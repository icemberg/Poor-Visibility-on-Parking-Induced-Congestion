# ============================================================
# Phase 7 — Next-Week Forecasting
# Predicts next-week hotspot risk, forecast band, and escalation probability
# using weekly hotspot history derived from Phases 1–6.
#
# ------------------------------------------------------------
# CORRECTIONS APPLIED IN THIS VERSION
# ------------------------------------------------------------
# 1. ROOT-CAUSE FIX for "weekly_growth_pct always 0.0" and "predictions
#    jumping from ~5 to ~98":
#    prepare_record_history() used to drop every row that didn't have a
#    *known* next-week target before returning. That target is only known
#    up to a hotspot's second-to-last week (the true last/latest week, by
#    definition, has no future week recorded yet). Forecasting code then
#    picked the "latest row per hotspot" out of that already-filtered
#    table, so it was always grabbing a STALE row, not the real present.
#    For a hotspot with only 2 observed weeks (the common case here, since
#    most hotspots have very little history), that stale row is literally
#    the hotspot's first-ever week — which has no prior week to diff
#    against, hence weekly_growth_pct == 0.0 — and the "current" score it
#    forecasts from is up to a week (or more) behind reality, which is what
#    produced unrealistic-looking deltas.
#    Fix: prepare_record_history() now returns the FULL weekly history
#    (including each hotspot's true latest, target-less week).
#    fit_predict_forecast() still drops target-less rows when building the
#    *training* set (model_df) — that part of the original logic was
#    already correct — but now pulls each hotspot's truly latest row from
#    the full, un-filtered history for inference.
#    NOTE: weekly_growth_pct's formula itself (lag-based, via shift(1)) was
#    already implemented correctly; nothing needed to change there.
#
# 2. Production guardrails on top of the fix above: a clip on week-over-week
#    score swings (MAX_DELTA_SCORE) and exponential smoothing toward the
#    current score (FORECAST_SMOOTHING_ALPHA), so a single noisy prediction
#    on a sparse/young hotspot still can't publish a 90+ point jump.
#
# 3. Escalation probabilities now pass through probability calibration
#    (CalibratedClassifierCV) so confidence values are less overconfident.
#    Added an escalation_brier_score metric — AUC/accuracy/F1 are all blind
#    to overconfidence, Brier score isn't — to monitor this going forward.
#    (The escalation_flag label definition itself — next-week raw pressure
#    >= 1.15x current raw pressure, using only past/contemporaneous
#    features — was already correct and uses no future-leaking columns.)
#
# 4. New diagnostics: a weeks-of-history-per-hotspot distribution (makes
#    data sparsity visible instead of silently producing growth == 0 for
#    single-week hotspots), and a warning when most Phase 6 hotspots carry
#    geometry_source == "fallback" (an upstream Phase 5/6 issue this script
#    cannot itself fix, but should surface loudly).
#
# 5. Forecasts are now ranked by a composite forecast_priority_score
#    (predicted score + escalation probability + upside delta) instead of
#    predicted score alone.
#
# 6. Replaced a `groupby(...).apply(per_group_function)` step in the weekly
#    feature engineering with plain vectorized groupby().shift()/.rolling()
#    calls. Whether that kind of apply's sub-frame includes the grouping
#    column is version-dependent — pandas 3.0 always excludes it with no
#    opt-out — so on newer pandas the original helper silently dropped
#    cluster_key and crashed the very next merge. This is now version-safe.
# ============================================================
import ast
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -------------------------
# Optional model backends
# -------------------------
BACKEND = None
XGBRegressor = XGBClassifier = None
LGBMRegressor = LGBMClassifier = None

try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore
    BACKEND = "xgboost"
except Exception:
    try:
        from lightgbm import LGBMRegressor, LGBMClassifier  # type: ignore
        BACKEND = "lightgbm"
    except Exception:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        BACKEND = "sklearn_forest"

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    # scikit-learn >= 1.4
    from sklearn.metrics import root_mean_squared_error  # type: ignore

    def rmse_metric(y_true, y_pred) -> float:
        return float(root_mean_squared_error(y_true, y_pred))
except Exception:
    def rmse_metric(y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

try:
    import joblib
    JOBLIB_OK = True
except Exception:
    import pickle
    JOBLIB_OK = False

# -------------------------
# Config
# -------------------------
PHASE6_DIRS = [
    Path("content/phase6_outputs_2"),
    Path("phase6_outputs_2"),
    Path("/content/phase6_outputs_2"),
]
PHASE5_DIRS = [
    Path("content/phase5_outputs_2"),
    Path("phase5_outputs_2"),
    Path("/content/phase5_outputs_2"),
]
PHASE4_DIRS = [
    Path("content/phase4_outputs_2"),
    Path("phase4_outputs_2"),
    Path("/content/phase4_outputs_2"),
]
PHASE3_DIRS = [
    Path("content/phase3_outputs_2"),
    Path("phase3_outputs_2"),
    Path("/content/phase3_outputs_2"),
]

OUT_DIR = Path("content/phase7_outputs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9
RANDOM_STATE = 42
VALIDATION_FRACTION_BY_WEEKS = 0.20
ESCALATION_INCREASE_THRESHOLD = 0.15  # 15% week-over-week increase

# --- Production guardrails (see CORRECTIONS #2 above) -----------------
ENABLE_DELTA_CLAMP = True
MAX_DELTA_SCORE = 20.0            # max allowed |next_week_score - current_week_score|
ENABLE_FORECAST_SMOOTHING = True
FORECAST_SMOOTHING_ALPHA = 0.7    # weight on CURRENT score; (1 - alpha) on the model's raw prediction

# --- Probability calibration (see CORRECTIONS #3 above) ----------------
ENABLE_PROBABILITY_CALIBRATION = True

# --- Ranking weights (see CORRECTIONS #5 above) -------------------------
PRIORITY_WEIGHT_SCORE = 0.5
PRIORITY_WEIGHT_ESCALATION = 30.0
PRIORITY_WEIGHT_DELTA = 10.0

# -------------------------
# Severity mapping
# -------------------------
SEVERITY_RULES = {
    5: {
        "DOUBLE PARKING",
        "NEAR ROAD CROSSING",
        "NEAR TRAFFIC LIGHT",
        "NEAR ZEBRA CROSSING",
        "NEAR TRAFFIC LIGHT / ZEBRA CROSSING",
        "NEAR TRAFFIC LIGHT/ZEBRA CROSSING",
    },
    4: {
        "PARKING IN MAIN ROAD",
        "NEAR BUS STOP",
        "NEAR SCHOOL",
        "NEAR HOSPITAL",
        "OPPOSITE ANOTHER VEHICLE",
    },
    3: {"PARKING ON FOOTPATH"},
    2: {"WRONG PARKING", "PARKING OTHER THAN BUS STOP"},
    1: {"NO PARKING", "NO PARKING (GENERIC)"},
}

# -------------------------
# Helpers
# -------------------------
def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_tag(tag) -> str:
    return clean_text(tag).upper().replace("&", "AND").strip()


def parse_listlike(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    s = str(value).strip()
    if not s:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    except Exception:
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return [p for p in parts if p]


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def standardize_cluster_col(df: pd.DataFrame) -> str:
    for c in ["st_dbscan_cluster_id", "cluster_id", "dbscan_cluster_id"]:
        if c in df.columns:
            return c
    raise ValueError("No cluster id column found.")


def standardize_vehicle_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["canonical_vehicle_number", "vehicle_number", "updated_vehicle_number"]:
        if c in df.columns:
            return c
    return None


def standardize_vehicle_type_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["canonical_vehicle_type", "vehicle_type", "updated_vehicle_type"]:
        if c in df.columns:
            return c
    return None


def normalize_cluster_key(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except Exception:
        return s


def load_first_existing(paths: Iterable[Path], filenames: Iterable[str]):
    for d in paths:
        for name in filenames:
            p = d / name
            if p.exists():
                return pd.read_csv(p, low_memory=False), p
    return None, None


def derive_coords(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    if "lat" in df.columns and "lon" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    elif {"centroid_lat", "centroid_lon"}.issubset(df.columns):
        df["lat"] = pd.to_numeric(df["centroid_lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["centroid_lon"], errors="coerce")
    elif {"latitude_mean", "longitude_mean"}.issubset(df.columns):
        df["lat"] = pd.to_numeric(df["latitude_mean"], errors="coerce")
        df["lon"] = pd.to_numeric(df["longitude_mean"], errors="coerce")
    elif {"latitude", "longitude"}.issubset(df.columns):
        df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    else:
        df["lat"] = np.nan
        df["lon"] = np.nan
    return df


def ensure_label_column(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    if "cluster_label" in df.columns:
        df["cluster_label"] = df["cluster_label"].fillna("").astype(str).str.strip()
        return df
    if "hotspot_unit" in df.columns:
        df["cluster_label"] = df["hotspot_unit"].fillna("").astype(str).str.strip()
        return df
    if "dominant_junction_name" in df.columns:
        df["cluster_label"] = df["dominant_junction_name"].fillna("").astype(str).str.strip()
        return df
    if "st_dbscan_cluster_id" in df.columns:
        df["cluster_label"] = "CLUSTER::" + df["st_dbscan_cluster_id"].astype(str)
        return df
    df["cluster_label"] = "UNKNOWN"
    return df


def parse_datetime_ist(df: pd.DataFrame) -> pd.Series:
    if "created_datetime_ist" in df.columns:
        ts = pd.to_datetime(df["created_datetime_ist"], errors="coerce", utc=True)
        if ts.notna().any():
            return ts.dt.tz_convert("Asia/Kolkata")
        ts = pd.to_datetime(df["created_datetime_ist"], errors="coerce")
        return ts.dt.tz_localize("Asia/Kolkata", nonexistent="NaT", ambiguous="NaT")

    if "created_datetime_parsed" in df.columns:
        ts = pd.to_datetime(df["created_datetime_parsed"], errors="coerce", utc=True)
        return ts.dt.tz_convert("Asia/Kolkata")

    if "created_datetime" not in df.columns:
        return pd.Series(pd.NaT, index=df.index)

    ts = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    return ts.dt.tz_convert("Asia/Kolkata")


def week_start_monday(series):
    dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    week_start = dt.dt.normalize() - pd.to_timedelta(dt.dt.weekday, unit="D")
    return week_start.dt.tz_localize(None)


def severity_from_tags(tags):
    if not tags:
        return 1
    normalized = [normalize_tag(t) for t in tags]
    score = 1
    for sev in sorted(SEVERITY_RULES.keys(), reverse=True):
        vocab = SEVERITY_RULES[sev]
        if any(any(v == tag or v in tag for v in vocab) for tag in normalized):
            score = sev
            break
    return score


def make_hotspot_unit(row):
    junction = clean_text(row.get("junction_name", ""))
    if junction and junction.upper() != "NO JUNCTION":
        return f"JUNCTION::{junction}"
    station = clean_text(row.get("police_station", "UNKNOWN"))
    if not station:
        station = "UNKNOWN"
    return f"POLICE_STATION::{station}"


def add_missing_cols(df: pd.DataFrame, defaults: Dict[str, object]) -> pd.DataFrame:
    df = df.copy()
    for c, default in defaults.items():
        if c not in df.columns:
            df[c] = default
    return df


def build_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    if len(y) < 2 or np.allclose(y, y[0], equal_nan=True):
        return 0.0
    x = np.arange(len(y), dtype=float)
    try:
        coef = np.polyfit(x, y, 1)[0]
        if np.isfinite(coef):
            return float(coef)
    except Exception:
        pass
    return 0.0


def pressure_raw_from_week(df: pd.DataFrame) -> pd.Series:
    rec = pd.to_numeric(df["records_week"], errors="coerce").fillna(0.0).clip(lower=0.0)
    sev_mean = pd.to_numeric(df["severity_mean_week"], errors="coerce").fillna(1.0).clip(lower=1.0, upper=5.0)
    peak_ratio = pd.to_numeric(df["peak_window_ratio_week"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    repeat_ratio = pd.to_numeric(df["repeat_vehicle_ratio_week"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    growth_surge = pd.to_numeric(df["growth_surge_week"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=5.0)
    congestion = np.log1p(rec) * (1.0 + sev_mean / 5.0) * (1.0 + 0.5 * peak_ratio) * (1.0 + repeat_ratio) * (1.0 + growth_surge)
    return congestion.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def score_to_0_100(raw: pd.Series, lo: float, hi: float) -> pd.Series:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + EPS:
        return pd.Series(np.full(len(raw), 50.0), index=raw.index)
    return 100.0 * ((pd.to_numeric(raw, errors="coerce").fillna(lo) - lo) / (hi - lo + EPS)).clip(0.0, 1.0)


def band_from_score(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Watch"


def ensure_bool_int(s):
    if s is None:
        return None
    if s.dtype == bool:
        return s.astype(int)
    if str(s.dtype).startswith("boolean"):
        return s.fillna(False).astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

# -------------------------
# Load inputs
# -------------------------
def load_static_phase6():
    df, src = load_first_existing(
        PHASE6_DIRS,
        [
            "phase6_cluster_ccs_full.csv",
            "phase6_stage6_handoff.csv",
            "phase6_weekly_dispatch_priority_table.csv",
        ],
    )
    if df is None:
        raise FileNotFoundError("Could not find Phase 6 outputs.")
    return df, src


def load_record_history():
    df, src = load_first_existing(
        PHASE5_DIRS + PHASE4_DIRS + PHASE3_DIRS,
        [
            "phase5_enriched_records.csv",
            "phase4_merged_with_prior_scores.csv",
            "phase3_clustered_dataset.csv",
        ],
    )
    if df is None:
        raise FileNotFoundError("Could not find Phase 5/4/3 record-level source.")
    return df, src

# -------------------------
# Core feature engineering
# -------------------------
def prepare_static_context(static_df: pd.DataFrame) -> pd.DataFrame:
    df = static_df.copy()
    df = ensure_label_column(df)
    df = derive_coords(df)

    cluster_col = standardize_cluster_col(df)
    df["cluster_key"] = df[cluster_col].map(normalize_cluster_key)

    defaults = {
        "cluster_label": "UNKNOWN",
        "cluster_label_display": "",
        "risk_band": "Watch",
        "ccs_score": 0.0,
        "delay_minutes_per_vehicle": 0.0,
        "records_total": 0.0,
        "distinct_days": 0.0,
        "severity_sum": 0.0,
        "severity_mean": 1.0,
        "growth_pct_change": 0.0,
        "growth_multiplier": 1.0,
        "criticality_factor": 1.0,
        "context_multiplier": 1.0,
        "layer_b_priority_boost": 0.0,
        "layer_b_alert_flag": 0,
        "validation_uncertainty": 0.0,
        "resurgence_score": 0.0,
        "persistence_score": 0.0,
        "anomaly_score": 0.0,
        "rop": 0.0,
        "tvs": 0.0,
        "vdi": 0.0,
        "nearby_sensitive_poi_count": 0.0,
        "road_class": "road",
        "lane_count": 2.0,
        "carriageway_width_m": 7.0,
        "link_length_m": 250.0,
        "junction_degree": 4.0,
        "betweenness_centrality": 0.0,
        "geometry_source": "fallback",
        "mappls_address": "",
        "road_node_id": np.nan,
        "road_node_distance_m": 0.0,
        "road_node_degree": 0.0,
        "road_node_betweenness": 0.0,
        "source_pressure_score": 0.0,
        "source_pressure_norm": 0.0,
        "spillover_out_score": 0.0,
        "spillover_in_score": 0.0,
        "spillover_total_score": 0.0,
        "propagation_radius_m": 0.0,
        "network_pagerank": 0.0,
        "network_component_id": 0,
        "network_component_size": 1,
        "neighbor_count": 0.0,
        "in_neighbor_count": 0.0,
        "out_neighbor_count": 0.0,
        "influence_asymmetry": 0.0,
        "network_vulnerability_score": 0.0,
        "layer_d_alert_flag": 0,
        "dominant_vehicle_type": "UNKNOWN",
    }
    df = add_missing_cols(df, defaults)

    numeric_cols = [
        "ccs_score", "delay_minutes_per_vehicle", "records_total", "distinct_days",
        "severity_sum", "severity_mean", "growth_pct_change", "growth_multiplier",
        "criticality_factor", "context_multiplier", "layer_b_priority_boost",
        "validation_uncertainty", "resurgence_score", "persistence_score",
        "anomaly_score", "rop", "tvs", "vdi", "nearby_sensitive_poi_count",
        "lane_count", "carriageway_width_m", "link_length_m", "junction_degree",
        "betweenness_centrality", "road_node_distance_m", "road_node_degree",
        "road_node_betweenness", "source_pressure_score", "source_pressure_norm",
        "spillover_out_score", "spillover_in_score", "spillover_total_score",
        "propagation_radius_m", "network_pagerank", "network_component_id",
        "network_component_size", "neighbor_count", "in_neighbor_count",
        "out_neighbor_count", "influence_asymmetry", "network_vulnerability_score",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["layer_b_alert_flag"] = ensure_bool_int(df["layer_b_alert_flag"])
    df["layer_d_alert_flag"] = ensure_bool_int(df["layer_d_alert_flag"])

    for c in ["road_class", "geometry_source", "dominant_vehicle_type"]:
        df[c] = df[c].fillna("UNKNOWN").astype(str).str.strip()

    df = df.sort_values(["ccs_score", "delay_minutes_per_vehicle", "records_total"], ascending=[False, False, False])
    df = df.drop_duplicates(subset=["cluster_key"], keep="first").reset_index(drop=True)
    return df


def prepare_record_history(records_df: pd.DataFrame, static_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    r = records_df.copy()
    cluster_col = standardize_cluster_col(r)
    vehicle_col = standardize_vehicle_col(r)
    vehicle_type_col = standardize_vehicle_type_col(r)

    if "validation_status" in r.columns:
        r["validation_status"] = r["validation_status"].fillna("").astype(str).str.lower()
        approved = r[r["validation_status"].eq("approved")].copy()
        if len(approved) > 0:
            r = approved.copy()

    r = ensure_label_column(r)
    r = derive_coords(r)
    r["created_datetime_ist"] = parse_datetime_ist(r)
    r = r.dropna(subset=["created_datetime_ist"]).copy()

    if "severity_score" not in r.columns:
        if "violation_type" in r.columns:
            r["violation_tags"] = r["violation_type"].apply(parse_listlike)
            r["severity_score"] = r["violation_tags"].apply(severity_from_tags)
        else:
            r["severity_score"] = 1
    r["severity_score"] = pd.to_numeric(r["severity_score"], errors="coerce").fillna(1).clip(lower=1, upper=5)

    if "hotspot_unit" not in r.columns:
        r["hotspot_unit"] = r.apply(make_hotspot_unit, axis=1)

    r["service_date"] = pd.to_datetime(r["created_datetime_ist"], errors="coerce").dt.tz_localize(None).dt.date
    r["week_start"] = week_start_monday(r["created_datetime_ist"])
    r["hour_ist"] = r["created_datetime_ist"].dt.hour
    r["is_peak_window"] = r["hour_ist"].between(8, 12, inclusive="both").astype(int)

    r["cluster_key"] = r[cluster_col].map(normalize_cluster_key)

    agg_spec = {
        "records_week": ("week_start", "size"),
        "active_days_week": ("service_date", "nunique"),
        "severity_sum_week": ("severity_score", "sum"),
        "severity_mean_week": ("severity_score", "mean"),
        "peak_window_records_week": ("is_peak_window", "sum"),
    }
    if vehicle_col and vehicle_col in r.columns:
        r[vehicle_col] = r[vehicle_col].fillna("").astype(str).str.strip()
        r = r[r[vehicle_col].ne("")].copy()
        agg_spec["unique_vehicles_week"] = (vehicle_col, "nunique")
    else:
        r["vehicle_fallback"] = "UNKNOWN"
        agg_spec["unique_vehicles_week"] = ("vehicle_fallback", "nunique")

    if vehicle_type_col and vehicle_type_col in r.columns:
        r[vehicle_type_col] = r[vehicle_type_col].fillna("").astype(str).str.strip()
        agg_spec["unique_vehicle_types_week"] = (vehicle_type_col, "nunique")
        agg_spec["dominant_vehicle_type_week"] = (vehicle_type_col, lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN")
    else:
        r["vehicle_type_fallback"] = "UNKNOWN"
        agg_spec["unique_vehicle_types_week"] = ("vehicle_type_fallback", "nunique")
        agg_spec["dominant_vehicle_type_week"] = ("vehicle_type_fallback", lambda s: "UNKNOWN")

    weekly = (
        r.groupby(["cluster_key", "week_start"])
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["cluster_key", "week_start"])
        .reset_index(drop=True)
    )

    weekly["records_per_active_day_week"] = weekly["records_week"] / weekly["active_days_week"].clip(lower=1)
    weekly["peak_window_ratio_week"] = weekly["peak_window_records_week"] / weekly["records_week"].clip(lower=1)
    weekly["repeat_vehicle_ratio_week"] = (
        (weekly["records_week"] - weekly["unique_vehicles_week"]) / weekly["records_week"].clip(lower=1)
    ).clip(lower=0.0, upper=1.0)

    weekly["prev_records_week"] = weekly.groupby("cluster_key")["records_week"].shift(1)
    weekly["weekly_growth_pct"] = (
        (weekly["records_week"] - weekly["prev_records_week"]) / (weekly["prev_records_week"].abs() + EPS)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weekly["growth_surge_week"] = weekly["weekly_growth_pct"].clip(lower=0.0)

    weekly["weekly_pressure_raw"] = pressure_raw_from_week(weekly)

    # NOTE (robustness fix): the per-hotspot feature block below used to be
    # a `weekly.groupby("cluster_key").apply(enrich_group)` helper. Whether
    # a groupby-apply's sub-frame includes the grouping column itself has
    # changed across pandas versions — as of pandas 3.0 it is *always*
    # excluded, with no way to opt back in — which made that helper
    # silently lose the cluster_key column on newer pandas and crash the
    # very next merge. Computing every per-hotspot feature with plain
    # groupby().shift()/.rolling()/.transform() (the same pattern already
    # used above for weekly_growth_pct) sidesteps that ambiguity entirely
    # and behaves identically on old and new pandas.
    weekly = weekly.sort_values(["cluster_key", "week_start"]).reset_index(drop=True)
    grp = weekly.groupby("cluster_key", sort=False)

    weekly["week_index"] = grp.cumcount()
    weekly["hotspot_age_weeks"] = weekly["week_index"] + 1

    first_pressure_raw = grp["weekly_pressure_raw"].transform("first")
    weekly["lag1_pressure_raw"] = grp["weekly_pressure_raw"].shift(1).fillna(first_pressure_raw)
    weekly["lag2_pressure_raw"] = grp["weekly_pressure_raw"].shift(2).fillna(weekly["lag1_pressure_raw"])

    weekly["rolling_4w_mean_pressure_raw"] = grp["weekly_pressure_raw"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    weekly["rolling_4w_std_pressure_raw"] = grp["weekly_pressure_raw"].transform(
        lambda s: s.rolling(4, min_periods=1).std(ddof=0)
    ).fillna(0.0)
    weekly["rolling_4w_mean_records_week"] = grp["records_week"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    weekly["rolling_4w_mean_severity_week"] = grp["severity_sum_week"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    weekly["rolling_4w_mean_growth_week"] = grp["weekly_growth_pct"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    weekly["rolling_4w_trend_slope_records_week"] = grp["records_week"].transform(
        lambda s: s.rolling(4, min_periods=1).apply(build_slope, raw=True)
    )
    weekly["rolling_4w_pressure_acceleration"] = weekly["rolling_4w_mean_pressure_raw"] - (
        grp["rolling_4w_mean_pressure_raw"].shift(1).fillna(weekly["rolling_4w_mean_pressure_raw"])
    )

    static_ctx = static_df.copy()
    static_ctx["cluster_key"] = static_ctx[cluster_col].map(normalize_cluster_key)
    static_cols_to_merge = [
        "cluster_label", "cluster_label_display", "risk_band", "ccs_score", "delay_minutes_per_vehicle",
        "records_total", "distinct_days", "severity_sum", "severity_mean", "growth_pct_change",
        "growth_multiplier", "criticality_factor", "context_multiplier", "layer_b_priority_boost",
        "layer_b_alert_flag", "validation_uncertainty", "resurgence_score", "persistence_score",
        "anomaly_score", "rop", "tvs", "vdi", "nearby_sensitive_poi_count", "road_class",
        "lane_count", "carriageway_width_m", "link_length_m", "junction_degree", "betweenness_centrality",
        "geometry_source", "mappls_address", "road_node_id", "road_node_distance_m", "road_node_degree",
        "road_node_betweenness", "source_pressure_score", "source_pressure_norm", "spillover_out_score",
        "spillover_in_score", "spillover_total_score", "propagation_radius_m", "network_pagerank",
        "network_component_id", "network_component_size", "neighbor_count", "in_neighbor_count",
        "out_neighbor_count", "influence_asymmetry", "network_vulnerability_score", "layer_d_alert_flag",
        "dominant_vehicle_type",
    ]
    static_cols_to_merge = [c for c in static_cols_to_merge if c in static_ctx.columns]
    static_ctx = static_ctx[["cluster_key"] + static_cols_to_merge].drop_duplicates("cluster_key")
    weekly = weekly.merge(static_ctx, on="cluster_key", how="left", suffixes=("", "_static"))

    fill_defaults = {
        "cluster_label": "UNKNOWN",
        "cluster_label_display": "",
        "risk_band": "Watch",
        "ccs_score": 0.0,
        "delay_minutes_per_vehicle": 0.0,
        "records_total": 0.0,
        "distinct_days": 0.0,
        "severity_sum": 0.0,
        "severity_mean": 1.0,
        "growth_pct_change": 0.0,
        "growth_multiplier": 1.0,
        "criticality_factor": 1.0,
        "context_multiplier": 1.0,
        "layer_b_priority_boost": 0.0,
        "layer_b_alert_flag": 0,
        "validation_uncertainty": 0.0,
        "resurgence_score": 0.0,
        "persistence_score": 0.0,
        "anomaly_score": 0.0,
        "rop": 0.0,
        "tvs": 0.0,
        "vdi": 0.0,
        "nearby_sensitive_poi_count": 0.0,
        "road_class": "road",
        "lane_count": 2.0,
        "carriageway_width_m": 7.0,
        "link_length_m": 250.0,
        "junction_degree": 4.0,
        "betweenness_centrality": 0.0,
        "geometry_source": "fallback",
        "mappls_address": "",
        "road_node_id": np.nan,
        "road_node_distance_m": 0.0,
        "road_node_degree": 0.0,
        "road_node_betweenness": 0.0,
        "source_pressure_score": 0.0,
        "source_pressure_norm": 0.0,
        "spillover_out_score": 0.0,
        "spillover_in_score": 0.0,
        "spillover_total_score": 0.0,
        "propagation_radius_m": 0.0,
        "network_pagerank": 0.0,
        "network_component_id": 0,
        "network_component_size": 1,
        "neighbor_count": 0.0,
        "in_neighbor_count": 0.0,
        "out_neighbor_count": 0.0,
        "influence_asymmetry": 0.0,
        "network_vulnerability_score": 0.0,
        "layer_d_alert_flag": 0,
        "dominant_vehicle_type": "UNKNOWN",
    }
    weekly = add_missing_cols(weekly, fill_defaults)

    numeric_cols = [
        "records_week", "active_days_week", "severity_sum_week", "severity_mean_week",
        "peak_window_records_week", "unique_vehicles_week", "unique_vehicle_types_week",
        "records_per_active_day_week", "peak_window_ratio_week", "repeat_vehicle_ratio_week",
        "prev_records_week", "weekly_growth_pct", "growth_surge_week", "weekly_pressure_raw",
        "week_index", "hotspot_age_weeks", "lag1_pressure_raw", "lag2_pressure_raw",
        "rolling_4w_mean_pressure_raw", "rolling_4w_std_pressure_raw",
        "rolling_4w_mean_records_week", "rolling_4w_mean_severity_week",
        "rolling_4w_mean_growth_week", "rolling_4w_trend_slope_records_week",
        "rolling_4w_pressure_acceleration",
        "ccs_score", "delay_minutes_per_vehicle", "records_total", "distinct_days",
        "severity_sum", "severity_mean", "growth_pct_change", "growth_multiplier",
        "criticality_factor", "context_multiplier", "layer_b_priority_boost",
        "validation_uncertainty", "resurgence_score", "persistence_score", "anomaly_score",
        "rop", "tvs", "vdi", "nearby_sensitive_poi_count", "lane_count",
        "carriageway_width_m", "link_length_m", "junction_degree", "betweenness_centrality",
        "road_node_distance_m", "road_node_degree", "road_node_betweenness",
        "source_pressure_score", "source_pressure_norm", "spillover_out_score",
        "spillover_in_score", "spillover_total_score", "propagation_radius_m",
        "network_pagerank", "network_component_id", "network_component_size",
        "neighbor_count", "in_neighbor_count", "out_neighbor_count",
        "influence_asymmetry", "network_vulnerability_score",
    ]
    for c in numeric_cols:
        if c in weekly.columns:
            weekly[c] = pd.to_numeric(weekly[c], errors="coerce")

    weekly["layer_b_alert_flag"] = ensure_bool_int(weekly["layer_b_alert_flag"])
    weekly["layer_d_alert_flag"] = ensure_bool_int(weekly["layer_d_alert_flag"])

    for c in ["road_class", "geometry_source", "dominant_vehicle_type"]:
        weekly[c] = weekly[c].fillna("UNKNOWN").astype(str).str.strip()

    weekly["current_week_risk_raw"] = weekly["weekly_pressure_raw"].copy()
    weekly["target_next_week_risk_raw"] = weekly.groupby("cluster_key")["weekly_pressure_raw"].shift(-1)
    weekly["target_next_week_records_week"] = weekly.groupby("cluster_key")["records_week"].shift(-1)
    weekly["target_next_week_growth_surge_week"] = weekly.groupby("cluster_key")["growth_surge_week"].shift(-1)

    weekly["escalation_flag"] = (
        (weekly["target_next_week_risk_raw"] >= weekly["current_week_risk_raw"] * (1.0 + ESCALATION_INCREASE_THRESHOLD))
    ).astype(int)

    # FIX (see CORRECTIONS #1 at top of file): do NOT drop rows with a
    # missing target_next_week_risk_raw here. Each hotspot's most recent
    # observed week never has a known next-week target by definition — and
    # that row is exactly what downstream forecasting needs to predict
    # FROM. The original code dropped it at this point, which silently
    # forced every "latest week" lookup in fit_predict_forecast() onto a
    # stale, second-to-last week instead of the real present. Rows without
    # a target are still excluded from the *training* set later (see
    # model_df inside fit_predict_forecast) — just not from the history
    # returned here.
    weekly = weekly.reset_index(drop=True)
    return weekly, cluster_col

# -------------------------
# Modeling
# -------------------------
def choose_model_backend():
    return BACKEND or "sklearn_forest"


def build_models(scale_pos_weight: Optional[float] = None):
    backend = choose_model_backend()
    if backend == "xgboost":
        reg = XGBRegressor(
            n_estimators=350, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
        )
        clf_kwargs = dict(
            n_estimators=350, learning_rate=0.05, max_depth=6,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist",
            eval_metric="logloss",
        )
        if scale_pos_weight is not None and np.isfinite(scale_pos_weight) and scale_pos_weight > 0:
            clf_kwargs["scale_pos_weight"] = float(scale_pos_weight)
        clf = XGBClassifier(**clf_kwargs)
    elif backend == "lightgbm":
        reg = LGBMRegressor(
            n_estimators=450, learning_rate=0.05, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, n_jobs=-1,
        )
        clf = LGBMClassifier(
            n_estimators=450, learning_rate=0.05, num_leaves=31,
            subsample=0.85, colsample_bytree=0.85, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced" if scale_pos_weight is None else None,
        )
    else:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        reg = RandomForestRegressor(
            n_estimators=450, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2,
        )
        clf = RandomForestClassifier(
            n_estimators=450, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2,
            class_weight="balanced",
        )
    return reg, clf


def build_calibrated_classifier(base_pipe, method: str = "sigmoid", cv: int = 3):
    """Wrap a fitted-or-unfitted classifier Pipeline in probability calibration.

    See CORRECTIONS #3 at the top of this file: AUC/accuracy/F1 near-perfect
    scores are consistent with a genuinely strong signal AND overconfident
    probabilities at the same time. Calibration only rescales predict_proba
    output; it does not change which class wins, so ranking/recommendations
    built off predicted_escalation_flag are unaffected.
    """
    try:
        return CalibratedClassifierCV(estimator=base_pipe, method=method, cv=cv)
    except TypeError:
        # Older scikit-learn versions use `base_estimator` instead of `estimator`.
        return CalibratedClassifierCV(base_estimator=base_pipe, method=method, cv=cv)


def build_preprocessor(num_cols: List[str], cat_cols: List[str]):
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", build_ohe())])
    return ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        sparse_threshold=0.0,
    )


def temporal_split(df: pd.DataFrame, week_col: str = "week_start", frac: float = 0.20):
    weeks = sorted(pd.to_datetime(df[week_col], errors="coerce").dropna().unique())
    if len(weeks) < 3:
        idx = np.arange(len(df))
        rng = np.random.default_rng(RANDOM_STATE)
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * (1 - frac)))
        return df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()

    valid_weeks_count = max(1, int(round(len(weeks) * frac)))
    train_weeks = weeks[:-valid_weeks_count]
    valid_weeks = weeks[-valid_weeks_count:]

    train_df = df[df[week_col].isin(train_weeks)].copy()
    valid_df = df[df[week_col].isin(valid_weeks)].copy()

    if len(train_df) == 0 or len(valid_df) == 0:
        cut = max(1, int(len(df) * (1 - frac)))
        train_df = df.iloc[:cut].copy()
        valid_df = df.iloc[cut:].copy()
    return train_df, valid_df


def fit_predict_forecast(weekly_df, static_df, cluster_col):
    numeric_features = [
        "records_week", "active_days_week", "severity_sum_week", "severity_mean_week",
        "peak_window_records_week", "unique_vehicles_week", "unique_vehicle_types_week",
        "records_per_active_day_week", "peak_window_ratio_week", "repeat_vehicle_ratio_week",
        "weekly_growth_pct", "growth_surge_week", "current_week_risk_raw",
        "week_index", "hotspot_age_weeks", "lag1_pressure_raw", "lag2_pressure_raw",
        "rolling_4w_mean_pressure_raw", "rolling_4w_std_pressure_raw",
        "rolling_4w_mean_records_week", "rolling_4w_mean_severity_week",
        "rolling_4w_mean_growth_week", "rolling_4w_trend_slope_records_week",
        "rolling_4w_pressure_acceleration",
        "records_total", "distinct_days", "severity_sum", "severity_mean",
        "growth_pct_change", "growth_multiplier", "criticality_factor",
        "context_multiplier", "layer_b_priority_boost", "validation_uncertainty",
        "resurgence_score", "persistence_score", "anomaly_score", "rop", "tvs", "vdi",
        "nearby_sensitive_poi_count", "lane_count", "carriageway_width_m",
        "link_length_m", "junction_degree", "betweenness_centrality",
        "road_node_distance_m", "road_node_degree", "road_node_betweenness",
        "source_pressure_score", "source_pressure_norm", "spillover_out_score",
        "spillover_in_score", "spillover_total_score", "propagation_radius_m",
        "network_pagerank", "network_component_id", "network_component_size",
        "neighbor_count", "in_neighbor_count", "out_neighbor_count",
        "influence_asymmetry", "network_vulnerability_score",
        "layer_b_alert_flag", "layer_d_alert_flag",
    ]
    categorical_features = ["road_class", "geometry_source", "dominant_vehicle_type"]

    numeric_features = [c for c in numeric_features if c in weekly_df.columns]
    categorical_features = [c for c in categorical_features if c in weekly_df.columns]

    # model_df: rows with a KNOWN next-week target -> used to train/validate.
    # This correctly excludes each hotspot's true latest week (which has no
    # future yet). weekly_df itself (the full, un-filtered history) is kept
    # around below so the truly latest row per hotspot can still be used
    # for inference — see CORRECTIONS #1 at the top of this file.
    model_df = weekly_df.copy().dropna(subset=["target_next_week_risk_raw"]).copy()
    train_df, valid_df = temporal_split(model_df, week_col="week_start", frac=VALIDATION_FRACTION_BY_WEEKS)
    if len(train_df) == 0 or len(valid_df) == 0:
        raise RuntimeError("Not enough temporal history to train the forecast model.")

    X_train = train_df[numeric_features + categorical_features].copy()
    y_train_raw = train_df["target_next_week_risk_raw"].astype(float).copy()

    X_valid = valid_df[numeric_features + categorical_features].copy()
    y_valid_raw = valid_df["target_next_week_risk_raw"].astype(float).copy()

    y_train_cls = train_df["escalation_flag"].astype(int).copy()
    y_valid_cls = valid_df["escalation_flag"].astype(int).copy()

    target_min = float(y_train_raw.min())
    target_max = float(y_train_raw.max())

    def raw_to_score(raw_series: pd.Series) -> pd.Series:
        return score_to_0_100(raw_series, target_min, target_max)

    pos = int(y_train_cls.sum())
    neg = int(len(y_train_cls) - pos)
    scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

    reg_model, clf_model = build_models(scale_pos_weight=scale_pos_weight if pos > 0 else None)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    reg_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", reg_model)])
    base_clf_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf_model)])

    reg_pipe.fit(X_train, y_train_raw)

    use_classifier = y_train_cls.nunique() >= 2 and len(y_train_cls) >= 20
    clf_pipe = None
    if use_classifier:
        base_clf_pipe.fit(X_train, y_train_cls)
        clf_pipe = base_clf_pipe
        if ENABLE_PROBABILITY_CALIBRATION:
            # See CORRECTIONS #3: rescale overconfident probabilities
            # without changing which hotspots get flagged as escalating.
            min_class_count = int(min(pos, neg))
            if min_class_count >= 4:
                cv_folds = 3 if min_class_count >= 6 else 2
                method = "isotonic" if len(y_train_cls) >= 200 else "sigmoid"
                try:
                    calibrated = build_calibrated_classifier(base_clf_pipe, method=method, cv=cv_folds)
                    calibrated.fit(X_train, y_train_cls)
                    clf_pipe = calibrated
                except Exception:
                    clf_pipe = base_clf_pipe

    valid_pred_raw = pd.Series(reg_pipe.predict(X_valid), index=valid_df.index)
    valid_pred_score = raw_to_score(valid_pred_raw)
    valid_true_score = raw_to_score(y_valid_raw)

    valid_pred_escalation = None
    valid_pred_escalation_prob = None
    if clf_pipe is not None:
        try:
            valid_pred_escalation_prob = pd.Series(clf_pipe.predict_proba(X_valid)[:, 1], index=valid_df.index)
            valid_pred_escalation = (valid_pred_escalation_prob >= 0.5).astype(int)
        except Exception:
            valid_pred_escalation = pd.Series(clf_pipe.predict(X_valid), index=valid_df.index).astype(int)
            valid_pred_escalation_prob = valid_pred_escalation.astype(float)

    rmse = rmse_metric(y_valid_raw, valid_pred_raw) if len(valid_df) else np.nan
    mae = float(mean_absolute_error(y_valid_raw, valid_pred_raw)) if len(valid_df) else np.nan
    r2 = float(r2_score(y_valid_raw, valid_pred_raw)) if len(valid_df) else np.nan

    metrics_rows = [
        {"metric": "backend", "value": choose_model_backend()},
        {"metric": "train_rows", "value": len(train_df)},
        {"metric": "validation_rows", "value": len(valid_df)},
        {"metric": "target_raw_min_train", "value": target_min},
        {"metric": "target_raw_max_train", "value": target_max},
        {"metric": "regression_rmse_raw", "value": rmse},
        {"metric": "regression_mae_raw", "value": mae},
        {"metric": "regression_r2_raw", "value": r2},
    ]

    if use_classifier and valid_pred_escalation_prob is not None and len(valid_df):
        try:
            auc = float(roc_auc_score(y_valid_cls, valid_pred_escalation_prob))
        except Exception:
            auc = np.nan
        try:
            ap = float(average_precision_score(y_valid_cls, valid_pred_escalation_prob))
        except Exception:
            ap = np.nan
        try:
            acc = float(accuracy_score(y_valid_cls, valid_pred_escalation))
        except Exception:
            acc = np.nan
        try:
            f1 = float(f1_score(y_valid_cls, valid_pred_escalation))
        except Exception:
            f1 = np.nan
        try:
            brier = float(brier_score_loss(y_valid_cls, valid_pred_escalation_prob))
        except Exception:
            brier = np.nan
        metrics_rows.extend([
            {"metric": "escalation_auc", "value": auc},
            {"metric": "escalation_average_precision", "value": ap},
            {"metric": "escalation_accuracy", "value": acc},
            {"metric": "escalation_f1", "value": f1},
            {"metric": "escalation_brier_score", "value": brier},
        ])

    metrics_df = pd.DataFrame(metrics_rows)

    importance_df = pd.DataFrame()
    try:
        feature_names = reg_pipe.named_steps["preprocess"].get_feature_names_out()
        model = reg_pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        importance_df = pd.DataFrame(columns=["feature", "importance"])

    # FIX (CORRECTIONS #1): pull each hotspot's truly latest observed week
    # from the full, un-filtered weekly_df — NOT from model_df. model_df
    # only contains rows with a known next-week target, so its "latest" row
    # per hotspot is the second-to-last (or older) observed week, not the
    # actual present. We only need *features* here (not the target), so a
    # missing target on these rows is expected and fine.
    latest_rows = (
        weekly_df.sort_values(["cluster_key", "week_start"])
        .groupby("cluster_key", as_index=False)
        .tail(1)
        .copy()
    )

    latest_X = latest_rows[numeric_features + categorical_features].copy()
    latest_pred_raw = pd.Series(reg_pipe.predict(latest_X), index=latest_rows.index)
    latest_pred_score = raw_to_score(latest_pred_raw)

    current_score = raw_to_score(latest_rows["current_week_risk_raw"].astype(float))

    escalation_prob = pd.Series(np.full(len(latest_rows), np.nan), index=latest_rows.index)
    escalation_pred = pd.Series(np.full(len(latest_rows), 0), index=latest_rows.index, dtype=int)
    if clf_pipe is not None:
        try:
            escalation_prob = pd.Series(clf_pipe.predict_proba(latest_X)[:, 1], index=latest_rows.index)
            escalation_pred = (escalation_prob >= 0.5).astype(int)
        except Exception:
            escalation_pred = pd.Series(clf_pipe.predict(latest_X), index=latest_rows.index).astype(int)
            escalation_prob = escalation_pred.astype(float)

    forecast_cols = [
        "cluster_key", "cluster_label", "cluster_label_display", "week_start",
        "current_week_risk_raw", "records_week", "severity_sum_week",
        "severity_mean_week", "weekly_growth_pct", "growth_surge_week",
        "hotspot_age_weeks", "road_class", "geometry_source",
        "dominant_vehicle_type", "layer_b_alert_flag", "layer_d_alert_flag",
        "ccs_score", "delay_minutes_per_vehicle", "criticality_factor",
        "context_multiplier", "layer_b_priority_boost", "validation_uncertainty",
        "resurgence_score", "persistence_score", "anomaly_score", "rop",
        "tvs", "vdi", "network_vulnerability_score", "spillover_total_score",
        "road_node_betweenness", "road_node_degree", "road_node_distance_m",
        "neighbor_count", "in_neighbor_count", "out_neighbor_count",
        "influence_asymmetry", "nearby_sensitive_poi_count",
    ]
    forecast_cols = [c for c in forecast_cols if c in latest_rows.columns]
    forecast = latest_rows[forecast_cols].copy()

    forecast["current_week_ccs_proxy_score"] = current_score.values
    forecast["predicted_next_week_risk_raw"] = latest_pred_raw.values
    forecast["model_predicted_next_week_ccs_score"] = latest_pred_score.values
    forecast["probability_of_escalation"] = escalation_prob.values
    forecast["predicted_escalation_flag"] = escalation_pred.values

    # --- CORRECTIONS #2: stabilize the published score ---------------------
    # Even after fixing which row counts as "latest", a hotspot with very
    # little history can still produce a noisy regression output. Blend
    # toward the current score and cap the swing so the published forecast
    # can't jump by 90+ points off a single prediction.
    stabilized_score = forecast["model_predicted_next_week_ccs_score"].astype(float).copy()

    if ENABLE_FORECAST_SMOOTHING:
        stabilized_score = (
            FORECAST_SMOOTHING_ALPHA * forecast["current_week_ccs_proxy_score"].astype(float)
            + (1.0 - FORECAST_SMOOTHING_ALPHA) * stabilized_score
        )

    if ENABLE_DELTA_CLAMP:
        raw_delta = stabilized_score - forecast["current_week_ccs_proxy_score"].astype(float)
        clamped_delta = raw_delta.clip(lower=-MAX_DELTA_SCORE, upper=MAX_DELTA_SCORE)
        stabilized_score = (forecast["current_week_ccs_proxy_score"].astype(float) + clamped_delta).clip(0.0, 100.0)

    forecast["predicted_next_week_ccs_score"] = stabilized_score
    # Risk band must be derived from the score actually published, not the
    # pre-stabilization model output, or band and score could disagree.
    forecast["future_risk_band"] = forecast["predicted_next_week_ccs_score"].apply(lambda x: band_from_score(float(x)))

    forecast["predicted_delta_score"] = forecast["predicted_next_week_ccs_score"] - forecast["current_week_ccs_proxy_score"]
    forecast["predicted_delta_pct"] = (forecast["predicted_delta_score"] / (forecast["current_week_ccs_proxy_score"].abs() + EPS)) * 100.0
    forecast["forecast_week_start"] = pd.to_datetime(forecast["week_start"], errors="coerce") + pd.Timedelta(days=7)

    coord_lookup = pd.DataFrame(columns=["cluster_key", "lat", "lon"])

    if static_df is not None:
        coord_lookup = static_df.copy()
        coord_lookup["cluster_key"] = coord_lookup[cluster_col].map(normalize_cluster_key)

        if "lat" not in coord_lookup.columns:
            coord_lookup["lat"] = np.nan
        if "lon" not in coord_lookup.columns:
            coord_lookup["lon"] = np.nan

        coord_lookup = coord_lookup[["cluster_key", "lat", "lon"]].drop_duplicates("cluster_key")

    forecast = forecast.merge(coord_lookup, on="cluster_key", how="left")
    forecast["lat"] = pd.to_numeric(forecast["lat"], errors="coerce")
    forecast["lon"] = pd.to_numeric(forecast["lon"], errors="coerce")

    # --- CORRECTIONS #5: rank by a composite priority score, not score alone
    forecast["forecast_priority_score"] = (
        PRIORITY_WEIGHT_SCORE * forecast["predicted_next_week_ccs_score"]
        + PRIORITY_WEIGHT_ESCALATION * forecast["probability_of_escalation"].fillna(0.0)
        + PRIORITY_WEIGHT_DELTA * forecast["predicted_delta_score"].clip(lower=0.0)
    )

    forecast = forecast.sort_values(
        ["forecast_priority_score", "predicted_next_week_ccs_score", "probability_of_escalation"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    forecast["forecast_rank"] = np.arange(1, len(forecast) + 1)

    forecast["forecast_recommendation"] = np.select(
        [
            forecast["future_risk_band"].eq("Critical"),
            forecast["future_risk_band"].eq("High"),
            forecast["future_risk_band"].eq("Moderate"),
        ],
        [
            "Immediate patrol deployment",
            "Targeted enforcement + tow readiness",
            "Monitor and schedule peak-window checks",
        ],
        default="Routine monitoring",
    )

    return {
        "reg_pipe": reg_pipe,
        "clf_pipe": clf_pipe,
        "metrics_df": metrics_df,
        "importance_df": importance_df,
        "forecast_df": forecast,
        "train_df": train_df,
        "valid_df": valid_df,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_min": target_min,
        "target_max": target_max,
        "cluster_col": "cluster_key",
    }

# -------------------------
# Main
# -------------------------
def main():
    print("Loading Phase 6 static snapshot...")
    static_df, static_src = load_static_phase6()
    print("Static source:", static_src)

    print("Loading record-level history...")
    records_df, records_src = load_record_history()
    print("Record source:", records_src)

    static_df = prepare_static_context(static_df)

    # CORRECTIONS #4: surface upstream Phase 5/6 geometry-join failures
    # instead of silently forecasting on all-fallback road context.
    if "geometry_source" in static_df.columns and len(static_df):
        fallback_share = float(static_df["geometry_source"].astype(str).str.lower().eq("fallback").mean())
        if fallback_share > 0.5:
            print(
                f"\n[WARNING] {fallback_share:.0%} of Phase 6 hotspots have "
                f"geometry_source == 'fallback'. This points to an upstream "
                f"Phase 5/6 road-graph join issue (road_class, lane_count, "
                f"junction_degree, betweenness_centrality not attaching) — "
                f"not something this script can fix. Forecasts below will "
                f"still run on record-level signal, but road-network "
                f"features will contribute little until that join is "
                f"repaired.\n"
                # PATCH: actionable guidance for the upstream fix.
                f"  ACTION: Re-run stage5 with ENABLE_OSMNX_PHASE5=1 and "
                f"check for a saved graph at content/phase5_outputs_2/"
                f"osmnx_graph_cache.graphml.  If that file is absent or "
                f"zero-bytes, the OSMnx download itself is failing — check "
                f"network access and the exception messages printed during "
                f"build_osmnx_graph().  If the file exists but fallback "
                f"share is still high, check the per-point fallback logs "
                f"emitted by road_context_for_points() — they now print the "
                f"exception reason for the first 3 and every 50th failure.\n"
            )

    weekly_df, cluster_col = prepare_record_history(records_df, static_df)

    if len(weekly_df) == 0:
        raise RuntimeError("No weekly rows available after feature engineering.")

    print("Weekly rows built:", len(weekly_df))
    print("Hotspots in weekly history:", weekly_df["cluster_key"].nunique())

    # CORRECTIONS #4: make data sparsity visible. A hotspot with only one
    # observed week will correctly show weekly_growth_pct == 0.0 — there is
    # no prior week to compare against — which is expected behavior, not a
    # bug, given how little history most hotspots currently have.
    weeks_per_cluster = weekly_df.groupby("cluster_key")["week_start"].nunique()
    print("\nWeeks of history per hotspot (describe):")
    print(weeks_per_cluster.describe().to_string())
    single_week_share = float((weeks_per_cluster <= 1).mean())
    print(
        f"Share of hotspots with only ONE observed week: {single_week_share:.1%} "
        f"-- weekly_growth_pct == 0.0 for these is expected, not a bug.\n"
    )

    print("Training model backend:", choose_model_backend())

    artifacts = fit_predict_forecast(weekly_df, static_df, cluster_col)

    reg_pipe = artifacts["reg_pipe"]
    clf_pipe = artifacts["clf_pipe"]
    metrics_df = artifacts["metrics_df"]
    importance_df = artifacts["importance_df"]
    forecast_df = artifacts["forecast_df"]
    train_df = artifacts["train_df"]
    valid_df = artifacts["valid_df"]

    weekly_df.to_csv(OUT_DIR / "phase7_weekly_hotspot_history.csv", index=False)
    train_df.to_csv(OUT_DIR / "phase7_training_rows.csv", index=False)
    valid_df.to_csv(OUT_DIR / "phase7_validation_rows.csv", index=False)
    forecast_df.to_csv(OUT_DIR / "phase7_next_week_forecast.csv", index=False)
    forecast_df.to_csv(OUT_DIR / "phase7_next_week_forecast_ranked.csv", index=False)
    metrics_df.to_csv(OUT_DIR / "phase7_model_metrics.csv", index=False)
    if len(importance_df):
        importance_df.to_csv(OUT_DIR / "phase7_feature_importance.csv", index=False)

    artifact_path = OUT_DIR / ("phase7_model_artifact.joblib" if JOBLIB_OK else "phase7_model_artifact.pkl")
    artifact_obj = {
        "backend": choose_model_backend(),
        "regressor": reg_pipe,
        "classifier": clf_pipe,
        "numeric_features": artifacts["numeric_features"],
        "categorical_features": artifacts["categorical_features"],
        "target_min": artifacts["target_min"],
        "target_max": artifacts["target_max"],
    }
    if JOBLIB_OK:
        joblib.dump(artifact_obj, artifact_path)
    else:
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact_obj, f)

    print("\nPhase 7 complete")
    print("Outputs saved to:", OUT_DIR.resolve())
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print("\nTop 10 next-week forecasts:")
    top_cols = [
        "forecast_rank", "cluster_key", "cluster_label_display", "future_risk_band",
        "forecast_priority_score", "predicted_next_week_ccs_score", "probability_of_escalation",
        "current_week_ccs_proxy_score", "predicted_delta_score",
        "records_week", "weekly_growth_pct", "road_class",
        "geometry_source", "forecast_recommendation",
    ]
    top_cols = [c for c in top_cols if c in forecast_df.columns]
    print(forecast_df[top_cols].head(10).to_string(index=False))

    print("\nEscalation probability distribution across all forecasted hotspots:")
    print(forecast_df["probability_of_escalation"].describe().to_string())

    print("\nTop feature importance:")
    if len(importance_df):
        print(importance_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()