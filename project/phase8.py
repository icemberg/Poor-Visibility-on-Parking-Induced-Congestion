
from __future__ import annotations

import ast
import math
import os
import warnings
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# Phase 8 — Prescriptive Enforcement Engine
# Inputs: Phase 7 forecasts + Phase 6/5/4/3 context
# Outputs: patrol allocation, tow readiness, time-window
# recommendations, zone ranking, route plan, offender actions.
# ============================================================

PHASE8_OUT_DIR = Path("content/phase8_outputs_2")
PHASE8_OUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE7_DIRS = [Path("content/phase7_outputs_2"), Path("phase7_outputs_2"), Path("/content/phase7_outputs_2")]
PHASE6_DIRS = [Path("content/phase6_outputs_2"), Path("phase6_outputs_2"), Path("/content/phase6_outputs_2")]
PHASE5_DIRS = [Path("content/phase5_outputs_2"), Path("phase5_outputs_2"), Path("/content/phase5_outputs_2")]
PHASE4_DIRS = [Path("content/phase4_outputs_2"), Path("phase4_outputs_2"), Path("/content/phase4_outputs_2")]
PHASE3_DIRS = [Path("content/phase3_outputs_2"), Path("phase3_outputs_2"), Path("/content/phase3_outputs_2")]

EPS = 1e-9
TOTAL_PATROL_UNITS = int(os.environ.get("PHASE8_TOTAL_PATROL_UNITS", "15"))
ROUTE_PLAN_TOP_N = int(os.environ.get("PHASE8_ROUTE_PLAN_TOP_N", "12"))
DEPOT_LAT = float(os.environ.get("PHASE8_DEPOT_LAT", "12.9716"))
DEPOT_LON = float(os.environ.get("PHASE8_DEPOT_LON", "77.5946"))
MAX_PATROLS_PER_HOTSPOT = int(os.environ.get("PHASE8_MAX_PATROLS_PER_HOTSPOT", "3"))
PATROL_ELIGIBILITY_SCORE = float(os.environ.get("PHASE8_PATROL_ELIGIBILITY_SCORE", "40"))
TOW_PRIORITY_THRESHOLD = float(os.environ.get("PHASE8_TOW_PRIORITY_THRESHOLD", "70"))
ESCALATION_THRESHOLD = float(os.environ.get("PHASE8_ESCALATION_THRESHOLD", "0.50"))
HIGH_BETWEENNESS_PERCENTILE = float(os.environ.get("PHASE8_HIGH_BETWEENNESS_PERCENTILE", "0.75"))

SEVERITY_RULES = {
    5: {"DOUBLE PARKING", "NEAR ROAD CROSSING", "NEAR TRAFFIC LIGHT", "NEAR ZEBRA CROSSING", "NEAR TRAFFIC LIGHT / ZEBRA CROSSING", "NEAR TRAFFIC LIGHT/ZEBRA CROSSING"},
    4: {"PARKING IN MAIN ROAD", "NEAR BUS STOP", "NEAR SCHOOL", "NEAR HOSPITAL", "OPPOSITE ANOTHER VEHICLE"},
    3: {"PARKING ON FOOTPATH"},
    2: {"WRONG PARKING", "PARKING OTHER THAN BUS STOP"},
    1: {"NO PARKING", "NO PARKING (GENERIC)"},
}


# ------------------------- helpers -------------------------
def clean_text(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


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
        return [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_first_existing(base_dirs: Iterable[Path], filenames: Iterable[str]):
    for d in base_dirs:
        for name in filenames:
            p = d / name
            if p.exists():
                return pd.read_csv(p, low_memory=False), p
    return None, None


def standardize_cluster_col(df: pd.DataFrame) -> str:
    for c in ["cluster_key", "st_dbscan_cluster_id", "cluster_id", "dbscan_cluster_id"]:
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


def dominant_label(series, exclude=None, default=""):
    s = pd.Series(series).dropna().astype(str).str.strip()
    s = s[s.ne("")]
    if exclude:
        excl = {clean_text(x).lower() for x in exclude}
        s = s[~s.str.lower().isin(excl)]
    if s.empty:
        return default
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    mn = valid.min()
    mx = valid.max()
    return (s.fillna(mn) - mn) / (mx - mn + EPS)


def band_from_score(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Moderate"
    return "Watch"


def band_index(band: str) -> int:
    return {"Watch": 0, "Moderate": 1, "High": 2, "Critical": 3}.get(str(band), 0)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    r = 6371000.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1 - a)))


def build_zone_name(df: pd.DataFrame) -> pd.Series:
    zone = pd.Series([""] * len(df), index=df.index, dtype="object")
    for c in ["dominant_police_station", "police_station", "dominant_junction_name", "hotspot_unit", "cluster_label_display", "cluster_label"]:
        if c in df.columns:
            s = df[c].fillna("").astype(str).str.strip()
            s = s.replace({"No Junction": "", "NO JUNCTION": ""})
            zone = zone.mask(zone.eq(""), s)
    return zone.replace({"": "UNASSIGNED"}).astype(str)


def coalesce_from_suffix(df: pd.DataFrame, base_cols: Iterable[str], suffix: str = "_static"):
    out = df.copy()
    for c in base_cols:
        s = f"{c}{suffix}"
        if s in out.columns:
            if c in out.columns:
                out[c] = out[c].combine_first(out[s])
            else:
                out[c] = out[s]
            out.drop(columns=[s], inplace=True)
    return out


def safe_to_int_series(s, default=0):
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(int)


def pressure_raw_from_week(df: pd.DataFrame) -> pd.Series:
    rec = pd.to_numeric(df["records_week"], errors="coerce").fillna(0.0).clip(lower=0.0)
    sev = pd.to_numeric(df["severity_mean_week"], errors="coerce").fillna(1.0).clip(lower=1.0, upper=5.0)
    peak = pd.to_numeric(df["peak_window_ratio_week"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    repeat = pd.to_numeric(df["repeat_vehicle_ratio_week"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    growth = pd.to_numeric(df["growth_surge_week"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=5.0)
    raw = np.log1p(rec) * (1.0 + sev / 5.0) * (1.0 + 0.5 * peak) * (1.0 + repeat) * (1.0 + growth)
    return pd.Series(raw, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_shannon_diversity_from_tags(tag_series: pd.Series) -> float:
    tags = pd.Series(tag_series).dropna().astype(str)
    tags = tags[tags.str.strip().ne("")]
    if tags.empty:
        return 0.0
    probs = tags.value_counts(normalize=True)
    entropy = float(-(probs * np.log(probs + EPS)).sum())
    max_entropy = float(np.log(len(probs) + EPS))
    return 0.0 if max_entropy <= EPS else float(np.clip(entropy / max_entropy, 0.0, 1.0))


def make_hotspot_unit(row):
    junction = clean_text(row.get("junction_name", ""))
    if junction and junction.upper() != "NO JUNCTION":
        return f"JUNCTION::{junction}"
    station = clean_text(row.get("police_station", "UNKNOWN")) or "UNKNOWN"
    return f"POLICE_STATION::{station}"


# ------------------------- load phase inputs -------------------------
def load_phase7_forecast():
    df, src = load_first_existing(PHASE7_DIRS, ["phase7_next_week_forecast_ranked.csv", "phase7_next_week_forecast.csv"])
    if df is None:
        raise FileNotFoundError("Could not find Phase 7 forecast outputs.")
    return df, src


def load_phase6_static():
    df, src = load_first_existing(PHASE6_DIRS, ["phase6_cluster_ccs_full.csv", "phase6_stage6_handoff.csv", "phase6_weekly_dispatch_priority_table.csv"])
    if df is None:
        raise FileNotFoundError("Could not find Phase 6 outputs.")
    return df, src


def load_record_history():
    df, src = load_first_existing(PHASE5_DIRS + PHASE4_DIRS + PHASE3_DIRS, ["phase5_enriched_records.csv", "phase4_merged_with_prior_scores.csv", "phase3_clustered_dataset.csv"])
    if df is None:
        raise FileNotFoundError("Could not find record-level source from Phase 5/4/3.")
    return df, src


def load_chronic_offenders():
    return load_first_existing(PHASE6_DIRS + PHASE5_DIRS, ["phase6_chronic_offender_list.csv", "phase5_chronic_offenders.csv"])


# ------------------------- preparation -------------------------
def prepare_static_context(static_df: pd.DataFrame) -> pd.DataFrame:
    df = static_df.copy()
    df = ensure_label_column(df)
    df = derive_coords(df)
    cluster_col = standardize_cluster_col(df)
    df["cluster_key"] = df[cluster_col].map(normalize_cluster_key)

    defaults = {
        "cluster_label": "UNKNOWN", "cluster_label_display": "", "risk_band": "Watch",
        "ccs_score": 0.0, "delay_minutes_per_vehicle": 0.0, "records_total": 0.0,
        "distinct_days": 0.0, "severity_sum": 0.0, "severity_mean": 1.0,
        "growth_pct_change": 0.0, "growth_multiplier": 1.0, "criticality_factor": 1.0,
        "context_multiplier": 1.0, "layer_b_priority_boost": 0.0, "layer_b_alert_flag": 0,
        "validation_uncertainty": 0.0, "resurgence_score": 0.0, "persistence_score": 0.0,
        "anomaly_score": 0.0, "rop": 0.0, "tvs": 0.0, "vdi": 0.0,
        "nearby_sensitive_poi_count": 0.0, "road_class": "road", "lane_count": 2.0,
        "carriageway_width_m": 7.0, "link_length_m": 250.0, "junction_degree": 4.0,
        "betweenness_centrality": 0.0, "geometry_source": "fallback", "mappls_address": "",
        "road_node_id": np.nan, "road_node_distance_m": 0.0, "road_node_degree": 0.0,
        "road_node_betweenness": 0.0, "source_pressure_score": 0.0, "source_pressure_norm": 0.0,
        "spillover_out_score": 0.0, "spillover_in_score": 0.0, "spillover_total_score": 0.0,
        "propagation_radius_m": 0.0, "network_pagerank": 0.0, "network_component_id": 0,
        "network_component_size": 1, "neighbor_count": 0.0, "in_neighbor_count": 0.0,
        "out_neighbor_count": 0.0, "influence_asymmetry": 0.0, "network_vulnerability_score": 0.0,
        "layer_d_alert_flag": 0, "dominant_vehicle_type": "UNKNOWN",
        "repeat_vehicle_count_2plus": 0.0, "chronic_vehicle_count_5plus": 0.0,
    }
    for c, default in defaults.items():
        if c not in df.columns:
            df[c] = default

    numeric_cols = [
        "ccs_score", "delay_minutes_per_vehicle", "records_total", "distinct_days", "severity_sum", "severity_mean",
        "growth_pct_change", "growth_multiplier", "criticality_factor", "context_multiplier", "layer_b_priority_boost",
        "validation_uncertainty", "resurgence_score", "persistence_score", "anomaly_score", "rop", "tvs", "vdi",
        "nearby_sensitive_poi_count", "lane_count", "carriageway_width_m", "link_length_m", "junction_degree",
        "betweenness_centrality", "road_node_distance_m", "road_node_degree", "road_node_betweenness",
        "source_pressure_score", "source_pressure_norm", "spillover_out_score", "spillover_in_score", "spillover_total_score",
        "propagation_radius_m", "network_pagerank", "network_component_id", "network_component_size",
        "neighbor_count", "in_neighbor_count", "out_neighbor_count", "influence_asymmetry", "network_vulnerability_score",
        "repeat_vehicle_count_2plus", "chronic_vehicle_count_5plus",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["layer_b_alert_flag", "layer_d_alert_flag"]:
        df[c] = safe_to_int_series(df[c], 0)

    for c in ["road_class", "geometry_source", "dominant_vehicle_type"]:
        df[c] = df[c].fillna("UNKNOWN").astype(str).str.strip()

    # Best row per hotspot
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
    r["day_of_week"] = r["created_datetime_ist"].dt.dayofweek
    r["is_peak_window"] = r["hour_ist"].between(8, 12, inclusive="both").astype(int)

    r["cluster_key"] = r[cluster_col].map(normalize_cluster_key)

    if vehicle_col and vehicle_col in r.columns:
        r[vehicle_col] = r[vehicle_col].fillna("").astype(str).str.strip()
        r = r[r[vehicle_col].ne("")].copy()
    else:
        vehicle_col = None

    if vehicle_type_col and vehicle_type_col in r.columns:
        r[vehicle_type_col] = r[vehicle_type_col].fillna("").astype(str).str.strip()
    else:
        vehicle_type_col = None

    agg_spec = {
        "records_week": ("week_start", "size"),
        "active_days_week": ("service_date", "nunique"),
        "severity_sum_week": ("severity_score", "sum"),
        "severity_mean_week": ("severity_score", "mean"),
        "peak_window_records_week": ("is_peak_window", "sum"),
        "unique_hotspots_week": ("hotspot_unit", "nunique"),
    }
    if vehicle_col:
        agg_spec["unique_vehicles_week"] = (vehicle_col, "nunique")
    else:
        r["vehicle_fallback"] = "UNKNOWN"
        agg_spec["unique_vehicles_week"] = ("vehicle_fallback", "nunique")

    if vehicle_type_col:
        agg_spec["unique_vehicle_types_week"] = (vehicle_type_col, "nunique")
        agg_spec["dominant_vehicle_type_week"] = (vehicle_type_col, lambda s: s.mode().iloc[0] if not s.mode().empty else "UNKNOWN")
    else:
        r["vehicle_type_fallback"] = "UNKNOWN"
        agg_spec["unique_vehicle_types_week"] = ("vehicle_type_fallback", "nunique")
        agg_spec["dominant_vehicle_type_week"] = ("vehicle_type_fallback", lambda s: "UNKNOWN")

    weekly = r.groupby(["cluster_key", "week_start"]).agg(**agg_spec).reset_index().sort_values(["cluster_key", "week_start"]).reset_index(drop=True)
    weekly["records_per_active_day_week"] = weekly["records_week"] / weekly["active_days_week"].clip(lower=1)
    weekly["peak_window_ratio_week"] = weekly["peak_window_records_week"] / weekly["records_week"].clip(lower=1)
    weekly["repeat_vehicle_ratio_week"] = ((weekly["records_week"] - weekly["unique_vehicles_week"]) / weekly["records_week"].clip(lower=1)).clip(0.0, 1.0)
    weekly["prev_records_week"] = weekly.groupby("cluster_key")["records_week"].shift(1)
    weekly["weekly_growth_pct"] = ((weekly["records_week"] - weekly["prev_records_week"]) / (weekly["prev_records_week"].abs() + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weekly["growth_surge_week"] = weekly["weekly_growth_pct"].clip(lower=0.0)
    weekly["weekly_pressure_raw"] = pressure_raw_from_week(weekly)

    def enrich_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("week_start").copy()
        g["week_index"] = np.arange(len(g), dtype=int)
        g["hotspot_age_weeks"] = g["week_index"] + 1
        g["lag1_pressure_raw"] = g["weekly_pressure_raw"].shift(1).fillna(g["weekly_pressure_raw"].iloc[0] if len(g) else 0.0)
        g["lag2_pressure_raw"] = g["weekly_pressure_raw"].shift(2).fillna(g["lag1_pressure_raw"])
        g["rolling_4w_mean_pressure_raw"] = g["weekly_pressure_raw"].rolling(4, min_periods=1).mean()
        g["rolling_4w_std_pressure_raw"] = g["weekly_pressure_raw"].rolling(4, min_periods=1).std(ddof=0).fillna(0.0)
        g["rolling_4w_mean_records_week"] = g["records_week"].rolling(4, min_periods=1).mean()
        g["rolling_4w_mean_severity_week"] = g["severity_sum_week"].rolling(4, min_periods=1).mean()
        g["rolling_4w_mean_growth_week"] = g["weekly_growth_pct"].rolling(4, min_periods=1).mean()
        vals = g["records_week"].to_numpy(dtype=float)
        slopes = []
        for i in range(len(g)):
            start = max(0, i - 3)
            y = vals[start:i+1]
            if len(y) < 2 or np.allclose(y, y[0], equal_nan=True):
                slopes.append(0.0)
            else:
                x = np.arange(len(y), dtype=float)
                try:
                    slopes.append(float(np.polyfit(x, y, 1)[0]))
                except Exception:
                    slopes.append(0.0)
        g["rolling_4w_trend_slope_records_week"] = slopes
        g["rolling_4w_pressure_acceleration"] = g["rolling_4w_mean_pressure_raw"] - g["rolling_4w_mean_pressure_raw"].shift(1).fillna(g["rolling_4w_mean_pressure_raw"])
        return g

    weekly = weekly.groupby("cluster_key", group_keys=False).apply(enrich_group).reset_index(drop=True)

    # Validation uncertainty
    if "validation_status" in r.columns:
        val = r.groupby("cluster_key")["validation_status"].agg(
            validation_total="size",
            validation_approved=lambda s: int((s == "approved").sum()),
            validation_rejected=lambda s: int((s == "rejected").sum()),
            validation_processing=lambda s: int((s == "processing").sum()),
        ).reset_index()
        val["validation_uncertainty"] = 1.0 - (val["validation_approved"] / val["validation_total"].clip(lower=1))
    else:
        val = pd.DataFrame({"cluster_key": weekly["cluster_key"].unique(), "validation_total": 0, "validation_uncertainty": 0.0})

    # Repeat offenders
    if vehicle_col:
        repeat_stats = r.groupby(["cluster_key", vehicle_col]).size().reset_index(name="vehicle_cluster_count")
        repeat_stats["repeat_flag_2plus"] = (repeat_stats["vehicle_cluster_count"] >= 2).astype(int)
        repeat_stats["chronic_flag_5plus"] = (repeat_stats["vehicle_cluster_count"] >= 5).astype(int)
        repeat_agg = repeat_stats.groupby("cluster_key").agg(
            repeat_vehicle_count_2plus=("repeat_flag_2plus", "sum"),
            chronic_vehicle_count_5plus=("chronic_flag_5plus", "sum"),
        ).reset_index()
    else:
        repeat_agg = pd.DataFrame({"cluster_key": weekly["cluster_key"].unique(), "repeat_vehicle_count_2plus": 0, "chronic_vehicle_count_5plus": 0})

    # Diversity (VDI)
    if "violation_tags" in r.columns:
        tag_rows = r[["cluster_key", "violation_tags"]].explode("violation_tags").copy()
        tag_rows["violation_tag"] = tag_rows["violation_tags"].map(normalize_tag)
        tag_rows = tag_rows[tag_rows["violation_tag"].astype(str).str.strip().ne("")]
        diversity = tag_rows.groupby("cluster_key")["violation_tag"].apply(compute_shannon_diversity_from_tags).reset_index(name="vdi")
    else:
        diversity = pd.DataFrame({"cluster_key": weekly["cluster_key"].unique(), "vdi": 0.0})

    # Summary stats per hotspot
    rows = []
    for cid, g in weekly.groupby("cluster_key"):
        counts = g["records_week"].to_numpy(dtype=float)
        if len(counts) < 2:
            tvs = 0.0
            resurgence = 0.0
            persistence = 1.0
        else:
            tvs = float(np.std(counts, ddof=0) / (np.mean(counts) + EPS))
            split = max(1, len(counts) // 2)
            first = float(np.mean(counts[:split]))
            second = float(np.mean(counts[split:])) if len(counts[split:]) else first
            resurgence = max(0.0, (second - first) / (first + EPS))
            persistence = float(np.mean(counts[-3:]) / (np.mean(counts[:3]) + EPS)) if len(counts) >= 3 else 1.0
        recent2 = float(np.mean(counts[-2:])) if len(counts) >= 2 else float(np.mean(counts)) if len(counts) else 0.0
        prev2 = float(np.mean(counts[-4:-2])) if len(counts) >= 4 else float(np.mean(counts[:-2])) if len(counts) > 2 else recent2
        resurgence_score = max(0.0, (recent2 - prev2) / (prev2 + EPS)) if prev2 > 0 else recent2

        hours = r.loc[r["cluster_key"].eq(cid), "hour_ist"]
        days = r.loc[r["cluster_key"].eq(cid), "day_of_week"]
        hour_counts = hours.value_counts().reindex(range(24), fill_value=0).astype(float)
        day_counts = days.value_counts().reindex(range(7), fill_value=0).astype(float)
        peak_hour = int(hour_counts.idxmax()) if hour_counts.sum() > 0 else 8
        rolling3 = hour_counts.rolling(3, min_periods=1).sum()
        start_hour = int(rolling3.idxmax()) if len(rolling3) else 8
        end_hour = min(24, start_hour + 3)
        best_day = int(day_counts.idxmax()) if day_counts.sum() > 0 else 0

        rows.append({
            "cluster_key": cid,
            "weekly_rows": len(g),
            "records_week_mean": float(np.mean(counts)) if len(counts) else 0.0,
            "records_week_std": float(np.std(counts, ddof=0)) if len(counts) else 0.0,
            "weekly_growth_pct_mean": float(g["weekly_growth_pct"].mean()),
            "weekly_growth_pct_max": float(g["weekly_growth_pct"].max()),
            "weekly_pressure_raw_mean": float(g["weekly_pressure_raw"].mean()),
            "weekly_pressure_raw_max": float(g["weekly_pressure_raw"].max()),
            "rolling_4w_mean_pressure_raw_max": float(g["rolling_4w_mean_pressure_raw"].max()),
            "rolling_4w_std_pressure_raw_max": float(g["rolling_4w_std_pressure_raw"].max()),
            "rolling_4w_mean_records_week_max": float(g["rolling_4w_mean_records_week"].max()),
            "rolling_4w_mean_severity_week_max": float(g["rolling_4w_mean_severity_week"].max()),
            "rolling_4w_trend_slope_records_week_max": float(g["rolling_4w_trend_slope_records_week"].max()),
            "rolling_4w_pressure_acceleration_max": float(g["rolling_4w_pressure_acceleration"].max()),
            "peak_hour": peak_hour,
            "best_window_start_hour": start_hour,
            "best_window_end_hour": end_hour,
            "best_day_of_week": best_day,
            "best_day_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][best_day],
            "peak_hours_top3": ",".join(str(i) for i in hour_counts.sort_values(ascending=False).head(3).index.tolist()),
            "peak_window_violation_ratio": float(hour_counts.iloc[start_hour:end_hour].sum() / (hour_counts.sum() + EPS)),
            "tvs": float(tvs),
            "resurgence_score": float(np.clip(resurgence_score, 0.0, 5.0)),
            "persistence_score": float(np.clip(persistence, 0.0, 5.0)),
        })

    summary = pd.DataFrame(rows).merge(val[["cluster_key", "validation_total", "validation_uncertainty"]], on="cluster_key", how="left")
    summary = summary.merge(repeat_agg, on="cluster_key", how="left")
    summary = summary.merge(diversity, on="cluster_key", how="left")

    static_ctx = static_df.copy()
    static_ctx["cluster_key"] = static_ctx[standardize_cluster_col(static_ctx)].map(normalize_cluster_key)
    static_cols = [
        "cluster_key", "lat", "lon", "cluster_label", "cluster_label_display", "risk_band", "ccs_score",
        "delay_minutes_per_vehicle", "records_total", "distinct_days", "severity_sum", "severity_mean",
        "growth_pct_change", "growth_multiplier", "criticality_factor", "context_multiplier",
        "layer_b_priority_boost", "layer_b_alert_flag", "validation_uncertainty", "resurgence_score",
        "persistence_score", "anomaly_score", "rop", "tvs", "vdi", "nearby_sensitive_poi_count",
        "road_class", "lane_count", "carriageway_width_m", "link_length_m", "junction_degree",
        "betweenness_centrality", "geometry_source", "mappls_address", "road_node_id", "road_node_distance_m",
        "road_node_degree", "road_node_betweenness", "source_pressure_score", "source_pressure_norm",
        "spillover_out_score", "spillover_in_score", "spillover_total_score", "propagation_radius_m",
        "network_pagerank", "network_component_id", "network_component_size", "neighbor_count",
        "in_neighbor_count", "out_neighbor_count", "influence_asymmetry", "network_vulnerability_score",
        "layer_d_alert_flag", "dominant_vehicle_type", "repeat_vehicle_count_2plus", "chronic_vehicle_count_5plus",
        "dominant_police_station", "police_station", "dominant_junction_name", "hotspot_unit",
    ]
    static_cols = [c for c in static_cols if c in static_ctx.columns]
    static_ctx = static_ctx[static_cols].drop_duplicates("cluster_key")
    static_ctx = static_ctx.rename(columns={c: f"static_{c}" for c in static_ctx.columns if c != "cluster_key"})
    summary = summary.merge(static_ctx, on="cluster_key", how="left")
    summary = coalesce_from_suffix(summary, [c for c in summary.columns if c.startswith("static_")], suffix="")  # no-op safe
    return weekly, summary, cluster_col


# ------------------------- prescriptive logic -------------------------
def compute_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    components = {
        "probability_of_escalation": 0.22,
        "predicted_next_week_ccs_score": 0.18,
        "current_week_ccs_proxy_score": 0.08,
        "criticality_factor": 0.08,
        "growth_multiplier": 0.07,
        "growth_pct_change": 0.05,
        "layer_b_priority_boost": 0.05,
        "validation_uncertainty": 0.05,
        "rop": 0.08,
        "tvs": 0.06,
        "vdi": 0.05,
        "resurgence_score": 0.05,
        "anomaly_score": 0.04,
        "network_vulnerability_score": 0.04,
        "spillover_total_score": 0.01,
        "nearby_sensitive_poi_count": 0.01,
    }
    for c in components:
        if c not in out.columns:
            out[c] = 0.0
        out[f"{c}_norm"] = minmax(pd.to_numeric(out[c], errors="coerce").fillna(0.0))
    raw = np.zeros(len(out), dtype=float)
    total_w = float(sum(components.values()))
    for c, w in components.items():
        raw += w * out[f"{c}_norm"].to_numpy(dtype=float)
    out["priority_score"] = 100.0 * np.clip(raw / max(total_w, EPS), 0.0, 1.0)
    out["priority_band"] = out["priority_score"].apply(band_from_score)
    out["final_risk_band"] = out.apply(
        lambda row: row["priority_band"] if band_index(row["priority_band"]) >= band_index(row.get("future_risk_band", "Watch")) else row.get("future_risk_band", "Watch"),
        axis=1,
    )
    return out


def compute_time_window(row):
    """
    Returns:
        recommended_time_window,
        recommended_day_name,
        recommended_shift
    """
    start_hour = row.get("peak_start_hour", np.nan)
    end_hour = row.get("peak_end_hour", np.nan)
    # fallback to peak hour
    if pd.isna(start_hour) or pd.isna(end_hour):
        peak_hour = row.get("peak_hour", np.nan)
        if pd.isna(peak_hour):
            peak_hour = 8
        peak_hour = int(round(float(peak_hour)))
        start_hour = max(0, peak_hour - 1)
        end_hour = min(24, peak_hour + 2)
    start_hour = int(start_hour)
    end_hour = int(end_hour)
    time_window = (
        f"{start_hour:02d}:00-{end_hour:02d}:00"
    )
    day_name = row.get("peak_day_name", "All Days")
    # shift classification
    if start_hour < 12:
        shift = "Morning"
    elif start_hour < 17:
        shift = "Afternoon"
    else:
        shift = "Evening"
    return (
        time_window,
        day_name,
        shift
    )


def allocate_patrols(df: pd.DataFrame, total_units: int = TOTAL_PATROL_UNITS) -> pd.DataFrame:
    out = df.copy()
    out["recommended_patrol_count"] = 0
    eligible = out["priority_score"] >= PATROL_ELIGIBILITY_SCORE
    if eligible.sum() == 0:
        eligible = out.sort_values("priority_score", ascending=False).head(min(total_units, len(out))).index
    eligible_idx = list(out.index[eligible]) if not isinstance(eligible, pd.Index) else list(eligible)

    if len(eligible_idx) <= total_units:
        out.loc[eligible_idx, "recommended_patrol_count"] = 1
        remain = total_units - len(eligible_idx)
        if remain > 0:
            weights = pd.to_numeric(out.loc[eligible_idx, "priority_score"], errors="coerce").fillna(0.0)
            weights = weights / max(weights.sum(), EPS)
            extra = weights * remain
            add = np.floor(extra).astype(int)
            out.loc[eligible_idx, "recommended_patrol_count"] += add.values
            leftover = remain - int(add.sum())
            if leftover > 0:
                frac = (extra - add).sort_values(ascending=False)
                for idx in frac.index[:leftover]:
                    out.at[idx, "recommended_patrol_count"] += 1
    else:
        top = out.sort_values("priority_score", ascending=False).head(total_units).index.tolist()
        out.loc[top, "recommended_patrol_count"] = 1

    out["recommended_patrol_count"] = pd.to_numeric(out["recommended_patrol_count"], errors="coerce").fillna(0).astype(int).clip(0, MAX_PATROLS_PER_HOTSPOT)

    critical_mask = out["final_risk_band"].isin(["Critical", "High"]) & (out["recommended_patrol_count"] == 0)
    for idx in out[critical_mask].sort_values("priority_score", ascending=False).index:
        if int(out["recommended_patrol_count"].sum()) >= total_units:
            break
        out.at[idx, "recommended_patrol_count"] = 1
    return out


def compute_tow_readiness(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    road = out["road_class"].fillna("road").astype(str).str.lower()
    arterial = road.isin({"motorway", "trunk", "primary", "secondary", "tertiary", "road"})
    width = pd.to_numeric(out["carriageway_width_m"], errors="coerce").fillna(7.0)
    lanes = pd.to_numeric(out["lane_count"], errors="coerce").fillna(2)
    bet = pd.to_numeric(out["betweenness_centrality"], errors="coerce").fillna(0.0)
    bet_hi = bet >= bet.quantile(HIGH_BETWEENNESS_PERCENTILE) if len(bet) else pd.Series(False, index=out.index)
    chronic = pd.to_numeric(out.get("chronic_vehicle_count_5plus", 0), errors="coerce").fillna(0)
    repeat2 = pd.to_numeric(out.get("repeat_vehicle_count_2plus", 0), errors="coerce").fillna(0)
    esc = pd.to_numeric(out["probability_of_escalation"], errors="coerce").fillna(0.0)
    pscore = pd.to_numeric(out["priority_score"], errors="coerce").fillna(0.0)

    tow = ((pscore >= TOW_PRIORITY_THRESHOLD) | out["final_risk_band"].isin(["Critical", "High"]) | (esc >= ESCALATION_THRESHOLD)) & (arterial | (width <= 7.5) | (lanes <= 2) | bet_hi) & ((chronic >= 1) | (repeat2 >= 3) | (pscore >= 80))
    out["tow_required"] = tow.astype(bool)
    out["tow_reason"] = np.where(out["tow_required"], "High-risk corridor with repeat/chronic pressure", "No tow required")
    return out


def build_enforcement_action(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def action(row):
        b = str(row.get("final_risk_band", "Watch"))
        tow = bool(row.get("tow_required", False))
        patrols = int(row.get("recommended_patrol_count", 0) or 0)
        if b == "Critical":
            return "Immediate patrol + tow standby + camera watch" if tow else "Immediate patrol deployment"
        if b == "High":
            return "Targeted patrol + tow readiness" if tow else "Targeted patrol + close monitoring"
        if b == "Moderate":
            return "Scheduled patrol + peak-window checks" if patrols >= 2 else "Peak-window checks"
        return "Routine monitoring"

    out["enforcement_action"] = out.apply(action, axis=1)
    out["dispatch_notes"] = out.apply(
        lambda row: " | ".join([
            f"forecast={safe_float(row.get('predicted_next_week_ccs_score', 0.0), 0.0):.2f}",
            f"esc_prob={safe_float(row.get('probability_of_escalation', 0.0), 0.0):.2f}",
            f"priority={safe_float(row.get('priority_score', 0.0), 0.0):.1f}",
            f"patrols={int(row.get('recommended_patrol_count', 0) or 0)}" if int(row.get("recommended_patrol_count", 0) or 0) > 0 else "patrols=0",
            f"tow={'yes' if bool(row.get('tow_required', False)) else 'no'}",
            f"window={clean_text(row.get('recommended_time_window', ''))}",
        ]),
        axis=1,
    )
    return out


def nearest_neighbor_route(df: pd.DataFrame, depot_lat: float = DEPOT_LAT, depot_lon: float = DEPOT_LON, top_n: int = ROUTE_PLAN_TOP_N) -> pd.DataFrame:
    cand = df[df["recommended_patrol_count"] > 0].copy()
    cand = cand.dropna(subset=["lat", "lon"]).copy()
    if len(cand) == 0:
        return pd.DataFrame(columns=["route_id", "stop_sequence", "cluster_key", "cluster_label_display", "lat", "lon", "distance_from_prev_m", "cumulative_distance_m", "priority_score"])

    cand = cand.sort_values(["priority_score", "probability_of_escalation"], ascending=[False, False]).head(top_n).copy().reset_index(drop=True)
    remaining = cand.copy()
    rows = []
    cur_lat, cur_lon = float(depot_lat), float(depot_lon)
    cum = 0.0
    seq = 0
    while len(remaining):
        dists = remaining.apply(lambda r: haversine_m(cur_lat, cur_lon, r["lat"], r["lon"]), axis=1)
        idx = dists.idxmin()
        row = remaining.loc[idx]
        d = float(dists.loc[idx]) if pd.notna(dists.loc[idx]) else 0.0
        cum += d
        seq += 1
        rows.append({
            "route_id": "route_1",
            "stop_sequence": seq,
            "cluster_key": row["cluster_key"],
            "cluster_label_display": row.get("cluster_label_display", row.get("cluster_label", "")),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "distance_from_prev_m": d,
            "cumulative_distance_m": cum,
            "priority_score": float(row["priority_score"]),
        })
        cur_lat, cur_lon = float(row["lat"]), float(row["lon"])
        remaining = remaining.drop(index=idx)
    return pd.DataFrame(rows)


def build_offender_actions(offenders_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if offenders_df is None or len(offenders_df) == 0:
        return pd.DataFrame(columns=["vehicle_number", "total_violations", "unique_clusters", "unique_hotspots", "first_seen", "last_seen", "dominant_vehicle_type", "offender_pressure_score", "offender_priority_band", "recommended_action"])

    out = offenders_df.copy()
    for c in ["vehicle_number", "dominant_vehicle_type"]:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str).str.strip()
    for c in ["total_violations", "unique_clusters", "unique_hotspots"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    if "first_seen" in out.columns:
        out["first_seen"] = pd.to_datetime(out["first_seen"], errors="coerce")
    if "last_seen" in out.columns:
        out["last_seen"] = pd.to_datetime(out["last_seen"], errors="coerce")

    out["offender_pressure_raw"] = np.log1p(pd.to_numeric(out.get("total_violations", 0), errors="coerce").fillna(0).clip(lower=0)) * (1.0 + minmax(pd.to_numeric(out.get("unique_hotspots", 0), errors="coerce").fillna(0))) * (1.0 + minmax(pd.to_numeric(out.get("unique_clusters", 0), errors="coerce").fillna(0)))
    out["offender_pressure_score"] = 100.0 * minmax(out["offender_pressure_raw"].fillna(0.0))
    out["offender_priority_band"] = out["offender_pressure_score"].apply(band_from_score)
    out["recommended_action"] = np.select(
        [
            out["offender_priority_band"].eq("Critical"),
            out["offender_priority_band"].eq("High"),
            out["offender_priority_band"].eq("Moderate"),
        ],
        [
            "Special watch + field verification",
            "Targeted surveillance + repeat checks",
            "Monitor vehicle pattern",
        ],
        default="Routine tracking",
    )
    return out.sort_values(["offender_pressure_score", "total_violations"], ascending=[False, False]).reset_index(drop=True)


# ------------------------- main -------------------------
def main():
    print("Loading Phase 7 forecast...")
    phase7_df, phase7_src = load_phase7_forecast()
    print("Phase 7 source:", phase7_src)

    print("Loading Phase 6 static context...")
    phase6_df, phase6_src = load_phase6_static()
    print("Phase 6 source:", phase6_src)

    print("Loading record-level history...")
    records_df, records_src = load_record_history()
    print("Record source:", records_src)

    offenders_df, offenders_src = load_chronic_offenders()

    static_df = prepare_static_context(phase6_df)

    forecast = phase7_df.copy()
    forecast = ensure_label_column(forecast)
    forecast = derive_coords(forecast)
    cluster_col = standardize_cluster_col(forecast)
    forecast["cluster_key"] = forecast[cluster_col].map(normalize_cluster_key)

    # If static coords/context exist, merge them as suffix columns and coalesce.
    merge_cols = [
        "cluster_key", "lat", "lon", "road_class", "lane_count", "carriageway_width_m", "link_length_m",
        "junction_degree", "betweenness_centrality", "criticality_factor", "context_multiplier",
        "layer_b_priority_boost", "layer_b_alert_flag", "validation_uncertainty", "resurgence_score",
        "persistence_score", "anomaly_score", "rop", "tvs", "vdi", "nearby_sensitive_poi_count",
        "geometry_source", "mappls_address", "road_node_id", "road_node_distance_m", "road_node_degree",
        "road_node_betweenness", "source_pressure_score", "source_pressure_norm", "spillover_out_score",
        "spillover_in_score", "spillover_total_score", "propagation_radius_m", "network_pagerank",
        "network_component_id", "network_component_size", "neighbor_count", "in_neighbor_count",
        "out_neighbor_count", "influence_asymmetry", "network_vulnerability_score", "layer_d_alert_flag",
        "dominant_vehicle_type", "cluster_label", "cluster_label_display", "risk_band", "ccs_score",
        "delay_minutes_per_vehicle", "records_total", "distinct_days", "severity_sum", "severity_mean",
        "growth_pct_change", "growth_multiplier", "repeat_vehicle_count_2plus", "chronic_vehicle_count_5plus",
        "dominant_police_station", "police_station", "dominant_junction_name", "hotspot_unit",
    ]
    merge_cols = [c for c in merge_cols if c in static_df.columns]
    static_lookup = static_df[merge_cols].drop_duplicates("cluster_key").rename(columns={c: f"static_{c}" for c in merge_cols if c != "cluster_key"})
    forecast = forecast.merge(static_lookup, on="cluster_key", how="left")
    forecast = coalesce_from_suffix(forecast, [c for c in merge_cols if c != "cluster_key"], suffix="_static")

    # Weekly history + novel features
    weekly_hist, hist_summary, _ = prepare_record_history(records_df, static_df)
    print("Weekly rows built:", len(weekly_hist))
    print("Hotspots in weekly history:", weekly_hist["cluster_key"].nunique())

    week_counts = weekly_hist.groupby("cluster_key")["week_start"].nunique()
    print("\nWeeks of history per hotspot (describe):")
    print(week_counts.describe().to_string())
    single_week_share = float((week_counts == 1).mean() * 100.0) if len(week_counts) else 0.0
    print(f"Share of hotspots with only ONE observed week: {single_week_share:.1f}% -- weekly_growth_pct == 0.0 for these is expected, not a bug.")

    # Latest history row per hotspot.
    hist_latest = weekly_hist.sort_values(["cluster_key", "week_start"]).groupby("cluster_key", as_index=False).tail(1).copy()
    hist_cols = [c for c in ["cluster_key", "peak_hour", "best_window_start_hour", "best_window_end_hour", "best_day_of_week", "best_day_name", "peak_hours_top3", "peak_window_violation_ratio", "tvs", "resurgence_score", "persistence_score", "validation_total", "validation_uncertainty", "repeat_vehicle_count_2plus", "chronic_vehicle_count_5plus", "vdi", "weekly_rows", "records_week_mean", "records_week_std", "severity_sum_week_mean", "peak_window_records_week_mean", "unique_vehicles_week_mean", "unique_vehicle_types_week_mean", "weekly_growth_pct_mean", "weekly_growth_pct_max", "weekly_pressure_raw_mean", "weekly_pressure_raw_max", "rolling_4w_mean_pressure_raw_max", "rolling_4w_std_pressure_raw_max", "rolling_4w_mean_records_week_max", "rolling_4w_mean_severity_week_max", "rolling_4w_trend_slope_records_week_max", "rolling_4w_pressure_acceleration_max"] if c in hist_latest.columns]
    hist_latest = hist_latest[hist_cols].drop_duplicates("cluster_key").rename(columns={c: f"hist_{c}" for c in hist_cols if c != "cluster_key"})

    forecast = forecast.merge(hist_latest, on="cluster_key", how="left")
    forecast = coalesce_from_suffix(forecast, [c for c in hist_cols if c != "cluster_key"], suffix="_hist")

    # Use history values as fallbacks.
    for c in ["peak_hour", "best_window_start_hour", "best_window_end_hour", "best_day_name", "peak_hours_top3", "peak_window_violation_ratio", "tvs", "resurgence_score", "persistence_score", "validation_total", "validation_uncertainty", "repeat_vehicle_count_2plus", "chronic_vehicle_count_5plus", "vdi", "weekly_rows", "records_week_mean", "records_week_std", "severity_sum_week_mean", "peak_window_records_week_mean", "unique_vehicles_week_mean", "unique_vehicle_types_week_mean", "weekly_growth_pct_mean", "weekly_growth_pct_max", "weekly_pressure_raw_mean", "weekly_pressure_raw_max", "rolling_4w_mean_pressure_raw_max", "rolling_4w_std_pressure_raw_max", "rolling_4w_mean_records_week_max", "rolling_4w_mean_severity_week_max", "rolling_4w_trend_slope_records_week_max", "rolling_4w_pressure_acceleration_max", "dominant_vehicle_type", "cluster_label", "cluster_label_display", "risk_band", "ccs_score", "delay_minutes_per_vehicle", "records_total", "distinct_days", "severity_sum", "severity_mean", "growth_pct_change", "growth_multiplier", "criticality_factor", "context_multiplier", "layer_b_priority_boost", "layer_b_alert_flag", "geometry_source", "road_class", "lane_count", "carriageway_width_m", "link_length_m", "junction_degree", "betweenness_centrality", "road_node_distance_m", "road_node_degree", "road_node_betweenness", "source_pressure_score", "source_pressure_norm", "spillover_out_score", "spillover_in_score", "spillover_total_score", "propagation_radius_m", "network_pagerank", "network_component_id", "network_component_size", "neighbor_count", "in_neighbor_count", "out_neighbor_count", "influence_asymmetry", "network_vulnerability_score"]:
        if c not in forecast.columns:
            forecast[c] = np.nan

    forecast["zone_name"] = build_zone_name(forecast)

    # Score + actions
    components = {
        "probability_of_escalation": 0.22,
        "predicted_next_week_ccs_score": 0.18,
        "current_week_ccs_proxy_score": 0.08,
        "criticality_factor": 0.08,
        "growth_multiplier": 0.07,
        "growth_pct_change": 0.05,
        "layer_b_priority_boost": 0.05,
        "validation_uncertainty": 0.05,
        "rop": 0.08,
        "tvs": 0.06,
        "vdi": 0.05,
        "resurgence_score": 0.05,
        "anomaly_score": 0.04,
        "network_vulnerability_score": 0.04,
        "spillover_total_score": 0.01,
        "nearby_sensitive_poi_count": 0.01,
    }
    for c in components:
        forecast[c] = pd.to_numeric(forecast.get(c, 0.0), errors="coerce").fillna(0.0)
        forecast[f"{c}_norm"] = minmax(forecast[c])

    raw = np.zeros(len(forecast), dtype=float)
    total_w = float(sum(components.values()))
    for c, w in components.items():
        raw += w * forecast[f"{c}_norm"].to_numpy(dtype=float)
    forecast["priority_score"] = 100.0 * np.clip(raw / max(total_w, EPS), 0.0, 1.0)
    forecast["priority_band"] = forecast["priority_score"].apply(band_from_score)
    forecast["final_risk_band"] = forecast.apply(
        lambda row: row["priority_band"] if band_index(row["priority_band"]) >= band_index(row.get("future_risk_band", "Watch")) else row.get("future_risk_band", "Watch"),
        axis=1,
    )

    # Time windows
    tw = forecast.apply(lambda r: pd.Series(compute_time_window(r), index=["recommended_time_window", "recommended_day_name", "recommended_shift"]), axis=1)
    forecast = pd.concat([forecast, tw], axis=1)

    # Tow + patrol + actions
    road = forecast["road_class"].fillna("road").astype(str).str.lower()
    arterial = road.isin({"motorway", "trunk", "primary", "secondary", "tertiary", "road"})
    width = pd.to_numeric(forecast["carriageway_width_m"], errors="coerce").fillna(7.0)
    lanes = pd.to_numeric(forecast["lane_count"], errors="coerce").fillna(2)
    bet = pd.to_numeric(forecast["betweenness_centrality"], errors="coerce").fillna(0.0)
    bet_hi = bet >= bet.quantile(HIGH_BETWEENNESS_PERCENTILE) if len(bet) else pd.Series(False, index=forecast.index)
    chronic = pd.to_numeric(forecast.get("chronic_vehicle_count_5plus", 0), errors="coerce").fillna(0)
    repeat2 = pd.to_numeric(forecast.get("repeat_vehicle_count_2plus", 0), errors="coerce").fillna(0)
    esc = pd.to_numeric(forecast["probability_of_escalation"], errors="coerce").fillna(0.0)
    pscore = pd.to_numeric(forecast["priority_score"], errors="coerce").fillna(0.0)

    forecast["tow_required"] = (((pscore >= TOW_PRIORITY_THRESHOLD) | forecast["final_risk_band"].isin(["Critical", "High"]) | (esc >= ESCALATION_THRESHOLD)) & (arterial | (width <= 7.5) | (lanes <= 2) | bet_hi) & ((chronic >= 1) | (repeat2 >= 3) | (pscore >= 80))).astype(bool)
    forecast["tow_reason"] = np.where(forecast["tow_required"], "High-risk corridor with repeat/chronic pressure", "No tow required")

    forecast = forecast.sort_values(["priority_score", "probability_of_escalation", "predicted_next_week_ccs_score"], ascending=[False, False, False]).reset_index(drop=True)
    forecast["dispatch_rank"] = np.arange(1, len(forecast) + 1)

    # Patrol allocation
    forecast["recommended_patrol_count"] = 0
    eligible = forecast["priority_score"] >= PATROL_ELIGIBILITY_SCORE
    if eligible.sum() == 0:
        eligible = forecast.index.isin(forecast.head(min(TOTAL_PATROL_UNITS, len(forecast))).index)
    eligible_idx = list(forecast.index[eligible]) if not isinstance(eligible, pd.Index) else list(eligible)
    if len(eligible_idx) <= TOTAL_PATROL_UNITS:
        forecast.loc[eligible_idx, "recommended_patrol_count"] = 1
        remain = TOTAL_PATROL_UNITS - len(eligible_idx)
        if remain > 0:
            w = pd.to_numeric(forecast.loc[eligible_idx, "priority_score"], errors="coerce").fillna(0.0)
            w = w / max(w.sum(), EPS)
            extra = w * remain
            add = np.floor(extra).astype(int)
            forecast.loc[eligible_idx, "recommended_patrol_count"] += add.values
            leftover = remain - int(add.sum())
            if leftover > 0:
                frac = (extra - add).sort_values(ascending=False)
                for idx in frac.index[:leftover]:
                    forecast.at[idx, "recommended_patrol_count"] += 1
    else:
        top = forecast.head(TOTAL_PATROL_UNITS).index.tolist()
        forecast.loc[top, "recommended_patrol_count"] = 1
    forecast["recommended_patrol_count"] = pd.to_numeric(forecast["recommended_patrol_count"], errors="coerce").fillna(0).astype(int).clip(0, MAX_PATROLS_PER_HOTSPOT)

    # Action text
    def action(row):
        b = str(row.get("final_risk_band", "Watch"))
        tow = bool(row.get("tow_required", False))
        patrols = int(row.get("recommended_patrol_count", 0) or 0)
        if b == "Critical":
            return "Immediate patrol + tow standby + camera watch" if tow else "Immediate patrol deployment"
        if b == "High":
            return "Targeted patrol + tow readiness" if tow else "Targeted patrol + close monitoring"
        if b == "Moderate":
            return "Scheduled patrol + peak-window checks" if patrols >= 2 else "Peak-window checks"
        return "Routine monitoring"
    forecast["enforcement_action"] = forecast.apply(action, axis=1)

    forecast["dispatch_notes"] = forecast.apply(
        lambda row: " | ".join([
            f"forecast={safe_float(row.get('predicted_next_week_ccs_score', 0.0), 0.0):.2f}",
            f"esc_prob={safe_float(row.get('probability_of_escalation', 0.0), 0.0):.2f}",
            f"priority={safe_float(row.get('priority_score', 0.0), 0.0):.1f}",
            f"patrols={int(row.get('recommended_patrol_count', 0) or 0)}",
            f"tow={'yes' if bool(row.get('tow_required', False)) else 'no'}",
            f"window={clean_text(row.get('recommended_time_window', ''))}",
        ]),
        axis=1,
    )

    # Zone summary
    zone_summary = forecast.groupby("zone_name").agg(
        zone_hotspots=("cluster_key", "size"),
        zone_priority_total=("priority_score", "sum"),
        zone_priority_mean=("priority_score", "mean"),
        zone_priority_max=("priority_score", "max"),
        zone_ccs_mean=("predicted_next_week_ccs_score", "mean"),
        zone_ccs_max=("predicted_next_week_ccs_score", "max"),
        zone_escalation_mean=("probability_of_escalation", "mean"),
        zone_tow_count=("tow_required", "sum"),
        zone_patrol_units=("recommended_patrol_count", "sum"),
        zone_critical_count=("final_risk_band", lambda s: int((s == "Critical").sum())),
        zone_high_count=("final_risk_band", lambda s: int((s == "High").sum())),
        zone_top_label=("cluster_label_display", dominant_label),
    ).reset_index()
    zone_summary["zone_rank"] = zone_summary["zone_priority_total"].rank(method="dense", ascending=False).astype(int)
    zone_summary = zone_summary.sort_values(["zone_priority_total", "zone_priority_max"], ascending=[False, False]).reset_index(drop=True)

    # Route plan (heuristic, top assigned hotspots)
    route_candidates = forecast[forecast["recommended_patrol_count"] > 0].copy()
    route_candidates = route_candidates.dropna(subset=["lat", "lon"]).copy()
    if len(route_candidates):
        route_candidates = route_candidates.sort_values(["priority_score", "probability_of_escalation"], ascending=[False, False]).head(ROUTE_PLAN_TOP_N).reset_index(drop=True)
        remaining = route_candidates.copy()
        route_rows = []
        cur_lat, cur_lon = float(DEPOT_LAT), float(DEPOT_LON)
        cum = 0.0
        seq = 0
        while len(remaining):
            dists = remaining.apply(lambda r: haversine_m(cur_lat, cur_lon, r["lat"], r["lon"]), axis=1)
            idx = dists.idxmin()
            row = remaining.loc[idx]
            d = float(dists.loc[idx]) if pd.notna(dists.loc[idx]) else 0.0
            cum += d
            seq += 1
            route_rows.append({
                "route_id": "route_1",
                "stop_sequence": seq,
                "cluster_key": row["cluster_key"],
                "cluster_label_display": row.get("cluster_label_display", row.get("cluster_label", "")),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "distance_from_prev_m": d,
                "cumulative_distance_m": cum,
                "priority_score": float(row["priority_score"]),
            })
            cur_lat, cur_lon = float(row["lat"]), float(row["lon"])
            remaining = remaining.drop(index=idx)
        route_plan = pd.DataFrame(route_rows)
    else:
        route_plan = pd.DataFrame(columns=["route_id", "stop_sequence", "cluster_key", "cluster_label_display", "lat", "lon", "distance_from_prev_m", "cumulative_distance_m", "priority_score"])

    # Offenders
    offender_actions = build_offender_actions(offenders_df)

    # Final tables
    dispatch_cols = [
        "dispatch_rank", "cluster_key", "cluster_label", "cluster_label_display", "zone_name",
        "final_risk_band", "future_risk_band", "priority_band", "priority_score",
        "probability_of_escalation", "predicted_next_week_ccs_score", "current_week_ccs_proxy_score",
        "predicted_delta_score", "predicted_delta_pct", "recommended_patrol_count", "tow_required",
        "tow_reason", "recommended_time_window", "recommended_day_name", "recommended_shift",
        "enforcement_action", "forecast_recommendation", "road_class", "geometry_source",
        "dominant_vehicle_type", "records_week", "weekly_growth_pct", "validation_uncertainty",
        "repeat_vehicle_count_2plus", "chronic_vehicle_count_5plus", "vdi", "tvs",
        "resurgence_score", "persistence_score", "criticality_factor", "context_multiplier",
        "layer_b_priority_boost", "layer_b_alert_flag", "nearby_sensitive_poi_count",
        "lane_count", "carriageway_width_m", "link_length_m", "junction_degree",
        "betweenness_centrality", "road_node_distance_m", "road_node_degree",
        "road_node_betweenness", "source_pressure_score", "source_pressure_norm",
        "spillover_out_score", "spillover_in_score", "spillover_total_score",
        "propagation_radius_m", "network_pagerank", "network_component_id",
        "network_component_size", "neighbor_count", "in_neighbor_count", "out_neighbor_count",
        "influence_asymmetry", "network_vulnerability_score", "dispatch_notes", "lat", "lon",
    ]
    dispatch_cols = [c for c in dispatch_cols if c in forecast.columns]
    dispatch_sheet = forecast[dispatch_cols].copy()

    if "forecast_recommendation" not in dispatch_sheet.columns:
        dispatch_sheet["forecast_recommendation"] = np.select(
            [
                dispatch_sheet["future_risk_band"].eq("Critical"),
                dispatch_sheet["future_risk_band"].eq("High"),
                dispatch_sheet["future_risk_band"].eq("Moderate"),
            ],
            [
                "Immediate patrol deployment",
                "Targeted enforcement + tow readiness",
                "Monitor and schedule peak-window checks",
            ],
            default="Routine monitoring",
        )

    allocation_output = dispatch_sheet[dispatch_sheet["recommended_patrol_count"] > 0].copy().reset_index(drop=True)

    # Save outputs
    dispatch_sheet.to_csv(PHASE8_OUT_DIR / "phase8_dispatch_sheet.csv", index=False)
    allocation_output.to_csv(PHASE8_OUT_DIR / "phase8_allocated_hotspots.csv", index=False)
    zone_summary.to_csv(PHASE8_OUT_DIR / "phase8_zone_summary.csv", index=False)
    route_plan.to_csv(PHASE8_OUT_DIR / "phase8_route_plan.csv", index=False)
    offender_actions.to_csv(PHASE8_OUT_DIR / "phase8_chronic_offender_actions.csv", index=False)

    summary_out = pd.DataFrame([{
        "phase7_source": str(phase7_src),
        "phase6_source": str(phase6_src),
        "records_source": str(records_src),
        "offenders_source": str(offenders_src) if offenders_src else "",
        "hotspots_forecasted": int(len(dispatch_sheet)),
        "hotspots_allocated": int(len(allocation_output)),
        "total_patrol_units_requested": int(TOTAL_PATROL_UNITS),
        "total_patrol_units_allocated": int(dispatch_sheet["recommended_patrol_count"].sum()),
        "tow_required_hotspots": int(dispatch_sheet["tow_required"].sum()),
        "critical_hotspots": int((dispatch_sheet["final_risk_band"] == "Critical").sum()),
        "high_hotspots": int((dispatch_sheet["final_risk_band"] == "High").sum()),
        "zone_count": int(len(zone_summary)),
        "route_stops": int(len(route_plan)),
        "mean_priority_score": float(pd.to_numeric(dispatch_sheet["priority_score"], errors="coerce").mean()) if len(dispatch_sheet) else 0.0,
        "mean_escalation_probability": float(pd.to_numeric(dispatch_sheet["probability_of_escalation"], errors="coerce").mean()) if len(dispatch_sheet) else 0.0,
    }])
    summary_out.to_csv(PHASE8_OUT_DIR / "phase8_summary.csv", index=False)

    # Console output
    print("\nPhase 8 complete")
    print("Outputs saved to:", PHASE8_OUT_DIR.resolve())
    print("\nSummary:")
    print(summary_out.to_string(index=False))

    print("\nTop 10 dispatch priorities:")
    top_cols = [c for c in ["dispatch_rank", "cluster_key", "cluster_label_display", "zone_name", "final_risk_band", "priority_score", "probability_of_escalation", "predicted_next_week_ccs_score", "recommended_patrol_count", "tow_required", "recommended_time_window", "enforcement_action"] if c in dispatch_sheet.columns]
    print(dispatch_sheet[top_cols].head(10).to_string(index=False))

    print("\nTop 10 zones:")
    zone_cols = [c for c in ["zone_rank", "zone_name", "zone_priority_total", "zone_priority_mean", "zone_priority_max", "zone_hotspots", "zone_patrol_units", "zone_tow_count", "zone_critical_count", "zone_high_count"] if c in zone_summary.columns]
    print(zone_summary[zone_cols].head(10).to_string(index=False))

    print("\nRoute plan:")
    if len(route_plan):
        print(route_plan.head(20).to_string(index=False))
    else:
        print("No route plan generated.")


if __name__ == "__main__":
    main()
