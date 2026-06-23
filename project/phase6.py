import ast
import warnings
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# Stage 6 - corrected hotspot scoring and output generation
# ============================================================
# Major fixes:
#   1) CCS no longer collapses when one factor is constant.
#   2) delay_minutes_per_vehicle is repaired with a unit-consistent
#      congestion model when the upstream value is tiny / flat.
#   3) criticality_factor is rebuilt from available graph/load proxies.
#   4) growth calculation uses smoothing to avoid repeated 3.0 / 4.0 clips.
#   5) duplicate physical hotspots are deduplicated by label + rounded coords.
#   6) output tables are preserved: full cluster table + distinct hotspot ranking.
# ============================================================

PHASE5_DIRS = [
    Path("phase5_outputs_2"),
    Path("/content/phase5_outputs_2"),
    Path("content/phase5_outputs_2"),
]

PHASE4_DIRS = [
    Path("phase4_outputs_2"),
    Path("/content/phase4_outputs_2"),
    Path("content/phase4_outputs_2"),
]

PHASE3_DIRS = [
    Path("phase3_outputs_2"),
    Path("/content/phase3_outputs_2"),
    Path("content/phase3_outputs_2"),
]

OUT_DIR = Path("content/phase6_outputs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9

# Use a modest smoothing term so sparse windows do not explode or collapse.
GROWTH_SMOOTHING = 2.0


# =========================
# Generic helpers
# =========================
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


def load_first_existing(base_dirs: Iterable[Path], filenames: Iterable[str]):
    for d in base_dirs:
        for name in filenames:
            p = d / name
            if p.exists():
                return pd.read_csv(p, low_memory=False), p
    return None, None


def standardize_cluster_col(df: pd.DataFrame) -> str:
    for c in ["st_dbscan_cluster_id", "cluster_id", "dbscan_cluster_id"]:
        if c in df.columns:
            return c
    raise ValueError("No cluster id column found.")


def standardize_vehicle_col(df: pd.DataFrame):
    for c in ["canonical_vehicle_number", "vehicle_number", "updated_vehicle_number"]:
        if c in df.columns:
            return c
    return None


def standardize_vehicle_type_col(df: pd.DataFrame):
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
        raise ValueError("Missing created_datetime / created_datetime_ist column.")

    ts = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    return ts.dt.tz_convert("Asia/Kolkata")


def week_start_monday(series):
    dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    week_start = dt.dt.normalize() - pd.to_timedelta(dt.dt.weekday, unit="D")
    return week_start.dt.tz_localize(None)


def safe_metric(val, digits=2):
    try:
        if pd.isna(val):
            return "0"
        val = float(val)
        if abs(val) >= 1000:
            return f"{val:,.0f}"
        return f"{val:,.{digits}f}"
    except Exception:
        return str(val)


def dominant_label(series, exclude=None, default=""):
    s = pd.Series(series).dropna().astype(str).str.strip()
    s = s[s.ne("")]
    if exclude:
        excl = {clean_text(x).lower() for x in exclude}
        s = s[~s.str.lower().isin(excl)]
    if s.empty:
        return default
    m = s.mode()
    if not m.empty:
        return m.iloc[0]
    return s.iloc[0]


# =========================
# Domain helpers
# =========================
def vehicle_width_m(vehicle_type):
    widths = {
        "SCOOTER": 0.80,
        "MOTOR CYCLE": 0.90,
        "MOTORCYCLE": 0.90,
        "BICYCLE": 0.60,
        "CYCLE": 0.60,
        "PASSENGER AUTO": 1.60,
        "AUTO": 1.60,
        "CAR": 1.90,
        "SUV": 2.00,
        "JEEP": 2.00,
        "VAN": 2.20,
        "TEMPO": 2.20,
        "BUS": 2.60,
        "TRUCK": 2.60,
        "LORRY": 2.60,
        "TANKER": 2.80,
        "TRACTOR": 2.20,
        "MINI TRUCK": 2.20,
        "AMBULANCE": 2.00,
        "UNKNOWN": 1.90,
    }
    return widths.get(clean_text(vehicle_type).upper(), 1.90)


def road_speed_kmh(road_class):
    speeds = {
        "motorway": 60.0,
        "trunk": 55.0,
        "primary": 45.0,
        "secondary": 40.0,
        "tertiary": 35.0,
        "unclassified": 30.0,
        "residential": 30.0,
        "living_street": 20.0,
        "service": 25.0,
        "road": 30.0,
    }
    return speeds.get(clean_text(road_class).lower(), 30.0)


def base_capacity_per_lane(road_class):
    cap = {
        "motorway": 2200.0,
        "trunk": 2100.0,
        "primary": 1900.0,
        "secondary": 1800.0,
        "tertiary": 1700.0,
        "unclassified": 1650.0,
        "residential": 1500.0,
        "living_street": 1200.0,
        "service": 1400.0,
        "road": 1600.0,
    }
    return cap.get(clean_text(road_class).lower(), 1600.0)


# =========================
# Normalization / scoring
# =========================
def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    mn = valid.min()
    mx = valid.max()
    return (s.fillna(mn) - mn) / (mx - mn + EPS)


def active_minmax(s: pd.Series) -> Tuple[pd.Series, bool]:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.dropna().nunique(dropna=True) <= 1:
        return pd.Series(np.nan, index=s.index, dtype=float), False
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    return (s.fillna(mn) - mn) / (mx - mn + EPS), True


def weighted_component_score(df: pd.DataFrame, components):
    """
    PATCH: restore the concept-note multiplicative CCS formula.

    Original design: CCS ∝ Delay × λ × Severity × Growth × Criticality
    Each factor is min-max normalised to [0,1] and then the *product* is
    taken, NOT a weighted average.  To prevent one zero factor from killing
    the whole score, constant / all-null components are replaced by their
    column median (floored at 0.1) instead of being dropped.

    The weighted-average variant was a safe fallback but loses the
    multiplicative interaction structure that distinguishes high-delay +
    high-growth hotspots from ones that are merely high on one axis.

    components: list of (source_col, weight, output_name)
      weight is used as the *exponent* in the product  — component^weight —
      so higher-weight factors have more influence while the score stays
      bounded in [0, 1].
    """
    out = df.copy()
    active = []
    for source_col, weight, out_col in components:
        if source_col not in out.columns:
            out[out_col] = np.nan
            continue
        norm, is_active = active_minmax(out[source_col])
        if not is_active:
            # Constant column: use a neutral value (0.5) so it does not kill
            # the product but also does not inflate it.
            norm = pd.Series(0.5, index=out.index, dtype=float)
        out[out_col] = norm.fillna(0.5)
        active.append((out_col, weight))

    if not active:
        out["ccs_score"] = 0.0
        out["ccs_raw_product"] = 0.0
        return out

    # Multiplicative combination: ∏ component_i ^ weight_i
    # Then rescale by the total weight so the result sits in [0, 1].
    total_weight = float(sum(w for _, w in active))
    log_sum = np.zeros(len(out), dtype=float)
    for out_col, weight in active:
        vals = out[out_col].to_numpy(dtype=float).clip(1e-6, 1.0)
        log_sum += (weight / max(total_weight, EPS)) * np.log(vals)

    raw = np.exp(log_sum)          # back to linear, in (0, 1]
    out["ccs_raw_product"] = raw
    out["ccs_score"] = 100.0 * np.clip(raw, 0.0, 1.0)
    return out


def risk_band_from_score(df: pd.DataFrame):
    """
    Prefer fixed thresholds on a calibrated 0-100 score. If the distribution is
    too compressed, fall back to quantile buckets so the output remains usable.
    """
    if df is None or len(df) == 0 or "ccs_score" not in df.columns:
        return df

    df = df.copy()
    score = pd.to_numeric(df["ccs_score"], errors="coerce").fillna(0.0)
    spread = float(score.quantile(0.90) - score.quantile(0.10)) if len(score) else 0.0

    if spread < 5.0:
        q80 = score.quantile(0.80)
        q60 = score.quantile(0.60)
        q40 = score.quantile(0.40)

        def band_q(x):
            if x >= q80:
                return "Critical"
            if x >= q60:
                return "High"
            if x >= q40:
                return "Moderate"
            return "Watch"

        df["risk_band"] = score.apply(band_q)
        return df

    def band_fixed(x):
        if x >= 80:
            return "Critical"
        if x >= 60:
            return "High"
        if x >= 40:
            return "Moderate"
        return "Watch"

    df["risk_band"] = score.apply(band_fixed)
    return df


# =========================
# Record-level backfill
# =========================
def load_record_level_source():
    df, path = load_first_existing(
        PHASE5_DIRS + PHASE4_DIRS + PHASE3_DIRS,
        [
            "phase5_enriched_records.csv",
            "phase4_merged_with_prior_scores.csv",
            "phase3_clustered_dataset.csv",
        ],
    )
    return df, path


def fill_missing_from_records(cluster_df, records_df, cluster_col, vehicle_col=None, vehicle_type_col=None):
    """
    Backfill cluster-level metrics from the record-level source.
    The record-level source is also used to derive stable growth metrics.
    """
    if records_df is None or len(records_df) == 0:
        return cluster_df.copy(), pd.DataFrame(), records_df

    r = records_df.copy()
    r = r.dropna(subset=[cluster_col]).copy()
    r[cluster_col] = pd.to_numeric(r[cluster_col], errors="coerce")
    r = r[r[cluster_col].ne(-1)].copy()

    r["created_datetime_ist"] = parse_datetime_ist(r)
    r = r.dropna(subset=["created_datetime_ist"]).copy()
    r["created_datetime_ist_naive"] = r["created_datetime_ist"].dt.tz_localize(None)
    r["service_date"] = r["created_datetime_ist_naive"].dt.date
    r["cluster_week_start"] = week_start_monday(r["created_datetime_ist"])
    r["hour_ist"] = r["created_datetime_ist"].dt.hour
    r["is_peak_window"] = r["hour_ist"].between(8, 12, inclusive="both").astype(int)

    has_vehicle = vehicle_col is not None and vehicle_col in r.columns
    has_vtype = vehicle_type_col is not None and vehicle_type_col in r.columns
    has_lat = "lat" in r.columns
    has_lon = "lon" in r.columns

    metrics = r.groupby(cluster_col).agg(
        records_total=(cluster_col, "size"),
        distinct_days=("service_date", "nunique"),
        peak_window_records=("is_peak_window", "sum"),
        severity_sum=("severity_score", "sum") if "severity_score" in r.columns else (cluster_col, "size"),
        severity_mean=("severity_score", "mean") if "severity_score" in r.columns else (cluster_col, "size"),
        unique_vehicles=(vehicle_col, "nunique") if has_vehicle else (cluster_col, "size"),
        unique_vehicle_types=(vehicle_type_col, "nunique") if has_vtype else (cluster_col, "size"),
        centroid_lat=("lat", "mean") if has_lat else (cluster_col, "size"),
        centroid_lon=("lon", "mean") if has_lon else (cluster_col, "size"),
        dominant_vehicle_type=(vehicle_type_col, dominant_label) if has_vtype else (cluster_col, lambda s: "UNKNOWN"),
        dominant_police_station=("police_station", dominant_label) if "police_station" in r.columns else (cluster_col, lambda s: ""),
        dominant_junction_name=("junction_name", dominant_label) if "junction_name" in r.columns else (cluster_col, lambda s: ""),
    ).reset_index()

    metrics["distinct_days"] = pd.to_numeric(metrics["distinct_days"], errors="coerce").fillna(1).clip(lower=1)
    metrics["lambda_hr_peak_window"] = metrics["peak_window_records"] / (metrics["distinct_days"] * 5.0)

    weekly = (
        r.groupby([cluster_col, "cluster_week_start"])
        .size()
        .reset_index(name="weekly_count")
        .sort_values([cluster_col, "cluster_week_start"])
    )

    growth_rows = []
    for cid, g in weekly.groupby(cluster_col):
        counts = g["weekly_count"].to_numpy(dtype=float)
        if len(counts) < 2:
            first_half = float(counts.mean()) if len(counts) else 0.0
            second_half = first_half
        else:
            mid = max(1, len(counts) // 2)
            first_half = float(counts[:mid].mean()) if len(counts[:mid]) else 0.0
            second_half = float(counts[mid:].mean()) if len(counts[mid:]) else 0.0

        smoothed_first = first_half + GROWTH_SMOOTHING
        smoothed_second = second_half + GROWTH_SMOOTHING
        growth_multiplier = smoothed_second / max(smoothed_first, EPS)
        growth_pct = growth_multiplier - 1.0

        # Keep the scale sane, but do not clip so hard that many clusters become identical.
        growth_pct = float(np.clip(growth_pct, -0.8, 1.5))
        growth_multiplier = float(np.clip(growth_multiplier, 0.5, 2.5))

        growth_rows.append(
            {
                cluster_col: cid,
                "growth_first_half_mean": first_half,
                "growth_second_half_mean": second_half,
                "growth_pct_change": growth_pct,
                "growth_multiplier": growth_multiplier,
            }
        )

    growth_df = pd.DataFrame(growth_rows)
    metrics = metrics.merge(growth_df, on=cluster_col, how="left")

    out = cluster_df.copy()
    if cluster_col not in out.columns:
        raise ValueError(f"Expected cluster column {cluster_col} in cluster_df.")

    out = out.merge(metrics, on=cluster_col, how="left", suffixes=("", "_rec"))

    for col in metrics.columns:
        if col == cluster_col:
            continue
        rec_col = f"{col}_rec"
        if rec_col in out.columns:
            if col in out.columns:
                if pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_numeric_dtype(out[rec_col]):
                    out[col] = pd.to_numeric(out[col], errors="coerce").combine_first(
                        pd.to_numeric(out[rec_col], errors="coerce")
                    )
                else:
                    out[col] = out[col].combine_first(out[rec_col])
            else:
                out[col] = out[rec_col]
            out.drop(columns=[rec_col], inplace=True)

    return out, weekly, r


# =========================
# Load Stage 5 table
# =========================
cluster_df, cluster_src = load_first_existing(
    PHASE5_DIRS,
    ["phase5_cluster_capacity_loss.csv", "phase5_priority_table_full.csv", "phase5_stage5_handoff.csv"],
)

if cluster_df is None:
    raise FileNotFoundError(
        "Could not find Stage 5 output. Expected phase5_cluster_capacity_loss.csv or phase5_priority_table_full.csv"
    )

cluster_col = standardize_cluster_col(cluster_df)
vehicle_col = standardize_vehicle_col(cluster_df)
vehicle_type_col = standardize_vehicle_type_col(cluster_df)

cluster_df = cluster_df.copy()
cluster_df = ensure_label_column(cluster_df)
cluster_df = derive_coords(cluster_df)

# Key columns that may be absent in some upstream outputs.
for col, default in [
    ("records_total", np.nan),
    ("distinct_days", np.nan),
    ("peak_window_records", np.nan),
    ("lambda_hr_peak_window", np.nan),
    ("lambda_hr_peak_hour", np.nan),
    ("mean_dwell_minutes", np.nan),
    ("mu_departures_per_hour", np.nan),
    ("blocking_vehicles_L", np.nan),
    ("severity_sum", np.nan),
    ("severity_mean", np.nan),
    ("growth_pct_change", np.nan),
    ("growth_multiplier", np.nan),
    ("criticality_factor", np.nan),
    ("delay_minutes_per_vehicle", np.nan),
    ("capacity_loss_pct", np.nan),
    ("junction_degree", np.nan),
    ("betweenness_centrality", np.nan),
    ("lane_count", np.nan),
    ("carriageway_width_m", np.nan),
    ("link_length_m", np.nan),
    ("base_capacity_pcu_hr", np.nan),
    ("reduced_capacity_pcu_hr", np.nan),
    ("unique_vehicles", np.nan),
    ("unique_vehicle_types", np.nan),
    ("repeat_vehicle_count_2plus", 0),
    ("chronic_vehicle_count_5plus", 0),
]:
    if col not in cluster_df.columns:
        cluster_df[col] = default

for col in [
    "records_total",
    "distinct_days",
    "peak_window_records",
    "lambda_hr_peak_window",
    "lambda_hr_peak_hour",
    "mean_dwell_minutes",
    "mu_departures_per_hour",
    "blocking_vehicles_L",
    "severity_sum",
    "severity_mean",
    "growth_pct_change",
    "growth_multiplier",
    "criticality_factor",
    "delay_minutes_per_vehicle",
    "capacity_loss_pct",
    "junction_degree",
    "betweenness_centrality",
    "lane_count",
    "carriageway_width_m",
    "link_length_m",
    "base_capacity_pcu_hr",
    "reduced_capacity_pcu_hr",
    "unique_vehicles",
    "unique_vehicle_types",
]:
    cluster_df[col] = pd.to_numeric(cluster_df[col], errors="coerce")

cluster_df["effective_capacity_pcu_hr"] = cluster_df["reduced_capacity_pcu_hr"]

# Safe label repair.
cluster_df["cluster_label"] = cluster_df["cluster_label"].fillna("").astype(str).str.strip()
cluster_df.loc[cluster_df["cluster_label"].eq(""), "cluster_label"] = np.nan

if "dominant_junction_name" in cluster_df.columns:
    dominant_junction_series = cluster_df["dominant_junction_name"].fillna("").astype(str)
else:
    dominant_junction_series = pd.Series("", index=cluster_df.index)

if "hotspot_unit" in cluster_df.columns:
    hotspot_unit_series = cluster_df["hotspot_unit"].fillna("").astype(str)
else:
    hotspot_unit_series = pd.Series("", index=cluster_df.index)

cluster_df.loc[cluster_df["cluster_label"].isna(), "cluster_label"] = dominant_junction_series
cluster_df.loc[cluster_df["cluster_label"].astype(str).str.strip().eq(""), "cluster_label"] = hotspot_unit_series
cluster_df.loc[cluster_df["cluster_label"].astype(str).str.strip().eq(""), "cluster_label"] = "CLUSTER::" + cluster_df[cluster_col].astype(str)

# =========================
# Backfill from record-level source
# =========================
records_df, records_src = load_record_level_source()
weekly_df = pd.DataFrame()

if records_df is not None and len(records_df):
    if cluster_col not in records_df.columns:
        rec_cluster_col = standardize_cluster_col(records_df)
        if rec_cluster_col != cluster_col:
            records_df = records_df.rename(columns={rec_cluster_col: cluster_col})
    if vehicle_col is None:
        vehicle_col = standardize_vehicle_col(records_df)
    if vehicle_type_col is None:
        vehicle_type_col = standardize_vehicle_type_col(records_df)

    records_df = records_df.copy()
    records_df = ensure_label_column(records_df)
    records_df = derive_coords(records_df)
    records_df["created_datetime_ist"] = parse_datetime_ist(records_df)

    cluster_df, weekly_df, records_df = fill_missing_from_records(
        cluster_df, records_df, cluster_col, vehicle_col, vehicle_type_col
    )

# Re-coerce after backfill merge.
for col in [
    "records_total",
    "distinct_days",
    "peak_window_records",
    "lambda_hr_peak_window",
    "lambda_hr_peak_hour",
    "mean_dwell_minutes",
    "mu_departures_per_hour",
    "blocking_vehicles_L",
    "severity_sum",
    "severity_mean",
    "growth_pct_change",
    "growth_multiplier",
    "criticality_factor",
    "delay_minutes_per_vehicle",
    "capacity_loss_pct",
    "junction_degree",
    "betweenness_centrality",
    "lane_count",
    "carriageway_width_m",
    "link_length_m",
    "unique_vehicles",
    "unique_vehicle_types",
    "base_capacity_pcu_hr",
    "reduced_capacity_pcu_hr",
]:
    if col in cluster_df.columns:
        cluster_df[col] = pd.to_numeric(cluster_df[col], errors="coerce")

# =========================
# Repair missing / broken row-wise metrics
# =========================
# Severity backfill.
mask = cluster_df["severity_sum"].isna() & cluster_df["severity_mean"].notna()
cluster_df.loc[mask, "severity_sum"] = cluster_df.loc[mask, "severity_mean"] * cluster_df.loc[mask, "records_total"].clip(lower=1)

mask = cluster_df["severity_mean"].isna() & cluster_df["severity_sum"].notna()
cluster_df.loc[mask, "severity_mean"] = cluster_df.loc[mask, "severity_sum"] / cluster_df.loc[mask, "records_total"].clip(lower=1)

# Peak-window arrival rate.
mask = cluster_df["lambda_hr_peak_window"].isna()
cluster_df.loc[mask, "lambda_hr_peak_window"] = (
    cluster_df.loc[mask, "peak_window_records"] / (cluster_df.loc[mask, "distinct_days"].clip(lower=1) * 5.0)
)

# Service rate from dwell time.
mask = cluster_df["mu_departures_per_hour"].isna()
cluster_df.loc[mask, "mu_departures_per_hour"] = 60.0 / (cluster_df.loc[mask, "mean_dwell_minutes"] + EPS)

# Blocking load.
mask = cluster_df["blocking_vehicles_L"].isna()
cluster_df.loc[mask, "blocking_vehicles_L"] = (
    cluster_df.loc[mask, "lambda_hr_peak_window"] / (cluster_df.loc[mask, "mu_departures_per_hour"] + EPS)
)

# Lane count / carriageway width repair.
if "lane_count" in cluster_df.columns:
    mask = cluster_df["lane_count"].isna() & cluster_df["carriageway_width_m"].notna()
    cluster_df.loc[mask, "lane_count"] = np.maximum(
        1,
        np.rint(cluster_df.loc[mask, "carriageway_width_m"] / 3.5),
    )

if "carriageway_width_m" in cluster_df.columns:
    mask = cluster_df["carriageway_width_m"].isna() & cluster_df["lane_count"].notna()
    cluster_df.loc[mask, "carriageway_width_m"] = cluster_df.loc[mask, "lane_count"].clip(lower=1) * 3.5

# Base / reduced capacity repair.
if "base_capacity_pcu_hr" in cluster_df.columns:
    mask = cluster_df["base_capacity_pcu_hr"].isna()
    cluster_df.loc[mask, "base_capacity_pcu_hr"] = (
        cluster_df.loc[mask, "road_class"].apply(base_capacity_per_lane).to_numpy()
        * cluster_df.loc[mask, "lane_count"].fillna(1).clip(lower=1).to_numpy()
    )

if "capacity_loss_pct" in cluster_df.columns:
    # If capacity loss is missing or completely collapsed, derive a usable proxy.
    cap_loss = pd.to_numeric(cluster_df["capacity_loss_pct"], errors="coerce")
    if cap_loss.notna().sum() == 0 or cap_loss.nunique(dropna=True) <= 1:
        util = (
            cluster_df["lambda_hr_peak_window"] / (cluster_df["mu_departures_per_hour"] + EPS)
        ).replace([np.inf, -np.inf], np.nan)
        util = util.fillna(util.median(skipna=True) if util.notna().any() else 0.0)
        util = util.clip(lower=0.0, upper=5.0)

        sev_norm = minmax(cluster_df["severity_sum"].fillna(0.0))
        blocking_norm = minmax(cluster_df["blocking_vehicles_L"].fillna(0.0))
        records_norm = minmax(cluster_df["records_total"].fillna(0.0))

        # Proxy: stronger when the cluster is loaded, blocked, and severe.
        capacity_loss_proxy = 100.0 * (
            0.40 * np.clip(util / 1.5, 0.0, 1.0)
            + 0.25 * blocking_norm
            + 0.20 * sev_norm
            + 0.15 * records_norm
        )
        cluster_df["capacity_loss_pct"] = capacity_loss_proxy.clip(lower=0.0, upper=95.0)
    else:
        cluster_df["capacity_loss_pct"] = cap_loss

# Reduced capacity.
if "reduced_capacity_pcu_hr" in cluster_df.columns:
    mask = cluster_df["reduced_capacity_pcu_hr"].isna() & cluster_df["base_capacity_pcu_hr"].notna()
    cluster_df.loc[mask, "reduced_capacity_pcu_hr"] = cluster_df.loc[mask, "base_capacity_pcu_hr"] * (
        1.0 - cluster_df.loc[mask, "capacity_loss_pct"].fillna(0.0) / 100.0
    )

cluster_df["effective_capacity_pcu_hr"] = cluster_df["reduced_capacity_pcu_hr"]

# Delay repair: if the upstream delay is tiny / flat / missing, rebuild it with
# a unit-consistent congestion proxy.
def rebuild_delay_minutes(df: pd.DataFrame) -> pd.Series:
    mean_dwell = pd.to_numeric(df.get("mean_dwell_minutes"), errors="coerce")
    mean_dwell = mean_dwell.fillna(mean_dwell.median(skipna=True) if mean_dwell.notna().any() else 2.0)
    mean_dwell = mean_dwell.clip(lower=0.5, upper=180.0)

    lam = pd.to_numeric(df.get("lambda_hr_peak_window"), errors="coerce").fillna(0.0).clip(lower=0.0)
    mu = pd.to_numeric(df.get("mu_departures_per_hour"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    mu = mu.fillna(mu.median(skipna=True) if mu.notna().any() else 1.0).clip(lower=0.1)

    util = (lam / (mu + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    util = util.clip(lower=0.0, upper=5.0)
    overflow = np.maximum(0.0, util - 1.0)

    sev_norm = minmax(pd.to_numeric(df.get("severity_sum"), errors="coerce").fillna(0.0))
    block_norm = minmax(pd.to_numeric(df.get("blocking_vehicles_L"), errors="coerce").fillna(0.0))

    # This is an extra delay proxy, not total travel time:
    #   - base delay grows with utilization,
    #   - overflow adds a nonlinear penalty,
    #   - severity / blocking increase the delay.
    delay = (
        mean_dwell
        * (1.0 + 0.4 * sev_norm + 0.3 * block_norm)
        * (1.0 + overflow)
    )

    return delay.clip(lower=0.5, upper=60.0)

existing_delay = pd.to_numeric(cluster_df["delay_minutes_per_vehicle"], errors="coerce")
if (
    existing_delay.notna().sum() == 0
    or existing_delay.nunique(dropna=True) <= 2
    or float(existing_delay.median(skipna=True) or 0.0) < 0.05
):
    cluster_df["delay_minutes_per_vehicle"] = rebuild_delay_minutes(cluster_df)
else:
    cluster_df["delay_minutes_per_vehicle"] = existing_delay

# Final safety-net fill for anything still missing.
for col in [
    "lambda_hr_peak_window",
    "lambda_hr_peak_hour",
    "mean_dwell_minutes",
    "mu_departures_per_hour",
    "blocking_vehicles_L",
    "delay_minutes_per_vehicle",
    "severity_sum",
    "severity_mean",
    "capacity_loss_pct",
    "base_capacity_pcu_hr",
    "reduced_capacity_pcu_hr",
]:
    if col in cluster_df.columns:
        cluster_df[col] = pd.to_numeric(cluster_df[col], errors="coerce").fillna(0.0)

# =========================
# Novel Feature B: criticality factor
# =========================
def build_criticality_factor(df: pd.DataFrame) -> pd.Series:
    # Primary source: actual graph proxies.
    deg = pd.to_numeric(df.get("junction_degree"), errors="coerce") if "junction_degree" in df.columns else pd.Series(np.nan, index=df.index)
    bet = pd.to_numeric(df.get("betweenness_centrality"), errors="coerce") if "betweenness_centrality" in df.columns else pd.Series(np.nan, index=df.index)

    deg_active = deg.notna().sum() > 0 and deg.nunique(dropna=True) > 1
    bet_active = bet.notna().sum() > 0 and bet.nunique(dropna=True) > 1

    if deg_active or bet_active:
        deg_norm = minmax(deg.fillna(deg.median(skipna=True) if deg.notna().any() else 0.0)) if deg_active else pd.Series(0.0, index=df.index)
        bet_norm = minmax(bet.fillna(bet.median(skipna=True) if bet.notna().any() else 0.0)) if bet_active else pd.Series(0.0, index=df.index)

        return 1.0 + 0.55 * deg_norm + 0.45 * bet_norm

    # Fallback: use load / severity / roadway proxy so the factor is still informative.
    records_norm = minmax(pd.to_numeric(df.get("records_total"), errors="coerce").fillna(0.0))
    unique_veh_norm = minmax(pd.to_numeric(df.get("unique_vehicles"), errors="coerce").fillna(0.0))
    block_norm = minmax(pd.to_numeric(df.get("blocking_vehicles_L"), errors="coerce").fillna(0.0))
    caploss_norm = minmax(pd.to_numeric(df.get("capacity_loss_pct"), errors="coerce").fillna(0.0))
    sev_norm = minmax(pd.to_numeric(df.get("severity_sum"), errors="coerce").fillna(0.0))

    road_class_factor = pd.Series(
        pd.to_numeric(df.get("road_class"), errors="coerce"), index=df.index
    ) if False else pd.Series(0.0, index=df.index)
    # The road class factor is intentionally omitted unless you map road class
    # categories to numeric order; the load/severity proxy is enough to prevent collapse.

    proxy = (
        0.30 * records_norm
        + 0.20 * unique_veh_norm
        + 0.20 * block_norm
        + 0.15 * caploss_norm
        + 0.15 * sev_norm
        + 0.00 * road_class_factor
    )
    return 1.0 + 0.5 * proxy.clip(0.0, 1.0)

cluster_df["criticality_factor"] = build_criticality_factor(cluster_df)

# If the source already had better non-null values than the proxy, preserve them.
existing_criticality = pd.to_numeric(cluster_df.get("criticality_factor"), errors="coerce")
if existing_criticality.notna().sum() > 0 and existing_criticality.nunique(dropna=True) > 1:
    # Keep the more informative of the two by replacing only missing rows.
    cluster_df["criticality_factor"] = existing_criticality.combine_first(cluster_df["criticality_factor"])

# Growth repair.
mask = cluster_df["growth_pct_change"].isna()
cluster_df.loc[mask, "growth_pct_change"] = 0.0
# PATCH: widen growth clip.  The original upper=3.0 cap compressed genuine
# strong-trend hotspots (e.g. 400% week-over-week growth) to the same value
# as 300%.  Upper=8.0 preserves differentiation while still guarding against
# data artifacts (e.g., a cluster with 1 record one week and 10 the next).
cluster_df["growth_pct_change"] = cluster_df["growth_pct_change"].clip(lower=-0.8, upper=8.0)

mask = cluster_df["growth_multiplier"].isna()
cluster_df.loc[mask, "growth_multiplier"] = 1.0 + cluster_df.loc[mask, "growth_pct_change"].clip(lower=0.0)
# Upper=9.0 matches the widened growth_pct_change ceiling (1 + 8.0).
cluster_df["growth_multiplier"] = cluster_df["growth_multiplier"].fillna(1.0).clip(lower=0.2, upper=9.0)

# =========================
# CCS computation
# =========================
# Weighted average on active components so one flat factor does not zero out the whole score.
cluster_df = weighted_component_score(
    cluster_df,
    components=[
        ("delay_minutes_per_vehicle", 0.25, "delay_norm"),
        ("lambda_hr_peak_window", 0.25, "lambda_norm"),
        ("severity_sum", 0.25, "severity_norm"),
        ("growth_multiplier", 0.10, "growth_norm"),
        ("criticality_factor", 0.15, "criticality_norm"),
    ],
)

# Also expose the underlying criticality proxy for inspection.
cluster_df["criticality_norm"] = pd.to_numeric(cluster_df["criticality_norm"], errors="coerce").fillna(0.0)
cluster_df["delay_norm"] = pd.to_numeric(cluster_df["delay_norm"], errors="coerce").fillna(0.0)
cluster_df["lambda_norm"] = pd.to_numeric(cluster_df["lambda_norm"], errors="coerce").fillna(0.0)
cluster_df["severity_norm"] = pd.to_numeric(cluster_df["severity_norm"], errors="coerce").fillna(0.0)
cluster_df["growth_norm"] = pd.to_numeric(cluster_df["growth_norm"], errors="coerce").fillna(0.0)

cluster_df = risk_band_from_score(cluster_df)

# =========================
# Dedupe physically identical hotspots
# =========================
def build_hotspot_key(df: pd.DataFrame) -> pd.Series:
    label = df.get("cluster_label", pd.Series("", index=df.index)).fillna("").astype(str).map(normalize_tag)
    if "lat" in df.columns and "lon" in df.columns:
        lat = pd.to_numeric(df["lat"], errors="coerce").round(5).astype(str)
        lon = pd.to_numeric(df["lon"], errors="coerce").round(5).astype(str)
    else:
        lat = pd.Series("", index=df.index)
        lon = pd.Series("", index=df.index)
    return label + "|" + lat + "|" + lon

cluster_df["physical_hotspot_key"] = build_hotspot_key(cluster_df)
cluster_df["cluster_label_display"] = (
    cluster_df["cluster_label"].astype(str) + " (Cluster " + cluster_df[cluster_col].astype(str) + ")"
)
cluster_df["priority_message"] = (
    cluster_df["cluster_label_display"] + " | "
    + cluster_df["risk_band"].astype(str)
    + " | CCS="
    + cluster_df["ccs_score"].round(3).astype(str)
)

# Sort by risk, then keep only one row per physical hotspot for the presentation tables.
cluster_df = cluster_df.sort_values(
    ["ccs_score", "delay_minutes_per_vehicle", "lambda_hr_peak_window", "records_total"],
    ascending=[False, False, False, False],
).reset_index(drop=True)

hotspot_df = cluster_df.drop_duplicates(subset=["physical_hotspot_key"], keep="first").copy().reset_index(drop=True)
cluster_df["ccs_rank"] = np.arange(1, len(cluster_df) + 1)
hotspot_df["ccs_rank"] = np.arange(1, len(hotspot_df) + 1)

# =========================
# Recommended actions
# =========================
hotspot_df["recommended_action"] = np.select(
    [
        hotspot_df["risk_band"].eq("Critical"),
        hotspot_df["risk_band"].eq("High"),
        hotspot_df["risk_band"].eq("Moderate"),
    ],
    [
        "Immediate patrol deployment",
        "Targeted enforcement + towing readiness",
        "Monitor and schedule peak-window checks",
    ],
    default="Routine monitoring",
)

# =========================
# Emerging hotspot alerts
# =========================
alerts = hotspot_df.copy()
if len(alerts):
    median_records = alerts["records_total"].median()
    min_record_threshold = max(10, int(round(median_records)) if pd.notna(median_records) else 10)

    # Require meaningful growth, but also let the distribution decide who is "emerging".
    positive_growth = alerts.loc[alerts["growth_pct_change"] > 0, "growth_pct_change"]
    growth_cut = float(positive_growth.quantile(0.70)) if len(positive_growth) else 0.25
    growth_cut = max(0.25, growth_cut)

    alerts["is_emerging"] = (
        (alerts["growth_pct_change"] >= growth_cut)
        & (alerts["records_total"] >= min_record_threshold)
    )
else:
    alerts["is_emerging"] = pd.Series(dtype=bool)

alerts = alerts[alerts["is_emerging"]].copy()

if len(alerts):
    alerts = alerts.sort_values(
        ["growth_pct_change", "growth_multiplier", "ccs_score"],
        ascending=[False, False, False],
    ).head(25).copy()
    q80 = alerts["growth_pct_change"].quantile(0.80)
    q60 = alerts["growth_pct_change"].quantile(0.60)

    def alert_level(x):
        if x >= q80:
            return "Emerging-Critical"
        if x >= q60:
            return "Emerging-High"
        return "Emerging-Watch"

    alerts["alert_level"] = alerts["growth_pct_change"].apply(alert_level)
else:
    alerts["alert_level"] = pd.Series(dtype=str)

# =========================
# Chronic offenders
# =========================
if records_df is not None and len(records_df):
    r = records_df.copy()
    rec_cluster_col = standardize_cluster_col(r)
    if cluster_col not in r.columns and rec_cluster_col:
        r = r.rename(columns={rec_cluster_col: cluster_col})
    if "created_datetime_ist" not in r.columns:
        r["created_datetime_ist"] = parse_datetime_ist(r)
    r["created_datetime_ist"] = pd.to_datetime(r["created_datetime_ist"], errors="coerce")
    r = r.dropna(subset=[cluster_col, "created_datetime_ist"]).copy()
    if vehicle_col is None:
        vehicle_col = standardize_vehicle_col(r)
    if vehicle_col is None:
        chronic_offenders = pd.DataFrame(
            columns=[
                "vehicle_number",
                "total_violations",
                "unique_clusters",
                "unique_hotspots",
                "first_seen",
                "last_seen",
                "dominant_vehicle_type",
                "chronic_offender_flag",
            ]
        )
    else:
        r[vehicle_col] = r[vehicle_col].fillna("").astype(str).str.strip()
        r = r[r[vehicle_col].ne("")].copy()
        if vehicle_type_col and vehicle_type_col in r.columns:
            dom_vtype = lambda s: dominant_label(s, default="UNKNOWN")
        else:
            dom_vtype = lambda s: "UNKNOWN"

        chronic_offenders = (
            r.groupby(vehicle_col)
            .agg(
                total_violations=(vehicle_col, "size"),
                unique_clusters=(cluster_col, "nunique"),
                unique_hotspots=("hotspot_unit", "nunique") if "hotspot_unit" in r.columns else (vehicle_col, "size"),
                first_seen=("created_datetime_ist", "min"),
                last_seen=("created_datetime_ist", "max"),
                dominant_vehicle_type=(vehicle_type_col, dom_vtype) if vehicle_type_col and vehicle_type_col in r.columns else (vehicle_col, lambda s: "UNKNOWN"),
            )
            .reset_index()
            .rename(columns={vehicle_col: "vehicle_number"})
            .sort_values(["total_violations", "unique_clusters"], ascending=[False, False])
        )
        chronic_offenders["chronic_offender_flag"] = (chronic_offenders["total_violations"] >= 5).astype(int)
        chronic_offenders = chronic_offenders[chronic_offenders["chronic_offender_flag"].eq(1)].copy()
else:
    chronic_offenders = pd.DataFrame(
        columns=[
            "vehicle_number",
            "total_violations",
            "unique_clusters",
            "unique_hotspots",
            "first_seen",
            "last_seen",
            "dominant_vehicle_type",
            "chronic_offender_flag",
        ]
    )

# =========================
# Final tables
# =========================
weekly_dispatch = hotspot_df.copy()

cols_hotspot = [
    "ccs_rank",
    cluster_col,
    "cluster_label",
    "cluster_label_display",
    "risk_band",
    "ccs_score",
    "ccs_raw_product",
    "delay_minutes_per_vehicle",
    "lambda_hr_peak_window",
    "lambda_hr_peak_hour",
    "severity_sum",
    "severity_mean",
    "growth_pct_change",
    "growth_multiplier",
    "criticality_factor",
    "junction_degree",
    "betweenness_centrality",
    "capacity_loss_pct",
    "base_capacity_pcu_hr",
    "effective_capacity_pcu_hr",
    "blocking_vehicles_L",
    "road_class",
    "lane_count",
    "carriageway_width_m",
    "link_length_m",
    "dominant_vehicle_type",
    "dominant_violation_tag",
    "records_total",
    "distinct_days",
    "unique_vehicles",
    "unique_vehicle_types",
    "geometry_source",
    "mappls_address",
    "physical_hotspot_key",
]

hotspot_ranking = hotspot_df[[c for c in cols_hotspot if c in hotspot_df.columns]].copy()

cols_reco = [
    "ccs_rank",
    cluster_col,
    "cluster_label_display",
    "risk_band",
    "recommended_action",
    "ccs_score",
    "delay_minutes_per_vehicle",
    "lambda_hr_peak_window",
    "severity_sum",
    "growth_pct_change",
    "criticality_factor",
    "road_class",
    "lane_count",
    "geometry_source",
    "physical_hotspot_key",
]

reco = hotspot_df[[c for c in cols_reco if c in hotspot_df.columns]].copy()

cols_handoff = [
    "ccs_rank",
    cluster_col,
    "cluster_label",
    "cluster_label_display",
    "risk_band",
    "ccs_score",
    "ccs_raw_product",
    "delay_minutes_per_vehicle",
    "lambda_hr_peak_window",
    "lambda_hr_peak_hour",
    "mean_dwell_minutes",
    "mu_departures_per_hour",
    "blocking_vehicles_L",
    "capacity_loss_pct",
    "base_capacity_pcu_hr",
    "effective_capacity_pcu_hr",
    "junction_degree",
    "betweenness_centrality",
    "growth_pct_change",
    "growth_multiplier",
    "criticality_factor",
    "road_class",
    "lane_count",
    "carriageway_width_m",
    "link_length_m",
    "dominant_vehicle_type",
    "dominant_violation_tag",
    "records_total",
    "distinct_days",
    "unique_vehicles",
    "unique_vehicle_types",
    "repeat_vehicle_count_2plus",
    "chronic_vehicle_count_5plus",
    "geometry_source",
    "mappls_address",
    "priority_message",
    "physical_hotspot_key",
]

handoff = hotspot_df[[c for c in cols_handoff if c in hotspot_df.columns]].copy()

# =========================
# Save outputs
# =========================
cluster_df.to_csv(OUT_DIR / "phase6_cluster_ccs_full.csv", index=False)
weekly_dispatch.to_csv(OUT_DIR / "phase6_weekly_dispatch_priority_table.csv", index=False)
hotspot_ranking.to_csv(OUT_DIR / "phase6_hotspot_ranking_ccs.csv", index=False)
alerts.to_csv(OUT_DIR / "phase6_emerging_hotspot_alerts.csv", index=False)
chronic_offenders.to_csv(OUT_DIR / "phase6_chronic_offender_list.csv", index=False)
reco.to_csv(OUT_DIR / "phase6_enforcement_recommendations.csv", index=False)
handoff.to_csv(OUT_DIR / "phase6_stage6_handoff.csv", index=False)

summary = pd.DataFrame(
    [
        {
            "clusters_scored": len(cluster_df),
            "distinct_hotspots": len(hotspot_df),
            "critical_hotspots": int((hotspot_df["risk_band"] == "Critical").sum()) if "risk_band" in hotspot_df.columns else 0,
            "high_hotspots": int((hotspot_df["risk_band"] == "High").sum()) if "risk_band" in hotspot_df.columns else 0,
            "emerging_alerts": len(alerts),
            "chronic_offenders": len(chronic_offenders),
            "ccs_mean": float(pd.to_numeric(hotspot_df["ccs_score"], errors="coerce").mean()) if len(hotspot_df) else 0.0,
            "ccs_std": float(pd.to_numeric(hotspot_df["ccs_score"], errors="coerce").std()) if len(hotspot_df) else 0.0,
        }
    ]
)
summary.to_csv(OUT_DIR / "phase6_summary.csv", index=False)

# =========================
# Console output
# =========================
print("Stage 6 complete")
print("Input source:", cluster_src)
print("Record source:", records_src)
print("Clusters scored:", len(cluster_df))
print("Distinct hotspots:", len(hotspot_df))
print("Top CCS cluster:", hotspot_df.iloc[0][cluster_col] if len(hotspot_df) else "N/A")
print("Top CCS score:", safe_metric(hotspot_df.iloc[0]["ccs_score"], 3) if len(hotspot_df) else "N/A")
print("Outputs saved to:", OUT_DIR.resolve())

print("\nCriticality factor check:")
print(hotspot_df["criticality_factor"].describe())

print("\nCCS check:")
print(hotspot_df["ccs_score"].describe())

print("\nTop 10 CCS hotspots:")
if len(hotspot_df):
    print(
        hotspot_df[[
            "ccs_rank",
            cluster_col,
            "cluster_label_display",
            "risk_band",
            "ccs_score",
            "delay_minutes_per_vehicle",
            "lambda_hr_peak_window",
            "severity_sum",
            "growth_pct_change",
            "criticality_factor",
        ]].head(10).to_string(index=False)
    )

print("\nTop 10 emerging alerts:")
if len(alerts):
    print(
        alerts[[
            "alert_level",
            cluster_col,
            "cluster_label_display",
            "growth_pct_change",
            "growth_multiplier",
            "ccs_score",
            "records_total",
        ]].head(10).to_string(index=False)
    )