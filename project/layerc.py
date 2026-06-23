import math
from pathlib import Path

import numpy as np
import pandas as pd

# =========================
# Layer C - Novel Feature Engineering
# ROP, TVS, VDI, Validation Uncertainty, Resurgence Score
# =========================

EPS = 1e-9
OUT_DIR = Path("content/layer_c_outputs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHASE5_CLUSTER_PATHS = [
    Path("content/phase5_outputs_2/phase5_cluster_capacity_loss.csv"),
    Path("content/phase5_outputs_2/phase5_priority_table_full.csv"),
    Path("content/phase5_outputs_2/phase5_stage5_handoff.csv"),
    Path("phase5_outputs_2/phase5_cluster_capacity_loss.csv"),
    Path("phase5_outputs_2/phase5_priority_table_full.csv"),
    Path("phase5_outputs_2/phase5_stage5_handoff.csv"),
    Path("/content/phase5_outputs_2/phase5_cluster_capacity_loss.csv"),
    Path("/content/phase5_outputs_2/phase5_priority_table_full.csv"),
    Path("/content/phase5_outputs_2/phase5_stage5_handoff.csv"),
]

PHASE5_RECORD_PATHS = [
    Path("content/phase5_outputs_2/phase5_enriched_records.csv"),
    Path("phase5_outputs_2/phase5_enriched_records.csv"),
    Path("/content/phase5_outputs_2/phase5_enriched_records.csv"),
]

VALIDATION_SOURCE_PATHS = [
    Path("content/phase4_outputs_2/phase4_merged_with_prior_scores.csv"),
    Path("phase4_outputs_2/phase4_merged_with_prior_scores.csv"),
    Path("/content/phase4_outputs_2/phase4_merged_with_prior_scores.csv"),
    Path("content/phase3_outputs_2/phase3_clustered_dataset.csv"),
    Path("phase3_outputs_2/phase3_clustered_dataset.csv"),
    Path("/content/phase3_outputs_2/phase3_clustered_dataset.csv"),
    Path("content/phase5_outputs_2/phase5_enriched_records.csv"),
    Path("phase5_outputs_2/phase5_enriched_records.csv"),
    Path("/content/phase5_outputs_2/phase5_enriched_records.csv"),
]

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


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_tag(tag):
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
        parsed = eval(s, {"__builtins__": {}})  # safe-ish for list-like literals only
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    except Exception:
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return [p for p in parts if p]


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def standardize_cluster_col(df):
    for c in ["st_dbscan_cluster_id", "cluster_id", "dbscan_cluster_id"]:
        if c in df.columns:
            return c
    raise ValueError("No cluster id column found.")


def standardize_vehicle_col(df):
    for c in ["canonical_vehicle_number", "vehicle_number", "updated_vehicle_number"]:
        if c in df.columns:
            return c
    return None


def standardize_vehicle_type_col(df):
    for c in ["canonical_vehicle_type", "vehicle_type", "updated_vehicle_type"]:
        if c in df.columns:
            return c
    return None


def ensure_label_column(df):
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


def derive_coords(df):
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


def parse_datetime_ist(df):
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
    for sev in sorted(SEVERITY_RULES.keys(), reverse=True):
        vocab = SEVERITY_RULES[sev]
        if any(any(v == tag or v in tag for v in vocab) for tag in normalized):
            return sev
    return 1


def dominant_label(series, default=""):
    s = pd.Series(series).dropna().astype(str).str.strip()
    s = s[s.ne("")]
    if s.empty:
        return default
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


def load_first_existing(paths):
    for p in paths:
        if p.exists():
            return pd.read_csv(p, low_memory=False), p
    return None, None


def minmax(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty or valid.nunique() <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    mn = valid.min()
    mx = valid.max()
    return (s.fillna(mn) - mn) / (mx - mn + EPS)


def smooth_norm(s, floor=0.10):
    return floor + (1.0 - floor) * minmax(s)


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


def monotone_chain(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x = np.array([p[0] for p in poly], dtype=float)
    y = np.array([p[1] for p in poly], dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def latlon_to_local_xy(lat, lon, ref_lat):
    r = 6371000.0
    x = np.deg2rad(lon) * r * np.cos(np.deg2rad(ref_lat))
    y = np.deg2rad(lat) * r
    return x, y


def estimate_cluster_area_m2(g):
    pts = g[["lat", "lon"]].dropna().drop_duplicates()
    if len(pts) == 0:
        return np.nan
    if len(pts) == 1:
        return 1000.0

    ref_lat = float(pts["lat"].mean())
    xy = np.array([latlon_to_local_xy(lat, lon, ref_lat) for lat, lon in pts[["lat", "lon"]].to_numpy()])

    if len(xy) >= 3:
        hull = monotone_chain(xy)
        area = polygon_area(hull)
        if area > 0:
            return max(area, 1000.0)

    x_span = float(np.ptp(xy[:, 0]))
    y_span = float(np.ptp(xy[:, 1]))
    padded_area = max((x_span + 50.0) * (y_span + 50.0), 1000.0)
    return padded_area


def compute_tvs(weekly_counts):
    counts = np.asarray(weekly_counts, dtype=float)
    if len(counts) <= 1:
        return 0.0
    mean = float(np.mean(counts))
    std = float(np.std(counts, ddof=0))
    return std / (mean + EPS)


def compute_resurgence_score(weekly_counts):
    counts = np.asarray(weekly_counts, dtype=float)
    if len(counts) == 0:
        return np.nan
    if len(counts) == 1:
        return 1.0

    recent_n = min(2, len(counts))
    recent_avg = float(np.mean(counts[-recent_n:]))
    baseline = counts[:-recent_n]
    if len(baseline) == 0:
        baseline = counts[:-1]
    baseline_avg = float(np.mean(baseline)) if len(baseline) else float(np.mean(counts))

    return float(np.clip((recent_avg + 1.0) / (baseline_avg + 1.0), 0.0, 5.0))


def load_layer_c_sources():
    cluster_df, cluster_src = load_first_existing(PHASE5_CLUSTER_PATHS)
    records_df, records_src = load_first_existing(PHASE5_RECORD_PATHS)
    review_df, review_src = load_first_existing(VALIDATION_SOURCE_PATHS)
    return cluster_df, cluster_src, records_df, records_src, review_df, review_src


# =========================
# Load data
# =========================
cluster_df, cluster_src, records_df, records_src, review_df, review_src = load_layer_c_sources()

if cluster_df is None:
    raise FileNotFoundError("Could not find the Stage 5 cluster table.")

cluster_col = standardize_cluster_col(cluster_df)
cluster_df = ensure_label_column(derive_coords(cluster_df.copy()))
cluster_df = cluster_df.drop_duplicates(subset=[cluster_col]).reset_index(drop=True)

if records_df is None:
    records_df = review_df.copy() if review_df is not None else cluster_df.copy()

# Build the analysis record table
records_df = records_df.copy()
if cluster_col not in records_df.columns:
    rec_cluster_col = standardize_cluster_col(records_df)
    if rec_cluster_col != cluster_col:
        records_df = records_df.rename(columns={rec_cluster_col: cluster_col})

records_df = ensure_label_column(records_df)
records_df = derive_coords(records_df)

if "validation_status" in records_df.columns:
    records_df["validation_status_clean"] = records_df["validation_status"].fillna("").astype(str).str.lower()
    approved = records_df[records_df["validation_status_clean"].eq("approved")].copy()
    if len(approved) == 0:
        approved = records_df.copy()
else:
    approved = records_df.copy()

approved["created_datetime_ist"] = parse_datetime_ist(approved)
approved = approved.dropna(subset=["created_datetime_ist"]).copy()
approved["service_date"] = approved["created_datetime_ist"].dt.tz_localize(None).dt.date
approved["week_start"] = week_start_monday(approved["created_datetime_ist"])

if "violation_tags" not in approved.columns and "violation_type" in approved.columns:
    approved["violation_tags"] = approved["violation_type"].apply(parse_listlike)
elif "violation_tags" in approved.columns:
    approved["violation_tags"] = approved["violation_tags"].apply(parse_listlike)
else:
    approved["violation_tags"] = [[] for _ in range(len(approved))]

approved["severity_score"] = approved["violation_tags"].apply(severity_from_tags)

vehicle_col = standardize_vehicle_col(approved)
vehicle_type_col = standardize_vehicle_type_col(approved)

if "hotspot_unit" not in approved.columns:
    if {"junction_name", "police_station"}.issubset(approved.columns):
        def make_hotspot_unit(row):
            junction = clean_text(row.get("junction_name", ""))
            if junction and junction.upper() != "NO JUNCTION":
                return f"JUNCTION::{junction}"
            station = clean_text(row.get("police_station", "UNKNOWN")) or "UNKNOWN"
            return f"POLICE_STATION::{station}"
        approved["hotspot_unit"] = approved.apply(make_hotspot_unit, axis=1)
    else:
        approved["hotspot_unit"] = "UNKNOWN"

# Global observation window
global_min_date = pd.to_datetime(approved["service_date"], errors="coerce").min()
global_max_date = pd.to_datetime(approved["service_date"], errors="coerce").max()
global_observation_days = max(int((global_max_date - global_min_date).days) + 1 if pd.notna(global_min_date) and pd.notna(global_max_date) else 1, 1)

# =========================
# Cluster-level feature engineering
# =========================
cluster_groups = approved.groupby(cluster_col, dropna=False)

base_metrics = cluster_groups.agg(
    records_total=(cluster_col, "size"),
    distinct_days=("service_date", "nunique"),
    unique_hotspots=("hotspot_unit", "nunique"),
    centroid_lat=("lat", "mean"),
    centroid_lon=("lon", "mean"),
    severity_sum=("severity_score", "sum"),
    severity_mean=("severity_score", "mean"),
).reset_index()

if vehicle_col is not None and vehicle_col in approved.columns:
    vcount = approved.groupby(cluster_col)[vehicle_col].nunique().reset_index(name="unique_vehicles")
    base_metrics = base_metrics.merge(vcount, on=cluster_col, how="left")
else:
    base_metrics["unique_vehicles"] = np.nan

if vehicle_type_col is not None and vehicle_type_col in approved.columns:
    vtype = approved.groupby(cluster_col)[vehicle_type_col].agg(dominant_label).reset_index(name="dominant_vehicle_type")
    base_metrics = base_metrics.merge(vtype, on=cluster_col, how="left")
else:
    base_metrics["dominant_vehicle_type"] = "UNKNOWN"

if "violation_tags" in approved.columns:
    exploded = approved[[cluster_col, "violation_tags"]].explode("violation_tags").copy()
    exploded["violation_tag"] = exploded["violation_tags"].map(normalize_tag)
    exploded = exploded[exploded["violation_tag"].ne("")]
    dom_tag = exploded.groupby(cluster_col)["violation_tag"].agg(dominant_label).reset_index(name="dominant_violation_tag")
    base_metrics = base_metrics.merge(dom_tag, on=cluster_col, how="left")
else:
    base_metrics["dominant_violation_tag"] = ""

# Weekly counts for temporal features
weekly_counts = (
    approved.groupby([cluster_col, "week_start"])
    .size()
    .reset_index(name="weekly_count")
    .sort_values([cluster_col, "week_start"])
)

weekly_lists = weekly_counts.groupby(cluster_col)["weekly_count"].apply(list).reset_index(name="weekly_count_list")
base_metrics = base_metrics.merge(weekly_lists, on=cluster_col, how="left")

# ROP: recurring occupancy persistence
base_metrics["active_days"] = base_metrics["distinct_days"].fillna(0).astype(float)
base_metrics["observation_days"] = global_observation_days
base_metrics["ROP"] = base_metrics["active_days"] / (base_metrics["observation_days"] + EPS)

# TVS: temporal volatility score
base_metrics["TVS"] = base_metrics["weekly_count_list"].apply(lambda x: compute_tvs(x if isinstance(x, list) else []))

# Resurgence score
base_metrics["resurgence_score"] = base_metrics["weekly_count_list"].apply(
    lambda x: compute_resurgence_score(x if isinstance(x, list) else [])
)

# VDI: violation density index
areas = []
for cid, g in approved.groupby(cluster_col):
    area_m2 = estimate_cluster_area_m2(g)
    areas.append((cid, area_m2))
area_df = pd.DataFrame(areas, columns=[cluster_col, "cluster_area_m2"])
base_metrics = base_metrics.merge(area_df, on=cluster_col, how="left")
base_metrics["VDI"] = base_metrics["records_total"] / (base_metrics["cluster_area_m2"].replace(0, np.nan) + EPS)

# Validation uncertainty from broader review source if available
validation_source = review_df.copy() if review_df is not None else None
if validation_source is not None and len(validation_source):
    if cluster_col not in validation_source.columns:
        rv_cluster_col = standardize_cluster_col(validation_source)
        if rv_cluster_col != cluster_col:
            validation_source = validation_source.rename(columns={rv_cluster_col: cluster_col})

    if "validation_status" in validation_source.columns:
        validation_source = validation_source.copy()
        validation_source["validation_status_clean"] = validation_source["validation_status"].fillna("").astype(str).str.lower()
        validation_source = validation_source.dropna(subset=[cluster_col]).copy()

        validation_summary = (
            validation_source.groupby(cluster_col)
            .agg(
                review_total=(cluster_col, "size"),
                review_present=("validation_status_clean", lambda s: int((s != "").sum())),
                approved_count=("validation_status_clean", lambda s: int((s == "approved").sum())),
            )
            .reset_index()
        )

        validation_summary["reviewed_ratio"] = validation_summary["review_present"] / (validation_summary["review_total"] + EPS)
        validation_summary["approval_rate"] = validation_summary["approved_count"] / (validation_summary["review_present"].replace(0, np.nan) + EPS)
        validation_summary["validation_confidence"] = validation_summary["reviewed_ratio"] * validation_summary["approval_rate"].fillna(0.0)
        validation_summary["validation_uncertainty"] = 1.0 - validation_summary["validation_confidence"]
        validation_summary["validation_uncertainty"] = validation_summary["validation_uncertainty"].clip(lower=0.0, upper=1.0)

        base_metrics = base_metrics.merge(
            validation_summary[[cluster_col, "review_total", "review_present", "approved_count", "reviewed_ratio", "approval_rate", "validation_confidence", "validation_uncertainty"]],
            on=cluster_col,
            how="left",
        )
    else:
        base_metrics["review_total"] = np.nan
        base_metrics["review_present"] = np.nan
        base_metrics["approved_count"] = np.nan
        base_metrics["reviewed_ratio"] = np.nan
        base_metrics["approval_rate"] = np.nan
        base_metrics["validation_confidence"] = np.nan
        base_metrics["validation_uncertainty"] = np.nan
else:
    base_metrics["review_total"] = np.nan
    base_metrics["review_present"] = np.nan
    base_metrics["approved_count"] = np.nan
    base_metrics["reviewed_ratio"] = np.nan
    base_metrics["approval_rate"] = np.nan
    base_metrics["validation_confidence"] = np.nan
    base_metrics["validation_uncertainty"] = np.nan

# Fill safe defaults
base_metrics["dominant_vehicle_type"] = base_metrics["dominant_vehicle_type"].fillna("UNKNOWN")
base_metrics["dominant_violation_tag"] = base_metrics["dominant_violation_tag"].fillna("")
base_metrics["validation_uncertainty"] = pd.to_numeric(base_metrics["validation_uncertainty"], errors="coerce")
base_metrics["validation_uncertainty"] = base_metrics["validation_uncertainty"].fillna(base_metrics["validation_uncertainty"].median(skipna=True) if base_metrics["validation_uncertainty"].notna().any() else 0.5)
base_metrics["validation_uncertainty"] = base_metrics["validation_uncertainty"].clip(lower=0.0, upper=1.0)

# Optional compact composite for downstream use
base_metrics["ROP_norm"] = minmax(base_metrics["ROP"])
base_metrics["TVS_norm"] = minmax(base_metrics["TVS"])
base_metrics["VDI_norm"] = minmax(base_metrics["VDI"])
base_metrics["validation_uncertainty_norm"] = minmax(base_metrics["validation_uncertainty"])
base_metrics["resurgence_norm"] = minmax(base_metrics["resurgence_score"])

base_metrics["layer_c_index"] = (
    0.25 * base_metrics["ROP_norm"]
    + 0.20 * base_metrics["TVS_norm"]
    + 0.20 * base_metrics["VDI_norm"]
    + 0.20 * base_metrics["validation_uncertainty_norm"]
    + 0.15 * base_metrics["resurgence_norm"]
)

# Merge back onto Stage 5 cluster table
layer_c_df = cluster_df.merge(base_metrics, on=cluster_col, how="left", suffixes=("", "_layerc"))

# Preserve / clean useful fields
layer_c_df["cluster_label"] = layer_c_df["cluster_label"].fillna("").astype(str).str.strip()
layer_c_df["cluster_label_display"] = (
    layer_c_df["cluster_label"].astype(str) + " (Cluster " + layer_c_df[cluster_col].astype(str) + ")"
)

# Handle duplicate suffix columns if any
for col in list(layer_c_df.columns):
    if col.endswith("_layerc"):
        layer_c_df.drop(columns=[col], inplace=True, errors="ignore")

# Keep one row per cluster
layer_c_df = layer_c_df.drop_duplicates(subset=[cluster_col]).reset_index(drop=True)

# Sorting for presentation
layer_c_df["layer_c_index"] = pd.to_numeric(layer_c_df["layer_c_index"], errors="coerce").fillna(0.0)
layer_c_df = layer_c_df.sort_values(
    ["layer_c_index", "resurgence_score", "VDI", "ROP"],
    ascending=[False, False, False, False],
).reset_index(drop=True)

layer_c_df["layer_c_rank"] = np.arange(1, len(layer_c_df) + 1)

# =========================
# Outputs
# =========================
layer_c_export_cols = [
    "layer_c_rank",
    cluster_col,
    "cluster_label",
    "cluster_label_display",
    "records_total",
    "distinct_days",
    "active_days",
    "observation_days",
    "ROP",
    "TVS",
    "VDI",
    "cluster_area_m2",
    "validation_uncertainty",
    "validation_confidence",
    "review_total",
    "review_present",
    "approved_count",
    "resurgence_score",
    "dominant_vehicle_type",
    "dominant_violation_tag",
    "layer_c_index",
    "lat",
    "lon",
]

layer_c_export = layer_c_df[[c for c in layer_c_export_cols if c in layer_c_df.columns]].copy()

layer_c_export.to_csv(OUT_DIR / "layer_c_novel_features.csv", index=False)
layer_c_df.to_csv(OUT_DIR / "phase5_with_layer_c.csv", index=False)

summary = pd.DataFrame([{
    "cluster_source": str(cluster_src),
    "records_source": str(records_src) if records_df is not None else "",
    "validation_source": str(review_src) if review_df is not None else "",
    "clusters_scored": int(len(layer_c_df)),
    "mean_ROP": float(pd.to_numeric(layer_c_df["ROP"], errors="coerce").mean()),
    "mean_TVS": float(pd.to_numeric(layer_c_df["TVS"], errors="coerce").mean()),
    "mean_VDI": float(pd.to_numeric(layer_c_df["VDI"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
    "mean_validation_uncertainty": float(pd.to_numeric(layer_c_df["validation_uncertainty"], errors="coerce").mean()),
    "mean_resurgence_score": float(pd.to_numeric(layer_c_df["resurgence_score"], errors="coerce").mean()),
}])
summary.to_csv(OUT_DIR / "layer_c_summary.csv", index=False)

weekly_counts.to_csv(OUT_DIR / "layer_c_weekly_counts.csv", index=False)

print("Layer C complete")
print("Cluster source:", cluster_src)
print("Record source:", records_src)
print("Validation source:", review_src)
print("Clusters scored:", len(layer_c_df))
print("Outputs saved to:", OUT_DIR.resolve())
print("\nSummary:")
print(summary.to_string(index=False))

print("\nTop Layer C hotspots:")
show_cols = [
    "layer_c_rank", cluster_col, "cluster_label_display",
    "ROP", "TVS", "VDI", "validation_uncertainty", "resurgence_score", "layer_c_index"
]
print(layer_c_export[[c for c in show_cols if c in layer_c_export.columns]].head(15).to_string(index=False))