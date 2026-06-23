# Stage 3 — ST-DBSCAN Spatial + Temporal Clustering
# Input : approved records with latitude, longitude, created_datetime, violation_type, vehicle_number
# Output: clustered records, cluster summary, noise points, weekly cluster counts

import ast
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
RAW_CSV = "jan to may police violation_anonymized791b166.csv"
PHASE2_APPROVED_PATH = Path("content/phase2_outputs_2/phase2_approved_severity_dataset.csv")

OUT_DIR = Path("content/phase3_outputs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS_SPATIAL_METERS = 150.0
EPS_TEMPORAL_DAYS = 3.0
MIN_PTS = 15
EARTH_RADIUS_M = 6371000.0

# =========================
# Helpers
# =========================
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

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

def parse_created_datetime(df: pd.DataFrame) -> pd.Series:
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

def ensure_hotspot_unit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hotspot_unit" in df.columns:
        df["hotspot_unit"] = df["hotspot_unit"].fillna("").astype(str).str.strip()
        df.loc[df["hotspot_unit"].eq(""), "hotspot_unit"] = np.nan
        return df

    if {"junction_name", "police_station"}.issubset(df.columns):
        def make_hotspot_unit(row):
            junction = clean_text(row.get("junction_name", ""))
            if junction and junction.upper() != "NO JUNCTION":
                return f"JUNCTION::{junction}"
            station = clean_text(row.get("police_station", "UNKNOWN"))
            if not station:
                station = "UNKNOWN"
            return f"POLICE_STATION::{station}"
        df["hotspot_unit"] = df.apply(make_hotspot_unit, axis=1)
        return df

    df["hotspot_unit"] = "UNKNOWN"
    return df

def ensure_severity(df: pd.DataFrame) -> pd.DataFrame:
    if "severity_score" in df.columns:
        df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce").fillna(1).astype(int)
        return df

    severity_map = {
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

    def severity_from_tags(tags):
        if not tags:
            return 1
        normalized = [clean_text(t).upper().replace("&", "AND").strip() for t in tags]
        score = 1
        for sev in sorted(severity_map.keys(), reverse=True):
            vocab = severity_map[sev]
            if any(any(v == tag or v in tag for v in vocab) for tag in normalized):
                score = sev
                break
        return score

    if "violation_type" in df.columns:
        df["violation_tags"] = df["violation_type"].apply(parse_listlike)
        df["severity_score"] = df["violation_tags"].apply(severity_from_tags)
    else:
        df["severity_score"] = 1

    return df

def project_latlon_to_xy(lat, lon):
    """
    Project lat/lon to a local meter-based plane using an equirectangular approximation.
    Works well for city-scale clustering.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    lat0 = np.deg2rad(np.nanmean(lat))
    lon0 = np.deg2rad(np.nanmean(lon))

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    x = EARTH_RADIUS_M * (lon_rad - lon0) * np.cos(lat0)
    y = EARTH_RADIUS_M * (lat_rad - lat0)
    return np.c_[x, y]

def st_dbscan(coords_xy, time_days, eps_spatial_m, eps_temporal_days, min_pts):
    """
    ST-DBSCAN using:
    - spatial radius search on projected XY coordinates (meters)
    - temporal proximity filter in days
    - DBSCAN-style BFS expansion
    """
    n = len(coords_xy)
    tree = BallTree(coords_xy, metric="euclidean")

    labels = np.full(n, -99, dtype=int)  # -99 = unassigned, -1 = noise
    visited = np.zeros(n, dtype=bool)
    core_mask = np.zeros(n, dtype=bool)
    neighbor_cache = {}

    def region_query(i):
        if i in neighbor_cache:
            return neighbor_cache[i]

        idx = tree.query_radius(coords_xy[i:i + 1], r=eps_spatial_m, return_distance=False)[0]
        if idx.size == 0:
            neighbor_cache[i] = idx
            return idx

        temporal_ok = np.abs(time_days[idx] - time_days[i]) <= eps_temporal_days
        neigh = idx[temporal_ok]
        neighbor_cache[i] = neigh
        return neigh

    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = region_query(i)

        if len(neighbors) < min_pts:
            labels[i] = -1
            continue

        cluster_id += 1
        labels[i] = cluster_id
        core_mask[i] = True

        seed_queue = deque()
        queued = set()

        for j in neighbors:
            if j != i and j not in queued:
                seed_queue.append(j)
                queued.add(j)

        while seed_queue:
            j = seed_queue.popleft()

            if not visited[j]:
                visited[j] = True
                j_neighbors = region_query(j)
                if len(j_neighbors) >= min_pts:
                    core_mask[j] = True
                    for k in j_neighbors:
                        if k not in queued:
                            seed_queue.append(k)
                            queued.add(k)

            if labels[j] in (-99, -1):
                labels[j] = cluster_id

    return labels, core_mask

def cluster_label_for_group(g: pd.DataFrame) -> str:
    if "junction_name" in g.columns:
        junctions = g["junction_name"].fillna("").astype(str)
        junctions = junctions[junctions.str.strip().ne("")]
        junctions = junctions[junctions.str.upper().ne("NO JUNCTION")]
        if len(junctions):
            mode = junctions.mode()
            if not mode.empty:
                return mode.iloc[0]

    if "hotspot_unit" in g.columns:
        units = g["hotspot_unit"].fillna("").astype(str)
        units = units[units.str.strip().ne("")]
        if len(units):
            mode = units.mode()
            if not mode.empty:
                return mode.iloc[0]

    if "police_station" in g.columns:
        stations = g["police_station"].fillna("").astype(str)
        stations = stations[stations.str.strip().ne("")]
        if len(stations):
            mode = stations.mode()
            if not mode.empty:
                return f"POLICE_STATION::{mode.iloc[0]}"

    return f"Cluster {g.name}"

def week_start_monday(series):
    dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    week_start = dt.dt.normalize() - pd.to_timedelta(dt.dt.weekday, unit="D")
    return week_start.dt.tz_localize(None)

# =========================
# Load
# =========================
def load_input():
    if PHASE2_APPROVED_PATH.exists():
        return pd.read_csv(PHASE2_APPROVED_PATH, low_memory=False), PHASE2_APPROVED_PATH
    return pd.read_csv(RAW_CSV, low_memory=False), RAW_CSV

def main():
    df, source = load_input()

    required_cols = {"latitude", "longitude", "created_datetime"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Stage 1 carry-through: approved records only
    if "validation_status" in df.columns:
        df["validation_status_clean"] = df["validation_status"].fillna("").astype(str).str.lower()
        df = df[df["validation_status_clean"].eq("approved")].copy()

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["created_datetime_ist"] = parse_created_datetime(df)
    df = df.dropna(subset=["created_datetime_ist"]).copy()

    df = ensure_hotspot_unit(df)
    df = ensure_severity(df)

    df["created_datetime_ist_naive"] = df["created_datetime_ist"].dt.tz_localize(None)
    df["cluster_week_start"] = week_start_monday(df["created_datetime_ist"])

    # Project coordinates to meters for spatial radius search
    coords_xy = project_latlon_to_xy(df["latitude"].values, df["longitude"].values)

    # Convert timestamps to days since epoch for temporal radius search
    time_days = (
        df["created_datetime_ist_naive"].astype("int64").values.astype(np.float64)
        / 86_400_000_000_000.0
    )

    print("Input source:", source)
    print("Approved rows used:", len(df))
    print("Running ST-DBSCAN with:")
    print(f"  spatial epsilon = {EPS_SPATIAL_METERS} meters")
    print(f"  temporal epsilon = {EPS_TEMPORAL_DAYS} days")
    print(f"  min_pts         = {MIN_PTS}")

    labels, core_mask = st_dbscan(
        coords_xy=coords_xy,
        time_days=time_days,
        eps_spatial_m=EPS_SPATIAL_METERS,
        eps_temporal_days=EPS_TEMPORAL_DAYS,
        min_pts=MIN_PTS,
    )

    df["st_dbscan_cluster_id"] = labels
    df["is_core_point"] = core_mask.astype(int)
    df["is_noise"] = (df["st_dbscan_cluster_id"] == -1).astype(int)

    # Human-readable labels
    cluster_ids = sorted([c for c in df["st_dbscan_cluster_id"].unique() if c != -1])
    cluster_summary_rows = []

    for cid in cluster_ids:
        g = df[df["st_dbscan_cluster_id"] == cid].copy()
        if g.empty:
            continue

        label = cluster_label_for_group(g)

        cluster_summary_rows.append({
            "st_dbscan_cluster_id": cid,
            "cluster_label": label,
            "records": len(g),
            "core_points": int(g["is_core_point"].sum()),
            "noise_points": int(g["is_noise"].sum()),
            "start_time_ist": g["created_datetime_ist"].min(),
            "end_time_ist": g["created_datetime_ist"].max(),
            "duration_days": float((g["created_datetime_ist"].max() - g["created_datetime_ist"].min()).total_seconds() / 86400.0) if len(g) else 0.0,
            "latitude_mean": float(g["latitude"].mean()),
            "longitude_mean": float(g["longitude"].mean()),
            "unique_vehicles": int(g["vehicle_number"].nunique()) if "vehicle_number" in g.columns else 0,
            "dominant_vehicle_type": (
                g["vehicle_type"].mode().iloc[0] if "vehicle_type" in g.columns and not g["vehicle_type"].mode().empty else "UNKNOWN"
            ),
            "dominant_police_station": (
                g["police_station"].mode().iloc[0] if "police_station" in g.columns and not g["police_station"].mode().empty else ""
            ),
            "dominant_junction_name": (
                g["junction_name"].mode().iloc[0] if "junction_name" in g.columns and not g["junction_name"].mode().empty else ""
            ),
            "severity_sum": float(g["severity_score"].sum()),
            "severity_mean": float(g["severity_score"].mean()),
        })

    cluster_summary = pd.DataFrame(cluster_summary_rows).sort_values(
        ["records", "severity_sum"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Weekly counts per cluster (for later trend modules)
    weekly_cluster_counts = (
        df[df["st_dbscan_cluster_id"] != -1]
        .groupby(["st_dbscan_cluster_id", "cluster_week_start"])
        .size()
        .reset_index(name="weekly_count")
        .sort_values(["st_dbscan_cluster_id", "cluster_week_start"])
    )

    # Noise points
    noise_df = df[df["st_dbscan_cluster_id"] == -1].copy()

    # Save outputs
    df.to_csv(OUT_DIR / "phase3_clustered_dataset.csv", index=False)
    cluster_summary.to_csv(OUT_DIR / "phase3_cluster_summary.csv", index=False)
    weekly_cluster_counts.to_csv(OUT_DIR / "phase3_cluster_weekly_counts.csv", index=False)
    noise_df.to_csv(OUT_DIR / "phase3_noise_points.csv", index=False)

    # Compact dispatch view
    dispatch_view_cols = [
        "st_dbscan_cluster_id", "cluster_label", "records", "severity_sum",
        "severity_mean", "unique_vehicles", "dominant_vehicle_type",
        "dominant_police_station", "dominant_junction_name",
        "latitude_mean", "longitude_mean", "start_time_ist", "end_time_ist"
    ]
    dispatch_view = cluster_summary[[c for c in dispatch_view_cols if c in cluster_summary.columns]].copy()
    dispatch_view.to_csv(OUT_DIR / "phase3_cluster_dispatch_view.csv", index=False)

    # Summary
    print("\nStage 3 complete")
    print("Clusters found:", len(cluster_summary))
    print("Noise points:", len(noise_df))
    print("Clustered rows:", len(df) - len(noise_df))
    print("Outputs saved to:", OUT_DIR.resolve())

    if len(cluster_summary):
        print("\nTop 10 clusters:")
        print(
            cluster_summary[[
                "st_dbscan_cluster_id", "cluster_label", "records",
                "severity_sum", "severity_mean", "unique_vehicles"
            ]].head(10).to_string(index=False)
        )

if __name__ == "__main__":
    main()