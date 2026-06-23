# Stage 4 — Implicit μ Estimation / Dwell-Time Estimation
# Input : Phase 3 clustered approved records
# Output: gap event log, cluster-level μ summary, cluster×vehicle-type μ summary,
#         vehicle-type summary, and a merged dataset for Stage 5

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
PHASE3_PATHS = [
    Path("content/phase3_outputs_2/phase3_clustered_dataset.csv"),
    Path("content/phase3_outputs_2/phase3_cluster_dispatch_view.csv"),
]

OUT_DIR = Path("content/phase4_outputs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9

# =========================
# Helpers
# =========================
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

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

def standardize_cluster_col(df: pd.DataFrame) -> str:
    for c in ["st_dbscan_cluster_id", "cluster_id", "dbscan_cluster_id"]:
        if c in df.columns:
            return c
    raise ValueError("No cluster id column found.")

def standardize_vehicle_col(df: pd.DataFrame) -> str:
    if "updated_vehicle_number" in df.columns and "vehicle_number" in df.columns:
        updated = df["updated_vehicle_number"].fillna("").astype(str).str.strip()
        original = df["vehicle_number"].fillna("").astype(str).str.strip()
        df["canonical_vehicle_number"] = np.where(updated.ne(""), updated, original)
        return "canonical_vehicle_number"

    if "vehicle_number" in df.columns:
        df["canonical_vehicle_number"] = df["vehicle_number"].fillna("").astype(str).str.strip()
        return "canonical_vehicle_number"

    raise ValueError("No vehicle number column found.")

def standardize_vehicle_type_col(df: pd.DataFrame) -> str:
    if "updated_vehicle_type" in df.columns and "vehicle_type" in df.columns:
        updated = df["updated_vehicle_type"].fillna("").astype(str).str.strip()
        original = df["vehicle_type"].fillna("").astype(str).str.strip()
        df["canonical_vehicle_type"] = np.where(updated.ne(""), updated, original)
        return "canonical_vehicle_type"

    if "vehicle_type" in df.columns:
        df["canonical_vehicle_type"] = df["vehicle_type"].fillna("UNKNOWN").astype(str).str.strip()
        df.loc[df["canonical_vehicle_type"].eq(""), "canonical_vehicle_type"] = "UNKNOWN"
        return "canonical_vehicle_type"

    df["canonical_vehicle_type"] = "UNKNOWN"
    return "canonical_vehicle_type"

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

def mode_non_empty(series, exclude=None, default=""):
    s = pd.Series(series).dropna().astype(str).str.strip()
    s = s[s.ne("")]

    #  remove excluded placeholders like "No Junction"
    if exclude:
        excl = {clean_text(x).lower() for x in exclude}
        s = s[~s.str.lower().isin(excl)]

    if s.empty:
        return default

    m = s.mode()
    if not m.empty:
        return m.iloc[0]
    return s.iloc[0]

def build_cluster_label_for_group(g: pd.DataFrame) -> str:
    # Prefer a real named junction
    if "junction_name" in g.columns:
        junction = mode_non_empty(g["junction_name"], exclude={"No Junction", "NO JUNCTION"}, default="")
        if junction:
            return junction

    # Then a stable hotspot_unit
    if "hotspot_unit" in g.columns:
        unit = mode_non_empty(g["hotspot_unit"], default="")
        if unit:
            return unit

    # Then police station
    if "police_station" in g.columns:
        station = mode_non_empty(g["police_station"], default="")
        if station:
            return f"POLICE_STATION::{station}"

    return f"Cluster {g.name}"

def week_start_monday(series):
    dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    week_start = dt.dt.normalize() - pd.to_timedelta(dt.dt.weekday, unit="D")
    return week_start.dt.tz_localize(None)

# =========================
# Load
# =========================
def load_input():
    for p in PHASE3_PATHS:
        if p.exists():
            return pd.read_csv(p, low_memory=False), p
    raise FileNotFoundError("Could not find Phase 3 output file.")

def main():
    df, source = load_input()

    if "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError("Phase 3 dataset must include latitude and longitude.")

    cluster_col = standardize_cluster_col(df)
    vehicle_col = standardize_vehicle_col(df)
    vehicle_type_col = standardize_vehicle_type_col(df)

    # Keep approved records only if the column exists
    if "validation_status_clean" in df.columns:
        df["validation_status_clean"] = df["validation_status_clean"].fillna("").astype(str).str.lower()
        df = df[df["validation_status_clean"].eq("approved")].copy()
    elif "validation_status" in df.columns:
        df["validation_status_clean"] = df["validation_status"].fillna("").astype(str).str.lower()
        df = df[df["validation_status_clean"].eq("approved")].copy()

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = ensure_hotspot_unit(df)
    df["created_datetime_ist"] = parse_datetime_ist(df)
    df = df.dropna(subset=["created_datetime_ist"]).copy()

    # Exclude noise
    df[cluster_col] = pd.to_numeric(df[cluster_col], errors="coerce")
    df = df[df[cluster_col].ne(-1)].copy()

    df = df.sort_values([cluster_col, vehicle_col, "created_datetime_ist"]).reset_index(drop=True)
    df["created_datetime_ist_naive"] = df["created_datetime_ist"].dt.tz_localize(None)
    df["service_date"] = df["created_datetime_ist_naive"].dt.date
    df["cluster_week_start"] = week_start_monday(df["created_datetime_ist"])

    # Same vehicle + same cluster + same day consecutive gaps
    group_keys = [cluster_col, vehicle_col, "service_date"]
    df["prev_created_datetime_ist"] = df.groupby(group_keys)["created_datetime_ist_naive"].shift(1)
    df["gap_minutes"] = (
        (df["created_datetime_ist_naive"] - df["prev_created_datetime_ist"])
        .dt.total_seconds() / 60.0
    )
    df["gap_hours"] = df["gap_minutes"] / 60.0

    gap_log = df[df["gap_minutes"].notna() & (df["gap_minutes"] > 0)].copy()

    # If no gaps are found, still save empty outputs and stop cleanly
    if gap_log.empty:
        empty_cols = [
            cluster_col, "cluster_label", "gap_count", "unique_vehicles",
            "unique_vehicle_types", "mean_dwell_minutes", "median_dwell_minutes",
            "std_dwell_minutes", "mu_departures_per_hour"
        ]
        empty_df = pd.DataFrame(columns=empty_cols)

        empty_df.to_csv(OUT_DIR / "phase4_cluster_mu_summary.csv", index=False)
        empty_df.to_csv(OUT_DIR / "phase4_cluster_vehicle_type_mu_summary.csv", index=False)
        empty_df.to_csv(OUT_DIR / "phase4_vehicle_type_mu_summary.csv", index=False)
        df.to_csv(OUT_DIR / "phase4_merged_with_prior_scores.csv", index=False)

        print("Stage 4 complete")
        print("Input source:", source)
        print("No valid inter-record gaps found.")
        print("Empty outputs saved to:", OUT_DIR.resolve())
        return

    # ---------- Cluster label map ----------
    cluster_label_map = (
        df.groupby(cluster_col, sort=False)
        .apply(build_cluster_label_for_group)
        .reset_index(name="cluster_label")
    )

    # ---------- Overall vehicle-type summary ----------
    vehicle_type_summary = (
        gap_log.groupby(vehicle_type_col, dropna=False)
        .agg(
            gap_count=("gap_minutes", "size"),
            unique_clusters=(cluster_col, "nunique"),
            unique_vehicles=(vehicle_col, "nunique"),
            mean_dwell_minutes=("gap_minutes", "mean"),
            median_dwell_minutes=("gap_minutes", "median"),
            std_dwell_minutes=("gap_minutes", lambda s: float(s.std(ddof=0)) if len(s) else 0.0),
            min_gap_minutes=("gap_minutes", "min"),
            max_gap_minutes=("gap_minutes", "max"),
        )
        .reset_index()
        .rename(columns={vehicle_type_col: "vehicle_type"})
        .sort_values(["gap_count", "mean_dwell_minutes"], ascending=[False, False])
    )
    vehicle_type_summary["mu_departures_per_hour"] = 60.0 / (vehicle_type_summary["mean_dwell_minutes"] + EPS)

    # ---------- Cluster-level summary ----------
    cluster_summary = (
        gap_log.groupby(cluster_col, dropna=False)
        .agg(
            gap_count=("gap_minutes", "size"),
            unique_vehicles=(vehicle_col, "nunique"),
            unique_vehicle_types=(vehicle_type_col, "nunique"),
            mean_dwell_minutes=("gap_minutes", "mean"),
            median_dwell_minutes=("gap_minutes", "median"),
            std_dwell_minutes=("gap_minutes", lambda s: float(s.std(ddof=0)) if len(s) else 0.0),
            min_gap_minutes=("gap_minutes", "min"),
            max_gap_minutes=("gap_minutes", "max"),
            first_gap_time=("prev_created_datetime_ist", "min"),
            last_gap_time=("created_datetime_ist_naive", "max"),
        )
        .reset_index()
    )

    cluster_summary = cluster_summary.merge(cluster_label_map, on=cluster_col, how="left")
    cluster_summary["mu_departures_per_hour"] = 60.0 / (cluster_summary["mean_dwell_minutes"] + EPS)

    # ---------- Cluster × vehicle-type summary ----------
    cluster_vehicle_type_summary = (
        gap_log.groupby([cluster_col, vehicle_type_col], dropna=False)
        .agg(
            gap_count=("gap_minutes", "size"),
            unique_vehicles=(vehicle_col, "nunique"),
            mean_dwell_minutes=("gap_minutes", "mean"),
            median_dwell_minutes=("gap_minutes", "median"),
            std_dwell_minutes=("gap_minutes", lambda s: float(s.std(ddof=0)) if len(s) else 0.0),
            min_gap_minutes=("gap_minutes", "min"),
            max_gap_minutes=("gap_minutes", "max"),
        )
        .reset_index()
        .rename(columns={vehicle_type_col: "vehicle_type"})
        .sort_values([cluster_col, "gap_count", "mean_dwell_minutes"], ascending=[True, False, False])
    )
    cluster_vehicle_type_summary["mu_departures_per_hour"] = 60.0 / (
        cluster_vehicle_type_summary["mean_dwell_minutes"] + EPS
    )

    # ---------- Useful merged record-level output for Stage 5 ----------
    merged = df.merge(
        cluster_summary[[
            cluster_col, "cluster_label", "gap_count", "mean_dwell_minutes",
            "median_dwell_minutes", "std_dwell_minutes", "mu_departures_per_hour"
        ]],
        on=cluster_col,
        how="left"
    )

    # ---------- Stage 5 handoff ----------
    mu_lookup = cluster_summary[[
        cluster_col, "cluster_label", "gap_count", "mean_dwell_minutes",
        "median_dwell_minutes", "std_dwell_minutes", "mu_departures_per_hour"
    ]].copy()

    # ---------- Save outputs ----------
    gap_cols = [
        c for c in [
            "id", cluster_col, "cluster_label", vehicle_col, vehicle_type_col,
            "service_date", "created_datetime_ist", "created_datetime_ist_naive",
            "prev_created_datetime_ist", "gap_minutes", "gap_hours",
            "latitude", "longitude", "junction_name", "police_station",
            "severity_score", "hotspot_unit"
        ] if c in gap_log.columns
    ]
    gap_log[gap_cols].to_csv(OUT_DIR / "phase4_gap_event_log.csv", index=False)

    cluster_summary.to_csv(OUT_DIR / "phase4_cluster_mu_summary.csv", index=False)
    cluster_vehicle_type_summary.to_csv(OUT_DIR / "phase4_cluster_vehicle_type_mu_summary.csv", index=False)
    vehicle_type_summary.to_csv(OUT_DIR / "phase4_vehicle_type_mu_summary.csv", index=False)
    merged.to_csv(OUT_DIR / "phase4_merged_with_prior_scores.csv", index=False)
    mu_lookup.to_csv(OUT_DIR / "phase4_stage5_mu_lookup.csv", index=False)

    # ---------- Console summary ----------
    overall_mean = float(gap_log["gap_minutes"].mean())
    overall_mu = 60.0 / (overall_mean + EPS)

    print("Stage 4 complete")
    print("Input source:", source)
    print("Approved clustered rows used:", len(df))
    print("Valid inter-record gaps:", len(gap_log))
    print("Overall mean dwell (min):", round(overall_mean, 3))
    print("Overall μ (departures/hour):", round(overall_mu, 6))
    print("Clusters with μ estimates:", len(cluster_summary))
    print("Vehicle types with μ estimates:", len(vehicle_type_summary))
    print("Outputs saved to:", OUT_DIR.resolve())

    print("\nTop 10 clusters by gap count:")
    if len(cluster_summary):
        print(
            cluster_summary[[
                cluster_col, "cluster_label", "gap_count",
                "mean_dwell_minutes", "mu_departures_per_hour"
            ]].sort_values("gap_count", ascending=False).head(10).to_string(index=False)
        )

if __name__ == "__main__":
    main()