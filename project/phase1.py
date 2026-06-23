# =========================
# Phase 1 — Dataset Intelligence
# Validation Analysis
# Vehicle Behaviour Analysis
# Temporal Analysis
# Violation Analysis
# =========================

import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# -------------------------
# Config
# -------------------------
INPUT_CSV = "jan to may police violation_anonymized791b166.csv"
OUT_DIR = Path("content/phase1_outputs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 25
EPS = 1e-9

# -------------------------
# Helpers
# -------------------------
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

def week_start_monday(series):
    dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    week_start = dt.dt.normalize() - pd.to_timedelta(dt.dt.weekday, unit="D")
    return week_start.dt.tz_localize(None)

def safe_div(a, b):
    return a / (b + EPS)

def ensure_required_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def severity_from_tags(tags):
    """
    Maps violation tags to a 1–5 severity score.
    Uses the highest applicable severity if multiple tags exist.
    """
    severity_map = {
        5: {
            "DOUBLE PARKING",
            "NEAR ROAD CROSSING",
            "NEAR TRAFFIC LIGHT",
            "NEAR TRAFFIC LIGHT / ZEBRA CROSSING",
            "NEAR ZEBRA CROSSING",
        },
        4: {
            "PARKING IN MAIN ROAD",
            "NEAR BUS STOP",
            "NEAR SCHOOL",
            "NEAR HOSPITAL",
            "OPPOSITE ANOTHER VEHICLE",
        },
        3: {
            "PARKING ON FOOTPATH",
        },
        2: {
            "WRONG PARKING",
            "PARKING OTHER THAN BUS STOP",
        },
        1: {
            "NO PARKING",
        },
    }

    if not tags:
        return 1

    normalized = [clean_text(t).upper() for t in tags]
    score = 1
    for s, vocab in severity_map.items():
        if any(tag in vocab for tag in normalized):
            score = max(score, s)
    return score

def dominant_tag(tags):
    if not tags:
        return ""
    return clean_text(tags[0]).upper()

# -------------------------
# Load
# -------------------------
df = pd.read_csv(INPUT_CSV, low_memory=False)

required_cols = [
    "id", "latitude", "longitude", "vehicle_number", "vehicle_type",
    "violation_type", "created_datetime", "police_station",
    "junction_name", "validation_status"
]
ensure_required_columns(df, required_cols)

for col in ["validation_status", "police_station", "junction_name", "vehicle_number", "vehicle_type"]:
    df[col] = df[col].astype("string")

# -------------------------
# Core parsing
# -------------------------
df["created_datetime_parsed"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
df["created_datetime_ist"] = df["created_datetime_parsed"].dt.tz_convert("Asia/Kolkata")
df["week_start"] = week_start_monday(df["created_datetime"])
df["hour_ist"] = df["created_datetime_ist"].dt.hour
df["day_of_week"] = df["created_datetime_ist"].dt.day_name()
df["month"] = df["created_datetime_ist"].dt.month
df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

df["violation_tags"] = df["violation_type"].apply(parse_listlike)
df["tag_count"] = df["violation_tags"].apply(len)
df["severity_score"] = df["violation_tags"].apply(severity_from_tags)
df["dominant_violation_tag"] = df["violation_tags"].apply(dominant_tag)

junction_clean = df["junction_name"].fillna("").astype(str).str.strip()
has_junction = junction_clean.ne("") & junction_clean.str.upper().ne("NO JUNCTION")
police_clean = df["police_station"].fillna("UNKNOWN").astype(str).str.strip()
police_clean = police_clean.where(police_clean.ne(""), "UNKNOWN")

df["hotspot_unit"] = np.where(
    has_junction,
    "JUNCTION::" + junction_clean,
    "POLICE_STATION::" + police_clean
)

validation_clean = df["validation_status"].fillna("").astype(str).str.lower()
df["validation_status_clean"] = validation_clean

# A simple numeric flag useful for later phases
status_score_map = {
    "approved": 1.0,
    "rejected": 0.0,
    "processing": 0.5,
    "duplicate": 0.25,
    "created1": 0.5,
}
df["validation_score"] = df["validation_status_clean"].map(status_score_map).fillna(0.5)

# Keep only rows with valid time and coordinates for analysis tables that need them
df_valid = df.dropna(subset=["created_datetime_parsed", "latitude", "longitude"]).copy()

# -------------------------
# 1) Validation Analysis
# -------------------------
validation_summary = (
    df["validation_status_clean"]
    .replace("", "missing")
    .value_counts(dropna=False)
    .rename_axis("validation_status")
    .reset_index(name="count")
)
validation_summary["percent"] = 100 * validation_summary["count"] / len(df)

approval_rate = safe_div(
    int((df["validation_status_clean"] == "approved").sum()),
    len(df)
)

validation_by_vehicle_type = (
    df.groupby("vehicle_type", dropna=False)
    .agg(
        total_records=("id", "size"),
        approved_records=("validation_status_clean", lambda s: (s == "approved").sum()),
        rejected_records=("validation_status_clean", lambda s: (s == "rejected").sum()),
        approved_rate=("validation_status_clean", lambda s: (s == "approved").mean()),
    )
    .reset_index()
    .sort_values("total_records", ascending=False)
)

validation_by_police_station = (
    df.groupby("police_station", dropna=False)
    .agg(
        total_records=("id", "size"),
        approved_rate=("validation_status_clean", lambda s: (s == "approved").mean()),
        rejected_rate=("validation_status_clean", lambda s: (s == "rejected").mean()),
    )
    .reset_index()
    .sort_values(["total_records", "approved_rate"], ascending=[False, False])
)

validation_by_hotspot = (
    df.groupby("hotspot_unit", dropna=False)
    .agg(
        total_records=("id", "size"),
        approved_rate=("validation_status_clean", lambda s: (s == "approved").mean()),
        rejected_rate=("validation_status_clean", lambda s: (s == "rejected").mean()),
        processed_rate=("validation_status_clean", lambda s: (s == "processing").mean()),
    )
    .reset_index()
    .sort_values("total_records", ascending=False)
)

# Validation uncertainty score: high when approval rate is low or mixed
validation_by_hotspot["validation_uncertainty"] = 1.0 - validation_by_hotspot["approved_rate"]
validation_by_vehicle_type["validation_uncertainty"] = 1.0 - validation_by_vehicle_type["approved_rate"]

# -------------------------
# 2) Vehicle Behaviour Analysis
# -------------------------
vehicle_summary = (
    df.groupby("vehicle_number", dropna=False)
    .agg(
        total_violations=("id", "size"),
        unique_hotspots=("hotspot_unit", "nunique"),
        unique_junctions=("junction_name", lambda s: s.fillna("").astype(str).replace("No Junction", "").nunique()),
        first_seen=("created_datetime_parsed", "min"),
        last_seen=("created_datetime_parsed", "max"),
        vehicle_type_mode=("vehicle_type", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
        approved_violations=("validation_status_clean", lambda s: (s == "approved").sum()),
    )
    .reset_index()
)

vehicle_summary["active_days"] = (
    df_valid.groupby("vehicle_number")["created_datetime_ist"]
    .apply(lambda s: s.dt.date.nunique())
    .reindex(vehicle_summary["vehicle_number"])
    .values
)

vehicle_summary["repeat_violation_flag"] = (vehicle_summary["total_violations"] >= 2).astype(int)
vehicle_summary["chronic_offender_flag"] = (vehicle_summary["total_violations"] >= 5).astype(int)

vehicle_summary["approval_rate"] = safe_div(
    vehicle_summary["approved_violations"],
    vehicle_summary["total_violations"]
)

vehicle_summary = vehicle_summary.sort_values(
    ["total_violations", "approved_violations", "unique_hotspots"],
    ascending=[False, False, False]
).reset_index(drop=True)

top_vehicle_offenders = vehicle_summary.head(TOP_N).copy()
chronic_offenders = vehicle_summary[vehicle_summary["chronic_offender_flag"] == 1].copy()

# Per vehicle-type behaviour
vehicle_type_summary = (
    df.groupby("vehicle_type", dropna=False)
    .agg(
        records=("id", "size"),
        unique_vehicles=("vehicle_number", "nunique"),
        mean_severity=("severity_score", "mean"),
        median_severity=("severity_score", "median"),
        approval_rate=("validation_status_clean", lambda s: (s == "approved").mean()),
    )
    .reset_index()
    .sort_values("records", ascending=False)
)

# -------------------------
# 3) Temporal Analysis
# -------------------------
hourly_distribution = (
    df_valid.groupby("hour_ist")
    .size()
    .reset_index(name="count")
    .sort_values("hour_ist")
)
daily_distribution = (
    df_valid.groupby("day_of_week")
    .size()
    .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    .reset_index(name="count")
)
weekly_distribution = (
    df_valid.groupby("week_start")
    .size()
    .reset_index(name="count")
    .sort_values("week_start")
)
monthly_distribution = (
    df_valid.groupby("month")
    .size()
    .reset_index(name="count")
    .sort_values("month")
)

temporal_by_hour_vehicle_type = (
    df_valid.groupby(["hour_ist", "vehicle_type"])
    .size()
    .reset_index(name="count")
)

temporal_by_day_validation = (
    df_valid.groupby(["day_of_week", "validation_status_clean"])
    .size()
    .reset_index(name="count")
)

peak_hour = int(hourly_distribution.loc[hourly_distribution["count"].idxmax(), "hour_ist"]) if len(hourly_distribution) else None
peak_day = daily_distribution.loc[daily_distribution["count"].idxmax(), "day_of_week"] if len(daily_distribution) else None

# Trend by week: useful for later emerging hotspot stage
weekly_trend_by_hotspot = (
    df_valid.groupby(["hotspot_unit", "week_start"])
    .size()
    .reset_index(name="weekly_count")
    .sort_values(["hotspot_unit", "week_start"])
)

# -------------------------
# 4) Violation Analysis
# -------------------------
# Tag-level analysis
tag_rows = []
for _, row in df.iterrows():
    tags = row["violation_tags"]
    if not tags:
        tag_rows.append({"violation_tag": "UNKNOWN", "id": row["id"], "vehicle_number": row["vehicle_number"], "hotspot_unit": row["hotspot_unit"]})
    else:
        for t in tags:
            tag_rows.append({
                "violation_tag": clean_text(t).upper(),
                "id": row["id"],
                "vehicle_number": row["vehicle_number"],
                "hotspot_unit": row["hotspot_unit"],
            })

tag_df = pd.DataFrame(tag_rows)

violation_tag_summary = (
    tag_df.groupby("violation_tag")
    .agg(
        tag_frequency=("id", "size"),
        unique_vehicles=("vehicle_number", "nunique"),
        unique_hotspots=("hotspot_unit", "nunique"),
    )
    .reset_index()
    .sort_values("tag_frequency", ascending=False)
)

severity_distribution = (
    df.groupby("severity_score")
    .size()
    .reset_index(name="count")
    .sort_values("severity_score")
)

violation_by_vehicle_type = (
    df.groupby("vehicle_type")
    .agg(
        records=("id", "size"),
        mean_severity=("severity_score", "mean"),
        approval_rate=("validation_status_clean", lambda s: (s == "approved").mean()),
    )
    .reset_index()
    .sort_values("records", ascending=False)
)

violation_by_hotspot = (
    df.groupby("hotspot_unit")
    .agg(
        records=("id", "size"),
        mean_severity=("severity_score", "mean"),
        approval_rate=("validation_status_clean", lambda s: (s == "approved").mean()),
        unique_vehicles=("vehicle_number", "nunique"),
    )
    .reset_index()
    .sort_values(["records", "mean_severity"], ascending=[False, False])
)

# Tag co-occurrence patterns
combo_counts = (
    df["violation_tags"]
    .apply(lambda x: tuple(sorted([clean_text(t).upper() for t in x])) if x else ("UNKNOWN",))
    .value_counts()
    .reset_index()
)
combo_counts.columns = ["violation_combo", "frequency"]
combo_counts["violation_combo"] = combo_counts["violation_combo"].apply(lambda x: " | ".join(x))

# -------------------------
# Consolidated Phase 1 feature table
# -------------------------
phase1_features = df.copy()

# Useful derived columns for later stages
phase1_features["validation_is_approved"] = (phase1_features["validation_status_clean"] == "approved").astype(int)
phase1_features["validation_is_rejected"] = (phase1_features["validation_status_clean"] == "rejected").astype(int)
phase1_features["vehicle_total_violations"] = phase1_features.groupby("vehicle_number")["id"].transform("count")
phase1_features["vehicle_chronic_flag"] = (phase1_features["vehicle_total_violations"] >= 5).astype(int)
phase1_features["vehicle_repeat_flag"] = (phase1_features["vehicle_total_violations"] >= 2).astype(int)

phase1_features["hotspot_total_records"] = phase1_features.groupby("hotspot_unit")["id"].transform("count")
phase1_features["hotspot_approval_rate"] = phase1_features.groupby("hotspot_unit")["validation_is_approved"].transform("mean")
phase1_features["hotspot_uncertainty"] = 1.0 - phase1_features["hotspot_approval_rate"]

phase1_features["vehicle_type_total_records"] = phase1_features.groupby("vehicle_type")["id"].transform("count")
phase1_features["vehicle_type_mean_severity"] = phase1_features.groupby("vehicle_type")["severity_score"].transform("mean")

phase1_features["tag_frequency_record"] = phase1_features["dominant_violation_tag"].map(
    violation_tag_summary.set_index("violation_tag")["tag_frequency"]
).fillna(0).astype(int)

# -------------------------
# Save outputs
# -------------------------
validation_summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)
validation_by_vehicle_type.to_csv(OUT_DIR / "validation_by_vehicle_type.csv", index=False)
validation_by_police_station.to_csv(OUT_DIR / "validation_by_police_station.csv", index=False)
validation_by_hotspot.to_csv(OUT_DIR / "validation_by_hotspot.csv", index=False)

vehicle_summary.to_csv(OUT_DIR / "vehicle_summary.csv", index=False)
top_vehicle_offenders.to_csv(OUT_DIR / "top_vehicle_offenders.csv", index=False)
chronic_offenders.to_csv(OUT_DIR / "chronic_offenders.csv", index=False)
vehicle_type_summary.to_csv(OUT_DIR / "vehicle_type_summary.csv", index=False)

hourly_distribution.to_csv(OUT_DIR / "hourly_distribution.csv", index=False)
daily_distribution.to_csv(OUT_DIR / "daily_distribution.csv", index=False)
weekly_distribution.to_csv(OUT_DIR / "weekly_distribution.csv", index=False)
monthly_distribution.to_csv(OUT_DIR / "monthly_distribution.csv", index=False)
temporal_by_hour_vehicle_type.to_csv(OUT_DIR / "temporal_by_hour_vehicle_type.csv", index=False)
temporal_by_day_validation.to_csv(OUT_DIR / "temporal_by_day_validation.csv", index=False)
weekly_trend_by_hotspot.to_csv(OUT_DIR / "weekly_trend_by_hotspot.csv", index=False)

violation_tag_summary.to_csv(OUT_DIR / "violation_tag_summary.csv", index=False)
severity_distribution.to_csv(OUT_DIR / "severity_distribution.csv", index=False)
violation_by_vehicle_type.to_csv(OUT_DIR / "violation_by_vehicle_type.csv", index=False)
violation_by_hotspot.to_csv(OUT_DIR / "violation_by_hotspot.csv", index=False)
combo_counts.to_csv(OUT_DIR / "violation_combo_summary.csv", index=False)

phase1_features.to_csv(OUT_DIR / "phase1_enriched_dataset.csv", index=False)

# -------------------------
# Print summary
# -------------------------
print("Phase 1 complete")
print("Total rows:", len(df))
print("Approved rows:", int((df["validation_status_clean"] == "approved").sum()))
print("Approval rate:", round(approval_rate * 100, 2), "%")
print("Unique vehicles:", df["vehicle_number"].nunique())
print("Unique hotspots:", df["hotspot_unit"].nunique())
print("Peak hour:", peak_hour)
print("Peak day:", peak_day)
print("Chronic offenders (>=5):", len(chronic_offenders))
print("Outputs saved to:", OUT_DIR.resolve())

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(10, 5))
plt.bar(validation_summary["validation_status"].astype(str), validation_summary["count"])
plt.title("Validation Status Distribution")
plt.xlabel("Validation Status")
plt.ylabel("Count")
save_plot(OUT_DIR / "validation_status_distribution.png")

plt.figure(figsize=(12, 5))
plt.bar(hourly_distribution["hour_ist"].astype(int), hourly_distribution["count"])
plt.title("Hourly Violation Distribution (IST)")
plt.xlabel("Hour")
plt.ylabel("Count")
save_plot(OUT_DIR / "hourly_distribution.png")

plt.figure(figsize=(12, 5))
plt.bar(daily_distribution["day_of_week"].astype(str), daily_distribution["count"])
plt.title("Day-of-Week Violation Distribution")
plt.xlabel("Day")
plt.ylabel("Count")
plt.xticks(rotation=30)
save_plot(OUT_DIR / "daily_distribution.png")

plt.figure(figsize=(12, 6))
top_tags = violation_tag_summary.head(TOP_N)
plt.barh(top_tags["violation_tag"][::-1], top_tags["tag_frequency"][::-1])
plt.title("Top Violation Tags")
plt.xlabel("Frequency")
plt.ylabel("Violation Tag")
save_plot(OUT_DIR / "top_violation_tags.png")

plt.figure(figsize=(12, 6))
top_vehicles = top_vehicle_offenders.head(TOP_N).sort_values("total_violations")
plt.barh(top_vehicles["vehicle_number"], top_vehicles["total_violations"])
plt.title("Top Vehicle Offenders")
plt.xlabel("Total Violations")
plt.ylabel("Vehicle Number")
save_plot(OUT_DIR / "top_vehicle_offenders.png")
