import ast
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

INPUT_CSV = "jan to may police violation_anonymized791b166.csv"
PHASE1_PATH = Path("content/phase1_outputs_2/phase1_enriched_dataset.csv")
OUT_DIR = Path("content/phase2_outputs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9
TOP_N = 25

SEVERITY_RULES = {
    5: [
        "DOUBLE PARKING",
        "NEAR ROAD CROSSING",
        "NEAR TRAFFIC LIGHT",
        "NEAR ZEBRA CROSSING",
        "NEAR TRAFFIC LIGHT / ZEBRA CROSSING",
        "NEAR TRAFFIC LIGHT/ZEBRA CROSSING",
    ],
    4: [
        "PARKING IN MAIN ROAD",
        "NEAR BUS STOP",
        "NEAR SCHOOL",
        "NEAR HOSPITAL",
        "OPPOSITE ANOTHER VEHICLE",
    ],
    3: [
        "PARKING ON FOOTPATH",
    ],
    2: [
        "WRONG PARKING",
        "PARKING OTHER THAN BUS STOP",
    ],
    1: [
        "NO PARKING",
        "NO PARKING (GENERIC)",
    ],
}

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

def normalize_tag(tag):
    s = clean_text(tag).upper()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    s = s.replace("&", "AND")
    return s.strip()

def make_hotspot_unit(row):
    junction = clean_text(row.get("junction_name", ""))
    if junction and junction.upper() != "NO JUNCTION":
        return f"JUNCTION::{junction}"
    station = clean_text(row.get("police_station", "UNKNOWN"))
    if not station:
        station = "UNKNOWN"
    return f"POLICE_STATION::{station}"

def severity_for_tags(tags):
    if not tags:
        return 1, [], "NO PARKING"

    normalized = [normalize_tag(t) for t in tags]
    hit_tags = []
    best = 1

    for tag in normalized:
        tag_best = 1
        matched = False
        for sev in sorted(SEVERITY_RULES.keys(), reverse=True):
            for pattern in SEVERITY_RULES[sev]:
                p = normalize_tag(pattern)
                if p == tag or p in tag:
                    tag_best = sev
                    matched = True
                    break
            if matched:
                break
        best = max(best, tag_best)
        if matched:
            hit_tags.append(tag)

    dominant = hit_tags[0] if hit_tags else normalized[0]
    return best, hit_tags, dominant

def safe_ratio(num, den):
    return num / (den + EPS)

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    source_path = PHASE1_PATH if PHASE1_PATH.exists() else Path(INPUT_CSV)
    df = pd.read_csv(source_path, low_memory=False)

    if "violation_type" not in df.columns:
        raise ValueError("Missing required column: violation_type")

    if "hotspot_unit" not in df.columns:
        if {"junction_name", "police_station"}.issubset(df.columns):
            df["hotspot_unit"] = df.apply(make_hotspot_unit, axis=1)
        else:
            df["hotspot_unit"] = "UNKNOWN"

    if "validation_status" in df.columns:
        df["validation_status_clean"] = df["validation_status"].fillna("").astype(str).str.lower()
    else:
        df["validation_status_clean"] = "approved"

    df["violation_tags"] = df["violation_type"].apply(parse_listlike)
    df["violation_tag_count"] = df["violation_tags"].apply(len)

    sev_res = df["violation_tags"].apply(severity_for_tags)
    df["severity_score"] = sev_res.apply(lambda x: int(x[0]))
    df["matched_severity_tags"] = sev_res.apply(lambda x: x[1])
    df["dominant_violation_tag"] = sev_res.apply(lambda x: x[2])
    df["severity_normalized"] = df["severity_score"] / 5.0

    df["severity_is_very_high"] = (df["severity_score"] == 5).astype(int)
    df["severity_is_high"] = (df["severity_score"] == 4).astype(int)
    df["severity_is_moderate"] = (df["severity_score"] == 3).astype(int)
    df["severity_is_low"] = (df["severity_score"] == 2).astype(int)
    df["severity_is_baseline"] = (df["severity_score"] == 1).astype(int)

    approved_mask = df["validation_status_clean"].eq("approved")
    approved = df.loc[approved_mask].copy()

    severity_record_cols = [
        c for c in [
            "id", "created_datetime", "created_datetime_parsed", "created_datetime_ist",
            "latitude", "longitude", "vehicle_number", "vehicle_type",
            "police_station", "junction_name", "hotspot_unit",
            "validation_status", "validation_status_clean",
            "violation_type", "violation_tags", "violation_tag_count",
            "severity_score", "severity_normalized", "matched_severity_tags",
            "dominant_violation_tag"
        ] if c in df.columns
    ]

    hotspot_summary = (
        approved.groupby("hotspot_unit", dropna=False)
        .agg(
            approved_records=("severity_score", "size"),
            mean_severity=("severity_score", "mean"),
            median_severity=("severity_score", "median"),
            severity_sum=("severity_score", "sum"),
            very_high_count=("severity_is_very_high", "sum"),
            high_count=("severity_is_high", "sum"),
            moderate_count=("severity_is_moderate", "sum"),
            low_count=("severity_is_low", "sum"),
            baseline_count=("severity_is_baseline", "sum"),
            unique_vehicles=("vehicle_number", "nunique") if "vehicle_number" in approved.columns else ("severity_score", "size"),
        )
        .reset_index()
    )

    hotspot_summary["severity_share"] = safe_ratio(hotspot_summary["severity_sum"], hotspot_summary["severity_sum"].sum())
    hotspot_summary["severity_priority_score"] = (
        0.50 * (hotspot_summary["severity_sum"] / (hotspot_summary["severity_sum"].max() + EPS)) +
        0.30 * (hotspot_summary["mean_severity"] / 5.0) +
        0.20 * safe_ratio(hotspot_summary["approved_records"], hotspot_summary["approved_records"].max())
    ) * 100.0

    hotspot_summary = hotspot_summary.sort_values(
        ["severity_priority_score", "severity_sum", "approved_records"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    hotspot_summary["severity_rank"] = np.arange(1, len(hotspot_summary) + 1)

    vehicle_type_summary = (
        approved.groupby("vehicle_type", dropna=False)
        .agg(
            approved_records=("severity_score", "size"),
            mean_severity=("severity_score", "mean"),
            median_severity=("severity_score", "median"),
            severity_sum=("severity_score", "sum"),
            unique_vehicles=("vehicle_number", "nunique") if "vehicle_number" in approved.columns else ("severity_score", "size"),
        )
        .reset_index()
        .sort_values(["severity_sum", "approved_records"], ascending=[False, False])
    )

    tag_rows = []
    for _, row in approved.iterrows():
        tags = row["violation_tags"]
        if not tags:
            tag_rows.append({"violation_tag": "NO TAG", "severity_score": row["severity_score"], "hotspot_unit": row["hotspot_unit"]})
        else:
            for t in tags:
                tag_rows.append({
                    "violation_tag": normalize_tag(t),
                    "severity_score": row["severity_score"],
                    "hotspot_unit": row["hotspot_unit"],
                })

    tag_df = pd.DataFrame(tag_rows)
    if tag_df.empty:
        tag_summary = pd.DataFrame(columns=["violation_tag", "frequency", "avg_record_severity", "hotspot_count"])
    else:
        tag_summary = (
            tag_df.groupby("violation_tag", dropna=False)
            .agg(
                frequency=("violation_tag", "size"),
                avg_record_severity=("severity_score", "mean"),
                hotspot_count=("hotspot_unit", "nunique"),
            )
            .reset_index()
            .sort_values(["frequency", "avg_record_severity"], ascending=[False, False])
        )

    validation_severity = (
        df.groupby("validation_status_clean", dropna=False)
        .agg(
            records=("severity_score", "size"),
            mean_severity=("severity_score", "mean"),
            severity_sum=("severity_score", "sum"),
        )
        .reset_index()
        .sort_values("records", ascending=False)
    )

    df[severity_record_cols].to_csv(OUT_DIR / "phase2_enriched_dataset.csv", index=False)
    approved[severity_record_cols].to_csv(OUT_DIR / "phase2_approved_severity_dataset.csv", index=False)
    hotspot_summary.to_csv(OUT_DIR / "phase2_hotspot_severity_scores.csv", index=False)
    vehicle_type_summary.to_csv(OUT_DIR / "phase2_vehicle_type_severity.csv", index=False)
    tag_summary.to_csv(OUT_DIR / "phase2_violation_tag_severity.csv", index=False)
    validation_severity.to_csv(OUT_DIR / "phase2_validation_severity.csv", index=False)

    print("Phase 2 complete")
    print("Input source:", source_path)
    print("Total rows:", len(df))
    print("Approved rows:", len(approved))
    print("Hotspots scored:", len(hotspot_summary))
    print("Mean severity (approved):", round(float(approved["severity_score"].mean()) if len(approved) else 0.0, 4))
    print("Top hotspot:", hotspot_summary.iloc[0]["hotspot_unit"] if len(hotspot_summary) else "N/A")
    print("Outputs saved to:", OUT_DIR.resolve())

    if len(hotspot_summary):
        plt.figure(figsize=(12, 6))
        top_hotspots = hotspot_summary.head(TOP_N).sort_values("severity_priority_score", ascending=True)
        plt.barh(top_hotspots["hotspot_unit"], top_hotspots["severity_priority_score"])
        plt.title("Top Hotspots by Severity Priority Score")
        plt.xlabel("Severity Priority Score")
        plt.ylabel("Hotspot")
        save_plot(OUT_DIR / "top_hotspots_by_severity_priority.png")

    if len(tag_summary):
        plt.figure(figsize=(12, 6))
        top_tags = tag_summary.head(TOP_N).sort_values("frequency", ascending=True)
        plt.barh(top_tags["violation_tag"], top_tags["frequency"])
        plt.title("Top Violation Tags (Approved Records)")
        plt.xlabel("Frequency")
        plt.ylabel("Violation Tag")
        save_plot(OUT_DIR / "top_violation_tags_phase2.png")

    if len(validation_severity):
        plt.figure(figsize=(10, 5))
        plt.bar(validation_severity["validation_status_clean"].astype(str), validation_severity["records"])
        plt.title("Records by Validation Status")
        plt.xlabel("Validation Status")
        plt.ylabel("Records")
        plt.xticks(rotation=30)
        save_plot(OUT_DIR / "validation_status_phase2.png")

if __name__ == "__main__":
    main()