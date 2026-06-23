import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import RobustScaler
except Exception as e:
    raise ImportError("scikit-learn is required for Layer A emerging hotspot detection.") from e

warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
INPUT_PATHS = [
    Path("content/phase5_outputs_2/phase5_enriched_records.csv"),
    Path("phase5_outputs_2/phase5_enriched_records.csv"),
    Path("/content/phase5_outputs_2/phase5_enriched_records.csv"),
    Path("content/phase4_outputs_2/phase4_merged_with_prior_scores.csv"),
    Path("phase4_outputs_2/phase4_merged_with_prior_scores.csv"),
    Path("/content/phase4_outputs_2/phase4_merged_with_prior_scores.csv"),
    Path("content/phase3_outputs_2/phase3_clustered_dataset.csv"),
    Path("phase3_outputs_2/phase3_clustered_dataset.csv"),
    Path("/content/phase3_outputs_2/phase3_clustered_dataset.csv"),
]

OUT_DIR = Path("content/layer_a_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9
TOP_N = 50
RECENT_WINDOW_WEEKS = 4
ALERT_FRACTIONS = (0.80, 0.60, 0.40)

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


def load_first_existing(paths):
    for p in paths:
        if p.exists():
            return pd.read_csv(p, low_memory=False), p
    return None, None


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


def derive_coords(df):
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


def normalize_tag(tag):
    return clean_text(tag).upper().replace("&", "AND").strip()


def minmax(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if len(valid) == 0 or valid.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    mn = valid.min()
    mx = valid.max()
    return (s.fillna(mn) - mn) / (mx - mn + EPS)


def robust_norm(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if len(valid) == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    lo = valid.quantile(0.05)
    hi = valid.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or abs(float(hi) - float(lo)) < EPS:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return ((s.fillna(lo) - lo) / (hi - lo + EPS)).clip(0.0, 1.0)


def safe_slope(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def dominant_label(series, default=""):
    s = pd.Series(series).dropna().astype(str).str.strip()
    s = s[s.ne("")]
    if s.empty:
        return default
    m = s.mode()
    if not m.empty:
        return m.iloc[0]
    return s.iloc[0]


def alert_level_from_score(score, q80, q60, q40):
    if score >= q80:
        return "Emerging-Critical"
    if score >= q60:
        return "Emerging-High"
    if score >= q40:
        return "Emerging-Watch"
    return "Stable"


def load_source_data():
    df, path = load_first_existing(INPUT_PATHS)
    if df is None:
        raise FileNotFoundError("No record-level source file found for Layer A.")
    return df, path


# =========================
# Layer A: Emerging Hotspot Detection
# =========================
def build_layer_a_emerging_hotspots():
    raw_df, source_path = load_source_data()
    df = raw_df.copy()

    cluster_col = standardize_cluster_col(df)
    vehicle_col = standardize_vehicle_col(df)
    vehicle_type_col = standardize_vehicle_type_col(df)

    if "validation_status" in df.columns:
        df["validation_status_clean"] = df["validation_status"].fillna("").astype(str).str.lower()
        df = df[df["validation_status_clean"].eq("approved")].copy()

    df = ensure_label_column(df)
    df = derive_coords(df)

    if "violation_tags" not in df.columns and "violation_type" in df.columns:
        df["violation_tags"] = df["violation_type"].apply(parse_listlike)
    elif "violation_tags" in df.columns:
        df["violation_tags"] = df["violation_tags"].apply(parse_listlike)
    else:
        df["violation_tags"] = [[] for _ in range(len(df))]

    df["created_datetime_ist"] = parse_datetime_ist(df)
    df = df.dropna(subset=["created_datetime_ist"]).copy()
    df["created_datetime_ist_naive"] = df["created_datetime_ist"].dt.tz_localize(None)
    df["service_date"] = df["created_datetime_ist_naive"].dt.date
    df["week_start"] = week_start_monday(df["created_datetime_ist"])
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df["is_peak_window"] = df["created_datetime_ist"].dt.hour.between(8, 12, inclusive="both").astype(int)

    if "severity_score" not in df.columns:
        severity_rules = {
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
            normalized = [normalize_tag(t) for t in tags]
            score = 1
            for sev in sorted(severity_rules.keys(), reverse=True):
                vocab = severity_rules[sev]
                if any(any(v == tag or v in tag for v in vocab) for tag in normalized):
                    score = sev
                    break
            return score

        df["severity_score"] = df["violation_tags"].apply(severity_from_tags)
    else:
        df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce").fillna(1.0)

    agg_kwargs = {
        "records_total": (cluster_col, "size"),
        "distinct_days": ("service_date", "nunique"),
        "severity_sum": ("severity_score", "sum"),
        "severity_mean": ("severity_score", "mean"),
        "peak_window_records": ("is_peak_window", "sum"),
        "unique_vehicles": (vehicle_col, "nunique") if vehicle_col and vehicle_col in df.columns else (cluster_col, "size"),
        "unique_vehicle_types": (vehicle_type_col, "nunique") if vehicle_type_col and vehicle_type_col in df.columns else (cluster_col, "size"),
        "lat": ("lat", "mean"),
        "lon": ("lon", "mean"),
        "cluster_label": ("cluster_label", dominant_label),
    }

    cluster = df.groupby(cluster_col).agg(**agg_kwargs).reset_index()
    cluster["distinct_days"] = pd.to_numeric(cluster["distinct_days"], errors="coerce").fillna(1).clip(lower=1)

    weekly = (
        df.groupby([cluster_col, "week_start"])
        .size()
        .reset_index(name="weekly_count")
        .sort_values([cluster_col, "week_start"])
    )

    trend_rows = []
    for cid, g in weekly.groupby(cluster_col):
        counts = g["weekly_count"].to_numpy(dtype=float)
        weeks = len(counts)
        if weeks == 0:
            first_half = second_half = 0.0
            recent_mean = prev_mean = 0.0
            slope = 0.0
            peak_ratio = 0.0
        else:
            mid = max(1, weeks // 2)
            first_half = float(counts[:mid].mean()) if len(counts[:mid]) else 0.0
            second_half = float(counts[mid:].mean()) if len(counts[mid:]) else first_half

            if weeks >= 2 * RECENT_WINDOW_WEEKS:
                recent_mean = float(counts[-RECENT_WINDOW_WEEKS:].mean())
                prev_mean = float(counts[-2 * RECENT_WINDOW_WEEKS:-RECENT_WINDOW_WEEKS].mean())
            elif weeks >= 2:
                recent_mean = second_half
                prev_mean = first_half
            else:
                recent_mean = prev_mean = float(counts.mean())

            slope = safe_slope(counts)
            peak_ratio = float(counts.max() / (counts.mean() + EPS)) if counts.mean() > 0 else float(counts.max())

        growth_pct = (second_half - first_half) / (first_half + EPS) if first_half > 0 else second_half
        recent_vs_prev = (recent_mean + 1.0) / (prev_mean + 1.0)
        resurgence_score = recent_vs_prev * (1.0 + max(0.0, growth_pct))
        growth_multiplier = 1.0 + max(0.0, growth_pct)

        trend_rows.append(
            {
                cluster_col: cid,
                "weekly_first_half_mean": first_half,
                "weekly_second_half_mean": second_half,
                "weekly_recent_mean": recent_mean,
                "weekly_previous_mean": prev_mean,
                "weekly_trend_slope": slope,
                "peak_week_ratio": peak_ratio,
                "growth_pct_change": float(np.clip(growth_pct, -0.8, 3.0)),
                "growth_multiplier": float(np.clip(growth_multiplier, 0.2, 4.0)),
                "recent_vs_prev_ratio": recent_vs_prev,
                "resurgence_score": resurgence_score,
            }
        )

    trend_df = pd.DataFrame(trend_rows)
    cluster = cluster.merge(trend_df, on=cluster_col, how="left")

    for c in [
        "weekly_first_half_mean",
        "weekly_second_half_mean",
        "weekly_recent_mean",
        "weekly_previous_mean",
        "weekly_trend_slope",
        "peak_week_ratio",
        "growth_pct_change",
        "growth_multiplier",
        "recent_vs_prev_ratio",
        "resurgence_score",
        "records_total",
        "distinct_days",
        "severity_sum",
        "severity_mean",
        "peak_window_records",
        "unique_vehicles",
        "unique_vehicle_types",
    ]:
        if c in cluster.columns:
            cluster[c] = pd.to_numeric(cluster[c], errors="coerce")

    cluster["recent_volume_norm"] = robust_norm(np.log1p(cluster["weekly_recent_mean"].fillna(0.0)))
    cluster["growth_norm"] = robust_norm(cluster["growth_pct_change"].fillna(0.0))
    cluster["trend_slope_norm"] = robust_norm(cluster["weekly_trend_slope"].fillna(0.0))
    cluster["resurgence_norm"] = robust_norm(np.log1p(cluster["resurgence_score"].fillna(0.0)))
    cluster["peak_ratio_norm"] = robust_norm(cluster["peak_week_ratio"].fillna(0.0))
    cluster["severity_norm"] = robust_norm(cluster["severity_sum"].fillna(0.0))
    cluster["records_norm"] = robust_norm(np.log1p(cluster["records_total"].fillna(0.0)))

    if len(cluster) >= 5:
        features = cluster[[
            "recent_volume_norm",
            "growth_norm",
            "trend_slope_norm",
            "resurgence_norm",
            "peak_ratio_norm",
            "severity_norm",
            "records_norm",
        ]].fillna(0.0)

        scaler = RobustScaler()
        X = scaler.fit_transform(features)

        iso = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=42,
            bootstrap=False,
        )
        iso.fit(X)
        anomaly_raw = -iso.decision_function(X)
        cluster["anomaly_score_raw"] = anomaly_raw
        cluster["anomaly_score"] = minmax(pd.Series(anomaly_raw, index=cluster.index))
    else:
        cluster["anomaly_score_raw"] = 0.0
        cluster["anomaly_score"] = 0.0

    cluster["emerging_hotspot_score"] = (
        0.32 * cluster["anomaly_score"].fillna(0.0)
        + 0.26 * cluster["growth_norm"].fillna(0.0)
        + 0.18 * cluster["resurgence_norm"].fillna(0.0)
        + 0.12 * cluster["trend_slope_norm"].fillna(0.0)
        + 0.12 * cluster["recent_volume_norm"].fillna(0.0)
    ) * 100.0

    cluster["emerging_hotspot_score"] = cluster["emerging_hotspot_score"].clip(0.0, 100.0)
    cluster = cluster.sort_values(
        ["emerging_hotspot_score", "growth_pct_change", "weekly_recent_mean", "records_total"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    cluster["emerging_rank"] = np.arange(1, len(cluster) + 1)

    q80 = cluster["emerging_hotspot_score"].quantile(ALERT_FRACTIONS[0]) if len(cluster) else 0.0
    q60 = cluster["emerging_hotspot_score"].quantile(ALERT_FRACTIONS[1]) if len(cluster) else 0.0
    q40 = cluster["emerging_hotspot_score"].quantile(ALERT_FRACTIONS[2]) if len(cluster) else 0.0

    cluster["alert_level"] = cluster["emerging_hotspot_score"].apply(
        lambda x: alert_level_from_score(x, q80, q60, q40)
    )
    cluster["is_emerging_hotspot"] = cluster["alert_level"].ne("Stable").astype(int)

    cluster["cluster_label"] = cluster["cluster_label"].fillna("").astype(str).str.strip()
    cluster.loc[cluster["cluster_label"].eq(""), "cluster_label"] = "CLUSTER::" + cluster[cluster_col].astype(str)

    cluster["priority_message"] = (
        cluster["cluster_label"]
        + " | "
        + cluster["alert_level"].astype(str)
        + " | Score="
        + cluster["emerging_hotspot_score"].round(2).astype(str)
    )

    weekly_out = weekly.merge(
        cluster[[cluster_col, "cluster_label", "emerging_hotspot_score", "alert_level", "emerging_rank"]],
        on=cluster_col,
        how="left",
    )

    top_alerts = cluster[cluster["is_emerging_hotspot"].eq(1)].copy()
    top_alerts = top_alerts.sort_values(
        ["alert_level", "emerging_hotspot_score", "growth_pct_change"],
        ascending=[True, False, False],
    ).head(TOP_N).reset_index(drop=True)

    cluster.to_csv(OUT_DIR / "layer_a_emerging_hotspots_full.csv", index=False)
    top_alerts.to_csv(OUT_DIR / "layer_a_emerging_hotspots_top.csv", index=False)
    weekly_out.to_csv(OUT_DIR / "layer_a_weekly_trends.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "input_source": str(source_path),
                "clusters_analyzed": len(cluster),
                "emerging_hotspots": int(cluster["is_emerging_hotspot"].sum()),
                "top_emerging_score": float(cluster["emerging_hotspot_score"].max()) if len(cluster) else 0.0,
                "mean_emerging_score": float(cluster["emerging_hotspot_score"].mean()) if len(cluster) else 0.0,
                "mean_growth_pct": float(cluster["growth_pct_change"].mean()) if len(cluster) else 0.0,
            }
        ]
    )
    summary.to_csv(OUT_DIR / "layer_a_summary.csv", index=False)

    print("Layer A complete")
    print("Input source:", source_path)
    print("Clusters analyzed:", len(cluster))
    print("Emerging hotspots:", int(cluster["is_emerging_hotspot"].sum()))
    print("Outputs saved to:", OUT_DIR.resolve())
    print("\nTop 10 emerging hotspots:")
    cols = [
        c for c in [
            "emerging_rank",
            cluster_col,
            "cluster_label",
            "alert_level",
            "emerging_hotspot_score",
            "anomaly_score",
            "growth_pct_change",
            "resurgence_score",
            "weekly_recent_mean",
            "records_total",
            "severity_sum",
        ] if c in top_alerts.columns
    ]
    if len(top_alerts):
        print(top_alerts[cols].head(10).to_string(index=False))

    return cluster, top_alerts, weekly_out, summary


# Run
layer_a_full, layer_a_top, layer_a_weekly, layer_a_summary = build_layer_a_emerging_hotspots()