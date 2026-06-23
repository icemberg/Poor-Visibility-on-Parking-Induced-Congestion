import ast
import io
import re
import zipfile
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="Bengaluru Parking Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; padding-bottom: 1rem; }
    .metric-card {
        background: linear-gradient(180deg, rgba(20,24,32,0.96), rgba(28,33,44,0.96));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px 18px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.14);
    }
    .small-label { font-size: 0.82rem; opacity: 0.75; margin-bottom: 0.2rem; }
    .big-value { font-size: 1.5rem; font-weight: 700; line-height: 1.15; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Config
# =========================
EPS = 1e-9
BENGALURU_CENTER = (12.9716, 77.5946)

PHASE6_DIRS = [
    Path("content/phase6_outputs_2"),
    Path("phase6_outputs_2"),
    Path("content/phase6_outputs_1"),
    Path("phase6_outputs_1"),
]

PHASE5_DIRS = [
    Path("content/phase5_outputs_2"),
    Path("phase5_outputs_2"),
    Path("content/phase5_outputs_1"),
    Path("phase5_outputs_1"),
]

RISK_COLORS = {
    "Critical": "red",
    "High": "orange",
    "Moderate": "blue",
    "Watch": "green",
    "Emerging-Critical": "purple",
    "Emerging-High": "darkred",
    "Emerging-Watch": "cadetblue",
}

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

VEHICLE_WIDTH_M = {
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

# =========================
# Helpers
# =========================
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def normalize_search_text(s: str) -> str:
    s = clean_text(s).lower()
    s = s.replace("::", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
    return clean_text(tag).upper().replace("&", "AND").strip()

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

def parse_created_datetime(df):
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

def minmax(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min() + EPS)

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

def color_for_band(band):
    return RISK_COLORS.get(str(band), "gray")

def standardize_cluster_col(df):
    for c in ["st_dbscan_cluster_id", "cluster_id", "dbscan_cluster_id"]:
        if c in df.columns:
            return c
    return None

def standardize_vehicle_col(df):
    if "canonical_vehicle_number" in df.columns:
        return "canonical_vehicle_number"
    if "vehicle_number" in df.columns:
        return "vehicle_number"
    return None

def standardize_vehicle_type_col(df):
    if "canonical_vehicle_type" in df.columns:
        return "canonical_vehicle_type"
    if "vehicle_type" in df.columns:
        return "vehicle_type"
    return None

def ensure_label_column(df):
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    if "cluster_label" in df.columns:
        df["cluster_label"] = df["cluster_label"].fillna("").astype(str).str.strip()
        df.loc[df["cluster_label"].eq(""), "cluster_label"] = np.nan
        return df
    if "hotspot_unit" in df.columns:
        df["cluster_label"] = df["hotspot_unit"].fillna("").astype(str).str.strip()
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

    elif {"cluster_lat", "cluster_lon"}.issubset(df.columns):
        df["lat"] = pd.to_numeric(df["cluster_lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["cluster_lon"], errors="coerce")

    elif {"mean_lat", "mean_lon"}.issubset(df.columns):
        df["lat"] = pd.to_numeric(df["mean_lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["mean_lon"], errors="coerce")

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

def risk_band_from_score(df):
    if df is None or len(df) == 0 or "ccs_score" not in df.columns:
        return df
    df = df.copy()
    q80 = df["ccs_score"].quantile(0.80)
    q60 = df["ccs_score"].quantile(0.60)
    q40 = df["ccs_score"].quantile(0.40)

    def band(x):
        if x >= q80:
            return "Critical"
        if x >= q60:
            return "High"
        if x >= q40:
            return "Moderate"
        return "Watch"

    df["risk_band"] = df["ccs_score"].apply(band)
    return df

def nearest_hotspot(df, lat, lon):
    if df is None or len(df) == 0 or pd.isna(lat) or pd.isna(lon):
        return None
    valid = df.dropna(subset=["lat", "lon"]).copy()
    if len(valid) == 0:
        return None
    d = np.sqrt((valid["lat"] - lat) ** 2 + (valid["lon"] - lon) ** 2)
    idx = d.idxmin()
    return valid.loc[idx]

def load_csv(uploaded_file, local_path):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, low_memory=False)
    if local_path and Path(local_path).exists():
        return pd.read_csv(local_path, low_memory=False)
    return None

def load_if_exists(path):
    return pd.read_csv(path, low_memory=False) if path.exists() else None

def fit_map_bounds(m, df):
    valid = df.dropna(subset=["lat", "lon"]).copy()
    if len(valid) == 0:
        return m
    if len(valid) == 1:
        row = valid.iloc[0]
        m.location = [float(row["lat"]), float(row["lon"])]
        m.zoom_start = 14
        return m
    bounds = [
        [float(valid["lat"].min()), float(valid["lon"].min())],
        [float(valid["lat"].max()), float(valid["lon"].max())],
    ]
    try:
        m.fit_bounds(bounds, padding=(30, 30))
    except Exception:
        pass
    return m

def add_hotspot_markers(m, df, selected_label=None, show_heatmap=False, heatmap_weight_col="ccs_score"):
    valid = df.dropna(subset=["lat", "lon"]).copy()
    if len(valid) == 0:
        return m

    if heatmap_weight_col not in valid.columns:
        valid[heatmap_weight_col] = 1.0

    if show_heatmap:
        heat_df = valid[["lat", "lon", heatmap_weight_col]].copy()
        heat_df[heatmap_weight_col] = smooth_norm(heat_df[heatmap_weight_col], floor=0.2)
        HeatMap(
            data=heat_df[["lat", "lon", heatmap_weight_col]].values.tolist(),
            name="Heatmap",
            radius=25,
            blur=18,
            min_opacity=0.18,
        ).add_to(m)

    for _, r in valid.iterrows():
        label = clean_text(r.get("cluster_label", r.get("hotspot_unit", "Hotspot")))
        band = clean_text(r.get("risk_band", "Watch"))
        ccs = float(r.get("ccs_score", 0.0))
        delay = float(r.get("delay_minutes_per_vehicle", 0.0))
        color = color_for_band(band)

        is_selected = selected_label is not None and str(label) == str(selected_label)
        if is_selected:
            color = "black"

        radius = 6 + 14 * float(minmax(pd.Series([ccs, 1.0])).iloc[0])
        if is_selected:
            radius += 5

        popup_html = f"""
        <div style="width:280px">
            <h4 style="margin-bottom:6px">{label}</h4>
            <b>CCS:</b> {safe_metric(ccs, 3)}<br>
            <b>Risk Band:</b> {band}<br>
            <b>Delay / vehicle:</b> {safe_metric(delay, 3)}<br>
            <b>Records:</b> {safe_metric(r.get('records_total', 0), 0)}<br>
            <b>Growth %:</b> {safe_metric(r.get('growth_pct_change', 0), 2)}<br>
            <b>Criticality:</b> {safe_metric(r.get('criticality_factor', 1), 2)}<br>
            <b>Action:</b> {clean_text(r.get('recommended_action', ''))}
        </div>
        """

        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=radius,
            color=color,
            weight=2 if is_selected else 1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=label,
            popup=folium.Popup(popup_html, max_width=340),
        ).add_to(m)

    return m

# =========================
# Precomputed bundle loaders
# =========================
@st.cache_data(show_spinner=False)
def load_precomputed_bundle():
    for phase6_dir in PHASE6_DIRS:
        if phase6_dir.exists():
            cluster = load_if_exists(phase6_dir / "phase6_cluster_ccs_full.csv")
            dispatch = load_if_exists(phase6_dir / "phase6_weekly_dispatch_priority_table.csv")
            alerts = load_if_exists(phase6_dir / "phase6_emerging_hotspot_alerts.csv")
            offenders = load_if_exists(phase6_dir / "phase6_chronic_offender_list.csv")
            reco = load_if_exists(phase6_dir / "phase6_enforcement_recommendations.csv")
            weekly = None
            for phase5_dir in PHASE5_DIRS:
                weekly = load_if_exists(phase5_dir / "phase5_weekly_cluster_counts.csv")
                if weekly is not None:
                    break

            if cluster is not None and dispatch is not None:
                bundle = {
                    "cluster": ensure_label_column(derive_coords(cluster)),
                    "dispatch": ensure_label_column(derive_coords(dispatch)),
                    "alerts": ensure_label_column(derive_coords(alerts)) if alerts is not None else None,
                    "offenders": offenders,
                    "recommendations": ensure_label_column(derive_coords(reco)) if reco is not None else None,
                    "weekly": weekly,
                    "raw": None,
                    "source": str(phase6_dir),
                }
                return bundle
    return None

def build_live_pipeline(raw_df):
    """
    Lightweight real-time pipeline for uploaded CSVs.
    Produces the same dashboard shape as precomputed outputs.
    """
    df = raw_df.copy()
    for c in ["validation_status", "police_station", "junction_name", "vehicle_number", "vehicle_type"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    if "validation_status" in df.columns:
        df["validation_status_clean"] = df["validation_status"].fillna("").astype(str).str.lower()
        approved = df[df["validation_status_clean"].eq("approved")].copy()
        if len(approved) == 0:
            approved = df.copy()
            approved["validation_status_clean"] = "approved"
    else:
        approved = df.copy()
        approved["validation_status_clean"] = "approved"

    approved["created_datetime_ist"] = parse_created_datetime(approved)
    approved = approved.dropna(subset=["created_datetime_ist"]).copy()

    if "violation_tags" not in approved.columns and "violation_type" in approved.columns:
        approved["violation_tags"] = approved["violation_type"].apply(parse_listlike)
    else:
        approved["violation_tags"] = approved.get("violation_tags", pd.Series([[]] * len(approved))).apply(
            lambda x: x if isinstance(x, list) else parse_listlike(x)
        )

    approved["severity_score"] = approved["violation_tags"].apply(severity_from_tags)

    if "hotspot_unit" not in approved.columns:
        approved["hotspot_unit"] = approved.apply(make_hotspot_unit, axis=1)

    approved["week_start"] = week_start_monday(approved["created_datetime_ist"])
    approved["week_start"] = pd.to_datetime(approved["week_start"], errors="coerce")

    agg_dict = {
        "records_total": ("hotspot_unit", "size"),
        "severity_sum": ("severity_score", "sum"),
        "severity_mean": ("severity_score", "mean"),
        "unique_vehicles": ("vehicle_number", "nunique") if "vehicle_number" in approved.columns else ("hotspot_unit", "size"),
        "lat": ("latitude", "mean") if "latitude" in approved.columns else ("hotspot_unit", "size"),
        "lon": ("longitude", "mean") if "longitude" in approved.columns else ("hotspot_unit", "size"),
        "dominant_vehicle_type": ("vehicle_type", lambda s: s.mode().iloc[0] if "vehicle_type" in approved.columns and not s.mode().empty else "UNKNOWN"),
    }
    cluster = approved.groupby("hotspot_unit").agg(**agg_dict).reset_index()

    if "latitude" not in approved.columns:
        cluster["lat"] = np.nan
        cluster["lon"] = np.nan

    weekly = (
        approved.groupby(["hotspot_unit", "week_start"])
        .size()
        .reset_index(name="weekly_count")
        .sort_values(["hotspot_unit", "week_start"])
    )

    growth_rows = []
    for hotspot, g in weekly.groupby("hotspot_unit"):
        counts = g["weekly_count"].to_numpy(dtype=float)
        if len(counts) < 2:
            first_half = float(counts.mean()) if len(counts) else 0.0
            second_half = first_half
            growth_pct = 0.0
        else:
            mid = max(1, len(counts) // 2)
            first_half = float(counts[:mid].mean()) if len(counts[:mid]) else 0.0
            second_half = float(counts[mid:].mean()) if len(counts[mid:]) else 0.0
            growth_pct = ((second_half - first_half) / (first_half + EPS)) if first_half > 0 else second_half

        growth_rows.append((hotspot, first_half, second_half, growth_pct))

    growth_df = pd.DataFrame(
        growth_rows,
        columns=["hotspot_unit", "growth_first_half_mean", "growth_second_half_mean", "growth_pct_change"],
    )

    cluster = cluster.merge(growth_df, on="hotspot_unit", how="left")
    cluster["growth_pct_change"] = cluster["growth_pct_change"].fillna(0.0)
    cluster["growth_multiplier"] = (1.0 + cluster["growth_pct_change"].clip(lower=0.0)).clip(lower=0.1)

    # simple live delay proxy
    cluster["delay_minutes_per_vehicle"] = (
        2.5 * minmax(cluster["severity_sum"]) +
        3.0 * minmax(cluster["records_total"]) +
        2.0 * minmax(cluster["growth_multiplier"])
    ) * 10.0

    cluster["criticality_factor"] = 1.0
    cluster["ccs_score"] = 100.0 * (
        0.35 * smooth_norm(cluster["delay_minutes_per_vehicle"]) +
        0.25 * smooth_norm(cluster["records_total"]) +
        0.20 * smooth_norm(cluster["severity_sum"]) +
        0.20 * smooth_norm(cluster["growth_multiplier"])
    )
    cluster["ccs_score"] = cluster["ccs_score"].fillna(0.0)
    cluster = risk_band_from_score(cluster)

    # labels/coords
    cluster["cluster_label"] = cluster["hotspot_unit"].astype(str)
    cluster = ensure_label_column(cluster)
    cluster = derive_coords(cluster)

    cluster["recommended_action"] = np.select(
        [
            cluster["risk_band"].eq("Critical"),
            cluster["risk_band"].eq("High"),
            cluster["risk_band"].eq("Moderate"),
        ],
        [
            "Immediate patrol deployment",
            "Targeted enforcement + towing readiness",
            "Monitor and schedule peak-window checks",
        ],
        default="Routine monitoring",
    )

    cluster["ccs_rank"] = cluster["ccs_score"].rank(method="first", ascending=False).astype(int)

    # offenders
    offenders = pd.DataFrame(columns=["vehicle_number", "total_violations", "unique_hotspots", "chronic_offender_flag"])
    if "vehicle_number" in approved.columns:
        offenders = approved.groupby("vehicle_number").agg(
            total_violations=("vehicle_number", "size"),
            unique_hotspots=("hotspot_unit", "nunique"),
            first_seen=("created_datetime_ist", "min"),
            last_seen=("created_datetime_ist", "max"),
        ).reset_index()
        offenders = offenders.sort_values("total_violations", ascending=False)
        offenders["chronic_offender_flag"] = (offenders["total_violations"] >= 5).astype(int)
        offenders = offenders[offenders["chronic_offender_flag"].eq(1)].copy()

    alerts = cluster.sort_values(["growth_pct_change", "ccs_score"], ascending=[False, False]).head(25).copy()
    alerts["alert_level"] = np.where(
        alerts["growth_pct_change"] >= alerts["growth_pct_change"].quantile(0.80),
        "Emerging-Critical",
        np.where(
            alerts["growth_pct_change"] >= alerts["growth_pct_change"].quantile(0.60),
            "Emerging-High",
            "Emerging-Watch",
        ),
    )

    recommendations = cluster.copy()
    recommendations["risk_band"] = cluster["risk_band"]

    return {
        "cluster": cluster.sort_values("ccs_score", ascending=False).reset_index(drop=True),
        "dispatch": cluster.sort_values("ccs_score", ascending=False).reset_index(drop=True),
        "alerts": alerts,
        "offenders": offenders,
        "recommendations": recommendations,
        "weekly": weekly,
        "raw": approved,
        "source": "live uploaded CSV",
    }

# =========================
# Sidebar
# =========================
st.sidebar.title("🚦 Parking Intelligence")
st.sidebar.caption("Hackathon prototype dashboard")

mode = st.sidebar.radio(
    "Data source",
    ["Use precomputed phase outputs", "Upload CSV and run analysis"],
    index=0,
)

uploaded_file = None
local_path = st.sidebar.text_input(
    "Local CSV path",
    value="jan to may police violation_anonymized791b166.csv",
)

if mode == "Upload CSV and run analysis":
    uploaded_file = st.sidebar.file_uploader("Upload violation CSV", type=["csv"])

use_cached = st.sidebar.checkbox("Prefer cached phase outputs", value=True)
show_heatmap = st.sidebar.checkbox("Show heatmap layer", value=True)
top_n_map = st.sidebar.slider("Map markers shown", 25, 300, 150, 25)
min_records_filter = st.sidebar.slider("Minimum records filter", 0, 5000, 0, 10)
search_text = st.sidebar.text_input("Search hotspot / junction / station")
selected_band = st.sidebar.selectbox(
    "Risk band",
    ["All", "Critical", "High", "Moderate", "Watch", "Emerging-Critical", "Emerging-High", "Emerging-Watch"],
)

run_clicked = st.sidebar.button("▶ Run analysis", type="primary", use_container_width=True)
reset_clicked = st.sidebar.button("Reset selected hotspot")

st.sidebar.markdown("---")
st.sidebar.caption("Click a hotspot on the map or choose one from search to focus it.")

if reset_clicked:
    st.session_state.pop("selected_label", None)
    st.session_state.pop("selected_row", None)
    st.rerun()

# =========================
# Load / run
# =========================
if run_clicked:
    with st.spinner("Running analysis..."):
        bundle = None
        source_label = None

        if mode == "Use precomputed phase outputs" and use_cached:
            bundle = load_precomputed_bundle()
            if bundle is not None:
                source_label = f"precomputed outputs ({bundle.get('source', 'phase folder')})"

        if bundle is None:
            raw_df = load_csv(uploaded_file, local_path)
            if raw_df is None:
                st.error("No dataset found. Upload a CSV or provide a valid local path.")
                st.stop()
            bundle = build_live_pipeline(raw_df)
            source_label = "live uploaded CSV"

        st.session_state["bundle"] = bundle
        st.session_state["source_label"] = source_label

elif "bundle" in st.session_state:
    bundle = st.session_state["bundle"]
else:
    bundle = None
    if mode == "Use precomputed phase outputs" and use_cached:
        bundle = load_precomputed_bundle()
    if bundle is None and mode == "Upload CSV and run analysis":
        st.info("Upload a CSV and click **Run analysis**.")
        st.stop()
    if bundle is None:
        st.info("No cached outputs found. Upload a CSV and click **Run analysis**.")
        st.stop()
    st.session_state["bundle"] = bundle
    st.session_state["source_label"] = f"precomputed outputs ({bundle.get('source', 'phase folder')})"

source_label = st.session_state.get("source_label", "session data")

# =========================
# Normalize dataframes
# =========================
cluster_df = bundle.get("cluster")
dispatch_df = bundle.get("dispatch")
alerts_df = bundle.get("alerts")
offenders_df = bundle.get("offenders")
reco_df = bundle.get("recommendations")
weekly_df = bundle.get("weekly")
raw_df = bundle.get("raw")

for name in ["cluster", "dispatch", "alerts", "offenders", "recommendations"]:
    df = bundle.get(name)
    if df is not None and len(df):
        df = ensure_label_column(df)
        df = derive_coords(df)
        if name == "cluster":
            cluster_df = df
        elif name == "dispatch":
            dispatch_df = df
        elif name == "alerts":
            alerts_df = df
        elif name == "offenders":
            offenders_df = df
        elif name == "recommendations":
            reco_df = df

if dispatch_df is None or len(dispatch_df) == 0:
    st.error("No hotspot table available.")
    st.stop()

dispatch_df = ensure_label_column(dispatch_df)
dispatch_df = derive_coords(dispatch_df)

if "ccs_score" not in dispatch_df.columns:
    dispatch_df["ccs_score"] = 0.0
if "risk_band" not in dispatch_df.columns:
    dispatch_df["risk_band"] = "Watch"
if "cluster_label" not in dispatch_df.columns:
    if "hotspot_unit" in dispatch_df.columns:
        dispatch_df["cluster_label"] = dispatch_df["hotspot_unit"].astype(str)
    else:
        dispatch_df["cluster_label"] = "Hotspot"
if "recommended_action" not in dispatch_df.columns:
    dispatch_df["recommended_action"] = "Target this zone"

dispatch_df["cluster_label"] = dispatch_df["cluster_label"].fillna("").astype(str)
if "hotspot_unit" in dispatch_df.columns:
    dispatch_df.loc[dispatch_df["cluster_label"].eq(""), "cluster_label"] = dispatch_df["hotspot_unit"].astype(str)
else:
    dispatch_df.loc[dispatch_df["cluster_label"].eq(""), "cluster_label"] = "Hotspot"

dispatch_df = risk_band_from_score(dispatch_df)
dispatch_df = derive_coords(dispatch_df)

# ---------------------------------------------------------------
# Merge coordinates from cluster_df into any df that lacks them.
# The Phase 6 dispatch / alerts / recommendations CSVs do NOT
# contain lat/lon columns — coordinates live only in cluster_df
# (as centroid_lat / centroid_lon).
# ---------------------------------------------------------------
def _merge_coords_from_cluster(target_df, source_df):
    """Merge lat/lon from source_df into target_df using cluster_label."""
    if target_df is None or len(target_df) == 0:
        return target_df
    if source_df is None or len(source_df) == 0:
        return target_df

    # Check if target already has valid coordinates
    target_df = derive_coords(target_df)
    if "lat" in target_df.columns and target_df["lat"].notna().any():
        return target_df

    src = source_df.copy()
    src = ensure_label_column(src)
    src = derive_coords(src)
    if "lat" not in src.columns or src["lat"].notna().sum() == 0:
        return target_df

    # Find a common key
    key = None
    for candidate in ["cluster_label", "hotspot_unit", "st_dbscan_cluster_id"]:
        if candidate in target_df.columns and candidate in src.columns:
            key = candidate
            break
    if key is None:
        return target_df

    coord_lookup = (
        src[[key, "lat", "lon"]]
        .dropna(subset=["lat", "lon"])
        .drop_duplicates(subset=[key])
    )

    # Drop existing NaN lat/lon before merge to avoid suffix collisions
    target_df = target_df.drop(columns=["lat", "lon"], errors="ignore")
    target_df = target_df.merge(coord_lookup, on=key, how="left")
    target_df["lat"] = pd.to_numeric(target_df["lat"], errors="coerce")
    target_df["lon"] = pd.to_numeric(target_df["lon"], errors="coerce")
    return target_df

dispatch_df = _merge_coords_from_cluster(dispatch_df, cluster_df)
alerts_df = _merge_coords_from_cluster(alerts_df, cluster_df)
reco_df = _merge_coords_from_cluster(reco_df, cluster_df)

# =========================
# Search / focus logic
# =========================
query_norm = normalize_search_text(search_text)

search_matches = pd.DataFrame()
if query_norm:
    fields = [c for c in ["cluster_label", "hotspot_unit", "road_class", "dominant_vehicle_type"] if c in dispatch_df.columns]
    if fields:
        mask = np.zeros(len(dispatch_df), dtype=bool)
        for c in fields:
            col_norm = dispatch_df[c].fillna("").astype(str).map(normalize_search_text)
            mask |= col_norm.str.contains(query_norm, regex=False).values
        search_matches = dispatch_df[mask].copy()

selected_label = st.session_state.get("selected_label")

if len(search_matches):
    match_labels = search_matches["cluster_label"].fillna("").astype(str).tolist()
    if selected_label not in match_labels:
        selected_label = match_labels[0]
    chosen_from_search = st.sidebar.selectbox(
        "Matching hotspots",
        match_labels,
        index=match_labels.index(selected_label) if selected_label in match_labels else 0,
    )
    selected_label = chosen_from_search
    sel_row = search_matches[search_matches["cluster_label"].astype(str) == str(selected_label)]
    if len(sel_row):
        st.session_state["selected_row"] = sel_row.iloc[0].to_dict()
        st.session_state["selected_label"] = selected_label
elif query_norm:
    st.sidebar.warning("No exact match found. Showing all hotspots on the map.")

# click-based selection will update this later
if "selected_row" not in st.session_state and len(dispatch_df):
    st.session_state["selected_row"] = dispatch_df.iloc[0].to_dict()
    st.session_state["selected_label"] = clean_text(dispatch_df.iloc[0].get("cluster_label", ""))

# =========================
# Filters
# =========================
base_filtered_df = dispatch_df.copy()
if selected_band != "All" and "risk_band" in base_filtered_df.columns:
    base_filtered_df = base_filtered_df[base_filtered_df["risk_band"] == selected_band]
if "records_total" in base_filtered_df.columns:
    base_filtered_df = base_filtered_df[base_filtered_df["records_total"] >= min_records_filter]

table_df = base_filtered_df.copy()
if query_norm:
    if len(search_matches):
        table_df = table_df[table_df["cluster_label"].astype(str).isin(search_matches["cluster_label"].astype(str))]
    else:
        table_df = table_df.iloc[0:0]

table_df = table_df.sort_values("ccs_score", ascending=False).reset_index(drop=True)
if len(table_df) > 0:
    table_df["ccs_rank"] = range(1, len(table_df) + 1)

# map scope uses the same base filters as the table
map_scope = base_filtered_df.copy().sort_values("ccs_score", ascending=False).reset_index(drop=True)

# Ensure the focused hotspot is always visible on the map
focus_label = st.session_state.get("selected_label")
if query_norm and len(search_matches):
    focus_label = selected_label

if focus_label and focus_label not in map_scope["cluster_label"].astype(str).tolist():
    focus_row = dispatch_df[dispatch_df["cluster_label"].astype(str) == str(focus_label)]
    if len(focus_row):
        map_scope = pd.concat([focus_row, map_scope], ignore_index=True)
        map_scope = map_scope.drop_duplicates(subset=["cluster_label"], keep="first").reset_index(drop=True)

# map top_n plus focus
if len(map_scope) > top_n_map:
    map_df = map_scope.head(top_n_map).copy()
    if focus_label and focus_label in map_scope["cluster_label"].astype(str).tolist():
        focus_row = map_scope[map_scope["cluster_label"].astype(str) == str(focus_label)]
        map_df = pd.concat([map_df, focus_row], ignore_index=True).drop_duplicates(subset=["cluster_label"], keep="first")
else:
    map_df = map_scope.copy()

# =========================
# Update selected row from map click if any
# =========================
selected_row_dict = st.session_state.get("selected_row", None)
selected_label = st.session_state.get("selected_label", None)

# =========================
# Header
# =========================
st.title("Bengaluru Parking-Induced Congestion Intelligence")
st.caption("Hotspot discovery, congestion ranking, emerging alerts, and enforcement prioritization.")

# =========================
# KPI cards
# =========================
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

kpi_source = table_df.copy()
top_ccs = float(kpi_source["ccs_score"].max()) if len(kpi_source) and "ccs_score" in kpi_source.columns else 0.0
top_delay = float(kpi_source["delay_minutes_per_vehicle"].max()) if len(kpi_source) and "delay_minutes_per_vehicle" in kpi_source.columns else 0.0
critical_count = int((kpi_source["risk_band"] == "Critical").sum()) if "risk_band" in kpi_source.columns else 0
emerging_count = len(alerts_df) if alerts_df is not None else 0
offender_count = len(offenders_df) if offenders_df is not None else 0
cluster_count = len(kpi_source)

metrics = [
    ("Clusters", f"{cluster_count:,}"),
    ("Top CCS", safe_metric(top_ccs, 3)),
    ("Top delay / vehicle", safe_metric(top_delay, 3)),
    ("Critical hotspots", f"{critical_count:,}"),
    ("Emerging alerts", f"{emerging_count:,}"),
    ("Chronic offenders", f"{offender_count:,}"),
]

for col, (title, value) in zip([kpi1, kpi2, kpi3, kpi4, kpi5, kpi6], metrics):
    with col:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="small-label">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big-value">{value}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.caption(f"Data source: {source_label}")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Executive view", "Interactive map", "Hotspot details", "Alerts", "Offenders & downloads"]
)

# =========================
# Tab 1 — Executive view
# =========================
with tab1:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Priority ranking")
        show_cols = [
            c for c in [
                "ccs_rank", "cluster_label", "risk_band", "ccs_score",
                "delay_minutes_per_vehicle", "lambda_hr_peak_window",
                "severity_sum", "growth_pct_change", "criticality_factor",
                "records_total", "road_class", "geometry_source"
            ] if c in table_df.columns
        ]
        if len(table_df):
            st.dataframe(table_df[show_cols].head(25), use_container_width=True, height=560)
        else:
            st.info("No hotspots match the current table filters.")

    with right:
        st.subheader("Operational summary")
        if len(table_df):
            band_counts = table_df["risk_band"].value_counts() if "risk_band" in table_df.columns else pd.Series(dtype=int)
            if len(band_counts):
                st.bar_chart(band_counts)

            st.markdown("**Top hotspot**")
            top_row = table_df.sort_values("ccs_score", ascending=False).iloc[0]
            st.write({
                "Hotspot": clean_text(top_row.get("cluster_label", "")),
                "CCS": round(float(top_row.get("ccs_score", 0.0)), 3),
                "Delay / vehicle": round(float(top_row.get("delay_minutes_per_vehicle", 0.0)), 3),
                "Growth %": round(float(top_row.get("growth_pct_change", 0.0)), 3),
                "Action": clean_text(top_row.get("recommended_action", "Target this zone")),
            })

            if "growth_pct_change" in table_df.columns:
                trend_df = table_df[["cluster_label", "growth_pct_change"]].head(12).sort_values("growth_pct_change", ascending=True)
                if len(trend_df):
                    st.subheader("Top growth trend")
                    st.bar_chart(trend_df.set_index("cluster_label"))
        else:
            st.info("No hotspots match the current table filters.")

# =========================
# Tab 2 — Map
# =========================
# -------------------------
# Helpers for map data
# -------------------------
def prepare_map_df(dispatch_df, cluster_df=None, raw_df=None):
    """
    Build a dataframe that always has lat/lon if any source contains coordinates.
    Priority:
    1) dispatch_df itself
    2) cluster_df merged in by hotspot/unit label
    3) raw_df aggregated by hotspot/unit label
    """
    df = dispatch_df.copy()
    df = ensure_label_column(df)

    # 1) Try direct coordinate derivation from the dispatch dataframe
    df = derive_coords(df)

    # 2) If still missing coordinates, merge from cluster_df
    if (("lat" not in df.columns) or ("lon" not in df.columns) or df[["lat", "lon"]].dropna().empty) and cluster_df is not None and len(cluster_df):
        cdf = ensure_label_column(cluster_df.copy())
        cdf = derive_coords(cdf)

        key_left = None
        key_right = None

        if "cluster_label" in df.columns and "cluster_label" in cdf.columns:
            key_left = "cluster_label"
            key_right = "cluster_label"
        elif "hotspot_unit" in df.columns and "hotspot_unit" in cdf.columns:
            key_left = "hotspot_unit"
            key_right = "hotspot_unit"

        if key_left and key_right:
            cdf_small = cdf[[key_right, "lat", "lon"]].dropna(subset=["lat", "lon"]).drop_duplicates(key_right)
            df = df.merge(cdf_small, left_on=key_left, right_on=key_right, how="left", suffixes=("", "_from_cluster"))
            if "lat_from_cluster" in df.columns:
                df["lat"] = df["lat"].combine_first(df["lat_from_cluster"])
                df["lon"] = df["lon"].combine_first(df["lon_from_cluster"])
                df.drop(columns=["lat_from_cluster", "lon_from_cluster"], inplace=True)
            if key_right != key_left and key_right in df.columns:
                df.drop(columns=[key_right], inplace=True)

    # 3) If still missing, derive from raw data aggregated by hotspot label
    if (("lat" not in df.columns) or ("lon" not in df.columns) or df[["lat", "lon"]].dropna().empty) and raw_df is not None and len(raw_df):
        rdf = raw_df.copy()
        rdf = ensure_label_column(rdf)
        rdf = derive_coords(rdf)

        if "cluster_label" in df.columns and "cluster_label" in rdf.columns:
            agg = (
                rdf.dropna(subset=["lat", "lon"])
                   .groupby("cluster_label", as_index=False)
                   .agg(lat=("lat", "mean"), lon=("lon", "mean"))
            )
            df = df.merge(agg, on="cluster_label", how="left", suffixes=("", "_raw"))
            if "lat_raw" in df.columns:
                df["lat"] = df["lat"].combine_first(df["lat_raw"])
                df["lon"] = df["lon"].combine_first(df["lon_raw"])
                df.drop(columns=["lat_raw", "lon_raw"], inplace=True)

    # Final cleanup
    df["lat"] = pd.to_numeric(df.get("lat", np.nan), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon", np.nan), errors="coerce")
    return df


def build_map(df, selected_label=None, show_heatmap=True, top_n=150):
    m = folium.Map(location=BENGALURU_CENTER, zoom_start=11, tiles="cartodbpositron")

    valid = df.dropna(subset=["lat", "lon"]).copy()
    if len(valid) == 0:
        return m

    if "ccs_score" not in valid.columns:
        valid["ccs_score"] = 0.0
    if "risk_band" not in valid.columns:
        valid["risk_band"] = "Watch"
    if "cluster_label" not in valid.columns:
        valid["cluster_label"] = valid.get("hotspot_unit", "Hotspot").astype(str)

    valid = valid.sort_values("ccs_score", ascending=False).head(top_n)
    if show_heatmap and len(valid) > 1:
        heat_rows = valid[["lat", "lon", "ccs_score"]].copy()
        heat_rows["ccs_score"] = smooth_norm(heat_rows["ccs_score"], floor=0.2)
        HeatMap(
            heat_rows[["lat", "lon", "ccs_score"]].values.tolist(),
            name="Heatmap",
            radius=25,
            blur=18,
            min_opacity=0.18,
        ).add_to(m)
    
    ccs_norm = minmax(valid["ccs_score"])
    valid["radius"] = 6 + 18*ccs_norm

    for _, r in valid.iterrows():
        label = clean_text(r.get("cluster_label", r.get("hotspot_unit", "Hotspot")))
        band = clean_text(r.get("risk_band", "Watch"))
        ccs = float(r.get("ccs_score", 0.0))
        delay = float(r.get("delay_minutes_per_vehicle", 0.0))
        color = color_for_band(band)

        is_selected = selected_label is not None and str(label) == str(selected_label)
        if is_selected:
            color = "black"

        popup_html = f"""
        <div style="width:280px">
            <h4 style="margin-bottom:6px">{label}</h4>
            <b>CCS:</b> {safe_metric(ccs, 3)}<br>
            <b>Risk Band:</b> {band}<br>
            <b>Delay / vehicle:</b> {safe_metric(delay, 3)}<br>
            <b>Records:</b> {safe_metric(r.get('records_total', 0), 0)}<br>
            <b>Growth %:</b> {safe_metric(r.get('growth_pct_change', 0), 2)}<br>
            <b>Criticality:</b> {safe_metric(r.get('criticality_factor', 1), 2)}<br>
            <b>Action:</b> {clean_text(r.get('recommended_action', ''))}
        </div>
        """

        radius = float(r["radius"])
        if is_selected:
            radius += 5

        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=radius,
            color=color,
            weight=2 if is_selected else 1,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=label,
            popup=folium.Popup(popup_html, max_width=340),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

with tab2:
    st.subheader("Hotspots on the map")

    # Build a robust map dataframe from the pre-filtered map_df
    map_df_prepared = prepare_map_df(
        map_df if map_df is not None else pd.DataFrame(),
        cluster_df=cluster_df,
        raw_df=raw_df
    )
    map_df_prepared = map_df_prepared.dropna(subset=["lat", "lon"]).copy()

    # Plot
    if len(map_df_prepared) > 0:
        map_obj = build_map(
            map_df_prepared,
            selected_label=st.session_state.get("selected_label"),
            show_heatmap=show_heatmap,
            top_n=top_n_map,
        )
        map_obj = fit_map_bounds(map_obj, map_df_prepared)

        map_state = st_folium(map_obj, use_container_width=True, height=720, key="hotspot_map")

        clicked = map_state.get("last_object_clicked") if isinstance(map_state, dict) else None
        if clicked and "lat" in clicked and "lng" in clicked:
            selected = nearest_hotspot(map_df_prepared, clicked["lat"], clicked["lng"])
            if selected is not None:
                st.session_state["selected_row"] = selected.to_dict()
                st.session_state["selected_label"] = clean_text(selected.get("cluster_label", ""))
                st.success(f"Selected hotspot: {clean_text(selected.get('cluster_label', ''))}")
    else:
        st.info("No hotspots match the current map filters.")

# =========================
# Tab 3 — Hotspot details
# =========================
with tab3:
    st.subheader("Hotspot detail explorer")

    detail_base = table_df if len(table_df) else dispatch_df
    if len(detail_base) == 0:
        st.info("No hotspots match the current filters.")
    else:
        labels = detail_base["cluster_label"].fillna("").astype(str).tolist()
        default_label = st.session_state.get("selected_label")
        if default_label not in labels:
            default_label = labels[0]

        chosen_label = st.selectbox("Choose a hotspot", labels, index=labels.index(default_label))
        chosen = detail_base[detail_base["cluster_label"].astype(str) == str(chosen_label)].iloc[0]

        a, b, c, d = st.columns(4)
        with a:
            st.metric("CCS rank", f"{int(chosen.get('ccs_rank', 0))}")
        with b:
            st.metric("CCS score", safe_metric(chosen.get("ccs_score", 0.0), 3))
        with c:
            st.metric("Delay / vehicle", safe_metric(chosen.get("delay_minutes_per_vehicle", 0.0), 3))
        with d:
            st.metric("Risk band", clean_text(chosen.get("risk_band", "Watch")))

        info1, info2 = st.columns([1.0, 1.0])

        with info1:
            st.markdown("**Operational attributes**")
            st.write({
                "Hotspot": clean_text(chosen.get("cluster_label", "")),
                "Records": int(chosen.get("records_total", 0)) if not pd.isna(chosen.get("records_total", 0)) else 0,
                "Severity sum": round(float(chosen.get("severity_sum", 0.0)), 3),
                "Growth %": round(float(chosen.get("growth_pct_change", 0.0)), 3),
                "Growth multiplier": round(float(chosen.get("growth_multiplier", 1.0)), 3),
                "Criticality factor": round(float(chosen.get("criticality_factor", 1.0)), 3),
                "Road class": clean_text(chosen.get("road_class", "road")),
                "Geometry source": clean_text(chosen.get("geometry_source", "fallback")),
                "Recommended action": clean_text(chosen.get("recommended_action", "Target this zone")),
            })

        with info2:
            st.markdown("**Geography**")
            st.write({
                "Latitude": round(float(chosen.get("lat", np.nan)), 6) if not pd.isna(chosen.get("lat", np.nan)) else None,
                "Longitude": round(float(chosen.get("lon", np.nan)), 6) if not pd.isna(chosen.get("lon", np.nan)) else None,
                "Lane count": int(chosen.get("lane_count", 2)) if not pd.isna(chosen.get("lane_count", 2)) else 2,
                "Carriageway width (m)": round(float(chosen.get("carriageway_width_m", 7.0)), 2) if not pd.isna(chosen.get("carriageway_width_m", 7.0)) else 7.0,
                "Link length (m)": round(float(chosen.get("link_length_m", 250.0)), 2) if not pd.isna(chosen.get("link_length_m", 250.0)) else 250.0,
                "Blocking vehicles L": round(float(chosen.get("blocking_vehicles_L", 0.0)), 3) if not pd.isna(chosen.get("blocking_vehicles_L", 0.0)) else 0.0,
                "Capacity loss %": round(float(chosen.get("capacity_loss_pct", 0.0)) * 100.0, 2) if not pd.isna(chosen.get("capacity_loss_pct", 0.0)) else 0.0,
            })

        st.markdown("**Weekly trend**")
        if weekly_df is not None and len(weekly_df):
            wk = weekly_df.copy()
            if "hotspot_unit" in wk.columns:
                series = wk[wk["hotspot_unit"].astype(str) == str(chosen.get("hotspot_unit", chosen_label))].copy()
            elif "st_dbscan_cluster_id" in wk.columns and "st_dbscan_cluster_id" in chosen.index:
                series = wk[wk["st_dbscan_cluster_id"].astype(str) == str(chosen.get("st_dbscan_cluster_id"))].copy()
            else:
                series = pd.DataFrame()

            if len(series):
                series = series.sort_values(series.columns[1])
                if "weekly_count" in series.columns:
                    st.line_chart(series.set_index(series.columns[1])["weekly_count"])
                else:
                    st.write(series.head(20))
            else:
                st.info("No weekly trend found for this hotspot.")
        else:
            st.info("Weekly trend file not available.")

# =========================
# Tab 4 — Alerts
# =========================
with tab4:
    st.subheader("Emerging hotspot alerts")

    if alerts_df is not None and len(alerts_df):
        alerts_show = ensure_label_column(derive_coords(alerts_df.copy()))
        show_cols = [
            c for c in [
                "alert_level", "cluster_label", "growth_pct_change",
                "growth_multiplier", "ccs_score", "records_total",
                "delay_minutes_per_vehicle", "risk_band"
            ] if c in alerts_show.columns
        ]
        st.dataframe(alerts_show[show_cols].head(25), use_container_width=True, height=420)

        st.markdown("**Growth leaders**")
        growth_show = alerts_show[["cluster_label", "growth_pct_change"]].head(15).sort_values("growth_pct_change", ascending=True)
        if len(growth_show):
            st.bar_chart(growth_show.set_index("cluster_label"))
    else:
        st.info("No alerts file available.")

    st.markdown("---")
    st.subheader("Recommended enforcement actions")
    if reco_df is not None and len(reco_df):
        reco_show = ensure_label_column(derive_coords(reco_df.copy()))
        reco_cols = [
            c for c in [
                "cluster_label", "risk_band", "recommended_action",
                "delay_minutes_per_vehicle", "lambda_hr_peak_window",
                "severity_sum", "growth_pct_change", "criticality_factor"
            ] if c in reco_show.columns
        ]
        st.dataframe(reco_show[reco_cols].head(20), use_container_width=True, height=360)
    else:
        st.info("No recommendation table available.")

# =========================
# Tab 5 — Offenders and downloads
# =========================
with tab5:
    left, right = st.columns([1.0, 1.0])

    with left:
        st.subheader("Chronic offenders")
        if offenders_df is not None and len(offenders_df):
            st.dataframe(
                offenders_df[[
                    c for c in [
                        "vehicle_number", "total_violations",
                        "unique_hotspots", "dominant_vehicle_type",
                        "first_seen", "last_seen"
                    ] if c in offenders_df.columns
                ]].head(50),
                use_container_width=True,
                height=520,
            )
        else:
            st.info("No chronic-offender table available.")

    with right:
        st.subheader("Downloads")
        if len(dispatch_df):
            st.download_button(
                "Download dispatch priority table",
                data=dispatch_df.to_csv(index=False).encode("utf-8"),
                file_name="weekly_dispatch_priority_table.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if len(table_df):
            st.download_button(
                "Download filtered hotspots",
                data=table_df.to_csv(index=False).encode("utf-8"),
                file_name="filtered_hotspots.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if alerts_df is not None and len(alerts_df):
            st.download_button(
                "Download emerging alerts",
                data=alerts_df.to_csv(index=False).encode("utf-8"),
                file_name="emerging_hotspot_alerts.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if offenders_df is not None and len(offenders_df):
            st.download_button(
                "Download chronic offenders",
                data=offenders_df.to_csv(index=False).encode("utf-8"),
                file_name="chronic_offender_list.csv",
                mime="text/csv",
                use_container_width=True,
            )

        files = {}
        if len(dispatch_df):
            files["weekly_dispatch_priority_table.csv"] = dispatch_df
        if len(table_df):
            files["filtered_hotspots.csv"] = table_df
        if alerts_df is not None and len(alerts_df):
            files["emerging_hotspot_alerts.csv"] = alerts_df
        if offenders_df is not None and len(offenders_df):
            files["chronic_offender_list.csv"] = offenders_df
        
        if files:
            bio = io.BytesIO()
            with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
                for filename, dfx in files.items():
                    zf.writestr(filename, dfx.to_csv(index=False))
            bio.seek(0)
            st.download_button(
                "Download all outputs as ZIP",
                data=bio,
                file_name="dashboard_outputs.zip",
                mime="application/zip",
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("Demo script")
    st.write(
        "Open the map, search a hotspot, and the app will highlight it instead of filtering the map to nothing. "
        "Use the detail tab to explain CCS, delay, growth, and recommended action."
    )
    st.write(
        "The dashboard supports both precomputed outputs and live analysis from an uploaded CSV."
    )