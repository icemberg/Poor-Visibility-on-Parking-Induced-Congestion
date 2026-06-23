import ast
import math
import os
import re
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import requests
except Exception:
    requests = None

try:
    import networkx as nx
except Exception:
    nx = None

try:
    import osmnx as ox
except Exception:
    ox = None


# ============================================================
# Config
# ============================================================
PHASE4_MERGED_PATHS = [
    Path("content/phase4_outputs_2/phase4_merged_with_prior_scores.csv"),
    Path("/content/phase4_outputs_2/phase4_merged_with_prior_scores.csv"),
    Path("phase4_outputs_2/phase4_merged_with_prior_scores.csv"),
]

PHASE3_PATHS = [
    Path("content/phase3_outputs_2/phase3_clustered_dataset.csv"),
    Path("/content/phase3_outputs_2/phase3_clustered_dataset.csv"),
    Path("phase3_outputs_2/phase3_clustered_dataset.csv"),
]

PHASE4_MU_PATHS = [
    Path("content/phase4_outputs_2/phase4_cluster_mu_summary.csv"),
    Path("/content/phase4_outputs_2/phase4_cluster_mu_summary.csv"),
    Path("phase4_outputs_2/phase4_cluster_mu_summary.csv"),
]

OUT_DIR = Path("content/phase5_outputs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Persistent on-disk GraphML cache so the expensive OSMnx download is only
# done once across runs.  Delete the file manually to force a fresh download.
GRAPH_CACHE_PATH = OUT_DIR / "osmnx_graph_cache.graphml"

EPS = 1e-9
ALPHA_BPR = 0.56
BETA_BPR = 2.12
DEFAULT_MEAN_DWELL_MINUTES = 94.4

DEFAULT_LANES = 2
DEFAULT_WIDTH_PER_LANE_M = 3.5
DEFAULT_CARRIAGEWAY_WIDTH_M = DEFAULT_LANES * DEFAULT_WIDTH_PER_LANE_M
DEFAULT_LINK_LENGTH_M = 250.0
DEFAULT_ROAD_CLASS = "road"

PEAK_HOURS = {8, 9, 10, 11, 12}  # 8 AM to 1 PM
WEEK_SPLIT_HOURS = 24

ENABLE_OSMNX = os.environ.get("ENABLE_OSMNX_PHASE5", "1").strip() == "1"
print(ENABLE_OSMNX)
ENABLE_MAPPLS = os.environ.get("ENABLE_MAPPLS_PHASE5", "1").strip() == "1"
print(ENABLE_MAPPLS)
MAPPLS_ADDRESS_TOP_N = int(os.environ.get("MAPPLS_ADDRESS_TOP_N", "10"))
MAPPLS_ACCESS_TOKEN = os.environ.get("MAPPLS_ACCESS_TOKEN", "").strip()
MAPPLS_REGION = os.environ.get("MAPPLS_REGION", "IND").strip().upper()

ROAD_CLASS_SPEED_KMH = {
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

ROAD_CLASS_BASE_CAPACITY_PER_LANE = {
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


# ============================================================
# Helpers
# ============================================================
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


def valid_coords(lat, lon):
    try:
        return (
            pd.notna(lat)
            and pd.notna(lon)
            and -90 <= float(lat) <= 90
            and -180 <= float(lon) <= 180
        )
    except Exception:
        return False


def minmax(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    if len(s) == 0:
        return pd.Series(dtype=float)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min() + EPS)


def smooth_norm(s, floor=0.10):
    return floor + (1.0 - floor) * minmax(s)


def safe_ratio(a, b):
    return a / (b + EPS)


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


def dominant_label_from_series(series, exclude=None, default=""):
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


def standardize_cluster_col(df):
    for c in ["st_dbscan_cluster_id", "cluster_id", "dbscan_cluster_id"]:
        if c in df.columns:
            return c
    raise ValueError("No cluster id column found.")


def standardize_vehicle_col(df):
    for c in ["canonical_vehicle_number", "vehicle_number", "updated_vehicle_number"]:
        if c in df.columns:
            return c
    raise ValueError("No vehicle number column found.")


def standardize_vehicle_type_col(df):
    for c in ["canonical_vehicle_type", "vehicle_type", "updated_vehicle_type"]:
        if c in df.columns:
            return c
    return None


def load_first_existing(paths):
    for p in paths:
        if p.exists():
            return pd.read_csv(p, low_memory=False), p
    return None, None


def load_input():
    df, src = load_first_existing(PHASE4_MERGED_PATHS)
    if df is not None:
        return df, src
    df, src = load_first_existing(PHASE3_PATHS)
    if df is not None:
        return df, src
    raise FileNotFoundError("Could not find phase 4 merged output or phase 3 clustered dataset.")


def load_phase4_mu():
    df, _ = load_first_existing(PHASE4_MU_PATHS)
    return df


def ensure_hotspot_unit(df):
    df = df.copy()
    if "hotspot_unit" in df.columns:
        df["hotspot_unit"] = df["hotspot_unit"].fillna("").astype(str).str.strip()
        df.loc[df["hotspot_unit"].eq(""), "hotspot_unit"] = np.nan
        return df

    def make_hotspot_unit(row):
        junction = clean_text(row.get("junction_name", ""))
        if junction and junction.upper() != "NO JUNCTION":
            return f"JUNCTION::{junction}"
        station = clean_text(row.get("police_station", "UNKNOWN"))
        if not station:
            station = "UNKNOWN"
        return f"POLICE_STATION::{station}"

    if "junction_name" in df.columns or "police_station" in df.columns:
        df["hotspot_unit"] = df.apply(make_hotspot_unit, axis=1)
    else:
        df["hotspot_unit"] = "UNKNOWN"
    return df


def ensure_label_column(df):
    df = df.copy()
    if "cluster_label" in df.columns:
        df["cluster_label"] = df["cluster_label"].fillna("").astype(str).str.strip()
        df.loc[df["cluster_label"].eq(""), "cluster_label"] = np.nan
        return df
    if "hotspot_unit" in df.columns:
        df["cluster_label"] = df["hotspot_unit"].fillna("").astype(str).str.strip()
        df.loc[df["cluster_label"].eq(""), "cluster_label"] = np.nan
        return df
    if "dominant_junction_name" in df.columns:
        df["cluster_label"] = df["dominant_junction_name"].fillna("").astype(str).str.strip()
        df.loc[df["cluster_label"].eq(""), "cluster_label"] = np.nan
        return df
    if "st_dbscan_cluster_id" in df.columns:
        df["cluster_label"] = "CLUSTER::" + df["st_dbscan_cluster_id"].astype(str)
        return df
    df["cluster_label"] = "UNKNOWN"
    return df


def derive_coords(df):
    df = df.copy()
    if {"lat", "lon"}.issubset(df.columns):
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


def ensure_severity(df):
    df = df.copy()
    if "severity_score" in df.columns:
        df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce").fillna(1).clip(lower=1, upper=5).astype(int)
        return df

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

    if "violation_type" in df.columns:
        df["violation_tags"] = df["violation_type"].apply(parse_listlike)
        df["severity_score"] = df["violation_tags"].apply(severity_from_tags)
    else:
        df["severity_score"] = 1
        df["violation_tags"] = [[] for _ in range(len(df))]
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
        raise ValueError("Missing created_datetime / created_datetime_ist column.")

    ts = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    return ts.dt.tz_convert("Asia/Kolkata")


def week_start_monday(series):
    dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    week_start = dt.dt.normalize() - pd.to_timedelta(dt.dt.weekday, unit="D")
    return week_start.dt.tz_localize(None)


def parse_lane_count(lanes_value):
    if pd.isna(lanes_value):
        return None
    if isinstance(lanes_value, list):
        lanes_value = lanes_value[0] if lanes_value else None
    s = clean_text(lanes_value)
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s.replace(";", ",").replace("|", ","))
    if m:
        try:
            return max(1, int(round(float(m.group(1)))))
        except Exception:
            pass
    return None


def parse_width_m(width_value):
    if pd.isna(width_value):
        return None
    if isinstance(width_value, list):
        width_value = width_value[0] if width_value else None
    s = clean_text(width_value).lower().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def road_class_from_highway(highway_value):
    if pd.isna(highway_value):
        return DEFAULT_ROAD_CLASS

    vals = highway_value if isinstance(highway_value, list) else parse_listlike(highway_value)
    if not vals:
        raw = clean_text(highway_value).lower()
        vals = [raw] if raw else []

    normalized = []
    for v in vals:
        s = clean_text(v).lower().replace(" ", "_")
        s = {
            "motorway_link": "motorway",
            "trunk_link": "trunk",
            "primary_link": "primary",
            "secondary_link": "secondary",
            "tertiary_link": "tertiary",
        }.get(s, s)
        if s in ROAD_CLASS_SPEED_KMH:
            normalized.append(s)
        else:
            normalized.append("road")

    priority = [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "living_street", "service", "road"
    ]
    for p in priority:
        if p in normalized:
            return p
    return normalized[0] if normalized else DEFAULT_ROAD_CLASS


def road_speed_kmh(road_class):
    return ROAD_CLASS_SPEED_KMH.get(str(road_class).strip().lower(), 30.0)


def base_capacity_per_lane(road_class):
    return ROAD_CLASS_BASE_CAPACITY_PER_LANE.get(str(road_class).strip().lower(), 1600.0)


def vehicle_width_m(vehicle_type):
    vt = clean_text(vehicle_type).upper()
    return VEHICLE_WIDTH_M.get(vt, VEHICLE_WIDTH_M["UNKNOWN"])


def reverse_geocode_mappls(lat, lon, session):
    out = {"ok": False, "address": "", "error": "", "raw": None, "source": "reverse_geocode"}
    if not valid_coords(lat, lon):
        out["error"] = "invalid_coordinates"
        return out
    if not MAPPLS_ACCESS_TOKEN:
        out["error"] = "missing_MAPPLS_ACCESS_TOKEN"
        return out
    if requests is None:
        out["error"] = "requests_not_available"
        return out
    try:
        resp = session.get(
            "https://search.mappls.com/search/address/rev-geocode",
            params={
                "lat": float(lat),
                "lng": float(lon),
                "region": MAPPLS_REGION,
                "access_token": MAPPLS_ACCESS_TOKEN,
            },
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        out["raw"] = data
        results = data.get("results") or []
        if results and isinstance(results, list) and isinstance(results[0], dict):
            first = results[0]
            address = first.get("formatted_address") or first.get("address") or ""
            if not address:
                parts = []
                for k in [
                    "houseNumber", "houseName", "poi", "street", "subSubLocality",
                    "subLocality", "locality", "subDistrict", "district", "city",
                    "state", "pincode"
                ]:
                    v = clean_text(first.get(k, ""))
                    if v:
                        parts.append(v)
                address = ", ".join(dict.fromkeys(parts))
            out["address"] = clean_text(address)
            out["ok"] = bool(out["address"])
            if not out["address"]:
                out["error"] = "empty_address"
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def build_osmnx_graph(unique_coords):
    if not (ENABLE_OSMNX and ox is not None and nx is not None):
        return None, None

    if unique_coords is None or len(unique_coords) == 0:
        return None, None

    try:
        if hasattr(ox, "settings"):
            try:
                ox.settings.use_cache = True
            except Exception:
                pass
            try:
                ox.settings.log_console = False  # reduce noise; errors logged below
            except Exception:
                pass
    except Exception:
        pass

    # ---------- PATCH: persistent on-disk GraphML cache ----------
    # If a cached graph exists from a previous run, load it directly
    # instead of re-downloading from the OSMnx tile server.
    G = None
    if GRAPH_CACHE_PATH.exists():
        try:
            G = ox.load_graphml(str(GRAPH_CACHE_PATH))
            print(f"[OSMnx] Loaded graph from cache: {GRAPH_CACHE_PATH}")
        except Exception as e:
            print(f"[OSMnx] Cache load failed ({e}), re-downloading...")
            G = None

    if G is None:
        lat_min = float(unique_coords["lat_key"].min())
        lat_max = float(unique_coords["lat_key"].max())
        lon_min = float(unique_coords["lon_key"].min())
        lon_max = float(unique_coords["lon_key"].max())

        pad = 0.02
        north = lat_max + pad
        south = lat_min - pad
        east = lon_max + pad
        west = lon_min - pad

        try:
            try:
                G = ox.graph_from_bbox(
                    # pyrefly: ignore [unexpected-keyword]
                    north=north,
                    # pyrefly: ignore [unexpected-keyword]
                    south=south,
                    # pyrefly: ignore [unexpected-keyword]
                    east=east,
                    # pyrefly: ignore [unexpected-keyword]
                    west=west,
                    network_type="drive",
                    simplify=True,
                    retain_all=False,
                )
                print("[OSMnx] Downloaded graph via bbox (keyword args)")
            except TypeError:
                G = ox.graph_from_bbox(
                    north, south, east, west,
                    network_type="drive",
                    simplify=True,
                    retain_all=False,
                )
                print("[OSMnx] Downloaded graph via bbox (positional args)")
        except Exception as e:
            print(f"[OSMnx] bbox download failed ({e}), trying city-level fallback...")
            try:
                G = ox.graph_from_place(
                    "Bengaluru, Karnataka, India",
                    network_type="drive",
                    simplify=True,
                    retain_all=False,
                )
                print("[OSMnx] Downloaded graph via place name")
            except Exception as e2:
                print(f"[OSMnx] place download also failed ({e2}). Road enrichment will use defaults.")
                G = None

        # Save successfully downloaded graph so the next run is instant.
        if G is not None:
            try:
                ox.save_graphml(G, str(GRAPH_CACHE_PATH))
                print(f"[OSMnx] Graph saved to cache: {GRAPH_CACHE_PATH}")
            except Exception as e:
                print(f"[OSMnx] Could not save graph cache ({e}); continuing anyway.")
    # ---------- end cache patch ----------

    if G is None:
        return None, None

    try:
        UG = nx.Graph(G)
    except Exception:
        UG = G.to_undirected() if hasattr(G, "to_undirected") else None

    bc = {}
    if UG is not None and len(UG.nodes) > 0:
        n_nodes = len(UG.nodes)
        print(f"[OSMnx] Graph has {n_nodes:,} nodes")
        try:
            if n_nodes > 5000:
                k = min(200, max(50, n_nodes // 200))
                print(f"[OSMnx] Approximate betweenness centrality (k={k})...")
                bc = nx.betweenness_centrality(UG, k=k, normalized=True, weight="length", seed=42)
            else:
                print(f"[OSMnx] Exact betweenness centrality on {n_nodes:,} nodes...")
                bc = nx.betweenness_centrality(UG, normalized=True, weight="length")
        except Exception as e:
            print(f"[OSMnx] Betweenness centrality failed ({e}); bc will be 0.0 for all nodes.")
            bc = {}

    return G, bc


def road_context_for_points(unique_coords):
    """
    Returns a DataFrame with road_class, lane_count, width, degree, betweenness,
    using one graph download + cached nearest-node lookups.
    """
    defaults = []
    if unique_coords is None or len(unique_coords) == 0:
        return pd.DataFrame(defaults)

    G, bc = build_osmnx_graph(unique_coords)
    if G is None:
        for _, row in unique_coords.iterrows():
            defaults.append({
                "lat_key": row["lat_key"],
                "lon_key": row["lon_key"],
                "geometry_source": "fallback",
                "road_class": DEFAULT_ROAD_CLASS,
                "lane_count": DEFAULT_LANES,
                "carriageway_width_m": DEFAULT_CARRIAGEWAY_WIDTH_M,
                "link_length_m": DEFAULT_LINK_LENGTH_M,
                "junction_degree": 4,
                "betweenness_centrality": 0.0,
            })
        return pd.DataFrame(defaults)

    total = len(unique_coords)
    print(f"Road context enrichment for {total:,} unique coordinates...")
    for i, row in enumerate(unique_coords.itertuples(index=False), start=1):
        lat = float(row.lat_key)
        lon = float(row.lon_key)
        if i == 1 or i % 25 == 0 or i == total:
            print(f"  road context {i:,}/{total:,}")

        try:
            node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
            edge = ox.distance.nearest_edges(G, X=lon, Y=lat)
            u, v, k = edge
            edge_data = G.edges[u, v, k]

            road_class = road_class_from_highway(edge_data.get("highway", DEFAULT_ROAD_CLASS))
            lane_count = parse_lane_count(edge_data.get("lanes", None))
            if lane_count is None:
                lane_count = DEFAULT_LANES

            width_m = parse_width_m(edge_data.get("width", None))
            if width_m is None:
                width_m = lane_count * DEFAULT_WIDTH_PER_LANE_M

            link_length_m = safe_float(edge_data.get("length", DEFAULT_LINK_LENGTH_M), DEFAULT_LINK_LENGTH_M)

            if hasattr(G, "degree"):
                degree = int(G.degree[node]) if node in G else 4
            else:
                degree = 4

            centrality = float(bc.get(node, 0.0)) if bc else 0.0

            defaults.append({
                "lat_key": lat,
                "lon_key": lon,
                "geometry_source": "osmnx",
                "road_class": road_class,
                "lane_count": lane_count,
                "carriageway_width_m": width_m,
                "link_length_m": link_length_m,
                "junction_degree": degree,
                "betweenness_centrality": centrality,
            })
        except Exception as _exc:
            # PATCH: log the failure reason so we can diagnose whether
            # nearest_nodes/nearest_edges is the source (projection issue,
            # empty graph section, etc.) rather than silently falling back.
            if i <= 3 or i % 50 == 0:
                print(f"  [road_context fallback] coord ({lat:.4f},{lon:.4f}): {_exc}")
            defaults.append({
                "lat_key": lat,
                "lon_key": lon,
                "geometry_source": "fallback",
                "road_class": DEFAULT_ROAD_CLASS,
                "lane_count": DEFAULT_LANES,
                "carriageway_width_m": DEFAULT_CARRIAGEWAY_WIDTH_M,
                "link_length_m": DEFAULT_LINK_LENGTH_M,
                "junction_degree": 4,
                "betweenness_centrality": 0.0,
            })

    return pd.DataFrame(defaults)


def compute_dwell_gaps(records_df, cluster_col, vehicle_col, vehicle_type_col):
    """
    Stage 4: implicit μ estimator.
    Same vehicle, same cluster, same calendar day.
    """
    r = records_df.copy()
    r = r.dropna(subset=[cluster_col]).copy()
    r[cluster_col] = pd.to_numeric(r[cluster_col], errors="coerce")
    r = r[r[cluster_col].ne(-1)].copy()

    if vehicle_col not in r.columns:
        raise ValueError("vehicle number column is required for Stage 4/5.")

    r[vehicle_col] = r[vehicle_col].fillna("").astype(str).str.strip()
    r = r[r[vehicle_col].ne("")]

    # Parse datetime
    r["created_datetime_ist"] = parse_datetime_ist(r)
    r = r.dropna(subset=["created_datetime_ist"]).copy()
    r["created_datetime_ist_naive"] = r["created_datetime_ist"].dt.tz_localize(None)
    r["service_date"] = r["created_datetime_ist_naive"].dt.date
    r["cluster_week_start"] = week_start_monday(r["created_datetime_ist"])
    r["hour_ist"] = r["created_datetime_ist"].dt.hour
    r["is_peak_window"] = r["hour_ist"].isin(PEAK_HOURS).astype(int)

    sort_cols = [cluster_col, vehicle_col, "service_date", "created_datetime_ist_naive"]
    sort_cols = [c for c in sort_cols if c in r.columns]
    r = r.sort_values(sort_cols).copy()

    r["prev_created_datetime_ist_naive"] = (
        r.groupby([cluster_col, vehicle_col, "service_date"])["created_datetime_ist_naive"].shift(1)
    )
    r["gap_minutes"] = (
        r["created_datetime_ist_naive"] - r["prev_created_datetime_ist_naive"]
    ).dt.total_seconds() / 60.0

    gaps = r[r["gap_minutes"].notna() & (r["gap_minutes"] > 0)].copy()

    # per-cluster dwell summary
    if len(gaps):
        dwell_summary = (
            gaps.groupby(cluster_col)
            .agg(
                gap_count=("gap_minutes", "size"),
                mean_dwell_minutes=("gap_minutes", "mean"),
                median_dwell_minutes=("gap_minutes", "median"),
                std_dwell_minutes=("gap_minutes", lambda s: float(s.std(ddof=0)) if len(s) else 0.0),
            )
            .reset_index()
        )
        dwell_summary["mu_departures_per_hour"] = 60.0 / (dwell_summary["mean_dwell_minutes"] + EPS)
    else:
        dwell_summary = pd.DataFrame(columns=[
            cluster_col, "gap_count", "mean_dwell_minutes", "median_dwell_minutes",
            "std_dwell_minutes", "mu_departures_per_hour"
        ])

    # per-cluster, per-vehicle-type dwell summary
    dwell_by_type = pd.DataFrame()
    if vehicle_type_col and vehicle_type_col in gaps.columns and len(gaps):
        dwell_by_type = (
            gaps.groupby([cluster_col, vehicle_type_col])
            .agg(
                gap_count=("gap_minutes", "size"),
                mean_dwell_minutes=("gap_minutes", "mean"),
                median_dwell_minutes=("gap_minutes", "median"),
                std_dwell_minutes=("gap_minutes", lambda s: float(s.std(ddof=0)) if len(s) else 0.0),
            )
            .reset_index()
        )

    weekly_counts = (
        r.groupby([cluster_col, "cluster_week_start"])
        .size()
        .reset_index(name="weekly_count")
        .sort_values([cluster_col, "cluster_week_start"])
    )

    # growth acceleration per cluster
    growth_rows = []
    for cid, g in weekly_counts.groupby(cluster_col):
        counts = g["weekly_count"].to_numpy(dtype=float)
        if len(counts) < 2:
            first_half = float(counts.mean()) if len(counts) else 0.0
            second_half = first_half
            growth_pct = 0.0
        else:
            mid = max(1, len(counts) // 2)
            first_half = float(counts[:mid].mean()) if len(counts[:mid]) else 0.0
            second_half = float(counts[mid:].mean()) if len(counts[mid:]) else 0.0
            growth_pct = safe_ratio(second_half - first_half, first_half) if first_half > 0 else second_half

        growth_multiplier = max(0.5, 1.0 + max(0.0, growth_pct))
        growth_rows.append({
            cluster_col: cid,
            "growth_first_half_mean": first_half,
            "growth_second_half_mean": second_half,
            "growth_pct_change": growth_pct,
            "growth_multiplier": growth_multiplier,
        })

    growth_df = pd.DataFrame(growth_rows)
    return r, gaps, dwell_summary, dwell_by_type, weekly_counts, growth_df


def compute_cluster_table(records_df, cluster_col, vehicle_col, vehicle_type_col, dwell_summary, growth_df):
    """
    Cluster-level aggregate table built from approved records.
    """
    agg = {
        "records_total": (cluster_col, "size"),
        "peak_window_records": ("is_peak_window", "sum"),
        "distinct_days": ("service_date", "nunique"),
        "severity_sum": ("severity_score", "sum"),
        "severity_mean": ("severity_score", "mean"),
        "unique_vehicles": (vehicle_col, "nunique"),
        "centroid_lat": ("lat", "mean"),
        "centroid_lon": ("lon", "mean"),
        "dominant_police_station": ("police_station", lambda s: dominant_label_from_series(s, default="")) if "police_station" in records_df.columns else (cluster_col, lambda s: ""),
        "dominant_junction_name": ("junction_name", lambda s: dominant_label_from_series(s, exclude={"No Junction", "NO JUNCTION"}, default="")) if "junction_name" in records_df.columns else (cluster_col, lambda s: ""),
    }

    if vehicle_type_col and vehicle_type_col in records_df.columns:
        agg["unique_vehicle_types"] = (vehicle_type_col, "nunique")
        agg["dominant_vehicle_type"] = (vehicle_type_col, lambda s: dominant_label_from_series(s, default="UNKNOWN"))
    else:
        agg["unique_vehicle_types"] = (cluster_col, lambda s: 1)
        agg["dominant_vehicle_type"] = (cluster_col, lambda s: "UNKNOWN")

    cluster_table = records_df.groupby(cluster_col).agg(**agg).reset_index()

    # Peak-hour maximum count within peak window
    peak_hour_counts = (
        records_df[records_df["is_peak_window"].eq(1)]
        .groupby([cluster_col, "hour_ist"])
        .size()
        .reset_index(name="hourly_count")
    )
    records_peak_hour = (
        peak_hour_counts.groupby(cluster_col)["hourly_count"]
        .max()
        .rename("records_peak_hour")
        .reset_index()
    )
    cluster_table = cluster_table.merge(records_peak_hour, on=cluster_col, how="left")

    # Merge dwell summary
    if dwell_summary is not None and len(dwell_summary):
        cluster_table = cluster_table.merge(dwell_summary, on=cluster_col, how="left")

    # Merge growth
    if growth_df is not None and len(growth_df):
        cluster_table = cluster_table.merge(growth_df, on=cluster_col, how="left")

    # defaults / cleanup
    for c in ["records_total", "peak_window_records", "distinct_days", "records_peak_hour", "unique_vehicles", "unique_vehicle_types"]:
        if c not in cluster_table.columns:
            cluster_table[c] = 0
        cluster_table[c] = pd.to_numeric(cluster_table[c], errors="coerce").fillna(0)

    cluster_table["distinct_days"] = cluster_table["distinct_days"].clip(lower=1)

    if "mean_dwell_minutes" not in cluster_table.columns:
        cluster_table["mean_dwell_minutes"] = np.nan
    if "mu_departures_per_hour" not in cluster_table.columns:
        cluster_table["mu_departures_per_hour"] = np.nan
    if "growth_first_half_mean" not in cluster_table.columns:
        cluster_table["growth_first_half_mean"] = 0.0
    if "growth_second_half_mean" not in cluster_table.columns:
        cluster_table["growth_second_half_mean"] = 0.0
    if "growth_pct_change" not in cluster_table.columns:
        cluster_table["growth_pct_change"] = 0.0
    if "growth_multiplier" not in cluster_table.columns:
        cluster_table["growth_multiplier"] = 1.0

    # lambda normalization: divide peak-window record count by (distinct active days x peak-hour span)
    GLOBAL_OBSERVATION_DAYS = records_df["service_date"].nunique()

    cluster_table["lambda_hr_peak_window"] = (
        cluster_table["peak_window_records"] /
        (GLOBAL_OBSERVATION_DAYS * 5.0)
    )
    cluster_table["lambda_hr_peak_hour"] = cluster_table["records_peak_hour"] / cluster_table["distinct_days"].clip(lower=1)

    # mu fallback if gap summary missing
    cluster_table["mean_dwell_minutes"] = pd.to_numeric(cluster_table["mean_dwell_minutes"], errors="coerce")
    cluster_table["mu_departures_per_hour"] = pd.to_numeric(cluster_table["mu_departures_per_hour"], errors="coerce")
    cluster_table["mean_dwell_minutes"] = cluster_table["mean_dwell_minutes"].fillna(DEFAULT_MEAN_DWELL_MINUTES)
    cluster_table["mu_departures_per_hour"] = cluster_table["mu_departures_per_hour"].fillna(60.0 / cluster_table["mean_dwell_minutes"].clip(lower=EPS))

    # One cluster label field
    cluster_table["cluster_label"] = np.nan
    cluster_table.loc[cluster_table["dominant_junction_name"].astype(str).str.strip().ne(""), "cluster_label"] = cluster_table["dominant_junction_name"]
    cluster_table.loc[cluster_table["cluster_label"].isna() | cluster_table["cluster_label"].astype(str).str.strip().eq(""), "cluster_label"] = cluster_table["dominant_police_station"].apply(lambda x: f"POLICE_STATION::{x}" if clean_text(x) else np.nan)
    cluster_table.loc[cluster_table["cluster_label"].isna() | cluster_table["cluster_label"].astype(str).str.strip().eq(""), "cluster_label"] = "CLUSTER::" + cluster_table[cluster_col].astype(str)

    # Centroids / coords
    cluster_table["centroid_lat"] = pd.to_numeric(cluster_table["centroid_lat"], errors="coerce")
    cluster_table["centroid_lon"] = pd.to_numeric(cluster_table["centroid_lon"], errors="coerce")
    cluster_table["lat"] = cluster_table["centroid_lat"]
    cluster_table["lon"] = cluster_table["centroid_lon"]

    # vehicle width average
    if vehicle_type_col and vehicle_type_col in records_df.columns:
        records_df["vehicle_width_m"] = records_df[vehicle_type_col].map(vehicle_width_m).fillna(VEHICLE_WIDTH_M["UNKNOWN"])
        avg_width = records_df.groupby(cluster_col)["vehicle_width_m"].mean().rename("vehicle_width_avg_m").reset_index()
        cluster_table = cluster_table.merge(avg_width, on=cluster_col, how="left")
    else:
        cluster_table["vehicle_width_avg_m"] = VEHICLE_WIDTH_M["UNKNOWN"]

    return cluster_table


def compute_repeat_offenders(records_df, cluster_col, vehicle_col, vehicle_type_col):
    r = records_df.copy()
    r[vehicle_col] = r[vehicle_col].fillna("").astype(str).str.strip()
    r = r[r[vehicle_col].ne("")].copy()

    if len(r) == 0:
        return pd.DataFrame(columns=[
            "vehicle_number", "total_violations", "unique_clusters", "unique_hotspots",
            "first_seen", "last_seen", "dominant_vehicle_type", "chronic_offender_flag"
        ])

    if vehicle_type_col and vehicle_type_col in r.columns:
        dom_vtype = lambda s: dominant_label_from_series(s, default="UNKNOWN")
    else:
        dom_vtype = lambda s: "UNKNOWN"

    offender_counts = (
        r.groupby(vehicle_col)
        .agg(
            total_violations=(vehicle_col, "size"),
            unique_clusters=(cluster_col, "nunique"),
            unique_hotspots=("hotspot_unit", "nunique") if "hotspot_unit" in r.columns else (vehicle_col, "size"),
            first_seen=("created_datetime_ist_naive", "min"),
            last_seen=("created_datetime_ist_naive", "max"),
            dominant_vehicle_type=(vehicle_type_col, dom_vtype) if vehicle_type_col and vehicle_type_col in r.columns else (vehicle_col, lambda s: "UNKNOWN"),
        )
        .reset_index()
        .rename(columns={vehicle_col: "vehicle_number"})
        .sort_values(["total_violations", "unique_clusters"], ascending=[False, False])
    )

    offender_counts["chronic_offender_flag"] = (offender_counts["total_violations"] >= 5).astype(int)
    offender_counts = offender_counts[offender_counts["chronic_offender_flag"].eq(1)].copy()
    return offender_counts


def compute_vehicle_mix(records_df, cluster_col, vehicle_col, vehicle_type_col):
    if not vehicle_type_col or vehicle_type_col not in records_df.columns:
        return pd.DataFrame(columns=[cluster_col, "vehicle_type", "count", "share"])

    vehicle_mix = (
        records_df.groupby([cluster_col, vehicle_type_col])
        .agg(count=(vehicle_col, "size"))
        .reset_index()
        .rename(columns={vehicle_type_col: "vehicle_type"})
    )
    vehicle_mix["share"] = vehicle_mix["count"] / vehicle_mix.groupby(cluster_col)["count"].transform("sum")
    return vehicle_mix


def compute_dominant_violation_tag(records_df, cluster_col):
    if "violation_tags" not in records_df.columns:
        return pd.DataFrame(columns=[cluster_col, "dominant_violation_tag"])

    exploded = records_df[[cluster_col, "violation_tags"]].explode("violation_tags").copy()
    exploded["violation_tag"] = exploded["violation_tags"].map(normalize_tag)
    exploded = exploded[exploded["violation_tag"].fillna("").ne("")].copy()

    if len(exploded) == 0:
        return pd.DataFrame(columns=[cluster_col, "dominant_violation_tag"])

    dom_tag = (
        exploded.groupby(cluster_col)["violation_tag"]
        .apply(lambda s: dominant_label_from_series(s, default=""))
        .reset_index(name="dominant_violation_tag")
    )
    dom_tag["dominant_violation_tag"] = dom_tag["dominant_violation_tag"].fillna("")
    return dom_tag


def apply_osmnx_context(cluster_table):
    """
    Adds road_class, lane_count, carriageway_width_m, link_length_m,
    junction_degree, betweenness_centrality, geometry_source.

    FIX: real OSMnx-derived values must take priority over the hardcoded
    defaults. The previous version pre-populated cluster_table with default
    values for these columns BEFORE merging road_context_df (which carries
    the same column names), then called
        cluster_table[c].combine_first(cluster_table[ctx_col])
    combine_first only fills NaNs in the LEFT series -- since the left
    column (`c`) was already fully populated with non-null defaults, the
    freshly-computed `_ctx` values were always discarded, regardless of
    whether OSMnx actually returned real geometry. Every cluster silently
    kept defaults (road_class="road", carriageway_width_m=7.0,
    junction_degree=4, betweenness_centrality=0.0) even when ENABLE_OSMNX
    was on and the graph download succeeded.

    The fix: do not pre-populate defaults before the merge. Merge first,
    then fill only genuine NaNs (i.e. points whose coordinates had no
    match in road_context_df, or rows where OSMnx itself fell back) with
    defaults -- so the real OSMnx value wins whenever it exists.
    """
    cluster_table = cluster_table.copy()

    # Round coordinates for cache / de-dup -- done BEFORE any default columns exist.
    cluster_table["lat_key"] = pd.to_numeric(cluster_table["centroid_lat"], errors="coerce").round(4)
    cluster_table["lon_key"] = pd.to_numeric(cluster_table["centroid_lon"], errors="coerce").round(4)

    unique_coords = (
        cluster_table[["lat_key", "lon_key"]]
        .dropna(subset=["lat_key", "lon_key"])
        .drop_duplicates()
        .copy()
    )

    road_context_df = pd.DataFrame()
    if ENABLE_OSMNX and ox is not None and nx is not None and len(unique_coords) > 0:
        road_context_df = road_context_for_points(unique_coords)

    if len(road_context_df) > 0:
        # Merge FIRST. cluster_table has no road_class/lane_count/etc. columns
        # yet at this point, so there is no name collision and no suffixing --
        # the real OSMnx values land directly under their plain column names.
        cluster_table = cluster_table.merge(road_context_df, on=["lat_key", "lon_key"], how="left")

    # Apply defaults AFTER the merge, filling only genuine NaNs:
    # rows with no OSMnx match at all (OSMnx disabled, graph download failed,
    # or this particular coordinate had no match in road_context_df).
    for c, default in [
        ("geometry_source", "fallback"),
        ("road_class", DEFAULT_ROAD_CLASS),
        ("lane_count", DEFAULT_LANES),
        ("carriageway_width_m", DEFAULT_CARRIAGEWAY_WIDTH_M),
        ("link_length_m", DEFAULT_LINK_LENGTH_M),
        ("junction_degree", 4),
        ("betweenness_centrality", 0.0),
    ]:
        if c not in cluster_table.columns:
            cluster_table[c] = default
        else:
            cluster_table[c] = cluster_table[c].fillna(default)

    cluster_table["road_class"] = cluster_table["road_class"].astype(str)
    cluster_table["lane_count"] = pd.to_numeric(cluster_table["lane_count"], errors="coerce").fillna(DEFAULT_LANES).astype(int)
    cluster_table["carriageway_width_m"] = pd.to_numeric(cluster_table["carriageway_width_m"], errors="coerce").fillna(DEFAULT_CARRIAGEWAY_WIDTH_M)
    cluster_table["link_length_m"] = pd.to_numeric(cluster_table["link_length_m"], errors="coerce").fillna(DEFAULT_LINK_LENGTH_M)
    cluster_table["junction_degree"] = pd.to_numeric(cluster_table["junction_degree"], errors="coerce").fillna(4.0)
    cluster_table["betweenness_centrality"] = pd.to_numeric(cluster_table["betweenness_centrality"], errors="coerce").fillna(0.0)
    cluster_table["geometry_source"] = cluster_table["geometry_source"].astype(str)
    return cluster_table


def enrich_mappls_address(cluster_table):
    """
    Uses Mappls reverse geocode on the top N clusters by delay. Produces a
    human-readable address label only -- by design this does not feed back
    into road_class / lane_count / capacity, since Mappls reverse-geocode
    returns an address string, not road-geometry attributes. Road geometry
    enrichment is OSMnx's job (apply_osmnx_context); Mappls' contribution
    here is display/traceability, not a scoring input.
    """
    cluster_table = cluster_table.copy()
    cluster_table["mappls_address"] = ""

    if not ENABLE_MAPPLS or requests is None or not MAPPLS_ACCESS_TOKEN or len(cluster_table) == 0:
        return cluster_table

    session = requests.Session()
    top_idx = cluster_table.sort_values("delay_minutes_per_vehicle", ascending=False).head(MAPPLS_ADDRESS_TOP_N).index.tolist()

    for n, idx in enumerate(top_idx, start=1):
        lat = safe_float(cluster_table.at[idx, "centroid_lat"])
        lon = safe_float(cluster_table.at[idx, "centroid_lon"])
        if not valid_coords(lat, lon):
            continue
        if n == 1 or n % 3 == 0:
            print(f"Mappls reverse-geocode {n}/{len(top_idx)} ...")
        out = reverse_geocode_mappls(lat, lon, session)
        cluster_table.at[idx, "mappls_address"] = out["address"] if out["ok"] else ""
    return cluster_table


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.perf_counter()
    print("Loading input...")
    records_df, source = load_input()
    print(f"Input source: {source}")
    print(f"Rows loaded: {len(records_df):,}")

    cluster_col = standardize_cluster_col(records_df)
    vehicle_col = standardize_vehicle_col(records_df)
    vehicle_type_col = standardize_vehicle_type_col(records_df)

    records_df = records_df.copy()

    # Keep approved evidence if validation exists; otherwise keep all rows.
    if "validation_status_clean" in records_df.columns:
        records_df["validation_status_clean"] = records_df["validation_status_clean"].fillna("").astype(str).str.lower()
        records_df = records_df[records_df["validation_status_clean"].eq("approved")].copy()
    elif "validation_status" in records_df.columns:
        records_df["validation_status_clean"] = records_df["validation_status"].fillna("").astype(str).str.lower()
        records_df = records_df[records_df["validation_status_clean"].eq("approved")].copy()

    print(f"Approved rows: {len(records_df):,}")

    # Normalize / enrich record-level fields
    records_df = ensure_hotspot_unit(records_df)
    records_df = ensure_severity(records_df)
    records_df = derive_coords(records_df)

    if "latitude" in records_df.columns and "longitude" in records_df.columns:
        records_df["lat"] = pd.to_numeric(records_df["latitude"], errors="coerce")
        records_df["lon"] = pd.to_numeric(records_df["longitude"], errors="coerce")
    else:
        records_df = derive_coords(records_df)

    records_df["created_datetime_ist"] = parse_datetime_ist(records_df)
    records_df = records_df.dropna(subset=["created_datetime_ist"]).copy()
    records_df["created_datetime_ist_naive"] = records_df["created_datetime_ist"].dt.tz_localize(None)
    records_df["service_date"] = records_df["created_datetime_ist_naive"].dt.date
    records_df["cluster_week_start"] = week_start_monday(records_df["created_datetime_ist"])
    records_df["hour_ist"] = records_df["created_datetime_ist"].dt.hour
    records_df["is_peak_window"] = records_df["hour_ist"].isin(PEAK_HOURS).astype(int)

    if vehicle_type_col and vehicle_type_col in records_df.columns:
        records_df["vehicle_width_m"] = records_df[vehicle_type_col].map(vehicle_width_m).fillna(VEHICLE_WIDTH_M["UNKNOWN"])
    else:
        records_df["vehicle_width_m"] = VEHICLE_WIDTH_M["UNKNOWN"]

    records_df[vehicle_col] = records_df[vehicle_col].fillna("").astype(str).str.strip()
    records_df = records_df[records_df[vehicle_col].ne("")].copy()
    records_df = records_df[records_df[cluster_col].notna()].copy()
    records_df[cluster_col] = pd.to_numeric(records_df[cluster_col], errors="coerce")
    records_df = records_df[records_df[cluster_col].ne(-1)].copy()

    print(f"Clusters: {records_df[cluster_col].nunique():,}")

    # --------------------------------------------------------
    # Stage 4: implicit μ estimator
    # --------------------------------------------------------
    print("\nStage 4: estimating dwell time...")
    records_df, gaps_df, dwell_summary, dwell_by_type, weekly_counts, growth_df = compute_dwell_gaps(
        records_df, cluster_col, vehicle_col, vehicle_type_col
    )
    print(f"Valid dwell gaps: {len(gaps_df):,}")
    if len(dwell_summary):
        print(f"Overall mean dwell (minutes): {float(dwell_summary['mean_dwell_minutes'].mean()):.2f}")

    # --------------------------------------------------------
    # Cluster table
    # --------------------------------------------------------
    print("\nAggregating cluster metrics...")
    cluster_table = compute_cluster_table(records_df, cluster_col, vehicle_col, vehicle_type_col, dwell_summary, growth_df)

    # Merge optional stage 4 μ summary if available
    mu_summary = load_phase4_mu()
    if mu_summary is not None and len(mu_summary):
        if cluster_col not in mu_summary.columns and "st_dbscan_cluster_id" in mu_summary.columns:
            mu_summary = mu_summary.rename(columns={"st_dbscan_cluster_id": cluster_col})
        keep_cols = [cluster_col]
        for c in ["cluster_label", "gap_count", "mean_dwell_minutes", "median_dwell_minutes", "std_dwell_minutes", "mu_departures_per_hour"]:
            if c in mu_summary.columns:
                keep_cols.append(c)
        mu_summary = mu_summary[keep_cols].drop_duplicates(cluster_col)
        cluster_table = cluster_table.merge(mu_summary, on=cluster_col, how="left", suffixes=("", "_mu"))

        for c in ["mean_dwell_minutes", "mu_departures_per_hour"]:
            mu_c = f"{c}_mu"
            if mu_c in cluster_table.columns:
                if c in cluster_table.columns:
                    cluster_table[c] = pd.to_numeric(cluster_table[c], errors="coerce").combine_first(pd.to_numeric(cluster_table[mu_c], errors="coerce"))
                else:
                    cluster_table[c] = cluster_table[mu_c]
                cluster_table.drop(columns=[mu_c], inplace=True)

    # Final dwell cleanup
    cluster_table["mean_dwell_minutes"] = pd.to_numeric(cluster_table["mean_dwell_minutes"], errors="coerce").fillna(DEFAULT_MEAN_DWELL_MINUTES)
    cluster_table["mu_departures_per_hour"] = pd.to_numeric(cluster_table["mu_departures_per_hour"], errors="coerce").fillna(60.0 / cluster_table["mean_dwell_minutes"].clip(lower=EPS))

    # --------------------------------------------------------
    # Stage 5a: M/M/∞ queueing model
    # --------------------------------------------------------
    print("\nStage 5a: computing blocking vehicles...")
    cluster_table["blocking_vehicles_L"] = cluster_table["lambda_hr_peak_window"] / (cluster_table["mu_departures_per_hour"] + EPS)

    # --------------------------------------------------------
    # Vehicle mix / repeated offenders
    # --------------------------------------------------------
    print("Building vehicle mix and chronic-offender list...")
    vehicle_mix = compute_vehicle_mix(records_df, cluster_col, vehicle_col, vehicle_type_col)
    chronic_offenders = compute_repeat_offenders(records_df, cluster_col, vehicle_col, vehicle_type_col)

    repeat_summary = (
        records_df.groupby([cluster_col, vehicle_col])
        .size()
        .reset_index(name="vehicle_cluster_count")
    )
    repeat_summary["repeat_flag_2plus"] = (repeat_summary["vehicle_cluster_count"] >= 2).astype(int)
    repeat_summary["chronic_flag_5plus"] = (repeat_summary["vehicle_cluster_count"] >= 5).astype(int)

    repeat_agg = (
        repeat_summary.groupby(cluster_col)
        .agg(
            repeat_vehicle_count_2plus=("repeat_flag_2plus", "sum"),
            chronic_vehicle_count_5plus=("chronic_flag_5plus", "sum"),
        )
        .reset_index()
    )
    cluster_table = cluster_table.merge(repeat_agg, on=cluster_col, how="left")
    cluster_table["repeat_vehicle_count_2plus"] = cluster_table["repeat_vehicle_count_2plus"].fillna(0).astype(int)
    cluster_table["chronic_vehicle_count_5plus"] = cluster_table["chronic_vehicle_count_5plus"].fillna(0).astype(int)

    # Dominant violation tag
    dom_tag = compute_dominant_violation_tag(records_df, cluster_col)
    if len(dom_tag):
        cluster_table = cluster_table.merge(dom_tag, on=cluster_col, how="left")
    else:
        cluster_table["dominant_violation_tag"] = ""
    cluster_table["dominant_violation_tag"] = cluster_table["dominant_violation_tag"].fillna("")

    # --------------------------------------------------------
    # Stage 5b: Zhao capacity reduction model
    # --------------------------------------------------------
    print("\nStage 5b: road-network context + capacity loss...")
    cluster_table = apply_osmnx_context(cluster_table)

    cluster_table["base_saturation_per_lane_pcu_hr"] = cluster_table["road_class"].map(base_capacity_per_lane).fillna(1600.0)
    cluster_table["base_capacity_pcu_hr"] = cluster_table["base_saturation_per_lane_pcu_hr"] * pd.to_numeric(cluster_table["lane_count"], errors="coerce").fillna(DEFAULT_LANES)

    cluster_table["carriageway_width_m"] = pd.to_numeric(cluster_table["carriageway_width_m"], errors="coerce").fillna(DEFAULT_CARRIAGEWAY_WIDTH_M)
    cluster_table["vehicle_width_avg_m"] = pd.to_numeric(cluster_table["vehicle_width_avg_m"], errors="coerce").fillna(VEHICLE_WIDTH_M["UNKNOWN"])
    cluster_table["blocking_vehicles_L"] = pd.to_numeric(cluster_table["blocking_vehicles_L"], errors="coerce").fillna(0.0)

    cluster_table["blocked_width_fraction"] = (
        cluster_table["blocking_vehicles_L"] * cluster_table["vehicle_width_avg_m"]
    ) / (cluster_table["carriageway_width_m"] + EPS)

    cluster_table["blocked_width_fraction"] = cluster_table["blocked_width_fraction"].clip(lower=0.0, upper=0.95)
    cluster_table["reduced_capacity_pcu_hr"] = cluster_table["base_capacity_pcu_hr"] * (1.0 - cluster_table["blocked_width_fraction"])
    cluster_table["reduced_capacity_pcu_hr"] = cluster_table["reduced_capacity_pcu_hr"].clip(lower=cluster_table["base_capacity_pcu_hr"] * 0.10)
    cluster_table["capacity_loss_pct"] = 1.0 - safe_ratio(cluster_table["reduced_capacity_pcu_hr"], cluster_table["base_capacity_pcu_hr"])

    # --------------------------------------------------------
    # Stage 5c: Modified BPR delay
    # --------------------------------------------------------
    print("Stage 5c: BPR delay computation...")
    cluster_table["free_flow_speed_kmh"] = cluster_table["road_class"].apply(road_speed_kmh)
    cluster_table["free_flow_time_min"] = cluster_table["link_length_m"] * 60.0 / (cluster_table["free_flow_speed_kmh"] * 1000.0 + EPS)

    cluster_table["V_over_C0"] = cluster_table["lambda_hr_peak_window"] / (cluster_table["base_capacity_pcu_hr"] + EPS)
    cluster_table["V_over_Cp"] = cluster_table["lambda_hr_peak_window"] / (cluster_table["reduced_capacity_pcu_hr"] + EPS)

    cluster_table["delay_minutes_per_vehicle"] = (
        cluster_table["free_flow_time_min"] * ALPHA_BPR *
        (np.power(cluster_table["V_over_Cp"], BETA_BPR) - np.power(cluster_table["V_over_C0"], BETA_BPR))
    )
    cluster_table["delay_minutes_per_vehicle"] = (
        pd.to_numeric(cluster_table["delay_minutes_per_vehicle"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    # --------------------------------------------------------
    # Final cleanup and ranking
    # --------------------------------------------------------
    print("\nFinalizing stage 5 outputs...")
    cluster_table["growth_first_half_mean"] = pd.to_numeric(cluster_table["growth_first_half_mean"], errors="coerce").fillna(0.0)
    cluster_table["growth_second_half_mean"] = pd.to_numeric(cluster_table["growth_second_half_mean"], errors="coerce").fillna(0.0)
    cluster_table["growth_pct_change"] = pd.to_numeric(cluster_table["growth_pct_change"], errors="coerce").fillna(0.0)
    cluster_table["growth_multiplier"] = pd.to_numeric(cluster_table["growth_multiplier"], errors="coerce").fillna(1.0).clip(lower=0.1)

    cluster_table["criticality_factor"] = (
        1.0
        + 0.5 * minmax(pd.to_numeric(cluster_table["junction_degree"], errors="coerce").fillna(0))
        + 0.5 * minmax(pd.to_numeric(cluster_table["betweenness_centrality"], errors="coerce").fillna(0))
    )
    cluster_table["criticality_factor"] = pd.to_numeric(cluster_table["criticality_factor"], errors="coerce").fillna(1.0)

    # Stage 5 rank based on delay
    cluster_table = cluster_table.sort_values(
        ["delay_minutes_per_vehicle", "lambda_hr_peak_window", "blocking_vehicles_L"],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    cluster_table["stage5_rank"] = np.arange(1, len(cluster_table) + 1)

    # Cleanup labels and coordinates
    cluster_table["cluster_label"] = cluster_table["cluster_label"].fillna("").astype(str).str.strip()
    cluster_table.loc[cluster_table["cluster_label"].eq(""), "cluster_label"] = "CLUSTER::" + cluster_table[cluster_col].astype(str)

    cluster_table["centroid_lat"] = pd.to_numeric(cluster_table["centroid_lat"], errors="coerce")
    cluster_table["centroid_lon"] = pd.to_numeric(cluster_table["centroid_lon"], errors="coerce")
    cluster_table["lat"] = cluster_table["centroid_lat"]
    cluster_table["lon"] = cluster_table["centroid_lon"]

    # Optional Mappls reverse geocode labels for top hotspots only
    cluster_table = enrich_mappls_address(cluster_table)

    # Merge Mappls address back to records
    if "mappls_address" not in records_df.columns:
        records_df["mappls_address"] = ""
    records_df = records_df.merge(
        cluster_table[[cluster_col, "mappls_address"]],
        on=cluster_col,
        how="left",
        suffixes=("", "_stage5")
    )
    if "mappls_address_stage5" in records_df.columns:
        records_df["mappls_address"] = records_df["mappls_address"].combine_first(records_df["mappls_address_stage5"])
        records_df.drop(columns=["mappls_address_stage5"], inplace=True)

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------
    print("\nSaving outputs...")

    cluster_table.to_csv(OUT_DIR / "phase5_cluster_capacity_loss.csv", index=False)
    cluster_table.to_csv(OUT_DIR / "phase5_priority_table_full.csv", index=False)

    stage5_handoff_cols = [
        cluster_col, "cluster_label", "stage5_rank", "records_total", "distinct_days",
        "peak_window_records", "records_peak_hour", "lambda_hr_peak_window",
        "lambda_hr_peak_hour", "mean_dwell_minutes", "mu_departures_per_hour",
        "blocking_vehicles_L", "road_class", "lane_count", "carriageway_width_m",
        "base_saturation_per_lane_pcu_hr", "base_capacity_pcu_hr",
        "reduced_capacity_pcu_hr", "capacity_loss_pct", "free_flow_time_min",
        "delay_minutes_per_vehicle", "growth_first_half_mean",
        "growth_second_half_mean", "growth_pct_change", "growth_multiplier",
        "criticality_factor", "dominant_vehicle_type", "dominant_violation_tag",
        "geometry_source", "repeat_vehicle_count_2plus", "chronic_vehicle_count_5plus",
        "unique_vehicles", "unique_vehicle_types", "centroid_lat", "centroid_lon",
        "mappls_address"
    ]
    stage5_handoff_cols = [c for c in stage5_handoff_cols if c in cluster_table.columns]
    cluster_table[stage5_handoff_cols].to_csv(OUT_DIR / "phase5_stage5_handoff.csv", index=False)

    records_df.to_csv(OUT_DIR / "phase5_enriched_records.csv", index=False)
    vehicle_mix.to_csv(OUT_DIR / "phase5_vehicle_mix.csv", index=False)
    weekly_counts.to_csv(OUT_DIR / "phase5_weekly_cluster_counts.csv", index=False)
    growth_df.to_csv(OUT_DIR / "phase5_growth_summary.csv", index=False)
    chronic_offenders.to_csv(OUT_DIR / "phase5_chronic_offenders.csv", index=False)

    if len(dwell_summary):
        dwell_summary.to_csv(OUT_DIR / "phase5_cluster_mu_summary.csv", index=False)
    if len(dwell_by_type):
        dwell_by_type.to_csv(OUT_DIR / "phase5_cluster_mu_by_vehicle_type.csv", index=False)

    road_context_export_cols = [
        "geometry_source", "road_class", "lane_count", "carriageway_width_m",
        "link_length_m", "junction_degree", "betweenness_centrality"
    ]
    export_cols = [cluster_col, "cluster_label"] + [c for c in road_context_export_cols if c in cluster_table.columns]
    cluster_table[export_cols].to_csv(OUT_DIR / "phase5_road_context_cache.csv", index=False)

    elapsed = time.perf_counter() - t0
    print("Stage 5 complete")
    print("Clusters scored:", len(cluster_table))
    print("Chronic offenders:", len(chronic_offenders))
    print("Outputs saved to:", OUT_DIR.resolve())
    print(f"Elapsed: {elapsed:.1f} sec")

    print("\nTop 10 clusters by delay:")
    top_show_cols = [
        "stage5_rank", cluster_col, "cluster_label", "delay_minutes_per_vehicle",
        "lambda_hr_peak_window", "mu_departures_per_hour", "blocking_vehicles_L",
        "capacity_loss_pct", "criticality_factor", "growth_multiplier"
    ]
    top_show_cols = [c for c in top_show_cols if c in cluster_table.columns]
    if len(cluster_table):
        print(cluster_table[top_show_cols].head(10).to_string(index=False))

    print("\nRoad-context sanity check (should show real variance, not constants):")
    sanity_cols = ["road_class", "carriageway_width_m", "base_capacity_pcu_hr",
                   "junction_degree", "betweenness_centrality", "criticality_factor", "geometry_source"]
    sanity_cols = [c for c in sanity_cols if c in cluster_table.columns]
    if sanity_cols:
        print(cluster_table[sanity_cols].describe(include="all"))
        if "geometry_source" in cluster_table.columns:
            print("\ngeometry_source counts:")
            print(cluster_table["geometry_source"].value_counts())


if __name__ == "__main__":
    main()