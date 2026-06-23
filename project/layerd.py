import ast
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import osmnx as ox

warnings.filterwarnings("ignore")

# ============================================================
# Layer D — Network Spillover / Propagation Analysis
# Optimized with:
#   - OSMnx GraphML caching (download once, reuse later)
#   - nearest_nodes() for hotspot-to-road-node assignment
#   - cached betweenness centrality
#   - pairwise hotspot graph built vectorially
# ============================================================

# -------------------------
# Config
# -------------------------
BASE_DIRS = [
    Path("content/phase5_outputs_2"),
    Path("phase5_outputs_2"),
    Path("/content/phase5_outputs_2"),
]

# If Layer B/C outputs exist, we can merge them; if not, code still runs.
LAYER_B_DIRS = [
    Path("content/layer_b_outputs_2"),
    Path("layer_b_outputs_2"),
    Path("/content/layer_b_outputs_2"),
]

LAYER_C_DIRS = [
    Path("content/layer_c_outputs_2"),
    Path("layer_c_outputs_2"),
    Path("/content/layer_c_outputs_2"),
]

OUT_DIR = Path("content/layer_d_outputs_2")
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-9

GRAPH_QUERY = os.environ.get("LAYER_D_GRAPH_QUERY", "Bengaluru, Karnataka, India").strip()
GRAPH_FALLBACK_DIST_M = int(os.environ.get("LAYER_D_GRAPH_FALLBACK_DIST_M", "25000"))
GRAPH_BETWEENNESS_K = int(os.environ.get("LAYER_D_BETWEENNESS_K", "128"))

NEIGHBOR_RADIUS_M = float(os.environ.get("LAYER_D_NEIGHBOR_RADIUS_M", "1200"))
SPILLOVER_DECAY_M = float(os.environ.get("LAYER_D_SPILLOVER_DECAY_M", "600"))
TOP_N = int(os.environ.get("LAYER_D_TOP_N", "0"))  # 0 = all

GRAPHML_CACHE = CACHE_DIR / "bengaluru_drive_graph.graphml"
CENTRALITY_CACHE = CACHE_DIR / "bengaluru_drive_graph_node_betweenness.csv"

# -------------------------
# Generic helpers
# -------------------------
def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

def safe_series(g, col, default=np.nan):
    if col in g.columns:
        return pd.to_numeric(g[col], errors="coerce")
    return pd.Series([default] * len(g), index=g.index, dtype=float)

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


def dominant_label(series, default=""):
    s = pd.Series(series).dropna().astype(str).str.strip()
    s = s[s.ne("")]
    if s.empty:
        return default
    m = s.mode()
    if not m.empty:
        return m.iloc[0]
    return s.iloc[0]


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if len(valid) == 0 or valid.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    mn = valid.min()
    mx = valid.max()
    return (s.fillna(mn) - mn) / (mx - mn + EPS)


def active_minmax(s: pd.Series) -> Tuple[pd.Series, bool]:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if len(valid) == 0 or valid.nunique(dropna=True) <= 1:
        return pd.Series(np.nan, index=s.index, dtype=float), False
    mn = valid.min()
    mx = valid.max()
    return (s.fillna(mn) - mn) / (mx - mn + EPS), True


def weighted_score(df: pd.DataFrame, specs, prefix: str):
    """
    specs = [(source_col, weight), ...]
    Only active (non-constant, non-empty) components contribute.
    """
    out = df.copy()
    active = []

    for source_col, weight in specs:
        norm_col = f"{prefix}_{source_col}_norm".replace(" ", "_")
        if source_col not in out.columns:
            out[norm_col] = np.nan
            continue

        norm, is_active = active_minmax(out[source_col])
        out[norm_col] = norm
        if is_active:
            active.append((norm_col, weight))

    if not active:
        out[f"{prefix}_raw"] = 0.0
        out[f"{prefix}_score"] = 0.0
        return out

    total_weight = float(sum(w for _, w in active))
    raw = np.zeros(len(out), dtype=float)
    for norm_col, weight in active:
        raw += weight * out[norm_col].fillna(0.0).to_numpy(dtype=float)

    raw = raw / max(total_weight, EPS)
    out[f"{prefix}_raw"] = raw
    out[f"{prefix}_score"] = 100.0 * np.clip(raw, 0.0, 1.0)
    return out


def make_hotspot_key(df: pd.DataFrame) -> pd.Series:
    label = df.get("cluster_label", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().str.lower()
    if "lat" in df.columns and "lon" in df.columns:
        lat = pd.to_numeric(df["lat"], errors="coerce").round(5).astype(str)
        lon = pd.to_numeric(df["lon"], errors="coerce").round(5).astype(str)
    else:
        lat = pd.Series("", index=df.index)
        lon = pd.Series("", index=df.index)
    return label + "|" + lat + "|" + lon


def haversine_matrix(lat_rad: np.ndarray, lon_rad: np.ndarray) -> np.ndarray:
    """
    Vectorized pairwise haversine distance matrix (meters).
    """
    R = 6371000.0
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def safe_merge_extra_columns(base: pd.DataFrame, extra: pd.DataFrame, key: str = "cluster_id") -> pd.DataFrame:
    """
    Merge extra enrichment into base using key, keeping base values first and
    filling only missing values from extra.
    """
    if extra is None or len(extra) == 0 or key not in extra.columns:
        return base

    # Coerce the join key to string on both sides so int64 vs object never
    # causes a ValueError — cluster_id can arrive as int64 from a CSV and as
    # str after astype(str) downstream.
    base = base.copy()
    base[key] = base[key].astype(str)
    extra = extra.copy()
    extra[key] = extra[key].astype(str)
    extra = extra.drop_duplicates(subset=[key], keep="first")
    extra_cols = [c for c in extra.columns if c != key]
    if not extra_cols:
        return base

    merged = base.merge(extra[[key] + extra_cols], on=key, how="left", suffixes=("", "_extra"))

    for c in extra_cols:
        ec = f"{c}_extra"
        if ec not in merged.columns:
            continue

        if c in merged.columns:
            if pd.api.types.is_numeric_dtype(merged[c]) or pd.api.types.is_numeric_dtype(merged[ec]):
                merged[c] = pd.to_numeric(merged[c], errors="coerce").combine_first(
                    pd.to_numeric(merged[ec], errors="coerce")
                )
            else:
                merged[c] = merged[c].combine_first(merged[ec])
        else:
            merged[c] = merged[ec]

        merged.drop(columns=[ec], inplace=True, errors="ignore")

    return merged


def collapse_physical_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate physical hotspots using rounded lat/lon + normalized label.
    """
    if df is None or len(df) == 0:
        return df

    work = df.copy()
    work["physical_hotspot_key"] = make_hotspot_key(work)

    rows = []
    for key, g in work.groupby("physical_hotspot_key", sort=False):
        g = g.copy()
        top = g.sort_values(
            ["ccs_score", "delay_minutes_per_vehicle", "records_total"],
            ascending=[False, False, False],
        ).iloc[0]

        row = {
            "physical_hotspot_key": key,
            "merged_cluster_ids": "|".join(sorted({str(x) for x in g["cluster_id"].dropna().tolist()})),
            "cluster_id": top.get("cluster_id"),
            "cluster_label": top.get("cluster_label", ""),
            "cluster_label_display": top.get("cluster_label_display", ""),
            "risk_band": top.get("risk_band", "Watch"),
            "lat": float(pd.to_numeric(g["lat"], errors="coerce").mean()),
            "lon": float(pd.to_numeric(g["lon"], errors="coerce").mean()),
            "ccs_score": float(pd.to_numeric(g["ccs_score"], errors="coerce").max()),
            "delay_minutes_per_vehicle": float(pd.to_numeric(g["delay_minutes_per_vehicle"], errors="coerce").max()),
            "records_total": float(pd.to_numeric(g.get("records_total", 0), errors="coerce").fillna(0).sum()),
            "distinct_days": float(pd.to_numeric(g.get("distinct_days", 0), errors="coerce").fillna(0).max()),
            "severity_sum": float(pd.to_numeric(g.get("severity_sum", 0), errors="coerce").fillna(0).sum()),
            "severity_mean": float(pd.to_numeric(g.get("severity_mean", 0), errors="coerce").fillna(0).mean()),
            "growth_pct_change": float(pd.to_numeric(g.get("growth_pct_change", 0), errors="coerce").fillna(0).max()),
            "growth_multiplier": float(pd.to_numeric(g.get("growth_multiplier", 1), errors="coerce").fillna(1).max()),
            "context_multiplier": float(pd.to_numeric(g.get("context_multiplier", 1), errors="coerce").fillna(1).max()),
            "layer_b_priority_boost": float(pd.to_numeric(g.get("layer_b_priority_boost", 0), errors="coerce").fillna(0).max()),
            "layer_b_alert_flag": bool(g.get("layer_b_alert_flag", False).astype(bool).any()) if "layer_b_alert_flag" in g.columns else False,
            "validation_uncertainty": float(pd.to_numeric(g.get("validation_uncertainty", np.nan), errors="coerce").fillna(np.nan).mean()),
            "resurgence_score": float(pd.to_numeric(g.get("resurgence_score", np.nan), errors="coerce").fillna(np.nan).max()),
            "persistence_score": float(pd.to_numeric(g.get("persistence_score", np.nan), errors="coerce").fillna(np.nan).max()),
            "anomaly_score": float(pd.to_numeric(g.get("anomaly_score", np.nan), errors="coerce").fillna(np.nan).max()),
            "rop": float(pd.to_numeric(g.get("rop", np.nan), errors="coerce").fillna(np.nan).max()),
            "tvs": float(pd.to_numeric(g.get("tvs", np.nan), errors="coerce").fillna(np.nan).max()),
            "vdi": float(pd.to_numeric(g.get("vdi", np.nan), errors="coerce").fillna(np.nan).max()),
            "nearby_sensitive_poi_count": float(pd.to_numeric(g.get("nearby_sensitive_poi_count", 0), errors="coerce").fillna(0).sum()),
            "road_class": clean_text(top.get("road_class", "road")),
            "lane_count": float(pd.to_numeric(g.get("lane_count", np.nan), errors="coerce").fillna(np.nan).mean()),
            "carriageway_width_m": float(pd.to_numeric(g.get("carriageway_width_m", np.nan), errors="coerce").fillna(np.nan).mean()),
            "link_length_m": float(pd.to_numeric(g.get("link_length_m", np.nan), errors="coerce").fillna(np.nan).mean()),
            "junction_degree": float(pd.to_numeric(g.get("junction_degree", np.nan), errors="coerce").fillna(np.nan).mean()),
            "betweenness_centrality": float(pd.to_numeric(g.get("betweenness_centrality", np.nan), errors="coerce").fillna(np.nan).mean()),
            "geometry_source": clean_text(top.get("geometry_source", "fallback")),
            "mappls_address": clean_text(top.get("mappls_address", "")),
            "road_node_id": top.get("road_node_id", np.nan),
            "road_node_distance_m": float(pd.to_numeric(g.get("road_node_distance_m", np.nan), errors="coerce").fillna(np.nan).mean()),
            "road_node_degree": float(pd.to_numeric(g.get("road_node_degree", np.nan), errors="coerce").fillna(np.nan).mean()),
            "road_node_betweenness": float(pd.to_numeric(g.get("road_node_betweenness", np.nan), errors="coerce").fillna(np.nan).mean()),
            "source_pressure_score": float(safe_series(g, "source_pressure_score").mean()),
            "source_pressure_norm": float(safe_series(g, "source_pressure_norm").mean()),
            "spillover_out_score": float(safe_series(g, "spillover_out_score").fillna(0.0).max()),
            "spillover_in_score": float(safe_series(g, "spillover_in_score").fillna(0.0).max()),
            "spillover_total_score": float(safe_series(g, "spillover_total_score").fillna(0.0).max()),
            "propagation_radius_m": float(safe_series(g, "propagation_radius_m").mean()),
            "network_pagerank": float(safe_series(g, "network_pagerank").mean()),
            "network_component_id": int(pd.to_numeric(g.get("network_component_id", 0), errors="coerce").fillna(0).iloc[0]) if "network_component_id" in g.columns else 0,
            "network_component_size": int(pd.to_numeric(g.get("network_component_size", 1), errors="coerce").fillna(1).iloc[0]) if "network_component_size" in g.columns else 1,
            "neighbor_count": int(safe_series(g, "neighbor_count",0).max()),
            "in_neighbor_count": int(safe_series(g, "in_neighbor_count",0).max()),
            "out_neighbor_count": int(safe_series(g, "out_neighbor_count",0).max()),
            "influence_asymmetry": float(safe_series(g, "influence_asymmetry",0).max()),
            "network_vulnerability_score": float(safe_series(g, "network_vulnerability_score").mean()),
            "layer_d_alert_flag": bool(g.get("layer_d_alert_flag", False).astype(bool).any()) if "layer_d_alert_flag" in g.columns else False,
        }

        rows.append(row)

    out = pd.DataFrame(rows)
    out["cluster_label"] = out["cluster_label"].fillna("").astype(str).str.strip()
    out["cluster_label_display"] = out["cluster_label_display"].fillna("").astype(str).str.strip()
    return out


# -------------------------
# Layer B / C optional load
# -------------------------
def load_optional_table(base_dirs: Iterable[Path], filenames: Iterable[str]):
    df, src = load_first_existing(base_dirs, filenames)
    return df, src


# -------------------------
# Road graph caching
# -------------------------
def load_or_build_road_graph(place_query: str, fallback_center: Tuple[float, float]):
    """
    Reuse GraphML if present. Otherwise download once and cache it.
    """
    if GRAPHML_CACHE.exists():
        try:
            G = ox.io.load_graphml(GRAPHML_CACHE)
            return G, f"cache:{GRAPHML_CACHE}"
        except Exception as e:
            print(f"Graph cache load failed, rebuilding: {e}")

    try:
        G = ox.graph_from_place(
            place_query,
            network_type="drive",
            simplify=True,
            retain_all=False,
        )
        ox.io.save_graphml(G, GRAPHML_CACHE)
        return G, f"downloaded_place:{place_query}"
    except Exception as place_err:
        lat0, lon0 = fallback_center
        try:
            G = ox.graph_from_point(
                (lat0, lon0),
                dist=GRAPH_FALLBACK_DIST_M,
                network_type="drive",
                simplify=True,
                retain_all=False,
            )
            ox.io.save_graphml(G, GRAPHML_CACHE)
            return G, f"downloaded_point:{place_err}"
        except Exception as point_err:
            raise RuntimeError(
                f"Failed to build road graph from place and point. place_error={place_err!r}, point_error={point_err!r}"
            )


def load_or_compute_node_betweenness(G):
    """
    Compute betweenness once and cache it. Uses approximate BC with k-sampling
    when the graph is large, because full betweenness can be costly.
    """
    if CENTRALITY_CACHE.exists():
        try:
            cache_df = pd.read_csv(CENTRALITY_CACHE)
            if {"node_id", "betweenness"}.issubset(cache_df.columns):
                return dict(zip(cache_df["node_id"], cache_df["betweenness"])), "cache"
        except Exception:
            pass

    # Simplify parallel edges before BC.
    DG = ox.convert.to_digraph(G, weight="length")
    Gu = DG.to_undirected()

    n_nodes = len(Gu)
    if n_nodes <= 1500:
        bc = nx.betweenness_centrality(Gu, normalized=True, weight="length")
        method = "exact"
    else:
        k = min(GRAPH_BETWEENNESS_K, max(64, int(np.sqrt(n_nodes))))
        bc = nx.betweenness_centrality(Gu, k=k, normalized=True, weight="length", seed=42)
        method = f"approx_k={k}"

    cache_df = pd.DataFrame(
        [{"node_id": node, "betweenness": val} for node, val in bc.items()]
    )
    cache_df.to_csv(CENTRALITY_CACHE, index=False)
    return bc, method


def assign_road_nodes(hotspots: pd.DataFrame, G, bc_map: Dict[Any, float]) -> pd.DataFrame:
    work = hotspots.copy()
    X = pd.to_numeric(work["lon"], errors="coerce").to_numpy(dtype=float)
    Y = pd.to_numeric(work["lat"], errors="coerce").to_numpy(dtype=float)

    print("Assigning nearest road nodes...")
    nearest_nodes, nearest_dists = ox.distance.nearest_nodes(G, X=X, Y=Y, return_dist=True)

    work["road_node_id"] = pd.Series(nearest_nodes, index=work.index)
    work["road_node_distance_m"] = pd.Series(nearest_dists, index=work.index).astype(float)

    # Degree from the road graph.
    road_deg = []
    road_bc = []
    for node in work["road_node_id"].tolist():
        try:
            road_deg.append(int(G.degree[node]))
        except Exception:
            road_deg.append(np.nan)
        road_bc.append(float(bc_map.get(node, np.nan)))

    work["road_node_degree"] = pd.Series(road_deg, index=work.index)
    work["road_node_betweenness"] = pd.Series(road_bc, index=work.index)

    return work


# -------------------------
# Layer D computation
# -------------------------
def ensure_feature_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create fallback feature columns if layer A/B/C outputs are not present.
    """
    work = df.copy()

    defaults = {
        "context_multiplier": 1.0,
        "layer_b_priority_boost": 0.0,
        "layer_b_alert_flag": False,
        "validation_uncertainty": np.nan,
        "resurgence_score": np.nan,
        "persistence_score": np.nan,
        "anomaly_score": np.nan,
        "rop": np.nan,
        "tvs": np.nan,
        "vdi": np.nan,
    }
    for c, default in defaults.items():
        if c not in work.columns:
            work[c] = default

    # Reasonable derived fallbacks if Layer C is not present.
    if "resurgence_score" not in df.columns or pd.to_numeric(work["resurgence_score"], errors="coerce").notna().sum() == 0:
        if "growth_pct_change" in work.columns:
            work["resurgence_score"] = pd.to_numeric(work["growth_pct_change"], errors="coerce").fillna(0.0).clip(lower=0.0)
        elif "growth_multiplier" in work.columns:
            work["resurgence_score"] = (pd.to_numeric(work["growth_multiplier"], errors="coerce").fillna(1.0) - 1.0).clip(lower=0.0)
        else:
            work["resurgence_score"] = 0.0

    if "persistence_score" not in df.columns or pd.to_numeric(work["persistence_score"], errors="coerce").notna().sum() == 0:
        records_norm = minmax(work.get("records_total", pd.Series(0, index=work.index)).fillna(0))
        distinct_days_norm = minmax(work.get("distinct_days", pd.Series(0, index=work.index)).fillna(0))
        repeat_norm = minmax(work.get("repeat_vehicle_count_2plus", pd.Series(0, index=work.index)).fillna(0))
        chronic_norm = minmax(work.get("chronic_vehicle_count_5plus", pd.Series(0, index=work.index)).fillna(0))
        work["persistence_score"] = 100.0 * (
            0.35 * records_norm +
            0.25 * distinct_days_norm +
            0.25 * repeat_norm +
            0.15 * chronic_norm
        )

    if "validation_uncertainty" not in df.columns or pd.to_numeric(work["validation_uncertainty"], errors="coerce").notna().sum() == 0:
        work["validation_uncertainty"] = 0.0

    if "anomaly_score" not in df.columns or pd.to_numeric(work["anomaly_score"], errors="coerce").notna().sum() == 0:
        work["anomaly_score"] = 0.0

    # Helpful context boost proxy for Layer B (if present, use it; otherwise zero).
    if "context_multiplier" in work.columns:
        work["context_boost"] = pd.to_numeric(work["context_multiplier"], errors="coerce").fillna(1.0) - 1.0
    else:
        work["context_boost"] = 0.0

    return work


def build_source_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite "pressure" score used as the source term in spillover propagation.
    """
    work = df.copy()

    # Positive-growth version of resurgence to avoid penalizing declining hotspots.
    if "growth_surge" not in work.columns:
        if "growth_pct_change" in work.columns:
            work["growth_surge"] = pd.to_numeric(work["growth_pct_change"], errors="coerce").fillna(0.0).clip(lower=0.0)
        elif "growth_multiplier" in work.columns:
            work["growth_surge"] = (pd.to_numeric(work["growth_multiplier"], errors="coerce").fillna(1.0) - 1.0).clip(lower=0.0)
        else:
            work["growth_surge"] = 0.0

    # Add a few composite terms before weighting.
    if "context_boost" not in work.columns:
        work["context_boost"] = pd.to_numeric(work.get("context_multiplier", 1.0), errors="coerce").fillna(1.0) - 1.0

    specs = [
        ("ccs_score", 0.25),
        ("delay_minutes_per_vehicle", 0.15),
        ("growth_surge", 0.10),
        ("layer_b_priority_boost", 0.10),
        ("context_boost", 0.08),
        ("persistence_score", 0.10),
        ("resurgence_score", 0.10),
        ("validation_uncertainty", 0.05),
        ("anomaly_score", 0.04),
        ("rop", 0.06),
        ("tvs", 0.04),
        ("vdi", 0.03),
    ]

    work = weighted_score(work, specs=specs, prefix="layer_d_pressure")
    work["source_pressure_score"] = pd.to_numeric(work["layer_d_pressure_score"], errors="coerce").fillna(0.0)

    # Fallback: if weighted_score produced all-zeros (every feature was constant
    # or missing), use a simple minmax of ccs_score so pressure is never flat.
    if work["source_pressure_score"].max() == 0.0:
        fallback = minmax(pd.to_numeric(work.get("ccs_score", pd.Series(0, index=work.index)), errors="coerce").fillna(0.0))
        if fallback.max() > 0:
            print("WARNING: all pressure features were constant — falling back to ccs_score minmax for source pressure.")
            work["source_pressure_score"] = fallback * 100.0
        else:
            # Last resort: uniform pressure so spillover at least runs on proximity alone.
            print("WARNING: ccs_score also constant — using uniform source pressure.")
            work["source_pressure_score"] = 50.0

    work["source_pressure_norm"] = work["source_pressure_score"] / 100.0
    return work


def build_spillover_network(hotspots: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build directed hotspot spillover graph from pairwise proximity and source pressure.
    """
    work = hotspots.copy().reset_index(drop=True)
    work = work.reset_index(drop=True)

    work["source_idx"] = np.arange(len(work))
    n = len(work)
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    lat = pd.to_numeric(work["lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(work["lon"], errors="coerce").to_numpy(dtype=float)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    print("Computing pairwise hotspot distances...")
    dist_m = haversine_matrix(lat_rad, lon_rad)

    # Candidate pairs within the spillover radius.
    pair_mask = np.triu((dist_m > 0) & (dist_m <= NEIGHBOR_RADIUS_M), k=1)
    pair_idx = np.column_stack(np.where(pair_mask))

    edge_rows = []
    print(f"Building spillover edges for {len(pair_idx)} hotspot pairs...")
    for idx, (i, j) in enumerate(pair_idx, start=1):
        d = float(dist_m[i, j])
        decay = float(np.exp(-d / max(SPILLOVER_DECAY_M, EPS)))

        src_pressure_i = float(work.loc[i, "source_pressure_norm"])
        src_pressure_j = float(work.loc[j, "source_pressure_norm"])

        cent_i = float(work.loc[i, "road_node_betweenness_norm"]) if "road_node_betweenness_norm" in work.columns else 0.0
        cent_j = float(work.loc[j, "road_node_betweenness_norm"]) if "road_node_betweenness_norm" in work.columns else 0.0

        # Directed influences.
        w_ij = src_pressure_i * decay * (1.0 + 0.5 * cent_j)
        w_ji = src_pressure_j * decay * (1.0 + 0.5 * cent_i)

        edge_rows.append(
            {
                "source_idx": i,
                "target_idx": j,
                "source_cluster_id": work.loc[i, "cluster_id"],
                "target_cluster_id": work.loc[j, "cluster_id"],
                "source_label": clean_text(work.loc[i, "cluster_label_display"]),
                "target_label": clean_text(work.loc[j, "cluster_label_display"]),
                "distance_m": d,
                "decay_factor": decay,
                "source_pressure_norm": src_pressure_i,
                "target_pressure_norm": src_pressure_j,
                "source_road_centrality_norm": cent_i,
                "target_road_centrality_norm": cent_j,
                "influence_weight": w_ij,
            }
        )
        edge_rows.append(
            {
                "source_idx": j,
                "target_idx": i,
                "source_cluster_id": work.loc[j, "cluster_id"],
                "target_cluster_id": work.loc[i, "cluster_id"],
                "source_label": clean_text(work.loc[j, "cluster_label_display"]),
                "target_label": clean_text(work.loc[i, "cluster_label_display"]),
                "distance_m": d,
                "decay_factor": decay,
                "source_pressure_norm": src_pressure_j,
                "target_pressure_norm": src_pressure_i,
                "source_road_centrality_norm": cent_j,
                "target_road_centrality_norm": cent_i,
                "influence_weight": w_ji,
            }
        )

        if idx % 1000 == 0:
            print(f"  processed {idx}/{len(pair_idx)} pairs")

    edge_df = pd.DataFrame(edge_rows)

    # Build directed graph for PageRank and an undirected graph for components.
    DG = nx.DiGraph()
    UG = nx.Graph()
    for idx, row in work.iterrows():
        node_key = int(idx)
        DG.add_node(node_key)
        UG.add_node(node_key)

    for r in edge_df.itertuples(index=False):
        w = float(r.influence_weight)
        if w <= 0.0:
            continue  # skip zero-weight edges — they carry no spillover signal
        DG.add_edge(int(r.source_idx), int(r.target_idx), weight=w)
        UG.add_edge(int(r.source_idx), int(r.target_idx), weight=w)

    # PageRank over spillover graph.
    if DG.number_of_edges() > 0:
        pr = nx.pagerank(DG, alpha=0.85, weight="weight", max_iter=200)
    else:
        # No real edges (all weights were zero) — uniform rank.
        uniform = 1.0 / max(len(DG), 1)
        pr = {node: uniform for node in DG.nodes()}

    pagerank_df = pd.DataFrame(
        [{"source_idx": k, "network_pagerank": v} for k, v in pr.items()]
    )

    # Component id / size.
    comp_rows = []
    for comp_id, comp in enumerate(nx.connected_components(UG), start=1):
        size = len(comp)
        for node in comp:
            comp_rows.append(
                {
                    "source_idx": int(node),
                    "network_component_id": comp_id,
                    "network_component_size": size,
                }
            )
    comp_df = pd.DataFrame(comp_rows)

    # ----------------------------------------------------------------
    # Aggregate spillover scores directly onto work via index maps.
    # We deliberately avoid merges here — merges on computed DataFrames
    # have caused silent column-missing bugs across pandas versions.
    # Instead we build plain Python dicts keyed by source_idx (integer
    # row position) and assign column arrays directly.
    # ----------------------------------------------------------------
    n_work = len(work)

    # Initialize all network columns to their neutral defaults first.
    out_score_arr   = np.zeros(n_work, dtype=float)
    in_score_arr    = np.zeros(n_work, dtype=float)
    out_nbr_arr     = np.zeros(n_work, dtype=int)
    in_nbr_arr      = np.zeros(n_work, dtype=int)
    prop_rad_arr    = np.full(n_work, np.nan, dtype=float)
    pr_arr          = np.zeros(n_work, dtype=float)
    comp_id_arr     = np.zeros(n_work, dtype=int)
    comp_sz_arr     = np.ones(n_work,  dtype=int)

    if len(edge_df):
        iw = pd.to_numeric(edge_df["influence_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        dm = pd.to_numeric(edge_df["distance_m"],       errors="coerce").fillna(0.0).to_numpy(dtype=float)
        si = edge_df["source_idx"].to_numpy(dtype=int)
        ti = edge_df["target_idx"].to_numpy(dtype=int)

        # Out-scores: sum influence weights leaving each source.
        np.add.at(out_score_arr, si, iw)

        # In-scores: sum influence weights arriving at each target.
        np.add.at(in_score_arr,  ti, iw)

        # Out-neighbor count: number of distinct targets per source.
        src_neighbors: dict = {}   # source_idx -> set of target_idx
        tgt_sources:   dict = {}   # target_idx -> set of source_idx
        for s_i, t_i in zip(si.tolist(), ti.tolist()):
            src_neighbors.setdefault(s_i, set()).add(t_i)
            tgt_sources.setdefault(t_i, set()).add(s_i)
        for s_i, nbrs in src_neighbors.items():
            out_nbr_arr[s_i] = len(nbrs)
        for t_i, srcs in tgt_sources.items():
            in_nbr_arr[t_i] = len(srcs)

        # Weighted-average propagation radius per source.
        # prop_radius[s] = sum(w*d) / sum(w)
        num_arr = np.zeros(n_work, dtype=float)
        den_arr = np.zeros(n_work, dtype=float)
        np.add.at(num_arr, si, iw * dm)
        np.add.at(den_arr, si, iw)
        with np.errstate(invalid="ignore", divide="ignore"):
            prop_rad_arr = np.where(den_arr > 0, num_arr / den_arr, np.nan)

    # PageRank: assign from dict (node -> value), node == source_idx.
    for node, val in pr.items():
        idx_i = int(node)
        if 0 <= idx_i < n_work:
            pr_arr[idx_i] = float(val)

    # Component id/size: assign from comp_rows list.
    for row in comp_rows:
        idx_i = int(row["source_idx"])
        if 0 <= idx_i < n_work:
            comp_id_arr[idx_i] = int(row["network_component_id"])
            comp_sz_arr[idx_i] = int(row["network_component_size"])

    # Assign all columns directly — no merges, no missing-column risk.
    work = work.copy()
    work["spillover_out_score"]      = out_score_arr
    work["spillover_in_score"]       = in_score_arr
    work["spillover_total_score"]    = out_score_arr + in_score_arr
    work["out_neighbor_count"]       = out_nbr_arr
    work["in_neighbor_count"]        = in_nbr_arr
    work["neighbor_count"]           = out_nbr_arr + in_nbr_arr
    work["propagation_radius_m"]     = prop_rad_arr
    work["network_pagerank"]         = pr_arr
    work["network_component_id"]     = comp_id_arr
    work["network_component_size"]   = comp_sz_arr
    work["influence_asymmetry"]      = out_score_arr - in_score_arr

    # Additional derived flags.
    pr_q80 = work["network_pagerank"].quantile(0.80) if len(work) else 0.0
    spill_q80 = work["spillover_total_score"].quantile(0.80) if len(work) else 0.0
    work["layer_d_alert_flag"] = (
        (work["network_pagerank"] >= pr_q80) |
        (work["spillover_total_score"] >= spill_q80)
    )

    return work, edge_df


def finalize_layer_d(hotspots: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """
    Compute final Layer D scores from source pressure + spillover + road-network metrics.
    """
    work = hotspots.copy()

    # Normalize road-node metrics and spillover metrics.
    if "road_node_betweenness" not in work.columns:
        work["road_node_betweenness"] = np.nan
    if "road_node_degree" not in work.columns:
        work["road_node_degree"] = np.nan
    if "road_node_distance_m" not in work.columns:
        work["road_node_distance_m"] = np.nan
    if "validation_uncertainty" not in work.columns:
        work["validation_uncertainty"] = np.nan
    if "resurgence_score" not in work.columns:
        work["resurgence_score"] = np.nan
    if "persistence_score" not in work.columns:
        work["persistence_score"] = np.nan
    if "anomaly_score" not in work.columns:
        work["anomaly_score"] = np.nan
    if "context_multiplier" not in work.columns:
        work["context_multiplier"] = 1.0
    if "layer_b_priority_boost" not in work.columns:
        work["layer_b_priority_boost"] = 0.0

    # Road-node centrality normalization.
    work["road_node_betweenness_norm"] = minmax(work["road_node_betweenness"].fillna(0.0))
    work["road_node_degree_norm"] = minmax(work["road_node_degree"].fillna(0.0))
    work["road_node_distance_norm"] = minmax(work["road_node_distance_m"].fillna(work["road_node_distance_m"].max() if work["road_node_distance_m"].notna().any() else 0.0))

    # Ensure source pressure exists.
    work = build_source_pressure(work)

    # If any Layer C/novel features are present, retain their normalized footprints.
    if "growth_pct_change" in work.columns:
        work["growth_surge"] = pd.to_numeric(work["growth_pct_change"], errors="coerce").fillna(0.0).clip(lower=0.0)
    elif "growth_multiplier" in work.columns:
        work["growth_surge"] = (pd.to_numeric(work["growth_multiplier"], errors="coerce").fillna(1.0) - 1.0).clip(lower=0.0)
    else:
        work["growth_surge"] = 0.0

    # Final vulnerability score:
    # source pressure + outgoing spillover + incoming spillover + road-network centrality + PageRank.
    work = weighted_score(
        work,
        specs=[
            ("source_pressure_score", 0.35),
            ("spillover_out_score", 0.20),
            ("spillover_in_score", 0.15),
            ("road_node_betweenness", 0.15),
            ("network_pagerank", 0.10),
            ("resurgence_score", 0.05),
            ("validation_uncertainty", 0.05),
        ],
        prefix="layer_d_vulnerability",
    )
    work["network_vulnerability_score"] = pd.to_numeric(work["layer_d_vulnerability_score"], errors="coerce").fillna(0.0)

    # Helpful transparency columns.
    work["spillover_total_score"] = pd.to_numeric(work.get("spillover_total_score", work["spillover_out_score"] + work["spillover_in_score"]), errors="coerce").fillna(0.0)
    work["secondary_risk_score"] = work["spillover_in_score"]
    work["source_pressure_norm"] = pd.to_numeric(work["source_pressure_norm"], errors="coerce").fillna(0.0)

    # Rank + alerting.
    work = work.sort_values(
        ["network_vulnerability_score", "spillover_total_score", "ccs_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    work["network_rank"] = np.arange(1, len(work) + 1)

    vuln_q80 = work["network_vulnerability_score"].quantile(0.80) if len(work) else 0.0
    spill_q80 = work["spillover_total_score"].quantile(0.80) if len(work) else 0.0
    work["layer_d_alert_flag"] = (
        (work["network_vulnerability_score"] >= vuln_q80) |
        (work["spillover_total_score"] >= spill_q80)
    )

    # Propagation class for dashboarding.
    work["propagation_class"] = np.select(
        [
            work["spillover_total_score"] >= work["spillover_total_score"].quantile(0.90),
            work["spillover_total_score"] >= work["spillover_total_score"].quantile(0.70),
            work["spillover_total_score"] >= work["spillover_total_score"].quantile(0.50),
        ],
        [
            "Severe Propagator",
            "Strong Propagator",
            "Moderate Propagator",
        ],
        default="Localized",
    )

    return work


def main():
    # -------------------------
    # Load base Layer B / Phase 5 table
    # -------------------------
    base_df, base_src = load_first_existing(
        BASE_DIRS,
        [
            "phase5_with_layer_b.csv",
            "phase5_cluster_capacity_loss.csv",
            "phase5_priority_table_full.csv",
            "phase5_stage5_handoff.csv",
        ],
    )
    if base_df is None:
        raise FileNotFoundError(
            "Could not find a Layer B / Stage 5 hotspot table. "
            "Expected phase5_with_layer_b.csv or phase5_cluster_capacity_loss.csv."
        )

    base_df = base_df.copy()
    base_cluster_col = standardize_cluster_col(base_df)

    # Standardize to a working cluster_id column, but keep original too.
    if "cluster_id" not in base_df.columns:
        base_df["cluster_id"] = base_df[base_cluster_col]
    else:
        base_df["cluster_id"] = base_df["cluster_id"].fillna(base_df[base_cluster_col])
    # Normalise to string — CSVs may load it as int64, but hotspot_df uses str.
    base_df["cluster_id"] = base_df["cluster_id"].astype(str)

    base_df = ensure_label_column(base_df)
    base_df = derive_coords(base_df)

    # Helpful display label.
    if "cluster_label_display" not in base_df.columns:
        base_df["cluster_label_display"] = (
            base_df["cluster_label"].astype(str) +
            " (Cluster " +
            base_df["cluster_id"].astype(str) +
            ")"
        )

    # -------------------------
    # Optional Layer C enrichment
    # -------------------------
    layer_c_df, layer_c_src = load_optional_table(
        LAYER_C_DIRS,
        [
            "layer_c_enriched_hotspots.csv",
            "layer_c_hotspot_features.csv",
            "layer_c_novel_features.csv",
            "phase5_with_layer_c.csv",
            "layer_c_final.csv",
        ],
    )
    if layer_c_df is not None and len(layer_c_df):
        layer_c_df = layer_c_df.copy()
        try:
            layer_c_cluster_col = standardize_cluster_col(layer_c_df)
            if "cluster_id" not in layer_c_df.columns:
                layer_c_df["cluster_id"] = layer_c_df[layer_c_cluster_col]
            layer_c_df["cluster_id"] = layer_c_df["cluster_id"].astype(str)
            layer_c_df = ensure_label_column(layer_c_df)
            layer_c_df = derive_coords(layer_c_df)
            base_df = safe_merge_extra_columns(base_df, layer_c_df, key="cluster_id")
        except Exception as e:
            print(f"Layer C merge skipped: {e}")

    # -------------------------
    # Ensure required columns exist
    # -------------------------
    required_numeric_defaults = {
        "ccs_score": 0.0,
        "delay_minutes_per_vehicle": 0.0,
        "records_total": 0.0,
        "distinct_days": 1.0,
        "severity_sum": 0.0,
        "severity_mean": 0.0,
        "growth_pct_change": 0.0,
        "growth_multiplier": 1.0,
        "layer_b_priority_boost": 0.0,
        "context_multiplier": 1.0,
        "validation_uncertainty": np.nan,
        "resurgence_score": np.nan,
        "persistence_score": np.nan,
        "anomaly_score": np.nan,
        "rop": np.nan,
        "tvs": np.nan,
        "vdi": np.nan,
        "repeat_vehicle_count_2plus": 0.0,
        "chronic_vehicle_count_5plus": 0.0,
        "nearby_sensitive_poi_count": 0.0,
        "lane_count": np.nan,
        "carriageway_width_m": np.nan,
        "link_length_m": np.nan,
        "junction_degree": np.nan,
        "betweenness_centrality": np.nan,
        "road_node_distance_m": np.nan,
        "road_node_degree": np.nan,
        "road_node_betweenness": np.nan,
        "network_pagerank": np.nan,
        "spillover_out_score": np.nan,
        "spillover_in_score": np.nan,
        "spillover_total_score": np.nan,
        "propagation_radius_m": np.nan,
    }
    for c, default in required_numeric_defaults.items():
        if c not in base_df.columns:
            base_df[c] = default

    # Coerce numerics.
    for c in required_numeric_defaults:
        base_df[c] = pd.to_numeric(base_df[c], errors="coerce")

    # Derive missing/flat Layer C-like features.
    base_df = ensure_feature_defaults(base_df)

    # If label display was not present in optional files, build it.
    if "cluster_label_display" not in base_df.columns:
        base_df["cluster_label_display"] = (
            base_df["cluster_label"].astype(str) +
            " (Cluster " +
            base_df["cluster_id"].astype(str) +
            ")"
        )

    # Use a consistent hotspot key.
    base_df["physical_hotspot_key"] = make_hotspot_key(base_df)

    # Keep only coordinates-bearing rows.
    base_df["lat"] = pd.to_numeric(base_df["lat"], errors="coerce")
    base_df["lon"] = pd.to_numeric(base_df["lon"], errors="coerce")
    base_df = base_df.dropna(subset=["lat", "lon"]).copy()

    # Optional top-N restriction.
    if TOP_N and TOP_N > 0:
        base_df = base_df.sort_values(
            ["ccs_score", "delay_minutes_per_vehicle", "records_total"],
            ascending=[False, False, False],
        ).head(TOP_N).copy()

    # -------------------------
    # Collapse duplicates (same physical location)
    # -------------------------
    print("Collapsing duplicate physical hotspots...")
    hotspot_df = collapse_physical_hotspots(base_df)

    if len(hotspot_df) == 0:
        raise RuntimeError("No hotspot rows found after collapsing duplicates.")

    hotspot_df["cluster_label_display"] = hotspot_df["cluster_label_display"].fillna("").astype(str)
    hotspot_df["cluster_label_display"] = np.where(
        hotspot_df["cluster_label_display"].str.strip().eq(""),
        hotspot_df["cluster_label"].astype(str) + " (Cluster " + hotspot_df["cluster_id"].astype(str) + ")",
        hotspot_df["cluster_label_display"],
    )

    # -------------------------
    # Build / load road graph once
    # -------------------------
    center_lat = float(pd.to_numeric(hotspot_df["lat"], errors="coerce").mean())
    center_lon = float(pd.to_numeric(hotspot_df["lon"], errors="coerce").mean())
    print("Loading road graph...")
    G, graph_source = load_or_build_road_graph(GRAPH_QUERY, fallback_center=(center_lat, center_lon))
    print("Road graph source:", graph_source)

    print("Loading / computing node betweenness...")
    bc_map, bc_method = load_or_compute_node_betweenness(G)
    print("Betweenness method:", bc_method)

    # -------------------------
    # Road-node assignment
    # -------------------------
    hotspot_df = assign_road_nodes(hotspot_df, G, bc_map)

    # Normalize road metrics for later use.
    hotspot_df["road_node_betweenness_norm"] = minmax(hotspot_df["road_node_betweenness"].fillna(0.0))
    hotspot_df["road_node_degree_norm"] = minmax(hotspot_df["road_node_degree"].fillna(0.0))
    hotspot_df["road_node_distance_norm"] = minmax(hotspot_df["road_node_distance_m"].fillna(hotspot_df["road_node_distance_m"].max() if hotspot_df["road_node_distance_m"].notna().any() else 0.0))

    # If layer B priority/context exists, turn it into a boost term.
    if "context_multiplier" in hotspot_df.columns:
        hotspot_df["context_boost"] = pd.to_numeric(hotspot_df["context_multiplier"], errors="coerce").fillna(1.0) - 1.0
    else:
        hotspot_df["context_boost"] = 0.0

    if "layer_b_priority_boost" not in hotspot_df.columns:
        hotspot_df["layer_b_priority_boost"] = 0.0

    # -------------------------
    # Build hotspot spillover graph
    # -------------------------
    hotspot_df = build_source_pressure(hotspot_df)

    # Make sure road centrality is present as a numeric proxy.
    hotspot_df["road_node_betweenness"] = pd.to_numeric(hotspot_df["road_node_betweenness"], errors="coerce").fillna(0.0)
    hotspot_df["road_node_degree"] = pd.to_numeric(hotspot_df["road_node_degree"], errors="coerce").fillna(0.0)

    print("Building hotspot spillover network...")
    network_df, edge_df = build_spillover_network(hotspot_df)

    # network_df IS the fully enriched hotspot frame — build_spillover_network
    # assigns all spillover/pagerank/component columns directly onto the work
    # DataFrame and returns it. There is no merge step needed; using hotspot_df
    # here would silently overwrite computed scores with the stub zeros that
    # collapse_physical_hotspots put in place earlier.
    hotspot_df = network_df.copy()

    # Standardize types on the network columns (guard against any edge case).
    for c in ["spillover_out_score", "spillover_in_score", "spillover_total_score",
              "neighbor_count", "in_neighbor_count", "out_neighbor_count",
              "influence_asymmetry", "propagation_radius_m",
              "network_pagerank", "network_component_id", "network_component_size"]:
        if c not in hotspot_df.columns:
            hotspot_df[c] = np.nan
        hotspot_df[c] = pd.to_numeric(hotspot_df[c], errors="coerce")

    hotspot_df["spillover_out_score"]   = hotspot_df["spillover_out_score"].fillna(0.0)
    hotspot_df["spillover_in_score"]    = hotspot_df["spillover_in_score"].fillna(0.0)
    hotspot_df["spillover_total_score"] = hotspot_df["spillover_total_score"].fillna(
        hotspot_df["spillover_out_score"] + hotspot_df["spillover_in_score"]
    )
    hotspot_df["neighbor_count"]          = hotspot_df["neighbor_count"].fillna(0).astype(int)
    hotspot_df["in_neighbor_count"]       = hotspot_df["in_neighbor_count"].fillna(0).astype(int)
    hotspot_df["out_neighbor_count"]      = hotspot_df["out_neighbor_count"].fillna(0).astype(int)
    hotspot_df["network_component_id"]   = hotspot_df["network_component_id"].fillna(0).astype(int)
    hotspot_df["network_component_size"] = hotspot_df["network_component_size"].fillna(1).astype(int)
    hotspot_df["layer_d_alert_flag"]     = hotspot_df["layer_d_alert_flag"].fillna(False).astype(bool) if "layer_d_alert_flag" in hotspot_df.columns else False

    # -------------------------
    # Final Layer D score
    # -------------------------
    hotspot_df = finalize_layer_d(hotspot_df, edge_df)

    # -------------------------
    # Clean display / output columns
    # -------------------------
    hotspot_df["cluster_id"] = hotspot_df["cluster_id"].astype(str)
    hotspot_df["cluster_label"] = hotspot_df["cluster_label"].fillna("").astype(str)
    hotspot_df["cluster_label_display"] = hotspot_df["cluster_label_display"].fillna("").astype(str)
    hotspot_df["road_class"] = hotspot_df.get("road_class", "road").fillna("road").astype(str)
    hotspot_df["mappls_address"] = hotspot_df.get("mappls_address", "").fillna("").astype(str)

    # Network confidence / quality fields
    hotspot_df["road_node_distance_m"] = pd.to_numeric(hotspot_df["road_node_distance_m"], errors="coerce")
    hotspot_df["road_node_degree"] = pd.to_numeric(hotspot_df["road_node_degree"], errors="coerce")
    hotspot_df["road_node_betweenness"] = pd.to_numeric(hotspot_df["road_node_betweenness"], errors="coerce")

    # -------------------------
    # Output files
    # -------------------------
    full_output = hotspot_df.copy()
    edge_output = edge_df.copy()

    # Ranking / hotspot summary.
    hotspot_summary_cols = [
        "network_rank",
        "cluster_id",
        "cluster_label",
        "cluster_label_display",
        "risk_band",
        "network_vulnerability_score",
        "source_pressure_score",
        "spillover_out_score",
        "spillover_in_score",
        "spillover_total_score",
        "secondary_risk_score",
        "propagation_radius_m",
        "neighbor_count",
        "road_node_betweenness",
        "road_node_degree",
        "road_node_distance_m",
        "network_pagerank",
        "network_component_id",
        "network_component_size",
        "layer_d_alert_flag",
        "propagation_class",
        "ccs_score",
        "delay_minutes_per_vehicle",
        "growth_pct_change",
        "growth_multiplier",
        "layer_b_priority_boost",
        "context_multiplier",
        "validation_uncertainty",
        "resurgence_score",
        "persistence_score",
        "anomaly_score",
        "rop",
        "tvs",
        "vdi",
        "records_total",
        "distinct_days",
        "severity_sum",
        "nearby_sensitive_poi_count",
        "road_class",
        "geometry_source",
        "mappls_address",
        "lat",
        "lon",
        "merged_cluster_ids",
        "physical_hotspot_key",
    ]
    hotspot_summary_cols = [c for c in hotspot_summary_cols if c in hotspot_df.columns]

    network_metric_cols = [
        "cluster_id",
        "cluster_label_display",
        "road_node_id",
        "road_node_distance_m",
        "road_node_degree",
        "road_node_betweenness",
        "road_node_degree_norm",
        "road_node_betweenness_norm",
        "source_pressure_score",
        "source_pressure_norm",
        "spillover_out_score",
        "spillover_in_score",
        "spillover_total_score",
        "influence_asymmetry",
        "neighbor_count",
        "in_neighbor_count",
        "out_neighbor_count",
        "propagation_radius_m",
        "network_pagerank",
        "network_component_id",
        "network_component_size",
        "layer_d_alert_flag",
        "network_vulnerability_score",
        "network_rank",
        "propagation_class",
    ]
    network_metric_cols = [c for c in network_metric_cols if c in hotspot_df.columns]

    # Save files.
    full_output.to_csv(OUT_DIR / "layer_d_full_hotspot_output.csv", index=False)
    hotspot_df[hotspot_summary_cols].to_csv(OUT_DIR / "layer_d_final_ranking.csv", index=False)
    hotspot_df[network_metric_cols].to_csv(OUT_DIR / "layer_d_network_metrics.csv", index=False)
    edge_output.to_csv(OUT_DIR / "layer_d_spillover_edges.csv", index=False)

    # Merge back to upstream file shape for downstream use.
    stage5_with_d = safe_merge_extra_columns(
        base_df.copy(),
        hotspot_df.rename(columns={"cluster_id": "cluster_id"}),
        key="cluster_id",
    )
    stage5_with_d.to_csv(OUT_DIR / "phase5_with_layer_d.csv", index=False)

    # Summary.
    summary = pd.DataFrame(
        [
            {
                "input_source": str(base_src),
                "layer_c_source": str(layer_c_src) if layer_c_src is not None else "",
                "graph_source": graph_source,
                "betweenness_method": bc_method,
                "clusters_scored": int(len(full_output)),
                "distinct_hotspots": int(len(hotspot_df)),
                "spillover_edges": int(len(edge_output)),
                "alerts_flagged": int(hotspot_df["layer_d_alert_flag"].sum()) if "layer_d_alert_flag" in hotspot_df.columns else 0,
                "mean_network_vulnerability_score": float(pd.to_numeric(hotspot_df["network_vulnerability_score"], errors="coerce").mean()) if len(hotspot_df) else np.nan,
                "max_network_vulnerability_score": float(pd.to_numeric(hotspot_df["network_vulnerability_score"], errors="coerce").max()) if len(hotspot_df) else np.nan,
                "mean_spillover_total_score": float(pd.to_numeric(hotspot_df["spillover_total_score"], errors="coerce").mean()) if len(hotspot_df) else np.nan,
                "mean_road_node_distance_m": float(pd.to_numeric(hotspot_df["road_node_distance_m"], errors="coerce").mean()) if len(hotspot_df) else np.nan,
            }
        ]
    )
    summary.to_csv(OUT_DIR / "layer_d_summary.csv", index=False)

    # -------------------------
    # Console output
    # -------------------------
    print("Layer D complete")
    print("Output directory:", OUT_DIR.resolve())
    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nTop 10 Layer D hotspots:")
    top_cols = [
        "network_rank",
        "cluster_id",
        "cluster_label_display",
        "risk_band",
        "network_vulnerability_score",
        "source_pressure_score",
        "spillover_total_score",
        "spillover_out_score",
        "spillover_in_score",
        "propagation_radius_m",
        "road_node_betweenness",
        "network_pagerank",
        "network_component_size",
        "layer_d_alert_flag",
    ]
    top_cols = [c for c in top_cols if c in hotspot_df.columns]
    print(hotspot_df[top_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()