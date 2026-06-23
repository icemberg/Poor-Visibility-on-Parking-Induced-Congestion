import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
INPUT_PATHS = [
    Path("content/phase6_outputs_2/phase6_stage6_handoff.csv"),
    Path("content/phase6_outputs_2/phase6_cluster_ccs_full.csv"),
    Path("phase6_outputs_2/phase6_stage6_handoff.csv"),
    Path("phase6_outputs_2/phase6_cluster_ccs_full.csv"),
    Path("/content/phase6_outputs_2/phase6_stage6_handoff.csv"),
    Path("/content/phase6_outputs_2/phase6_cluster_ccs_full.csv"),
    Path("phase6_outputs/phase6_stage6_handoff.csv"),
    Path("phase6_outputs/phase6_cluster_ccs_full.csv"),
    Path("content/phase6_outputs/phase6_stage6_handoff.csv"),
    Path("content/phase6_outputs/phase6_cluster_ccs_full.csv"),
]

AUX_PATHS = [
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

OUT_DIR = Path("content/validation_outputs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = int(os.environ.get("VALIDATION_TOP_N", "20"))
VALIDATION_CORRIDOR_METERS = float(os.environ.get("VALIDATION_CORRIDOR_METERS", "500"))
REQUEST_TIMEOUT = int(os.environ.get("VALIDATION_REQUEST_TIMEOUT", "25"))

MAPPLS_ACCESS_TOKEN = os.environ.get("MAPPLS_ACCESS_TOKEN", "").strip()
MAPPLS_REGION = os.environ.get("MAPPLS_REGION", "ind").strip().lower()
MAPPLS_PROFILE = os.environ.get("MAPPLS_PROFILE", "driving").strip().lower()

DEFAULT_ROAD_CLASS = "road"
EPS = 1e-9

# Same table Stage 5 uses, so the "free-flow ETA" computed here for the
# synthetic validation corridor is on the same basis as the model's own
# free_flow_time_min — this is what makes the comparison meaningful.
ROAD_CLASS_SPEED_KMH = {
    "motorway": 60.0, "trunk": 55.0, "primary": 45.0, "secondary": 40.0,
    "tertiary": 35.0, "unclassified": 30.0, "residential": 30.0,
    "living_street": 20.0, "service": 25.0, "road": 30.0,
}


# =========================
# Helpers
# =========================
def clean_text(x):
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

def valid_coords(lat, lon):
    try:
        return (
            pd.notna(lat) and pd.notna(lon)
            and -90 <= float(lat) <= 90
            and -180 <= float(lon) <= 180
        )
    except Exception:
        return False

def road_speed_kmh(road_class):
    return ROAD_CLASS_SPEED_KMH.get(clean_text(road_class).lower(), 30.0)

def load_first_existing(paths):
    for p in paths:
        if p.exists():
            return pd.read_csv(p, low_memory=False), p
    raise FileNotFoundError("Could not find a Stage 6 output file.")

def load_all_existing(paths):
    frames = []
    for p in paths:
        if p.exists():
            try:
                frames.append((p, pd.read_csv(p, low_memory=False)))
            except Exception:
                pass
    return frames

def standardize_cluster_col(df):
    for c in ["st_dbscan_cluster_id", "cluster_id", "dbscan_cluster_id"]:
        if c in df.columns:
            return c
    raise ValueError("No cluster id column found.")

def derive_lat_lon(df):
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

def offset_point(lat, lon, meters=500.0, bearing_deg=90.0):
    """Arbitrary nearby point used only to probe live road speed near the
    hotspot (a 'is traffic moving normally here right now' check) — it is
    NOT meant to represent the hotspot's actual congested approach/exit."""
    R = 6371000.0
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d = meters / R
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)

def deep_first_value(obj, keys):
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] not in (None, "", [], {}):
                return obj[k]
        for v in obj.values():
            found = deep_first_value(v, keys)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = deep_first_value(item, keys)
            if found not in (None, "", [], {}):
                return found
    return None

def extract_point_to_point(matrix, src_idx=0, dst_idx=1):
    """This script only ever submits exactly one source + one destination,
    so the Mappls response is a 1xN (N>=2) row-ordered matrix and the value
    we want is unambiguously matrix[src_idx][dst_idx] (per Mappls Distance
    Matrix docs: row-ordered source->target array). No need to scan for a
    'minimum off-diagonal' value — that was only safe by coincidence for a
    2x2 case and would silently pick the wrong cell for larger matrices."""
    if not isinstance(matrix, list) or len(matrix) <= src_idx:
        return None
    row = matrix[src_idx]
    if not isinstance(row, list) or len(row) <= dst_idx:
        return None
    try:
        v = float(row[dst_idx])
        return v if np.isfinite(v) and v >= 0 else None
    except Exception:
        return None

def parse_reverse_geocode_address(payload):
    if not isinstance(payload, dict):
        return ""
    results = payload.get("results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            addr = first.get("formatted_address") or first.get("address") or ""
            if addr:
                return str(addr).strip()
            parts = []
            for k in [
                "houseNumber", "houseName", "poi", "street", "subSubLocality",
                "subLocality", "locality", "subDistrict", "district", "city",
                "state", "pincode"
            ]:
                v = clean_text(first.get(k, ""))
                if v:
                    parts.append(v)
            if parts:
                return ", ".join(dict.fromkeys(parts))
    addr = deep_first_value(payload, ["formatted_address", "address"])
    return clean_text(addr) if addr is not None else ""

def parse_route_response(payload):
    """Per Mappls Route Driving Directions API: {"code":"Ok","routes":[{"distance":..,
    "duration":..,"legs":[{"distance":..,"duration":..}]}]} — distance is in
    meters, duration is in seconds, both directly on routes[0]."""
    if not isinstance(payload, dict):
        return None, None, "invalid"

    routes = payload.get("routes")
    if isinstance(routes, list) and routes:
        r0 = routes[0]
        d = safe_float(r0.get("distance"), None)
        t = safe_float(r0.get("duration"), None)
        if d is not None or t is not None:
            return d, t, "routes[0]"

        legs = r0.get("legs")
        if isinstance(legs, list) and legs:
            leg0 = legs[0]
            d = safe_float(leg0.get("distance"), None)
            t = safe_float(leg0.get("duration"), None)
            if d is not None or t is not None:
                return d, t, "routes[0].legs[0]"
            summary = leg0.get("summary")
            if isinstance(summary, dict):
                d = safe_float(summary.get("distance"), None)
                t = safe_float(summary.get("duration"), None)
                if d is not None or t is not None:
                    return d, t, "routes[0].legs[0].summary"

    return None, None, "unknown"

def parse_matrix_any(payload):
    """Per Mappls Distance Matrix API: {"results":{"code":"Ok","distances":[[...]],
    "durations":[[...]]}} — row-ordered, source->target. distances in meters,
    durations in seconds."""
    out = {"ok": False, "distance_m": None, "duration_sec": None, "source": "unknown", "raw": payload}
    if not isinstance(payload, dict):
        return out

    results = payload.get("results")
    if isinstance(results, list) and results:
        results = results[0]

    if isinstance(results, dict):
        if "distances" in results or "durations" in results:
            d = extract_point_to_point(results.get("distances"))
            t = extract_point_to_point(results.get("durations"))
            if d is not None or t is not None:
                out.update({"ok": True, "distance_m": d, "duration_sec": t, "source": "results.distances/durations"})
                return out

        stt = results.get("sources_to_targets")
        if isinstance(stt, list):
            dists, times = [], []
            for row in stt:
                if isinstance(row, list):
                    for cell in row:
                        if isinstance(cell, dict):
                            for key, arr in [("distance", dists), ("time", times), ("duration", times)]:
                                if key in cell:
                                    try:
                                        v = float(cell[key])
                                        if np.isfinite(v) and v >= 0:
                                            arr.append(v)
                                    except Exception:
                                        pass
            if dists or times:
                out.update({
                    "ok": True,
                    "distance_m": dists[0] if dists else None,
                    "duration_sec": times[0] if times else None,
                    "source": "results.sources_to_targets",
                })
                return out

    d = deep_first_value(payload, ["distance", "distances", "length"])
    t = deep_first_value(payload, ["duration", "durations", "time"])
    d = extract_point_to_point(d) if isinstance(d, list) else (safe_float(d, None) if d is not None else None)
    t = extract_point_to_point(t) if isinstance(t, list) else (safe_float(t, None) if t is not None else None)

    if d is not None or t is not None:
        out.update({"ok": True, "distance_m": d, "duration_sec": t, "source": "generic"})
    return out

def make_key(series):
    return series.fillna("").astype(str).str.strip().str.lower()

def fill_coords_from_lookup(df, lookup, key):
    if key not in df.columns or key not in lookup.columns:
        return df.copy()
    work = df.copy()
    lk = lookup.copy()
    if "lat" not in work.columns:
        work["lat"] = np.nan
    if "lon" not in work.columns:
        work["lon"] = np.nan
    work["_join_key"] = make_key(work[key])
    lk["_join_key"] = make_key(lk[key])
    lk = lk[lk["_join_key"].ne("")].copy()
    if lk.empty:
        work.drop(columns=["_join_key"], inplace=True, errors="ignore")
        return work
    lk = lk.groupby("_join_key", as_index=False).agg(lat=("lat", "mean"), lon=("lon", "mean"))
    merged = work.merge(lk, on="_join_key", how="left", suffixes=("", "_aux"))
    if "lat_aux" in merged.columns:
        merged["lat"] = merged["lat"].combine_first(merged["lat_aux"])
        merged["lon"] = merged["lon"].combine_first(merged["lon_aux"])
        merged.drop(columns=["lat_aux", "lon_aux"], inplace=True, errors="ignore")
    merged.drop(columns=["_join_key"], inplace=True, errors="ignore")
    return merged

def backfill_coordinates(df, aux_frames, cluster_col):
    work = df.copy()
    if "lat" not in work.columns:
        work["lat"] = np.nan
    if "lon" not in work.columns:
        work["lon"] = np.nan
    join_keys = [cluster_col, "cluster_label", "hotspot_unit", "dominant_junction_name"]
    for _, aux in aux_frames:
        a = derive_lat_lon(aux.copy()).dropna(subset=["lat", "lon"], how="any")
        if a.empty:
            continue
        for key in join_keys:
            if key in work.columns and key in a.columns:
                work = fill_coords_from_lookup(work, a[[key, "lat", "lon"]].copy(), key)
    return derive_lat_lon(work)

def count_status(df, status):
    if df.empty or "validation_status" not in df.columns:
        return 0
    return int((df["validation_status"] == status).sum())


# =========================
# Mappls client
# =========================
class MapplsClient:
    def __init__(self, access_token: str, timeout: int = 25):
        self.access_token = access_token.strip()
        self.timeout = timeout
        self.session = requests.Session()

    def _access_query(self):
        if not self.access_token:
            raise RuntimeError(
                "MAPPLS_ACCESS_TOKEN is missing. The current Mappls REST docs use the static "
                "access_token query parameter for reverse geocode, routing, and distance matrix."
            )
        return {"access_token": self.access_token}

    def reverse_geocode(self, lat: float, lng: float):
        return self.session.get(
            "https://search.mappls.com/search/address/rev-geocode",
            params={"lat": float(lat), "lng": float(lng), **self._access_query()},
            timeout=self.timeout,
        )

    def _request_first_success(self, url_for_resource, resources):
        """Shared GET loop for route()/distance_matrix(). Tries each resource
        in order and returns the requests.Response for the first one that is
        both an HTTP success AND reports an 'Ok' code in the payload (Mappls
        can return HTTP 200 with an internal error code, e.g. 'NoRoute')."""
        last_resp = None
        for resource in resources:
            params = {**self._access_query(), "rtype": 0}
            if MAPPLS_REGION:
                params["region"] = MAPPLS_REGION
            resp = self.session.get(url_for_resource(resource), params=params, timeout=self.timeout)
            last_resp = resp
            if resp.status_code >= 400:
                continue
            try:
                payload = resp.json()
            except Exception:
                continue
            code = None
            if isinstance(payload, dict):
                code = payload.get("code")
                if code is None and isinstance(payload.get("results"), dict):
                    code = payload["results"].get("code")
            if code is not None and str(code).lower() != "ok":
                continue
            return resp
        return last_resp

    def route(self, start_lon, start_lat, end_lon, end_lat, prefer_live_traffic=True):
        """Returns a requests.Response (so callers can do resp.json()),
        mirroring distance_matrix(). Previously this returned a plain dict,
        called a nonexistent self.get_json(), didn't accept
        prefer_live_traffic, and referenced an unset last_error — all fixed
        by sharing _request_first_success with distance_matrix()."""
        resources = []
        if prefer_live_traffic:
            resources.extend(["route_eta", "route_traffic"])
        resources.append("route_adv")
        return self._request_first_success(
            lambda resource: (
                f"https://route.mappls.com/route/direction/{resource}/{MAPPLS_PROFILE}/"
                f"{start_lon},{start_lat};{end_lon},{end_lat}"
            ),
            resources,
        )

    def distance_matrix(self, start_lon, start_lat, end_lon, end_lat, prefer_live_traffic=True):
        resources = []
        if prefer_live_traffic:
            resources.extend(["distance_matrix_eta", "distance_matrix_traffic"])
        resources.append("distance_matrix")
        return self._request_first_success(
            lambda resource: (
                f"https://route.mappls.com/route/dm/{resource}/{MAPPLS_PROFILE}/"
                f"{start_lon},{start_lat};{end_lon},{end_lat}"
            ),
            resources,
        )


# =========================
# Auxiliary source loading
# =========================
def load_aux_frames():
    frames = []
    for path, df in load_all_existing(AUX_PATHS):
        if "st_dbscan_cluster_id" in df.columns or "cluster_id" in df.columns or "dbscan_cluster_id" in df.columns:
            frames.append((path, df))
    return frames


# =========================
# Main
# =========================
def main():
    if not MAPPLS_ACCESS_TOKEN:
        raise RuntimeError("MAPPLS_ACCESS_TOKEN is missing. Put the Mappls static key into this environment variable.")

    client = MapplsClient(MAPPLS_ACCESS_TOKEN, timeout=REQUEST_TIMEOUT)

    df, source = load_first_existing(INPUT_PATHS)
    df = df.copy()
    cluster_col = standardize_cluster_col(df)

    if "cluster_label" not in df.columns:
        df["cluster_label"] = df["hotspot_unit"] if "hotspot_unit" in df.columns else "CLUSTER::" + df[cluster_col].astype(str)
    if "cluster_label_display" not in df.columns:
        df["cluster_label_display"] = df["cluster_label"].astype(str) + " (Cluster " + df[cluster_col].astype(str) + ")"

    for col, default in [("ccs_score", 0.0), ("delay_minutes_per_vehicle", 0.0), ("risk_band", "Watch"),
                          ("recommended_action", ""), ("road_class", DEFAULT_ROAD_CLASS),
                          ("link_length_m", np.nan), ("free_flow_time_min", np.nan)]:
        if col not in df.columns:
            df[col] = default

    df["ccs_score"] = pd.to_numeric(df["ccs_score"], errors="coerce").fillna(0.0)
    df["delay_minutes_per_vehicle"] = pd.to_numeric(df["delay_minutes_per_vehicle"], errors="coerce").fillna(0.0)
    df["link_length_m"] = pd.to_numeric(df["link_length_m"], errors="coerce")
    df["free_flow_time_min"] = pd.to_numeric(df["free_flow_time_min"], errors="coerce")

    df = derive_lat_lon(df)

    aux_frames = load_aux_frames()
    if aux_frames:
        df = backfill_coordinates(df, aux_frames, cluster_col)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Dedup by cluster_col (the actual unique key), NOT by cluster_label. Multiple
    # distinct ST-DBSCAN clusters can legitimately fall back to the same
    # "POLICE_STATION::X" label when they have no junction name; that is real
    # data, not a duplicate row, and deduping on the label would silently
    # discard genuinely separate hotspots. cluster_label_display (label +
    # cluster ID) is what disambiguates them for humans reading the output.
    df = df.sort_values(
        ["ccs_score", "delay_minutes_per_vehicle"], ascending=[False, False]
    ).drop_duplicates(subset=[cluster_col], keep="first").reset_index(drop=True)

    validate_df = df.head(TOP_N).copy()

    print("Primary source:", source)
    print("Aux sources loaded:", [str(p) for p, _ in aux_frames])
    print("Top rows available:", len(validate_df))
    print("Coordinate rows available:", int(validate_df[["lat", "lon"]].dropna().shape[0]))
    print(validate_df[["cluster_label_display", "lat", "lon"]].head(10).to_string(index=False))

    rows = []
    for _, row in validate_df.iterrows():
        lat = safe_float(row.get("lat"))
        lon = safe_float(row.get("lon"))
        label = clean_text(row.get("cluster_label", ""))
        label_display = clean_text(row.get("cluster_label_display", "")) or f"{label} (Cluster {row.get(cluster_col)})"
        risk_band = clean_text(row.get("risk_band", "Watch"))
        ccs = safe_float(row.get("ccs_score", 0.0), 0.0)
        model_delay_min = safe_float(row.get("delay_minutes_per_vehicle", 0.0), 0.0)
        road_class = clean_text(row.get("road_class", "")) or DEFAULT_ROAD_CLASS
        free_flow_speed = road_speed_kmh(road_class)

        # Free-flow ETA over the SAME synthetic corridor used for the live
        # route/matrix probe below — comparing live-over-corridor against
        # free-flow-over-the-same-corridor is a same-units congestion ratio,
        # unlike comparing it against the model's incremental delay (a
        # different physical quantity measured over a different link).
        corridor_km = VALIDATION_CORRIDOR_METERS / 1000.0
        corridor_free_flow_eta_min = (corridor_km / free_flow_speed) * 60.0 if free_flow_speed > 0 else np.nan

        # The model's own delay is "extra minutes over this link's free-flow
        # time" — express it as a percentage of that same free-flow time
        # rather than dividing it by an unrelated travel time.
        existing_fft = safe_float(row.get("free_flow_time_min"), np.nan)
        link_length_m = safe_float(row.get("link_length_m"), np.nan)
        if pd.notna(existing_fft) and existing_fft > 0:
            free_flow_time_min_link = existing_fft
        elif pd.notna(link_length_m) and free_flow_speed > 0:
            free_flow_time_min_link = link_length_m * 60.0 / (free_flow_speed * 1000.0)
        else:
            free_flow_time_min_link = np.nan
        delay_pct_of_freeflow = (
            100.0 * model_delay_min / free_flow_time_min_link
            if pd.notna(free_flow_time_min_link) and free_flow_time_min_link > 0 else np.nan
        )

        row_out = {
            "cluster_id": row.get(cluster_col),
            "cluster_label": label,
            "cluster_label_display": label_display,
            "risk_band": risk_band,
            "ccs_score": ccs,
            "delay_minutes_per_vehicle_model": model_delay_min,
            "road_class": road_class,
            "free_flow_time_min_link": free_flow_time_min_link,
            "delay_pct_of_freeflow": delay_pct_of_freeflow,
            "latitude": lat,
            "longitude": lon,
            "reverse_geocode_ok": False,
            "reverse_geocode_address": "",
            "reverse_geocode_error": "",
            "route_eta_ok": False,
            "route_eta_distance_m": np.nan,
            "route_eta_duration_sec": np.nan,
            "route_eta_duration_min": np.nan,
            "route_eta_error": "",
            "route_eta_source": "",
            "distance_matrix_ok": False,
            "distance_matrix_source": "",
            "distance_matrix_distance_m": np.nan,
            "distance_matrix_duration_sec": np.nan,
            "distance_matrix_duration_min": np.nan,
            "distance_matrix_error": "",
            "corridor_free_flow_eta_min": corridor_free_flow_eta_min,
            "corridor_congestion_ratio_route": np.nan,
            "corridor_congestion_ratio_matrix": np.nan,
            "validation_status": "Not Validated",
        }

        if not valid_coords(lat, lon):
            row_out["reverse_geocode_error"] = "invalid_coordinates"
            row_out["route_eta_error"] = "invalid_coordinates"
            row_out["distance_matrix_error"] = "invalid_coordinates"
            rows.append(row_out)
            continue

        # Reverse geocode — informational location sanity-check only. It
        # confirms the coordinates resolve to a real address; it says
        # nothing about congestion, so it does NOT feed validation_status.
        try:
            rev_data = client.reverse_geocode(lat, lon).json()
            addr = parse_reverse_geocode_address(rev_data)
            row_out["reverse_geocode_ok"] = bool(addr)
            row_out["reverse_geocode_address"] = addr
            row_out["reverse_geocode_error"] = "" if addr else "empty_or_unparsed"
        except Exception as e:
            row_out["reverse_geocode_error"] = str(e)

        dest_lat, dest_lon = offset_point(lat, lon, meters=VALIDATION_CORRIDOR_METERS, bearing_deg=90.0)

        # Route ETA
        try:
            route_resp = client.route(lon, lat, dest_lon, dest_lat, prefer_live_traffic=True)
            route_data = route_resp.json()
            d_m, t_s, src = parse_route_response(route_data)
            if d_m is not None or t_s is not None:
                row_out["route_eta_ok"] = True
                row_out["route_eta_distance_m"] = d_m
                row_out["route_eta_duration_sec"] = t_s
                row_out["route_eta_duration_min"] = (t_s / 60.0) if t_s is not None and pd.notna(t_s) else np.nan
                row_out["route_eta_source"] = src
            else:
                row_out["route_eta_error"] = "unparsed_route_payload"
        except Exception as e:
            row_out["route_eta_error"] = str(e)

        # Distance matrix
        try:
            dm_data = client.distance_matrix(lon, lat, dest_lon, dest_lat, prefer_live_traffic=True).json()
            parsed = parse_matrix_any(dm_data)
            if parsed["ok"]:
                row_out["distance_matrix_ok"] = True
                row_out["distance_matrix_source"] = parsed["source"]
                row_out["distance_matrix_distance_m"] = parsed["distance_m"]
                row_out["distance_matrix_duration_sec"] = parsed["duration_sec"]
                row_out["distance_matrix_duration_min"] = (
                    parsed["duration_sec"] / 60.0
                    if parsed["duration_sec"] is not None and pd.notna(parsed["duration_sec"]) else np.nan
                )
            else:
                row_out["distance_matrix_error"] = "unparsed_matrix_payload"
        except Exception as e:
            row_out["distance_matrix_error"] = str(e)

        if pd.notna(corridor_free_flow_eta_min) and corridor_free_flow_eta_min > 0:
            if row_out["route_eta_ok"] and pd.notna(row_out["route_eta_duration_min"]):
                row_out["corridor_congestion_ratio_route"] = row_out["route_eta_duration_min"] / corridor_free_flow_eta_min
            if row_out["distance_matrix_ok"] and pd.notna(row_out["distance_matrix_duration_min"]):
                row_out["corridor_congestion_ratio_matrix"] = row_out["distance_matrix_duration_min"] / corridor_free_flow_eta_min

        # Validation status reflects only the two live road-network checks
        # (route ETA, distance matrix) — reverse geocode is excluded since it
        # validates an address, not congestion.
        congestion_checks = [row_out["route_eta_ok"], row_out["distance_matrix_ok"]]
        n_ok = sum(1 for c in congestion_checks if c)
        row_out["validation_status"] = (
            "Validated" if n_ok == len(congestion_checks)
            else "Partially Validated" if n_ok > 0
            else "Not Validated"
        )

        rows.append(row_out)

    validation_df = pd.DataFrame(rows)

    expected_validation_cols = [
        "cluster_id", "cluster_label", "cluster_label_display", "risk_band", "ccs_score",
        "delay_minutes_per_vehicle_model", "road_class", "free_flow_time_min_link", "delay_pct_of_freeflow",
        "latitude", "longitude",
        "reverse_geocode_ok", "reverse_geocode_address", "reverse_geocode_error",
        "route_eta_ok", "route_eta_distance_m", "route_eta_duration_sec",
        "route_eta_duration_min", "route_eta_error", "route_eta_source",
        "distance_matrix_ok", "distance_matrix_source", "distance_matrix_distance_m",
        "distance_matrix_duration_sec", "distance_matrix_duration_min", "distance_matrix_error",
        "corridor_free_flow_eta_min", "corridor_congestion_ratio_route", "corridor_congestion_ratio_matrix",
        "validation_status",
    ]
    for c in expected_validation_cols:
        if c not in validation_df.columns:
            validation_df[c] = np.nan

    merge_cols = [
        "cluster_id", "validation_status", "reverse_geocode_ok", "reverse_geocode_address",
        "route_eta_ok", "route_eta_duration_min", "distance_matrix_ok", "distance_matrix_duration_min",
        "corridor_congestion_ratio_route", "corridor_congestion_ratio_matrix", "delay_pct_of_freeflow",
    ]
    for c in merge_cols:
        if c not in validation_df.columns:
            validation_df[c] = np.nan

    merged = df.merge(
        validation_df[merge_cols].rename(columns={"cluster_id": cluster_col}),
        on=cluster_col, how="left",
    )

    summary = pd.DataFrame([{
        "input_source": str(source),
        "validated_hotspots": count_status(validation_df, "Validated"),
        "partially_validated_hotspots": count_status(validation_df, "Partially Validated"),
        "not_validated_hotspots": count_status(validation_df, "Not Validated"),
        "mean_route_eta_min": float(validation_df["route_eta_duration_min"].dropna().mean()) if validation_df["route_eta_duration_min"].notna().any() else np.nan,
        "mean_distance_matrix_eta_min": float(validation_df["distance_matrix_duration_min"].dropna().mean()) if validation_df["distance_matrix_duration_min"].notna().any() else np.nan,
        "mean_model_delay_min": float(validation_df["delay_minutes_per_vehicle_model"].dropna().mean()) if validation_df["delay_minutes_per_vehicle_model"].notna().any() else np.nan,
        "mean_corridor_congestion_ratio_route": float(validation_df["corridor_congestion_ratio_route"].dropna().mean()) if validation_df["corridor_congestion_ratio_route"].notna().any() else np.nan,
        "mean_corridor_congestion_ratio_matrix": float(validation_df["corridor_congestion_ratio_matrix"].dropna().mean()) if validation_df["corridor_congestion_ratio_matrix"].notna().any() else np.nan,
        "mean_delay_pct_of_freeflow": float(validation_df["delay_pct_of_freeflow"].dropna().mean()) if validation_df["delay_pct_of_freeflow"].notna().any() else np.nan,
    }])

    validation_df.to_csv(OUT_DIR / "validation_layer_top_hotspots.csv", index=False)
    merged.to_csv(OUT_DIR / "phase6_with_validation.csv", index=False)
    summary.to_csv(OUT_DIR / "validation_summary.csv", index=False)

    print("Validation layer complete")
    print("Input source:", source)
    print("Top hotspots validated:", len(validation_df))
    print("Outputs saved to:", OUT_DIR.resolve())
    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nTop validation rows:")
    show_cols = [
        "cluster_id", "cluster_label_display", "risk_band", "validation_status",
        "reverse_geocode_address", "route_eta_duration_min", "distance_matrix_duration_min",
        "corridor_congestion_ratio_matrix", "delay_minutes_per_vehicle_model", "delay_pct_of_freeflow",
        "distance_matrix_source",
    ]
    print(validation_df[[c for c in show_cols if c in validation_df.columns]].head(TOP_N).to_string(index=False))


if __name__ == "__main__":
    main()