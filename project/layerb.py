# ============================================================
# Layer B — Mappls / MapMyIndia Context Enrichment (CORRECTED)
# ============================================================
#
# ROOT CAUSES OF NaN in route_traffic_duration_min / route_optimal_duration_min:
#
# 1. WRONG AUTH METHOD for route_eta():
#    - Original code passes `access_token` as a QUERY PARAM to route.mappls.com
#    - The modern Mappls Route API uses Bearer token in the Authorization HEADER
#    - URL format is also wrong: route.mappls.com/routev2/direction/route is not
#      the correct public endpoint; use:
#      https://apis.mappls.com/advancedmaps/v1/{rest_key}/route_eta/{profile}/{coords}
#
# 2. WRONG AUTH METHOD for distance_matrix():
#    - Legacy distance-matrix endpoint uses the REST license key in the URL path
#      (which the code does), but the access_token query param doesn't work for it.
#    - Returns durations in MINUTES not seconds in the `results.durations` matrix.
#
# 3. WRONG ENDPOINT for nearby_places():
#    - search.mappls.com/search/places/nearby/json does not exist publicly.
#    - Correct endpoint: https://apis.mappls.com/advancedmaps/v1/{rest_key}/nearby
#      OR use the OAuth bearer token approach:
#      https://apis.mappls.com/places/search/json with `keywords` + `refLocation`
#
# 4. WRONG ENDPOINT for reverse_geocode():
#    - search.mappls.com/search/address/rev-geocode is not public.
#    - Correct: https://apis.mappls.com/advancedmaps/v1/{rest_key}/rev_geocode
#      (legacy, uses REST key in path)
#
# 5. route_eta() calls the SAME method twice for traffic vs optimal,
#    getting identical results. The original route API has no separate
#    traffic/optimal toggle in a simple free-tier call — so we call the
#    same endpoint and use both as the same value; ratio will be 1.0.
#    If your key supports the ETA variant (route_eta vs route), use that.
#
# SUMMARY OF CORRECTIONS:
# - reverse_geocode: use legacy path-key endpoint with REST key
# - nearby_places: use legacy nearby endpoint with REST key
# - route_eta: use Bearer auth header on route.mappls.com
# - distance_matrix: use advancedmaps/v1/{key}/distance_matrix endpoint;
#   durations returned in SECONDS for distance_matrix_eta, in MINUTES for
#   distance_matrix — normalize correctly.
# - Separate traffic vs optimal calls by using route_eta vs route resource.
# ============================================================

import os
import math
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# -------------------------
# Config
# -------------------------
PHASE5_DIRS = [
    Path("content/phase5_outputs_2"),
    Path("phase5_outputs_2"),
    Path("/content/phase5_outputs_2"),
]

OUT_DIR = Path("content/layer_b_outputs_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = int(os.environ.get("LAYER_B_TOP_N", "50"))
REQUEST_TIMEOUT = int(os.environ.get("MAPPLS_REQUEST_TIMEOUT", "25"))
REQUEST_RETRIES = int(os.environ.get("MAPPLS_REQUEST_RETRIES", "2"))
REQUEST_BACKOFF_SEC = float(os.environ.get("MAPPLS_REQUEST_BACKOFF_SEC", "0.8"))
VALIDATION_CORRIDOR_METERS = float(os.environ.get("LAYER_B_CORRIDOR_METERS", "500"))

# Bearer/OAuth access token — used in Authorization header for route.mappls.com
MAPPLS_ACCESS_TOKEN = os.environ.get("MAPPLS_ACCESS_TOKEN", "").strip()

# REST license key — used IN THE URL PATH for legacy advancedmaps/v1/{key}/... endpoints
MAPPLS_REST_KEY = os.environ.get("MAPPLS_REST_KEY", "").strip()

MAPPLS_REGION = os.environ.get("MAPPLS_REGION", "IND").strip().upper()
MAPPLS_PROFILE = os.environ.get("MAPPLS_PROFILE", "driving").strip().lower()

SENSITIVE_NEARBY_KEYWORDS = "school;hospital;bus stop;metro station;police station;junction"

# -------------------------
# Generic helpers
# -------------------------
EPS = 1e-9


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_tag(tag) -> str:
    return clean_text(tag).upper().replace("&", "AND").strip()


def load_first_existing(base_dirs: Iterable[Path], filenames: Iterable[str]):
    for d in base_dirs:
        for name in filenames:
            p = d / name
            if p.exists():
                return pd.read_csv(p, low_memory=False), p
    return None, None


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def valid_coords(lat, lon) -> bool:
    try:
        return (
            pd.notna(lat)
            and pd.notna(lon)
            and -90 <= float(lat) <= 90
            and -180 <= float(lon) <= 180
        )
    except Exception:
        return False


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


def offset_point(lat, lon, meters=500.0, bearing_deg=90.0):
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


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    mn = valid.min()
    mx = valid.max()
    return (s.fillna(mn) - mn) / (mx - mn + EPS)


def make_hotspot_key(df: pd.DataFrame) -> pd.Series:
    label = df.get("cluster_label", pd.Series("", index=df.index)).fillna("").astype(str).str.strip().str.lower()
    if "lat" in df.columns and "lon" in df.columns:
        lat = pd.to_numeric(df["lat"], errors="coerce").round(5).astype(str)
        lon = pd.to_numeric(df["lon"], errors="coerce").round(5).astype(str)
    else:
        lat = pd.Series("", index=df.index)
        lon = pd.Series("", index=df.index)
    return label + "|" + lat + "|" + lon


def get_requests_session() -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=10,
        pool_maxsize=10,
        max_retries=0,
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def request_json(
    session: requests.Session,
    method: str,
    url: str,
    headers: Optional[dict] = None,
    **kwargs,
) -> Tuple[Optional[dict], str, int]:
    last_err = ""
    last_status = 0
    for attempt in range(REQUEST_RETRIES + 1):
        try:
            resp = session.request(method, url, timeout=REQUEST_TIMEOUT, headers=headers or {}, **kwargs)
            last_status = int(resp.status_code)
            if resp.status_code == 204:
                return None, "", 204
            resp.raise_for_status()
            try:
                return resp.json(), "", resp.status_code
            except Exception:
                return {"_raw_text": resp.text}, "", resp.status_code
        except Exception as e:
            last_err = str(e)
            if attempt < REQUEST_RETRIES:
                time.sleep(REQUEST_BACKOFF_SEC * (attempt + 1))
    return None, last_err, last_status


# -------------------------
# Parsers
# -------------------------
def parse_reverse_geocode(payload: dict) -> Dict[str, Any]:
    """
    Mappls legacy rev_geocode response:
    {
      "results": {
        "houseNumber": "", "houseName": "", "poi": "", "street": "",
        "subSubLocality": "", "subLocality": "", "locality": "",
        "village": "", "subDistrict": "", "district": "",
        "city": "", "state": "", "pincode": "",
        "formattedAddress": "...",
        "eLoc": "...", "geocodeLevel": "...", "confidenceScore": 0.8
      }
    }
    Note: copResults is for the geocoding (forward) API.
    Rev-geocode uses `results` key (can be dict or list).
    """
    out = {
        "ok": False,
        "address": "",
        "houseNumber": "",
        "houseName": "",
        "poi": "",
        "street": "",
        "subSubLocality": "",
        "subLocality": "",
        "locality": "",
        "village": "",
        "subDistrict": "",
        "district": "",
        "city": "",
        "state": "",
        "pincode": "",
        "geocodeLevel": "",
        "confidenceScore": np.nan,
        "eloc": "",
    }

    if not isinstance(payload, dict):
        return out

    # rev_geocode uses "results" (can be dict or list).
    # Forward geocode uses "copResults". Handle both for safety.
    candidates = None
    if "results" in payload:
        candidates = payload["results"]
    elif "copResults" in payload:
        candidates = payload["copResults"]
    else:
        candidates = payload  # root-level fallback

    if isinstance(candidates, list) and candidates:
        first = candidates[0] if isinstance(candidates[0], dict) else {}
    elif isinstance(candidates, dict):
        first = candidates
    else:
        first = {}

    if not isinstance(first, dict):
        return out

    out["houseNumber"] = clean_text(first.get("houseNumber", ""))
    out["houseName"] = clean_text(first.get("houseName", ""))
    out["poi"] = clean_text(first.get("poi", ""))
    out["street"] = clean_text(first.get("street", ""))
    out["subSubLocality"] = clean_text(first.get("subSubLocality", first.get("subsubLocality", "")))
    out["subLocality"] = clean_text(first.get("subLocality", ""))
    out["locality"] = clean_text(first.get("locality", ""))
    out["village"] = clean_text(first.get("village", ""))
    out["subDistrict"] = clean_text(first.get("subDistrict", ""))
    out["district"] = clean_text(first.get("district", ""))
    out["city"] = clean_text(first.get("city", ""))
    out["state"] = clean_text(first.get("state", ""))
    out["pincode"] = clean_text(first.get("pincode", ""))
    out["geocodeLevel"] = clean_text(first.get("geocodeLevel", ""))
    out["confidenceScore"] = safe_float(first.get("confidenceScore"), np.nan)
    out["eloc"] = clean_text(first.get("eLoc", first.get("eloc", "")))

    formatted = (
        first.get("formattedAddress")
        or first.get("formatted_address")
        or ""
    )
    if not formatted:
        parts = [
            out["houseNumber"], out["houseName"], out["poi"], out["street"],
            out["subSubLocality"], out["subLocality"], out["locality"],
            out["village"], out["subDistrict"], out["district"],
            out["city"], out["state"], out["pincode"],
        ]
        parts = [p for p in parts if p]
        formatted = ", ".join(dict.fromkeys(parts))

    out["address"] = clean_text(formatted)
    out["ok"] = bool(out["address"])
    return out


def parse_nearby_places(payload: dict) -> Dict[str, Any]:
    """
    Mappls nearby API response:
    { "suggestedLocations": [ { "placeName": "...", "placeAddress": "...", "distance": 123, ... }, ... ] }
    """
    out = {
        "ok": False,
        "count": 0,
        "names": [],
        "addresses": [],
        "min_distance_m": np.nan,
    }
    if not isinstance(payload, dict):
        return out

    items = (
        payload.get("suggestedLocations")
        or payload.get("results")
        or []
    )
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return out

    names, addresses, distances = [], [], []
    for item in items:
        if not isinstance(item, dict):
            continue
        nm = clean_text(item.get("placeName", item.get("name", "")))
        addr = clean_text(item.get("placeAddress", item.get("address", "")))
        d = safe_float(item.get("distance"), np.nan)
        if nm:
            names.append(nm)
        if addr:
            addresses.append(addr)
        if pd.notna(d) and d >= 0:
            distances.append(float(d))

    out["count"] = len(items)
    out["names"] = names[:10]
    out["addresses"] = addresses[:10]
    out["min_distance_m"] = float(np.nanmin(distances)) if distances else np.nan
    out["ok"] = len(items) > 0
    return out


def parse_route_eta_payload(payload: dict) -> Dict[str, Any]:
    """
    Mappls Route ETA / Direction API response shapes:
    Shape 1 (advancedmaps route_eta / route):
      { "results": { "trips": [ { "duration": <seconds>, "distance": <meters>, ... } ] } }
    Shape 2 (advancedmaps route with older format):
      { "results": { "duration": <seconds>, "distance": <meters> } }
    Shape 3 (OSRM-style, used by some route.mappls.com responses):
      { "routes": [ { "duration": <seconds>, "distance": <meters>, "legs": [...] } ] }
    Shape 4 (trip-style):
      { "trip": { "summary": { "length": <km>, "time": <seconds> } } }
    """
    out = {"ok": False, "distance_m": None, "duration_sec": None, "source": "unknown"}

    if not isinstance(payload, dict):
        return out

    # Shape 1 & 2: results key
    results = payload.get("results")
    if isinstance(results, dict):
        trips = results.get("trips")
        if isinstance(trips, list) and trips:
            t = trips[0]
            dur = safe_float(t.get("duration", t.get("time")), None)
            dist = safe_float(t.get("distance", t.get("length")), None)
            if dist is not None and dist < 500:  # likely km, convert
                dist = dist * 1000.0
            if dur is not None or dist is not None:
                out.update({"ok": True, "duration_sec": dur, "distance_m": dist, "source": "results.trips"})
                return out

        dur = safe_float(results.get("duration", results.get("time")), None)
        dist = safe_float(results.get("distance", results.get("length")), None)
        if dist is not None and dist < 500:
            dist = dist * 1000.0
        if dur is not None or dist is not None:
            out.update({"ok": True, "duration_sec": dur, "distance_m": dist, "source": "results"})
            return out

    # Shape 3: OSRM-style routes array
    routes = payload.get("routes")
    if isinstance(routes, list) and routes:
        r = routes[0]
        if isinstance(r, dict):
            dur = safe_float(r.get("duration"), None)
            dist = safe_float(r.get("distance"), None)
            if dist is not None and dist < 500:
                dist = dist * 1000.0
            if dur is not None or dist is not None:
                out.update({"ok": True, "duration_sec": dur, "distance_m": dist, "source": "routes[0]"})
                return out

    # Shape 4: trip summary
    trip = payload.get("trip")
    if isinstance(trip, dict):
        summary = trip.get("summary", {})
        if isinstance(summary, dict):
            dur = safe_float(summary.get("time", summary.get("duration")), None)
            dist = safe_float(summary.get("length", summary.get("distance")), None)
            if dist is not None and dist < 500:
                dist = dist * 1000.0
            if dur is not None or dist is not None:
                out.update({"ok": True, "duration_sec": dur, "distance_m": dist, "source": "trip.summary"})
                return out

    return out


def parse_matrix_payload(payload: dict) -> Dict[str, Any]:
    """
    Mappls Distance Matrix API (advancedmaps/v1/{key}/distance_matrix/driving/{coords}):
    Response:
    {
      "results": {
        "distances": [[0, 1234], [1234, 0]],    -- in METERS
        "durations": [[0, 456], [456, 0]]         -- in SECONDS for distance_matrix_eta
      }
    }
    OR for distance_matrix (non-ETA):
    {
      "results": {
        "distances": [[0, 1.23], [1.23, 0]],    -- in KILOMETERS
        "durations": [[0, 7.6], [7.6, 0]]         -- in MINUTES
      }
    }
    We try distance_matrix_eta first (seconds), fall back to distance_matrix (minutes).
    """
    out = {"ok": False, "distance_m": None, "duration_sec": None, "source": "unknown"}

    if not isinstance(payload, dict):
        return out

    results = payload.get("results")
    if isinstance(results, list) and results:
        results = results[0]

    if isinstance(results, dict):
        dists_raw = results.get("distances")
        durs_raw = results.get("durations")

        def extract_offdiag(matrix):
            if not isinstance(matrix, list):
                return None
            vals = []
            for i, row in enumerate(matrix):
                if isinstance(row, list):
                    for j, v in enumerate(row):
                        if i != j:
                            try:
                                f = float(v)
                                if f >= 0 and np.isfinite(f):
                                    vals.append(f)
                            except Exception:
                                pass
                else:
                    try:
                        f = float(row)
                        if f >= 0 and np.isfinite(f):
                            vals.append(f)
                    except Exception:
                        pass
            return min(vals) if vals else None

        d_val = extract_offdiag(dists_raw)
        t_val = extract_offdiag(durs_raw)

        if d_val is not None or t_val is not None:
            # Normalize: if d_val is small (<500) it's likely km; convert to meters
            if d_val is not None and d_val < 500:
                d_val = d_val * 1000.0
            # Duration: if t_val is very small (<30) it's likely minutes; convert to seconds
            # If t_val > 3600 it's almost certainly seconds already
            # Heuristic: distance_matrix_eta returns seconds, distance_matrix returns minutes
            # We store the endpoint type to decide (passed via `resource` arg at call site)
            out.update({
                "ok": True,
                "distance_m": d_val,
                "duration_sec": t_val,  # caller normalizes based on resource type
                "source": "results.distances/durations",
            })
            return out

    return out


# -------------------------
# Mappls client (CORRECTED)
# -------------------------
class MapplsLayerBClient:
    """
    Key corrections vs original:
    1. reverse_geocode: use legacy advancedmaps/v1/{rest_key}/rev_geocode endpoint
    2. nearby_places: use advancedmaps/v1/{rest_key}/nearby endpoint
    3. route_eta: use Authorization Bearer header on route.mappls.com OR
       advancedmaps/v1/{rest_key}/route_eta/{profile}/{coords}
    4. distance_matrix: use advancedmaps/v1/{rest_key}/distance_matrix_eta/{profile}/{coords}
       and handle duration in seconds; fall back to distance_matrix (duration in minutes)
    """

    def __init__(self, access_token: str, rest_key: str = "", timeout: int = 25):
        self.access_token = access_token.strip()
        self.rest_key = (rest_key or access_token).strip()
        self.timeout = timeout
        self.session = get_requests_session()

    def _bearer_headers(self) -> dict:
        """For endpoints that accept OAuth Bearer token in header."""
        return {"Authorization": f"bearer {self.access_token}"}

    def reverse_geocode(self, lat: float, lng: float) -> Dict[str, Any]:
        """
        CORRECTED: Use legacy REST key endpoint.
        GET https://apis.mappls.com/advancedmaps/v1/{rest_key}/rev_geocode?lat=&lng=&region=
        Response: { "results": { "formattedAddress": "...", "city": "...", ... } }
        """
        url = f"https://apis.mappls.com/advancedmaps/v1/{self.rest_key}/rev_geocode"
        params = {
            "lat": float(lat),
            "lng": float(lng),
            "region": MAPPLS_REGION,
        }
        payload, err, status = request_json(self.session, "GET", url, params=params)
        if payload is None:
            # Try bearer auth fallback
            url2 = "https://apis.mappls.com/places/geocode"
            params2 = {"address": f"{lat},{lng}", "region": MAPPLS_REGION}
            payload, err, status = request_json(
                self.session, "GET", url2,
                headers=self._bearer_headers(),
                params=params2,
            )
        if payload is None:
            return {"ok": False, "error": err or f"HTTP_{status}", "raw": None}
        parsed = parse_reverse_geocode(payload)
        parsed["raw"] = payload
        parsed["error"] = "" if parsed["ok"] else "empty_or_unparsed"
        return parsed

    def nearby_places(
        self,
        lat: float,
        lng: float,
        keywords: str = SENSITIVE_NEARBY_KEYWORDS,
        radius: int = 1000,
    ) -> Dict[str, Any]:
        """
        CORRECTED: Use legacy REST key nearby endpoint.
        GET https://apis.mappls.com/advancedmaps/v1/{rest_key}/nearby
            ?keywords=school;hospital&refLocation=lat,lng&region=IND&radius=1000&sortBy=dist:asc
        Response: { "suggestedLocations": [ { "placeName": "", "placeAddress": "", "distance": 123 }, ... ] }
        """
        url = f"https://apis.mappls.com/advancedmaps/v1/{self.rest_key}/nearby"
        params = {
            "keywords": keywords,
            "refLocation": f"{float(lat)},{float(lng)}",
            "region": MAPPLS_REGION,
            "radius": int(radius),
            "sortBy": "dist:asc",
        }
        payload, err, status = request_json(self.session, "GET", url, params=params)
        if payload is None:
            # Fallback: bearer token-based places API
            url2 = "https://apis.mappls.com/places/nearby/json"
            params2 = {
                "keywords": keywords,
                "refLocation": f"{float(lat)},{float(lng)}",
                "region": MAPPLS_REGION,
                "radius": int(radius),
                "sortBy": "dist:asc",
            }
            payload, err, status = request_json(
                self.session, "GET", url2,
                headers=self._bearer_headers(),
                params=params2,
            )
        if payload is None:
            return {"ok": False, "error": err or f"HTTP_{status}", "raw": None}
        parsed = parse_nearby_places(payload)
        parsed["raw"] = payload
        parsed["error"] = ""
        return parsed

    def route_eta(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        resource: str = "route_eta",
    ) -> Dict[str, Any]:
        """
        CORRECTED: Mappls Route ETA API.

        Primary: advancedmaps/v1/{rest_key}/{resource}/{profile}/{lon,lat;lon,lat}
          - resource = "route_eta"  → live traffic ETA (duration in seconds)
          - resource = "route"      → free-flow / optimal (duration in seconds)
        URL: https://apis.mappls.com/advancedmaps/v1/{rest_key}/route_eta/driving/lon1,lat1;lon2,lat2

        Fallback: route.mappls.com with Bearer header
          GET https://route.mappls.com/route/driving/{lon,lat};{lon,lat}
              ?overview=false  (Authorization: bearer {token})
        Response: { "routes": [{ "duration": <sec>, "distance": <meters> }] }

        IMPORTANT: coordinates are lon,lat ORDER (not lat,lon) for this API.
        """
        # Primary: legacy advancedmaps endpoint
        coords = f"{float(start_lon)},{float(start_lat)};{float(end_lon)},{float(end_lat)}"
        url = f"https://apis.mappls.com/advancedmaps/v1/{self.rest_key}/{resource}/{MAPPLS_PROFILE}/{coords}"
        params = {"region": MAPPLS_REGION, "rtype": "0"}
        payload, err, status = request_json(self.session, "GET", url, params=params)
        if payload is not None:
            parsed = parse_route_eta_payload(payload)
            if parsed["ok"]:
                parsed["raw"] = payload
                parsed["error"] = ""
                parsed["resource"] = resource
                parsed["endpoint"] = url
                return parsed

        # Fallback: route.mappls.com with Bearer auth
        url2 = f"https://route.mappls.com/route/{MAPPLS_PROFILE}/{coords}"
        params2 = {"overview": "false", "region": MAPPLS_REGION}
        payload2, err2, status2 = request_json(
            self.session, "GET", url2,
            headers=self._bearer_headers(),
            params=params2,
        )
        if payload2 is not None:
            parsed2 = parse_route_eta_payload(payload2)
            if parsed2["ok"]:
                parsed2["raw"] = payload2
                parsed2["error"] = ""
                parsed2["resource"] = resource
                parsed2["endpoint"] = url2
                return parsed2

        return {
            "ok": False,
            "distance_m": None,
            "duration_sec": None,
            "raw": None,
            "error": err2 or err or f"HTTP_{status2 or status}",
            "resource": resource,
        }

    def distance_matrix(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
    ) -> Dict[str, Any]:
        """
        CORRECTED: Mappls Distance Matrix API.

        URL format: https://apis.mappls.com/advancedmaps/v1/{rest_key}/{resource}/{profile}/{coords}
        Coordinates: lon,lat ORDER separated by semicolons.

        resource options (tried in order):
          - distance_matrix_eta   → durations in SECONDS, distances in METERS
          - distance_matrix_traffic → durations in SECONDS, distances in METERS
          - distance_matrix       → durations in MINUTES, distances in KILOMETERS

        Response: {
          "results": {
            "distances": [[0, d01], [d10, 0]],
            "durations": [[0, t01], [t10, 0]]
          }
        }
        """
        coords = f"{float(start_lon)},{float(start_lat)};{float(end_lon)},{float(end_lat)}"

        resources_seconds = ["distance_matrix_eta", "distance_matrix_traffic"]
        resources_minutes = ["distance_matrix"]

        for resource in resources_seconds:
            url = f"https://apis.mappls.com/advancedmaps/v1/{self.rest_key}/{resource}/{MAPPLS_PROFILE}/{coords}"
            params = {"region": MAPPLS_REGION, "rtype": "0"}
            payload, err, status = request_json(self.session, "GET", url, params=params)
            if payload is None:
                continue
            parsed = parse_matrix_payload(payload)
            if parsed["ok"]:
                # duration already in seconds for _eta and _traffic
                parsed["raw"] = payload
                parsed["error"] = ""
                parsed["endpoint"] = url
                parsed["resource"] = resource
                return parsed

        for resource in resources_minutes:
            url = f"https://apis.mappls.com/advancedmaps/v1/{self.rest_key}/{resource}/{MAPPLS_PROFILE}/{coords}"
            params = {"region": MAPPLS_REGION, "rtype": "0"}
            payload, err, status = request_json(self.session, "GET", url, params=params)
            if payload is None:
                continue
            parsed = parse_matrix_payload(payload)
            if parsed["ok"]:
                # Convert: durations in MINUTES → seconds, distances in KM → meters
                if parsed["duration_sec"] is not None:
                    parsed["duration_sec"] = parsed["duration_sec"] * 60.0
                if parsed["distance_m"] is not None and parsed["distance_m"] < 500:
                    parsed["distance_m"] = parsed["distance_m"] * 1000.0
                parsed["raw"] = payload
                parsed["error"] = ""
                parsed["endpoint"] = url
                parsed["resource"] = resource
                return parsed

        return {
            "ok": False,
            "distance_m": None,
            "duration_sec": None,
            "raw": None,
            "error": "all_distance_matrix_resources_failed",
            "source": "distance_matrix",
        }


# -------------------------
# Layer B logic (unchanged from original, except for the fixes above)
# -------------------------
def load_stage5_table():
    df, source = load_first_existing(
        PHASE5_DIRS,
        [
            "phase5_cluster_capacity_loss.csv",
            "phase5_priority_table_full.csv",
            "phase5_stage5_handoff.csv",
        ],
    )
    if df is None:
        raise FileNotFoundError(
            "Could not find Stage 5 output."
        )
    return df, source


def select_hotspots_for_enrichment(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    work = df.copy()
    work = ensure_label_column(work)
    work = derive_coords(work)
    if "physical_hotspot_key" not in work.columns:
        work["physical_hotspot_key"] = make_hotspot_key(work)
    if "ccs_score" not in work.columns:
        work["ccs_score"] = 0.0
    if "delay_minutes_per_vehicle" not in work.columns:
        work["delay_minutes_per_vehicle"] = 0.0
    if "risk_band" not in work.columns:
        work["risk_band"] = "Watch"
    if "records_total" not in work.columns:
        work["records_total"] = 0.0

    work["ccs_score"] = pd.to_numeric(work["ccs_score"], errors="coerce").fillna(0.0)
    work["delay_minutes_per_vehicle"] = pd.to_numeric(work["delay_minutes_per_vehicle"], errors="coerce").fillna(0.0)
    work = work.sort_values(
        ["ccs_score", "delay_minutes_per_vehicle", "records_total"],
        ascending=[False, False, False],
    ).drop_duplicates(subset=["physical_hotspot_key"], keep="first").reset_index(drop=True)
    work = work.dropna(subset=["lat", "lon"]).copy()
    if top_n and top_n > 0:
        work = work.head(top_n).copy()
    return work.reset_index(drop=True)


def build_layer_b_geofence_candidates(hotspot_df: pd.DataFrame) -> pd.DataFrame:
    if hotspot_df is None or len(hotspot_df) == 0:
        return pd.DataFrame(columns=[
            "cluster_id", "cluster_label", "lat", "lon", "radius_m",
            "reason", "nearby_sensitive_poi_count", "nearby_sensitive_poi_names"
        ])
    rows = []
    for _, r in hotspot_df.iterrows():
        nearby_names = r.get("nearby_sensitive_poi_names", [])
        if isinstance(nearby_names, str):
            nearby_names = [x.strip() for x in nearby_names.split("||") if x.strip()]
        count = int(r.get("nearby_sensitive_poi_count", 0) or 0)
        if count <= 0:
            continue
        radius_m = 100 + min(400, 50 * count)
        rows.append({
            "cluster_id": r.get("cluster_id", r.get("st_dbscan_cluster_id", "")),
            "cluster_label": r.get("cluster_label_display", r.get("cluster_label", "")),
            "lat": safe_float(r.get("lat")),
            "lon": safe_float(r.get("lon")),
            "radius_m": radius_m,
            "reason": "Sensitive POI proximity",
            "nearby_sensitive_poi_count": count,
            "nearby_sensitive_poi_names": "||".join([str(x) for x in nearby_names[:10]]),
        })
    return pd.DataFrame(rows)


def enrich_hotspot_row(client: MapplsLayerBClient, row: pd.Series) -> Dict[str, Any]:
    lat = safe_float(row.get("lat"))
    lon = safe_float(row.get("lon"))

    out = {
        "cluster_id": row.get("cluster_id", row.get("st_dbscan_cluster_id", "")),
        "cluster_label": clean_text(row.get("cluster_label", "")),
        "cluster_label_display": clean_text(row.get("cluster_label_display", row.get("cluster_label", ""))),
        "risk_band": clean_text(row.get("risk_band", "Watch")),
        "ccs_score": safe_float(row.get("ccs_score", 0.0), 0.0),
        "delay_minutes_per_vehicle": safe_float(row.get("delay_minutes_per_vehicle", 0.0), 0.0),
        "records_total": safe_float(row.get("records_total", 0.0), 0.0),
        "lat": lat,
        "lon": lon,
        "reverse_geocode_ok": False,
        "reverse_geocode_address": "",
        "reverse_geocode_level": "",
        "reverse_geocode_poi": "",
        "reverse_geocode_street": "",
        "reverse_geocode_locality": "",
        "reverse_geocode_city": "",
        "reverse_geocode_district": "",
        "reverse_geocode_state": "",
        "nearby_sensitive_poi_ok": False,
        "nearby_sensitive_poi_count": 0,
        "nearby_sensitive_poi_names": [],
        "nearby_sensitive_poi_addresses": [],
        "nearby_sensitive_poi_min_distance_m": np.nan,
        "route_traffic_ok": False,
        "route_traffic_distance_m": np.nan,
        "route_traffic_duration_sec": np.nan,
        "route_traffic_duration_min": np.nan,
        "route_optimal_ok": False,
        "route_optimal_distance_m": np.nan,
        "route_optimal_duration_sec": np.nan,
        "route_optimal_duration_min": np.nan,
        "route_traffic_vs_optimal_ratio": np.nan,
        "distance_matrix_ok": False,
        "distance_matrix_source": "",
        "distance_matrix_distance_m": np.nan,
        "distance_matrix_duration_sec": np.nan,
        "distance_matrix_duration_min": np.nan,
        "distance_matrix_vs_route_ratio": np.nan,
        "context_multiplier": 1.0,
        "layer_b_priority_boost": 0.0,
        "layer_b_alert_flag": False,
        "layer_b_error": "",
    }

    if not valid_coords(lat, lon):
        out["layer_b_error"] = "invalid_coordinates"
        return out

    # 1. Reverse geocode
    try:
        rev = client.reverse_geocode(lat, lon)
        if rev.get("ok"):
            out["reverse_geocode_ok"] = True
            out["reverse_geocode_address"] = rev.get("address", "")
            out["reverse_geocode_level"] = rev.get("geocodeLevel", "")
            out["reverse_geocode_poi"] = rev.get("poi", "")
            out["reverse_geocode_street"] = rev.get("street", "")
            out["reverse_geocode_locality"] = rev.get("locality", "")
            out["reverse_geocode_city"] = rev.get("city", "")
            out["reverse_geocode_district"] = rev.get("district", "")
            out["reverse_geocode_state"] = rev.get("state", "")
    except Exception as e:
        out["layer_b_error"] = f"reverse_geocode: {e}"

    # 2. Nearby POIs
    try:
        nearby = client.nearby_places(lat, lon, keywords=SENSITIVE_NEARBY_KEYWORDS, radius=1000)
        if nearby.get("ok"):
            out["nearby_sensitive_poi_ok"] = True
            out["nearby_sensitive_poi_count"] = int(nearby.get("count", 0) or 0)
            out["nearby_sensitive_poi_names"] = nearby.get("names", [])
            out["nearby_sensitive_poi_addresses"] = nearby.get("addresses", [])
            out["nearby_sensitive_poi_min_distance_m"] = nearby.get("min_distance_m", np.nan)
    except Exception as e:
        out["layer_b_error"] = f"{out['layer_b_error']} | nearby: {e}".strip(" |")

    # Corridor destination for routing
    dest_lat, dest_lon = offset_point(lat, lon, meters=VALIDATION_CORRIDOR_METERS, bearing_deg=90.0)

    # 3. Route ETA — traffic (resource=route_eta uses live traffic)
    try:
        route_traffic = client.route_eta(lat, lon, dest_lat, dest_lon, resource="route_eta")
        if route_traffic.get("ok"):
            dur_sec = safe_float(route_traffic.get("duration_sec"), np.nan)
            out["route_traffic_ok"] = True
            out["route_traffic_distance_m"] = safe_float(route_traffic.get("distance_m"), np.nan)
            out["route_traffic_duration_sec"] = dur_sec
            out["route_traffic_duration_min"] = dur_sec / 60.0 if pd.notna(dur_sec) else np.nan
    except Exception as e:
        out["layer_b_error"] = f"{out['layer_b_error']} | route_traffic: {e}".strip(" |")

    # 4. Route — optimal/free-flow (resource=route, no traffic)
    try:
        route_opt = client.route_eta(lat, lon, dest_lat, dest_lon, resource="route")
        if route_opt.get("ok"):
            dur_sec = safe_float(route_opt.get("duration_sec"), np.nan)
            out["route_optimal_ok"] = True
            out["route_optimal_distance_m"] = safe_float(route_opt.get("distance_m"), np.nan)
            out["route_optimal_duration_sec"] = dur_sec
            out["route_optimal_duration_min"] = dur_sec / 60.0 if pd.notna(dur_sec) else np.nan
    except Exception as e:
        out["layer_b_error"] = f"{out['layer_b_error']} | route_optimal: {e}".strip(" |")

    # Traffic vs optimal ratio
    t_dur = safe_float(out["route_traffic_duration_sec"], np.nan)
    o_dur = safe_float(out["route_optimal_duration_sec"], np.nan)
    if pd.notna(t_dur) and pd.notna(o_dur) and o_dur > 0:
        out["route_traffic_vs_optimal_ratio"] = t_dur / max(o_dur, EPS)

    # 5. Distance matrix
    try:
        dm = client.distance_matrix(lat, lon, dest_lat, dest_lon)
        if dm.get("ok"):
            dur_sec = safe_float(dm.get("duration_sec"), np.nan)
            out["distance_matrix_ok"] = True
            out["distance_matrix_source"] = dm.get("resource", dm.get("source", ""))
            out["distance_matrix_distance_m"] = safe_float(dm.get("distance_m"), np.nan)
            out["distance_matrix_duration_sec"] = dur_sec
            out["distance_matrix_duration_min"] = dur_sec / 60.0 if pd.notna(dur_sec) else np.nan
    except Exception as e:
        out["layer_b_error"] = f"{out['layer_b_error']} | distance_matrix: {e}".strip(" |")

    dm_dur = safe_float(out["distance_matrix_duration_sec"], np.nan)
    rt_dur = safe_float(out["route_traffic_duration_sec"], np.nan)
    if pd.notna(dm_dur) and pd.notna(rt_dur) and rt_dur > 0:
        out["distance_matrix_vs_route_ratio"] = dm_dur / max(rt_dur, EPS)

    # Context multiplier
    poi_count = int(out["nearby_sensitive_poi_count"] or 0)
    traffic_ratio = safe_float(out["route_traffic_vs_optimal_ratio"], np.nan)
    poi_boost = min(0.50, 0.08 * poi_count)
    traffic_boost = 0.0
    if pd.notna(traffic_ratio) and traffic_ratio > 1.0:
        traffic_boost = min(0.50, 0.25 * (traffic_ratio - 1.0))

    out["layer_b_priority_boost"] = poi_boost + traffic_boost
    out["context_multiplier"] = float(np.clip(1.0 + out["layer_b_priority_boost"], 1.0, 2.0))
    out["layer_b_alert_flag"] = bool(
        (poi_count > 0) or (pd.notna(traffic_ratio) and traffic_ratio >= 1.15)
    )

    return out


def main():
    if not MAPPLS_REST_KEY:
        raise RuntimeError("MAPPLS_REST_KEY is missing. Set your Mappls REST license key.")

    client = MapplsLayerBClient(
        access_token=MAPPLS_ACCESS_TOKEN,
        rest_key=MAPPLS_REST_KEY,
        timeout=REQUEST_TIMEOUT,
    )

    stage5_df, stage5_src = load_stage5_table()
    stage5_df = stage5_df.copy()
    stage5_df = ensure_label_column(stage5_df)
    stage5_df = derive_coords(stage5_df)
    hotspots = select_hotspots_for_enrichment(stage5_df, top_n=TOP_N)

    print("Stage 5 source:", stage5_src)
    print("Hotspots selected for Layer B:", len(hotspots))

    enriched_rows = []
    for i, row in enumerate(hotspots.itertuples(index=False), start=1):
        row_series = pd.Series(row._asdict())
        enriched = enrich_hotspot_row(client, row_series)
        enriched_rows.append(enriched)
        if i % 5 == 0 or i == len(hotspots):
            print(f"Processed {i}/{len(hotspots)} hotspots")

    layer_b_df = pd.DataFrame(enriched_rows)

    if len(layer_b_df):
        layer_b_df["cluster_label_display"] = (
            layer_b_df["cluster_label"].astype(str)
            + " (Cluster "
            + layer_b_df["cluster_id"].astype(str)
            + ")"
        )
        layer_b_df["nearby_sensitive_poi_names_str"] = layer_b_df["nearby_sensitive_poi_names"].apply(
            lambda xs: " || ".join(xs[:10]) if isinstance(xs, list) else clean_text(xs)
        )
        layer_b_df["nearby_sensitive_poi_addresses_str"] = layer_b_df["nearby_sensitive_poi_addresses"].apply(
            lambda xs: " || ".join(xs[:10]) if isinstance(xs, list) else clean_text(xs)
        )

    geofence_candidates = build_layer_b_geofence_candidates(layer_b_df)

    merge_cols = [
        "cluster_id",
        "reverse_geocode_ok", "reverse_geocode_address", "reverse_geocode_level",
        "reverse_geocode_poi", "reverse_geocode_street", "reverse_geocode_locality",
        "reverse_geocode_city", "reverse_geocode_district", "reverse_geocode_state",
        "nearby_sensitive_poi_ok", "nearby_sensitive_poi_count",
        "nearby_sensitive_poi_names", "nearby_sensitive_poi_addresses",
        "nearby_sensitive_poi_min_distance_m",
        "route_traffic_ok", "route_traffic_distance_m",
        "route_traffic_duration_sec", "route_traffic_duration_min",
        "route_optimal_ok", "route_optimal_distance_m",
        "route_optimal_duration_sec", "route_optimal_duration_min",
        "route_traffic_vs_optimal_ratio",
        "distance_matrix_ok", "distance_matrix_source",
        "distance_matrix_distance_m", "distance_matrix_duration_sec",
        "distance_matrix_duration_min", "distance_matrix_vs_route_ratio",
        "context_multiplier", "layer_b_priority_boost",
        "layer_b_alert_flag", "layer_b_error",
    ]
    merge_cols = [c for c in merge_cols if c in layer_b_df.columns]

    stage5_with_b = stage5_df.merge(
        layer_b_df[merge_cols].rename(columns={"cluster_id": standardize_cluster_col(stage5_df)}),
        on=standardize_cluster_col(stage5_df),
        how="left",
    )

    for col, default in [("context_multiplier", 1.0), ("layer_b_priority_boost", 0.0), ("layer_b_alert_flag", False)]:
        if col not in stage5_with_b.columns:
            stage5_with_b[col] = default

    summary = pd.DataFrame([{
        "input_source": str(stage5_src),
        "hotspots_enriched": int(len(layer_b_df)),
        "reverse_geocoded": int(layer_b_df["reverse_geocode_ok"].sum()) if len(layer_b_df) else 0,
        "nearby_context_hits": int(layer_b_df["nearby_sensitive_poi_ok"].sum()) if len(layer_b_df) else 0,
        "traffic_routes_ok": int(layer_b_df["route_traffic_ok"].sum()) if len(layer_b_df) else 0,
        "distance_matrix_ok": int(layer_b_df["distance_matrix_ok"].sum()) if len(layer_b_df) else 0,
        "mean_route_traffic_min": float(
            pd.to_numeric(layer_b_df["route_traffic_duration_min"], errors="coerce").dropna().mean()
        ) if len(layer_b_df) else np.nan,
        "mean_route_optimal_min": float(
            pd.to_numeric(layer_b_df["route_optimal_duration_min"], errors="coerce").dropna().mean()
        ) if len(layer_b_df) else np.nan,
        "mean_context_multiplier": float(
            pd.to_numeric(layer_b_df["context_multiplier"], errors="coerce").dropna().mean()
        ) if len(layer_b_df) else np.nan,
        "alerts_flagged": int(layer_b_df["layer_b_alert_flag"].sum()) if len(layer_b_df) else 0,
    }])

    layer_b_df.to_csv(OUT_DIR / "layer_b_mappls_enriched_hotspots.csv", index=False)
    geofence_candidates.to_csv(OUT_DIR / "layer_b_geofence_candidates.csv", index=False)
    stage5_with_b.to_csv(OUT_DIR / "phase5_with_layer_b.csv", index=False)
    summary.to_csv(OUT_DIR / "layer_b_summary.csv", index=False)

    print("Layer B complete")
    print("Output directory:", OUT_DIR.resolve())
    print("\nSummary:")
    print(summary.to_string(index=False))

    show_cols = [
        "cluster_id", "cluster_label_display", "reverse_geocode_address",
        "nearby_sensitive_poi_count", "route_traffic_duration_min",
        "route_optimal_duration_min", "distance_matrix_duration_min",
        "route_traffic_vs_optimal_ratio", "context_multiplier",
        "layer_b_priority_boost", "layer_b_alert_flag",
    ]
    show_cols = [c for c in show_cols if c in layer_b_df.columns]
    if len(layer_b_df) and show_cols:
        print("\nTop 10 Layer B rows:")
        print(layer_b_df[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()