from __future__ import annotations
import json, os, re, sys, math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("train-assistant")

try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(ROOT / ".env")
except Exception:
    pass

SERPAPI_KEY = os.getenv("SERPAPI_KEY") or os.getenv("SERP_API_KEY")
if not SERPAPI_KEY:
    print("[MCP train] Missing SERPAPI_KEY", file=sys.stderr)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data")).resolve()
TRAIN_DIR = (DATA_ROOT / "trains"); TRAIN_DIR.mkdir(parents=True, exist_ok=True)
REQ_TIMEOUT = 35

# ------------------ Nominatim geocoding (no hardcoded coords) -----------------
NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
GEOCODER_UA   = os.getenv("GEOCODER_UA", "train-assistant/1.0 (set GEOCODER_UA with your contact)")
GEO_CACHE_PATH = DATA_ROOT / "geo_cache_trains.json"
try:
    _GEO_CACHE: Dict[str, Dict[str, float]] = json.loads(GEO_CACHE_PATH.read_text("utf-8"))
    if not isinstance(_GEO_CACHE, dict):
        _GEO_CACHE = {}
except Exception:
    _GEO_CACHE = {}

def _geo_cache_get(name: str) -> Optional[tuple[float, float]]:
    key = (name or "").strip().lower()
    rec = _GEO_CACHE.get(key)
    if isinstance(rec, dict) and "lat" in rec and "lon" in rec:
        try:
            return float(rec["lat"]), float(rec["lon"])
        except Exception:
            return None
    return None

def _geo_cache_put(name: str, lat: float, lon: float) -> None:
    key = (name or "").strip().lower()
    _GEO_CACHE[key] = {"lat": float(lat), "lon": float(lon)}
    try:
        GEO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        GEO_CACHE_PATH.write_text(json.dumps(_GEO_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _geocode_city(name: str) -> Optional[tuple[float, float]]:
    if not name:
        return None
    cached = _geo_cache_get(name)
    if cached:
        return cached
    try:
        params = {
            "q": name,
            "format": "json",
            "limit": 1,
        }
        headers = {"User-Agent": GEOCODER_UA, "Accept-Language": "en"}
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            item = data[0]
            lat = float(item.get("lat"))
            lon = float(item.get("lon"))
            _geo_cache_put(name, lat, lon)
            return (lat, lon)
    except Exception:
        pass
    return None

def _haversine_km(a: tuple[float,float], b: tuple[float,float]) -> float:
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def _city_km(src: str, dst: str) -> Optional[float]:
    s = _geocode_city(src)
    d = _geocode_city(dst)
    if not s or not d: return None
    return _haversine_km(s, d)

# ------------------ Regex (hardened) ------------------------------------------
RE_RUPEES       = re.compile(r"(?:₹|INR|Rs\.?)\s*([0-9][0-9,]*)", re.IGNORECASE)
RE_RUPEES_RANGE = re.compile(r"(?:₹|INR|Rs\.?)\s*([0-9][0-9,]*)\s*[-–]\s*(?:₹|INR|Rs\.?)\s*([0-9][0-9,]*)", re.IGNORECASE)
RE_RUPEES_K     = re.compile(r"(?:₹|INR|Rs\.?)\s*([0-9]+(?:\.[0-9]+)?)\s*[kK]\b", re.IGNORECASE)

RE_H_M       = re.compile(r"\b(\d{1,2})\s*h(?:ours?)?\s*(\d{1,2})\s*m(?:in(?:s)?)?\b", re.IGNORECASE)
RE_HRS_MINS  = re.compile(r"\b(\d{1,2})\s*hrs?\s*(\d{1,2})\s*mins?\b", re.IGNORECASE)
RE_HH_MM     = re.compile(r"\b(\d{1,2})\s*[:h]\s*(\d{2})\b")
RE_HOURS     = re.compile(r"\b(\d{1,2}(?:\.\d{1,2})?)\s*(?:hrs?|hours?)\b", re.IGNORECASE)
RE_RANGE_H   = re.compile(r"\b(\d{1,2})\s*[-–]\s*\d{1,2}\s*(?:hrs?|hours?)\b", re.IGNORECASE)
RE_MIN_ONLY  = re.compile(r"\b(\d{2,3})\s*(?:m|min|mins|minute|minutes)\b", re.IGNORECASE)

def _safe(s: str | None) -> str:
    if not s: return ""
    s = re.sub(r'[:<>"/\\|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:100]

def _slug(s: str | None) -> str:
    if not s: return ""
    return re.sub(r"\W+", "-", s).strip("-").lower()

def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _search_id(src: str, dst: str, date: Optional[str]) -> str:
    return f"{_safe(src)}_{_safe(dst)}_{_safe(date) or 'NA'}_{_ts()}"

def _save_json(folder: Path, key: str, data: Dict[str, Any]) -> Path:
    p = folder / f"{key}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

def _to_int(s: Optional[str]) -> Optional[int]:
    if not s: return None
    try: return int(s.replace(",", ""))
    except Exception: return None

def _parse_price(text: str) -> Optional[int]:
    t = text or ""
    mr = RE_RUPEES_RANGE.search(t)
    if mr:
        lo = _to_int(mr.group(1))
        if lo: return lo
    mk = RE_RUPEES_K.search(t)
    if mk:
        try:
            return int(round(float(mk.group(1)) * 1000))
        except Exception:
            pass
    m = RE_RUPEES.search(t)
    if m:
        return _to_int(m.group(1))
    return None

def _parse_duration(text: str) -> Optional[float]:
    t = text or ""
    for rx in (RE_H_M, RE_HRS_MINS):
        m = rx.search(t)
        if m:
            try:
                h = int(m.group(1)); mm = int(m.group(2))
                return round(h + mm/60.0, 2)
            except Exception:
                pass
    m = RE_HH_MM.search(t)
    if m:
        try:
            return round(int(m.group(1)) + int(m.group(2))/60.0, 2)
        except Exception:
            pass
    mr = RE_RANGE_H.search(t)
    if mr:
        try:
            return float(mr.group(1))
        except Exception:
            pass
    mh = RE_HOURS.search(t)
    if mh:
        try:
            return float(mh.group(1))
        except Exception:
            pass
    mm = RE_MIN_ONLY.search(t)
    if mm:
        try:
            return round(int(mm.group(1)) / 60.0, 2)
        except Exception:
            pass
    return None

def _serpapi_search(q: str, gl="in", hl="en", num=20) -> Dict[str, Any]:
    params = {"engine": "google", "q": q, "gl": gl, "hl": hl, "num": num, "api_key": SERPAPI_KEY}
    r = requests.get("https://serpapi.com/search", params=params, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()

# ------------------ Structured API (optional fast path from the repo idea) ----
RAIL_API_URL = os.getenv("RAIL_API_URL", "https://railwayapi.amithv.xyz")
RAIL_API_KEY = os.getenv("RAIL_API_KEY")  # if absent, we skip structured path

def _rail_post(path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not RAIL_API_KEY:
        return None
    try:
        r = requests.post(
            f"{RAIL_API_URL}{path}",
            json=payload,
            headers={"Content-Type": "application/json", "X-API-KEY": RAIL_API_KEY},
            timeout=REQ_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _parse_date_yyyy_mm_dd(date_str: Optional[str]) -> datetime:
    # Accept "YYYY-MM-DD" else use UTC today
    if date_str:
        try:
            return datetime.fromisoformat(date_str)
        except Exception:
            pass
    return datetime.utcnow()

def _compose_iso(date_dt: datetime, time_hh_mm: Optional[str]) -> str:
    try:
        hh, mm = (time_hh_mm or "09:00").split(":")[:2]
        dt = datetime(year=date_dt.year, month=date_dt.month, day=date_dt.day,
                      hour=int(hh), minute=int(mm))
        return dt.isoformat()
    except Exception:
        return date_dt.replace(hour=9, minute=0).isoformat()

def _add_hours_iso(depart_iso: str, hours: float) -> str:
    try:
        dt = datetime.fromisoformat(depart_iso)
        return (dt + timedelta(hours=float(hours))).isoformat()
    except Exception:
        return depart_iso
    
TRAIN_PREFER = (
    "ixigo.com/trains", "trainman.in", "confirmtkt.com",
    "railyatri.in", "etrain.info", "yatra.com/trains", "paytm.com/train-tickets", "irctc.co.in"
)

def _looks_preferred(url: str | None) -> bool:
    if not url: return False
    u = url.lower()
    return any(dom in u for dom in TRAIN_PREFER)

def _extract_from_item(item: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    url = item.get("link") or ""
    title = item.get("title") or ""
    snippet = item.get("snippet") or ""
    if isinstance(snippet, list): snippet = " ".join(snippet)
    text = f"{title} {snippet}"

    dur = _parse_duration(text)
    price = _parse_price(text)

    for sl_key in ("sitelinks", "inline_sitelinks"):
        sl = item.get(sl_key) or {}
        links = sl.get("links") if isinstance(sl, dict) else sl
        if isinstance(links, list):
            for s in links:
                desc = s.get("snippet") or s.get("desc") or s.get("title") or ""
                if not dur:
                    d2 = _parse_duration(desc);  dur = d2 if d2 else dur
                if not price:
                    p2 = _parse_price(desc);     price = p2 if p2 else price

    rs = item.get("rich_snippet") or {}
    if isinstance(rs, dict):
        top = rs.get("top") or {}
        if isinstance(top, dict):
            exts = top.get("extensions") or []
            if isinstance(exts, list):
                joined = " ".join(map(str, exts))
                if not price:
                    p3 = _parse_price(joined);   price = p3 if p3 else price
                if not dur:
                    d3 = _parse_duration(joined); dur = d3 if d3 else dur

    return {"url": url, "dur": dur, "price": price, "preferred": _looks_preferred(url)}

def _estimate_train_duration(dist_km: Optional[float]) -> float:
    if not dist_km:
        return 18.0
    avg_speed = 70.0 if dist_km >= 300 else 58.0
    est = dist_km / avg_speed
    buf = 0.5 if dist_km < 200 else (1.0 if dist_km < 600 else 1.5)
    return round(max(3.0, min(55.0, est + buf)), 2)

def _valid_train_duration(hours: Optional[float], dist_km: Optional[float]) -> bool:
    if hours is None:
        return False
    if dist_km:
        avg = dist_km / max(hours, 0.1)
        if dist_km >= 300 and (avg > 110 or avg < 25):
            return False
    return 3.0 <= hours <= 60.0

def _build_options(
    organic: List[Dict[str, Any]],
    src_city: str,
    dst_city: str,
    base_dep: datetime,
    la: Optional[datetime],
    dist_km: Optional[float],
    max_results: int,
    debug: List[str],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for it in organic:
        parsed = _extract_from_item(it)
        u, dur, price, pref = parsed["url"], parsed["dur"], parsed["price"], parsed["preferred"]
        if not u:
            continue
        if dur is None:
            dur = _estimate_train_duration(dist_km)
        items.append({"url": u, "dur": dur, "price": price, "preferred": pref})

    items.sort(key=lambda z: (not z["preferred"], z["dur"] is None, z["price"] is None, z.get("price", 1e9)))

    options: List[Dict[str, Any]] = []
    kept = 0
    for i, z in enumerate(items, 1):
        dur = z["dur"]
        if not _valid_train_duration(dur, dist_km):
            debug.append(f"drop_invalid_dur:{dur} url:{z['url']}")
            continue
        depart = base_dep + timedelta(hours=3*(kept))
        arrive = depart + timedelta(hours=dur or 18.0)
        if la and arrive > la:
            debug.append(f"drop_by_la: {z['url']}")
            continue
        options.append({
            "mode": "train",
            "src": src_city, "dst": dst_city,
            "depart": depart.isoformat(),
            "arrive": arrive.isoformat(),
            "duration_hours": float(dur) if dur is not None else None,
            "price_inr": z["price"],
            "ref": f"T{i:02d}",
            "provider": {"name": "SerpAPI GoogleSearch", "link": z["url"]},
        })
        kept += 1
        if kept >= max_results:
            break
    return options

@mcp.tool()
def search_trains(
    src_city: str,
    dst_city: str,
    travel_date: Optional[str] = None,
    earliest_departure: Optional[str] = None,
    latest_arrival: Optional[str] = None,
    max_results: int = 12,
) -> str:
    """
    Multi-pass Google search via SerpAPI; structured-data fast path if configured.
    Writes debug trail in the snapshot.
    """
    debug: List[str] = []
    try:
        if not SERPAPI_KEY:
            return json.dumps({"error": "Missing SERPAPI_KEY"})
        try:
            base_dep = datetime.fromisoformat(earliest_departure) if earliest_departure else None
        except Exception:
            base_dep = None
        if not base_dep:
            base_dep = datetime.utcnow() + timedelta(hours=12)

        la = None
        try:
            la = datetime.fromisoformat(latest_arrival) if latest_arrival else None
        except Exception:
            la = None

        dist_km = _city_km(src_city, dst_city)

        if RAIL_API_KEY:
            dt = _parse_date_yyyy_mm_dd(travel_date)
            body = {"from_station": src_city, "to_station": dst_city}
            body["date"] = dt.strftime("%Y%m%d")

            data = _rail_post("/search-trains", body)
            if data and isinstance(data.get("trains"), list):
                options: List[Dict[str, Any]] = []
                for i, t in enumerate(data["trains"], 1):
                    dep_t = t.get("departureTime") or "09:00"
                    dur_txt = t.get("duration") or ""
                    dur_h  = _parse_duration(dur_txt) or _estimate_train_duration(dist_km)
                    depart_iso = _compose_iso(dt, dep_t)
                    arrive_iso = _add_hours_iso(depart_iso, dur_h)

                    options.append({
                        "mode": "train",
                        "src": src_city, "dst": dst_city,
                        "depart": depart_iso,
                        "arrive": arrive_iso,
                        "duration_hours": float(dur_h),
                        "price_inr": None, 
                        "ref": f"T{i:02d}",
                        "provider": {
                            "name": "IndianRail API",
                            "link": f"https://www.ixigo.com/trains/{_slug(src_city)}-to-{_slug(dst_city)}",
                        },
                    })
                    if len(options) >= max_results:
                        break

                sid = _search_id(src_city, dst_city, travel_date)
                snapshot = {
                    "search_metadata": {
                        "search_id": sid,
                        "src": src_city, "dst": dst_city,
                        "travel_date": travel_date,
                        "earliest_departure": earliest_departure,
                        "latest_arrival": latest_arrival,
                        "timestamp_utc": datetime.utcnow().isoformat(),
                        "query_primary": "structured_api",
                        "distance_km": dist_km,
                    },
                    "options": options,
                    "debug": ["used_structured_api"],
                }
                _save_json(TRAIN_DIR, sid, snapshot)
                return json.dumps({"search_id": sid, "count": len(options)})

        q1 = (f"{src_city} to {dst_city} trains time table fare duration "
              f"site:ixigo.com/trains OR site:trainman.in OR site:confirmtkt.com OR site:railyatri.in OR "
              f"site:etrain.info OR site:yatra.com/trains OR site:paytm.com/train-tickets OR site:irctc.co.in")
        if travel_date: q1 += f" {travel_date}"
        raw1 = _serpapi_search(q1, gl="in", hl="en", num=20)
        organic1 = raw1.get("organic_results") or raw1.get("organic") or []
        debug.append(f"pass1_organic: {len(organic1)}")

        options = _build_options(organic1, src_city, dst_city, base_dep, la, dist_km, max_results, debug)

        if len(options) < max_results // 2:
            q2 = f"{src_city} to {dst_city} trains schedule price duration"
            if travel_date: q2 += f" {travel_date}"
            raw2 = _serpapi_search(q2, gl="in", hl="en", num=20)
            organic2 = raw2.get("organic_results") or raw2.get("organic") or []
            debug.append(f"pass2_organic: {len(organic2)}")
            more = _build_options(organic2, src_city, dst_city, base_dep, la, dist_km, max_results - len(options), debug)
            options.extend(more)

        if not options and la:
            debug.append("no_options_due_to_latest_arrival; relaxing LA constraint for 2 items")
            options = _build_options(organic1, src_city, dst_city, base_dep, None, dist_km, 2, debug)

        if not options:
            est_h = _estimate_train_duration(dist_km)
            est_price = int(max(150, min(5000, round((dist_km or 800) * 0.9))))
            ixigo_url = f"https://www.ixigo.com/trains/{_slug(src_city)}-to-{_slug(dst_city)}"
            for j in range(1, 3+1):
                d = base_dep + timedelta(hours=3*(j-1))
                a = d + timedelta(hours=est_h + (j-1)*0.75)
                if la and a > la: continue
                options.append({
                    "mode": "train",
                    "src": src_city, "dst": dst_city,
                    "depart": d.isoformat(),
                    "arrive": a.isoformat(),
                    "duration_hours": float(est_h + (j-1)*0.75),
                    "price_inr": est_price,
                    "ref": f"T_FALLBACK_{j}",
                    "provider": {"name": "SerpAPI GoogleSearch", "link": ixigo_url},
                })
            debug.append("used_fallback_generator")

        sid = _search_id(src_city, dst_city, travel_date)
        snapshot = {
            "search_metadata": {
                "search_id": sid,
                "src": src_city, "dst": dst_city,
                "travel_date": travel_date,
                "earliest_departure": earliest_departure,
                "latest_arrival": latest_arrival,
                "timestamp_utc": datetime.utcnow().isoformat(),
                "query_primary": q1,
                "distance_km": dist_km,
            },
            "options": options,
            "debug": debug,
        }
        _save_json(TRAIN_DIR, sid, snapshot)
        return json.dumps({"search_id": sid, "count": len(options)})
    except Exception as e:
        debug.append(f"exception:{e}")
        return json.dumps({"error": str(e)})

@mcp.tool()
def get_train_details(search_id: str) -> str:
    p = TRAIN_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    return p.read_text(encoding="utf-8")

@mcp.tool()
def normalize_trains(search_id: str) -> str:
    p = TRAIN_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    data = json.loads(p.read_text(encoding="utf-8"))
    return json.dumps(data.get("options") or [])

@mcp.resource("trains://searches")
def list_train_searches() -> str:
    lines = ["# Train Searches", ""]
    for f in sorted(TRAIN_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            meta = data.get("search_metadata", {})
            lines += [
                f"## {f.stem}",
                f"- route: {meta.get('src')} → {meta.get('dst')}",
                f"- date: {meta.get('travel_date')}",
                f"- found: {len(data.get('options', []))}",
                f"- when: {meta.get('timestamp_utc')}",
                "",
                "---",
                ""
            ]
        except Exception:
            continue
    if len(lines) == 2:
        lines += ["No searches yet."]
    return "\n".join(lines)

if __name__ == "__main__":
    mcp.run(transport="stdio")
