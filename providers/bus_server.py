# providers/bus_server.py
from __future__ import annotations
import json, os, re, sys, math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("bus-assistant")

try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(ROOT / ".env")
except Exception:
    pass

SERPAPI_KEY = os.getenv("SERPAPI_KEY") or os.getenv("SERP_API_KEY")
if not SERPAPI_KEY:
    print("[MCP bus] Missing SERPAPI_KEY", file=sys.stderr)
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data")).resolve()
BUS_DIR = (DATA_ROOT / "buses"); BUS_DIR.mkdir(parents=True, exist_ok=True)
REQ_TIMEOUT = 35
RE_RUPEES = re.compile(r"(?:₹|INR)\s*([0-9][0-9,]*)", re.IGNORECASE)
RE_HOURS  = re.compile(r"\b(\d{1,2}(?:\.\d{1,2})?)\s*(?:hrs?|hours?)\b", re.IGNORECASE)
RE_HH_MM  = re.compile(r"\b(\d{1,2})\s*[:h]\s*(\d{2})\b")

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
    m = RE_RUPEES.search(text or "")
    return _to_int(m.group(1)) if m else None

def _parse_duration(text: str) -> Optional[float]:
    t = text or ""
    m = RE_HOURS.search(t)
    if m:
        try: return float(m.group(1))
        except Exception: pass
    m2 = RE_HH_MM.search(t)
    if m2:
        try:
            h = int(m2.group(1)); mm = int(m2.group(2))
            return round(h + mm/60.0, 2)
        except Exception:
            pass
    return None

def _serpapi_search(q: str, gl="in", hl="en", num=20) -> Dict[str, Any]:
    params = {"engine": "google", "q": q, "gl": gl, "hl": hl, "num": num, "api_key": SERPAPI_KEY}
    r = requests.get("https://serpapi.com/search", params=params, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()

_COORDS = {
    "indore": (22.7196, 75.8577), "bengaluru": (12.9716, 77.5946), "bangalore": (12.9716, 77.5946),
    "mumbai": (19.0760, 72.8777), "delhi": (28.6139, 77.2090), "pune": (18.5204, 73.8567),
    "hyderabad": (17.3850, 78.4867), "chennai": (13.0827, 80.2707), "ahmedabad": (23.0225, 72.5714),
    "jaipur": (26.9124, 75.7873), "lucknow": (26.8467, 80.9462), "kolkata": (22.5726, 88.3639),
    "goa": (15.2993, 74.1240), "surat": (21.1702, 72.8311), "nagpur": (21.1458, 79.0882),
    "kochi": (9.9312, 76.2673), "coimbatore": (11.0168, 76.9558), "bhopal": (23.2599, 77.4126),
    "noida": (28.5355, 77.3910), "gurgaon": (28.4595, 77.0266), "gurugram": (28.4595, 77.0266),
}

def _haversine_km(a: tuple[float,float], b: tuple[float,float]) -> float:
    R = 6371.0
    import math
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def _city_km(src: str, dst: str) -> Optional[float]:
    s = _COORDS.get((src or "").strip().lower())
    d = _COORDS.get((dst or "").strip().lower())
    if not s or not d: return None
    return _haversine_km(s, d)

BUS_PREFER = (
    "redbus.in", "abhibus.com", "intrcity.com", "zingbus.com",
    "paytm.com/bus-tickets", "makemytrip.com/bus-tickets", "yatra.com/bus",
    "ksrtc.in", "msrtc.maharashtra.gov.in", "gsrtc.in"
)

def _looks_preferred(url: str | None) -> bool:
    if not url: return False
    u = url.lower()
    return any(dom in u for dom in BUS_PREFER)

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
                desc = s.get("snippet") or s.get("desc") or ""
                if not dur:   dur = _parse_duration(desc)
                if not price: price = _parse_price(desc)

    rs = item.get("rich_snippet") or {}
    if isinstance(rs, dict):
        top = rs.get("top") or {}
        if isinstance(top, dict):
            exts = top.get("extensions") or []
            if isinstance(exts, list):
                joined = " ".join(map(str, exts))
                if not price: price = _parse_price(joined)

    return {"url": url, "dur": dur, "price": price, "preferred": _looks_preferred(url)}

def _valid_bus_duration(hours: Optional[float], dist_km: Optional[float]) -> bool:
    if hours is None: return False
    if hours < 6.0 or hours > 60.0: return False
    if dist_km:
        if hours < (dist_km / 70.0 - 1.0):
            return False
    return True

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
        items.append({"url": u, "dur": dur, "price": price, "preferred": pref})

    items.sort(key=lambda z: (not z["preferred"], z["dur"] is None))

    options: List[Dict[str, Any]] = []
    kept = 0
    for i, z in enumerate(items, 1):
        dur = z["dur"]
        if dur is None and dist_km is not None:
            dur = max(8.0, min(44.0, round(dist_km / 52.0, 1)))
        if not _valid_bus_duration(dur, dist_km):
            continue
        depart = base_dep + timedelta(hours=2*(kept))
        arrive = depart + timedelta(hours=dur or 24.0)
        if la and arrive > la:
            debug.append(f"drop_by_la: {z['url']}")
            continue
        options.append({
            "mode": "bus",
            "src": src_city, "dst": dst_city,
            "depart": depart.isoformat(),
            "arrive": arrive.isoformat(),
            "duration_hours": float(dur) if dur is not None else None,
            "price_inr": z["price"],
            "ref": f"B{i:02d}",
            "provider": {"name": "SerpAPI GoogleSearch", "link": z["url"]},
        })
        kept += 1
        if kept >= max_results:
            break
    return options

@mcp.tool()
def search_buses(
    src_city: str,
    dst_city: str,
    travel_date: Optional[str] = None,
    earliest_departure: Optional[str] = None,
    latest_arrival: Optional[str] = None,
    max_results: int = 12,
) -> str:
    """
    Multi-pass Google search via SerpAPI; rich parsing + strong fallback.
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

        q1 = (f"{src_city} to {dst_city} bus tickets price duration "
              f"site:redbus.in OR site:abhibus.com OR site:intrcity.com OR site:zingbus.com OR "
              f"site:paytm.com/bus-tickets OR site:makemytrip.com/bus-tickets OR site:yatra.com/bus")
        if travel_date: q1 += f" {travel_date}"
        raw1 = _serpapi_search(q1, gl="in", hl="en", num=20)
        organic1 = raw1.get("organic_results") or raw1.get("organic") or []
        debug.append(f"pass1_organic: {len(organic1)}")

        options = _build_options(organic1, src_city, dst_city, base_dep, la, dist_km, max_results, debug)

        if len(options) < max_results // 2:
            q2 = f"{src_city} to {dst_city} bus price duration booking"
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
            est_h = 30.0
            if dist_km: est_h = max(10.0, min(44.0, round(dist_km / 50.0, 1)))
            est_price = int(max(300, min(5000, round((dist_km or 800) * 0.9))))
            rb_url = f"https://www.redbus.in/bus-tickets/{_slug(src_city)}-to-{_slug(dst_city)}-bus-tickets"
            for j in range(1, 3+1):
                d = base_dep + timedelta(hours=2*(j-1))
                a = d + timedelta(hours=est_h + j-1)
                if la and a > la: continue
                options.append({
                    "mode": "bus",
                    "src": src_city, "dst": dst_city,
                    "depart": d.isoformat(),
                    "arrive": a.isoformat(),
                    "duration_hours": float(est_h + j-1),
                    "price_inr": est_price,
                    "ref": f"B_FALLBACK_{j}",
                    "provider": {"name": "SerpAPI GoogleSearch", "link": rb_url},
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
        _save_json(BUS_DIR, sid, snapshot)
        return json.dumps({"search_id": sid, "count": len(options)})
    except Exception as e:
        debug.append(f"exception:{e}")
        return json.dumps({"error": str(e)})

@mcp.tool()
def get_bus_details(search_id: str) -> str:
    p = BUS_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    return p.read_text(encoding="utf-8")

@mcp.tool()
def normalize_buses(search_id: str) -> str:
    p = BUS_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    data = json.loads(p.read_text(encoding="utf-8"))
    return json.dumps(data.get("options") or [])

@mcp.resource("buses://searches")
def list_bus_searches() -> str:
    lines = ["# Bus Searches", ""]
    for f in sorted(BUS_DIR.glob("*.json")):
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
