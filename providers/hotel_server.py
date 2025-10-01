# providers/hotel_server.py
from __future__ import annotations
import json, os, re, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hotel-assistant")

try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(ROOT / ".env")
except Exception:
    pass

SERPAPI_KEY = os.getenv("SERPAPI_KEY") or os.getenv("SERP_API_KEY")
if not SERPAPI_KEY:
    print("[MCP] Missing SERPAPI_KEY in environment (.env not found or not exported).", file=sys.stderr)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data")).resolve()
HOTEL_DIR = (DATA_ROOT / "hotels"); HOTEL_DIR.mkdir(parents=True, exist_ok=True)
REQ_TIMEOUT = 35

RE_KM = re.compile(r"(\d+(?:\.\d+)?)\s*km", re.IGNORECASE)

def _safe(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r'[:<>"/\\|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:100]

def _date_tag(dt_iso: Optional[str], fallback: str) -> str:
    if not dt_iso:
        return fallback
    try:
        dt = datetime.fromisoformat(dt_iso)
        return dt.strftime("%Y%m%dT%H%M%S")
    except Exception:
        return fallback

def _ensure_key():
    if not SERPAPI_KEY:
        raise RuntimeError("Missing SERPAPI_KEY")

def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _search_id(city: str, near: Optional[str], checkin: Optional[str], checkout: Optional[str]) -> str:
    city_tag = _safe(city or "city")
    near_tag = _safe((near or "near"))
    cin_tag  = _date_tag(checkin, "CIN")
    cout_tag = _date_tag(checkout, "COUT")
    return f"{city_tag}_{near_tag}_{cin_tag}_{cout_tag}_{_ts()}"

def _save_json(folder: Path, key: str, data: Dict[str, Any]) -> Path:
    p = folder / f"{key}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

def _load_json(folder: Path, key: str) -> Dict[str, Any]:
    p = folder / f"{key}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = re.sub(r"[^\d.]", "", x)
        return float(x)
    except Exception:
        return None

def _parse_km(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = RE_KM.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _normalize_hotels(
    raw: Dict[str, Any],
    city: str,
    near: Optional[str],
    checkin: Optional[str],
    checkout: Optional[str],
) -> List[Dict[str, Any]]:
    props = raw.get("properties") or []
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(props, 1):
        name = p.get("name") or p.get("title")
        if not name:
            continue

        price = None
        if "rate_per_night" in p and isinstance(p["rate_per_night"], dict):
            price = p["rate_per_night"].get("extracted_lowest") or p["rate_per_night"].get("lowest")
        if price is None:
            price = p.get("total_rate") or p.get("price")

        rating = p.get("overall_rating") or p.get("rating")
        distance_text = p.get("distance") or p.get("distance_from_search")

        out.append({
            "name": str(name)[:80],
            "price_inr": _to_float(price),
            "distance_km": _parse_km(distance_text),
            "rating": _to_float(rating),
            "city": city,
            "near": near,
            "checkin": checkin,
            "checkout": checkout,
            "ref": p.get("property_token") or f"H{i:02d}",
            "provider": {"name": "SerpAPI GoogleHotels", "link": p.get("link")},
        })
    return out

def _serpapi_hotels(params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get("https://serpapi.com/search", params=params, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()

@mcp.tool()
def search_hotels(
    city: str,
    near: Optional[str] = None,
    checkin: Optional[str] = None,         # "YYYY-MM-DD" or ISO
    checkout: Optional[str] = None,        # "YYYY-MM-DD" or ISO
    adults: int = 1,
    currency: str = "INR",
    country: str = "in",
    language: str = "en",
    children: int = 0,
    children_ages: Optional[List[int]] = None,
    sort_by: Optional[int] = None,         # e.g. 3=Lowest price, 8=Highest rating, 13=Most reviewed
    hotel_class: Optional[List[int]] = None,  # e.g. [4,5]
    amenities: Optional[List[int]] = None,
    property_types: Optional[List[int]] = None,
    brands: Optional[List[int]] = None,
    free_cancellation: bool = False,
    special_offers: bool = False,
    vacation_rentals: bool = False,
    bedrooms: Optional[int] = None,
    max_results: int = 30
) -> str:
    """
    Live hotel search via SerpAPI Google Hotels (rich filters supported).
    Saves a snapshot under data/hotels and returns: {"search_id","count"}.
    """
    try:
        _ensure_key()
        q = f"hotels near {near} {city}" if near else f"hotels in {city}"
        params: Dict[str, Any] = {
            "engine": "google_hotels",
            "api_key": SERPAPI_KEY,
            "q": q,
            "gl": country,
            "hl": language,
            "currency": currency,
            "adults": adults,
            "check_in_date": checkin,
            "check_out_date": checkout,
        }
        if children:
            params["children"] = children
        if children_ages:
            params["children_ages"] = ",".join(str(a) for a in children_ages)
        if sort_by:
            params["sort_by"] = sort_by
        if hotel_class:
            params["hotel_class"] = ",".join(str(h) for h in hotel_class)
        if amenities:
            params["amenities"] = ",".join(str(a) for a in amenities)
        if property_types:
            params["property_types"] = ",".join(str(p) for p in property_types)
        if brands:
            params["brands"] = ",".join(str(b) for b in brands)
        if free_cancellation:
            params["free_cancellation"] = "true"
        if special_offers:
            params["special_offers"] = "true"
        if vacation_rentals:
            params["vacation_rentals"] = "true"
        if bedrooms is not None:
            params["bedrooms"] = bedrooms

        params = {k: v for k, v in params.items() if v is not None}

        raw = _serpapi_hotels(params)

        sid = _search_id(city, near, checkin, checkout)
        snapshot = {
            "search_metadata": {
                "search_id": sid,
                "city": city, "near": near,
                "checkin": checkin, "checkout": checkout,
                "adults": adults, "children": children, "children_ages": children_ages,
                "currency": currency, "gl": country, "hl": language,
                "query": q,
                "filters": {
                    "sort_by": sort_by,
                    "hotel_class": hotel_class,
                    "amenities": amenities,
                    "property_types": property_types,
                    "brands": brands,
                    "free_cancellation": free_cancellation,
                    "special_offers": special_offers,
                    "vacation_rentals": vacation_rentals,
                    "bedrooms": bedrooms,
                },
                "timestamp_utc": datetime.utcnow().isoformat(),
            },
            "properties": (raw.get("properties") or [])[: max_results],
            "serpapi_context": {k: raw.get(k) for k in ("search_metadata","search_parameters","search_information") if k in raw},
        }
        _save_json(HOTEL_DIR, sid, snapshot)
        return json.dumps({"search_id": sid, "count": len(snapshot["properties"])})
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def get_hotel_details(search_id: str) -> str:
    p = HOTEL_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    return p.read_text(encoding="utf-8")

@mcp.tool()
def normalize_hotels(search_id: str) -> str:
    p = HOTEL_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    data = json.loads(p.read_text(encoding="utf-8"))
    meta = data.get("search_metadata", {})
    norm = _normalize_hotels(
        data,
        meta.get("city", ""),
        meta.get("near"),
        meta.get("checkin"),
        meta.get("checkout"),
    )
    return json.dumps(norm)

@mcp.tool()
def filter_hotels_by_price(search_id: str, max_price: Optional[float] = None, min_price: Optional[float] = None) -> str:
    p = HOTEL_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    data = json.loads(p.read_text(encoding="utf-8"))
    props = data.get("properties", [])
    def _price_of(prop: Dict[str, Any]) -> float:
        rpn = prop.get("rate_per_night", {})
        v = rpn.get("extracted_lowest") or rpn.get("lowest") or prop.get("total_rate") or prop.get("price")
        f = _to_float(v)
        return f if f is not None else 0.0
    filtered = []
    for h in props:
        pr = _price_of(h)
        if min_price is not None and pr < min_price: continue
        if max_price is not None and pr > max_price: continue
        filtered.append(h)
    out = {"search_id": search_id, "total_filtered": len(filtered), "filtered_properties": filtered}
    return json.dumps(out)

@mcp.tool()
def filter_hotels_by_rating(search_id: str, min_rating: float = 4.0) -> str:
    p = HOTEL_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    data = json.loads(p.read_text(encoding="utf-8"))
    props = data.get("properties", [])
    filtered = [h for h in props if (h.get("overall_rating") or 0) >= min_rating]
    out = {"search_id": search_id, "total_filtered": len(filtered), "filtered_properties": filtered}
    return json.dumps(out)

@mcp.tool()
def filter_hotels_by_amenities(search_id: str, required_amenities: List[str]) -> str:
    p = HOTEL_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    data = json.loads(p.read_text(encoding="utf-8"))
    props = data.get("properties", [])
    req = [a.lower() for a in (required_amenities or [])]
    def ok(h: Dict[str, Any]) -> bool:
        ams = [a.lower() for a in (h.get("amenities") or [])]
        return all(a in ams for a in req)
    filtered = [h for h in props if ok(h)]
    out = {"search_id": search_id, "total_filtered": len(filtered), "filtered_properties": filtered}
    return json.dumps(out)

@mcp.resource("hotels://searches")
def list_hotel_searches() -> str:
    lines = ["# Hotel Searches", ""]
    for f in sorted(HOTEL_DIR.glob("*.json")):
        sid = f.stem
        try:
            meta = json.loads(f.read_text(encoding="utf-8")).get("search_metadata", {})
            lines += [
                f"## {sid}",
                f"- city: {meta.get('city')}, near: {meta.get('near')}",
                f"- dates: {meta.get('checkin')} → {meta.get('checkout')}",
                f"- adults: {meta.get('adults')}, children: {meta.get('children')}",
                f"- when: {meta.get('timestamp_utc')}",
                "",
                "---",
                ""
            ]
        except Exception:
            continue
    if len(lines) == 2:
        lines += ["No hotel searches found."]
    return "\n".join(lines)

if __name__ == "__main__":
    mcp.run(transport="stdio")

