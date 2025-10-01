# providers/flight_server.py
from __future__ import annotations
import asyncio, json, os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("flight-assistant")

import os, sys
from pathlib import Path
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
FLIGHT_DIR = (DATA_ROOT / "flights"); FLIGHT_DIR.mkdir(parents=True, exist_ok=True)
REQ_TIMEOUT = 35

def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def _ensure_key():
    if not SERPAPI_KEY:
        raise RuntimeError("Missing SERPAPI_KEY")

def _search_id(dep: str, arr: str, out_date: str, ret: Optional[str]) -> str:
    rid = f"{dep}_{arr}_{out_date}"
    if ret: rid += f"_{ret}"
    rid += f"_{_ts()}"
    return rid

def _save_json(folder: Path, key: str, data: Dict[str, Any]) -> Path:
    p = folder / f"{key}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return p

def _to_float(x: Any) -> Optional[float]:
    try: return float(x)
    except Exception: return None

def _flatten_options(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    best = payload.get("best_flights") or []
    other = payload.get("other_flights") or []
    items: List[Dict[str, Any]] = []
    for opt in [*best, *other]:
        legs = opt.get("flights") or []
        if not legs: continue
        depart = legs[0].get("departure_airport", {}).get("time")
        arrive = legs[-1].get("arrival_airport", {}).get("time")
        src = legs[0].get("departure_airport", {}).get("id")
        dst = legs[-1].get("arrival_airport", {}).get("id")
        total_min = opt.get("total_duration")
        dur_h = round(total_min/60.0, 2) if total_min else None
        items.append({
            "mode": "flight",
            "src": src, "dst": dst,
            "depart": depart, "arrive": arrive,
            "duration_hours": dur_h,
            "price_inr": _to_float(opt.get("price")),
            "ref": opt.get("link") or opt.get("id") or "",
            "provider": {"name": "SerpAPI GoogleFlights"},
        })
    return items

@mcp.tool()
def search_flights(
    departure_id: str,
    arrival_id: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
    infants_in_seat: int = 0,
    infants_on_lap: int = 0,
    travel_class: int = 2,         # 1=Economy, 2=Premium Econ, 3=Business, 4=First
    currency: str = "INR",
    country: str = "in",
    language: str = "en",
    trip_type: int = 2,            # 2=one-way, 1=round
) -> str:
    """Live flight search via SerpAPI Google Flights; saves snapshot; returns summary."""
    try:
        _ensure_key()
        params = {
            "engine": "google_flights",
            "type": "2" if trip_type == 2 else "1",
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": outbound_date,
            "gl": country, "hl": language, "currency": currency,
            "adults": adults, "children": children,
            "infants_in_seat": infants_in_seat, "infants_on_lap": infants_on_lap,
            "travel_class": travel_class,
            "api_key": SERPAPI_KEY,
        }
        if trip_type == 1:
            if not return_date:
                return json.dumps({"error": "return_date required for round trip"})
            params["return_date"] = return_date

        r = requests.get("https://serpapi.com/search", params=params, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        raw = r.json()

        snapshot = {
            "search_metadata": {
                "search_id": _search_id(departure_id, arrival_id, outbound_date, return_date),
                "route": f"{departure_id}->{arrival_id}",
                "outbound_date": outbound_date, "return_date": return_date,
                "adults": adults, "children": children,
                "infants_in_seat": infants_in_seat, "infants_on_lap": infants_on_lap,
                "class": travel_class, "currency": currency, "gl": country, "hl": language,
                "timestamp_utc": datetime.utcnow().isoformat(),
            },
            "best_flights": raw.get("best_flights") or [],
            "other_flights": raw.get("other_flights") or [],
            "price_insights": raw.get("price_insights"),
            "airports": raw.get("airports"),
        }
        sid = snapshot["search_metadata"]["search_id"]
        _save_json(FLIGHT_DIR, sid, snapshot)

        prices = [f.get("price") for f in [*snapshot["best_flights"], *snapshot["other_flights"]] if f.get("price") is not None]
        return json.dumps({
            "search_id": sid,
            "count_best": len(snapshot["best_flights"]),
            "count_other": len(snapshot["other_flights"]),
            "min_price": min(prices) if prices else None,
            "max_price": max(prices) if prices else None,
            "currency": currency,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def get_flight_details(search_id: str) -> str:
    p = FLIGHT_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    return p.read_text()

@mcp.tool()
def normalize_flights(search_id: str) -> str:
    p = FLIGHT_DIR / f"{search_id}.json"
    if not p.exists():
        return json.dumps({"error": f"not found: {search_id}"})
    data = json.loads(p.read_text())
    return json.dumps(_flatten_options(data))

if __name__ == "__main__":
    mcp.run(transport="stdio")
