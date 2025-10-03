from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, date
import json
import os
import re
from langgraph.graph import StateGraph, END
from providers.client import (flights_client, buses_client, trains_client, hotels_client)
from src.llm import OpenRouterClient, DEFAULT_MODEL
from dateutil import tz as dateutil_tz

LOCAL_TZ_NAME = os.getenv("TIMEZONE", "Asia/Kolkata")
LOCAL_TZ = dateutil_tz.gettz(LOCAL_TZ_NAME) or dateutil_tz.gettz("Asia/Kolkata")
LLM_IATA_ENABLED = os.getenv("LLM_IATA_ENABLED", "true").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_MODEL)
LLM_TIMEOUT_SECS = int(os.getenv("LLM_TIMEOUT_SECS", "18"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
IATA_COUNTRY_BIAS = os.getenv("IATA_COUNTRY_BIAS", "IN")
_IATA_CACHE: Dict[str, Dict[str, Any]] = {}

def _arrival_point_name(r: Dict[str, Any]) -> str:
    # Try structured fields first
    for k in ("arrival_airport", "arrival_station", "arrival_terminal",
              "arrival_stop", "arrival_point", "to_name"):
        v = r.get(k)
        if isinstance(v, dict):
            name = v.get("name") or v.get("short") or v.get("code")
            if name: return str(name)
        elif isinstance(v, str) and v.strip():
            return v.strip()
    # Fall back by mode + city
    city = (r.get("arrival_city") or r.get("to_city") or "").strip()
    mode = (r.get("mode") or "").strip().lower()
    if city:
        if mode == "flight": return f"{city} Airport"
        if mode == "train":  return f"{city} Railway Station"
        if mode == "bus":    return f"{city} Bus Stand"
    return (r.get("mode") or "arrival").title()

def _google_maps_link(origin: str, destination: str, travelmode: str = "driving") -> str:
    import urllib.parse as _up
    o = _up.quote_plus(origin or "")
    d = _up.quote_plus(destination or "")
    tm = _up.quote_plus(travelmode)
    return f"https://www.google.com/maps/dir/?api=1&origin={o}&destination={d}&travelmode={tm}"

def _coarse_travel_minutes(origin_name: str, dest_text: Optional[str]) -> int:
    o = (origin_name or "").lower()
    if "airport" in o: return 50      # metro-ish default
    if any(x in o for x in ("railway", "station", "bus", "stand", "terminal")): return 35
    return 40

def _rough_cab_fare_inr(minutes: int) -> int:
    return int(round(minutes * 22, 0))


def _normalize_option_headers(text: str) -> str:
    def fix_line(lbl: str, s: str) -> str:
        s = re.sub(rf"(?mi)^\s*•\s*{lbl}\s*:\s*(.+?)\s*$", rf"{lbl}\n• Mode: \1", s)
        return s
    s = text
    for lbl in ("Value", "Fastest", "Cheapest"):
        s = fix_line(lbl, s)
    return s

def _normalize_duration_units(text: str) -> str:
    return re.sub(
        r"(?mi)^\s*•\s*Duration:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
        r"• Duration: \1 hours",
        text
    )

def _ensure_option_subheaders(text: str) -> str:
    kinds = ["Value", "Fastest", "Cheapest"]
    header_pat = re.compile(r"^\s*(Value|Fastest|Cheapest)\s*$", re.I)
    mode_pat = re.compile(r"^\s*•\s*Mode:", re.I)
    m = re.search(r"(?m)^\s*\*{0,2}\s*Options\s*\*{0,2}\s*$", text)
    if not m:
        return text
    start = m.end()
    m2 = re.search(
        r"(?m)^\s*\*{0,2}\s*(Hotels|Soumya's Suggested Plan|Cheapest Plan|Next Steps)\s*\*{0,2}\s*$",
        text[start:]
    )
    end = start + (m2.start() if m2 else len(text[start:]))
    block = text[start:end]
    lines = block.splitlines()
    used_headers: List[str] = []
    out: List[str] = []

    def is_header(s: str) -> bool:
        return bool(header_pat.match(s))

    def last_nonempty(arr: List[str]) -> str:
        for L in reversed(arr):
            if L.strip():
                return L
        return ""
    for line in lines:
        if is_header(line):
            hdr = line.strip().title()
            if not (out and out[-1].strip().title() == hdr):
                out.append(hdr)
            if hdr not in used_headers:
                used_headers.append(hdr)
            continue
        if mode_pat.match(line):
            prev = last_nonempty(out)
            if not is_header(prev):
                for k in kinds:
                    if k not in used_headers:
                        out.append(k)
                        used_headers.append(k)
                        break
        out.append(line)

    new_block = "\n".join(out)
    return text[:start] + new_block + text[end:]

def _normalize_city_phrase(city_phrase: str) -> Tuple[str, Optional[str]]:
    s = (city_phrase or "").strip()
    if not s:
        return "", None
    resolved = _llm_resolve_iata(s, IATA_COUNTRY_BIAS)
    if resolved:
        norm = (resolved.get("normalized_city") or "").strip()
        iata = (resolved.get("primary_iata") or "").strip().upper() or None
        if norm:
            return norm, iata
        if iata and len(s) == 3 and s.isalpha():
            return s.upper(), iata
    return s.title(), None

def _compute_default_rationale(kind: str, route: Dict[str, Any]) -> str:
    base = {
        "Value": "Best balance of time and cost",
        "Fastest": "Shortest overall travel time",
        "Cheapest": "Lowest total price",
        "Suggested": "Best overall pick",
        "CheapestPlan": "Lowest total price",
    }.get(kind, "—")
    after_window = bool(route.get("after_window")) or any(
        note for note in (route.get("notes") or []) if "arrives_after_latest_arrival" in str(note)
    )
    provisional = (route.get("duration_note") == "duration_unreliable")
    if after_window:
        base += "; arrives after preferred window (optional)"
    if provisional:
        base += "; timing provisional—confirm on booking"
    return base

def _route_booking_link_or_provider(r: Dict[str, Any]) -> Optional[str]:
    if r.get("booking_link"):
        return r["booking_link"]
    prov = r.get("provider") or {}
    if isinstance(prov, dict):
        cand = prov.get("link") or prov.get("booking_link") or (prov.get("urls") or {}).get("booking")
        if cand: return cand
    for k in ("provider_link", "link", "url"):
        if r.get(k): return r[k]
    return None

def _patch_options_and_plans(text: str, value: Dict[str, Any], fastest: Dict[str, Any],
                             cheapest: Dict[str, Any], hotel_value: Dict[str, Any],
                             hotel_cheapest: Dict[str, Any]) -> str:
    kinds = ["Value", "Fastest", "Cheapest"]
    routes = [value or {}, fastest or {}, cheapest or {}]

    def fmt_price(r):
        return _format_inr(r.get("price_inr"))

    def fix_block(block: str, kind: Optional[str], route: Dict[str, Any]) -> str:
        if not route:
            return block
        block = re.sub(
            r"(?mi)^\s*•\s*Price:\s*₹\s*\(no price available\)\s*$",
            f"• Price: {fmt_price(route)}",
            block
        )
        real_link = _route_booking_link_or_provider(route)
        if real_link:
            if re.search(r"(?mi)^\s*•\s*Booking Link:\s*.*$", block):
                block = re.sub(r"(?mi)^\s*•\s*Booking Link:\s*.*$", f"• Booking Link: {real_link}", block)
            else:
                block = block.rstrip() + f"\n• Booking Link: {real_link}\n"
        if kind:
            if re.search(r"(?mi)^\s*•\s*Rationale:\s*(?:—\s*|)\s*$", block) or not re.search(r"(?mi)^\s*•\s*Rationale:", block):
                rationale = _compute_default_rationale(kind, route)
                if re.search(r"(?mi)^\s*•\s*Rationale:", block):
                    block = re.sub(r"(?mi)^\s*•\s*Rationale:\s*(?:—\s*|)\s*$", f"• Rationale: {rationale}", block)
                else:
                    if re.search(r"(?mi)^\s*•\s*Price:.*$", block):
                        block = re.sub(r"(?mi)(^\s*•\s*Price:.*$)", r"\1\n" + f"• Rationale: {rationale}", block)
                    elif re.search(r"(?mi)^\s*•\s*Booking Link:.*$", block):
                        block = re.sub(r"(?mi)(^\s*•\s*Booking Link:.*$)", r"\1\n" + f"• Rationale: {rationale}", block)
                    else:
                        block = block.rstrip() + f"\n• Rationale: {rationale}\n"
        return block

    m  = re.search(r"(?m)^\s*\*{0,2}\s*Options\s*\*{0,2}\s*$", text)
    if m:
        start = m.end()
        m2 = re.search(r"(?m)^\s*\*{0,2}\s*(Hotels|Soumya's Suggested Plan|Cheapest Plan|Next Steps)\s*\*{0,2}\s*$", text[start:])
        end = start + (m2.start() if m2 else len(text[start:]))
        block = text[start:end]
        parts = re.split(r"(?=^\s*•\s*Mode:)", block, flags=re.M)
        fixed_parts: List[str] = []
        route_idx = 0
        for part in parts:
            if not part.strip():
                fixed_parts.append(part)
                continue
            if not re.search(r"(?m)^\s*•\s*Mode:", part):
                fixed_parts.append(part)
                continue
            kind = kinds[route_idx] if route_idx < len(kinds) else None
            route = routes[route_idx] if route_idx < len(routes) else {}
            fixed_parts.append(fix_block(part, kind, route))
            route_idx += 1
        text = text[:start] + "".join(fixed_parts) + text[end:]

    m  = re.search(r"(?m)^\s*\*{0,2}\s*Soumya's Suggested Plan\s*\*{0,2}\s*$", text)
    if m:
        start = m.end()
        m2 = re.search(r"(?m)^\s*\*{0,2}\s*(Cheapest Plan|Next Steps)\s*\*{0,2}\s*$", text[start:])
        end = start + (m2.start() if m2 else len(text[start:]))
        block = text[start:end]
        block = fix_block(block, "Suggested", value or {})
        text = text[:start] + block + text[end:]

    m  = re.search(r"(?m)^\s*\*{0,2}\s*Cheapest Plan\s*\*{0,2}\s*$", text)
    if m:
        start = m.end()
        m2 = re.search(r"(?m)^\s*\*{0,2}\s*(Next Steps)\s*\*{0,2}\s*$", text[start:])
        end = start + (m2.start() if m2 else len(text[start:]))
        block = text[start:end]
        block = fix_block(block, "CheapestPlan", cheapest or {})
        text = text[:start] + block + text[end:]
    return text

def _format_inr(n: float | int | None) -> str:
    try:
        return f"₹ {int(float(n)):,}"
    except Exception:
        return "₹ (no price available)"

def _slug(s: str) -> str:
    import re as _re
    s = (s or "").lower()
    s = _re.sub(r"[^a-z0-9\s-]", "", s)
    s = _re.sub(r"\s+", "-", s).strip("-")
    s = _re.sub(r"-+", "-", s)
    return s

def _slugify_city(city: Optional[str]) -> str:
    return _slug(city or "")

def _normalize_money_block(text: str) -> str:
    def _norm(m):
        amount = m.group(1).replace(",", "")
        try:
            return _format_inr(float(amount))
        except Exception:
            return m.group(0)
    text = re.sub(r"₹\s*([0-9][0-9,]*\.?[0-9]*)", _norm, text)
    text = re.sub(r"(₹)\s*(₹)\s*", r"\1 ", text)
    text = re.sub(r"Price:\s*₹\s*\(no price available\)", "Price: ₹ (no price available)", text, flags=re.I)
    return text

def _ensure_booking_links(text: str,
                          base_city: Optional[str],
                          dest_city: Optional[str],
                          base_iata: Optional[str],
                          dest_iata: Optional[str]) -> str:
    from_slug = _slugify_city(base_city)
    to_slug   = _slugify_city(dest_city)
    FROM = (base_iata or (base_city or "")[:3]).upper() if (base_city or base_iata) else ""
    TO   = (dest_iata or (dest_city or "")[:3]).upper() if (dest_city or dest_iata) else ""
    def gen_url(mode: str) -> str:
        m = (mode or "").strip().lower()
        if m == "bus" and from_slug and to_slug:
            return f"https://www.redbus.in/bus-tickets/{from_slug}-to-{to_slug}"
        if m == "train" and base_city and dest_city:
            return f"https://www.irctc.co.in/nget/train-search?from={base_city}&to={dest_city}"
        if m == "flight" and FROM and TO:
            return f"https://www.makemytrip.com/flights/{FROM}-to-{TO}-flights.html"
        return ""
    out, current_mode = [], None
    for line in text.splitlines():
        m = re.match(r"^\s*Mode:\s*(.+?)\s*$", line, flags=re.I)
        if m:
            current_mode = m.group(1).strip()
            out.append(line)
            continue
        if re.match(r"^\s*Booking Link:\s*$", line, flags=re.I):
            url = gen_url(current_mode or "")
            out.append(f"Booking Link: {url}" if url else line)
            continue
        out.append(line)
    return "\n".join(out)

def _ensure_rationale_dash(text: str) -> str:
    out = []
    for line in text.splitlines():
        if re.match(r"^\s*Rationale:\s*$", line) or re.match(r"^\s*Rationale:\s*[-–—]\s*$", line):
            out.append("Rationale: —")
        else:
            out.append(line)
    return "\n".join(out)

def _enrich_rationales(text: str) -> str:
    lines = text.splitlines()
    out: List[str] = []
    block = None
    def default_for(b: str) -> str:
        return {
            "Value": "Best balance of time and cost",
            "Fastest": "Shortest overall travel time",
            "Cheapest": "Lowest total price",
            "Suggested": "Best overall pick",
            "CheapestPlan": "Lowest total price",
        }.get(b, "—")
    for i, line in enumerate(lines):
        if re.match(r"^\s*Value\s*$", line): block = "Value"
        elif re.match(r"^\s*Fastest\s*$", line): block = "Fastest"
        elif re.match(r"^\s*Cheapest\s*$", line): block = "Cheapest"
        elif re.match(r"^\s*Soumya's Suggested Plan\s*$", line): block = "Suggested"
        elif re.match(r"^\s*Cheapest Plan\s*$", line): block = "CheapestPlan"
        if re.match(r"^\s*Rationale:\s*(?:—\s*)?$", line):
            base = default_for(block or "")
            window_note = False
            provisional = False
            scan = "\n".join(lines[max(0, i-6):i+6])
            if re.search(r"Arrives after preferred window \(optional\)", scan, flags=re.I):
                window_note = True
            if re.search(r"Timing provisional—confirm on booking", scan, flags=re.I):
                provisional = True
            if window_note:
                base = f"{base}; arrives after preferred window (optional)"
            if provisional:
                base = f"{base}; timing provisional—confirm on booking"
            out.append(f"Rationale: {base}")
        else:
            out.append(line)
    return "\n".join(out)

def _force_bullets_in_options(text: str) -> str:
    lines = text.splitlines()
    out, in_options, in_sub = [], False, False
    label_pat = re.compile(r"^\s*(Mode|Depart|Arrive|Duration|Price|Rationale|Booking Link):", re.I)
    for i, line in enumerate(lines):
        if re.match(r"^\s*Options\s*$", line):
            in_options = True
            out.append("**Options**")
            continue
        if in_options and re.match(r"^\s*(Hotels|Soumya's Suggested Plan|Cheapest Plan|Next Steps)\s*$", line):
            in_options = False
            out.append(line if line.startswith("**") else f"**{line.strip()}**")
            continue
        if in_options and re.match(r"^\s*(Value|Fastest|Cheapest)\s*$", line):
            in_sub = True
            out.append(f"**{line.strip()}**")
            continue
        if in_options and in_sub and label_pat.match(line):
            out.append("• " + line.strip())
            continue
        out.append(line)
    return "\n".join(out)

def _bold_static_headers(text: str) -> str:
    text = re.sub(r"^\s*Trip Plan for (.+?)\s*$", r"**Trip Plan for \1**", text, flags=re.M)
    for sec in ["Hotels", "Soumya's Suggested Plan", "Cheapest Plan", "Next Steps"]:
        text = re.sub(rf"^\s*{re.escape(sec)}\s*$", f"**{sec}**", text, flags=re.M)
    return text

def _fmt_dt_for_display(dt_str: Optional[str]) -> Optional[str]:
    if not dt_str:
        return None
    try:
        d = datetime.fromisoformat(dt_str)
    except Exception:
        try:
            from dateutil import parser as dtp
            d = dtp.isoparse(dt_str)
        except Exception:
            return dt_str
    if d.tzinfo is None:
        d = d.replace(tzinfo=LOCAL_TZ)
    else:
        d = d.astimezone(LOCAL_TZ)
    return d.strftime("%Y-%m-%d %H:%M")

def _recompute_duration_hours(route: Dict[str, Any]) -> None:
    dep = route.get("depart"); arr = route.get("arrive")
    if not dep or not arr: return
    try:
        from dateutil import parser as dtp
        d1 = dtp.isoparse(dep); d2 = dtp.isoparse(arr)
        d1 = d1.replace(tzinfo=LOCAL_TZ) if d1.tzinfo is None else d1.astimezone(LOCAL_TZ)
        d2 = d2.replace(tzinfo=LOCAL_TZ) if d2.tzinfo is None else d2.astimezone(LOCAL_TZ)
        hrs = max(0.0, (d2 - d1).total_seconds() / 3600.0)
        route["duration_hours"] = round(hrs, 1)
    except Exception:
        pass

def _to_local_aware(dt_in: Optional[datetime]) -> Optional[datetime]:
    if dt_in is None: return None
    return dt_in.replace(tzinfo=LOCAL_TZ) if dt_in.tzinfo is None else dt_in.astimezone(LOCAL_TZ)

def _parse_route_iso_local(s: str) -> Optional[datetime]:
    try:
        d = datetime.fromisoformat(s)
    except Exception:
        try:
            from dateutil import parser as dtp
            d = dtp.isoparse(s)
        except Exception:
            return None
    return d.replace(tzinfo=LOCAL_TZ) if d.tzinfo is None else d.astimezone(LOCAL_TZ)

def _ordered_picks(state: "AgentState") -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("Value", state.get("value") or {}),
        ("Fastest", state.get("fastest") or {}),
        ("Cheapest", state.get("cheapest") or {}),
    ]

def _clean_route_times_for_display(r: Dict[str, Any]) -> None:
    if "depart" in r: r["depart"] = _fmt_dt_for_display(r.get("depart"))
    if "arrive" in r: r["arrive"] = _fmt_dt_for_display(r.get("arrive"))

class AgentState(TypedDict, total=False):
    base_city: Optional[str]
    destination_city: Optional[str]
    meeting_start: Optional[datetime]
    meeting_end: Optional[datetime]
    earliest_departure: Optional[datetime]
    latest_arrival: Optional[datetime]
    latest_return: Optional[datetime]
    venue_text: Optional[str]
    base_city_norm: Optional[str]
    dest_city_norm: Optional[str]
    base_iata: Optional[str]
    dest_iata: Optional[str]
    flights: List[Dict[str, Any]]
    trains: List[Dict[str, Any]]
    buses: List[Dict[str, Any]]
    hotels: List[Dict[str, Any]]
    hotel_value: Dict[str, Any]
    hotel_cheapest: Dict[str, Any]
    venue_latlng: Optional[Tuple[float, float]]                   
    cheapest: Dict[str, Any]
    fastest: Dict[str, Any]
    value: Dict[str, Any]
    ordered_recommendations: List[Tuple[str, Dict[str, Any]]]
    debug: List[str]
    itinerary_text: str
    summary_json: Dict[str, Any]
    last_mile: Dict[str, Any]  

class ToolMux:
    def __init__(self):
        self.flights = flights_client()
        self.buses = buses_client()
        self.trains = trains_client()
        self.hotels = hotels_client()
    def call(self, tool: str, **kwargs):
        t = tool.lower()
        if "flight" in t: return self.flights.call(tool, **kwargs)
        if "hotel"  in t: return self.hotels.call(tool, **kwargs)
        if "train"  in t: return self.trains.call(tool, **kwargs)
        if "bus"    in t: return self.buses.call(tool, **kwargs)
        raise ValueError(f"Unknown tool namespace for: {tool}")

_tools = ToolMux()

def _iso(dt_in: datetime | None) -> str | None:
    return dt_in.isoformat() if dt_in else None

_IATA_SYSTEM = (
    "You map city/location phrases to IATA airport codes. "
    "Return strict JSON ONLY with this schema:\n"
    "{\n"
    '  "normalized_city": string|null,\n'
    '  "primary_iata": string|null,\n'
    '  "alternates": [ {"code": string, "note": string} ]\n'
    "}\n"
    "- Prefer the main commercial airport code for the given city.\n"
    "- Consider country bias if provided (e.g., IN for India).\n"
    "- If user already provided a 3-letter IATA code, echo it as primary.\n"
    "- If ambiguous, still pick the most likely city+airport and put others in alternates."
)

def _llm_resolve_iata(city_text: str, country_bias: Optional[str]) -> Optional[Dict[str, Any]]:
    if not city_text: return None
    key = f"{city_text.strip().lower()}|{(country_bias or '').upper()}"
    if key in _IATA_CACHE: return _IATA_CACHE[key]
    s = city_text.strip()
    if len(s) == 3 and s.isalpha():
        result = {"normalized_city": None, "primary_iata": s.upper(), "alternates": []}
        _IATA_CACHE[key] = result
        return result
    if not LLM_IATA_ENABLED: return None
    prompt = {"city_text": city_text, "country_bias": (country_bias or "").upper() or None}
    try:
        client = OpenRouterClient()
        messages = [
            {"role": "system", "content": _IATA_SYSTEM},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        out = client.chat(messages, model=LLM_MODEL, temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT_SECS)
        try:
            data = json.loads(out)
        except Exception:
            i, j = out.find("{"), out.rfind("}")
            data = json.loads(out[i:j+1]) if (i != -1 and j != -1 and j > i) else None
        if isinstance(data, dict):
            norm_city = (data.get("normalized_city") or None)
            primary = (data.get("primary_iata") or None)
            alts = data.get("alternates") or []
            if primary and isinstance(primary, str) and len(primary) == 3:
                result = {"normalized_city": norm_city, "primary_iata": primary.upper(), "alternates": alts}
                _IATA_CACHE[key] = result
                return result
    except Exception:
        pass
    return None

def _iata(code_or_city: Optional[str], dbg: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[str]]:
    if not code_or_city: return None, None
    raw = code_or_city.strip()
    if len(raw) == 3 and raw.isalpha(): return raw.upper(), None
    resolved = _llm_resolve_iata(raw, IATA_COUNTRY_BIAS)
    if resolved and resolved.get("primary_iata"):
        if dbg is not None:
            if resolved.get("normalized_city"):
                dbg.append(f"IATA resolved: {raw} → {resolved['primary_iata']} (as {resolved['normalized_city']})")
            else:
                dbg.append(f"IATA resolved: {raw} → {resolved['primary_iata']}")
        return resolved["primary_iata"], resolved.get("normalized_city")
    if dbg is not None: dbg.append(f"IATA unresolved for: {raw}")
    return None, None

def _clamp_future(d_in: date) -> date:
    today = date.today()
    return d_in if d_in >= today else today

def _choose_outbound_date(state: "AgentState") -> str:
    if state.get("earliest_departure"): return _clamp_future(state["earliest_departure"].date()).isoformat()
    if state.get("meeting_start"): return _clamp_future(state["meeting_start"].date()).isoformat()
    return date.today().isoformat()

def gather_constraints(state: "AgentState") -> "AgentState":
    if state.get("meeting_start") and not state.get("meeting_end"):
        state["meeting_end"] = state["meeting_start"] + timedelta(hours=2)
    state.setdefault("debug", [])
    dbg = state["debug"]
    biata, bnorm = _iata(state.get("base_city"), dbg)
    diata, dnorm = _iata(state.get("destination_city"), dbg)
    state["base_iata"] = biata; state["dest_iata"] = diata
    if bnorm: state["base_city_norm"] = bnorm
    if dnorm: state["dest_city_norm"] = dnorm
    return state

def plan_outbound(state: "AgentState") -> "AgentState":
    src_city = state.get("base_city_norm") or state.get("base_city")
    dst_city = state.get("dest_city_norm") or state.get("destination_city")
    dbg = state.setdefault("debug", [])
    dep_id = state.get("base_iata"); arr_id = state.get("dest_iata")
    flights: List[Dict[str, Any]] = []
    if not dep_id or not arr_id:
        dbg.append(f"Flights skipped: missing IATA (src={src_city}→{dep_id}, dst={dst_city}→{arr_id}).")
    else:
        outbound_date = _choose_outbound_date(state)
        f_search = _tools.call(
            "search_flights", departure_id=dep_id, arrival_id=arr_id,
            outbound_date=outbound_date, trip_type=2, currency="INR",
            country="in", language="en", adults=1, travel_class=1,
        )
        if isinstance(f_search, dict) and "search_id" in f_search:
            f_norm = _tools.call("normalize_flights", search_id=f_search["search_id"])
            if isinstance(f_norm, list):
                flights = f_norm; dbg.append(f"Flights: {len(flights)} options on {outbound_date}.")
            else:
                dbg.append("Flights normalize returned non-list/empty.")
        else:
            dbg.append(f"Flights search returned no search_id: {f_search!r}")
    trains: List[Dict[str, Any]] = []
    t_search = _tools.call(
        "search_trains", src_city=src_city, dst_city=dst_city,
        earliest_departure=_iso(state.get("earliest_departure")),
        latest_arrival=_iso(state.get("latest_arrival")),
    )
    if isinstance(t_search, dict) and "search_id" in t_search:
        t_norm = _tools.call("normalize_trains", search_id=t_search["search_id"])
        if isinstance(t_norm, list):
            trains = t_norm; dbg.append(f"Trains: {len(trains)} options.")
        else:
            dbg.append("Trains normalize returned non-list/empty.")
    else:
        dbg.append(f"Trains search returned no search_id: {t_search!r}")
    buses: List[Dict[str, Any]] = []
    b_search = _tools.call(
        "search_buses", src_city=src_city, dst_city=dst_city,
        earliest_departure=_iso(state.get("earliest_departure")),
        latest_arrival=_iso(state.get("latest_arrival")),
    )
    if isinstance(b_search, dict) and "search_id" in b_search:
        b_norm = _tools.call("normalize_buses", search_id=b_search["search_id"])
        if isinstance(b_norm, list):
            buses = b_norm; dbg.append(f"Buses: {len(buses)} options.")
        else:
            dbg.append("Buses normalize returned non-list/empty.")
    else:
        dbg.append(f"Buses search returned no search_id: {b_search!r}")
    for coll in (flights, trains, buses):
        for r in coll or []:
            _recompute_duration_hours(r)
    state["flights"], state["trains"], state["buses"] = flights, trains, buses
    return state

def suggest_hotels(state: "AgentState") -> "AgentState":
    checkin = state.get("meeting_start")
    checkout = state.get("meeting_end") or (checkin + timedelta(hours=2) if checkin else None)

    def _first_route_dt() -> Optional[datetime]:
        for key in ("value", "fastest", "cheapest"):
            r = state.get(key) or {}
            d = r.get("arrive") or r.get("depart")
            if d:
                try:
                    from dateutil import parser as dtp
                    x = dtp.isoparse(d)
                    return x.replace(tzinfo=LOCAL_TZ) if x.tzinfo is None else x.astimezone(LOCAL_TZ)
                except Exception:
                    pass
        return None

    if not checkin:
        cand = _first_route_dt() or state.get("latest_arrival")
        if isinstance(cand, datetime):
            checkin = cand
    if not checkout and isinstance(checkin, datetime):
        checkout = checkin + timedelta(days=1)

    if checkin and checkin.date() < date.today():
        checkin = datetime.combine(date.today(), (checkin.time() if isinstance(checkin, datetime) else datetime.min.time()))
    if checkout and checkout.date() <= (checkin.date() if isinstance(checkin, datetime) else date.today()):
        checkout = (checkin + timedelta(days=1)) if isinstance(checkin, datetime) else datetime.combine(date.today() + timedelta(days=1), datetime.min.time())

    if not checkin:
        checkin = datetime.combine(date.today(), datetime.min.time()).replace(tzinfo=LOCAL_TZ)
    if not checkout:
        checkout = (checkin + timedelta(days=1)) if isinstance(checkin, datetime) else datetime.combine(date.today() + timedelta(days=1), datetime.min.time()).replace(tzinfo=LOCAL_TZ)

    hotels: List[Dict[str, Any]] = []
    h_search = _tools.call(
        "search_hotels",
        city=state.get("dest_city_norm") or state.get("destination_city"),
        near=state.get("venue_text"),
        checkin=_iso(checkin),
        checkout=_iso(checkout),
        adults=1, currency="INR", country="in", language="en",
    )
    if isinstance(h_search, dict) and "search_id" in h_search:
        h_norm = _tools.call("normalize_hotels", search_id=h_search["search_id"])
        if isinstance(h_norm, list):
            hotels = h_norm; state.setdefault("debug", []).append(f"Hotels: {len(hotels)} options.")
        else:
            state.setdefault("debug", []).append("Hotels normalize returned non-list/empty.")
    else:
        state.setdefault("debug", []).append(f"Hotels search returned no search_id: {h_search!r}")

    def price_of(h): return float(h.get("price_inr")) if isinstance(h.get("price_inr"), (int, float)) else 1e12
    def rating_of(h):
        r = h.get("rating")
        try:
            return float(r) if r is not None else 0.0
        except Exception:
            return 0.0
    def value_score(h):
        return price_of(h) / max(rating_of(h), 2.5)

    state["hotels"] = hotels
    state["hotel_cheapest"] = min(hotels, key=price_of) if hotels else {}
    state["hotel_value"] = min(hotels, key=value_score) if hotels else {}
    return state

def _num_or(val: Any, default: float) -> float:
    if isinstance(val, (int, float)) and val is not None: return float(val)
    return float(default)

def _safe_price(r: Dict[str, Any], default: float = 1e9) -> float: return _num_or(r.get("price_inr"), default)
def _safe_dur(r: Dict[str, Any], default: float = 1e9) -> float: return _num_or(r.get("duration_hours"), default)

def _pick_cheapest(routes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    return min(routes, key=lambda r: _safe_price(r, 1e9)) if routes else None

def _pick_fastest(routes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    return min(routes, key=lambda r: _safe_dur(r, 1e9)) if routes else None

def _score_value(r: Dict[str, Any]) -> float:
    price = _safe_price(r, 1e9); dur = _safe_dur(r, 24.0)
    return price * (dur ** 0.7)

def _pick_value(routes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    return min(routes, key=_score_value) if routes else None

def _flag_unreliable_duration(r: Dict[str, Any]) -> None:
    mode = (r.get("mode") or "").lower(); dur = r.get("duration_hours")
    try: dur_f = float(dur) if dur is not None else None
    except Exception: dur_f = None
    if mode == "bus" and (dur_f is not None) and dur_f < 8.0: r["duration_note"] = "duration_unreliable"
    if mode == "train" and (dur_f is not None) and dur_f < 10.0: r["duration_note"] = "duration_unreliable"

def rank_routes(state: "AgentState") -> "AgentState":
    def _clean(r: Dict[str, Any]):
        _flag_unreliable_duration(r); _clean_route_times_for_display(r)
    la = _to_local_aware(state.get("latest_arrival"))
    dbg = state.setdefault("debug", [])
    def _merge(apply_la: bool, relax_non_flights_only: bool = False) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for coll in ("flights", "trains", "buses"):
            for r in state.get(coll, []) or []:
                arr = _parse_route_iso_local(r.get("arrive", ""))
                if not arr: continue
                violates = False
                if apply_la and la:
                    if relax_non_flights_only:
                        if r.get("mode") == "flight" and arr > la: violates = True
                    else:
                        if arr > la: violates = True
                if violates: continue
                merged.append(r)
        return merged
    merged = _merge(apply_la=True, relax_non_flights_only=False)
    only_flights = all(r.get("mode") == "flight" for r in merged) if merged else True
    if not merged or only_flights:
        relaxed_bt = _merge(apply_la=True, relax_non_flights_only=True)
        if la:
            for r in relaxed_bt:
                try:
                    if r.get("mode") in ("bus", "train"):
                        arr_bt = _parse_route_iso_local(r.get("arrive", ""))
                        if arr_bt and arr_bt > la:
                            r.setdefault("notes", []).append("arrives_after_latest_arrival")
                            r["after_window"] = True
                except Exception:
                    pass
        ids_seen = {id(x) for x in merged}
        for r in relaxed_bt:
            if id(r) not in ids_seen and r.get("mode") in ("train", "bus"):
                merged.append(r)
        if relaxed_bt:
            dbg.append("Relaxed latest_arrival for bus/train to expose cheaper surface options.")
    if not merged:
        merged = _merge(apply_la=False)
        dbg.append("No routes within latest_arrival; fully relaxed arrival constraint for ranking.")
    state["cheapest"] = _pick_cheapest(merged) or {}
    state["fastest"] = _pick_fastest(merged) or {}
    state["value"]   = _pick_value(merged) or {}
    state["ordered_recommendations"] = _ordered_picks(state)
    base_city = state.get("base_city_norm") or state.get("base_city") or ""
    dest_city = state.get("dest_city_norm") or state.get("destination_city") or ""
    from_slug, to_slug = _slug(base_city), _slug(dest_city)
    FROM = (state.get("base_iata") or base_city).upper()
    TO   = (state.get("dest_iata") or dest_city).upper()
    def _depart_ddmmyyyy(r: Dict[str, Any]) -> Optional[str]:
        try:
            from dateutil import parser as dtp
            d = dtp.isoparse(r.get("depart"))
            d = d.replace(tzinfo=LOCAL_TZ) if d.tzinfo is None else d.astimezone(LOCAL_TZ)
            return d.strftime("%d/%m/%Y")
        except Exception:
            return None
    def _attach_link(r: Dict[str, Any]) -> None:
        if r.get("booking_link"):
            return
        prov = r.get("provider") or {}
        if isinstance(prov, dict):
            cand = (
                prov.get("link")
                or prov.get("booking_link")
                or (prov.get("urls") or {}).get("booking")
            )
            if cand:
                r["booking_link"] = cand
                return
        for k in ("provider_link", "link", "url"):
            if r.get(k):
                r["booking_link"] = r[k]
                return
        mode = (r.get("mode") or "").lower()
        if mode == "flight":
            ddmmyyyy = _depart_ddmmyyyy(r)
            if ddmmyyyy:
                r["booking_link"] = (
                    f"https://www.makemytrip.com/flight/search?"
                    f"itinerary={FROM}-{TO}-{ddmmyyyy}&tripType=O&paxType=A-1_C-0_I-0&intl=false&cabinClass=E"
                )
            else:
                r["booking_link"] = f"https://www.makemytrip.com/flights/{FROM}-to-{TO}-flights.html"
        elif mode == "bus":
            r["booking_link"] = f"https://www.redbus.in/bus-tickets/{from_slug}-to-{to_slug}"
        elif mode == "train":
            r["booking_link"] = f"https://www.irctc.co.in/nget/train-search?from={base_city}&to={dest_city}"
    for key in ("cheapest", "fastest", "value"):
        r = state.get(key) or {}
        if r:
            _clean(r); _attach_link(r)
    state["summary_json"] = {
        "cheapest": state["cheapest"],
        "fastest":  state["fastest"],
        "value":    state["value"],
        "hotels_top5": (state.get("hotels") or [])[:5],
        "hotel_value": state.get("hotel_value") or {},
        "hotel_cheapest": state.get("hotel_cheapest") or {},
        "debug": state.get("debug", []),
        "iata": {"from": state.get("base_iata"), "to": state.get("dest_iata")},
        "base_city": base_city,
        "destination_city": dest_city,
    }
    if not merged: dbg.append("No routes merged after filters (all empty).")
    return state

_SYSTEM_PROMPT = (
    "You are TripPlanner, an assistant that crafts concise, decision-ready trip plans.\n"
    "\n"
    "ABSOLUTE FORMAT RULES (STRICT):\n"
    "• Output must BEGIN with exactly: Trip Plan for {base_city} to {destination_city}\n"
    "• Do NOT add any preface or explanation.\n"
    "• Do NOT add anything after the 'Next Steps' section.\n"
    "\n"
    "• Do not invent data; use values as provided. If unknown, use the exact placeholders below.\n"
    "\n"
    "OPTIONS SECTION — NON-NEGOTIABLE LAYOUT:\n"
    "• The “Options” section must contain EXACTLY THREE blocks, in this ORDER: Value, Fastest, Cheapest.\n"
    "• Each block header MUST be a bare line (no bullets, no colon, no extra words): one of: Value / Fastest / Cheapest.\n"
    "• Never repeat Value/Fastest/Cheapest headers anywhere; they appear once each, inside “Options”, in that order.\n"
    "• Inside each block render EXACTLY SEVEN bullet lines, in THIS order (no extras, no omissions, no repeats):\n"
    "  • Mode: <text>\n"
    "  • Depart: <datetime or —>\n"
    "  • Arrive: <datetime or —>\n"
    "  • Duration: <X.Y hours or —>\n"
    "  • Price: <₹ 12,345 or ₹ (no price available)>\n"
    "  • Rationale: <text or —>\n"
    "  • Booking Link: <url or empty>\n"
    "• Never output “• Value: …”, “• Fastest: …”, or “• Cheapest: …”.\n"
    "• Never output more than one “• Mode:” line in any block.\n"
    "• If Duration is a bare number, append “ hours”. If unknown, use “—”.\n"
    "• If price is missing/null, render exactly: “₹ (no price available)”.\n"
    "\n"
    "ROUTES & CONSTRAINTS:\n"
    "- If a route has note 'arrives_after_latest_arrival' or field after_window=true, set its Rationale to "
    "'Arrives after preferred window (optional)'.\n"
    "- If a route has duration_note='duration_unreliable':\n"
    "    • Render Duration as '—'\n"
    "    • Append 'Timing provisional—confirm on booking' to the Rationale\n"
    "- If price is missing or null, render exactly: '₹ (no price available)'.\n"
    "- Use the provided booking_link field for the 'Booking Link' line.\n"
    "- Do not invent data; use values as provided.\n"
    "- Datetimes are already formatted for display.\n"
    "\n"
    "STRUCTURE (EXACTLY):\n"
    "1) Title: “Trip Plan for {base_city} to {destination_city}”\n"
    "2) Section “Options” with three sub-sections in this order: Value, Fastest, Cheapest.\n"
    "   For EACH sub-section, render these lines (IN THIS ORDER), each prefixed with '• ' (bullet):\n"
    "     • Mode: <mode>\n"
    "     • Depart: <datetime>\n"
    "     • Arrive: <datetime>\n"
    "     • Duration: <hours or '—'>\n"
    "     • Price: <₹ ... or ₹ (no price available)>\n"
    "     • Rationale: <one short line; if not provided, write '—'>\n"
    "     • Booking Link: <booking_link>\n"
    "     • Transfer: <last_mile[Key].from> → <last_mile[Key].to> (<last_mile[Key].minutes> min by <last_mile[Key].mode>; ~₹ <last_mile[Key].approx_fare_inr>)\n"
    "     • Leave By: <last_mile[Key].leave_by_local or '—'>; ETA at venue: <last_mile[Key].eta_at_venue_local or '—'>\n"
    "     • Maps Link: <last_mile[Key].maps_link>\n"
    "   (Where Key is Value/Fastest/Cheapest matching the subsection.)\n"
    "3) Section “Hotels”\n"
    "   - List up to 5 hotels as '<Name>: ₹ <price>, <rating>/5'\n"
    "   - If a Booking Link is available for a hotel, add a new line immediately after: 'Booking Link: <url>'\n"
    "4) Section “Soumya's Suggested Plan”\n"
    "   - Render the selected route details with the SAME seven bullet lines as above (Mode..Booking Link)\n"
    "   - THEN add two bullets for hotel integration:\n"
    "     • Hotel Suggestion: <value_hotel_name> (₹ <price>, <rating>/5)\n"
    "     • Hotel Link: <value_hotel_link_or_city_link>\n"
    "   - THEN add the three last-mile bullets using last_mile['value']:\n"
    "     • Transfer: ...\n"
    "     • Leave By: ...\n"
    "     • Maps Link: ...\n"
    "5) Section “Cheapest Plan”\n"
    "   - Render the cheapest route details with the SAME seven bullet lines\n"
    "   - THEN add:\n"
    "     • Hotel Suggestion: <cheapest_hotel_name> (₹ <price>, <rating>/5)\n"
    "     • Hotel Link: <cheapest_hotel_link_or_city_link>\n"
    "   - THEN add the three last-mile bullets using last_mile['cheapest'].\n"
    "6) Section “Next Steps” (final; nothing after this)\n"
    "   - Exactly these bullets:\n"
    "     1) Book your {mode of travel} and hotel with the links\n"
    "     2) Reach the hotel and freshen up\n"
    "     3) Enjoy your trip\n"
    "   - {mode of travel} = Value's mode; if missing use Fastest; else Cheapest.\n"
    "\n"
    "MANDATORY SELF-CHECK (before sending):\n"
    "• Options has exactly three headers (Value, Fastest, Cheapest) in that order, each exactly once, no bullets/colons.\n"
    "• Each Options block has exactly seven bullets and only one “• Mode:”.\n"
    "• No Value/Fastest/Cheapest headers appear outside Options.\n"
    "• Duration units are normalized (e.g., “1.5 hours”) or “—”.\n"
    "• No extra text anywhere; sections appear only in the specified order.\n"
)

def _hotel_fallback_link(city: Optional[str]) -> str:
    slug = _slugify_city(city or "")
    return f"https://www.goibibo.com/hotels/hotels-in-{slug}-ct/" if slug else ""

def _hotel_line(name: str, price: Any, rating: Any) -> str:
    p = _format_inr(price)
    try:
        rt = f"{float(rating):.1f}/5"
    except Exception:
        rt = "—/5"
    return f"{name}: {p}, {rt}"

def _inject_plan_hotels(text: str,
                        dest_city: Optional[str],
                        hotel_value: Dict[str, Any],
                        hotel_cheapest: Dict[str, Any]) -> str:
    def has_hotel_lines(block: str) -> bool:
        return bool(re.search(r"Hotel Suggestion:", block)) and bool(re.search(r"Hotel Link:", block))
    def inject(block: str, which: str) -> str:
        if has_hotel_lines(block): return block
        h = hotel_value if which == "value" else hotel_cheapest
        if not h: return block
        name = h.get("name") or "Hotel"
        price = h.get("price_inr")
        rating = h.get("rating")
        link = h.get("booking_link") or _hotel_fallback_link(dest_city)
        lines = block.rstrip().splitlines()
        insert_at = len(lines)
        for i, ln in enumerate(lines[::-1]):
            if re.match(r"^\s*•?\s*Booking Link:", ln):
                insert_at = len(lines) - i
                break
        extra = [
            f"• Hotel Suggestion: {_hotel_line(name, price, rating)}",
            f"• Hotel Link: {link}" if link else ""
        ]
        extra = [e for e in extra if e]
        lines[insert_at:insert_at] = extra
        return "\n".join(lines)
    parts = re.split(r"(^\s*Soumya's Suggested Plan\s*$)", text, flags=re.M)
    if len(parts) >= 3:
        body_parts = re.split(r"(^\s*Cheapest Plan\s*$)", parts[2], flags=re.M)
        body_parts[0] = inject(body_parts[0], "value")
        if len(body_parts) >= 3:
            body_parts[2] = inject(body_parts[2], "cheapest")
            parts[2] = "".join(body_parts)
        else:
            parts[2] = "".join(body_parts)
        text = "".join(parts)
    return text

def add_last_mile_transfers(state: "AgentState") -> "AgentState":
    venue = (state.get("venue_text") or "").strip()
    if not venue:
        state["last_mile"] = {}
        return state

    def _calc_for(r: Dict[str, Any]) -> Dict[str, Any]:
        if not r:
            return {}
        origin = _arrival_point_name(r)
        minutes = _coarse_travel_minutes(origin, venue)
        fare = _rough_cab_fare_inr(minutes)
        maps = _google_maps_link(origin, venue, travelmode="driving")

        leave_by = None
        try:
            if state.get("meeting_start"):
                lb = state["meeting_start"]
                if lb.tzinfo is None:
                    lb = lb.replace(tzinfo=LOCAL_TZ)
                lb = lb - timedelta(minutes=minutes + 15)  # buffer
                leave_by = lb.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

        eta = None
        try:
            if r.get("arrive"):
                from dateutil import parser as dtp
                arr = dtp.isoparse(r["arrive"])
                arr = arr.replace(tzinfo=LOCAL_TZ) if arr.tzinfo is None else arr.astimezone(LOCAL_TZ)
                eta = (arr + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass

        return {
            "from": origin,
            "to": venue,
            "mode": "cab",
            "minutes": minutes,
            "approx_fare_inr": fare,
            "leave_by_local": leave_by,         # may be None
            "eta_at_venue_local": eta,          # may be None
            "maps_link": maps,
        }

    lm = {}
    for key in ("value", "fastest", "cheapest"):
        lm[key] = _calc_for(state.get(key) or {})
    state["last_mile"] = lm
    return state

def llm_itinerary(state: "AgentState") -> "AgentState":
    summary = {
        "base_city": state.get("base_city_norm") or state.get("base_city"),
        "destination_city": state.get("dest_city_norm") or state.get("destination_city"),
        "meeting_start": state.get("meeting_start").isoformat() if state.get("meeting_start") else None,
        "meeting_end": state.get("meeting_end").isoformat() if state.get("meeting_end") else None,
        "latest_arrival": state.get("latest_arrival").isoformat() if state.get("latest_arrival") else None,
        "best": {
            "cheapest": state.get("cheapest"),
            "fastest": state.get("fastest"),
            "value": state.get("value"),
        },
        "ordered_recommendations": state.get("ordered_recommendations", []),
        "top_hotels": (state.get("hotels") or [])[:5],
        "hotel_value": state.get("hotel_value") or {},
        "hotel_cheapest": state.get("hotel_cheapest") or {},
        "last_mile": state.get("last_mile") or {},
        "notes": state.get("debug", []),
        "links_provided": True,
        "iata": {"from": state.get("base_iata"), "to": state.get("dest_iata")},
    }
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "Turn the following search results into the described output:\n\n" + json.dumps(summary, ensure_ascii=False, indent=2)},
    ]
    try:
        client = OpenRouterClient()
        raw = client.chat(messages, model=LLM_MODEL, temperature=0.2, timeout=45)
        m = re.search(r"(Trip Plan for .*$)", raw, flags=re.S)
        text = m.group(1) if m else raw
        text = _normalize_option_headers(text)
        text = _normalize_money_block(text)
        text = _ensure_rationale_dash(text)
        text = _patch_options_and_plans(
            text,
            value=state.get("value") or {},
            fastest=state.get("fastest") or {},
            cheapest=state.get("cheapest") or {},
            hotel_value=state.get("hotel_value") or {},
            hotel_cheapest=state.get("hotel_cheapest") or {},
        )
        text = _enrich_rationales(text)
        text = _ensure_booking_links(
            text,
            base_city=summary["base_city"],
            dest_city=summary["destination_city"],
            base_iata=(state.get("base_iata") or None),
            dest_iata=(state.get("dest_iata") or None),
        )
        text = _ensure_option_subheaders(text)
        text = _normalize_duration_units(text) 
        text = _force_bullets_in_options(text)
        text = _bold_static_headers(text)
        text = _inject_plan_hotels(
            text,
            dest_city=summary["destination_city"],
            hotel_value=summary.get("hotel_value") or {},
            hotel_cheapest=summary.get("hotel_cheapest") or {},
        )
        state["itinerary_text"] = text
        state.setdefault("debug", []).append(f"LLM itinerary generated ({len(text)} chars).")
    except Exception as e:
        state.setdefault("debug", []).append(f"LLM error: {e}")
        state["itinerary_text"] = "Itinerary generation failed. See debug."
    return state

_INTENT_SYSTEM = (
    "Classify the user's message into one of: "
    "FETCH_INVITES, PLAN_TRIP, SET_BASE, SMALL_TALK, HELP, OTHER. "
    "If PLAN_TRIP, also extract the 1-based invite index number if any. "
    "If SET_BASE, extract {city} when present."
)

_FEW_SHOTS = [
    ("can you check my mail for meetings or invites?", {"intent": "FETCH_INVITES"}),
    ("fetch my emails", {"intent": "FETCH_INVITES"}),
    ("scan inbox for invitations", {"intent": "FETCH_INVITES"}),
    ("plan trip for invite 2", {"intent": "PLAN_TRIP", "index": 2}),
    ("plan for invite number 1", {"intent": "PLAN_TRIP", "index": 1}),
    ("set my base to indore", {"intent": "SET_BASE", "city": "Indore"}),
    ("i am currently in indore", {"intent": "SET_BASE", "city": "Indore"}),
    ("my location is indore", {"intent": "SET_BASE", "city": "Indore"}),
    ("i will travel from indore", {"intent": "SET_BASE", "city": "Indore"}),
    ("base indore", {"intent": "SET_BASE", "city": "Indore"}),
    ("what can you do", {"intent": "HELP"}),
    ("help", {"intent": "HELP"}),
    ("hi", {"intent": "SMALL_TALK"}),
    ("hello there", {"intent": "SMALL_TALK"}),
    ("how are you", {"intent": "SMALL_TALK"}),
    ("invite #3", {"intent": "PLAN_TRIP", "index": 3}),
    ("plan #2", {"intent": "PLAN_TRIP", "index": 2}),
    ("set base=blr", {"intent": "SET_BASE", "city": "BLR"}),
    ("from hyd", {"intent": "SET_BASE", "city": "Hyderabad"}),
]

FETCH_PAT = re.compile(r"\b(fetch|scan|check|get|look up|look into)\b.*\b(mail|mails|email|emails|inbox|invites?|invitations?)\b", re.I)
PLAN_PAT  = re.compile(r"\bplan\b.*\b(invite\s*)?(\d+)\b", re.I)
SET_PAT_1 = re.compile(r"\b(set|update|change)\b.*\b(base|home)\b.*?(?:to\s+)?([A-Za-z][A-Za-z\s]+)$", re.I)
SET_PAT_2 = re.compile(r"\b(?:i(?:'m| am)?|currently|now)\s+(?:in|at)\s+([A-Za-z][A-Za-z\s]+)$", re.I)
SET_PAT_3 = re.compile(r"\b(?:from|travel from)\s+([A-Za-z][A-Za-z\s]+)$", re.I)
SET_PAT_4 = re.compile(r"\b(?:my|current)\s+location\s+is\s+([A-Za-z][A-Za-z\s]+)$", re.I)
SET_PAT_5 = re.compile(r"\bbase\s+([A-Za-z][A-Za-z\s]+)$", re.I)
SMALL_TALK_PAT = re.compile(r"\b(hi|hii|hello|hey|good (?:morning|afternoon|evening)|how are you)\b", re.I)
QUESTION_PAT   = re.compile(r"^\s*(what|who|when|where|why|how)\b", re.I)
WEATHER_PAT    = re.compile(r"\bweather\b", re.I)
PLAN_IDX_ONLY_PAT = re.compile(r"\b(?:invite|#|number)?\s*(\d+)\b", re.I)

def _canonical_city_from_text(text: str) -> Optional[str]:
    tokens = re.findall(r"[A-Za-z]+", text)
    n = len(tokens)
    for i in range(n - 1):
        cand = f"{tokens[i]} {tokens[i+1]}"
        if cand.strip(): return cand.title()
    for t in tokens:
        if t.strip(): return t.title()
    return None

def _validated_set_base(city_phrase: Optional[str], full_utterance: str) -> Optional[str]:
    raw = (city_phrase or "").strip()
    if not raw:
        raw = (_canonical_city_from_text(full_utterance) or "").strip()
        if not raw:
            return None
    norm, _ = _normalize_city_phrase(raw)
    return norm or None

def llm_intent(user_msg: str) -> Dict[str, Any]:
    if QUESTION_PAT.search(user_msg) or WEATHER_PAT.search(user_msg):
        pass
    try:
        client = OpenRouterClient()
        shots = "\n".join([f"User: {u}\nLabel: {json.dumps(l)}" for u, l in _FEW_SHOTS])
        messages = [
            {"role": "system", "content": _INTENT_SYSTEM},
            {"role": "user", "content": shots + f"\n\nUser: {user_msg}\nLabel:"},
        ]
        out = client.chat(messages, model=LLM_MODEL, temperature=0.0, timeout=12)
        m = re.search(r"\{.*\}", out, re.S)
        if m:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict) and "intent" in parsed:
                if str(parsed.get("intent", "")).upper() == "SET_BASE":
                    safe_city = _validated_set_base(parsed.get("city"), user_msg)
                    if not safe_city: return {"intent": "OTHER"}
                    return {"intent": "SET_BASE", "city": safe_city}
                return parsed
    except Exception:
        pass
    if SMALL_TALK_PAT.search(user_msg): return {"intent": "SMALL_TALK"}
    if FETCH_PAT.search(user_msg): return {"intent": "FETCH_INVITES"}
    m = PLAN_PAT.search(user_msg)
    if m:
        try: return {"intent": "PLAN_TRIP", "index": int(m.group(2))}
        except Exception: pass
    for pat in (SET_PAT_1, SET_PAT_2, SET_PAT_3, SET_PAT_4, SET_PAT_5):
        mm = pat.search(user_msg)
        if mm:
            raw = mm.group(mm.lastindex or 1).strip()
            safe_city = _validated_set_base(raw, user_msg)
            if safe_city: return {"intent": "SET_BASE", "city": safe_city}
            break
    if QUESTION_PAT.search(user_msg) or WEATHER_PAT.search(user_msg):
        return {"intent": "OTHER"}
    return {"intent": "OTHER"}

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("gather_constraints", gather_constraints)
    g.add_node("plan_outbound", plan_outbound)
    g.add_node("suggest_hotels", suggest_hotels)
    g.add_node("rank_routes", rank_routes)
    g.add_node("add_last_mile_transfers", add_last_mile_transfers)
    g.add_node("llm_itinerary", llm_itinerary)
    g.set_entry_point("gather_constraints")
    g.add_edge("gather_constraints", "plan_outbound")
    g.add_edge("plan_outbound", "suggest_hotels")
    g.add_edge("suggest_hotels", "rank_routes")
    g.add_edge("rank_routes", "add_last_mile_transfers")
    g.add_edge("add_last_mile_transfers", "llm_itinerary") 
    g.add_edge("llm_itinerary", END)
    return g.compile()

def run_planner(tripfacts: Any) -> Dict[str, Any]:
    meeting = getattr(tripfacts, "meeting", None)
    state: AgentState = {
        "base_city": getattr(tripfacts, "base_city", None),
        "destination_city": getattr(tripfacts, "destination_city", None),
        "meeting_start": getattr(meeting, "start", None) if meeting else None,
        "meeting_end": getattr(meeting, "end", None) if meeting else None,
        "earliest_departure": getattr(tripfacts, "earliest_departure", None),
        "latest_arrival": getattr(tripfacts, "latest_arrival", None),
        "latest_return": getattr(tripfacts, "latest_return", None),
        "venue_text": getattr(meeting, "location_text", None) if meeting else None,
        "debug": [],
    }
    app = _build_graph()
    return app.invoke(state)

class ChatAgent:
    SOUMYA_READY = "I'm good — ready to plan your trip! You can say things like “fetch my invites” or “set my base to Indore.”"
    SOUMYA_IRRELEVANT = "I can only help you with planning your trip (fetch invites, set your base city, plan for an invite)."
    SOUMYA_NEED_BASE = "To plan your trip, I need your base city. Tell me something like **“set my base to Indore”** or **“I am currently in Pune.”**"

    def _try_plan_by_index(
        self, idx: int, session: Dict[str, Any], user_email: str,
        extract_trip_facts_fn, plan_fn,
    ) -> Dict[str, Any]:
        if not session.get("base_city"):
            session["awaiting_base_city"] = True
            session["pending_plan_index"] = idx
            return {"reply": self.SOUMYA_NEED_BASE}
        invites = session.get("invites") or []
        i = idx - 1
        if i < 0 or i >= len(invites):
            return {"reply": f"I only have {len(invites)} invite(s) cached. Pick 1…{len(invites)}."}
        facts = extract_trip_facts_fn(user_email=user_email, base_city=session.get("base_city"), emails=[invites[i]])
        result = plan_fn(facts[0])
        reply = result.get("itinerary_text") or "Itinerary ready."
        summary_json = result.get("summary_json") or {}
        return {"reply": reply, "summary": summary_json}

    def _maybe_extract_and_set_base(self, user_msg: str, session: Dict[str, Any]) -> Optional[str]:
        for pat in (SET_PAT_1, SET_PAT_2, SET_PAT_3, SET_PAT_4, SET_PAT_5):
            m = pat.search(user_msg)
            if m:
                raw = m.group(m.lastindex or 1).strip()
                city = _validated_set_base(raw, user_msg)
                if city:
                    session["base_city"] = city
                    return city
        return None

    def process(
        self, user_msg: str, session: Dict[str, Any], user_email: str,
        fetch_invites_fn, plan_fn, extract_trip_facts_fn,
    ) -> Dict[str, Any]:
        user_msg = (user_msg or "").strip()
        if not session.get("greeted"): session["greeted"] = True
        pre_city = self._maybe_extract_and_set_base(user_msg, session)
        if pre_city and session.get("awaiting_base_city"): session["awaiting_base_city"] = False
        parts: List[str] = []
        invites_payload = None
        wants_fetch = bool(FETCH_PAT.search(user_msg))
        m1 = PLAN_PAT.search(user_msg); m2 = PLAN_IDX_ONLY_PAT.search(user_msg)
        plan_idx = int(m1.group(2)) if m1 else (int(m2.group(1)) if m2 else None)
        if wants_fetch:
            invites = fetch_invites_fn()
            session["invites"] = invites
            invites_payload = invites
            parts.append(f"Here are your recent invites ({len(invites)} found):")
        if isinstance(plan_idx, int):
            if not session.get("base_city"): _ = self._maybe_extract_and_set_base(user_msg, session)
            if not session.get("invites"):
                invites = fetch_invites_fn()
                session["invites"] = invites
                if invites_payload is None:
                    invites_payload = invites
                    parts.append(f"Here are your recent invites ({len(invites)} found):")
            out = self._try_plan_by_index(plan_idx, session, user_email, extract_trip_facts_fn, plan_fn)
            if out.get("reply") != self.SOUMYA_NEED_BASE:
                parts.append(out.get("reply") or "Itinerary ready.")
                combined = "\n\n".join(parts) if parts else (out.get("reply") or "")
                payload: Dict[str, Any] = {"reply": combined}
                if invites_payload is not None: payload["invites"] = invites_payload
                payload["summary"] = out.get("summary")
                return payload
        if session.get("awaiting_base_city"):
            city = self._maybe_extract_and_set_base(user_msg, session)
            if city:
                session["awaiting_base_city"] = False
                pending_idx = session.pop("pending_plan_index", None)
                if isinstance(pending_idx, int):
                    return self._try_plan_by_index(pending_idx, session, user_email, extract_trip_facts_fn, plan_fn)
                return {"reply": f"Got it — base city set to **{city}**. How can I help you next?"}
            return {"reply": self.SOUMYA_NEED_BASE}
        if session.get("awaiting_plan_index"):
            m_only = PLAN_IDX_ONLY_PAT.search(user_msg)
            if m_only:
                try:
                    idx = int(m_only.group(1))
                    session["awaiting_plan_index"] = False
                    return self._try_plan_by_index(idx, session, user_email, extract_trip_facts_fn, plan_fn)
                except Exception:
                    pass
            return {"reply": "Please tell me the invite number (e.g., **2** or **invite 2**)."}
        intent = llm_intent(user_msg)
        intent_name = (intent.get("intent") or "").upper()
        if intent_name == "SMALL_TALK": return {"reply": self.SOUMYA_READY}
        if intent_name == "HELP":
            return {"reply": ("I can fetch your invites, set your base city, and plan a trip for a specific invite.\n"
                              "Try: **fetch my invites**, **set my base to Indore**, or **plan trip for invite 2**.")}
        if intent_name == "SET_BASE":
            city = (intent.get("city") or "").strip()
            if not city:
                city = (_canonical_city_from_text(user_msg or "") or "").strip()
                if not city:
                    return {"reply": self.SOUMYA_NEED_BASE}
            norm_city, _ = _normalize_city_phrase(city)
            session["base_city"] = norm_city
            pending_idx = session.pop("pending_plan_index", None)
            session["awaiting_base_city"] = False
            if isinstance(pending_idx, int):
                return self._try_plan_by_index(pending_idx, session, user_email, extract_trip_facts_fn, plan_fn)
            return {"reply": f"Got it — base city set to **{session['base_city']}**."}

        if intent_name == "FETCH_INVITES":
            invites = fetch_invites_fn()
            session["invites"] = invites
            return {"reply": f"Here are your recent invites ({len(invites)} found):", "invites": invites}
        if intent_name == "PLAN_TRIP":
            idx = intent.get("index")
            if isinstance(idx, int):
                return self._try_plan_by_index(idx, session, user_email, extract_trip_facts_fn, plan_fn)
            if not session.get("invites"):
                invites = fetch_invites_fn()
                session["invites"] = invites
                session["awaiting_plan_index"] = True
                invite_count = len(invites)
                prefix = f"Here are your recent invites ({invite_count} found):" if invite_count else "I couldn't find any invites right now."
                return {"reply": prefix + "\n\nPlease tell me the invite number (e.g., **2**).", "invites": invites}
            session["awaiting_plan_index"] = True
            return {"reply": "Which invite number should I plan for? (e.g., **2** or **invite 2**)."}
        return {"reply": self.SOUMYA_IRRELEVANT}

