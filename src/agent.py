from __future__ import annotations
from typing import TypedDict, Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta, date
import json
import re
from langgraph.graph import StateGraph, END
from providers.client import (flights_client, buses_client, trains_client, hotels_client)
from src.llm import OpenRouterClient, OpenRouterError, DEFAULT_MODEL 

def _ordered_picks(state: "AgentState") -> List[Tuple[str, Dict[str, Any]]]:
    return [
        ("Value", state.get("value") or {}),
        ("Fastest", state.get("fastest") or {}),
        ("Cheapest", state.get("cheapest") or {})
    ]

class AgentState(TypedDict, total=False):
    base_city: Optional[str]
    destination_city: Optional[str]
    meeting_start: Optional[datetime]
    meeting_end: Optional[datetime]
    earliest_departure: Optional[datetime]
    latest_arrival: Optional[datetime]
    latest_return: Optional[datetime]
    venue_text: Optional[str]

    flights: List[Dict[str, Any]]
    trains: List[Dict[str, Any]]
    buses: List[Dict[str, Any]]
    hotels: List[Dict[str, Any]]

    cheapest: Dict[str, Any]
    fastest: Dict[str, Any]
    value: Dict[str, Any]
    cost_effective: Dict[str, Any]

    ordered_recommendations: List[Tuple[str, Dict[str, Any]]]
    debug: List[str]
    itinerary_text: str
    summary_json: Dict[str, Any]

class ToolMux:
    def __init__(self):
        self.flights = flights_client()
        self.buses = buses_client()
        self.trains = trains_client()
        self.hotels = hotels_client()

    def call(self, tool: str, **kwargs):
        t = tool.lower()
        if "flight" in t:
            return self.flights.call(tool, **kwargs)
        if "hotel" in t:
            return self.hotels.call(tool, **kwargs)
        if "train" in t:
            return self.trains.call(tool, **kwargs)
        if "bus" in t:
            return self.buses.call(tool, **kwargs)
        raise ValueError(f"Unknown tool namespace for: {tool}")

_tools = ToolMux()

def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None

_IATA = {
    "mumbai": "BOM", "bombay": "BOM",
    "delhi": "DEL", "new delhi": "DEL", "noida": "DEL",
    "gurgaon": "DEL", "gurugram": "DEL",
    "bengaluru": "BLR", "bangalore": "BLR",
    "hyderabad": "HYD",
    "chennai": "MAA",
    "pune": "PNQ",
    "kolkata": "CCU", "calcutta": "CCU",
    "ahmedabad": "AMD",
    "jaipur": "JAI",
    "lucknow": "LKO",
    "indore": "IDR",
    "bhopal": "BHO",
    "kochi": "COK",
    "coimbatore": "CJB",
    "nagpur": "NAG",
    "goa": "GOI",
    "surat": "STV",
}

def _iata(code_or_city: Optional[str]) -> Optional[str]:
    if not code_or_city:
        return None
    s = code_or_city.strip()
    if len(s) == 3 and s.isalpha():
        return s.upper()
    return _IATA.get(s.lower())

def _clamp_future(d: date) -> date:
    today = date.today()
    return d if d >= today else today

def _choose_outbound_date(state: AgentState) -> str:
    if state.get("earliest_departure"):
        return _clamp_future(state["earliest_departure"].date()).isoformat()
    if state.get("meeting_start"):
        return _clamp_future(state["meeting_start"].date()).isoformat()
    return date.today().isoformat()

def gather_constraints(state: AgentState) -> AgentState:
    if state.get("meeting_start") and not state.get("meeting_end"):
        state["meeting_end"] = state["meeting_start"] + timedelta(hours=2)
    state.setdefault("debug", [])
    return state

def plan_outbound(state: AgentState) -> AgentState:
    src_city = state.get("base_city")
    dst_city = state.get("destination_city")
    dbg = state.setdefault("debug", [])

    dep_id = _iata(src_city)
    arr_id = _iata(dst_city)
    flights: List[Dict[str, Any]] = []
    if not dep_id or not arr_id:
        dbg.append(f"Flights skipped: missing IATA (src={src_city}→{dep_id}, dst={dst_city}→{arr_id}).")
    else:
        outbound_date = _choose_outbound_date(state)
        f_search = _tools.call(
            "search_flights",
            departure_id=dep_id,
            arrival_id=arr_id,
            outbound_date=outbound_date,
            trip_type=2,
            currency="INR",
            country="in",
            language="en",
            adults=1,
            travel_class=1,
        )
        if isinstance(f_search, dict) and "search_id" in f_search:
            f_norm = _tools.call("normalize_flights", search_id=f_search["search_id"])
            if isinstance(f_norm, list):
                flights = f_norm
                dbg.append(f"Flights: {len(flights)} options on {outbound_date}.")
            else:
                dbg.append("Flights normalize returned non-list/empty.")
        else:
            dbg.append(f"Flights search returned no search_id: {f_search!r}")

    trains: List[Dict[str, Any]] = []
    t_search = _tools.call(
        "search_trains",
        src_city=src_city,
        dst_city=dst_city,
        earliest_departure=_iso(state.get("earliest_departure")),
        latest_arrival=_iso(state.get("latest_arrival")),
    )
    if isinstance(t_search, dict) and "search_id" in t_search:
        t_norm = _tools.call("normalize_trains", search_id=t_search["search_id"])
        if isinstance(t_norm, list):
            trains = t_norm
            dbg.append(f"Trains: {len(trains)} options.")
        else:
            dbg.append("Trains normalize returned non-list/empty.")
    else:
        dbg.append(f"Trains search returned no search_id: {t_search!r}")

    buses: List[Dict[str, Any]] = []
    b_search = _tools.call(
        "search_buses",
        src_city=src_city,
        dst_city=dst_city,
        earliest_departure=_iso(state.get("earliest_departure")),
        latest_arrival=_iso(state.get("latest_arrival")),
    )
    if isinstance(b_search, dict) and "search_id" in b_search:
        b_norm = _tools.call("normalize_buses", search_id=b_search["search_id"])
        if isinstance(b_norm, list):
            buses = b_norm
            dbg.append(f"Buses: {len(buses)} options.")
        else:
            dbg.append("Buses normalize returned non-list/empty.")
    else:
        dbg.append(f"Buses search returned no search_id: {b_search!r}")

    state["flights"], state["trains"], state["buses"] = flights, trains, buses
    return state

def plan_return(state: AgentState) -> AgentState:
    return state

def suggest_hotels(state: AgentState) -> AgentState:
    checkin = state.get("meeting_start")
    checkout = state.get("meeting_end") or (checkin + timedelta(hours=2) if checkin else None)

    if checkin and checkin.date() < date.today():
        checkin = datetime.combine(date.today(), checkin.time())
    if checkout and checkout.date() < date.today():
        checkout = datetime.combine(date.today() + timedelta(days=1), checkout.time())

    hotels: List[Dict[str, Any]] = []
    h_search = _tools.call(
        "search_hotels",
        city=state.get("destination_city"),
        near=state.get("venue_text"),
        checkin=_iso(checkin),
        checkout=_iso(checkout),
        adults=1,
        currency="INR",
        country="in",
        language="en",
    )
    if isinstance(h_search, dict) and "search_id" in h_search:
        h_norm = _tools.call("normalize_hotels", search_id=h_search["search_id"])
        if isinstance(h_norm, list):
            hotels = h_norm
            state.setdefault("debug", []).append(f"Hotels: {len(hotels)} options.")
        else:
            state.setdefault("debug", []).append("Hotels normalize returned non-list/empty.")
    else:
        state.setdefault("debug", []).append(f"Hotels search returned no search_id: {h_search!r}")

    state["hotels"] = hotels
    return state

def _num_or(val: Any, default: float) -> float:
    if isinstance(val, (int, float)) and val is not None:
        return float(val)
    return float(default)

def _safe_price(r: Dict[str, Any], default: float = 1e9) -> float:
    return _num_or(r.get("price_inr"), default)

def _safe_dur(r: Dict[str, Any], default: float = 1e9) -> float:
    return _num_or(r.get("duration_hours"), default)

def _pick_cheapest(routes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    return min(routes, key=lambda r: _safe_price(r, 1e9)) if routes else None

def _pick_fastest(routes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    return min(routes, key=lambda r: _safe_dur(r, 1e9)) if routes else None

def _score_value(r: Dict[str, Any]) -> float:
    price = _safe_price(r, 1e9)
    dur = _safe_dur(r, 24.0)
    return price * (dur ** 0.7)

def _pick_value(routes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    return min(routes, key=_score_value) if routes else None

def _pick_cost_effective(routes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    cheapest = _pick_cheapest(routes)
    fastest = _pick_fastest(routes)
    if not cheapest or not fastest:
        return cheapest or fastest
    c_price = _safe_price(cheapest, 1e9)
    f_price = _safe_price(fastest, 1e9)
    c_dur = _safe_dur(cheapest, 1e9)
    f_dur = _safe_dur(fastest, 1e9)
    price_gap = f_price - c_price
    time_save = c_dur - f_dur
    if price_gap <= 1500 and time_save >= 4.0:
        return fastest
    return cheapest

def rank_routes(state: AgentState) -> AgentState:
    la = state.get("latest_arrival")
    dbg = state.setdefault("debug", [])

    def _merge(apply_la: bool, relax_non_flights_only: bool = False) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for coll in ("flights", "trains", "buses"):
            for r in state.get(coll, []) or []:
                try:
                    arr = datetime.fromisoformat(r["arrive"])
                except Exception:
                    continue
                violates = False
                if apply_la and la:
                    if relax_non_flights_only:
                        if r.get("mode") == "flight" and arr > la:
                            violates = True
                    else:
                        if arr > la:
                            violates = True
                if violates:
                    continue
                merged.append(r)
        return merged

    merged = _merge(apply_la=True, relax_non_flights_only=False)
    only_flights = all(r.get("mode") == "flight" for r in merged) if merged else True

    if not merged or only_flights:
        relaxed_bt = _merge(apply_la=True, relax_non_flights_only=True)
        if la:
            for r in relaxed_bt:
                try:
                    if r.get("mode") in ("bus", "train") and datetime.fromisoformat(r["arrive"]) > la:
                        r.setdefault("notes", []).append("arrives_after_latest_arrival")
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
    state["value"] = _pick_value(merged) or {}
    state["cost_effective"] = _pick_cost_effective(merged) or {}
    state["ordered_recommendations"] = _ordered_picks(state)
    state["summary_json"] = {
        "cheapest": state["cheapest"],
        "fastest": state["fastest"],
        "value": state["value"],
        "hotels_top5": (state.get("hotels") or [])[:5],
        "debug": state.get("debug", []),
    }
    if not merged:
        dbg.append("No routes merged after filters (all empty).")
    return state

# ------------------------ UPDATED PROMPT ONLY ------------------------
_SYSTEM_PROMPT = (
    "You are TripPlanner, an assistant that crafts concise, decision-ready trip plans.\n"
    "- Respect meeting timing when possible; if a bus/train arrives after the preferred window, call it 'optional'.\n"
    "- Compare tradeoffs clearly. Favor common sense.\n"
    "- Output: brief intro + three labeled options (Cheapest / Fastest / Value), each with:\n"
    "  • mode, depart→arrive, duration, price, one-line rationale.\n"
    "  • A REQUIRED 'Booking Link' line that depends on the mode:\n"
    "    - If mode is BUS: add a redBus link with base city and destination prefilled.\n"
    "      Format: https://www.redbus.in/bus-tickets/{from_slug}-to-{to_slug}\n"
    "      where from_slug/to_slug = lowercase(city), spaces→'-', remove non-alphanumerics, collapse dashes.\n"
    "    - If mode is FLIGHT: add a MakeMyTrip link with base and destination and (if available) departure date.\n"
    "      Primary format (if you know depart date from the route):\n"
    "        https://www.makemytrip.com/flight/search?itinerary={FROM}-{TO}-{DD/MM/YYYY}&tripType=O&paxType=A-1_C-0_I-0&intl=false&cabinClass=E\n"
    "      Use IATA codes if known in the summary; else use city codes in uppercase (e.g., DEL, BOM). If date unknown, omit it and add a general link:\n"
    "        https://www.makemytrip.com/flights/{FROM}-to-{TO}-flights.html\n"
    "    - If mode is TRAIN: add an IRCTC link with base and destination referenced.\n"
    "        https://www.irctc.co.in/nget/train-search?from={BASE_CITY}&to={DESTINATION_CITY}\n"
    "      (Note: site may still need manual selection; link is for convenience.)\n"
    "  • Ensure the 'Booking Link' appears on its own line under each option.\n"
    "- Hotels: If hotels exist, list up to 5 with price/rating. For EACH hotel line, append a Goibibo link for the destination city:\n"
    "    https://www.goibibo.com/hotels/hotels-in-{dest_slug}-ct/\n"
    "  where dest_slug = lowercase(destination_city), spaces→'-', remove non-alphanumerics, collapse dashes.\n"
    "- Keep markdown tidy. Use real city names from the summary: base_city and destination_city. If any info is missing, still render the link using available placeholders.\n"
    "- End with a short next-steps checklist."
    '''- End with a "Next Steps" section as the final section (do not add anything after it). Use exactly these three bullets, nothing else:
            1) Book your {mode of travel} and hotel with the links
            2) Reach the hotel and freshen up
            3) Enjoy your trip
        - Set {mode of travel} to the mode used in the Value option; if Value is missing, use Fastest; if that’s also missing, use Cheapest.
        - Structure the output exactly in this order and with these rules:
        1) Title: “Trip Plan for {base_city} to {destination_city}”
        2) Section “Options” with three sub-sections in this order: Value, Fastest, Cheapest.
            For EACH of these three sub-sections, include ONLY:
            • Mode
            • Depart
            • Arrive
            • Duration
            • Price
            • Rationale (one short line)
            • A single “Booking Link:” line based on the mode:
                - BUS → https://www.redbus.in/bus-tickets/{from_slug}-to-{to_slug}
                - FLIGHT → https://www.makemytrip.com/flight/search?itinerary={FROM}-{TO}-{DD/MM/YYYY}&tripType=O&paxType=A-1_C-0_I-0&intl=false&cabinClass=E
                            (if date unknown, use https://www.makemytrip.com/flights/{FROM}-to-{TO}-flights.html)
                - TRAIN → https://www.irctc.co.in/nget/train-search?from={BASE_CITY}&to={DESTINATION_CITY}
            IMPORTANT: Do NOT list any hotel lines inside Cheapest/Fastest/Value. No “Hotel Options” under these.

        3) Section “Hotels”
            - List up to 5 hotels (name, price, rating).
            - For each hotel add one “* Booking Link:” line that points to:
            https://www.goibibo.com/hotels/hotels-in-{dest_slug}-ct/
            - This “Hotels” section appears ONCE only (do not repeat it under other sections).

        4) Section “Soumya's Suggested Plan”
            - Show ONLY the selected route details (mode, depart, arrive, duration, price, rationale).
            - Add ONE “Booking Link:” line for that route (same rules as above).
            - Do NOT include any hotel lines here.

        5) Section “Cheapest Plan”
            - Show ONLY the cheapest route details (mode, depart, arrive, duration, price, rationale).
            - Add ONE “Booking Link:” line for that route.
            - Do NOT include any hotel lines here.

        6) Section “Next Steps” (FINAL section; do not add anything after this)
            - Render exactly these three bullets, nothing else:
            1) Book your {mode of travel} and hotel with the links
            2) Reach the hotel and freshen up
            3) Enjoy your trip
            - {mode of travel} = use the mode from the Value option; if missing use Fastest; else Cheapest.

        - Slug rules:
        - {from_slug}/{to_slug}/{dest_slug}: lowercase city, spaces→'-', remove non-alphanumerics, collapse dashes.
        - {FROM}/{TO}: IATA codes when available; else uppercase city codes.
        - {DD/MM/YYYY}: use the route’s depart date if available.
'''
)
# --------------------------------------------------------------------

def llm_itinerary(state: "AgentState") -> "AgentState":
    summary = {
        "base_city": state.get("base_city"),
        "destination_city": state.get("destination_city"),
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
        "notes": state.get("debug", []),
    }
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "Turn the following search results into the described output:\n\n" + json.dumps(summary, ensure_ascii=False, indent=2)},
    ]
    try:
        client = OpenRouterClient()
        text = client.chat(messages, model=DEFAULT_MODEL, temperature=0.2, timeout=45)
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
PLAN_PAT  = re.compile(r"\bplan\b.*\b(invite\s*)?(\d+)\b", re.I)
PLAN_IDX_ONLY_PAT = re.compile(r"\b(?:invite|#|number)?\s*(\d+)\b", re.I)

def _canonical_city_from_text(text: str) -> Optional[str]:
    tokens = re.findall(r"[A-Za-z]+", text)
    n = len(tokens)
    for i in range(n - 1):
        cand = f"{tokens[i]} {tokens[i+1]}"
        if _iata(cand):
            return cand.title()
    for t in tokens:
        if _iata(t):
            return t.title()
    return None

def _validated_set_base(city_phrase: Optional[str], full_utterance: str) -> Optional[str]:
    if city_phrase:
        city = _canonical_city_from_text(city_phrase)
        if city:
            return city
    city = _canonical_city_from_text(full_utterance)
    return city

def llm_intent(user_msg: str) -> Dict[str, Any]:
    if QUESTION_PAT.search(user_msg) or WEATHER_PAT.search(user_msg):
        pass

    try:
        client = OpenRouterClient()
        shots = "\n".join([f"User: {u}\nLabel: {json.dumps(l)}" for u,l in _FEW_SHOTS])
        messages = [
            {"role": "system", "content": _INTENT_SYSTEM},
            {"role": "user", "content": shots + f"\n\nUser: {user_msg}\nLabel:"},
        ]
        out = client.chat(messages, model=DEFAULT_MODEL, temperature=0.0, timeout=12)
        m = re.search(r"\{.*\}", out, re.S)
        if m:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict) and "intent" in parsed:
                if str(parsed.get("intent", "")).upper() == "SET_BASE":
                    safe_city = _validated_set_base(parsed.get("city"), user_msg)
                    if not safe_city:
                        return {"intent": "OTHER"}
                    return {"intent": "SET_BASE", "city": safe_city}
                return parsed
    except Exception:
        pass

    if SMALL_TALK_PAT.search(user_msg):
        return {"intent": "SMALL_TALK"}

    if FETCH_PAT.search(user_msg):
        return {"intent": "FETCH_INVITES"}

    m = PLAN_PAT.search(user_msg)
    if m:
        try:
            return {"intent": "PLAN_TRIP", "index": int(m.group(2))}
        except Exception:
            pass

    for pat in (SET_PAT_1, SET_PAT_2, SET_PAT_3, SET_PAT_4, SET_PAT_5):
        mm = pat.search(user_msg)
        if mm:
            raw = mm.group(mm.lastindex or 1).strip()
            safe_city = _validated_set_base(raw, user_msg)
            if safe_city:
                return {"intent": "SET_BASE", "city": safe_city}
            break

    if QUESTION_PAT.search(user_msg) or WEATHER_PAT.search(user_msg):
        return {"intent": "OTHER"}

    return {"intent": "OTHER"}

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("gather_constraints", gather_constraints)
    g.add_node("plan_outbound", plan_outbound)
    g.add_node("plan_return", plan_return)
    g.add_node("suggest_hotels", suggest_hotels)
    g.add_node("rank_routes", rank_routes)
    g.add_node("llm_itinerary", llm_itinerary)
    g.set_entry_point("gather_constraints")
    g.add_edge("gather_constraints", "plan_outbound")
    g.add_edge("plan_outbound", "plan_return")
    g.add_edge("plan_return", "suggest_hotels")
    g.add_edge("suggest_hotels", "rank_routes")
    g.add_edge("rank_routes", "llm_itinerary")
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
    SOUMYA_GREETING = (
        "Hi, I'm **Soumya**, your trip planner. I can fetch your invites, set your base city, "
        "and plan an itinerary for a selected invite. How can I help you today?"
    )
    SOUMYA_READY = "I'm good — ready to plan your trip! You can say things like “fetch my invites” or “set my base to Indore.”"
    SOUMYA_IRRELEVANT = "I can only help you with planning your trip (fetch invites, set your base city, plan for an invite)."
    SOUMYA_NEED_BASE = "To plan your trip, I need your base city. Tell me something like **“set my base to Indore”** or **“I am currently in Pune.”**"

    def _try_plan_by_index(
        self,
        idx: int,
        session: Dict[str, Any],
        user_email: str,
        extract_trip_facts_fn,
        plan_fn,
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
        best_value = (summary_json.get("value") if isinstance(summary_json, dict) else None) or {}
        cheapest_route = (summary_json.get("cheapest") if isinstance(summary_json, dict) else None) or {}
        hotels_top = (summary_json.get("hotels_top5") if isinstance(summary_json, dict) else None) or []
    
        def _fmt_price(v):
            try:
                return f"₹{int(float(v)):,.0f}"
            except Exception:
                return None

        def _fmt_route_line(prefix: str, r: Dict[str, Any]) -> Optional[str]:
            if not r:
                return None
            mode = (r.get("mode") or "Route").title()
            depart = r.get("depart")
            arrive = r.get("arrive")
            dur = r.get("duration_hours")
            price = _fmt_price(r.get("price_inr"))
            parts = []
            if depart and arrive: parts.append(f"{depart} → {arrive}")
            if dur is not None: parts.append(f"{dur} h")
            if price: parts.append(price)
            return f"- {prefix}: **{mode}**" + (f" — {', '.join(parts)}" if parts else "")

        def _pick_good_hotel(hotels: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not hotels:
                return {}
            rated = [h for h in hotels if (h.get("rating") or 0) >= 4.0]
            pool = rated if rated else hotels
            try:
                return min(pool, key=lambda h: float(h.get("price_inr") or 1e12))
            except Exception:
                return pool[0]

        def _pick_cheapest_hotel(hotels: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not hotels:
                return {}
            try:
                return min(hotels, key=lambda h: float(h.get("price_inr") or 1e12))
            except Exception:
                return hotels[0]

        def _fmt_hotel_line(h: Dict[str, Any]) -> Optional[str]:
            if not h:
                return None
            name = h.get("name") or "Hotel (TBD)"
            rating = h.get("rating")
            price = _fmt_price(h.get("price_inr"))
            bits = [f"**{name}**"]
            if rating: bits.append(f"★{rating}")
            if price: bits.append(price)
            return "- Hotel: " + " — ".join(bits)

        good_hotel = _pick_good_hotel(hotels_top)
        cheap_hotel = _pick_cheapest_hotel(hotels_top)

        sug_lines = [l for l in (_fmt_route_line("Route", best_value), _fmt_hotel_line(good_hotel)) if l]
        cheap_lines = [l for l in (_fmt_route_line("Route", cheapest_route), _fmt_hotel_line(cheap_hotel)) if l]

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
        city = _canonical_city_from_text(user_msg)
        if city:
            session["base_city"] = city
            return city
        return None

    def process(
        self,
        user_msg: str,
        session: Dict[str, Any],
        user_email: str,
        fetch_invites_fn,
        plan_fn,
        extract_trip_facts_fn,
    ) -> Dict[str, Any]:
        user_msg = (user_msg or "").strip()

        if not session.get("greeted"):
            session["greeted"] = True

        pre_city = self._maybe_extract_and_set_base(user_msg, session)
        if pre_city and session.get("awaiting_base_city"):
            session["awaiting_base_city"] = False

        parts: List[str] = []
        invites_payload = None

        wants_fetch = bool(FETCH_PAT.search(user_msg))
        plan_match = PLAN_PAT.search(user_msg) or PLAN_IDX_ONLY_PAT.search(user_msg)
        plan_idx = None
        m1 = PLAN_PAT.search(user_msg)
        m2 = PLAN_IDX_ONLY_PAT.search(user_msg)
        if m1:
            plan_idx = int(m1.group(2))
        elif m2:
            plan_idx = int(m2.group(1))

        if wants_fetch:
            invites = fetch_invites_fn()
            session["invites"] = invites
            invites_payload = invites
            parts.append(f"Here are your recent invites ({len(invites)} found):")

        if isinstance(plan_idx, int):
            if not session.get("base_city"):
                pre_city2 = self._maybe_extract_and_set_base(user_msg, session)
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
                if invites_payload is not None:
                    payload["invites"] = invites_payload
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

        if intent_name == "SMALL_TALK":
            return {"reply": self.SOUMYA_READY}

        if intent_name == "HELP":
            return {
                "reply": (
                    "I can fetch your invites, set your base city, and plan a trip for a specific invite.\n"
                    "Try: **fetch my invites**, **set my base to Indore**, or **plan trip for invite 2**."
                )
            }

        if intent_name == "SET_BASE":
            city = intent.get("city")
            if not city or not _iata(city):
                city = _canonical_city_from_text(user_msg or "")
                if not city:
                    return {"reply": self.SOUMYA_NEED_BASE}
            session["base_city"] = city
            pending_idx = session.pop("pending_plan_index", None)
            session["awaiting_base_city"] = False
            if isinstance(pending_idx, int):
                return self._try_plan_by_index(pending_idx, session, user_email, extract_trip_facts_fn, plan_fn)
            return {"reply": f"Got it — base city set to **{city}**."}

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
                prefix = f"Here are your recent invites ({invite_count} found):" if invite_count else "I couldn’t find any invites right now."
                return {"reply": prefix + "\n\nPlease tell me the invite number (e.g., **2**).", "invites": invites}
            session["awaiting_plan_index"] = True
            return {"reply": "Which invite number should I plan for? (e.g., **2** or **invite 2**)."}

        return {"reply": self.SOUMYA_IRRELEVANT}

