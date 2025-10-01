from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import os
import json
from dateutil import parser as dtparser
from dateutil import tz as dateutil_tz
from utils.schemas import TripFacts, Meeting
from src.llm import OpenRouterClient, DEFAULT_MODEL

LLM_EXTRACT_ENABLED = os.getenv("LLM_EXTRACT_ENABLED", "true").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_MODEL)
LLM_TIMEOUT_SECS = int(os.getenv("LLM_TIMEOUT_SECS", "18"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
MAX_TEXT_CHARS = int(os.getenv("MAX_BODY_CHARS", "8000"))
LOCAL_TZ_NAME = os.getenv("TIMEZONE", "Asia/Kolkata")
LOCAL_TZ = dateutil_tz.gettz(LOCAL_TZ_NAME) or dateutil_tz.gettz("Asia/Kolkata")
DEPARTURE_BUFFER_HOURS = float(os.getenv("DEPARTURE_BUFFER_HOURS", "12"))   # earliest_departure = start - 12h
ARRIVAL_BUFFER_HOURS   = float(os.getenv("ARRIVAL_BUFFER_HOURS", "3"))      # latest_arrival   = start - 3h
RETURN_BUFFER_HOURS    = float(os.getenv("RETURN_BUFFER_HOURS", "7"))       # latest_return    = start + 7h

_LLM_SYSTEM = (
    "You extract trip-related facts from emails that describe meetings, visits, events or appointments.\n"
    "Return STRICT JSON ONLY with this schema:\n"
    "{\n"
    '  "title": string|null,\n'
    '  "destination_city": string|null,\n'
    '  "venue_text": string|null,\n'
    '  "venue_city": string|null,\n'
    '  "venue_state": string|null,\n'
    '  "venue_country": string|null,\n'
    '  "pincode": string|null,\n'
    '  "is_physical": true|false,\n'
    '  "is_online": true|false,\n'
    '  "online_platform": "zoom|teams|meet|webex|other|null",\n'
    '  "when": {\n'
    '    "start_iso": string|null,   // ISO8601 if confidently normalized; include timezone if known\n'
    '    "end_iso": string|null,\n'
    '    "raw": string|null,         // the original date/time phrase if not normalizable\n'
    '    "timezone": string|null     // e.g., IST, Asia/Kolkata, PST, etc.\n'
    "  }\n"
    "}\n"
    "- Do NOT hallucinate. If you cannot confidently identify a field, set it to null.\n"
    "- Prefer the MOST LIKELY single meeting slot if multiple are listed.\n"
    "- If only relative time is present (e.g., 'tomorrow 3 PM'), include it in 'raw' and set start_iso=null.\n"
)

def _llm_extract(subject: str, body: str, snippet: str | None = None) -> Optional[dict]:
    """Call the LLM once to extract meeting/venue/date facts."""
    if not LLM_EXTRACT_ENABLED:
        return None
    text = {
        "subject": (subject or "")[:512],
        "snippet": (snippet or "")[:768],
        "body": (body or "")[:MAX_TEXT_CHARS],
    }
    client = OpenRouterClient()
    messages = [
        {"role": "system", "content": _LLM_SYSTEM},
        {"role": "user", "content": json.dumps(text, ensure_ascii=False)},
    ]
    out = client.chat(messages, model=LLM_MODEL, temperature=LLM_TEMPERATURE, timeout=LLM_TIMEOUT_SECS)
    try:
        return json.loads(out)
    except Exception:
        try:
            s = out.strip()
            i, j = s.find("{"), s.rfind("}")
            if i != -1 and j != -1 and j > i:
                return json.loads(s[i:j+1])
        except Exception:
            return None
    return None

def _parse_to_local(iso_str: Optional[str], raw_hint: Optional[str]) -> Optional[datetime]:
    """Parse an ISO string (or a raw phrase) into a timezone-aware datetime in LOCAL_TZ."""
    if iso_str:
        try:
            dt = dtparser.isoparse(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=LOCAL_TZ)
            return dt.astimezone(LOCAL_TZ)
        except Exception:
            pass
    if raw_hint:
        try:
            dt = dtparser.parse(raw_hint, fuzzy=True, dayfirst=False)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=LOCAL_TZ)
            return dt.astimezone(LOCAL_TZ)
        except Exception:
            return None
    return None

def _derive_windows(start_dt: Optional[datetime]) -> tuple[Optional[datetime], Optional[datetime], Optional[datetime]]:
    if not start_dt:
        return None, None, None
    earliest_departure = start_dt - timedelta(hours=DEPARTURE_BUFFER_HOURS)
    latest_arrival     = start_dt - timedelta(hours=ARRIVAL_BUFFER_HOURS)
    latest_return      = start_dt + timedelta(hours=RETURN_BUFFER_HOURS)
    return earliest_departure, latest_arrival, latest_return

def _first_non_empty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v and isinstance(v, str) and v.strip():
            return v.strip()
    return None

def extract_one_email(user_email: str, base_city: Optional[str], email_obj: Dict[str, Any]) -> TripFacts:
    """
    Extract TripFacts from a single email dict:
       {id, subject, snippet, body, internalDate, invite_fact?}
    Prefers 'invite_fact' (attached by gmail_filter.py). If absent, calls LLM here.
    """
    subject = email_obj.get("subject", "") or ""
    snippet = email_obj.get("snippet", "") or ""
    body = (email_obj.get("body") or "")[:MAX_TEXT_CHARS]
    fact = email_obj.get("invite_fact") if isinstance(email_obj, dict) else None

    if not fact:
        fact = _llm_extract(subject=subject, body=body, snippet=snippet) or {}
    ev = fact if fact else {}
    if "event" in ev and isinstance(ev["event"], dict):
        event = ev["event"]
        title = event.get("title")
        venue_text = event.get("venue_text")
        venue_city = event.get("venue_city")
        venue_state = event.get("venue_state")
        venue_country = event.get("venue_country")
        pincode = event.get("pincode")
        online_platform = event.get("online_platform")
        when = event.get("when") or {}
        start_iso = when.get("start_iso")
        end_iso = when.get("end_iso")
        when_raw = when.get("raw")
        tz_hint = when.get("timezone")
        destination_city = event.get("venue_city") or None
    else:
        title = ev.get("title")
        venue_text = ev.get("venue_text")
        venue_city = ev.get("venue_city")
        venue_state = ev.get("venue_state")
        venue_country = ev.get("venue_country")
        pincode = ev.get("pincode")
        online_platform = ev.get("online_platform")
        when = ev.get("when") or {}
        start_iso = when.get("start_iso") if isinstance(when, dict) else ev.get("start_iso")
        end_iso = when.get("end_iso")   if isinstance(when, dict) else ev.get("end_iso")
        when_raw = when.get("raw")      if isinstance(when, dict) else ev.get("when_raw")
        tz_hint = when.get("timezone")  if isinstance(when, dict) else ev.get("timezone")
        destination_city = ev.get("destination_city") or ev.get("venue_city") or None

    title = _first_non_empty(title, subject, "Meeting")
    meet_start = _parse_to_local(start_iso, when_raw)
    meet_end = _parse_to_local(end_iso, None)
    dest_city = destination_city or venue_city
    meet_obj: Optional[Meeting] = None
    if meet_start:
        meet_obj = Meeting(
            title=title[:140] if title else "Meeting",
            start=meet_start,
            end=meet_end,
            location_text=venue_text,
            city=dest_city,
        )
    earliest_departure, latest_arrival, latest_return = _derive_windows(meet_start)

    notes: List[str] = []
    if not dest_city:
        notes.append("Destination city not confidently detected.")
    if not meet_start:
        notes.append("Meeting date/time not confidently detected.")
    if venue_text and dest_city and dest_city.lower() not in venue_text.lower():
        notes.append("Venue found (may be neighborhood/landmark); verify it matches destination city.")
    if tz_hint and isinstance(tz_hint, str) and tz_hint.strip():
        notes.append(f"Time interpreted in local zone ({LOCAL_TZ_NAME}); original tz hint: {tz_hint.strip()}")
    if online_platform:
        notes.append(f"Online platform noted: {online_platform}")

    return TripFacts(
        user_email=user_email,
        base_city=base_city,
        destination_city=dest_city,
        meeting=meet_obj,
        earliest_departure=earliest_departure,
        latest_arrival=latest_arrival,
        latest_return=latest_return,
        notes=notes,
    )

def extract_trip_facts(user_email: str, base_city: Optional[str], emails: List[Dict[str, Any]]) -> List[TripFacts]:
    out: List[TripFacts] = []
    for m in emails:
        try:
            out.append(extract_one_email(user_email, base_city, m))
        except Exception as e:
            tf = TripFacts(user_email=user_email, base_city=base_city, notes=[f"Extraction error: {e}"])
            out.append(tf)
    return out

