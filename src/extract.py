from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime, time, timedelta
import re
from dateutil import parser as dtparser
from utils.schemas import TripFacts, Meeting

CITY_PAT = re.compile(
    r"\b("
    r"New Delhi|Delhi|Mumbai|Bombay|Bengaluru|Bangalore|Hyderabad|Chennai|Pune|Kolkata|Calcutta|Ahmedabad|Jaipur|Lucknow|Surat|Noida|Gurgaon|Gurugram|Goa|Indore|Bhopal|Nagpur|Kochi|Coimbatore"
    r")\b",
    re.IGNORECASE,
)

WHEN_PAT = re.compile(
    r"\b("
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"                      # 05/10/2025 or 5-10-25
    r"|[A-Za-z]{3,9}\s+\d{1,2},?\s*\d{2,4}"                # Oct 5, 2025 / October 5 2025
    r"| \d{1,2}\s+[A-Za-z]{3,9}\s*\d{2,4}"                 # 5 Oct 2025
    r")\b"
)

TIME_PAT = re.compile(
    r"\b(\d{1,2}(:\d{2})?\s*(AM|PM|am|pm))\b"
)

VENUE_PAT = re.compile(
    r"\b(venue|office|meeting\s*room|conference|location|address)\s*[:\-]\s*([^\r\n]+)",
    re.IGNORECASE,
)

def _loose_datetime(date_str: str, time_str: Optional[str], default_hour: int = 10) -> datetime:
    """Parse a date string, optionally combining with a time string. Defaults to 10:00 if no time."""
    base = dtparser.parse(date_str, dayfirst=False, fuzzy=True)
    if time_str:
        t = dtparser.parse(time_str, fuzzy=True).time()
    else:
        t = time(default_hour, 0)
    return datetime.combine(base.date(), t)

def _pick_destination(text: str) -> Optional[str]:
    hits = CITY_PAT.findall(text)
    if not hits:
        return None
    return hits[-1].title()

def extract_one_email(user_email: str, base_city: Optional[str], email_obj: Dict[str, Any]) -> TripFacts:
    """Extract TripFacts from a single email dict:
       {id, subject, snippet, body, internalDate}
    """
    subject = email_obj.get("subject", "")
    body = email_obj.get("body", "")
    text = f"{subject}\n{body}"
    dest = _pick_destination(text)
    dmatch = WHEN_PAT.search(text)
    tmatch = TIME_PAT.search(text)
    meet_start: Optional[datetime] = None
    if dmatch:
        try:
            meet_start = _loose_datetime(dmatch.group(0), tmatch.group(1) if tmatch else None)
        except Exception:
            meet_start = None

    venue = None
    vmatch = VENUE_PAT.search(text)
    if vmatch:
        venue = vmatch.group(2).strip()

    meet_obj = None
    if meet_start:
        meet_obj = Meeting(
            title=subject[:140] if subject else "Meeting",
            start=meet_start,
            end=None,
            location_text=venue,
            city=dest,
        )

    earliest_departure = meet_start - timedelta(hours=12) if meet_start else None
    latest_arrival = meet_start - timedelta(hours=3) if meet_start else None
    latest_return = meet_start + timedelta(hours=7) if meet_start else None

    notes: List[str] = []
    if not dest:
        notes.append("Destination city not confidently detected.")
    if not meet_start:
        notes.append("Meeting date/time not confidently detected.")
    if venue and dest and dest.lower() not in venue.lower():
        notes.append("Venue found (may be address/area); verify it matches the destination city.")

    return TripFacts(
        user_email=user_email,
        base_city=base_city,
        destination_city=dest,
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

