from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import re
from utils.indian_cities import INDIAN_CITIES as _CITY_LIST


CITY_WORDS = {c.lower() for c in _CITY_LIST}

INVITE_TERMS = [
    "invite", "invitation", "you're invited", "meeting", "conference", "seminar",
    "workshop", "summit", "symposium", "program", "event", "offsite", "onsite",
    "client visit", "review meeting", "kickoff", "standup", "townhall", "expo", "outing"
]

PROMO_TERMS = [
    "sale","offer","offers","discount","deal","deals","promo","coupon","cashback",
    "newsletter","subscribe","promotion","marketing","blast","limited time","% off","off%",
]

ONLINE_ONLY_TERMS = [
    "webinar","virtual","online","zoom","google meet","meet.google.com","microsoft teams","teams meeting",
]

LOCATION_HINTS = [
    "venue","address","location","at ","office","campus","hall","auditorium","center","centre","hotel",
]

DATE_HINT = re.compile(r"\b(\d{1,2}[:/.-]\d{1,2}[:/.-]\d{2,4}|[A-Za-z]{3,9}\s+\d{1,2},?\s*\d{2,4})\b")
TIME_HINT = re.compile(r"\b(\d{1,2}(:\d{2})?\s*(AM|PM|am|pm))\b")
PINCODE = re.compile(r"\b\d{6}\b")
GOOGLE_CAL = re.compile(r"Invitation from Google Calendar|Event details|calendar.google.com", re.IGNORECASE)
OUTLOOK_CAL = re.compile(r"Meeting Invitation|Event Invitation|Outlook\.com", re.IGNORECASE)
ICS_MARKER = re.compile(r"BEGIN:VCALENDAR|.ics", re.IGNORECASE)

def build_query(
    newer_than_days: int = 45,
    require_ics: bool = False,
    extra_and: str = "",
) -> str:
    """
    Build a Gmail query for invitation-style emails in the last N days.
    - Excludes promotions/social/forums.
    - Focuses on invite/meeting/conference style subjects.
    - Optionally requires ICS attachments (safer but stricter).
    """
    subj_clause = " OR ".join([f'subject:"{t}"' if " " in t else f"subject:{t}" for t in INVITE_TERMS])
    base = f"({subj_clause})"

    if require_ics:
        base += " (has:attachment filename:ics)"
    else:
        base += " OR (has:attachment filename:(ics OR pdf))"

    base += " -category:promotions -category:social -category:forums"
    base += " -(" + " OR ".join([f"subject:{w}" for w in PROMO_TERMS]) + ")"
    base += " -(" + " OR ".join([f"subject:\"{w}\"" if ' ' in w else f"subject:{w}" for w in ONLINE_ONLY_TERMS]) + ")"

    if newer_than_days and newer_than_days > 0:
        base += f" newer_than:{int(newer_than_days)}d"

    if extra_and.strip():
        base += f" {extra_and.strip()}"
    return base

def _text_of(msg: Dict[str, Any]) -> str:
    subject = msg.get("subject") or ""
    snippet = msg.get("snippet") or ""
    body = (msg.get("body") or "")[:8000]
    return f"{subject}\n{snippet}\n{body}"

def _has_city(text: str) -> Optional[str]:
    t = text.lower()
    for c in CITY_WORDS:
        if c in t:
            return c
    return None

def _has_location_hint(text: str) -> bool:
    tl = text.lower()
    return any(h in tl for h in LOCATION_HINTS) or bool(PINCODE.search(tl))

def _is_online_only(text: str) -> bool:
    tl = text.lower()
    return any(w in tl for w in ONLINE_ONLY_TERMS)

def _looks_like_invite(subject_body: str) -> bool:
    tl = subject_body.lower()
    return any(w in tl for w in ["invite", "invitation", "you're invited", "meeting", "conference",
                                 "seminar", "workshop", "summit", "symposium", "program", "event",
                                 "offsite", "onsite", "client visit", "review meeting", "kickoff",
                                 "townhall", "expo", "outing"])

def classify_message(msg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Keep if:
      A) Strong calendar signal (Google/Outlook/ICS), and either city or location hint present.
      B) Invitation wording + (date or time) + (city or location hint).
      (Online-only invites are dropped unless a physical location signal exists.)
    """
    t = _text_of(msg)
    reasons: List[str] = []
    has_cal = bool(GOOGLE_CAL.search(t) or OUTLOOK_CAL.search(t) or ICS_MARKER.search(t))
    has_date_time = bool(DATE_HINT.search(t) or TIME_HINT.search(t))
    city = _has_city(t)
    has_loc = _has_location_hint(t)
    online_only = _is_online_only(t)
    is_invite = _looks_like_invite(t)

    if has_cal and (city or has_loc):
        if online_only and not (city or PINCODE.search(t)):
            return False, ["calendar invite but online-only (no physical location)"]
        reasons.append("calendar invite (Google/Outlook/ICS) with location")
        if city: reasons.append(f"city:{city}")
        return True, reasons

    if is_invite and has_date_time and (city or has_loc):
        if online_only and not (city or PINCODE.search(t)):
            return False, ["invite with date/time but online-only (no physical location)"]
        reasons.append("invite + date/time + location")
        if city: reasons.append(f"city:{city}")
        return True, reasons
    return False, ["not a location-bound invitation"]

def filter_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for m in messages:
        keep, reasons = classify_message(m)
        if keep:
            mm = dict(m)
            mm["filter_reasons"] = reasons
            kept.append(mm)
    return kept


# # gmail_filter.py
# from __future__ import annotations
# from typing import List, Dict, Any, Tuple
# import re
# import json
# from datetime import datetime, timezone

# # -------------------- Gmail query builder (broader) --------------------
# def build_query(
#     newer_than_days: int = 45,     # broad window so LLM can decide
#     require_ics: bool = False,
#     extra_and: str = "",
# ) -> str:
#     """
#     Broad inbox search; exclude obvious clutter categories.
#     We rely on the LLM to decide keep vs drop.
#     """
#     base = "in:inbox -category:social -category:forums -in:drafts -category:spam -unsubscribe -newsletter"
#     if newer_than_days and newer_than_days > 0:
#         base += f" newer_than:{int(newer_than_days)}d"

#     # Pull likely candidates; NOT required, just helps surface useful mail.
#     keywords = (
#         '(invite OR invitation OR meeting OR conference OR seminar OR workshop OR summit OR symposium '
#         'OR program OR event OR offsite OR onsite OR "team outing" OR "site visit" OR "client visit" '
#         'OR "plant visit" OR "field visit" OR interview OR onboarding OR installation OR audit OR expo '
#         'OR itinerary OR booking OR ticket OR reservation OR schedule OR agenda)'
#     )
#     base = f"({base}) ({keywords})"

#     if require_ics:
#         base += " (has:attachment filename:ics)"
#     else:
#         base += " OR (has:attachment filename:(ics OR pdf))"

#     if extra_and.strip():
#         base += f" {extra_and.strip()}"
#     return base


# # -------------------- Helpers --------------------
# def _text_of(msg: Dict[str, Any]) -> str:
#     subject = msg.get("subject") or ""
#     snippet = msg.get("snippet") or ""
#     body = (msg.get("body") or "")[:8000]
#     return f"{subject}\n{snippet}\n{body}"

# # PIN code pattern for India (used by the simple online-only sanity guard)
# PINCODE = re.compile(r"\b\d{6}\b")

# # -------------------- LLM-based classifier --------------------
# # NOTE: This uses the same OpenRouter client your project already has.
# try:
#     from src.llm import OpenRouterClient, DEFAULT_MODEL
# except Exception:
#     OpenRouterClient = None
#     DEFAULT_MODEL = None

# _SYSTEM_PROMPT = (
#     "You are an email triage assistant for a trip-planning tool.\n"
#     "INPUT: One email’s text (subject + snippet + body).\n"
#     "TODAY (UTC): {TODAY_ISO}\n\n"
#     "GOAL: Decide if the email is relevant for planning an IN-PERSON TRIP IN THE FUTURE.\n"
#     "Mark keep=true ONLY if the email clearly implies the user may need to physically travel and the meeting/event/trip is not already in the past relative to TODAY.\n\n"
#     "KEEP (keep=true) when ANY of these hold AND the date is in the future (or at least not past TODAY):\n"
#     "  • Calendar invite (Google/Outlook/ICS) for an in-person meeting/event.\n"
#     "  • A meeting/visit with a date/time AND any physical venue/location/address/office/hotel.\n"
#     "  • In-person work activities: team outing/offsite, onsite/client/vendor/partner meeting, site/plant/factory/field visit, interview at a location, training at a location, audit at a location, expo/trade fair with venue.\n"
#     "  • Travel confirmations (flight/train/bus/hotel) that clearly relate to a real trip (routes/cities/dates/PNR/etc.).\n\n"
#     "REJECT (keep=false) when ANY of these hold:\n"
#     "  • Online-only (webinar/Zoom/Google Meet/Teams) AND no physical venue/address.\n"
#     "  • Generic promotions/marketing/newsletters, courses, study materials, or product announcements with no concrete in-person event or trip.\n"
#     "  • The referenced meeting/event/trip date is clearly in the past relative to TODAY.\n"
#     "  • Vague emails with no date/time and no physical location/venue signal.\n\n"
#     "IMPORTANT:\n"
#     "  • Do NOT require the word 'invite' in the subject. Team outings/offsites and site/plant/field visits should be kept if physical and time-bound.\n"
#     "  • If promotional but contains a specific future date/time AND a concrete physical venue/address, then KEEP (because it may require travel).\n\n"
#     "OUTPUT: Strict JSON ONLY (no extra text):\n"
#     "{ \"keep\": true|false, \"reasons\": [\"short reason\", ...] }\n"
# )

# def _classify_with_llm(text: str) -> Tuple[bool, List[str]]:
#     if not OpenRouterClient or not DEFAULT_MODEL:
#         return False, ["LLM unavailable"]

#     # SAFELY inject today's date without .format() (avoids KeyError from JSON braces)
#     today_iso = datetime.now(timezone.utc).date().isoformat()
#     if "{TODAY_ISO}" in _SYSTEM_PROMPT:
#         sys_prompt = _SYSTEM_PROMPT.replace("{TODAY_ISO}", today_iso)
#     else:
#         sys_prompt = _SYSTEM_PROMPT + f"\n\nTODAY (UTC): {today_iso}"

#     messages = [
#         {"role": "system", "content": sys_prompt},
#         {"role": "user", "content": text[:12000]},
#     ]
#     try:
#         client = OpenRouterClient()
#         out = client.chat(messages, model=DEFAULT_MODEL, temperature=0.0, timeout=12)
#         m = re.search(r"\{.*\}", out, flags=re.S)
#         if not m:
#             return False, ["LLM returned no JSON"]
#         data = json.loads(m.group(0))
#         keep = bool(data.get("keep", False))
#         reasons = data.get("reasons") or []
#         if not isinstance(reasons, list):
#             reasons = [str(reasons)]
#         return keep, reasons
#     except Exception as e:
#         return False, [f"LLM error: {type(e).__name__}: {str(e)[:120]}"]


# # -------------------- Public API --------------------
# def classify_message(msg: Dict[str, Any]) -> Tuple[bool, List[str]]:
#     text = _text_of(msg)

#     # 1) LLM decision
#     keep, reasons = _classify_with_llm(text)

#     # 2) Hard safety guard: if it’s clearly online-only and there is no address/pincode,
#     #    force reject. This is not a promo keyword list—just a sanity check for online links.
#     online_only = bool(re.search(r"\b(webinar|zoom|google meet|meet\.google\.com|microsoft teams|teams meeting)\b", text, re.I))
#     has_physical = bool(PINCODE.search(text) or re.search(r"\b(venue|address|location|office|campus|hotel|auditorium|hall|centre|center)\b", text, re.I))
#     if keep and online_only and not has_physical:
#         return False, reasons + ["guard: online-only (no physical venue)"]

#     return (keep, reasons) if keep else (False, reasons or ["llm:false"])

# def filter_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     kept: List[Dict[str, Any]] = []
#     for m in messages:
#         keep, reasons = classify_message(m)
#         if keep:
#             mm = dict(m)
#             mm["filter_reasons"] = reasons
#             kept.append(mm)
#     return kept
