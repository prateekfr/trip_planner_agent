from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import json
from datetime import datetime, timezone
from src.llm import OpenRouterClient, DEFAULT_MODEL

def build_query(
    newer_than_days: int = 45,     
    require_ics: bool = False,
    extra_and: str = "",
) -> str:
    """
    Broad inbox search; exclude obvious clutter categories.
    We rely on the LLM to decide keep vs drop.
    """
    base = "in:inbox -category:social -category:forums -in:drafts -category:spam -unsubscribe -newsletter"
    if newer_than_days and newer_than_days > 0:
        base += f" newer_than:{int(newer_than_days)}d"
    keywords = (
        '(invite OR invitation OR meeting OR conference OR seminar OR workshop OR summit OR symposium '
        'OR program OR event OR offsite OR onsite OR "team outing" OR "site visit" OR "client visit" '
        'OR "plant visit" OR "field visit" OR interview OR onboarding OR installation OR audit OR expo '
        'OR itinerary OR booking OR ticket OR reservation OR schedule OR agenda)'
    )
    base = f"({base}) ({keywords})"

    if require_ics:
        base += " (has:attachment filename:ics)"
    else:
        base += " OR (has:attachment filename:(ics OR pdf))"
    if extra_and.strip():
        base += f" {extra_and.strip()}"
    return base

def _text_of(msg: Dict[str, Any]) -> str:
    subject = msg.get("subject") or ""
    snippet = msg.get("snippet") or ""
    body = (msg.get("body") or "")[:8000]
    return f"{subject}\n{snippet}\n{body}"

PINCODE = re.compile(r"\b\d{6}\b")

_SYSTEM_PROMPT = (
    "You are an email triage assistant for a trip-planning tool.\n"
    "INPUT: One email's text (subject + snippet + body).\n"
    "TODAY (UTC): {TODAY_ISO}\n\n"
    "GOAL: Decide if the email is relevant for planning an IN-PERSON TRIP IN THE FUTURE.\n"
    "Mark keep=true ONLY if the email clearly implies the user may need to physically travel and the meeting/event/trip is not already in the past relative to TODAY.\n\n"
    "KEEP (keep=true) when ANY of these hold AND the date is in the future (or at least not past TODAY):\n"
    "  • Calendar invite (Google/Outlook/ICS) for an in-person meeting/event.\n"
    "  • A meeting/visit with a date/time AND any physical venue/location/address/office/hotel.\n"
    "  • In-person work activities: team outing/offsite, onsite/client/vendor/partner meeting, site/plant/factory/field visit, interview at a location, training at a location, audit at a location, expo/trade fair with venue.\n"
    "  • Travel confirmations (flight/train/bus/hotel) that clearly relate to a real trip (routes/cities/dates/PNR/etc.).\n\n"
    "REJECT (keep=false) when ANY of these hold:\n"
    "  • Online-only (webinar/Zoom/Google Meet/Teams) AND no physical venue/address.\n"
    "  • Generic promotions/marketing/newsletters, courses, study materials, or product announcements with no concrete in-person event or trip.\n"
    "  • The referenced meeting/event/trip date is clearly in the past relative to TODAY.\n"
    "  • Vague emails with no date/time and no physical location/venue signal.\n\n"
    "IMPORTANT:\n"
    "  • Do NOT require the word 'invite' in the subject. Team outings/offsites and site/plant/field visits should be kept if physical and time-bound.\n"
    "  • If promotional but contains a specific future date/time AND a concrete physical venue/address, then KEEP (because it may require travel).\n\n"
    "OUTPUT: Strict JSON ONLY (no extra text):\n"
    "{ \"keep\": true|false, \"reasons\": [\"short reason\", ...] }\n"
)

def _classify_with_llm(text: str) -> Tuple[bool, List[str]]:
    if not OpenRouterClient or not DEFAULT_MODEL:
        return False, ["LLM unavailable"]
    today_iso = datetime.now(timezone.utc).date().isoformat()
    if "{TODAY_ISO}" in _SYSTEM_PROMPT:
        sys_prompt = _SYSTEM_PROMPT.replace("{TODAY_ISO}", today_iso)
    else:
        sys_prompt = _SYSTEM_PROMPT + f"\n\nTODAY (UTC): {today_iso}"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text[:12000]},
    ]
    try:
        client = OpenRouterClient()
        out = client.chat(messages, model=DEFAULT_MODEL, temperature=0.0, timeout=12)
        m = re.search(r"\{.*\}", out, flags=re.S)
        if not m:
            return False, ["LLM returned no JSON"]
        data = json.loads(m.group(0))
        keep = bool(data.get("keep", False))
        reasons = data.get("reasons") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        return keep, reasons
    except Exception as e:
        return False, [f"LLM error: {type(e).__name__}: {str(e)[:120]}"]

def classify_message(msg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    text = _text_of(msg)

    keep, reasons = _classify_with_llm(text)
    online_only = bool(re.search(r"\b(webinar|zoom|google meet|meet\.google\.com|microsoft teams|teams meeting)\b", text, re.I))
    has_physical = bool(PINCODE.search(text) or re.search(r"\b(venue|address|location|office|campus|hotel|auditorium|hall|centre|center)\b", text, re.I))
    if keep and online_only and not has_physical:
        return False, reasons + ["guard: online-only (no physical venue)"]

    return (keep, reasons) if keep else (False, reasons or ["llm:false"])

def filter_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for m in messages:
        keep, reasons = classify_message(m)
        if keep:
            mm = dict(m)
            mm["filter_reasons"] = reasons
            kept.append(mm)
    return kept
