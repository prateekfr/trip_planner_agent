from datetime import datetime
from dateutil import tz

def now_aware(tz_name: str = "Asia/Kolkata") -> datetime:
    return datetime.now(tz.gettz(tz_name))

def hours_between(a: datetime, b: datetime) -> float:
    return abs((b - a).total_seconds()) / 3600.0
