from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel
from datetime import datetime

class Meeting(BaseModel):
    title: str
    start: datetime
    end: Optional[datetime] = None
    location_text: Optional[str] = None
    city: Optional[str] = None

class TripFacts(BaseModel):
    user_email: str
    base_city: Optional[str] = None
    destination_city: Optional[str] = None
    meeting: Optional[Meeting] = None
    earliest_departure: Optional[datetime] = None
    latest_arrival: Optional[datetime] = None
    latest_return: Optional[datetime] = None
    notes: List[str] = []

class Leg(BaseModel):
    mode: Literal["flight","train","bus","car"]
    provider: str
    from_city: str
    to_city: str
    depart: datetime
    arrive: datetime
    price: float
    meta: Dict[str, Any] = {}

class Itinerary(BaseModel):
    legs: List[Leg] = []
