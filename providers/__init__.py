from datetime import datetime, timedelta
from typing import List
from utils.schemas import Leg

def flights(origin: str, destination: str, earliest: datetime, latest: datetime) -> List[Leg]:
    # stub: returns one mock flight
    depart = max(earliest, datetime(earliest.year, earliest.month, earliest.day, 7, 0))
    arrive = depart + timedelta(hours=2.5)
    return [Leg(mode="flight", provider="MockAir", from_city=origin, to_city=destination, depart=depart, arrive=arrive, price=5000.0)]

def trains(origin: str, destination: str, earliest: datetime, latest: datetime) -> List[Leg]:
    depart = max(earliest, datetime(earliest.year, earliest.month, earliest.day, 6, 0))
    arrive = depart + timedelta(hours=8)
    return [Leg(mode="train", provider="MockRail", from_city=origin, to_city=destination, depart=depart, arrive=arrive, price=1200.0)]

def buses(origin: str, destination: str, earliest: datetime, latest: datetime) -> List[Leg]:
    depart = max(earliest, datetime(earliest.year, earliest.month, earliest.day, 7, 0))
    arrive = depart + timedelta(hours=10)
    return [Leg(mode="bus", provider="MockBus", from_city=origin, to_city=destination, depart=depart, arrive=arrive, price=900.0)]

def hotels(near_city: str, checkin_date, checkout_date):
    # simple dicts; later we’ll define a Hotel schema if needed
    return [
        {"name": f"Business Inn {near_city}", "night": 3200.0, "distance_km": 0.8, "notes": "Free breakfast, Wi-Fi"},
        {"name": f"Comfort Stay {near_city}", "night": 2500.0, "distance_km": 1.3, "notes": "Good reviews"},
    ]
