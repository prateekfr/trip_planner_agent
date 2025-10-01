from difflib import get_close_matches
from typing import List

INDIAN_CITIES: List[str] = [
    # A
    "Agartala","Agra","Ahmedabad","Aizawl","Ajmer","Akola","Alappuzha","Aligarh","Allahabad","Alwar","Ambala",
    "Amravati","Amritsar","Anantapur","Ara","Arrah","Asansol","Aurangabad",
    # B
    "Badlapur","Bagalkot","Bahadurgarh","Ballari","Bally","Balurghat","Banda","Bareilly","Baripada","Barmer",
    "Bathinda","Begusarai","Belagavi","Belgaum","Bellary","Berhampur","Bettiah","Bhagalpur","Bharatpur","Bhavnagar",
    "Bhilai","Bhilwara","Bhimavaram","Bhiwadi","Bhiwani","Bhopal","Bhubaneswar","Bhuj","Bhusawal","Bidar","Bihar Sharif",
    "Bijapur","Bikaner","Bilaspur","Bokaro","Botad","Bulandshahr",
    # C
    "Chandigarh","Chandrapur","Chennai","Chhapra","Chikkamagaluru","Chinsurah","Chitradurga","Chittoor",
    "Coimbatore","Cuddalore","Cuttack","Calicut",
    # D
    "Daman","Darbhanga","Darjeeling","Davangere","Dehradun","Delhi","Dhanbad","Dhar","Dharamshala","Dharwad",
    "Dhenkanal","Dibrugarh","Dimapur","Durg","Durgapur",
    # E
    "Eluru","Erode",
    # F
    "Faridabad","Firozabad",
    # G
    "Gadag","Gandhinagar","Gaya","Ghaziabad","Gokak","Gorakhpur","Gudur","Gulbarga","Guna","Guntur","Guwahati","Gwalior",
    # H
    "Habra","Haldia","Haldwani","Hansi","Hardoi","Haridwar","Hassan","Hazaribagh","Himatnagar","Hisar","Hoshiarpur","Howrah","Hubli",
    # I
    "Imphal","Indore","Itanagar",
    # J
    "Jabalpur","Jagdalpur","Jaipur","Jalandhar","Jalgaon","Jalna","Jammu","Jamnagar","Jamshedpur","Jhansi","Jind","Jodhpur","Junagadh",
    # K
    "Kadapa","Kakinada","Kalaburagi","Kalyan","Kancheepuram","Kannur","Kanpur","Karaikudi","Karimnagar","Karnal","Karur","Karwar",
    "Kashipur","Katni","Kharagpur","Khanna","Kochi","Kodaikanal","Kolar","Kolhapur","Kolkata","Kollam","Korba","Kota","Kottayam",
    "Kozhikode","Kumbakonam","Kurnool",
    # L
    "Latur","Lonavala","Lucknow","Ludhiana",
    # M
    "Madurai","Maheshtala","Malegaon","Mangalore","Mangaluru","Mathura","Meerut","Mehsana","Mirzapur","Moradabad","Motihari",
    "Mumbai","Muzaffarnagar","Muzaffarpur","Mysore","Mysuru",
    # N
    "Nadia","Nagercoil","Nagpur","Nanded","Nashik","Navi Mumbai","Nellore","Noida",
    # O
    "Ongole","Ooty",
    # P
    "Pali","Panaji","Panchkula","Panipat","Parbhani","Pathankot","Patiala","Patna","Phagwara","Pimpri-Chinchwad",
    "Pondicherry","Porbandar","Prayagraj","Puducherry","Pune","Puri",
    # Q
    "Quilon",
    # R
    "Raebareli","Raichur","Raiganj","Raipur","Rajahmundry","Rajkot","Rajnandgaon","Ramanagara","Ranchi",
    "Ratlam","Rewa","Rewari","Rohtak","Roorkee","Rourkela",
    # S
    "Sabarkantha","Sagar","Saharanpur","Salem","Sambalpur","Sangli","Satara","Satna","Secunderabad","Shahjahanpur","Shillong",
    "Shimla","Shivamogga","Sikar","Siliguri","Silvassa","Sirsa","Solapur","Sonipat","Srikakulam","Srinagar","Surat",
    # T
    "Tadepalligudem","Thane","Thanjavur","Thiruvananthapuram","Thoothukudi","Thrissur","Tinsukia","Tiruchirappalli",
    "Tirunelveli","Tirupati","Tiruppur","Tumakuru","Tuticorin",
    # U
    "Udaipur","Udupi","Ujjain","Unnao",
    # V
    "Vadodara","Valsad","Varanasi","Vasai","Vellore","Vijayawada","Visakhapatnam","Vizianagaram",
    # W
    "Warangal",
]

def suggest_cities(prefix: str, n: int = 7) -> List[str]:
    """Return up to n suggestions, prioritizing prefix matches then fuzzy matches."""
    if not prefix:
        return []
    prefix_lower = prefix.lower()
    starts = [c for c in INDIAN_CITIES if c.lower().startswith(prefix_lower)]
    if len(starts) >= n:
        return starts[:n]
    remaining = [c for c in INDIAN_CITIES if c not in starts]
    fuzzy = get_close_matches(prefix, remaining, n=n - len(starts))
    out = starts + [c for c in fuzzy if c not in starts]
    return out[:n]
