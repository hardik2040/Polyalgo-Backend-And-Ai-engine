"""
Weather service using Open-Meteo API (free, no key needed).
Analyzes temperature-based Polymarket questions.
"""
import re
import requests
from datetime import datetime, timedelta
from typing import Optional


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"

CITY_COORDS = {
    "new york":      (40.7128,  -74.0060),
    "nyc":           (40.7128,  -74.0060),
    "los angeles":   (34.0522, -118.2437),
    "chicago":       (41.8781,  -87.6298),
    "houston":       (29.7604,  -95.3698),
    "phoenix":       (33.4484, -112.0740),
    "philadelphia":  (39.9526,  -75.1652),
    "san antonio":   (29.4241,  -98.4936),
    "san diego":     (32.7157, -117.1611),
    "dallas":        (32.7767,  -96.7970),
    "miami":         (25.7617,  -80.1918),
    "atlanta":       (33.7490,  -84.3880),
    "boston":        (42.3601,  -71.0589),
    "seattle":       (47.6062, -122.3321),
    "denver":        (39.7392, -104.9903),
    "london":        (51.5074,   -0.1278),
    "paris":         (48.8566,    2.3522),
    "tokyo":         (35.6762,  139.6503),
    "sydney":       (-33.8688,  151.2093),
    "toronto":       (43.6532,  -79.3832),
}


def is_weather_question(question: str) -> bool:
    q = question.lower()
    weather_terms = [
        "temperature", "degrees", "fahrenheit", "celsius",
        "high", "low", "weather", "°f", "°c", "temp"
    ]
    return any(t in q for t in weather_terms)


def _extract_city(question: str) -> Optional[str]:
    q = question.lower()
    for city in sorted(CITY_COORDS.keys(), key=len, reverse=True):
        if city in q:
            return city
    patterns = [
        r"in ([a-z ]+?) (be|exceed|reach|on|will|the)",
        r"temperature in ([a-z ]+)",
        r"high in ([a-z ]+)",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return m.group(1).strip()
    return None


def _extract_threshold_f(question: str) -> Optional[float]:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*°?\s*f\b",
        r"(\d+(?:\.\d+)?)\s*degrees?\s*fahrenheit",
        r"exceed\s+(\d+(?:\.\d+)?)",
        r"above\s+(\d+(?:\.\d+)?)",
        r"reach\s+(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*degrees",
        r"\b(\d{2,3})\b",
    ]
    for pat in patterns:
        m = re.search(pat, question.lower())
        if m:
            val = float(m.group(1))
            if 0 < val < 150:
                return val
    return None


def _get_coords(city: str):
    if city in CITY_COORDS:
        return CITY_COORDS[city]
    try:
        r = requests.get(GEOCODING_URL, params={"name": city, "count": 1}, timeout=5)
        results = r.json().get("results", [])
        if results:
            return results[0]["latitude"], results[0]["longitude"]
    except Exception:
        pass
    return None, None


def _fetch_max_temp_f(lat: float, lon: float, date_str: Optional[str] = None) -> Optional[float]:
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max",
            "temperature_unit": "fahrenheit",
            "forecast_days": 7,
            "timezone": "auto",
        }
        r = requests.get(FORECAST_URL, params=params, timeout=8)
        data = r.json()
        temps = data.get("daily", {}).get("temperature_2m_max", [])
        if temps:
            return float(temps[0])
    except Exception:
        pass
    return None


def analyze_temperature_question(question: str) -> dict:
    city = _extract_city(question)
    threshold_f = _extract_threshold_f(question)

    if not city or not threshold_f:
        return {
            "probability": 0.5,
            "confidence": 0.05,
            "reason": f"could_not_parse_city_or_threshold city={city} threshold={threshold_f}",
            "source": "open-meteo",
        }

    lat, lon = _get_coords(city)
    if lat is None:
        return {
            "probability": 0.5,
            "confidence": 0.05,
            "reason": f"city_not_found:{city}",
            "source": "open-meteo",
        }

    actual_f = _fetch_max_temp_f(lat, lon)
    if actual_f is None:
        return {
            "probability": 0.5,
            "confidence": 0.05,
            "reason": "weather_api_unavailable",
            "source": "open-meteo",
        }

    actual_c = round((actual_f - 32) * 5 / 9, 1)
    threshold_c = round((threshold_f - 32) * 5 / 9, 1)
    diff = actual_f - threshold_f

    # Confidence scales with how far the forecast is from the threshold
    confidence = min(0.95, 0.50 + abs(diff) * 0.025)

    if diff > 0:
        # Forecast exceeds threshold → YES likely
        probability = min(0.97, 0.60 + abs(diff) * 0.02)
    else:
        # Forecast below threshold → NO likely
        probability = max(0.03, 0.40 - abs(diff) * 0.02)

    return {
        "probability": round(probability, 3),
        "confidence": round(confidence, 3),
        "actual_temp_f": round(actual_f, 1),
        "actual_temp_c": actual_c,
        "threshold_f": threshold_f,
        "threshold_c": threshold_c,
        "city": city,
        "reason": f"forecast_{actual_f:.1f}F_vs_threshold_{threshold_f:.1f}F",
        "source": "open-meteo",
    }
