"""
Weather service using Open-Meteo API (free, no key needed).
Analyzes temperature-based Polymarket questions.
"""
import re
import math
import requests
from datetime import datetime, timedelta
from typing import Optional


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"

# Day-ahead max temperature forecast error std-dev (°C) by climate type.
# Tropical cities are more stable; continental/maritime are more variable.
FORECAST_ERROR_STD_C = 1.5   # global default
CITY_FORECAST_STD: dict = {
    "jakarta": 1.0, "singapore": 1.0, "kuala lumpur": 1.0, "bangkok": 1.0,
    "manila": 1.2, "hong kong": 1.2, "taipei": 1.2, "guangzhou": 1.2,
    "shenzhen": 1.2, "karachi": 1.2, "dubai": 1.0,
    "london": 2.0, "amsterdam": 2.0, "brussels": 2.0,
    "seattle": 2.0, "chicago": 2.2, "new york": 2.0, "boston": 2.0,
}

CITY_COORDS = {
    # North America
    "new york":      (40.7128,  -74.0060),
    "nyc":           (40.7128,  -74.0060),
    "new york city": (40.7128,  -74.0060),
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
    "toronto":       (43.6532,  -79.3832),
    # Europe
    "london":        (51.5074,   -0.1278),
    "paris":         (48.8566,    2.3522),
    "berlin":        (52.5200,   13.4050),
    "madrid":        (40.4168,   -3.7038),
    "rome":          (41.9028,   12.4964),
    "milan":         (45.4642,    9.1900),
    "amsterdam":     (52.3676,    4.9041),
    "brussels":      (50.8503,    4.3517),
    "vienna":        (48.2082,   16.3738),
    "warsaw":        (52.2297,   21.0122),
    "munich":        (48.1351,   11.5820),
    "prague":        (50.0755,   14.4378),
    "zurich":        (47.3769,    8.5417),
    "stockholm":     (59.3293,   18.0686),
    "oslo":          (59.9139,   10.7522),
    "copenhagen":    (55.6761,   12.5683),
    "helsinki":      (60.1699,   24.9384),
    "lisbon":        (38.7223,   -9.1393),
    "athens":        (37.9838,   23.7275),
    "budapest":      (47.4979,   19.0402),
    "bucharest":     (44.4268,   26.1025),
    "kiev":          (50.4501,   30.5234),
    "kyiv":          (50.4501,   30.5234),
    "moscow":        (55.7558,   37.6173),
    "istanbul":      (41.0082,   28.9784),
    "ankara":        (39.9334,   32.8597),
    # Asia-Pacific
    "tokyo":         (35.6762,  139.6503),
    "seoul":         (37.5665,  126.9780),
    "beijing":       (39.9042,  116.4074),
    "shanghai":      (31.2304,  121.4737),
    "guangzhou":     (23.1291,  113.2644),
    "shenzhen":      (22.5431,  114.0579),
    "chengdu":       (30.5728,  104.0668),
    "chongqing":     (29.4316,  106.9123),
    "hong kong":     (22.3193,  114.1694),
    "taipei":        (25.0330,  121.5654),
    "singapore":     (1.3521,   103.8198),
    "jakarta":       (-6.2088,  106.8456),
    "manila":        (14.5995,  120.9842),
    "bangkok":       (13.7563,  100.5018),
    "kuala lumpur":  (3.1390,   101.6869),
    "ho chi minh":   (10.8231,  106.6297),
    "dhaka":         (23.8103,   90.4125),
    "mumbai":        (19.0760,   72.8777),
    "delhi":         (28.6139,   77.2090),
    "karachi":       (24.8607,   67.0011),
    "lahore":        (31.5204,   74.3587),
    "sydney":        (-33.8688,  151.2093),
    "melbourne":     (-37.8136,  144.9631),
    "auckland":      (-36.8485,  174.7633),
    "wellington":    (-41.2865,  174.7762),
    "osaka":         (34.6937,  135.5023),
    # Middle East / Africa
    "dubai":         (25.2048,   55.2708),
    "riyadh":        (24.7136,   46.6753),
    "jeddah":        (21.3891,   39.8579),
    "cairo":         (30.0444,   31.2357),
    "cape town":     (-33.9249,  18.4241),
    "lagos":         (6.5244,    3.3792),
    "nairobi":       (-1.2921,   36.8219),
    # South America
    "sao paulo":     (-23.5558,  -46.6396),
    "buenos aires":  (-34.6037,  -58.3816),
    "rio de janeiro":(-22.9068,  -43.1729),
    "lima":          (-12.0464,  -77.0428),
    "bogota":        (4.7110,   -74.0721),
    "santiago":      (-33.4489,  -70.6693),
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


def _extract_target_celsius(question: str) -> Optional[float]:
    """Extract target temperature (°C) from Celsius market questions."""
    q = question.lower()
    # Range: "between 22-23°c"
    m = re.search(r"between\s+(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*°c", q)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    # Exact: "be 22°c" or "22 celsius" or "22°c"
    for pat in [r"(\d+(?:\.\d+)?)\s*°c", r"(\d+(?:\.\d+)?)\s*celsius",
                r"(\d+(?:\.\d+)?)\s*degrees\s*celsius"]:
        m = re.search(pat, q)
        if m:
            return float(m.group(1))
    return None


def _extract_threshold_f(question: str) -> Optional[float]:
    """Extract threshold temperature (°F) from Fahrenheit market questions."""
    q = question.lower()
    # Range: "74-75°f" or "between 74-75°f"
    m = re.search(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\s*°?\s*f\b", q)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    for pat in [
        r"(\d+(?:\.\d+)?)\s*°?\s*f\b",
        r"(\d+(?:\.\d+)?)\s*degrees?\s*fahrenheit",
        r"exceed\s+(\d+(?:\.\d+)?)",
        r"above\s+(\d+(?:\.\d+)?)",
        r"reach\s+(\d+(?:\.\d+)?)",
    ]:
        m = re.search(pat, q)
        if m:
            val = float(m.group(1))
            if 0 < val < 150:
                return val
    return None


def _is_exact_market(question: str) -> bool:
    """True for 'will temp BE exactly X' markets; False for 'exceed/above/or higher' markets."""
    q = question.lower()
    threshold_words = ["or higher", "or above", "or more", "exceed", "above", "at least", "over"]
    return not any(w in q for w in threshold_words)


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


def _fetch_max_temp_f(lat: float, lon: float) -> Optional[float]:
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


def _gaussian_prob(diff_c: float, sigma_c: float) -> float:
    """
    Probability that the actual max temp hits a 1°C-wide bucket centered at target,
    given the forecast is `diff_c` degrees away (forecast − target).
    Uses a Gaussian with peak ≈ 0.28 (empirical for 1°C buckets, σ=1.5°C).
    """
    peak = 0.28
    return peak * math.exp(-0.5 * (diff_c / sigma_c) ** 2)


def analyze_temperature_question(question: str) -> dict:
    city  = _extract_city(question)
    q     = question.lower()

    is_celsius = "°c" in q or " celsius" in q
    is_exact   = _is_exact_market(question)

    # ── 1. Extract target temperature in °C ──────────────────────────────────
    if is_celsius:
        target_c = _extract_target_celsius(question)
        if target_c is None:
            return {"probability": 0.5, "confidence": 0.05,
                    "reason": "celsius_parse_fail", "source": "open-meteo"}
    else:
        threshold_f = _extract_threshold_f(question)
        if threshold_f is None:
            return {"probability": 0.5, "confidence": 0.05,
                    "reason": "fahrenheit_parse_fail", "source": "open-meteo"}
        target_c = (threshold_f - 32) * 5 / 9

    if not city:
        return {"probability": 0.5, "confidence": 0.05,
                "reason": "city_parse_fail", "source": "open-meteo"}

    lat, lon = _get_coords(city)
    if lat is None:
        return {"probability": 0.5, "confidence": 0.05,
                "reason": f"city_not_found:{city}", "source": "open-meteo"}

    actual_f = _fetch_max_temp_f(lat, lon)
    if actual_f is None:
        return {"probability": 0.5, "confidence": 0.05,
                "reason": "weather_api_unavailable", "source": "open-meteo"}

    actual_c = (actual_f - 32) * 5 / 9
    diff_c   = actual_c - target_c                    # positive → forecast warmer than target
    sigma    = CITY_FORECAST_STD.get(city, FORECAST_ERROR_STD_C)

    if is_exact:
        # ── Gaussian model for "will temp BE exactly X°C?" markets ───────────
        # The bot's edge comes ONLY when the Open-Meteo forecast closely matches
        # the market's target temperature.  Off-target temperatures give negative
        # YES EV and a positive NO EV (which the EV calculator will prefer).
        prob_yes = max(0.02, min(0.75, _gaussian_prob(diff_c, sigma)))

        abs_diff = abs(diff_c)
        if abs_diff < 0.5:
            # Forecast ≈ target → strong YES signal
            confidence = 0.80
        elif abs_diff < 1.5:
            # One degree off → marginal edge; market may still resolve here
            confidence = 0.62
        elif abs_diff < 3.0:
            # Clearly off-target → confident NO signal
            confidence = 0.75
        else:
            # Way off → very confident NO
            confidence = 0.88
    else:
        # ── Threshold/exceedance model ("will temp EXCEED X°C/°F?") ──────────
        # Original logic, now using correct °C diff throughout.
        confidence = min(0.95, 0.50 + abs(diff_c) * 0.10)
        if diff_c > 0:
            prob_yes = min(0.97, 0.60 + diff_c * 0.08)
        else:
            prob_yes = max(0.03, 0.40 + diff_c * 0.08)

    return {
        "probability":     round(prob_yes, 3),
        "confidence":      round(confidence, 3),
        "actual_temp_f":   round(actual_f, 1),
        "actual_temp_c":   round(actual_c, 1),
        "threshold_f":     round(target_c * 9 / 5 + 32, 1),
        "threshold_c":     round(target_c, 1),
        "diff_c":          round(diff_c, 1),
        "is_exact_market": is_exact,
        "city":            city,
        "reason":          f"forecast_{actual_c:.1f}C_vs_target_{target_c:.1f}C_diff={diff_c:+.1f}C",
        "source":          "open-meteo",
    }
