"""
Polymarket market discovery and price fetching via Gamma API and CLOB API.
"""
import json
import requests
from typing import Optional, List

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"


def _safe_get(url: str, params: dict = None, timeout: int = 10):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Discovery] GET {url} failed: {e}")
        return {}


def _parse_json_field(value) -> list:
    """Parse a field that may be a JSON string or already a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _normalize_market(raw) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    cid      = raw.get("conditionId")
    question = raw.get("question", "")
    if not cid or not question:
        return None

    # outcomes and prices are stored as JSON strings e.g. '["Yes","No"]'
    outcomes      = _parse_json_field(raw.get("outcomes", "[]"))
    outcome_prices = _parse_json_field(raw.get("outcomePrices", "[]"))
    clob_token_ids = _parse_json_field(raw.get("clobTokenIds", "[]"))

    # Map YES/NO prices and token IDs by matching outcomes list
    yes_price = 0.5
    no_price  = 0.5
    yes_token = None
    no_token  = None

    for i, outcome in enumerate(outcomes):
        o = str(outcome).upper()
        price = 0.5
        token = clob_token_ids[i] if i < len(clob_token_ids) else None
        if i < len(outcome_prices):
            try:
                price = float(outcome_prices[i])
            except (TypeError, ValueError):
                price = 0.5

        if o in ("YES", "Y"):
            yes_price = price
            yes_token = token
        elif o in ("NO", "N"):
            no_price  = price
            no_token  = token

    # Fallback for markets with non-standard outcome names (first = YES, second = NO)
    if yes_token is None and len(clob_token_ids) >= 2:
        yes_token = clob_token_ids[0]
        no_token  = clob_token_ids[1]
        if len(outcome_prices) >= 2:
            try:
                yes_price = float(outcome_prices[0])
                no_price  = float(outcome_prices[1])
            except (TypeError, ValueError):
                pass

    try:
        liquidity = float(raw.get("liquidityNum") or raw.get("liquidity") or 0)
    except (TypeError, ValueError):
        liquidity = 0.0

    return {
        "conditionId":   cid,
        "question":      question,
        "active":        raw.get("active", True),
        "probabilities": {"YES": round(yes_price, 4), "NO": round(no_price, 4)},
        "liquidity":     liquidity,
        "volume24h":     float(raw.get("volume24hr") or raw.get("volume24hrClob") or 0),
        "endDate":       raw.get("endDate") or raw.get("endDateIso"),
        "yesTokenId":    yes_token,
        "noTokenId":     no_token,
        "negRisk":       raw.get("negRisk", False),
    }


def fetch_all_active_markets(limit: int = 100) -> List[dict]:
    PAGE_SIZE = 100
    result = []
    offset = 0

    while True:
        data = _safe_get(f"{GAMMA_API}/markets", params={
            "active":    "true",
            "closed":    "false",
            "limit":     PAGE_SIZE,
            "offset":    offset,
            "order":     "volume24hr",
            "ascending": "false",
        })

        raw_list = []
        if isinstance(data, list):
            raw_list = data
        elif isinstance(data, dict):
            raw_list = data.get("markets") or data.get("data") or []

        if not raw_list:
            break

        for item in raw_list:
            norm = _normalize_market(item)
            if norm:
                result.append(norm)

        offset += PAGE_SIZE
        if len(raw_list) < PAGE_SIZE or len(result) >= limit:
            break

    print(f"[Discovery] {len(result)} markets fetched (limit={limit})")
    return result[:limit]


def fetch_weather_markets() -> List[dict]:
    weather_terms = ["temperature", "degrees", "fahrenheit", "celsius", "°f", "°c", "high temp", "weather"]
    markets = fetch_all_active_markets(limit=1000)
    filtered = [
        m for m in markets
        if any(t in m.get("question", "").lower() for t in weather_terms)
    ]
    print(f"[Discovery] {len(filtered)} weather markets found")
    return filtered


def fetch_clob_midpoint(token_id: str) -> Optional[float]:
    if not token_id:
        return None
    try:
        data = _safe_get(f"{CLOB_API}/midpoint", params={"token_id": token_id})
        if isinstance(data, dict):
            mid = data.get("mid")
            if mid is not None:
                val = float(mid)
                if 0.01 <= val <= 0.99:
                    return val
    except Exception:
        pass
    return None


def fetch_gamma_price(condition_id: str, side: str = "YES") -> Optional[float]:
    try:
        data = _safe_get(f"{GAMMA_API}/markets/{condition_id}")
        if not isinstance(data, dict):
            return None
        outcomes       = _parse_json_field(data.get("outcomes", "[]"))
        outcome_prices = _parse_json_field(data.get("outcomePrices", "[]"))
        for i, outcome in enumerate(outcomes):
            if str(outcome).upper() == side.upper() and i < len(outcome_prices):
                val = float(outcome_prices[i])
                if 0.01 <= val <= 0.99:
                    return val
    except Exception:
        pass
    return None


def fetch_market_price(token_id: str) -> Optional[float]:
    if not token_id:
        return None
    try:
        data = _safe_get(f"{CLOB_API}/book", params={"token_id": token_id})
        if not isinstance(data, dict):
            return None
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 0
        if best_bid > 0 and best_ask > 0 and (best_ask - best_bid) < 0.4:
            return round((best_bid + best_ask) / 2, 4)
    except Exception:
        pass
    return None


def get_user_positions(wallet_address: str) -> List[dict]:
    data = _safe_get(
        "https://data-api.polymarket.com/positions",
        params={"user": wallet_address, "sizeThreshold": "0.01", "limit": 50}
    )
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("positions") or data.get("data") or []
    return []


def get_user_trades(wallet_address: str) -> List[dict]:
    data = _safe_get(f"{CLOB_API}/trades", params={
        "maker_address": wallet_address, "limit": 50
    })
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("data") or []
    return []


def get_portfolio_value(wallet_address: str) -> dict:
    positions = get_user_positions(wallet_address)
    total = 0.0
    for p in positions:
        if isinstance(p, dict):
            try:
                total += float(p.get("currentValue") or p.get("value") or p.get("size") or 0)
            except (TypeError, ValueError):
                pass
    return {"total_value": round(total, 2), "positions_count": len(positions)}
