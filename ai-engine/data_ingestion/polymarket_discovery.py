"""
Polymarket market discovery and price fetching via Gamma API and CLOB API.
"""
import requests
from typing import Optional, List

GAMMA_API   = "https://gamma-api.polymarket.com"
CLOB_API    = "https://clob.polymarket.com"


def _safe_get(url: str, params: dict = None, timeout: int = 10) -> dict:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Discovery] GET {url} failed: {e}")
        return {}


def _normalize_market(raw: dict) -> Optional[dict]:
    cid = raw.get("conditionId") or raw.get("condition_id")
    if not cid:
        return None

    tokens = raw.get("tokens") or raw.get("outcomes") or []
    yes_token = next((t.get("token_id") or t.get("tokenId") for t in tokens
                      if str(t.get("outcome", "")).upper() == "YES"), None)
    no_token  = next((t.get("token_id") or t.get("tokenId") for t in tokens
                      if str(t.get("outcome", "")).upper() == "NO"), None)

    yes_price = 0.5
    no_price  = 0.5
    for t in tokens:
        outcome = str(t.get("outcome", "")).upper()
        price   = float(t.get("price") or t.get("outcomePrices", [0.5])[0] if isinstance(t.get("outcomePrices"), list) else 0.5)
        if outcome == "YES":
            yes_price = price
        elif outcome == "NO":
            no_price = price

    return {
        "conditionId":  cid,
        "question":     raw.get("question", ""),
        "active":       raw.get("active", True),
        "probabilities": {"YES": round(yes_price, 4), "NO": round(no_price, 4)},
        "liquidity":    float(raw.get("liquidity") or 0),
        "volume24h":    float(raw.get("volume24hr") or raw.get("volume") or 0),
        "endDate":      raw.get("endDate") or raw.get("end_date_iso"),
        "yesTokenId":   yes_token,
        "noTokenId":    no_token,
        "negRisk":      raw.get("negRisk", False),
    }


def fetch_all_active_markets(limit: int = 100) -> List[dict]:
    data = _safe_get(f"{GAMMA_API}/markets", params={
        "active": "true", "closed": "false", "limit": limit, "order": "volume24hr", "ascending": "false"
    })
    raw_markets = data if isinstance(data, list) else data.get("markets", [])
    result = []
    for m in raw_markets:
        norm = _normalize_market(m)
        if norm:
            result.append(norm)
    return result


def fetch_weather_markets() -> List[dict]:
    weather_terms = ["temperature", "degrees", "fahrenheit", "celsius", "high", "weather"]
    markets = fetch_all_active_markets(limit=200)
    return [
        m for m in markets
        if any(t in m["question"].lower() for t in weather_terms)
    ]


def fetch_clob_midpoint(token_id: str) -> Optional[float]:
    if not token_id:
        return None
    try:
        data = _safe_get(f"{CLOB_API}/midpoint", params={"token_id": token_id})
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
        tokens = data.get("tokens") or data.get("outcomes") or []
        for t in tokens:
            if str(t.get("outcome", "")).upper() == side.upper():
                price = t.get("price") or t.get("outcomePrices")
                if isinstance(price, list):
                    price = price[0]
                if price is not None:
                    val = float(price)
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
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 0
        if best_bid > 0 and best_ask > 0:
            spread = best_ask - best_bid
            if spread < 0.4:
                return round((best_bid + best_ask) / 2, 4)
    except Exception:
        pass
    return None


def get_user_positions(wallet_address: str) -> List[dict]:
    data = _safe_get(f"{GAMMA_API}/positions", params={"user": wallet_address})
    positions = data if isinstance(data, list) else data.get("positions", [])
    return positions


def get_user_trades(wallet_address: str) -> List[dict]:
    data = _safe_get(f"{CLOB_API}/trades", params={"maker_address": wallet_address, "limit": 50})
    return data if isinstance(data, list) else data.get("data", [])


def get_portfolio_value(wallet_address: str) -> dict:
    positions = get_user_positions(wallet_address)
    total = sum(
        float(p.get("currentValue") or p.get("value") or 0)
        for p in positions
    )
    return {"total_value": round(total, 2), "positions_count": len(positions)}
