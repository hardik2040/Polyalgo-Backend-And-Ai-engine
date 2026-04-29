"""
Polymarket market discovery and price fetching via Gamma API and CLOB API.
"""
import requests
from typing import Optional, List

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"


def _safe_get(url: str, params: dict = None, timeout: int = 10) -> any:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Discovery] GET {url} failed: {e}")
        return {}


def _normalize_market(raw: any) -> Optional[dict]:
    # Skip non-dict items entirely
    if not isinstance(raw, dict):
        return None

    # Support both camelCase and snake_case field names
    cid = raw.get("conditionId") or raw.get("condition_id")
    if not cid:
        return None

    question = raw.get("question", "")
    if not question:
        return None

    # Extract token IDs and prices from tokens array
    tokens = raw.get("tokens") or raw.get("outcomes") or []
    yes_token = None
    no_token  = None
    yes_price = 0.5
    no_price  = 0.5

    for t in tokens:
        if not isinstance(t, dict):
            continue
        outcome = str(t.get("outcome", "")).upper()
        token_id = t.get("token_id") or t.get("tokenId") or t.get("id")
        price = t.get("price")
        if price is None:
            op = t.get("outcomePrices")
            price = op[0] if isinstance(op, list) and op else 0.5
        try:
            price = float(price)
        except (TypeError, ValueError):
            price = 0.5

        if outcome == "YES":
            yes_token = token_id
            yes_price = price
        elif outcome == "NO":
            no_token = token_id
            no_price = price

    # Fallback: some markets store prices at root level
    if yes_price == 0.5:
        op = raw.get("outcomePrices")
        if isinstance(op, list) and len(op) >= 2:
            try:
                yes_price = float(op[0])
                no_price  = float(op[1])
            except (TypeError, ValueError):
                pass

    liquidity = 0.0
    try:
        liquidity = float(raw.get("liquidity") or raw.get("liquidityNum") or 0)
    except (TypeError, ValueError):
        pass

    return {
        "conditionId":   cid,
        "question":      question,
        "active":        raw.get("active", True),
        "probabilities": {"YES": round(yes_price, 4), "NO": round(no_price, 4)},
        "liquidity":     liquidity,
        "volume24h":     float(raw.get("volume24hr") or raw.get("volume") or 0),
        "endDate":       raw.get("endDate") or raw.get("end_date_iso") or raw.get("endDateIso"),
        "yesTokenId":    yes_token,
        "noTokenId":     no_token,
        "negRisk":       raw.get("negRisk", False),
    }


def fetch_all_active_markets(limit: int = 100) -> List[dict]:
    data = _safe_get(f"{GAMMA_API}/markets", params={
        "active": "true",
        "closed": "false",
        "limit":  limit,
        "order":  "volume24hr",
        "ascending": "false",
    })

    # Handle both list response and dict with markets key
    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict):
        raw_list = data.get("markets") or data.get("data") or []
    else:
        raw_list = []

    result = []
    for item in raw_list:
        norm = _normalize_market(item)
        if norm:
            result.append(norm)

    print(f"[Discovery] {len(result)} markets normalized from {len(raw_list)} raw")
    return result


def fetch_weather_markets() -> List[dict]:
    weather_terms = ["temperature", "degrees", "fahrenheit", "celsius", "°f", "°c", "weather", "high temp"]
    markets = fetch_all_active_markets(limit=200)
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
        tokens = data.get("tokens") or data.get("outcomes") or []
        for t in tokens:
            if not isinstance(t, dict):
                continue
            if str(t.get("outcome", "")).upper() == side.upper():
                price = t.get("price")
                if price is None:
                    op = t.get("outcomePrices")
                    price = op[0] if isinstance(op, list) and op else None
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
    # Correct Polymarket Data API endpoint for positions
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
