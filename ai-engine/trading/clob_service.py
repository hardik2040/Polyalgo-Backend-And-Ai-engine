"""
Polymarket CLOB order execution service.
Uses py_clob_client for live trading, simulates for paper trading.
"""
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

PRIVATE_KEY    = os.environ.get("PRIVATE_KEY", "")
WALLET_ADDRESS = os.environ.get("WALLET_ADDRESS", "")
SIGNATURE_TYPE = int(os.environ.get("SIGNATURE_TYPE", "1"))
CLOB_API       = "https://clob.polymarket.com"
CHAIN_ID       = 137  # Polygon mainnet


def _get_clob_client():
    if not PRIVATE_KEY:
        return None
    try:
        from py_clob_client.client import ClobClient
        client = ClobClient(
            host=CLOB_API,
            key=PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=SIGNATURE_TYPE,
            funder=WALLET_ADDRESS or None,
        )
        return client
    except Exception as e:
        print(f"[CLOB] Client init error: {e}")
        return None


def place_order(token_id: str, price: float, size_usd: float,
                side: str = "BUY", neg_risk: bool = False,
                paper_trade: bool = True) -> dict:
    if paper_trade or not PRIVATE_KEY:
        order_id = str(uuid.uuid4())[:16]
        print(f"[PAPER] {side} token={token_id[:16]} price={price:.3f} size=${size_usd:.2f}")
        return {"orderId": order_id, "mode": "paper_trade", "status": "filled"}

    try:
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL

        client = _get_clob_client()
        if not client:
            raise Exception("CLOB client unavailable — check PRIVATE_KEY in .env")

        clob_side = BUY if side.upper() == "BUY" else SELL
        size = round(size_usd / price, 4) if side.upper() == "BUY" else size_usd

        order_args = OrderArgs(
            token_id=token_id,
            price=round(price, 4),
            size=size,
            side=clob_side,
            neg_risk=neg_risk,
        )
        signed_order = client.create_order(order_args)
        resp = client.post_order(signed_order, OrderType.GTC)
        order_id = resp.get("orderID") or resp.get("id") or str(uuid.uuid4())[:16]
        print(f"[LIVE] {side} order placed: {order_id} token={token_id[:16]} price={price:.3f} size={size}")
        return {"orderId": order_id, "mode": "live", "status": "open", "response": resp}

    except Exception as e:
        print(f"[CLOB] Order error: {e}")
        raise


def cancel_order(order_id: str) -> dict:
    if not PRIVATE_KEY:
        return {"cancelled": True, "mode": "paper_trade"}
    try:
        client = _get_clob_client()
        if client:
            resp = client.cancel(order_id)
            return {"cancelled": True, "response": resp}
    except Exception as e:
        print(f"[CLOB] Cancel error: {e}")
    return {"cancelled": False}


def get_wallet_balance(paper_trade: bool = True, mock_balance: float = 2000.0) -> float:
    if paper_trade or not PRIVATE_KEY:
        return mock_balance
    try:
        import requests
        r = requests.get(
            f"https://api.polygonscan.com/api",
            params={
                "module": "account", "action": "tokenbalance",
                "contractaddress": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                "address": WALLET_ADDRESS, "tag": "latest",
            },
            timeout=5,
        )
        result = r.json().get("result", "0")
        return round(int(result) / 1e6, 2)
    except Exception:
        return 0.0
