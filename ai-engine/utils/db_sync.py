import requests

BACKEND_URL = "http://127.0.0.1:3000"


def sync_trade_to_db(trade_data: dict):
    try:
        requests.post(f"{BACKEND_URL}/api/sync/trade", json=trade_data, timeout=5)
    except Exception as e:
        print(f"[DB-Sync] Trade sync failed: {e}")


def sync_prediction_to_db(pred_data: dict):
    try:
        requests.post(f"{BACKEND_URL}/api/sync/prediction", json=pred_data, timeout=5)
    except Exception as e:
        print(f"[DB-Sync] Prediction sync failed: {e}")
