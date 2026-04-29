"""
PolyAlgo AI Engine — Full Automated Trading Backend
FastAPI server exposing all AI, weather, RL, and trading endpoints.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv

load_dotenv()

from models.predictor import ModelPredictor
from strategy.ev_calculator import calculate_ev, kelly_criterion
from data_ingestion.external_apis import fetch_external_signals
from data_ingestion.weather_service import analyze_temperature_question, is_weather_question
from data_ingestion.polymarket_discovery import (
    fetch_all_active_markets, fetch_weather_markets,
    get_user_positions, get_user_trades, get_portfolio_value
)
from rl.q_agent import agent as rl_agent, encode_state
from trading.orchestrator import orchestrator

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[AI-ENGINE] Server starting up... Auto-starting trading bot.")
    orchestrator.start()
    yield
    print("[AI-ENGINE] Server shutting down... Stopping orchestrator.")
    orchestrator.stop()

app = FastAPI(title="PolyAlgo AI Engine", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ModelPredictor()

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class MarketContext(BaseModel):
    conditionId: str
    question: str
    marketProbability: float
    liquidity: float
    orderType: Optional[str] = "LIMIT"

class ExternalPrediction(BaseModel):
    conditionId: str
    question: str
    marketProbability: float
    liquidity: float
    probability: float
    confidence: float
    orderType: Optional[str] = "LIMIT"

class RLUpdateRequest(BaseModel):
    entry_price: float
    exit_price: float
    stake_usd: float
    entry_state: dict
    action_idx: int

class RLActionRequest(BaseModel):
    ev: float
    confidence: float
    pnl_pct: Optional[float] = 0.0
    market_prob: Optional[float] = 0.5
    has_position: Optional[bool] = False

class PinRequest(BaseModel):
    pinned: bool

# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH & STATUS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    try:
        rl_stats = rl_agent.get_stats()
    except Exception as e:
        rl_stats = {"error": str(e)}
    return {
        "status": "ok",
        "service": "ai-engine",
        "version": "3.0.0",
        "ai_provider": "puter.js (client-side GPT) + Open-Meteo (server-side weather)",
        "rl_status": rl_stats
    }

# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/weather/predict")
def predict_weather_market(question: str):
    if not is_weather_question(question):
        return {
            "is_weather_question": False,
            "probability": 0.50,
            "confidence": 0.05,
            "reason": "not_a_weather_question"
        }
    result = analyze_temperature_question(question)
    result["is_weather_question"] = True
    return result


@app.get("/markets/weather")
def get_weather_markets():
    try:
        markets = fetch_weather_markets()
        enriched = []
        for m in markets[:30]:
            analysis = analyze_temperature_question(m["question"])
            ev = calculate_ev(
                analysis.get("probability", m["probabilities"]["YES"]),
                m["probabilities"]["YES"]
            )
            m["aiAnalysis"] = analysis
            m["ev"] = ev
            enriched.append(m)
        return {"count": len(enriched), "markets": enriched}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/markets/all")
def get_all_markets(limit: int = 100, weather_only: bool = False):
    try:
        if weather_only:
            markets = fetch_weather_markets()
        else:
            markets = fetch_all_active_markets(limit=limit)
        return {"count": len(markets), "markets": markets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# AI PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/predict")
def predict_market_outcome(context: MarketContext):
    try:
        if is_weather_question(context.question):
            analysis = analyze_temperature_question(context.question)
            true_prob = analysis.get("probability", 0.5)
            confidence = analysis.get("confidence", 0.1)
            signals = {"source": "open-meteo-real-data", **analysis}
        else:
            signals = fetch_external_signals(context.question)
            sentiment = signals.get("sentiment_score", 0)
            true_prob = max(0.02, min(0.98, context.marketProbability + sentiment * 0.1))
            confidence = 0.25 + abs(sentiment) * 0.2

        ev = calculate_ev(true_prob, context.marketProbability)
        kelly_fraction = kelly_criterion(true_prob, context.marketProbability)

        rl_decision = rl_agent.choose_action(
            ev=ev, confidence=confidence,
            market_prob=context.marketProbability,
            has_position=False
        )

        return {
            "conditionId": context.conditionId,
            "trueProbability": true_prob,
            "marketProbabilityAtTime": context.marketProbability,
            "confidence": confidence,
            "ev": ev,
            "kellyFraction": kelly_fraction,
            "signals": signals,
            "rl_recommendation": rl_decision,
            "ai_source": "open-meteo" if is_weather_question(context.question) else "sentiment_model"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-external")
def predict_from_puter(payload: ExternalPrediction):
    try:
        signals = fetch_external_signals(payload.question)
        pred_result = predictor.predict_from_external(payload.probability, payload.confidence)
        true_prob = pred_result["probability"]
        confidence = pred_result["confidence"]

        ev = calculate_ev(true_prob, payload.marketProbability)
        kelly_fraction = kelly_criterion(true_prob, payload.marketProbability)

        rl_decision = rl_agent.choose_action(
            ev=ev, confidence=confidence,
            market_prob=payload.marketProbability,
            has_position=False
        )

        return {
            "conditionId": payload.conditionId,
            "trueProbability": true_prob,
            "marketProbabilityAtTime": payload.marketProbability,
            "confidence": confidence,
            "ev": ev,
            "kellyFraction": kelly_fraction,
            "signals": signals,
            "rl_recommendation": rl_decision,
            "ai_source": "puter.js GPT (client-side)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/rl/stats")
def get_rl_stats():
    return rl_agent.get_stats()


@app.post("/rl/action")
def get_rl_action(req: RLActionRequest):
    return rl_agent.choose_action(
        ev=req.ev,
        confidence=req.confidence,
        pnl_pct=req.pnl_pct or 0.0,
        market_prob=req.market_prob or 0.5,
        has_position=req.has_position or False
    )


@app.post("/rl/update")
def update_rl(req: RLUpdateRequest):
    result = rl_agent.record_trade_outcome(
        entry_price=req.entry_price,
        exit_price=req.exit_price,
        stake_usd=req.stake_usd,
        entry_state=req.entry_state,
        action_idx=req.action_idx
    )
    return result


@app.get("/rl/qtable")
def get_qtable_summary():
    states = list(rl_agent.q_table.items())[:50]
    return {
        "total_states": len(rl_agent.q_table),
        "sample_states": [
            {
                "state": s,
                "q_values": v,
                "best_action": ["HOLD", "BUY_SM", "BUY_MD", "BUY_LG", "SELL"][v.index(max(v))]
            }
            for s, v in states
        ]
    }

# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-TRADING BOT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/bot/start")
def start_bot(background_tasks: BackgroundTasks):
    orchestrator.start()
    return {"started": True, "status": orchestrator.get_status()}


@app.post("/bot/stop")
def stop_bot():
    orchestrator.stop()
    return {"stopped": True}


@app.get("/bot/status")
def bot_status():
    return orchestrator.get_status()


@app.post("/bot/mode")
def set_bot_mode(payload: Dict[str, bool]):
    is_paper = payload.get("is_paper", True)
    orchestrator.paper_trade_mode = is_paper
    orchestrator._save_state()
    return {"success": True, "paper_trade_mode": orchestrator.paper_trade_mode}


@app.post("/bot/settings")
def update_bot_settings(payload: Dict[str, Any]):
    print(f"[AI-ENGINE] Settings Update Received: {payload}")
    if "min_ev_threshold" in payload:
        orchestrator.min_ev_threshold = float(payload["min_ev_threshold"])
    if "min_confidence" in payload:
        orchestrator.min_confidence = float(payload["min_confidence"])
    if "paper_balance" in payload:
        orchestrator.paper_balance = float(payload["paper_balance"])
    if "weather_only" in payload:
        orchestrator.weather_only = bool(payload["weather_only"])
    orchestrator._save_state()
    return {"success": True, "settings": orchestrator.get_status()}


@app.get("/bot/positions")
def bot_positions():
    return {"positions": orchestrator.get_positions()}


@app.get("/bot/signals")
def bot_signals():
    return {"signals": orchestrator.get_signals()}


@app.get("/bot/trades")
def bot_trades():
    return {"trades": orchestrator.get_trade_log()}


@app.post("/positions/{condition_id}/sell")
def manual_sell_position(condition_id: str):
    result = orchestrator.manual_sell(condition_id)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Failed to sell"))
    return result


@app.post("/positions/{condition_id}/pin")
def pin_position_endpoint(condition_id: str, payload: PinRequest):
    result = orchestrator.pin_position(condition_id, payload.pinned)
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error", "Position not found"))
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO / WALLET
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/portfolio/positions")
def live_positions():
    wallet = os.environ.get("WALLET_ADDRESS", "")
    if not wallet:
        return {"positions": [], "error": "WALLET_ADDRESS not configured"}
    return {"positions": get_user_positions(wallet)}


@app.get("/portfolio/value")
def portfolio_value():
    wallet = os.environ.get("WALLET_ADDRESS", "")
    if not wallet:
        return {"total_value": 0, "error": "WALLET_ADDRESS not configured"}
    return get_portfolio_value(wallet)


@app.get("/portfolio/trades")
def trade_history():
    wallet = os.environ.get("WALLET_ADDRESS", "")
    if not wallet:
        return {"trades": []}
    return {"trades": get_user_trades(wallet)}
