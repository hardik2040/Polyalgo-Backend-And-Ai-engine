"""
Full Automated Trading Orchestrator
"""
import os
import json
import time
import uuid
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

from data_ingestion.polymarket_discovery import (
    fetch_weather_markets, fetch_all_active_markets,
    get_user_positions, fetch_market_price, fetch_gamma_price, fetch_clob_midpoint
)
from data_ingestion.weather_service import analyze_temperature_question, is_weather_question
from data_ingestion.external_apis import fetch_external_signals
from strategy.ev_calculator import calculate_ev, kelly_criterion
from rl.q_agent import agent as rl_agent, encode_state
from trading.clob_service import place_order, cancel_order, get_wallet_balance
from utils.db_sync import sync_trade_to_db, sync_prediction_to_db

MAX_STAKE_USD              = float(os.environ.get("MAX_STAKE_USD", "1000"))
MAX_TOTAL_EXPOSURE         = float(os.environ.get("MAX_TOTAL_EXPOSURE", "5000"))
MIN_EV_THRESHOLD           = float(os.environ.get("MIN_EV_THRESHOLD", "0.05"))
MIN_CONFIDENCE             = float(os.environ.get("MIN_CONFIDENCE", "0.60"))
MIN_CONFIDENCE_NON_WEATHER = float(os.environ.get("MIN_CONFIDENCE_NON_WEATHER", "0.35"))
STOP_LOSS_PCT              = float(os.environ.get("STOP_LOSS_PCT", "-95"))
TAKE_PROFIT_PCT            = float(os.environ.get("TAKE_PROFIT_PCT", "80"))
WALLET_ADDRESS             = os.environ.get("WALLET_ADDRESS", "").strip()
SCAN_INTERVAL_SEC          = int(os.environ.get("SCAN_INTERVAL_SEC", "60"))
MIN_DAYS_TO_EXPIRY         = float(os.environ.get("MIN_DAYS_TO_EXPIRY", "0.15"))
MIN_MARKET_PRICE           = float(os.environ.get("MIN_MARKET_PRICE", "0.10"))
MAX_MARKET_PRICE           = float(os.environ.get("MAX_MARKET_PRICE", "0.90"))
MIN_LIQUIDITY_USD          = float(os.environ.get("MIN_LIQUIDITY_USD", "100"))
CLOSED_MARKET_COOLDOWN_HOURS = int(os.environ.get("CLOSED_MARKET_COOLDOWN_HOURS", "24"))
MIN_WIN_RATE_FOR_LARGE_STAKES = 0.0

STATE_PATH     = os.path.join(os.path.dirname(__file__), "bot_state.json")
TRADE_LOG_PATH = os.path.join(os.path.dirname(__file__), "trade_log.jsonl")
SCAN_LOG_PATH  = os.path.join(os.path.dirname(__file__), "scan_log.jsonl")


def _days_until_expiry(end_date_str: Optional[str]) -> Optional[float]:
    if not end_date_str:
        return None
    try:
        end_dt = datetime.strptime(end_date_str[:19], "%Y-%m-%dT%H:%M:%S")
        return (end_dt - datetime.utcnow()).total_seconds() / 86400
    except Exception:
        pass
    try:
        end_dt = datetime.strptime(end_date_str[:10], "%Y-%m-%d")
        return (end_dt - datetime.utcnow()).total_seconds() / 86400
    except Exception:
        return None


def _write_scan_log(entry: dict):
    try:
        with open(SCAN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        print(f"[ScanLog] Write error: {e}")


def _check_data_quality(market: dict) -> dict:
    end_date  = market.get("endDate")
    days_left = _days_until_expiry(end_date)
    yes_prob  = market.get("probabilities", {}).get("YES")
    no_prob   = market.get("probabilities", {}).get("NO")
    liquidity = float(market.get("liquidity") or 0)
    yes_token = market.get("yesTokenId")
    no_token  = market.get("noTokenId")

    issues = []
    if not market.get("conditionId"):
        issues.append("missing_condition_id")
    if not end_date:
        issues.append("missing_end_date")
    if days_left is None:
        issues.append("end_date_unparseable")
    if yes_prob is None or (yes_prob == 0.5 and no_prob == 0.5):
        issues.append("default_50_50_price")
    if not yes_token:
        issues.append("missing_yes_token_id")
    if not no_token:
        issues.append("missing_no_token_id")
    if liquidity == 0:
        issues.append("zero_liquidity")
    if yes_prob is not None and (yes_prob <= 0 or yes_prob >= 1):
        issues.append("invalid_probability")

    return {
        "has_end_date":   bool(end_date),
        "end_date_raw":   end_date,
        "days_to_expiry": round(days_left, 3) if days_left is not None else None,
        "yes_prob":       yes_prob,
        "no_prob":        no_prob,
        "liquidity":      liquidity,
        "has_yes_token":  bool(yes_token),
        "has_no_token":   bool(no_token),
        "issues":         issues,
        "ok":             len(issues) == 0,
    }


def _city_date_key(question: str, end_date: Optional[str]) -> str:
    q = question.lower()
    date_part = (end_date or "unknown")[:10]
    city = "unknown"
    if "temperature in " in q:
        after = q.split("temperature in ")[1]
        city = after.split(" be ")[0].strip()
    return f"{city}_{date_part}"


class TradingOrchestrator:

    def __init__(self):
        self.is_running = False
        self.positions: Dict[str, dict] = {}
        self.signals: List[dict] = []
        self.trade_log: List[dict] = []
        self.total_pnl = 0.0
        self.scan_count = 0
        self.paper_trade_mode = True
        self.weather_only = True
        self.min_ev_threshold = MIN_EV_THRESHOLD
        self.min_confidence = MIN_CONFIDENCE
        self.paper_balance = 2000.0
        self.recently_closed: Dict[str, str] = {}
        self.pinned_positions: set = set()
        self._thread: Optional[threading.Thread] = None
        self._load_state()
        print(
            f"[Orchestrator] Init. MaxStake=${MAX_STAKE_USD}, MaxExp=${MAX_TOTAL_EXPOSURE}, "
            f"WeatherOnly={self.weather_only}, MinDays={MIN_DAYS_TO_EXPIRY}"
        )

    def _load_state(self):
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH) as f:
                    data = json.load(f)
                self.positions        = data.get("positions", {})
                self.trade_log        = data.get("trade_log", [])[-100:]
                self.total_pnl        = data.get("total_pnl", 0.0)
                self.paper_trade_mode = data.get("paper_trade_mode", True)
                self.weather_only     = data.get("weather_only", True)
                self.min_ev_threshold = data.get("min_ev_threshold", MIN_EV_THRESHOLD)
                self.min_confidence   = data.get("min_confidence", MIN_CONFIDENCE)
                self.paper_balance    = data.get("paper_balance", 2000.0)
                self.recently_closed  = data.get("recently_closed", {})
                self.pinned_positions = set(data.get("pinned_positions", []))
                self._purge_expired_cooldowns()
            except Exception as e:
                print(f"[Orchestrator] State load error: {e}")

    def _save_state(self):
        self._purge_expired_cooldowns()
        try:
            with open(STATE_PATH, "w") as f:
                json.dump({
                    "positions":        self.positions,
                    "trade_log":        self.trade_log[-100:],
                    "total_pnl":        self.total_pnl,
                    "paper_trade_mode": self.paper_trade_mode,
                    "weather_only":     self.weather_only,
                    "min_ev_threshold": self.min_ev_threshold,
                    "min_confidence":   self.min_confidence,
                    "paper_balance":    self.paper_balance,
                    "recently_closed":  self.recently_closed,
                    "pinned_positions": list(self.pinned_positions),
                    "last_updated":     datetime.utcnow().isoformat(),
                }, f, indent=2)
        except Exception as e:
            print(f"[Orchestrator] State save error: {e}")

    def _purge_expired_cooldowns(self):
        cutoff = datetime.utcnow() - timedelta(hours=CLOSED_MARKET_COOLDOWN_HOURS)
        expired = [
            cid for cid, ts in self.recently_closed.items()
            if datetime.fromisoformat(ts) < cutoff
        ]
        for cid in expired:
            del self.recently_closed[cid]

    def _is_in_cooldown(self, condition_id: str) -> bool:
        if condition_id not in self.recently_closed:
            return False
        closed_at = datetime.fromisoformat(self.recently_closed[condition_id])
        return (datetime.utcnow() - closed_at).total_seconds() < CLOSED_MARKET_COOLDOWN_HOURS * 3600

    def _current_win_rate(self) -> float:
        closed = [t for t in self.trade_log
                  if t.get("status") == "CLOSED" and t.get("action") == "SELL"]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.get("realizedPnl", 0) > 0)
        return wins / len(closed)

    # ── Core loop ──────────────────────────────────────────────────────────────

    def _scan_and_trade(self):
        self.scan_count += 1
        win_rate = self._current_win_rate()
        print(f"\n[Scan #{self.scan_count}] {datetime.utcnow().isoformat()} "
              f"| WeatherOnly={self.weather_only} | WinRate={win_rate:.1%}")

        total_exposure = sum(p.get("stake_usd", 0) for p in self.positions.values())
        if total_exposure < MAX_TOTAL_EXPOSURE:
            self._discover_and_enter_trades(win_rate)

        self._monitor_positions()
        self._save_state()

    def _discover_and_enter_trades(self, win_rate: float):
        scan_ts = datetime.utcnow().isoformat()
        try:
            if self.weather_only:
                markets = fetch_weather_markets()
            else:
                markets = fetch_all_active_markets(limit=150)
        except Exception as e:
            print(f"[Discovery] Error: {e}")
            return

        candidates = []
        for market in markets:
            cid             = market.get("conditionId", "")
            question        = market.get("question", "")
            market_prob_yes = market.get("probabilities", {}).get("YES", 0.5)
            end_date        = market.get("endDate")
            liquidity       = float(market.get("liquidity") or 0)
            dq              = _check_data_quality(market)

            if cid in self.positions:
                continue
            if self._is_in_cooldown(cid):
                continue

            days_left = dq["days_to_expiry"]
            if days_left is None or days_left < MIN_DAYS_TO_EXPIRY:
                continue
            if market_prob_yes < MIN_MARKET_PRICE or market_prob_yes > MAX_MARKET_PRICE:
                continue
            if liquidity < MIN_LIQUIDITY_USD:
                continue

            candidates.append((market, days_left, dq))

        scored = []
        new_signals = []

        for market, days_left, dq in candidates:
            cid             = market["conditionId"]
            question        = market["question"]
            market_prob_yes = market["probabilities"]["YES"]
            end_date        = market.get("endDate")

            if is_weather_question(question):
                analysis    = analyze_temperature_question(question)
                true_prob   = analysis.get("probability", 0.5)
                confidence  = analysis.get("confidence", 0.1)
                data_source = analysis.get("source", "open-meteo")
                thinking    = (
                    f"Weather: {analysis.get('actual_temp_f','?')}°F vs {analysis.get('threshold_f','?')}°F. "
                    f"Prob={true_prob:.2f} conf={confidence:.2f}."
                )
            else:
                signals     = fetch_external_signals(question)
                sentiment   = signals.get("sentiment_score", 0)
                true_prob   = max(0.02, min(0.98, market_prob_yes + sentiment * 0.12))
                confidence  = signals.get("confidence", 0.25)
                data_source = signals.get("source", "sentiment_model")
                thinking    = f"Sentiment={sentiment:.2f} Prob={true_prob:.2f} conf={confidence:.2f}."

            yes_ev = calculate_ev(true_prob, market_prob_yes)
            no_true_prob   = 1.0 - true_prob
            no_market_prob = 1.0 - market_prob_yes
            no_ev = calculate_ev(no_true_prob, no_market_prob) if no_market_prob > 0.01 else -999

            if yes_ev >= no_ev and yes_ev >= self.min_ev_threshold:
                trade_side = "YES"
                ev         = yes_ev
                kelly      = kelly_criterion(true_prob, market_prob_yes)
                token_id   = market.get("yesTokenId")
                thinking  += f" YES EV={yes_ev:+.3f} -> Trade YES"
            elif no_ev > yes_ev and no_ev >= self.min_ev_threshold:
                trade_side = "NO"
                ev         = no_ev
                kelly      = kelly_criterion(no_true_prob, no_market_prob)
                token_id   = market.get("noTokenId")
                thinking  += f" NO EV={no_ev:+.3f} -> Trade NO"
            else:
                trade_side = None
                ev         = max(yes_ev, no_ev)
                kelly      = 0.0
                token_id   = None

            signal = {
                "conditionId":             cid,
                "question":                question[:100],
                "marketProbabilityAtTime": market_prob_yes,
                "trueProbability":         true_prob,
                "confidence":              confidence,
                "ev":                      ev,
                "tradeSide":               trade_side or "SKIP",
                "kellyFraction":           kelly,
                "dataSource":              data_source,
                "daysToExpiry":            round(days_left, 2),
                "timestamp":               datetime.utcnow().isoformat(),
                "thinking":                thinking,
            }
            new_signals.append(signal)

            required_confidence = self.min_confidence if is_weather_question(question) else MIN_CONFIDENCE_NON_WEATHER
            if confidence < required_confidence:
                continue
            if trade_side is None or not token_id:
                continue

            scored.append({
                "market":           market,
                "cid":              cid,
                "question":         question,
                "days_left":        days_left,
                "end_date":         end_date,
                "trade_side":       trade_side,
                "true_prob":        true_prob,
                "confidence":       confidence,
                "ev":               ev,
                "kelly":            kelly,
                "token_id":         token_id,
                "market_prob_yes":  market_prob_yes,
                "signal":           signal,
                "score":            confidence * ev,
            })

        scored.sort(key=lambda x: -x["score"])
        city_date_committed: set = set()

        for item in scored:
            cd_key = _city_date_key(item["question"], item["end_date"])
            if cd_key in city_date_committed:
                continue
            city_date_committed.add(cd_key)
            self._execute_trade(item, win_rate)

        self.signals = (new_signals + self.signals)[:50]

    def _execute_trade(self, item: dict, win_rate: float):
        cid        = item["cid"]
        question   = item["question"]
        trade_side = item["trade_side"]
        token_id   = item["token_id"]
        ev         = item["ev"]
        confidence = item["confidence"]
        kelly      = item["kelly"]
        true_prob  = item["true_prob"]
        market     = item["market"]
        days_left  = item["days_left"]
        market_prob_yes = item["market_prob_yes"]

        rl_decision = rl_agent.choose_action(
            ev=ev, confidence=confidence,
            pnl_pct=0.0, market_prob=market_prob_yes,
            has_position=False
        )

        if rl_decision["action"] == "HOLD" or rl_decision["stake_multiplier"] == 0:
            return

        stake_mult = rl_decision["stake_multiplier"]
        if win_rate < MIN_WIN_RATE_FOR_LARGE_STAKES:
            stake_mult = min(stake_mult, 0.25)

        available  = self.paper_balance if self.paper_trade_mode else MAX_STAKE_USD * 20
        dynamic_max = min(MAX_STAKE_USD * 5, available * 0.05)
        ev_boost    = min(1.5, 1.0 + max(0.0, ev - 0.30) * 1.0)
        kelly_stake = kelly * available * 0.10
        stake = round(min(dynamic_max * stake_mult * ev_boost, kelly_stake, dynamic_max), 2)

        if stake < 1.0:
            return

        price = round(market_prob_yes if trade_side == "YES" else 1.0 - market_prob_yes, 2)

        try:
            order = place_order(
                token_id=token_id, price=price, size_usd=stake,
                side="BUY", neg_risk=market.get("negRisk", False),
                paper_trade=self.paper_trade_mode
            )

            trade_id = str(uuid.uuid4())[:16]
            pos = {
                "tradeId":           trade_id,
                "conditionId":       cid,
                "question":          question[:80],
                "tokenId":           token_id,
                "side":              trade_side,
                "entryPrice":        price,
                "stake_usd":         stake,
                "sharesHeld":        round(stake / price, 4) if price > 0 else 0,
                "orderId":           order.get("orderId"),
                "rl_state":          rl_decision,
                "rl_action_idx":     rl_decision["action_idx"],
                "status":            "OPEN",
                "enteredAt":         datetime.utcnow().isoformat(),
                "ev":                ev,
                "daysToExpiry":      round(days_left, 1),
                "trueProbability":   true_prob,
                "marketProbAtEntry": price,
                "confidence":        confidence,
                "mode":              order.get("mode", "live"),
                "pnl_usd":           0.0,
                "pnl_pct":           0.0,
                "currentPrice":      price,
                "sellThinking":      "Monitoring — waiting for take-profit or stop-loss trigger.",
            }
            self.positions[cid] = pos

            if order.get("mode") == "paper_trade":
                self.paper_balance = round(self.paper_balance - stake, 2)
                print(f"[PAPER] -{stake}. Balance: ${self.paper_balance}")

            buy_log = {**pos, "action": "BUY", "rl_decision": rl_decision,
                       "loggedAt": datetime.utcnow().isoformat()}
            self.trade_log.append(buy_log)
            self._append_trade_log(buy_log)

            print(f"[TRADE-BUY] tradeId={trade_id} side={trade_side} ev={ev:.3f} "
                  f"stake=${stake} price={price:.3f} bal=${self.paper_balance:.0f}")

            sync_prediction_to_db(item["signal"])
            sync_trade_to_db(pos)

        except Exception as e:
            print(f"[Trade Error] {cid}: {e}")

    def _monitor_positions(self):
        for cid, pos in list(self.positions.items()):
            if pos.get("status") != "OPEN":
                continue

            try:
                entered_at  = datetime.fromisoformat(pos.get("enteredAt", "1970-01-01"))
                age_seconds = (datetime.utcnow() - entered_at).total_seconds()
                if age_seconds < 300:
                    continue
            except Exception:
                pass

            token_id    = pos.get("tokenId")
            entry_price = pos.get("entryPrice", 0.5)

            current_price = fetch_clob_midpoint(token_id) if token_id else None
            if not current_price or current_price <= 0:
                current_price = fetch_gamma_price(cid, pos.get("side", "YES"))
            if not current_price or current_price <= 0:
                current_price = fetch_market_price(token_id) if token_id else None
            if not current_price or current_price <= 0:
                continue

            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            pnl_usd = (current_price - entry_price) * pos.get("stake_usd", 0) / entry_price

            pos["currentPrice"] = current_price
            pos["pnl_pct"]      = round(pnl_pct, 2)
            pos["pnl_usd"]      = round(pnl_usd, 2)

            if cid in self.pinned_positions:
                pos["sellThinking"] = (
                    f"Pinned by user — holding despite any exit signals. "
                    f"P&L {pnl_pct:+.1f}%. Unpin to allow auto-management."
                )
                continue

            exit_rec = rl_agent.get_exit_recommendation(
                current_pnl_pct=pnl_pct,
                market_prob=current_price,
                ev=pos.get("ev", 0),
                confidence=pos.get("confidence", 0.5)
            )

            if exit_rec["action"] == "SELL":
                pos["sellThinking"] = f"CRITICAL: {exit_rec['reason']}. Executing SELL at {current_price:.3f}."
                self._close_position(cid, pos, current_price, exit_rec["reason"])
            else:
                pos["sellThinking"] = (
                    f"Holding: P&L {pnl_pct:+.1f}%. "
                    f"RL: HOLD ({exit_rec.get('reason', 'no_exit_signal')})."
                )

    def _close_position(self, cid: str, pos: dict, exit_price: float, reason: str):
        try:
            token_id    = pos.get("tokenId")
            trade_id    = pos.get("tradeId", "unknown")
            entry_price = pos["entryPrice"]

            if token_id:
                shares_held = pos.get("sharesHeld") or (
                    round(pos["stake_usd"] / entry_price, 4) if entry_price > 0 else 0
                )
                if shares_held > 0:
                    try:
                        place_order(
                            token_id=token_id, price=exit_price, size_usd=shares_held,
                            side="SELL", neg_risk=False, paper_trade=self.paper_trade_mode
                        )
                    except Exception as sell_err:
                        print(f"   [SELL-ORDER-ERROR] {trade_id}: {sell_err}")

            outcome = rl_agent.record_trade_outcome(
                entry_price=entry_price,
                exit_price=exit_price,
                stake_usd=pos["stake_usd"],
                entry_state=pos.get("rl_state", {}),
                action_idx=pos.get("rl_action_idx", 0)
            )

            pos["status"]      = "CLOSED"
            pos["exitPrice"]   = exit_price
            pos["exitReason"]  = reason
            pos["closedAt"]    = datetime.utcnow().isoformat()
            pos["realizedPnl"] = outcome.get("pnl_usd", 0)

            self.total_pnl += outcome.get("pnl_usd", 0)

            if pos.get("mode") == "paper_trade":
                realized_amt = pos.get("stake_usd", 0) + outcome.get("pnl_usd", 0)
                self.paper_balance = round(self.paper_balance + realized_amt, 2)
                print(f"[PAPER] +${realized_amt:.2f} back. Balance: ${self.paper_balance}")

            self.recently_closed[cid] = datetime.utcnow().isoformat()
            sell_log = {**pos, "action": "SELL", "rl_outcome": outcome,
                        "tradeId": trade_id, "loggedAt": datetime.utcnow().isoformat()}
            self.trade_log.append(sell_log)
            self._append_trade_log(sell_log)
            del self.positions[cid]

            pnl = outcome.get("pnl_usd", 0)
            print(f"[TRADE-SELL] tradeId={trade_id} reason={reason} "
                  f"entry={entry_price:.3f} exit={exit_price:.3f} P&L=${pnl:.2f}")

            sync_trade_to_db({**pos, "status": "CLOSED", "realizedPnl": pnl})

        except Exception as e:
            print(f"[Close Error] {cid}: {e}")

    def _append_trade_log(self, entry: dict):
        try:
            with open(TRADE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            print(f"[TradeLog] Write error: {e}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Orchestrator] Bot started.")

    def stop(self):
        self.is_running = False
        print("[Orchestrator] Bot stopped.")

    def _loop(self):
        while self.is_running:
            try:
                self._scan_and_trade()
            except Exception as e:
                print(f"[Orchestrator] Loop error: {e}")
            time.sleep(SCAN_INTERVAL_SEC)

    def get_status(self) -> dict:
        win_rate = self._current_win_rate()
        return {
            "is_running":       self.is_running,
            "scan_count":       self.scan_count,
            "open_positions":   len(self.positions),
            "total_pnl":        round(self.total_pnl, 2),
            "rl_stats":         rl_agent.get_stats(),
            "trade_count":      len(self.trade_log),
            "balance":          get_wallet_balance(paper_trade=self.paper_trade_mode, mock_balance=self.paper_balance),
            "paper_trade_mode": self.paper_trade_mode,
            "weather_only":     self.weather_only,
            "min_ev_threshold": self.min_ev_threshold,
            "min_confidence":   self.min_confidence,
            "paper_balance":    self.paper_balance,
            "win_rate":         round(win_rate, 3),
        }

    def get_positions(self) -> list:
        result = []
        for cid, pos in self.positions.items():
            p = dict(pos)
            p["pinned"] = cid in self.pinned_positions
            result.append(p)
        return result

    def manual_sell(self, cid: str) -> dict:
        if cid not in self.positions:
            return {"success": False, "error": "Position not found"}
        pos = self.positions[cid]
        self.pinned_positions.discard(cid)
        token_id = pos.get("tokenId")
        current_price = fetch_clob_midpoint(token_id) if token_id else None
        if not current_price or current_price <= 0:
            current_price = fetch_gamma_price(cid, pos.get("side", "YES"))
        if not current_price or current_price <= 0:
            current_price = pos.get("currentPrice") or pos.get("entryPrice", 0.5)
        self._close_position(cid, pos, current_price, "manual_sell")
        self._save_state()
        return {"success": True, "exitPrice": current_price, "pnl": pos.get("realizedPnl", 0)}

    def pin_position(self, cid: str, pinned: bool) -> dict:
        if cid not in self.positions:
            return {"success": False, "error": "Position not found"}
        if pinned:
            self.pinned_positions.add(cid)
        else:
            self.pinned_positions.discard(cid)
        self._save_state()
        return {"success": True, "conditionId": cid, "pinned": pinned}

    def get_signals(self) -> list:
        return self.signals[:20]

    def get_trade_log(self) -> list:
        return self.trade_log[-30:]


orchestrator = TradingOrchestrator()
