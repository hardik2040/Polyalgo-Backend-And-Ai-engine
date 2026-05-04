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
    get_user_positions, fetch_market_price, fetch_gamma_price, fetch_clob_midpoint,
    fetch_market_resolution
)
from data_ingestion.weather_service import analyze_temperature_question, is_weather_question
from data_ingestion.external_apis import fetch_external_signals
from strategy.ev_calculator import calculate_ev, kelly_criterion
from rl.q_agent import agent as rl_agent, encode_state
from trading.clob_service import place_order, cancel_order, get_wallet_balance
from utils.db_sync import sync_trade_to_db, sync_prediction_to_db

MIN_EV_THRESHOLD           = float(os.environ.get("MIN_EV_THRESHOLD", "0.05"))
MIN_CONFIDENCE             = float(os.environ.get("MIN_CONFIDENCE", "0.60"))
MIN_CONFIDENCE_NON_WEATHER = float(os.environ.get("MIN_CONFIDENCE_NON_WEATHER", "0.35"))
STOP_LOSS_PCT              = float(os.environ.get("STOP_LOSS_PCT", "-40"))
TAKE_PROFIT_PCT            = float(os.environ.get("TAKE_PROFIT_PCT", "80"))
WALLET_ADDRESS             = os.environ.get("WALLET_ADDRESS", "").strip()
SCAN_INTERVAL_SEC          = int(os.environ.get("SCAN_INTERVAL_SEC", "60"))
MIN_DAYS_TO_EXPIRY         = float(os.environ.get("MIN_DAYS_TO_EXPIRY", "0.15"))
MIN_MARKET_PRICE           = float(os.environ.get("MIN_MARKET_PRICE", "0.10"))
MAX_MARKET_PRICE           = float(os.environ.get("MAX_MARKET_PRICE", "0.90"))
MIN_LIQUIDITY_USD          = float(os.environ.get("MIN_LIQUIDITY_USD", "100"))
CLOSED_MARKET_COOLDOWN_HOURS = int(os.environ.get("CLOSED_MARKET_COOLDOWN_HOURS", "24"))
MIN_WIN_RATE_FOR_LARGE_STAKES = 0.0

# ── Percentage-based position sizing ──────────────────────────────────────────
# MAX_STAKE_PCT  : max % of balance per single trade  (default 5%)
# MAX_EXPOSURE_PCT: max % of balance across all open positions (default 80%)
MAX_STAKE_PCT    = float(os.environ.get("MAX_STAKE_PCT",    "5"))   / 100
MAX_EXPOSURE_PCT = float(os.environ.get("MAX_EXPOSURE_PCT", "80"))  / 100

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
        self.paper_balance = float(os.environ.get("PAPER_BALANCE", "100"))
        self.recently_closed: Dict[str, str] = {}
        self.pinned_positions: set = set()
        self._thread: Optional[threading.Thread] = None
        self._signal_items: Dict[str, dict] = {}  # full market data keyed by conditionId
        self._load_state()
        print(
            f"[Orchestrator] Init. MaxStakePct={MAX_STAKE_PCT:.0%}, MaxExposurePct={MAX_EXPOSURE_PCT:.0%}, "
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
                self.paper_balance    = data.get("paper_balance", float(os.environ.get("PAPER_BALANCE", "100")))
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

        available      = self.paper_balance if self.paper_trade_mode else get_wallet_balance(paper_trade=False)
        max_exposure   = available * MAX_EXPOSURE_PCT
        total_exposure = sum(p.get("stake_usd", 0) for p in self.positions.values())
        if total_exposure < max_exposure:
            self._discover_and_enter_trades(win_rate)

        self._monitor_positions()
        self._save_state()

    def _discover_and_enter_trades(self, win_rate: float):
        scan_ts = datetime.utcnow().isoformat()
        try:
            if self.weather_only:
                markets = fetch_weather_markets()
            else:
                markets = fetch_all_active_markets(limit=700)
        except Exception as e:
            import traceback
            print(f"[Discovery] Error: {e}")
            traceback.print_exc()
            return

        skip_already  = 0
        skip_cooldown = 0
        skip_expiry   = 0
        skip_price    = 0
        skip_liq      = 0

        candidates = []
        for market in markets:
            cid             = market.get("conditionId", "")
            question        = market.get("question", "")
            market_prob_yes = market.get("probabilities", {}).get("YES", 0.5)
            end_date        = market.get("endDate")
            liquidity       = float(market.get("liquidity") or 0)
            dq              = _check_data_quality(market)

            if cid in self.positions:
                skip_already += 1
                continue
            if self._is_in_cooldown(cid):
                skip_cooldown += 1
                continue

            days_left = dq["days_to_expiry"]
            if days_left is None or days_left < MIN_DAYS_TO_EXPIRY:
                skip_expiry += 1
                print(f"   [SKIP:expiry] {question[:60]} | days={days_left}")
                continue
            if market_prob_yes < MIN_MARKET_PRICE or market_prob_yes > MAX_MARKET_PRICE:
                skip_price += 1
                print(f"   [SKIP:price] {question[:60]} | YES={market_prob_yes:.3f}")
                continue
            if liquidity < MIN_LIQUIDITY_USD:
                skip_liq += 1
                print(f"   [SKIP:liquidity] {question[:60]} | liq=${liquidity:.0f}")
                continue

            candidates.append((market, days_left, dq))

        print(f"   [FILTER] total={len(markets)} candidates={len(candidates)} "
              f"skip: holding={skip_already} cooldown={skip_cooldown} "
              f"expiry={skip_expiry} price={skip_price} liq={skip_liq}")

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

            item = {
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
            }
            self._signal_items[cid] = item
            scored.append(item)

        scored.sort(key=lambda x: -x["score"])
        city_date_committed: set = set()

        for item in scored:
            cd_key = _city_date_key(item["question"], item["end_date"])
            if cd_key in city_date_committed:
                continue
            city_date_committed.add(cd_key)
            self._execute_trade(item, win_rate)

        self.signals = (new_signals + self.signals)[:50]
        active_cids = {s["conditionId"] for s in self.signals}
        self._signal_items = {cid: item for cid, item in self._signal_items.items() if cid in active_cids}

    def _execute_trade(self, item: dict, win_rate: float, user_initiated: bool = False):
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

        if not user_initiated:
            if rl_decision["action"] == "HOLD" or rl_decision["stake_multiplier"] == 0:
                return

        stake_mult = rl_decision["stake_multiplier"]
        if stake_mult == 0:
            stake_mult = 0.5  # user-initiated: default to BUY_MEDIUM
        if win_rate < MIN_WIN_RATE_FOR_LARGE_STAKES:
            stake_mult = min(stake_mult, 0.25)

        available   = self.paper_balance if self.paper_trade_mode else get_wallet_balance(paper_trade=False, mock_balance=self.paper_balance)
        max_stake   = available * MAX_STAKE_PCT          # e.g. 5% of $100 = $5
        ev_boost    = min(1.5, 1.0 + max(0.0, ev - 0.30) * 1.0)
        kelly_stake = kelly * available * 0.10           # Kelly fraction of 10% of balance
        stake = round(min(max_stake * stake_mult * ev_boost, kelly_stake, max_stake), 2)

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
                "source":            "user" if user_initiated else "bot",
                "endDate":           item.get("end_date"),
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

            # No warmup delay — monitor immediately for stop-loss protection
            try:
                entered_at  = datetime.fromisoformat(pos.get("enteredAt", "1970-01-01"))
                age_seconds = (datetime.utcnow() - entered_at).total_seconds()
                if age_seconds < 30:   # only skip first 30s to let order settle
                    continue
            except Exception:
                pass

            token_id    = pos.get("tokenId")
            entry_price = pos.get("entryPrice", 0.5)
            side        = pos.get("side", "YES")

            current_price = fetch_clob_midpoint(token_id) if token_id else None
            if not current_price or current_price <= 0:
                current_price = fetch_gamma_price(cid, side)
            if not current_price or current_price <= 0:
                current_price = fetch_market_price(token_id) if token_id else None
            if not current_price or current_price <= 0:
                current_price = entry_price  # fallback to entry price for P&L display

            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            pnl_usd = (current_price - entry_price) * pos.get("stake_usd", 0) / entry_price

            pos["currentPrice"] = current_price
            pos["pnl_pct"]      = round(pnl_pct, 2)
            pos["pnl_usd"]      = round(pnl_usd, 2)

            # ── 1. Market resolution check (highest priority) ─────────────────
            resolution = fetch_market_resolution(cid)
            if resolution["resolved"]:
                winner = resolution["winner"]
                if winner == side:
                    exit_price = 1.0
                    reason     = f"market_resolved_{winner}_WIN"
                    pos["sellThinking"] = f"Market resolved {winner} — WIN! Collecting full payout at $1.00/share."
                elif winner is not None:
                    exit_price = 0.0
                    reason     = f"market_resolved_{winner}_LOSS"
                    pos["sellThinking"] = f"Market resolved {winner} — LOSS. Position expired worthless."
                else:
                    exit_price = current_price
                    reason     = "market_resolved_unknown"
                    pos["sellThinking"] = "Market resolved with unknown outcome — closing at last price."
                self._close_position(cid, pos, exit_price, reason)
                continue

            # ── 2. Expiry check — close at current price if endDate passed ────
            end_date  = pos.get("endDate")
            days_left = _days_until_expiry(end_date) if end_date else None
            if days_left is not None and days_left <= 0:
                pos["sellThinking"] = f"Market expired {abs(days_left):.1f}d ago — closing at last known price."
                self._close_position(cid, pos, current_price, "expiry_close")
                continue

            # ── 3. Pinned — skip auto-exit ────────────────────────────────────
            if cid in self.pinned_positions:
                pos["sellThinking"] = (
                    f"Pinned by user — holding despite any exit signals. "
                    f"P&L {pnl_pct:+.1f}%. Unpin to allow auto-management."
                )
                continue

            # ── 4. RL stop-loss / take-profit / hold ──────────────────────────
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
                days_info = f" | {days_left:.1f}d left" if days_left is not None else ""
                pos["sellThinking"] = (
                    f"Holding: P&L {pnl_pct:+.1f}%{days_info}. "
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

    def trade_signal(self, cid: str) -> dict:
        if cid in self.positions:
            return {"success": False, "error": "Already have an open position in this market"}
        if self._is_in_cooldown(cid):
            return {"success": False, "error": "Market is in cooldown after a recent close"}
        if cid not in self._signal_items:
            return {"success": False, "error": "Signal not found — wait for the next scan (signals refresh every 60s)"}

        # Reject if signal is stale (> 10 minutes old)
        signal_data = next((s for s in self.signals if s["conditionId"] == cid), None)
        if signal_data:
            try:
                ts = datetime.fromisoformat(signal_data["timestamp"])
                age_sec = (datetime.utcnow() - ts).total_seconds()
                if age_sec > 600:
                    return {"success": False, "error": f"Signal is {age_sec/60:.0f}m old — wait for next scan to refresh"}
            except Exception:
                pass

        item = self._signal_items[cid]
        if not item.get("token_id"):
            return {"success": False, "error": "No valid token ID for this market — cannot place order"}
        if not item.get("trade_side") or item["trade_side"] == "SKIP":
            return {"success": False, "error": "Signal has no actionable trade direction (EV too low)"}

        win_rate = self._current_win_rate()
        try:
            self._execute_trade(item, win_rate, user_initiated=True)
        except Exception as e:
            return {"success": False, "error": str(e)}

        if cid in self.positions:
            self._save_state()
            return {
                "success": True,
                "position": self.positions[cid],
                "message": "Trade entered. AI engine will monitor and exit automatically via stop-loss/take-profit/RL.",
            }
        return {"success": False, "error": "Trade rejected — balance too low or stake below $1 minimum"}

    def get_signals(self) -> list:
        return self.signals[:20]

    def get_trade_log(self) -> list:
        return self.trade_log[-30:]

    def get_trade_reports(self) -> dict:
        _REASON_LABELS = {
            "stop_loss":                "Stop Loss (-40%)",
            "take_profit":              "Take Profit (+80%)",
            "manual_sell":              "Manual Sell",
            "market_resolved_YES_WIN":  "Resolved YES — WIN",
            "market_resolved_NO_WIN":   "Resolved NO  — WIN",
            "market_resolved_YES_LOSS": "Resolved YES — LOSS",
            "market_resolved_NO_LOSS":  "Resolved NO  — LOSS",
            "market_resolved_unknown":  "Resolved (unknown)",
            "expiry_close":             "Market Expired",
            "rl_exit":                  "RL Agent Exit",
        }

        closed = [t for t in self.trade_log if t.get("action") == "SELL"]
        reports = []
        for t in closed:
            entry  = float(t.get("entryPrice") or 0)
            exit_  = float(t.get("exitPrice")  or 0)
            stake  = float(t.get("stake_usd")  or 0)
            pnl    = float(t.get("realizedPnl") or 0)
            pnl_pct = round(((exit_ - entry) / entry * 100) if entry > 0 else 0, 1)
            reason = t.get("exitReason", "unknown")
            reports.append({
                "tradeId":     t.get("tradeId"),
                "question":    t.get("question"),
                "side":        t.get("side"),
                "mode":        t.get("mode", "paper_trade"),
                "entryPrice":  round(entry, 4),
                "exitPrice":   round(exit_, 4),
                "enteredAt":   t.get("enteredAt"),
                "closedAt":    t.get("closedAt"),
                "stakeUsd":    round(stake, 2),
                "realizedPnl": round(pnl, 2),
                "pnlPct":      pnl_pct,
                "exitReason":  reason,
                "exitLabel":   _REASON_LABELS.get(reason, reason),
                "ev":          round(float(t.get("ev") or 0), 4),
                "confidence":  round(float(t.get("confidence") or 0), 3),
                "daysToExpiry": t.get("daysToExpiry"),
            })

        reports.reverse()  # most recent first

        wins      = [r for r in reports if r["realizedPnl"] > 0]
        losses    = [r for r in reports if r["realizedPnl"] < 0]
        total_pnl = sum(r["realizedPnl"] for r in reports)

        by_reason: dict = {}
        for r in reports:
            lbl = r["exitLabel"]
            if lbl not in by_reason:
                by_reason[lbl] = {"count": 0, "pnl": 0.0}
            by_reason[lbl]["count"] += 1
            by_reason[lbl]["pnl"]    = round(by_reason[lbl]["pnl"] + r["realizedPnl"], 2)

        return {
            "reports": reports,
            "summary": {
                "total_trades": len(reports),
                "wins":         len(wins),
                "losses":       len(losses),
                "win_rate":     round(len(wins) / len(reports), 3) if reports else 0,
                "total_pnl":    round(total_pnl, 2),
                "avg_win":      round(sum(r["realizedPnl"] for r in wins)   / len(wins),   2) if wins   else 0,
                "avg_loss":     round(sum(r["realizedPnl"] for r in losses) / len(losses), 2) if losses else 0,
                "by_reason":    by_reason,
            },
        }


orchestrator = TradingOrchestrator()
