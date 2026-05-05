"""
Q-Learning agent for trade entry and exit decisions.
"""
import json
import os
import random
from typing import Dict, List, Optional

Q_TABLE_PATH        = os.path.join(os.path.dirname(__file__), "q_table.json")
RL_STATS_PATH       = os.path.join(os.path.dirname(__file__), "rl_stats.json")
CITY_ACCURACY_PATH  = os.path.join(os.path.dirname(__file__), "city_accuracy.json")

ACTIONS = {0: "HOLD", 1: "BUY_SMALL", 2: "BUY_MEDIUM", 3: "BUY_LARGE", 4: "SELL"}
STAKE_MULTIPLIERS = {0: 0.0, 1: 0.25, 2: 0.5, 3: 1.0, 4: 0.0}

ALPHA     = 0.15
GAMMA     = 0.90
EPS_START = 0.20   # was 0.40 — less random since we now have good signals
EPS_MIN   = 0.05
EPS_DECAY = 0.995

# Minimum number of city trades before accuracy data is trusted
MIN_CITY_TRADES = 5

STOP_LOSS_PCT   = float(os.environ.get("STOP_LOSS_PCT", "-40"))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "80"))


def encode_state(ev: float, confidence: float, pnl_pct: float,
                 market_prob: float, city_win_rate: float = 0.5) -> str:
    """
    State = (ev_bin, conf_bin, pnl_bin, mkt_bin, city_acc_bin).
    city_acc_bin: 0=unknown/low(<40%), 1=medium(40-65%), 2=high(>65%)
    """
    ev_b   = 0 if ev < 0 else (1 if ev < 0.05 else (2 if ev < 0.15 else 3))
    conf_b = 0 if confidence < 0.3 else (1 if confidence < 0.65 else 2)
    pnl_b  = 0 if pnl_pct < -20 else (1 if pnl_pct < 0 else (2 if pnl_pct < 20 else 3))
    mkt_b  = 0 if market_prob < 0.2 else (1 if market_prob < 0.5 else (2 if market_prob < 0.8 else 3))
    acc_b  = 0 if city_win_rate < 0.40 else (1 if city_win_rate < 0.65 else 2)
    return f"{ev_b}_{conf_b}_{pnl_b}_{mkt_b}_{acc_b}"


class QLearningAgent:
    def __init__(self):
        self.q_table: Dict[str, List[float]] = {}
        self.city_accuracy: Dict[str, Dict]  = {}
        self.epsilon      = EPS_START
        self.episodes     = 0
        self.wins         = 0
        self.losses       = 0
        self.total_reward = 0.0
        self._load()

    def _default_q(self) -> List[float]:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    def _load(self):
        if os.path.exists(Q_TABLE_PATH):
            try:
                with open(Q_TABLE_PATH) as f:
                    self.q_table = json.load(f)
            except Exception:
                self.q_table = {}
        if os.path.exists(RL_STATS_PATH):
            try:
                with open(RL_STATS_PATH) as f:
                    s = json.load(f)
                    self.epsilon      = s.get("epsilon", EPS_START)
                    self.episodes     = s.get("episodes", 0)
                    self.wins         = s.get("wins", 0)
                    self.losses       = s.get("losses", 0)
                    self.total_reward = s.get("total_reward", 0.0)
            except Exception:
                pass
        if os.path.exists(CITY_ACCURACY_PATH):
            try:
                with open(CITY_ACCURACY_PATH) as f:
                    self.city_accuracy = json.load(f)
            except Exception:
                self.city_accuracy = {}

    def _save(self):
        try:
            with open(Q_TABLE_PATH, "w") as f:
                json.dump(self.q_table, f)
            with open(RL_STATS_PATH, "w") as f:
                json.dump({
                    "epsilon":      self.epsilon,
                    "episodes":     self.episodes,
                    "wins":         self.wins,
                    "losses":       self.losses,
                    "total_reward": self.total_reward,
                }, f)
            with open(CITY_ACCURACY_PATH, "w") as f:
                json.dump(self.city_accuracy, f, indent=2)
        except Exception as e:
            print(f"[RL] Save error: {e}")

    # ── City accuracy ──────────────────────────────────────────────────────────

    def get_city_win_rate(self, city: str) -> Optional[float]:
        """Return historical win rate for city, or None if < MIN_CITY_TRADES data."""
        if not city or city not in self.city_accuracy:
            return None
        d = self.city_accuracy[city]
        if d.get("total", 0) < MIN_CITY_TRADES:
            return None
        return d["wins"] / d["total"]

    def _update_city_accuracy(self, city: str, won: bool):
        if not city:
            return
        if city not in self.city_accuracy:
            self.city_accuracy[city] = {"wins": 0, "total": 0}
        self.city_accuracy[city]["total"] += 1
        if won:
            self.city_accuracy[city]["wins"] += 1

    # ── Entry decision ─────────────────────────────────────────────────────────

    def choose_action(self, ev: float, confidence: float,
                      pnl_pct: float = 0.0, market_prob: float = 0.5,
                      has_position: bool = False,
                      city_win_rate: float = 0.5) -> dict:
        state = encode_state(ev, confidence, pnl_pct, market_prob, city_win_rate)
        if state not in self.q_table:
            self.q_table[state] = self._default_q()

        if random.random() < self.epsilon:
            valid      = [4] if has_position else [0, 1, 2, 3]
            action_idx = random.choice(valid)
        else:
            q = self.q_table[state]
            if has_position:
                action_idx = 4 if q[4] > q[0] else 0
            else:
                action_idx = int(max(range(4), key=lambda i: q[i]))

        return {
            "action":           ACTIONS[action_idx],
            "action_idx":       action_idx,
            "stake_multiplier": STAKE_MULTIPLIERS[action_idx],
            "q_value":          self.q_table[state][action_idx],
            "epsilon":          round(self.epsilon, 4),
            "state":            state,
        }

    # ── Exit recommendation ────────────────────────────────────────────────────

    def get_exit_recommendation(self, current_pnl_pct: float, market_prob: float,
                                 ev: float, confidence: float,
                                 city_win_rate: float = 0.5) -> dict:
        # Hard limits always override RL
        if current_pnl_pct <= STOP_LOSS_PCT:
            return {"action": "SELL", "reason": f"stop_loss_{current_pnl_pct:.1f}pct"}
        if current_pnl_pct >= TAKE_PROFIT_PCT:
            return {"action": "SELL", "reason": f"take_profit_{current_pnl_pct:.1f}pct"}

        state = encode_state(ev, confidence, current_pnl_pct, market_prob, city_win_rate)
        if state not in self.q_table:
            self.q_table[state] = self._default_q()

        q = self.q_table[state]
        if q[4] > q[0] + 0.05:
            return {"action": "SELL", "reason": "rl_sell_signal"}
        return {"action": "HOLD", "reason": f"rl_hold_pnl_{current_pnl_pct:.1f}pct"}

    # ── Learning ───────────────────────────────────────────────────────────────

    def record_trade_outcome(self, entry_price: float, exit_price: float,
                             stake_usd: float, entry_state: dict,
                             action_idx: int, city: str = "") -> dict:
        if entry_price <= 0:
            return {"pnl_usd": 0, "pnl_pct": 0}

        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        pnl_usd = (exit_price - entry_price) * stake_usd / entry_price
        won     = pnl_usd > 0

        # Reward: normalised P&L, clipped to [-1, +1]
        reward = max(-1.0, min(1.0, pnl_pct / 100))

        state = entry_state.get("state", "0_0_1_1_1")
        if state not in self.q_table:
            self.q_table[state] = self._default_q()

        # Q-learning update (terminal state: no next-state value)
        old_q = self.q_table[state][action_idx]
        self.q_table[state][action_idx] = old_q + ALPHA * (reward - old_q)

        self._update_city_accuracy(city, won)

        self.total_reward += reward
        self.episodes += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1

        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)
        self._save()

        return {
            "pnl_usd": round(pnl_usd, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reward":  round(reward, 4),
        }

    def get_stats(self) -> dict:
        total    = self.wins + self.losses
        win_rate = self.wins / total if total > 0 else 0.0
        return {
            "episodes":        self.episodes,
            "wins":            self.wins,
            "losses":          self.losses,
            "win_rate":        round(win_rate, 4),
            "epsilon":         round(self.epsilon, 4),
            "exploration_pct": round(self.epsilon * 100, 1),
            "total_reward":    round(self.total_reward, 3),
            "q_table_states":  len(self.q_table),
            "cities_tracked":  len(self.city_accuracy),
        }

    def get_city_stats(self) -> dict:
        return {
            city: {
                "wins":     d["wins"],
                "total":    d["total"],
                "win_rate": round(d["wins"] / d["total"], 3) if d["total"] > 0 else 0,
            }
            for city, d in sorted(self.city_accuracy.items(),
                                   key=lambda x: -x[1].get("total", 0))
        }


agent = QLearningAgent()
