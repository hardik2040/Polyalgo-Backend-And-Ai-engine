"""
Q-Learning agent for trade entry and exit decisions.
"""
import json
import os
import random
from typing import Dict, List

Q_TABLE_PATH  = os.path.join(os.path.dirname(__file__), "q_table.json")
RL_STATS_PATH = os.path.join(os.path.dirname(__file__), "rl_stats.json")

ACTIONS = {0: "HOLD", 1: "BUY_SMALL", 2: "BUY_MEDIUM", 3: "BUY_LARGE", 4: "SELL"}
STAKE_MULTIPLIERS = {0: 0.0, 1: 0.25, 2: 0.5, 3: 1.0, 4: 0.0}

ALPHA   = 0.15
GAMMA   = 0.90
EPS_START = 0.40
EPS_MIN   = 0.05
EPS_DECAY = 0.995

STOP_LOSS_PCT   = float(os.environ.get("STOP_LOSS_PCT", "-40"))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "80"))


def encode_state(ev: float, confidence: float, pnl_pct: float, market_prob: float) -> str:
    ev_b   = 0 if ev < 0 else (1 if ev < 0.05 else (2 if ev < 0.15 else 3))
    conf_b = 0 if confidence < 0.3 else (1 if confidence < 0.6 else 2)
    pnl_b  = 0 if pnl_pct < -20 else (1 if pnl_pct < 0 else (2 if pnl_pct < 20 else 3))
    mkt_b  = 0 if market_prob < 0.2 else (1 if market_prob < 0.5 else (2 if market_prob < 0.8 else 3))
    return f"{ev_b}_{conf_b}_{pnl_b}_{mkt_b}"


class QLearningAgent:
    def __init__(self):
        self.q_table: Dict[str, List[float]] = {}
        self.epsilon = EPS_START
        self.episodes = 0
        self.wins = 0
        self.losses = 0
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
        except Exception as e:
            print(f"[RL] Save error: {e}")

    def choose_action(self, ev: float, confidence: float,
                      pnl_pct: float = 0.0, market_prob: float = 0.5,
                      has_position: bool = False) -> dict:
        state = encode_state(ev, confidence, pnl_pct, market_prob)
        if state not in self.q_table:
            self.q_table[state] = self._default_q()

        if random.random() < self.epsilon:
            valid = [4] if has_position else [0, 1, 2, 3]
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

    def get_exit_recommendation(self, current_pnl_pct: float, market_prob: float,
                                 ev: float, confidence: float) -> dict:
        # Hard limits always override RL
        if current_pnl_pct <= STOP_LOSS_PCT:
            return {"action": "SELL", "reason": f"stop_loss_{current_pnl_pct:.1f}pct"}
        if current_pnl_pct >= TAKE_PROFIT_PCT:
            return {"action": "SELL", "reason": f"take_profit_{current_pnl_pct:.1f}pct"}

        state = encode_state(ev, confidence, current_pnl_pct, market_prob)
        if state not in self.q_table:
            self.q_table[state] = self._default_q()

        q = self.q_table[state]
        if q[4] > q[0] + 0.05:
            return {"action": "SELL", "reason": "rl_sell_signal"}
        return {"action": "HOLD", "reason": f"rl_hold_pnl_{current_pnl_pct:.1f}pct"}

    def record_trade_outcome(self, entry_price: float, exit_price: float,
                             stake_usd: float, entry_state: dict,
                             action_idx: int) -> dict:
        if entry_price <= 0:
            return {"pnl_usd": 0, "pnl_pct": 0}

        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        pnl_usd = (exit_price - entry_price) * stake_usd / entry_price
        reward  = max(-1.0, min(1.0, pnl_pct / 100))

        state = entry_state.get("state", "0_0_1_1")
        if state not in self.q_table:
            self.q_table[state] = self._default_q()

        old_q = self.q_table[state][action_idx]
        self.q_table[state][action_idx] = old_q + ALPHA * (reward - old_q)

        self.total_reward += reward
        self.episodes += 1
        if pnl_usd > 0:
            self.wins += 1
        else:
            self.losses += 1

        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)
        self._save()

        return {
            "pnl_usd":    round(pnl_usd, 2),
            "pnl_pct":    round(pnl_pct, 2),
            "reward":     round(reward, 4),
        }

    def get_stats(self) -> dict:
        total = self.wins + self.losses
        win_rate = self.wins / total if total > 0 else 0.0
        return {
            "episodes":       self.episodes,
            "wins":           self.wins,
            "losses":         self.losses,
            "win_rate":       round(win_rate, 4),
            "epsilon":        round(self.epsilon, 4),
            "exploration_pct": round(self.epsilon * 100, 1),
            "total_reward":   round(self.total_reward, 3),
            "q_table_states": len(self.q_table),
        }


agent = QLearningAgent()
