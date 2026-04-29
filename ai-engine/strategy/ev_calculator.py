def calculate_ev(true_prob: float, market_prob: float) -> float:
    """Expected value of a prediction market bet."""
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    return true_prob - market_prob


def kelly_criterion(true_prob: float, market_prob: float) -> float:
    """Kelly fraction for optimal bet sizing on a binary market."""
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    b = (1.0 - market_prob) / market_prob
    p = true_prob
    q = 1.0 - true_prob
    kelly = (b * p - q) / b
    return max(0.0, min(0.5, kelly))
