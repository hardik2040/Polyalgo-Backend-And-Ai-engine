import re


def fetch_external_signals(question: str) -> dict:
    """
    Simple keyword-based sentiment analysis for prediction market questions.
    Returns a sentiment score and confidence without requiring any API key.
    """
    q = question.lower()
    score = 0.0

    positive_keywords = [
        "will", "exceed", "above", "over", "higher", "increase",
        "rise", "grow", "win", "pass", "approve", "yes"
    ]
    negative_keywords = [
        "fail", "below", "under", "lower", "decrease", "fall",
        "drop", "lose", "reject", "no", "not"
    ]

    for word in positive_keywords:
        if word in q:
            score += 0.05

    for word in negative_keywords:
        if word in q:
            score -= 0.05

    score = max(-0.3, min(0.3, score))
    confidence = 0.15 + min(0.15, abs(score))

    return {
        "sentiment_score": round(score, 3),
        "confidence": round(confidence, 3),
        "source": "keyword_sentiment",
    }
