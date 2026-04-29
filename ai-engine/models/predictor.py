class ModelPredictor:
    """Blends external GPT probability with model-level adjustments."""

    def predict_from_external(self, probability: float, confidence: float) -> dict:
        prob = max(0.02, min(0.98, float(probability)))
        conf = max(0.01, min(0.99, float(confidence)))
        return {"probability": prob, "confidence": conf}
