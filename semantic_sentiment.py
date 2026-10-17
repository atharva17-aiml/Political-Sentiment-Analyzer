from functools import lru_cache

from transformers import pipeline


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@lru_cache(maxsize=1)
def load_semantic_model():
    """
    Load the transformer sentiment model only once.
    """

    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
    )


LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",

    "NEGATIVE": "NEGATIVE",
    "NEUTRAL": "NEUTRAL",
    "POSITIVE": "POSITIVE",
}


def semantic_predict(text):
    """
    Transformer-based semantic sentiment prediction.

    Returns:
        {
            "label": "POSITIVE/NEGATIVE/NEUTRAL",
            "confidence": float
        }
    """

    if not text or not text.strip():
        return {
            "label": "NEUTRAL",
            "confidence": 0.0,
        }

    try:
        model = load_semantic_model()

        result = model(
            text,
            truncation=True,
            max_length=512,
        )[0]

        raw_label = str(result["label"]).upper()

        label = LABEL_MAP.get(
            raw_label,
            raw_label,
        )

        confidence = float(result["score"]) * 100

        return {
            "label": label,
            "confidence": confidence,
        }

    except Exception as e:
        print(f"Semantic model error: {e}")

        return {
            "label": "NEUTRAL",
            "confidence": 0.0,
        }