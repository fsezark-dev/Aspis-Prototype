import json
from pathlib import Path

from app.checks.clip_embedding import cosine_similarity, embed_image

CALIBRATED_PATH = Path("data_pipeline/product_identity_threshold.json")

with open(CALIBRATED_PATH) as file:
    calibrated = json.load(file)

LOW_SIMILARITY_THRESHOLD = calibrated["threshold"]


def score_product_identity(
    claim_image_paths: list[str],
    listing_image_path: str | None,
) -> dict:
    if listing_image_path is None:
        return {
            "signal": "product_identity",
            "method": "model",
            "score": 0.0,
            "reasons": [
                "no listing image available for this ASIN — signal not applicable"
            ],
        }

    if not claim_image_paths:
        return {
            "signal": "product_identity",
            "method": "model",
            "score": 0.0,
            "reasons": ["no claim images submitted"],
        }

    try:
        listing_embedding = embed_image(listing_image_path)
    except (FileNotFoundError, OSError) as error:
        return {
            "signal": "product_identity",
            "method": "model",
            "score": 0.0,
            "reasons": [f"could not load listing image: {error}"],
        }

    similarities = []

    for image_path in claim_image_paths:
        try:
            claim_embedding = embed_image(image_path)
            similarity = cosine_similarity(claim_embedding, listing_embedding)
            similarities.append((image_path, similarity))
        except (FileNotFoundError, OSError):
            continue

    if not similarities:
        return {
            "signal": "product_identity",
            "method": "model",
            "score": 0.0,
            "reasons": ["no analyzable claim images"],
        }

    worst_path, worst_similarity = min(
        similarities,
        key=lambda item: item[1],
    )

    is_suspicious = worst_similarity < LOW_SIMILARITY_THRESHOLD

    score = (
        max(
            0.0,
            min((LOW_SIMILARITY_THRESHOLD - worst_similarity) + 0.5, 1.0),
        )
        if is_suspicious
        else 0.0
    )

    comparison = "below" if is_suspicious else "within normal range vs."

    reason = (
        f"claim image similarity to listing photo is {worst_similarity:.3f}, "
        f"{comparison} calibrated threshold "
        f"{LOW_SIMILARITY_THRESHOLD:.3f} (image: {worst_path})"
    )

    return {
        "signal": "product_identity",
        "method": "model",
        "score": round(score, 3),
        "reasons": [reason],
    }