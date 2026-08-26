import json
from pathlib import Path

from app.checks.frequency_features import radial_high_frequency_ratio

CALIBRATED_PATH = Path("data_pipeline/authenticity_threshold.json")

with open(CALIBRATED_PATH) as file:
    calibrated = json.load(file)

LOW_FREQ_RATIO_THRESHOLD = calibrated["threshold"]


def score_image_authenticity(image_paths: list[str]) -> dict:
    reasons: list[str] = []

    if not image_paths:
        return {"signal": "image_authenticity", "method": "model", "score": 0.0,
                "reasons": ["no images submitted"]}

    ratios = []
    for image_path in image_paths:
        try:
            ratio = radial_high_frequency_ratio(image_path)
            ratios.append((image_path, ratio))
        except (FileNotFoundError, OSError) as error:
            reasons.append(f"could not analyze {image_path}: {error}")

    if not ratios:
        return {"signal": "image_authenticity", "method": "model", "score": 0.0,
                "reasons": reasons or ["no analyzable images"]}

    worst_path, worst_ratio = min(ratios, key=lambda item: item[1])
    is_suspicious = worst_ratio < LOW_FREQ_RATIO_THRESHOLD
    score = min(LOW_FREQ_RATIO_THRESHOLD / worst_ratio, 1.0) if worst_ratio > 0 else 1.0

    if is_suspicious:
        reasons.append(
            f"high-frequency energy ratio {worst_ratio:.3f} is BELOW calibrated threshold "
            f"{LOW_FREQ_RATIO_THRESHOLD:.3f} (image: {worst_path}) — unusually smooth for a real photo"
        )
    else:
        reasons.append(
            f"high-frequency energy ratio {worst_ratio:.3f}, within normal range "
            f"(threshold {LOW_FREQ_RATIO_THRESHOLD:.3f})"
        )

    return {"signal": "image_authenticity", "method": "model", "score": round(score, 3), "reasons": reasons}