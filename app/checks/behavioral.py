import json
from pathlib import Path
from typing import Optional

CALIBRATED_PATH = Path("data_pipeline/calibrated_thresholds.json")

with open(CALIBRATED_PATH) as f:
    calibrated = json.load(f)

PRIOR_CLAIM_COUNT_THRESHOLD = calibrated["prior_claim_count_threshold"]
RETURN_CHARGEBACK_THRESHOLD = calibrated["return_chargeback_count_threshold"]
TIGHT_GAP_MS = calibrated["avg_gap_ms_threshold"]

RECENT_CLAIM_COUNT_HIGH = 3


def score_behavioral(features: dict) -> dict:
    reasons: list[str] = []
    score = 0.0

    prior_claim_count = features.get("prior_claim_count") or 0
    recent_claim_count = features.get("recent_claim_count_last_5") or 0
    return_chargeback_count = features.get("return_chargeback_count") or 0
    avg_gap_ms: Optional[float] = features.get("avg_gap_between_claims_ms")

    if prior_claim_count >= PRIOR_CLAIM_COUNT_THRESHOLD:
        score += 0.4
        reasons.append(
            f"{prior_claim_count} prior claims (>= {PRIOR_CLAIM_COUNT_THRESHOLD:.0f}, "
            f"the 95th-percentile value for genuine customers)"
        )

    if return_chargeback_count >= RETURN_CHARGEBACK_THRESHOLD:
        score += 0.4
        reasons.append(
            f"{return_chargeback_count} prior return/chargebacks (>= {RETURN_CHARGEBACK_THRESHOLD:.0f}, "
            f"the 95th-percentile value for genuine customers)"
        )

    if recent_claim_count >= RECENT_CLAIM_COUNT_HIGH:
        score += 0.2
        reasons.append(
            f"{recent_claim_count} claims in last 5 (>= {RECENT_CLAIM_COUNT_HIGH}, uncalibrated)"
        )

    if avg_gap_ms is not None and avg_gap_ms < TIGHT_GAP_MS:
        hours = avg_gap_ms / (60 * 60 * 1000)
        threshold_hours = TIGHT_GAP_MS / (60 * 60 * 1000)
        score += 0.3
        reasons.append(
            f"avg gap between claims is {hours:.1f}h (< {threshold_hours:.1f}h, "
            f"the 5th-percentile value for genuine customers)"
        )

    score = min(score, 1.0)

    if not reasons:
        reasons.append(
            "no prior claim history" if prior_claim_count == 0 else "no threshold triggered"
        )

    return {
        "signal": "behavioral",
        "method": "rule",
        "score": round(score, 3),
        "reasons": reasons,
    }
