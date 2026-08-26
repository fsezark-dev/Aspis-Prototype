import json
from pathlib import Path

FEATURES_PATH = Path("dataset/electronics/behavioral_features_genuine.jsonl")
OUTPUT_PATH = Path("data_pipeline/calibrated_thresholds.json")

EXPECTED_GENUINE_COUNT = 800
CONTAMINATION_TOLERANCE = 20
PERCENTILE = 0.95


def percentile(values, percentile_value):
    if not values:
        return None

    values = sorted(values)
    index = (len(values) - 1) * percentile_value
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(values) - 1)

    if lower_index == upper_index:
        return values[lower_index]

    return (
        values[lower_index]
        + (values[upper_index] - values[lower_index])
        * (index - lower_index)
    )


def load_features(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def main():
    rows = load_features(FEATURES_PATH)

    if abs(len(rows) - EXPECTED_GENUINE_COUNT) > CONTAMINATION_TOLERANCE:
        raise RuntimeError(
            f"Refusing to calibrate: {FEATURES_PATH} has {len(rows)} claims, "
            f"expected ~{EXPECTED_GENUINE_COUNT} (the clean genuine baseline). "
            f"This file was likely generated from a fraud-injected manifest. "
            f"Re-run feature_builder.py with MANIFEST pointed at "
            f"manifest_with_price.jsonl (not the *_and_fraud.jsonl version) "
            f"before calibrating."
        )

    prior_claim_counts = [
        row["prior_claim_count"]
        for row in rows
    ]

    return_chargeback_counts = [
        row["return_chargeback_count"]
        for row in rows
    ]

    gaps = [
        row["avg_gap_between_claims_ms"]
        for row in rows
        if row.get("avg_gap_between_claims_ms") is not None
    ]

    thresholds = {
        "percentile_used": PERCENTILE,
        "computed_from": str(FEATURES_PATH),
        "n_claims": len(rows),
        "prior_claim_count_threshold": percentile(
            prior_claim_counts,
            PERCENTILE,
        ),
        "return_chargeback_count_threshold": percentile(
            return_chargeback_counts,
            PERCENTILE,
        ),
        "avg_gap_ms_threshold": percentile(
            gaps,
            1 - PERCENTILE,
        ),
        "note": (
            "Thresholds are the value at the stated percentile of each "
            "feature in the GENUINE baseline (pre-fraud-injection). "
            "prior_claim_count and return_chargeback_count use the 95th "
            "percentile (flag the unusually high tail). avg_gap_ms uses "
            "the 5th percentile (flag the unusually LOW tail — claims "
            "clustered too close together in time)."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            thresholds,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print("CALIBRATED THRESHOLDS")
    print("=" * 72)
    print(
        f"n claims (genuine baseline)      : "
        f"{thresholds['n_claims']:,}"
    )
    print(
        f"prior_claim_count (p{int(PERCENTILE * 100)})        : "
        f"{thresholds['prior_claim_count_threshold']}"
    )
    print(
        f"return_chargeback_count (p{int(PERCENTILE * 100)})  : "
        f"{thresholds['return_chargeback_count_threshold']}"
    )
    print(
        f"avg_gap_ms (p{int((1 - PERCENTILE) * 100)})            : "
        f"{thresholds['avg_gap_ms_threshold']}"
    )
    print()
    print(f"Written to {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
