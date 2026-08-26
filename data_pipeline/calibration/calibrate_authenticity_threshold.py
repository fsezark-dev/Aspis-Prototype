import json
from pathlib import Path

from app.checks.frequency_features import radial_high_frequency_ratio

MANIFEST_PATH = Path("dataset/authenticity_test/manifest.jsonl")
OUTPUT_PATH = Path("data_pipeline/authenticity_threshold.json")
PERCENTILE = 0.05 # bottom 5% of REAL photos' ratio


def percentile(values, percentile_value):
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


def main():
    with open(MANIFEST_PATH) as file:
        rows = [json.loads(line) for line in file]

    real_ratios = [
        radial_high_frequency_ratio(row["path"])
        for row in rows
        if row["label"] == 0
    ]

    threshold = percentile(real_ratios, PERCENTILE)

    with open(OUTPUT_PATH, "w") as file:
        json.dump(
            {
                "threshold": threshold,
                "percentile_used": PERCENTILE,
                "n_real_images": len(real_ratios),
                "computed_from": (
                    "COCO_AI real subset (proxy dataset, "
                    "not domain-matched to product photos)"
                ),
            },
            file,
            indent=2,
        )

    print(
        f"threshold (p{int(PERCENTILE * 100)} of real): "
        f"{threshold:.4f}"
    )
    print(f"written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()