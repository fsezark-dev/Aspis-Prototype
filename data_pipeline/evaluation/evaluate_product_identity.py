import json
from pathlib import Path

from app.checks.product_identity import score_product_identity

MANIFEST_PATH = Path("dataset/electronics/manifest_with_price_and_fraud.jsonl")
LOOKUP_PATH = Path("dataset/electronics/listing_image_lookup.json")


def main():
    with open(LOOKUP_PATH) as file:
        lookup = json.load(file)

    with open(MANIFEST_PATH) as file:
        rows = [json.loads(line) for line in file]

    def score_row(row):
        listing_path = lookup.get(row.get("asin"))
        claim_images = [
            image["path"]
            for image in row.get("images", [])
        ]

        result = score_product_identity(
            claim_images,
            listing_path,
        )

        return result["score"], listing_path is not None

    mismatch_rows = [
        row
        for row in rows
        if row.get("fraud_type") == "text_image_mismatch"
    ]

    mismatch_scores = [
        score_row(row)
        for row in mismatch_rows
    ]

    evaluable_mismatch = [
        (score, has_listing)
        for score, has_listing in mismatch_scores
        if has_listing
    ]

    print(
        f"text_image_mismatch claims total: "
        f"{len(mismatch_rows)}"
    )

    print(
        f"  with a matched listing image (evaluable): "
        f"{len(evaluable_mismatch)}"
    )

    if evaluable_mismatch:
        flagged = sum(
            1
            for score, _ in evaluable_mismatch
            if score > 0.0
        )

        print(
            f"  flagged (score > 0): "
            f"{flagged}/{len(evaluable_mismatch)} "
            f"({flagged / len(evaluable_mismatch):.1%})"
        )

    genuine_rows = [
        row
        for row in rows
        if row.get("label") == 0
    ]

    genuine_scores = [
        score_row(row)
        for row in genuine_rows
    ]

    evaluable_genuine = [
        (score, has_listing)
        for score, has_listing in genuine_scores
        if has_listing
    ]

    print(f"\ngenuine claims total: {len(genuine_rows)}")

    print(
        f"  with a matched listing image (evaluable): "
        f"{len(evaluable_genuine)}"
    )

    if evaluable_genuine:
        false_positives = sum(
            1
            for score, _ in evaluable_genuine
            if score > 0.0
        )

        print(
            f"  false positives (score > 0): "
            f"{false_positives}/{len(evaluable_genuine)} "
            f"({false_positives / len(evaluable_genuine):.1%})"
        )

    if not evaluable_mismatch:
        print(
            "\nWARNING: 0 text_image_mismatch claims have a matched "
            "listing image with only 100 listing images downloaded so far. "
            "Recall can't be measured at all until MAX_ASINS is raised "
            "in fetch_product_images.py and it's re-run."
        )


if __name__ == "__main__":
    main()