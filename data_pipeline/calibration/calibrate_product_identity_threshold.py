import json
from pathlib import Path

from app.checks.clip_embedding import cosine_similarity, embed_image

MANIFEST_PATH = Path("dataset/electronics/manifest_with_price_and_fraud.jsonl")
LOOKUP_PATH = Path("dataset/electronics/listing_image_lookup.json")
OUTPUT_PATH = Path("data_pipeline/product_identity_threshold.json")

PERCENTILE = 0.05


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
    with open(LOOKUP_PATH) as file:
        lookup = json.load(file)

    with open(MANIFEST_PATH) as file:
        rows = [json.loads(line) for line in file]

    similarities = []
    skipped_no_listing = 0
    skipped_no_images = 0

    for row in rows:
        if row.get("label") != 0:
            continue

        listing_path = lookup.get(row.get("asin"))

        if listing_path is None:
            skipped_no_listing += 1
            continue

        claim_images = row.get("images", [])

        if not claim_images:
            skipped_no_images += 1
            continue

        try:
            listing_embedding = embed_image(listing_path)
            claim_embedding = embed_image(claim_images[0]["path"])
            similarity = cosine_similarity(claim_embedding, listing_embedding)
            similarities.append(similarity)
        except (FileNotFoundError, OSError):
            continue

    print(
        f"genuine claims with a usable listing+claim image pair: "
        f"{len(similarities)}"
    )
    print(f"skipped (no matched listing image): {skipped_no_listing}")
    print(f"skipped (no claim images): {skipped_no_images}")

    if len(similarities) < 10:
        print(
            "WARNING: fewer than 10 samples — this threshold will be noisy. "
            "Consider raising MAX_ASINS in fetch_product_images.py and re-running."
        )

    threshold = percentile(similarities, PERCENTILE)

    with open(OUTPUT_PATH, "w") as file:
        json.dump(
            {
                "threshold": threshold,
                "percentile_used": PERCENTILE,
                "n_genuine_pairs": len(similarities),
                "min_similarity": min(similarities),
                "max_similarity": max(similarities),
                "note": (
                    "Calibrated from genuine (label=0) claims with "
                    "a matched listing image only."
                ),
            },
            file,
            indent=2,
        )

    print(
        f"threshold (p{int(PERCENTILE * 100)}): "
        f"{threshold:.4f}"
    )
    print(
        f"range across genuine pairs: "
        f"{min(similarities):.4f} to {max(similarities):.4f}"
    )
    print(f"written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()