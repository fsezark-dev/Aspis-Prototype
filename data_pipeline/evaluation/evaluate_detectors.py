import json
from pathlib import Path

from app.checks.behavioral import score_behavioral
from app.checks.duplicate_evidence import (
    build_index_from_manifest,
    score_duplicate_evidence,
)

MANIFEST_PATH = Path("dataset/electronics/manifest_with_price_and_fraud.jsonl")
FEATURES_PATH = Path("dataset/electronics/behavioral_features_with_fraud.jsonl")

BEHAVIORAL_FLAG_THRESHOLD = 0.5


def load_jsonl(path):
    with open(path) as file:
        return [json.loads(line) for line in file]


def main():
    manifest_rows = load_jsonl(MANIFEST_PATH)
    features_rows = load_jsonl(FEATURES_PATH)

    features_by_claim = {
        row["claim_id"]: row
        for row in features_rows
    }

    images_by_claim = {
        row["claim_id"]: [
            image["path"]
            for image in row.get("images", [])
        ]
        for row in manifest_rows
    }

    label_by_claim = {
        row["claim_id"]: row.get("label", 0)
        for row in manifest_rows
    }

    fraud_type_by_claim = {
        row["claim_id"]: row.get("fraud_type")
        for row in manifest_rows
    }

    build_index_from_manifest(MANIFEST_PATH)

    behavioral_flags = {}
    duplicate_flags = {}

    for row in manifest_rows:
        claim_id = row["claim_id"]
        features = features_by_claim.get(claim_id)

        if features is None:
            continue

        behavioral_result = score_behavioral(features)

        behavioral_flags[claim_id] = (
            behavioral_result["score"] >= BEHAVIORAL_FLAG_THRESHOLD
        )

        image_paths = images_by_claim.get(claim_id, [])
        duplicate_result = score_duplicate_evidence(
            claim_id,
            image_paths,
        )

        duplicate_flags[claim_id] = duplicate_result["score"] >= 0.5

    fraud_types = [
        "duplicate_evidence_ring",
        "text_image_mismatch",
        "behavioral_cluster",
    ]

    print("=" * 72)
    print(
        "PER-FRAUD-TYPE RECALL "
        "(does each signal fire on the type it targets?)"
    )
    print("=" * 72)

    for fraud_type in fraud_types:
        claim_ids = [
            claim_id
            for claim_id, claim_type in fraud_type_by_claim.items()
            if claim_type == fraud_type
        ]

        claim_count = len(claim_ids)

        if claim_count == 0:
            continue

        behavioral_caught = sum(
            1
            for claim_id in claim_ids
            if behavioral_flags.get(claim_id)
        )

        duplicate_caught = sum(
            1
            for claim_id in claim_ids
            if duplicate_flags.get(claim_id)
        )

        print(f"{fraud_type} (n={claim_count}):")
        print(
            f"  behavioral recall        : "
            f"{behavioral_caught}/{claim_count} "
            f"({behavioral_caught / claim_count:.1%})"
        )
        print(
            f"  duplicate_evidence recall: "
            f"{duplicate_caught}/{claim_count} "
            f"({duplicate_caught / claim_count:.1%})"
        )

    print()

    genuine_ids = [
        claim_id
        for claim_id, label in label_by_claim.items()
        if label == 0
    ]

    genuine_count = len(genuine_ids)

    behavioral_false_positives = sum(
        1
        for claim_id in genuine_ids
        if behavioral_flags.get(claim_id)
    )

    duplicate_false_positives = sum(
        1
        for claim_id in genuine_ids
        if duplicate_flags.get(claim_id)
    )

    print("=" * 72)
    print(
        f"FALSE POSITIVE RATE ON GENUINE CLAIMS "
        f"(n={genuine_count})"
    )
    print("=" * 72)
    print(
        f"behavioral false positive rate        : "
        f"{behavioral_false_positives}/{genuine_count} "
        f"({behavioral_false_positives / genuine_count:.2%})"
    )
    print(
        f"duplicate_evidence false positive rate: "
        f"{duplicate_false_positives}/{genuine_count} "
        f"({duplicate_false_positives / genuine_count:.2%})"
    )
    print()

    fraud_ids = [
        claim_id
        for claim_id, label in label_by_claim.items()
        if label == 1
    ]

    fraud_count = len(fraud_ids)

    def precision_recall(flag_dict, name):
        true_positives = sum(
            1
            for claim_id in fraud_ids
            if flag_dict.get(claim_id)
        )

        false_positives = sum(
            1
            for claim_id in genuine_ids
            if flag_dict.get(claim_id)
        )

        false_negatives = fraud_count - true_positives

        precision = (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else float("nan")
        )

        recall = (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else float("nan")
        )

        print(
            f"{name}: "
            f"precision={precision:.1%}  "
            f"recall={recall:.1%}  "
            f"(tp={true_positives}, "
            f"fp={false_positives}, "
            f"fn={false_negatives})"
        )

    print("=" * 72)
    print(
        "OVERALL PRECISION/RECALL "
        "(all fraud types combined, single signal alone)"
    )
    print("=" * 72)

    precision_recall(behavioral_flags, "behavioral")
    precision_recall(duplicate_flags, "duplicate_evidence")


if __name__ == "__main__":
    main()
