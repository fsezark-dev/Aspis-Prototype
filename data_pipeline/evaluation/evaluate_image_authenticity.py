import json
from pathlib import Path

from app.checks.image_authenticity import score_image_authenticity

MANIFEST_PATH = Path("dataset/authenticity_test/manifest.jsonl")
FLAG_SCORE = 1.0


def main():
    with open(MANIFEST_PATH) as file:
        rows = [json.loads(line) for line in file]

    results = []

    for row in rows:
        result = score_image_authenticity([row["path"]])
        results.append((row["label"], result["score"]))

    real_scores = [
        score
        for label, score in results
        if label == 0
    ]

    fake_scores = [
        score
        for label, score in results
        if label == 1
    ]

    print(
        f"real images (n={len(real_scores)}): "
        f"mean {sum(real_scores) / len(real_scores):.3f}, "
        f"min {min(real_scores):.3f}, "
        f"max {max(real_scores):.3f}"
    )

    print(
        f"fake images (n={len(fake_scores)}): "
        f"mean {sum(fake_scores) / len(fake_scores):.3f}, "
        f"min {min(fake_scores):.3f}, "
        f"max {max(fake_scores):.3f}"
    )

    true_positives = sum(
        1
        for label, score in results
        if label == 1 and score >= FLAG_SCORE
    )

    false_positives = sum(
        1
        for label, score in results
        if label == 0 and score >= FLAG_SCORE
    )

    false_negatives = sum(
        1
        for label, score in results
        if label == 1 and score < FLAG_SCORE
    )

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
        f"precision={precision:.1%} "
        f"recall={recall:.1%} "
        f"(tp={true_positives}, "
        f"fp={false_positives}, "
        f"fn={false_negatives})"
    )


if __name__ == "__main__":
    main()