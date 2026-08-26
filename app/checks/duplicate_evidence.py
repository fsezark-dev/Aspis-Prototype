from pathlib import Path
import imagehash
from PIL import Image

DUPLICATE_DISTANCE_THRESHOLD = 5

hash_index: dict[str, list[imagehash.ImageHash]] = {}


def build_index_from_manifest(manifest_path: Path) -> None:
    import json

    with open(manifest_path) as f:
        for line in f:
            row = json.loads(line)
            claim_id = row["claim_id"]
            hashes = []

            for img in row.get("images", []):
                try:
                    hashes.append(imagehash.phash(Image.open(img["path"])))
                except (FileNotFoundError, OSError):
                    continue

            hash_index[claim_id] = hashes


def score_duplicate_evidence(claim_id: str, image_paths: list[str]) -> dict:
    reasons: list[str] = []
    best_distance = None
    matched_claim_id = None

    current_hashes = []

    for path in image_paths:
        try:
            current_hashes.append(imagehash.phash(Image.open(path)))
        except (FileNotFoundError, OSError):
            reasons.append(f"could not open image: {path}")
            continue

    for other_claim_id, other_hashes in hash_index.items():
        if other_claim_id == claim_id:
            continue

        for h1 in current_hashes:
            for h2 in other_hashes:
                distance = h1 - h2

                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    matched_claim_id = other_claim_id

    is_duplicate = (
        best_distance is not None
        and best_distance <= DUPLICATE_DISTANCE_THRESHOLD
    )

    score = 1.0 if is_duplicate else 0.0

    if is_duplicate:
        reasons.append(
            f"image matches claim {matched_claim_id} "
            f"(Hamming distance {best_distance})"
        )
    elif best_distance is not None:
        reasons.append(
            f"nearest match is claim {matched_claim_id} "
            f"(distance {best_distance}, not flagged)"
        )
    else:
        reasons.append("no comparable images found")

    if current_hashes:
        hash_index[claim_id] = current_hashes

    return {
        "signal": "duplicate_evidence",
        "method": "rule",
        "score": score,
        "reasons": reasons,
    }