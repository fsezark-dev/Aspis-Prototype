import itertools
import json
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path("dataset/authenticity_test")
IMAGES_DIR = OUTPUT_DIR / "images"
MANIFEST_PATH = OUTPUT_DIR / "manifest.jsonl"

N_SAMPLES = 60
GENERATOR_COLUMN = "sdxl_image"


def main():
    IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_dataset(
        "NasrinImp/COCO_AI",
        split="train",
        streaming=True,
    )

    manifest_rows = []

    for index, row in enumerate(
        itertools.islice(dataset, N_SAMPLES)
    ):
        for label, column, tag in [
            (0, "coco_image", "real"),
            (1, GENERATOR_COLUMN, "ai_generated"),
        ]:
            image = row[column]

            filename = f"{index:04d}_{tag}.jpg"
            output_path = IMAGES_DIR / filename

            image.convert("RGB").save(
                output_path,
                "JPEG",
            )

            manifest_rows.append(
                {
                    "id": f"{index:04d}_{tag}",
                    "path": str(output_path),
                    "label": label,
                    "source": tag,
                    "generator": (
                        GENERATOR_COLUMN
                        if tag == "ai_generated"
                        else None
                    ),
                }
            )

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        for row in manifest_rows:
            file.write(
                json.dumps(row)
                + "\n"
            )

    real_count = sum(
        1
        for row in manifest_rows
        if row["label"] == 0
    )

    fake_count = sum(
        1
        for row in manifest_rows
        if row["label"] == 1
    )

    print(
        f"wrote {len(manifest_rows)} images "
        f"({real_count} real, {fake_count} AI-generated) "
        f"to {IMAGES_DIR}"
    )

    print(f"manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
