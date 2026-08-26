import json
import time
from pathlib import Path
import requests

META_PATH = Path("dataset/electronics/meta.jsonl")
MANIFEST_PATH = Path("dataset/electronics/manifest_with_price_and_fraud.jsonl")
OUTPUT_DIR = Path("dataset/electronics/listing_images")
LOOKUP_OUTPUT = Path("dataset/electronics/listing_image_lookup.json")

MAX_ASINS = 100  # Can raise upto 780


def get_needed_asins() -> set[str]:
    asins = set()
    with open(MANIFEST_PATH) as f:
        for line in f:
            row = json.loads(line)
            if row.get("asin"):
                asins.add(row["asin"])
    return asins


def build_asin_to_url(needed_asins: set[str]) -> dict[str, str]:
    asin_to_url = {}
    with open(META_PATH) as f:
        for line in f:
            row = json.loads(line)
            asin = row.get("parent_asin")
            if asin not in needed_asins or asin in asin_to_url:
                continue
            large_urls = row.get("images", {}).get("large", [])
            if large_urls:
                asin_to_url[asin] = large_urls[0]
    return asin_to_url


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    needed_asins = get_needed_asins()
    print(f"ASINs referenced in manifest: {len(needed_asins)}")

    asin_to_url = build_asin_to_url(needed_asins)
    print(f"ASINs with a listing image URL found: {len(asin_to_url)}")

    sample_asins = list(asin_to_url.items())[:MAX_ASINS]
    lookup = {}
    failed = []

    for asin, url in sample_asins:
        out_path = OUTPUT_DIR / f"{asin}.jpg"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            lookup[asin] = str(out_path)
        except Exception as e:
            failed.append((asin, str(e)))
        time.sleep(0.1)

    with open(LOOKUP_OUTPUT, "w") as f:
        json.dump(lookup, f, indent=2)

    print(f"downloaded: {len(lookup)}")
    print(f"failed: {len(failed)}")
    if failed[:5]:
        print("sample failures:", failed[:5])
    print(f"lookup written to {LOOKUP_OUTPUT}")


if __name__ == "__main__":
    main()