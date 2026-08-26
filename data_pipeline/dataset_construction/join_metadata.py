import json
from pathlib import Path


MANIFEST = Path("dataset/electronics/manifest.jsonl")
META = Path("dataset/electronics/meta.jsonl")
OUTPUT = Path("dataset/electronics/manifest_with_price.jsonl")
REPORT = Path("dataset/electronics/price_join_report.json")


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: bad JSON in {path} line {line_number}: {e}")
    return rows


def parse_price(raw_price):
    if raw_price is None:
        return None
    if isinstance(raw_price, (int, float)):
        return float(raw_price)
    if isinstance(raw_price, str):
        cleaned = raw_price.replace("$", "").replace(",", "").strip()
        if cleaned.lower() in ("", "none", "n/a", "null"):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def build_price_lookup(meta_rows):
    lookup = {}
    for row in meta_rows:
        price = parse_price(row.get("price"))
        if price is None:
            continue

        for key_field in ("asin", "parent_asin"):
            key = row.get(key_field)
            if key:
                lookup.setdefault(key, price)
    return lookup


def main():
    manifest_rows = load_jsonl(MANIFEST)
    meta_rows = load_jsonl(META)

    price_lookup = build_price_lookup(meta_rows)

    matched = 0
    unmatched = 0
    no_price_in_meta = 0

    enriched = []
    for row in manifest_rows:
        asin = row.get("asin")
        parent_asin = row.get("parent_asin")

        price = price_lookup.get(asin)
        if price is None and parent_asin:
            price = price_lookup.get(parent_asin)

        if price is not None:
            matched += 1
        else:
            unmatched += 1

        row = dict(row)
        row["price"] = price
        enriched.append(row)

    meta_asins_with_price = len(price_lookup)
    meta_asins_total = len({
        r.get("asin") or r.get("parent_asin")
        for r in meta_rows
        if r.get("asin") or r.get("parent_asin")
    })
    no_price_in_meta = meta_asins_total - meta_asins_with_price

    report = {
        "manifest_claims": len(manifest_rows),
        "matched_with_price": matched,
        "unmatched_no_price": unmatched,
        "match_fraction": matched / len(manifest_rows) if manifest_rows else 0,
        "meta_records_loaded": len(meta_rows),
        "meta_asins_with_price": meta_asins_with_price,
        "meta_asins_missing_price_field": no_price_in_meta,
        "note": (
            "unmatched_no_price claims keep price = None rather than being "
            "dropped or defaulted to 0 — feature_builder.py already reports "
            "these explicitly via feature_sources['avg_prior_transaction_amount'] "
            "= 'missing', so the gap stays visible end to end."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for row in enriched:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("PRICE JOIN")
    print("=" * 72)
    print(f"Manifest claims          : {report['manifest_claims']:,}")
    print(f"Matched with price       : {report['matched_with_price']:,} "
          f"({report['match_fraction']:.1%})")
    print(f"Unmatched (price=None)   : {report['unmatched_no_price']:,}")
    print(f"Meta ASINs w/ price      : {report['meta_asins_with_price']:,}")
    print()
    print(f"Enriched manifest -> {OUTPUT}")
    print(f"Report            -> {REPORT}")
    print("=" * 72)


if __name__ == "__main__":
    main()