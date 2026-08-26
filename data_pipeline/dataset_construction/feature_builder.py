import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_PATH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("dataset/electronics/manifest_with_price.jsonl")
)

OUTPUT_PATH = Path("dataset/electronics/behavioral_features.jsonl")
SUMMARY_OUTPUT_PATH = Path(
    "dataset/electronics/behavioral_features_summary.json"
)

RECENT_WINDOW = 5
RANDOM_SEED = 42

REGIONS = [
    "New York",
    "London",
    "Tokyo",
    "Singapore",
    "Dubai",
    "Toronto",
    "Sydney",
    "Paris",
    "Berlin",
    "Amsterdam",
]

DEVICE_TYPES = [
    "android_phone",
    "ios_phone",
    "desktop_web",
    "mobile_web",
]


def load_manifest(path):
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found:\n{path}")

    records = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(
                    f"WARNING: invalid JSON on line "
                    f"{line_number}: {error}"
                )

    return records


def assign_synthetic_identity(user_ids):
    random_generator = random.Random(RANDOM_SEED)
    identities = {}

    for user_id in sorted(user_ids):
        region = random_generator.choice(REGIONS)
        device_type = random_generator.choice(DEVICE_TYPES)
        device_id = f"dev_{random_generator.randrange(10**8, 10**9)}"

        identities[user_id] = {
            "region": region,
            "device_type": device_type,
            "device_id": device_id,
        }

    return identities


def build_user_histories(records):
    records_by_user = defaultdict(list)

    for record in records:
        user_id = record.get("user_id")

        if user_id:
            records_by_user[user_id].append(record)

    for user_id in records_by_user:
        records_by_user[user_id].sort(
            key=lambda record: record.get("timestamp") or 0
        )

    return records_by_user


def get_price(record):
    return record.get("price")


def compute_features(records):
    records_by_user = build_user_histories(records)
    identities = assign_synthetic_identity(records_by_user.keys())

    feature_rows = []

    for user_id, user_history in records_by_user.items():
        identity = identities[user_id]

        region = identity["region"]
        device_type = identity["device_type"]
        device_id = identity["device_id"]

        for index, record in enumerate(user_history):
            previous_claims = user_history[:index]

            prior_claim_count = len(previous_claims)

            previous_categories = []

            for previous_claim in previous_claims:
                previous_categories.extend(
                    previous_claim.get(
                        "evidence",
                        {},
                    ).get(
                        "categories",
                        [],
                    )
                )

            return_chargeback_count = len(previous_categories)

            timestamp = record.get("timestamp")
            average_gap_ms = None

            if len(previous_claims) >= 1 and timestamp is not None:
                gaps = []

                for first, second in zip(
                    previous_claims,
                    previous_claims[1:] + [record],
                ):
                    first_timestamp = first.get("timestamp")
                    second_timestamp = second.get("timestamp")

                    if (
                        first_timestamp is not None
                        and second_timestamp is not None
                    ):
                        gaps.append(
                            second_timestamp - first_timestamp
                        )

                if gaps:
                    average_gap_ms = sum(gaps) / len(gaps)

            recent_claims = previous_claims[-RECENT_WINDOW:]
            recent_claim_count = len(recent_claims)

            previous_prices = [
                get_price(previous_claim)
                for previous_claim in previous_claims
                if get_price(previous_claim) is not None
            ]

            average_prior_amount = (
                sum(previous_prices) / len(previous_prices)
                if previous_prices
                else None
            )

            previous_ratings = [
                previous_claim.get("rating")
                for previous_claim in previous_claims
                if previous_claim.get("rating") is not None
            ]

            average_prior_rating = (
                sum(previous_ratings) / len(previous_ratings)
                if previous_ratings
                else None
            )

            feature_rows.append(
                {
                    "claim_id": record.get("claim_id"),
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "timestamp_iso": (
                        datetime.fromtimestamp(
                            timestamp / 1000,
                            tz=timezone.utc,
                        ).isoformat()
                        if isinstance(timestamp, (int, float))
                        else None
                    ),
                    "prior_claim_count": prior_claim_count,
                    f"recent_claim_count_last_{RECENT_WINDOW}": (
                        recent_claim_count
                    ),
                    "return_chargeback_count": (
                        return_chargeback_count
                    ),
                    "avg_gap_between_claims_ms": average_gap_ms,
                    "avg_prior_transaction_amount": (
                        average_prior_amount
                    ),
                    "avg_prior_rating": average_prior_rating,
                    "verified_purchase": record.get(
                        "verified_purchase"
                    ),
                    "evidence_categories": record.get(
                        "evidence",
                        {},
                    ).get(
                        "categories",
                        [],
                    ),
                    "region": region,
                    "device_type": device_type,
                    "device_id": device_id,
                    "feature_sources": {
                        "prior_claim_count": "real",
                        "recent_claim_count": "real",
                        "return_chargeback_count": "real",
                        "avg_gap_between_claims_ms": "real",
                        "avg_prior_transaction_amount": (
                            "real" if previous_prices else "missing"
                        ),
                        "avg_prior_rating": (
                            "real" if previous_ratings else "missing"
                        ),
                        "region": "synthetic",
                        "device_type": "synthetic",
                        "device_id": "synthetic",
                    },
                }
            )

    return feature_rows


def summarize(feature_rows):
    prior_claim_counts = [
        row["prior_claim_count"]
        for row in feature_rows
    ]

    claims_with_prior_history = sum(
        1
        for claim_count in prior_claim_counts
        if claim_count > 0
    )

    missing_amount = sum(
        1
        for row in feature_rows
        if row["feature_sources"][
            "avg_prior_transaction_amount"
        ] == "missing"
    )

    missing_rating = sum(
        1
        for row in feature_rows
        if row["feature_sources"][
            "avg_prior_rating"
        ] == "missing"
    )

    return {
        "total_claims": len(feature_rows),
        "claims_with_prior_history": claims_with_prior_history,
        "fraction_with_prior_history": (
            claims_with_prior_history / len(feature_rows)
            if feature_rows
            else 0
        ),
        "max_prior_claim_count": (
            max(prior_claim_counts)
            if prior_claim_counts
            else 0
        ),
        "claims_missing_amount_signal": missing_amount,
        "claims_missing_rating_signal": missing_rating,
        "note": (
            "region/device_type/device_id are SYNTHETIC — not present "
            "in source data. Assigned one stable identity per user_id "
            "via a fixed-seed rule. Fraud-ring patterns "
            "(shared/rotating identity) are injected separately, "
            "on top of this clean baseline."
        ),
    }


def main():
    records = load_manifest(MANIFEST_PATH)
    feature_rows = compute_features(records)
    summary = summarize(feature_rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        for row in feature_rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(
        SUMMARY_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 72)
    print("BEHAVIORAL FEATURE BUILD")
    print("=" * 72)
    print(
        f"Claims processed          : "
        f"{summary['total_claims']:,}"
    )
    print(
        f"Claims with prior history : "
        f"{summary['claims_with_prior_history']:,} "
        f"({summary['fraction_with_prior_history']:.1%})"
    )
    print(
        f"Max prior claims (1 user) : "
        f"{summary['max_prior_claim_count']}"
    )
    print(
        f"Missing amount signal     : "
        f"{summary['claims_missing_amount_signal']:,}"
    )
    print(
        f"Missing rating signal     : "
        f"{summary['claims_missing_rating_signal']:,}"
    )
    print()
    print(f"Features written to : {OUTPUT_PATH}")
    print(f"Summary written to  : {SUMMARY_OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()