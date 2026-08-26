import json
import random
from pathlib import Path

MANIFEST_PATH = Path("dataset/electronics/manifest_with_price.jsonl")
OUTPUT_PATH = Path("dataset/electronics/manifest_with_price_and_fraud.jsonl")

random.seed(42)  # reproducibility -- same injected set every run

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
DEVICE_TYPES = ["desktop_web", "mobile_web", "android_phone", "ios_phone"]

N_MISMATCH_CASES = 15


def load_real_claims() -> list[dict]:
    with open(MANIFEST_PATH) as f:
        return [json.loads(line) for line in f]


def make_duplicate_ring(real_claims: list[dict], ring_id: int, n_claims: int = 3) -> list[dict]:
    source = random.choice(real_claims)
    if not source.get("images"):
        return []

    shared_region = random.choice(REGIONS)
    shared_device_type = random.choice(DEVICE_TYPES)
    shared_device_id = f"dev_fraud_ring_{ring_id}"
    base_ts = source["timestamp"]

    fake_claims = []
    for i in range(n_claims):
        fake_claim_id = f"FRAUD_DUP_RING_{ring_id:03d}_{i}"
        fake_claims.append({
            "claim_id": fake_claim_id,
            "review_key": f"synthetic|{fake_claim_id}",
            "source": "fraud_injection",
            "fraud_type": "duplicate_evidence_ring",
            "label": 1,
            "category": source["category"],
            "user_id": f"FRAUD_USER_{ring_id:03d}_{i}",
            "asin": source["asin"],
            "parent_asin": source["parent_asin"],
            "timestamp": base_ts + i * random.randint(60_000, 900_000),
            "rating": source["rating"],
            "title": source["title"],
            "text": source["text"],
            "verified_purchase": False,
            "helpful_vote": 0,
            "evidence": source["evidence"],
            "images": source["images"],  # reused ON PURPOSE -- this fraud type IS duplication
            "price": source["price"],
            "region": shared_region,
            "device_type": shared_device_type,
            "device_id": shared_device_id,
        })
    return fake_claims


def make_text_image_mismatch(text_source: dict, image_donor: dict, idx: int) -> dict | None:
    if not image_donor.get("images"):
        return None

    fake_claim_id = f"FRAUD_TEXT_MISMATCH_{idx:03d}"
    return {
        "claim_id": fake_claim_id,
        "review_key": f"synthetic|{fake_claim_id}",
        "source": "fraud_injection",
        "fraud_type": "text_image_mismatch",
        "label": 1,
        "category": text_source["category"],
        "user_id": f"FRAUD_USER_TEXT_{idx:03d}",
        "asin": text_source["asin"],
        "parent_asin": text_source["parent_asin"],
        "timestamp": text_source["timestamp"],
        "rating": text_source["rating"],
        "title": text_source["title"],
        "text": text_source["text"],
        "verified_purchase": False,
        "helpful_vote": 0,
        "evidence": text_source["evidence"],
        "images": image_donor["images"],  # unique in the corpus -- donor removed from baseline
        "price": text_source["price"],
        "region": random.choice(REGIONS),
        "device_type": random.choice(DEVICE_TYPES),
        "device_id": f"dev_fraud_{idx:03d}",
    }


def make_behavioral_fraud_cluster(cluster_id: int, n_claims: int = 6) -> list[dict]:
    shared_user = f"FRAUD_BEHAVIORAL_USER_{cluster_id:03d}"
    shared_region = random.choice(REGIONS)
    shared_device_type = random.choice(DEVICE_TYPES)
    shared_device_id = f"dev_fraud_behavioral_{cluster_id}"
    base_ts = random.randint(1_600_000_000_000, 1_690_000_000_000)

    fake_claims = []
    for i in range(n_claims):
        fake_claim_id = f"FRAUD_BEHAVIORAL_{cluster_id:03d}_{i}"
        fake_claims.append({
            "claim_id": fake_claim_id,
            "review_key": f"synthetic|{fake_claim_id}",
            "source": "fraud_injection",
            "fraud_type": "behavioral_cluster",
            "label": 1,
            "category": "Electronics",
            "user_id": shared_user,
            "asin": f"SYNTHETIC_HIGH_VALUE_ASIN_{cluster_id}",
            "parent_asin": f"SYNTHETIC_HIGH_VALUE_ASIN_{cluster_id}",
            "timestamp": base_ts + i * random.randint(3_600_000, 21_600_000),
            "rating": random.choice([1.0, 2.0]),
            "title": "Item arrived damaged",
            "text": "The item arrived damaged and does not match the listing description.",
            "verified_purchase": True,
            "helpful_vote": 0,
            "evidence": {"score": 3, "categories": ["damaged"], "matched_terms": ["damaged"], "negative_matches": []},
            "images": [],
            "price": None,
            "region": shared_region,
            "device_type": shared_device_type,
            "device_id": shared_device_id,
        })
    return fake_claims


def main():
    real_claims = load_real_claims()
    eligible_donors = [c for c in real_claims if c.get("images")]
    donors = random.sample(eligible_donors, min(N_MISMATCH_CASES, len(eligible_donors)))
    donor_ids = {d["claim_id"] for d in donors}
    real_claims = [c for c in real_claims if c["claim_id"] not in donor_ids]

    injected: list[dict] = []

    for ring_id in range(5):
        injected.extend(make_duplicate_ring(real_claims, ring_id))

    for idx, donor in enumerate(donors):
        text_source = random.choice(real_claims)  # any remaining real claim's text
        claim = make_text_image_mismatch(text_source, donor, idx)
        if claim:
            injected.append(claim)

    for cluster_id in range(6):
        injected.extend(make_behavioral_fraud_cluster(cluster_id))
        
    for claim in real_claims:
        claim["fraud_type"] = None
        claim["label"] = 0

    all_claims = real_claims + injected

    with open(OUTPUT_PATH, "w") as f:
        for claim in all_claims:
            f.write(json.dumps(claim) + "\n")

    print(f"real claims (after removing {len(donors)} mismatch donors): {len(real_claims)}")
    print(f"injected fraud claims: {len(injected)}")
    print(f"  duplicate_evidence_ring: {sum(1 for c in injected if c['fraud_type'] == 'duplicate_evidence_ring')}")
    print(f"  text_image_mismatch: {sum(1 for c in injected if c['fraud_type'] == 'text_image_mismatch')}")
    print(f"  behavioral_cluster: {sum(1 for c in injected if c['fraud_type'] == 'behavioral_cluster')}")
    print(f"total written to {OUTPUT_PATH}: {len(all_claims)}")


if __name__ == "__main__":
    main()