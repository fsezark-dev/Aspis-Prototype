import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.checks.behavioral import score_behavioral
from app.checks.duplicate_evidence import (
    build_index_from_manifest,
    score_duplicate_evidence,
)
from app.checks.image_authenticity import score_image_authenticity
from app.checks.product_identity import score_product_identity

FEATURES_PATH = Path("dataset/electronics/behavioral_features_with_fraud.jsonl")
MANIFEST_PATH = Path("dataset/electronics/manifest_with_price_and_fraud.jsonl")
LISTING_LOOKUP_PATH = Path("dataset/electronics/listing_image_lookup.json")

features_by_claim: dict[str, dict] = {}
images_by_claim: dict[str, list[str]] = {}
asin_by_claim: dict[str, str] = {}
listing_image_by_asin: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(FEATURES_PATH) as file:
        for line in file:
            row = json.loads(line)
            features_by_claim[row["claim_id"]] = row

    with open(MANIFEST_PATH) as file:
        for line in file:
            row = json.loads(line)
            images_by_claim[row["claim_id"]] = [
                image["path"] for image in row.get("images", [])
            ]
            asin_by_claim[row["claim_id"]] = row.get("asin")

    with open(LISTING_LOOKUP_PATH) as file:
        listing_image_by_asin.update(json.load(file))

    build_index_from_manifest(MANIFEST_PATH)

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


class ClaimSubmission(BaseModel):
    claim_id: str

@app.post("/score-claim")
def score_claim(claim: ClaimSubmission):
    features = features_by_claim.get(claim.claim_id)

    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown claim_id: {claim.claim_id}",
        )

    image_paths = images_by_claim.get(claim.claim_id, [])
    asin = asin_by_claim.get(claim.claim_id)

    listing_image_path = (
        listing_image_by_asin.get(asin)
        if asin
        else None
    )

    signals = [
        score_behavioral(features),
        score_duplicate_evidence(claim.claim_id, image_paths),
        score_image_authenticity(image_paths),
        score_product_identity(image_paths, listing_image_path),
    ]

    return {
        "claim_id": claim.claim_id,
        "signals": signals,
    }