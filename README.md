# Aspis

**A risk-scoring API for e-commerce return and chargeback fraud.**

Aspis scores a submitted return/chargeback claim across four independent
signals and returns a plain-English reason for every score, no
black-box number, no hidden logic. Each signal is explicitly labeled as
a transparent rule or a calibrated model, and every threshold is derived
from real data rather than hardcoded.

```json
{
  "claim_id": "AMZ_ELECTRONICS_00001",
  "signals": [
    { "signal": "behavioral", "method": "rule", "score": 0.0,
      "reasons": ["no prior claim history"] },
    { "signal": "duplicate_evidence", "method": "rule", "score": 0.0,
      "reasons": ["nearest match is claim AMZ_ELECTRONICS_00745 (distance 16, not flagged)"] },
    { "signal": "image_authenticity", "method": "model", "score": 0.592,
      "reasons": ["high-frequency energy ratio 0.428, within normal range (threshold 0.254)"] },
    { "signal": "product_identity", "method": "model", "score": 0.0,
      "reasons": ["claim image similarity to listing photo is 0.699, within normal range vs. threshold 0.449"] }
  ]
}
```

---

## Table of contents

- [Overview](#overview)
- [Signals](#signals)
- [Architecture](#architecture)
- [Results](#results)
- [Getting started](#getting-started)
- [Data pipeline](#data-pipeline)
- [Testing](#testing)
- [Limitations & roadmap](#limitations--roadmap)

---

## Overview

A merchant's return/chargeback claim typically comes with a stated
reason ("item arrived damaged") and one or more photos. Aspis evaluates
that claim across four detectors, each targeting a different fraud
pattern:

| Signal | Detects | Method |
|---|---|---|
| `behavioral` | Abnormal claim velocity or return/chargeback history | Rule-based, percentile-calibrated |
| `duplicate_evidence` | Reused "proof" photos across claims (fraud rings) | Rule-based, perceptual hashing |
| `image_authenticity` | AI-generated fake evidence photos | Frequency-domain model, calibrated |
| `product_identity` | Wrong-item claims (photo ≠ purchased product) | CLIP embedding similarity, calibrated |

Results are returned as four independent scores, not a single combined
number — see [Limitations & roadmap](#limitations--roadmap) for why.

Built on top of the [McAuley-Lab/Amazon-Reviews-2023](https://amazon-reviews-2023.github.io/) dataset.
(Electronics category), with real reviews reshaped into synthetic claims
and labeled fraud cases injected on top for evaluation. Full data
methodology is documented in [`data.md`](DATA.md).

---

## Signals

### `behavioral` : rule-based

Flags claims with unusually high prior claim counts, return/chargeback
history, claim velocity, or unusually tight timing between claims.
Thresholds are calibrated at the 95th percentile (5th, for time-gap)
of real genuine-customer behavior, not guessed constants.

### `duplicate_evidence` : rule-based

Uses perceptual image hashing (pHash) to detect when a claim's photo
matches or nearly matches one already used elsewhere, a signature of
coordinated fraud rings reusing evidence. Deterministic and hard to
game with minor edits.

### `image_authenticity` : model-based

Analyzes an image's frequency spectrum to detect AI-generated fakes.
Real camera photos carry more high-frequency energy (sensor noise,
compression artifacts) than typical generator output; the threshold is
calibrated against a real-vs-AI-generated benchmark set.

### `product_identity` : model-based

Compares a claim's photo against the purchased product's listing photo
using CLIP embeddings. Low similarity suggests a "wrong item" claim.
**Known limitation:** base CLIP does not reliably separate two different
products photographed in a similar studio style, this signal has a
measured recall ceiling as a result (see
[Limitations & roadmap](#limitations--roadmap)).

---

## Architecture

```
aspis/
├── app/
│   ├── main.py
│   └── checks/
│       ├── behavioral.py
│       ├── duplicate_evidence.py
│       ├── image_authenticity.py
│       ├── frequency_features.py
│       ├── product_identity.py
│       └── clip_embedding.py
├── data_pipeline/
│   ├── dataset_construction/
│   │   ├── fetch_meta.py
│   │   ├── join_metadata.py
│   │   ├── feature_builder.py
│   │   └── fraud_injection.py
│   ├── calibration/
│   │   ├── calibrate_thresholds.py
│   │   ├── calibrate_authenticity_threshold.py
│   │   ├── calibrate_product_identity_threshold.py
│   │   ├── authenticity_threshold.json
│   │   ├── calibrated_thresholds.json
│   │   └──  product_identity_threshold.json
│   ├── testset_acquisition/
│   │   ├── fetch_authenticity_test_set.py
│   │   └── fetch_product_images.py
│   └── evaluation/
│       ├── evaluate_detectors.py
│       ├── evaluate_image_authenticity.py
│       └── evaluate_product_identity.py
├── dataset/
│   ├── authenticity_test/
│   │   ├── authenticity_images
│   │   └── manifest.jsonl
│   └── electronics/
│       ├── images/
│       ├── listing_images/
│       ├── behavioral_features_genuine.jsonl
│       ├── behavioral_features_summary.json
│       ├── behavioral_features_with_fraud.jsonl
│       ├── listing_image_lookup.json
│       ├── manifest_with_price_and_fraud.jsonl
│       ├── manifest_with_price.jsonl
│       ├── manifest.jsonl
│       └── price_join_report.json
├── tests/
│   └── test_checks.py
└── docs/
    └── DATA.md
```

Each `model`- based signal's core logic (`frequency_features.py`,
`clip_embedding.py`) is split from its threshold-loading wrapper, so the
underlying math has no dependency on a calibration file, the
calibration scripts can import the same logic without a circular
dependency.

---

## Results

| Signal | Precision | Recall | Test set |
|---|---|---|---|
| `behavioral` | 30.5% | 27.3% | 66 synthetic fraud claims / 785 genuine |
| `duplicate_evidence` | 71.4% | 22.7% | 66 synthetic fraud claims / 785 genuine |
| `image_authenticity` | 92.2% | 78.3% | 60 real / 60 AI-generated images |
| `product_identity` | — | not yet reliably measurable (n=5) | 9.1% false-positive rate on 430 genuine claims |

Fraud used for evaluation of `behavioral` and `duplicate_evidence` is
synthetically injected (see [`data.md`](data.md)), not drawn
from real-world fraud cases, these numbers measure whether each
detector's underlying mechanism works, not real-world accuracy.

---

## Getting started

```bash
git clone <repo-url> && cd aspis
pip install fastapi uvicorn pydantic imagehash Pillow numpy torch transformers requests --break-system-packages

uvicorn app.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/score-claim \
  -H "Content-Type: application/json" \
  -d '{"claim_id": "AMZ_ELECTRONICS_00001"}'
```

Interactive API docs: `http://127.0.0.1:8000/docs`

---

## Data pipeline

To rebuild the dataset and calibrated thresholds from scratch:

```bash
python3 data_pipeline/fetch_meta.py
python3 data_pipeline/join_metadata.py
python3 data_pipeline/fraud_injection.py

python3 data_pipeline/feature_builder.py dataset/electronics/manifest_with_price.jsonl
mv dataset/electronics/behavioral_features.jsonl dataset/electronics/behavioral_features_genuine.jsonl
python3 data_pipeline/feature_builder.py dataset/electronics/manifest_with_price_and_fraud.jsonl
mv dataset/electronics/behavioral_features.jsonl dataset/electronics/behavioral_features_with_fraud.jsonl

python3 data_pipeline/calibrate_thresholds.py
python3 data_pipeline/fetch_authenticity_test_set.py
PYTHONPATH=. python3 data_pipeline/calibrate_authenticity_threshold.py
python3 data_pipeline/fetch_product_images.py
PYTHONPATH=. python3 data_pipeline/calibrate_product_identity_threshold.py
```

Full details on dataset sourcing, the reshaping of reviews into claims,
and each signal's exact image-processing method are in
[`data.md`](data.md).

---

## Testing

```bash
PYTHONPATH=. pytest tests/test_checks.py -v
```

13 unit tests across all four signals, mostly self-contained with
synthetic fixtures (no dataset download required). `product_identity`'s
tests require the downloaded listing-image set, since CLIP is a semantic
model and doesn't respond meaningfully to synthetic test patterns, run
`fetch_product_images.py` first if starting from a fresh clone.

---

## Limitations & roadmap

- **No combined risk score.** Four signals return four separate scores
  rather than one, a weighted combiner was deliberately not built
  without validated weights to back it.
- **Evaluation fraud is synthetic**, not drawn from real cases.
- **`/score-claim` currently scores claims from a pre-loaded dataset**,
  looked up by ID — not yet arbitrary new submissions. `duplicate_evidence`
  and `image_authenticity` would generalize to a new photo directly;
  `product_identity` would need a live listing-image fetch; `behavioral`
  already handles a first-time claimant correctly (no history → score 0).
- No authentication, no persistence beyond flat files, single process,
  a prototype, not a production system.
