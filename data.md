# Data methodology

This document covers exactly where Aspis's data comes from and how it
was processed, dataset provenance, the reshaping of reviews into
claims, fraud injection, and the precise image-processing method behind
each signal.

## 1. Source dataset

Claims derive from **McAuley-Lab/Amazon-Reviews-2023** (Hugging Face),
Electronics category, `raw_review_Electronics` for reviews and
`raw_meta_Electronics` for product metadata and listing photos.

No public dataset of e-commerce return/chargeback fraud claims exists.
This dataset was repurposed for that use, which is stated explicitly
here rather than implied to be a native fraud dataset.

## 2. Reshaping reviews into claims

Each review became a synthetic claim record in
`dataset/electronics/manifest.jsonl`, with fields: `claim_id`,
`review_key`, `source`, `category`, `user_id`, `asin`, `parent_asin`,
`timestamp` (ms since epoch), `rating`, `title`, `text`,
`verified_purchase`, `helpful_vote`, `evidence` (a `categories` list,
e.g. `damaged`, `missing`, `wrong_item`, derived from keyword-matching
the review text), and `images` (each with a local `path`, `sha256`,
`bytes`, `content_type`, and source `url`).

Images live at `dataset/electronics/images/<claim_id>/<NN>.jpg`.

## 3. Dataset statistics

From a one-time audit (`dataset/electronics/dataset_audit.json`,
committed as a snapshot. The script that produced it has since been
removed):

- 800 reviews, 2,078 images, 2.6 images/review average
- Evidence categories: damaged 70.2%, missing 26.2%, defective 7.9%,
  wrong_item 4.8%, not_as_described 2.5%, packaging_damage 0.4%
- 72.6% verified purchase
- 657 unique users (73 with more than one claim, max 18)
- 789 unique ASINs (4 with more than one claim, max 9)
- 0 duplicate image hashes at baseline
- Timestamps span 2005-02-11 to 2023-03-18

## 4. Price join

`join_metadata.py` joins `price` from `raw_meta_Electronics` by ASIN
(falling back to `parent_asin`). **56.4% match rate (451/800)**, 
unmatched prices are left `null`, not defaulted to zero, since a missing
price reflects a real metadata gap (delisted products, incomplete
records). Output: `manifest_with_price.jsonl`.

## 5. Behavioral features

`feature_builder.py` groups claims by `user_id`, sorts by timestamp, and
computes each claim's features using only that user's *prior* claims, 
never future ones, avoiding lookahead leakage. Computed:
`prior_claim_count`, `recent_claim_count_last_5`,
`return_chargeback_count`, `avg_gap_between_claims_ms`,
`avg_prior_transaction_amount`, `avg_prior_rating`.

Each user also gets one stable synthetic identity (`region`,
`device_type`, `device_id`), generated once via a fixed seed and reused
across that user's claims, explicitly tagged `"synthetic"` in a
`feature_sources` map on every row.

Only 17.9% of claims (143/800) have any prior history, this is expected, since
most reviewers appear once. This sparsity is why fraud injection exists.

`feature_builder.py` takes its manifest path as a CLI argument, since
two different downstream needs (calibration vs. evaluation) require it
run against two different inputs, outputs are renamed immediately to
`behavioral_features_genuine.jsonl` and
`behavioral_features_with_fraud.jsonl` to avoid the two overwriting each
other.

## 6. Fraud injection

`fraud_injection.py` generates three labeled synthetic fraud types on
top of the 800-claim genuine baseline (a fourth type, AI-generated fake
evidence was scoped but not implemented):

- **`duplicate_evidence_ring`** (15 claims, 5 rings of 3): one real
  claim's image reused unchanged across fabricated claim IDs sharing a
  device/region identity, filed minutes apart.
- **`text_image_mismatch`** (15 claims): a real claim's text paired with
  a **donor** claim's image. Donors are reserved and removed from the
  genuine baseline before any other processing, so each donor's image is
  used exactly once in the whole corpus, this prevents
  `duplicate_evidence` from catching these cases by coincidence.
- **`behavioral_cluster`** (36 claims, 6 clusters of 6): no images at
  all, one fabricated identity files 6 claims in quick succession,
  isolating whether `behavioral.py` fires on velocity alone.

All randomness uses a fixed seed (`random.seed(42)`) for reproducible
metrics. Output: `manifest_with_price_and_fraud.jsonl`, 785 genuine
(800 minus 15 removed donors) + 66 injected = 851 claims, each carrying
an explicit `label` (0/1) and `fraud_type`.

## 7. Listing images

`fetch_product_images.py` downloads one photo per unique ASIN from
`raw_meta_Electronics`'s `images` field (which has `hi_res`, `large`,
`thumb`, and `variant` URL lists), the **`large`** variant is used, for
a balance of resolution and download size. Matched via `parent_asin`,
consistent with the price join's fallback field.

**422 of 780 referenced ASINs (54.1%) had a listing image available**, 
a gap nearly identical to the price join's, from the same underlying
metadata-completeness issue. Downloaded images go to
`dataset/electronics/listing_images/`, indexed in
`listing_image_lookup.json`.

## 8. AI-generated image test set

No public dataset of AI-generated e-commerce product photos exists.
**`NasrinImp/COCO_AI`** (Hugging Face) is used as the closest available
proxy: real COCO photos paired with AI-generated versions of the same
caption across six generators (SD 3.5, SD 3, SD 2.1, SDXL, DALL-E,
Midjourney). `fetch_authenticity_test_set.py` streams 60 real/generated
pairs (using the `sdxl_image` column) to `dataset/authenticity_test/`.

## 9. Per-signal image processing

- **`duplicate_evidence`**: perceptual hashing (`imagehash.phash`), a
  64-bit DCT-based fingerprint of a downsampled, low-detail version of
  the image. Two images are duplicates if their Hamming distance <= 5.
  Deliberately invariant to brightness/color, encoding structure only.
- **`image_authenticity`**: grayscale conversion -> 2D FFT
  (`numpy.fft.fft2`) -> `fftshift` -> the fraction of total spectral
  energy in the outer 40% of the frequency spectrum by radius
  (`radius > 0.6 * max_radius`).
- **`product_identity`**: CLIP (`openai/clip-vit-base-patch32`) image
  embeddings via `model.vision_model(...)` -> `model.visual_projection(...)`
  (not the higher-level `get_image_features()`, whose return shape
  differs across `transformers` versions), compared by cosine
  similarity.
