import numpy as np
import pytest
from PIL import Image

from app.checks.behavioral import (
    score_behavioral,
    PRIOR_CLAIM_COUNT_THRESHOLD,
    RETURN_CHARGEBACK_THRESHOLD,
    TIGHT_GAP_MS,
)
from app.checks.duplicate_evidence import score_duplicate_evidence, hash_index
from app.checks.image_authenticity import score_image_authenticity


def _make_features(**overrides):
    base = {
        "prior_claim_count": 0,
        "recent_claim_count_last_5": 0,
        "return_chargeback_count": 0,
        "avg_gap_between_claims_ms": None,
    }
    base.update(overrides)
    return base


def test_behavioral_no_history_scores_zero():
    result = score_behavioral(_make_features())
    assert result["score"] == 0.0
    assert result["method"] == "rule"
    assert "no prior claim history" in result["reasons"][0]


def test_behavioral_high_prior_claim_count_flags():
    features = _make_features(prior_claim_count=int(PRIOR_CLAIM_COUNT_THRESHOLD) + 1)
    result = score_behavioral(features)
    assert result["score"] > 0.0
    assert any("prior claims" in r for r in result["reasons"])


def test_behavioral_high_return_chargeback_flags():
    features = _make_features(return_chargeback_count=int(RETURN_CHARGEBACK_THRESHOLD) + 1)
    result = score_behavioral(features)
    assert result["score"] > 0.0


def test_behavioral_tight_gap_flags():
    features = _make_features(avg_gap_between_claims_ms=TIGHT_GAP_MS - 1)
    result = score_behavioral(features)
    assert result["score"] > 0.0
    assert any("avg gap" in r for r in result["reasons"])


def test_behavioral_score_never_exceeds_one():
    features = _make_features(
        prior_claim_count=999,
        return_chargeback_count=999,
        recent_claim_count_last_5=999,
        avg_gap_between_claims_ms=0,
    )
    result = score_behavioral(features)
    assert result["score"] <= 1.0


@pytest.fixture(autouse=True)
def clear_hash_index():
    """hash_index is module-level state that persists across calls --
    clear it before/after each test so tests don't leak into each other."""
    hash_index.clear()
    yield
    hash_index.clear()


@pytest.fixture
def two_distinct_images(tmp_path):
    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"

    gradient = np.fromfunction(lambda y, x: (x + y) % 256, (64, 64), dtype=int).astype(np.uint8)
    Image.fromarray(gradient).save(path_a)

    checkerboard = np.indices((64, 64)).sum(axis=0) % 2 * 255
    Image.fromarray(checkerboard.astype(np.uint8)).save(path_b)

    return str(path_a), str(path_b)


def test_duplicate_evidence_flags_reused_image(two_distinct_images):
    path_a, _ = two_distinct_images
    score_duplicate_evidence("claim_original", [path_a])
    result = score_duplicate_evidence("claim_copy", [path_a])
    assert result["score"] == 1.0
    assert "claim_original" in result["reasons"][0]


def test_duplicate_evidence_does_not_flag_distinct_images(two_distinct_images):
    path_a, path_b = two_distinct_images
    score_duplicate_evidence("claim_original", [path_a])
    result = score_duplicate_evidence("claim_other", [path_b])
    assert result["score"] == 0.0


def test_duplicate_evidence_no_images_returns_zero():
    result = score_duplicate_evidence("claim_no_images", [])
    assert result["score"] == 0.0

@pytest.fixture
def smooth_and_noisy_images(tmp_path):
    smooth_path = tmp_path / "smooth.jpg"
    noisy_path = tmp_path / "noisy.jpg"

    Image.fromarray(np.full((128, 128), 128, dtype=np.uint8)).save(smooth_path)

    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    Image.fromarray(noisy).save(noisy_path)

    return str(smooth_path), str(noisy_path)


def test_image_authenticity_flags_unusually_smooth_image(smooth_and_noisy_images):
    smooth_path, _ = smooth_and_noisy_images
    result = score_image_authenticity([smooth_path])
    assert result["score"] > 0.0


def test_image_authenticity_does_not_flag_noisy_image(smooth_and_noisy_images):
    _, noisy_path = smooth_and_noisy_images
    result = score_image_authenticity([noisy_path])
    assert result["score"] < 1.0


def test_image_authenticity_no_images_returns_zero():
    result = score_image_authenticity([])
    assert result["score"] == 0.0
    assert result["reasons"] == ["no images submitted"]# additions to tests/test_checks.py

from app.checks.product_identity import score_product_identity


@pytest.fixture
def similar_and_dissimilar_images():
    import json

    lookup = json.load(open("dataset/electronics/listing_image_lookup.json"))
    listing_paths = list(lookup.values())
    assert len(listing_paths) >= 1, "need at least 1 downloaded listing image"

    scene_rows = [json.loads(l) for l in open("dataset/authenticity_test/manifest.jsonl")]
    scene_path = scene_rows[0]["path"]

    return listing_paths[0], listing_paths[0], scene_path


def test_product_identity_similar_image_not_flagged(similar_and_dissimilar_images):
    listing_path, similar_path, _ = similar_and_dissimilar_images
    result = score_product_identity([similar_path], listing_path)
    assert result["score"] == 0.0


def test_product_identity_dissimilar_image_flagged(similar_and_dissimilar_images):
    listing_path, _, dissimilar_path = similar_and_dissimilar_images
    result = score_product_identity([dissimilar_path], listing_path)
    assert result["score"] > 0.0
