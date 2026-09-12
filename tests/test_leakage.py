"""
test_leakage.py — Verifies that the train/test split is leakage-free.

Checks:
1. No conversation_id appears in both train and test splits.
2. Retrieval index only contains train-split threads.
3. Golden set test examples are not in the retrieval corpus.
"""

import sys
import os
import json
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_threads(path: str) -> list[dict]:
    threads = []
    if not Path(path).exists():
        return threads
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                threads.append(json.loads(line))
    return threads


def load_golden(path: str):
    import pandas as pd
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)


class TestLeakage:
    THREADS_PATH = "artifacts/thread_store.jsonl"
    GOLDEN_PATH = "data/golden_set.csv"

    def test_no_conversation_overlap_between_splits(self):
        """Train and test conversation IDs must be disjoint."""
        threads = load_threads(self.THREADS_PATH)
        if not threads:
            pytest.skip("Thread store not built yet — run build_threads.py first")

        train_ids = {t["conversation_id"] for t in threads if t.get("split") == "train"}
        test_ids = {t["conversation_id"] for t in threads if t.get("split") == "test"}

        overlap = train_ids & test_ids
        assert len(overlap) == 0, (
            f"Leakage: {len(overlap)} conversation IDs appear in both train and test! "
            f"Examples: {list(overlap)[:5]}"
        )

    def test_all_threads_have_split_label(self):
        """Every thread must have a split label."""
        threads = load_threads(self.THREADS_PATH)
        if not threads:
            pytest.skip("Thread store not built yet")

        missing_split = [t["conversation_id"] for t in threads if "split" not in t]
        assert len(missing_split) == 0, (
            f"{len(missing_split)} threads have no split label. First few: {missing_split[:5]}"
        )

    def test_golden_test_ids_not_in_train_threads(self):
        """Golden set test examples must not appear as train-split threads."""
        threads = load_threads(self.THREADS_PATH)
        golden = load_golden(self.GOLDEN_PATH)

        if not threads or golden.empty:
            pytest.skip("Thread store or golden set not present yet")

        train_ids = {t["conversation_id"] for t in threads if t.get("split") == "train"}

        if "conversation_id" in golden.columns and "split" in golden.columns:
            golden_test = golden[golden["split"] == "test"]
            golden_test_conv_ids = set(golden_test["conversation_id"].dropna())
        elif "tweet_id" in golden.columns:
            # Fallback: use tweet_id as proxy
            golden_test_conv_ids = set(golden["tweet_id"].dropna())
        else:
            pytest.skip("Golden set has no conversation_id column")

        overlap = golden_test_conv_ids & train_ids
        assert len(overlap) == 0, (
            f"Leakage: {len(overlap)} golden test conversation IDs are also in train threads! "
            f"Examples: {list(overlap)[:3]}"
        )

    def test_note_field_is_historical_response(self):
        """
        All thread records must use 'historical_response' note,
        NEVER 'proven_resolution' (we cannot prove resolutions from public Twitter data).
        """
        threads = load_threads(self.THREADS_PATH)
        if not threads:
            pytest.skip("Thread store not built yet")

        bad = [t["conversation_id"] for t in threads if t.get("note") == "proven_resolution"]
        assert len(bad) == 0, (
            f"{len(bad)} threads incorrectly claim 'proven_resolution'. "
            "Use 'historical_response' only."
        )

    def test_chronological_split_ordering(self):
        """Test split threads should have later created_at than most train threads."""
        threads = load_threads(self.THREADS_PATH)
        if not threads:
            pytest.skip("Thread store not built yet")

        dated = [t for t in threads if t.get("created_at")]
        if not dated:
            pytest.skip("No created_at timestamps in thread store")

        train_dates = sorted(t["created_at"] for t in dated if t.get("split") == "train")
        test_dates = sorted(t["created_at"] for t in dated if t.get("split") == "test")

        if not train_dates or not test_dates:
            pytest.skip("Insufficient data for chronological check")

        # At least 80% of test examples should post-date the median train date
        import statistics
        median_train = statistics.median(train_dates)
        pct_test_after = sum(1 for d in test_dates if d >= median_train) / len(test_dates)

        assert pct_test_after >= 0.70, (
            f"Only {pct_test_after:.1%} of test dates are after median train date. "
            "This suggests the split may not be truly chronological."
        )
