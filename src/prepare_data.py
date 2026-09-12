"""
prepare_data.py — Filter the raw Kaggle Twitter CS dataset to SpotifyCares,
normalize text, and write a compact sample to data/sample_threads.jsonl.

Usage:
    python src/prepare_data.py                    # uses configs/experiment.yaml
    python src/prepare_data.py --raw-path data/twcs.csv --max-threads 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── Text normalisation ────────────────────────────────────────────────────────

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")
HASHTAG_RE = re.compile(r"#(\w+)")


def normalize(text: str, keep_mentions: bool = False) -> str:
    """Lowercase, strip URLs, optionally strip @mentions, collapse whitespace."""
    text = URL_RE.sub(" [URL] ", text)
    if not keep_mentions:
        text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    text = WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


# ─── Kaggle dataset download (optional) ───────────────────────────────────────

def maybe_download_kaggle(raw_path: str) -> bool:
    """
    Download twcs.csv using kagglehub (primary) or kaggle CLI (fallback).
    Returns True if the file is available after the call.
    """
    if Path(raw_path).exists():
        log.info(f"Raw dataset already present at {raw_path}")
        return True

    dest_dir = Path(raw_path).parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    # ── Primary: kagglehub ────────────────────────────────────────────────
    try:
        import kagglehub  # type: ignore
        log.info("Downloading via kagglehub …")
        downloaded_path = kagglehub.dataset_download(
            "thoughtvector/customer-support-on-twitter"
        )
        log.info(f"kagglehub path: {downloaded_path}")

        # kagglehub stores files in its own cache dir; copy twcs.csv to data/
        import shutil
        src = Path(downloaded_path)
        # find twcs.csv anywhere under the downloaded path
        csv_files = list(src.rglob("twcs.csv"))
        if csv_files:
            shutil.copy(csv_files[0], raw_path)
            log.info(f"Copied twcs.csv → {raw_path}")
            return True
        else:
            log.warning("kagglehub download succeeded but twcs.csv not found in cache.")
    except Exception as exc:
        log.warning(f"kagglehub download failed: {exc}")

    # ── Fallback: kaggle CLI ───────────────────────────────────────────────
    try:
        import kaggle  # type: ignore
        log.info("Falling back to kaggle CLI …")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "thoughtvector/customer-support-on-twitter",
            path=str(dest_dir),
            unzip=True,
        )
        if Path(raw_path).exists():
            log.info("kaggle CLI download complete.")
            return True
    except Exception as exc:
        log.warning(f"kaggle CLI download failed: {exc}")

    return False


# ─── Core filtering + sampling ─────────────────────────────────────────────────

def load_brand_tweets(raw_path: str, brand: str) -> pd.DataFrame:
    """
    Load the raw CSV and return only tweets that are part of conversations
    involving `brand`.  The Kaggle TWCS dataset has columns:
        tweet_id, author_id, inbound, created_at, text,
        response_tweet_id, in_response_to_tweet_id
    """
    log.info(f"Loading raw data from {raw_path} …")
    df = pd.read_csv(raw_path, dtype=str, low_memory=False)

    # Standardise column names (sometimes appear as lowercase or with spaces)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"tweet_id", "author_id", "inbound", "created_at", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # inbound == True → customer tweet; False → brand tweet
    df["inbound"] = df["inbound"].astype(str).str.lower().isin(["true", "1", "yes"])

    brand_lower = brand.lower()

    # A conversation involves the brand if any non-inbound tweet is from `brand`.
    brand_mask = df["author_id"].str.lower() == brand_lower
    brand_tweet_ids = set(df.loc[brand_mask, "tweet_id"].dropna())

    # Keep tweets that are IN a brand-involving conversation.
    # Build conversation IDs: the root of each reply chain.
    # Simple approach: collect all tweet_ids that replied to a brand tweet,
    # or that a brand tweet replied to.
    in_response_col = (
        "in_response_to_tweet_id"
        if "in_response_to_tweet_id" in df.columns
        else None
    )

    if in_response_col:
        replied_to_brand = df[in_response_col].isin(brand_tweet_ids)
        is_brand_tweet = brand_mask
        relevant_ids = set(
            df.loc[replied_to_brand | is_brand_tweet, "tweet_id"].dropna()
        )
        # Also include tweets that brand replied to (customer messages)
        brand_replied_to = set(
            df.loc[is_brand_tweet, in_response_col].dropna()
        )
        relevant_ids |= brand_replied_to
    else:
        relevant_ids = brand_tweet_ids

    filtered = df[df["tweet_id"].isin(relevant_ids)].copy()
    log.info(f"Found {len(filtered):,} tweets in {brand} conversations.")
    return filtered


def build_conversation_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Group tweet_ids into conversation threads.
    Uses in_response_to_tweet_id to build parent->children chains.
    Returns: {root_tweet_id: [ordered tweet_ids in thread]}
    """
    in_resp_col = "in_response_to_tweet_id" if "in_response_to_tweet_id" in df.columns else None

    id_to_row = {row["tweet_id"]: row for _, row in df.iterrows()}
    children: dict[str, list[str]] = {}

    if in_resp_col:
        for _, row in df.iterrows():
            parent = row.get(in_resp_col)
            if pd.notna(parent) and parent in id_to_row:
                children.setdefault(parent, []).append(row["tweet_id"])

    # Find roots: tweets with no parent in the dataset
    all_ids = set(df["tweet_id"])
    has_parent_ids = set()
    if in_resp_col:
        for _, row in df.iterrows():
            parent = row.get(in_resp_col)
            if pd.notna(parent) and parent in id_to_row:
                has_parent_ids.add(row["tweet_id"])
    roots = all_ids - has_parent_ids

    # BFS to collect threads
    threads: dict[str, list[str]] = {}
    for root in roots:
        thread: list[str] = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            thread.append(node)
            queue.extend(children.get(node, []))
        threads[root] = thread

    return threads


def deduplicate_text(texts: list[str], threshold: float = 0.85) -> list[int]:
    """
    Return indices of texts that are NOT near-duplicates of an earlier text.
    Uses simple token Jaccard as a fast dedup heuristic.
    """
    kept: list[int] = []
    kept_sets: list[set] = []

    for i, t in enumerate(texts):
        tokens = set(t.lower().split())
        is_dup = any(
            len(tokens & s) / max(len(tokens | s), 1) >= threshold
            for s in kept_sets
        )
        if not is_dup:
            kept.append(i)
            kept_sets.append(tokens)

    return kept


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def prepare(
    raw_path: str,
    out_path: str,
    brand: str = "SpotifyCares",
    max_threads: int = 5000,
    train_cutoff: str = "2017-09-01",
    seed: int = 42,
) -> None:
    random.seed(seed)

    # 1. Try to download if needed
    if not Path(raw_path).exists():
        success = maybe_download_kaggle(raw_path)
        if not success:
            log.error(
                f"Raw data not found at {raw_path} and Kaggle download failed.\n"
                "Please download twcs.csv manually from:\n"
                "  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter\n"
                "and place it at data/twcs.csv"
            )
            sys.exit(1)

    # 2. Load and filter
    df = load_brand_tweets(raw_path, brand)

    # 3. Build conversation index
    threads = build_conversation_index(df)
    log.info(f"Identified {len(threads):,} conversation threads.")

    id_to_row = {row["tweet_id"]: row.to_dict() for _, row in df.iterrows()}

    # 4. Convert threads to output records
    records = []
    for root_id, tweet_ids in threads.items():
        rows = [id_to_row[tid] for tid in tweet_ids if tid in id_to_row]
        if not rows:
            continue

        customer_msgs = [
            normalize(r["text"])
            for r in rows
            if str(r.get("inbound", "false")).lower() in ("true", "1")
        ]
        brand_resps = [
            normalize(r["text"], keep_mentions=False)
            for r in rows
            if str(r.get("inbound", "false")).lower() not in ("true", "1")
        ]

        if not customer_msgs or not brand_resps:
            continue

        # thread quality = number of brand turns
        thread_quality = len(brand_resps)

        # has_followup = customer replied after last brand response
        inbound_flags = [
            str(r.get("inbound", "false")).lower() in ("true", "1")
            for r in rows
        ]
        has_followup = bool(
            inbound_flags and inbound_flags[-1]  # last tweet is from customer
            and thread_quality >= 1
        )

        created_at = rows[0].get("created_at", "")

        records.append(
            {
                "conversation_id": root_id,
                "tweet_ids": tweet_ids,
                "customer_messages": customer_msgs,
                "brand_responses": brand_resps,
                "thread_quality": thread_quality,
                "has_followup": has_followup,
                "created_at": created_at,
                "searchable_text": " ".join(customer_msgs),
                "split": "train" if created_at < train_cutoff else "test",
            }
        )

    # 5. Deduplicate by searchable text
    texts = [r["searchable_text"] for r in records]
    keep_idx = deduplicate_text(texts, threshold=0.85)
    records = [records[i] for i in keep_idx]
    log.info(f"After deduplication: {len(records):,} threads.")

    # 6. Sample up to max_threads (preserve train/test balance)
    if len(records) > max_threads:
        random.shuffle(records)
        records = records[:max_threads]

    # 7. Write output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train_n = sum(1 for r in records if r["split"] == "train")
    test_n = sum(1 for r in records if r["split"] == "test")
    log.info(
        f"Wrote {len(records):,} threads to {out_path}  "
        f"(train={train_n}, test={test_n})"
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SpotifyCares dataset")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--raw-path", default=None)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--brand", default=None)
    parser.add_argument("--max-threads", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    raw_path = args.raw_path or cfg["data"]["raw_path"]
    out_path = args.out_path or cfg["data"]["sample_path"]
    brand = args.brand or cfg["data"]["brand"]
    max_threads = args.max_threads or cfg["data"]["max_threads"]
    train_cutoff = cfg["data"]["train_cutoff_date"]
    seed = cfg["seed"]

    prepare(raw_path, out_path, brand, max_threads, train_cutoff, seed)


if __name__ == "__main__":
    main()
