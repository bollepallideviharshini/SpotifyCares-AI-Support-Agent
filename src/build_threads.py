"""
build_threads.py — Convert sample_threads.jsonl into an indexed thread store
with per-thread metadata used by the retriever and reply generator.

Adds:
  - Normalized searchable_text
  - A "representative_customer_message" (longest customer turn)
  - Apparent resolution signal (last_turn == brand AND no followup)
  - Thread chronological rank within the dataset

Usage:
    python src/build_threads.py
"""

from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def resolution_signal(record: dict) -> str:
    """
    Estimate apparent resolution.
    - 'likely_resolved'  : brand responded AND no customer followup after brand
    - 'uncertain'        : brand responded but customer replied again
    - 'no_brand_response': no brand turn found
    Note: This is an observation, NOT proof of resolution.
    """
    if not record.get("brand_responses"):
        return "no_brand_response"
    if record.get("has_followup"):
        return "uncertain"
    return "likely_resolved"


def build_threads(sample_path: str, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    records = []
    with open(sample_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log.info(f"Loaded {len(records):,} raw threads from {sample_path}")

    enriched = []
    for rank, rec in enumerate(records):
        customer_msgs: list[str] = rec.get("customer_messages", [])
        brand_resps: list[str] = rec.get("brand_responses", [])

        # Representative customer message = longest customer turn
        rep_msg = max(customer_msgs, key=len) if customer_msgs else ""

        # Apparent outcome signal
        outcome = resolution_signal(rec)

        thread_record = {
            "conversation_id": rec["conversation_id"],
            "tweet_ids": rec["tweet_ids"],
            "customer_messages": customer_msgs,
            "brand_responses": brand_resps,
            "representative_customer_message": rep_msg,
            "searchable_text": rec.get("searchable_text", " ".join(customer_msgs)),
            "thread_quality": rec.get("thread_quality", len(brand_resps)),
            "has_followup": rec.get("has_followup", False),
            "apparent_outcome": outcome,
            "note": "historical_response",  # NEVER claim 'proven_resolution'
            "created_at": rec.get("created_at", ""),
            "split": rec.get("split", "train"),
            "chronological_rank": rank,
        }
        enriched.append(thread_record)

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in enriched:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train = sum(1 for r in enriched if r["split"] == "train")
    test = sum(1 for r in enriched if r["split"] == "test")
    log.info(
        f"Wrote {len(enriched):,} enriched threads to {out_path} "
        f"(train={train}, test={test})"
    )

    # Summary statistics
    outcomes = {}
    for r in enriched:
        o = r["apparent_outcome"]
        outcomes[o] = outcomes.get(o, 0) + 1
    log.info(f"Apparent outcome distribution: {outcomes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build thread store")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_threads(
        sample_path=cfg["data"]["sample_path"],
        out_path=cfg["retrieval"]["thread_store_path"],
    )


if __name__ == "__main__":
    main()
