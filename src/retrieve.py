"""
retrieve.py — Hybrid BM25 + Dense retriever for SpotifyCares historical threads.

Builds a FAISS index from training thread embeddings and a BM25 index from
the same threads. At inference, combines lexical and dense scores and applies
optional intent-based filtering and reranking.

Usage:
    # Build indexes (run once after train_intent.py)
    python src/retrieve.py --build-index

    # Demo retrieval
    python src/retrieve.py --query "spotify keeps crashing on my iphone" --intent app_bug_crash
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from schemas import RetrievedExample

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ─── Thread Store ──────────────────────────────────────────────────────────────

class ThreadStore:
    """Loads and indexes thread records for retrieval."""

    def __init__(self, threads_path: str) -> None:
        self.threads: list[dict] = []
        self.train_threads: list[dict] = []
        self._load(threads_path)

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    self.threads.append(rec)
                    if rec.get("split") == "train":
                        self.train_threads.append(rec)
        log.info(
            f"ThreadStore: {len(self.threads)} total, "
            f"{len(self.train_threads)} in train split"
        )

    def get_train_texts(self) -> list[str]:
        return [t["searchable_text"] for t in self.train_threads]

    def get_train_ids(self) -> list[str]:
        return [t["conversation_id"] for t in self.train_threads]


# ─── BM25 Lexical Index ────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self) -> None:
        self._bm25 = None
        self._tokenized: list[list[str]] = []

    def build(self, texts: list[str]) -> "BM25Index":
        from rank_bm25 import BM25Okapi  # type: ignore
        self._tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(self._tokenized)
        log.info(f"BM25 index built over {len(self._tokenized)} documents.")
        return self

    def score(self, query: str, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, indices) of top_k results, sorted descending."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        # Normalise BM25 scores to [0, 1]
        max_s = scores.max() if scores.max() > 0 else 1.0
        return scores[top_idx] / max_s, top_idx

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─── FAISS Dense Index ─────────────────────────────────────────────────────────

class DenseIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._index = None
        self._model = None
        self._dim: int = 0

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, texts: list[str]) -> "DenseIndex":
        import faiss  # type: ignore
        model = self._get_model()
        log.info(f"Encoding {len(texts)} threads for FAISS index…")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
        embeddings = embeddings.astype("float32")
        # L2-normalise for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        self._dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings)
        log.info(f"FAISS IndexFlatIP built: {self._index.ntotal} vectors, dim={self._dim}")
        return self

    def encode_query(self, query: str) -> np.ndarray:
        import faiss  # type: ignore
        model = self._get_model()
        emb = model.encode([query]).astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def search(self, query: str, top_k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, indices) sorted descending."""
        emb = self.encode_query(query)
        scores, indices = self._index.search(emb, top_k)
        return scores[0], indices[0]

    def save(self, path: str) -> None:
        import faiss  # type: ignore
        faiss_path = path + ".faiss"
        meta_path = path + ".meta.pkl"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"model_name": self.model_name, "dim": self._dim}, f)
        log.info(f"Saved FAISS index to {faiss_path}")

    @classmethod
    def load(cls, path: str) -> "DenseIndex":
        import faiss  # type: ignore
        faiss_path = path + ".faiss"
        meta_path = path + ".meta.pkl"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        obj = cls(model_name=meta["model_name"])
        obj._index = faiss.read_index(faiss_path)
        obj._dim = meta["dim"]
        log.info(f"Loaded FAISS index: {obj._index.ntotal} vectors")
        return obj


# ─── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 (lexical) and FAISS (dense) retrieval with optional intent
    filtering and reranking by thread quality.

    Intentionally excludes test-split threads from the retrieval index
    to prevent leakage (train/test conversation-level split).
    """

    def __init__(
        self,
        thread_store: ThreadStore,
        bm25_index: BM25Index,
        dense_index: DenseIndex,
        cfg: dict,
    ) -> None:
        self.store = thread_store
        self.bm25 = bm25_index
        self.dense = dense_index
        self.bm25_w = cfg["retrieval"]["bm25_weight"]
        self.dense_w = cfg["retrieval"]["dense_weight"]
        self.intent_bonus = cfg["retrieval"]["intent_match_bonus"]
        self.top_k = cfg["retrieval"]["top_k"]
        # Thread-level intent labels (optional, populated by classify step)
        self._thread_intents: dict[str, str] = {}

    def set_thread_intents(self, intents: dict[str, str]) -> None:
        """Optionally provide pre-computed intent labels per conversation_id."""
        self._thread_intents = intents

    def retrieve(
        self,
        query: str,
        predicted_intent: str,
        exclude_conversation_ids: Optional[set[str]] = None,
        top_k: Optional[int] = None,
        retrieval_mode: str = "hybrid",  # "hybrid" | "bm25" | "dense" | "random"
    ) -> list[RetrievedExample]:
        """
        Retrieve top-k historical threads relevant to the query.

        Args:
            query: Normalized customer message text.
            predicted_intent: Classifier output — used for intent bonus.
            exclude_conversation_ids: IDs to exclude (e.g., the current test thread).
            top_k: Override config top_k.
            retrieval_mode: Which retriever to use (for ablation studies).

        Returns:
            List of RetrievedExample sorted by descending hybrid score.
        """
        k = top_k or self.top_k
        threads = self.store.train_threads
        n = len(threads)
        exclude = exclude_conversation_ids or set()

        if retrieval_mode == "random":
            import random
            candidates = [
                i for i, t in enumerate(threads)
                if t["conversation_id"] not in exclude
            ]
            chosen = random.sample(candidates, min(k, len(candidates)))
            scores = np.ones(len(chosen))
            idx = np.array(chosen)
        elif retrieval_mode == "bm25":
            raw_scores, idx = self.bm25.score(query, top_k=min(k * 5, n))
            scores = raw_scores
        elif retrieval_mode == "dense":
            raw_scores, idx = self.dense.search(query, top_k=min(k * 5, n))
            scores = raw_scores
        else:  # hybrid
            bm25_scores, bm25_idx = self.bm25.score(query, top_k=min(k * 10, n))
            dense_scores, dense_idx = self.dense.search(query, top_k=min(k * 10, n))

            # Merge into unified score dict
            score_dict: dict[int, float] = {}
            for s, i in zip(bm25_scores, bm25_idx):
                score_dict[int(i)] = score_dict.get(int(i), 0) + self.bm25_w * float(s)
            for s, i in zip(dense_scores, dense_idx):
                if i >= 0:
                    score_dict[int(i)] = score_dict.get(int(i), 0) + self.dense_w * float(s)

            # Intent bonus
            for i, thread in enumerate(threads):
                t_intent = self._thread_intents.get(thread["conversation_id"], "")
                if t_intent == predicted_intent and i in score_dict:
                    score_dict[i] += self.intent_bonus

            if not score_dict:
                return []

            sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            idx = np.array([item[0] for item in sorted_items])
            scores = np.array([item[1] for item in sorted_items])

        # Filter excluded and build results
        results: list[RetrievedExample] = []
        for raw_idx, score in zip(idx, scores):
            i = int(raw_idx)
            if i < 0 or i >= len(threads):
                continue
            thread = threads[i]
            if thread["conversation_id"] in exclude:
                continue

            brand_resp = (
                thread["brand_responses"][0]
                if thread["brand_responses"]
                else "No brand response available."
            )
            customer_text = thread.get("representative_customer_message", "")
            first_tweet_id = thread["tweet_ids"][0] if thread["tweet_ids"] else thread["conversation_id"]

            results.append(
                RetrievedExample(
                    tweet_id=first_tweet_id,
                    conversation_id=thread["conversation_id"],
                    customer_text=customer_text,
                    brand_response=brand_resp,
                    score=round(float(score), 4),
                    intent=self._thread_intents.get(thread["conversation_id"]),
                    note="historical_response",
                )
            )

            if len(results) >= k:
                break

        return results


# ─── Build & Save Indexes ──────────────────────────────────────────────────────

def build_and_save_indexes(cfg: dict) -> None:
    threads_path = cfg["retrieval"]["thread_store_path"]
    faiss_path = cfg["retrieval"]["faiss_index_path"]
    bm25_path = faiss_path.replace(".bin", "_bm25.pkl")

    store = ThreadStore(threads_path)
    texts = store.get_train_texts()

    log.info("Building BM25 index…")
    bm25 = BM25Index().build(texts)
    bm25.save(bm25_path)

    log.info("Building FAISS dense index…")
    dense = DenseIndex(model_name=cfg["retrieval"]["embedding_model"]).build(texts)
    dense.save(faiss_path)

    log.info("Indexes saved.")


def load_retriever(cfg: dict) -> HybridRetriever:
    faiss_path = cfg["retrieval"]["faiss_index_path"]
    bm25_path = faiss_path.replace(".bin", "_bm25.pkl")
    threads_path = cfg["retrieval"]["thread_store_path"]

    store = ThreadStore(threads_path)
    bm25 = BM25Index.load(bm25_path)
    dense = DenseIndex.load(faiss_path)
    return HybridRetriever(store, bm25, dense, cfg)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--query", default=None)
    parser.add_argument("--intent", default="other_unclear")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.build_index:
        build_and_save_indexes(cfg)
        return

    if args.query:
        retriever = load_retriever(cfg)
        results = retriever.retrieve(args.query, args.intent, top_k=args.top_k)
        print(f"\nTop {len(results)} results for: '{args.query}' (intent={args.intent})")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] Score={r.score:.4f}  tweet_id={r.tweet_id}")
            print(f"    Customer: {r.customer_text[:120]}")
            print(f"    Brand:    {r.brand_response[:120]}")


if __name__ == "__main__":
    main()
