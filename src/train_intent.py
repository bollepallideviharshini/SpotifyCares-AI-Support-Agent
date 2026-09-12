"""
train_intent.py — Train and evaluate intent classifiers for SpotifyCares.

Trains three systems:
  1. Trivial baseline   — majority-class predictor
  2. Simple baseline    — TF-IDF (word+char n-grams) + Logistic Regression (calibrated)
  3. Proposed           — Sentence-transformer embeddings + calibrated KNN

All models are saved to artifacts/ for use by evaluate.py.

Usage:
    python src/train_intent.py
    python src/train_intent.py --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from schemas import INTENTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── Seed everything ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Pseudo-labelling via LLM (used to build a labelled training set) ─────────

PSEUDO_LABEL_PROMPT = """You are a customer support intent classifier for Spotify.
Classify the following customer tweet into EXACTLY ONE of these intents:
{intents}

Rules:
- Output only the intent label, nothing else.
- If the message is vague or off-topic, output: other_unclear

Tweet: {tweet}
Intent:"""


def pseudo_label_batch(
    texts: list[str],
    model: str = "gpt-4o-mini",
    batch_size: int = 20,
) -> list[str]:
    """
    Use an LLM to produce pseudo-labels for training data.
    Falls back to 'other_unclear' on any API error.
    Returns one label per text.
    """
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()
    except ImportError:
        log.warning("openai not installed — all pseudo-labels will be 'other_unclear'")
        return ["other_unclear"] * len(texts)

    intents_str = "\n".join(f"- {i}" for i in INTENTS)
    labels: list[str] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            prompt = PSEUDO_LABEL_PROMPT.format(intents=intents_str, tweet=text[:280])
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=20,
                )
                label = resp.choices[0].message.content.strip().lower()
                label = label.replace('"', "").replace("'", "")
                if label not in INTENTS:
                    label = "other_unclear"
            except Exception as exc:
                log.warning(f"LLM labelling error: {exc} — defaulting to other_unclear")
                label = "other_unclear"
            labels.append(label)
        log.info(f"  Pseudo-labelled {min(i + batch_size, len(texts))}/{len(texts)}")

    return labels


# ─── Data loading ──────────────────────────────────────────────────────────────

def load_labelled_data(
    threads_path: str,
    golden_path: str,
    split: str = "train",
    pseudo_label: bool = False,
    llm_model: str = "gpt-4o-mini",
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (texts, labels, ids) for the given split.

    Priority:
    1. Golden set examples (ground-truth labels)
    2. Threads with pseudo-labels from LLM (training split only)
    """
    # Load golden set
    golden_ids: set[str] = set()
    golden_texts: list[str] = []
    golden_labels: list[str] = []
    golden_tweet_ids: list[str] = []

    if Path(golden_path).exists():
        import pandas as pd
        gdf = pd.read_csv(golden_path, dtype=str)
        for _, row in gdf.iterrows():
            if pd.notna(row.get("intent")) and row["intent"] in INTENTS:
                # Use conversation_id to avoid leakage
                cid = str(row.get("conversation_id", row["tweet_id"]))
                golden_ids.add(cid)
                golden_texts.append(str(row["text"]))
                golden_labels.append(str(row["intent"]))
                golden_tweet_ids.append(str(row["tweet_id"]))

    log.info(f"Loaded {len(golden_texts)} golden examples.")

    # Load threads
    threads: list[dict] = []
    if Path(threads_path).exists():
        with open(threads_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("split") == split:
                        threads.append(rec)

    log.info(f"Loaded {len(threads)} threads for split='{split}'.")

    thread_texts: list[str] = []
    thread_labels: list[str] = []
    thread_ids: list[str] = []

    if split == "train" and threads:
        texts_to_label = [t["searchable_text"] for t in threads]
        ids_to_label = [t["conversation_id"] for t in threads]

        if pseudo_label:
            log.info(f"Pseudo-labelling {len(texts_to_label)} training threads…")
            labels = pseudo_label_batch(texts_to_label, model=llm_model)
        else:
            # Load cached pseudo-labels if they exist
            cache_path = Path("artifacts/pseudo_labels_train.json")
            if cache_path.exists():
                with open(cache_path) as f:
                    cache = json.load(f)
                labels = [cache.get(tid, "other_unclear") for tid in ids_to_label]
                log.info("Loaded cached pseudo-labels.")
            else:
                log.warning(
                    "No pseudo-labels found. Using all-'other_unclear' fallback. "
                    "Run with --pseudo-label to generate real labels."
                )
                labels = ["other_unclear"] * len(texts_to_label)

        thread_texts = texts_to_label
        thread_labels = labels
        thread_ids = ids_to_label

    # Combine: golden takes priority
    all_texts = golden_texts + thread_texts
    all_labels = golden_labels + thread_labels
    all_ids = golden_tweet_ids + thread_ids

    return all_texts, all_labels, all_ids


# ─── Trivial baseline ──────────────────────────────────────────────────────────

class MajorityClassifier:
    """Always predicts the majority class with confidence 1.0."""

    def __init__(self) -> None:
        self.majority_class_: Optional[str] = None
        self.classes_: list[str] = INTENTS

    def fit(self, X: list[str], y: list[str]) -> "MajorityClassifier":
        from collections import Counter
        self.majority_class_ = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X: list[str]) -> list[str]:
        return [self.majority_class_] * len(X)

    def predict_proba(self, X: list[str]) -> np.ndarray:
        n = len(X)
        proba = np.zeros((n, len(self.classes_)))
        idx = self.classes_.index(self.majority_class_)
        proba[:, idx] = 1.0
        return proba


# ─── TF-IDF + Logistic Regression (simple baseline) ──────────────────────────

def build_tfidf_pipeline(cfg: dict) -> Pipeline:
    tfidf_cfg = cfg["intent"]["tfidf"]
    lr_cfg = cfg["intent"]["logistic_regression"]

    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(tfidf_cfg["word_ngram_range"]),
        max_features=tfidf_cfg["max_features"] // 2,
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        strip_accents="unicode",
        min_df=2,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(tfidf_cfg["char_ngram_range"]),
        max_features=tfidf_cfg["max_features"] // 2,
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        min_df=2,
    )

    from sklearn.pipeline import FeatureUnion

    features = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

    lr = LogisticRegression(
        C=lr_cfg["C"],
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"],
        random_state=42,
        solver="lbfgs",
        multi_class="multinomial",
    )

    calibrated_lr = CalibratedClassifierCV(lr, method=lr_cfg["calibration_method"], cv=5)

    pipe = Pipeline([("features", features), ("clf", calibrated_lr)])
    return pipe


# ─── Embedding + KNN (proposed) ───────────────────────────────────────────────

def build_embedding_model(cfg: dict) -> "SentenceTransformer":  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    model_name = cfg["intent"]["embedding"]["model_name"]
    log.info(f"Loading sentence-transformer: {model_name}")
    return SentenceTransformer(model_name)


def get_or_compute_embeddings(
    texts: list[str],
    ids: list[str],
    cfg: dict,
    force_recompute: bool = False,
) -> np.ndarray:
    cache_path = Path(cfg["intent"]["embedding"]["cache_path"])
    ids_path = Path(cfg["intent"]["embedding"]["ids_path"])

    if not force_recompute and cache_path.exists() and ids_path.exists():
        cached_ids = np.load(str(ids_path), allow_pickle=True).tolist()
        if cached_ids == ids:
            log.info(f"Loading cached embeddings from {cache_path}")
            return np.load(str(cache_path))
        log.info("Cache IDs mismatch — recomputing embeddings.")

    model = build_embedding_model(cfg)
    log.info(f"Computing embeddings for {len(texts)} texts…")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), embeddings)
    np.save(str(ids_path), np.array(ids))
    log.info(f"Saved embeddings to {cache_path}")
    return embeddings


class EmbeddingKNNClassifier:
    """KNN over sentence-transformer embeddings with softmax probability calibration."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.k = cfg["intent"]["knn"]["k"]
        self.classes_: list[str] = INTENTS
        self._embeddings: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._le = LabelEncoder()

    def fit(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> "EmbeddingKNNClassifier":
        self._embeddings = embeddings
        self._labels = np.array(labels)
        self._le.fit(INTENTS)
        labels_path = Path(self.cfg["intent"]["embedding"]["labels_path"])
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(labels_path), self._labels)
        return self

    def _cosine_sim(self, query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
        c_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-10)
        return q_norm @ c_norm.T

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        sims = self._cosine_sim(embeddings, self._embeddings)
        topk_idx = np.argsort(sims, axis=1)[:, -self.k :]

        n = embeddings.shape[0]
        proba = np.zeros((n, len(self.classes_)))

        for i in range(n):
            neighbor_labels = self._labels[topk_idx[i]]
            neighbor_sims = sims[i, topk_idx[i]]
            # Weight by similarity
            for lab, sim in zip(neighbor_labels, neighbor_sims):
                if lab in self.classes_:
                    idx = self.classes_.index(lab)
                    proba[i, idx] += max(sim, 0)

            # Softmax normalise
            total = proba[i].sum()
            if total > 0:
                proba[i] /= total
            else:
                proba[i, self.classes_.index("other_unclear")] = 1.0

        return proba

    def predict(self, embeddings: np.ndarray) -> list[str]:
        proba = self.predict_proba(embeddings)
        return [self.classes_[i] for i in np.argmax(proba, axis=1)]


# ─── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_classifier(
    name: str,
    y_true: list[str],
    y_pred: list[str],
    y_proba: Optional[np.ndarray] = None,
) -> dict:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=INTENTS, zero_division=0, output_dict=True
    )

    result = {
        "system": name,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_intent": {
            intent: {
                "precision": round(report.get(intent, {}).get("precision", 0), 4),
                "recall": round(report.get(intent, {}).get("recall", 0), 4),
                "f1": round(report.get(intent, {}).get("f1-score", 0), 4),
                "support": report.get(intent, {}).get("support", 0),
            }
            for intent in INTENTS
        },
    }

    if y_proba is not None:
        from sklearn.metrics import brier_score_loss
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(y_true, classes=INTENTS)
        brier = np.mean([
            brier_score_loss(y_bin[:, i], y_proba[:, i])
            for i in range(len(INTENTS))
        ])
        result["brier_score"] = round(float(brier), 4)

    log.info(
        f"[{name}] Macro F1={macro_f1:.4f}  Weighted F1={weighted_f1:.4f}"
    )
    return result


# ─── Main training pipeline ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--pseudo-label",
        action="store_true",
        help="Call LLM to pseudo-label training threads (requires OPENAI_API_KEY)",
    )
    parser.add_argument("--force-embed", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    Path("artifacts").mkdir(exist_ok=True)

    threads_path = cfg["retrieval"]["thread_store_path"]
    golden_path = cfg["data"]["golden_set_path"]

    # ── Load training data ────────────────────────────────────────────────
    train_texts, train_labels, train_ids = load_labelled_data(
        threads_path, golden_path, split="train",
        pseudo_label=args.pseudo_label,
        llm_model=cfg["generation"]["model"],
        seed=cfg["seed"],
    )

    if not train_texts:
        log.error("No training data found. Run prepare_data.py and build_threads.py first.")
        return

    log.info(f"Training on {len(train_texts)} examples.")

    # ── Load test data (golden set only) ─────────────────────────────────
    import pandas as pd
    test_texts: list[str] = []
    test_labels: list[str] = []
    if Path(golden_path).exists():
        gdf = pd.read_csv(golden_path, dtype=str)
        # Only use examples not in training (conversation-level split)
        # The golden set split column determines train/test
        test_rows = gdf[gdf.get("split", pd.Series(["test"] * len(gdf))) == "test"]
        if "split" not in gdf.columns:
            # If no split column, use last 30% as test
            n = len(gdf)
            test_rows = gdf.iloc[int(n * 0.7):]

        for _, row in test_rows.iterrows():
            if pd.notna(row.get("intent")) and row["intent"] in INTENTS:
                test_texts.append(str(row["text"]))
                test_labels.append(str(row["intent"]))

    log.info(f"Test set: {len(test_texts)} examples.")

    if not test_texts:
        log.warning("No test examples — using train as proxy (do NOT use for final eval).")
        test_texts = train_texts[:50]
        test_labels = train_labels[:50]

    all_results: list[dict] = []

    # ── 1. Trivial baseline ───────────────────────────────────────────────
    log.info("=== Training: Trivial Majority Baseline ===")
    majority = MajorityClassifier().fit(train_texts, train_labels)
    maj_pred = majority.predict(test_texts)
    maj_proba = majority.predict_proba(test_texts)
    all_results.append(
        evaluate_classifier("majority_baseline", test_labels, maj_pred, maj_proba)
    )
    with open("artifacts/intent_model_majority.pkl", "wb") as f:
        pickle.dump(majority, f)

    # ── 2. TF-IDF + LR (simple baseline) ─────────────────────────────────
    log.info("=== Training: TF-IDF + Logistic Regression ===")
    tfidf_pipe = build_tfidf_pipeline(cfg)
    tfidf_pipe.fit(train_texts, train_labels)
    tfidf_pred = tfidf_pipe.predict(test_texts)
    tfidf_proba = tfidf_pipe.predict_proba(test_texts)
    all_results.append(
        evaluate_classifier("tfidf_lr_baseline", test_labels, tfidf_pred, tfidf_proba)
    )
    with open("artifacts/intent_model_tfidf.pkl", "wb") as f:
        pickle.dump(tfidf_pipe, f)

    # ── 3. Embedding KNN (proposed) ───────────────────────────────────────
    log.info("=== Training: Sentence-Transformer KNN ===")
    train_embeddings = get_or_compute_embeddings(
        train_texts, train_ids, cfg, force_recompute=args.force_embed
    )
    knn_clf = EmbeddingKNNClassifier(cfg).fit(train_embeddings, train_labels)

    # Embed test texts
    emb_model = build_embedding_model(cfg)
    test_embeddings = emb_model.encode(test_texts, show_progress_bar=False)
    knn_pred = knn_clf.predict(test_embeddings)
    knn_proba = knn_clf.predict_proba(test_embeddings)
    all_results.append(
        evaluate_classifier("embedding_knn_proposed", test_labels, knn_pred, knn_proba)
    )
    with open("artifacts/intent_model_knn.pkl", "wb") as f:
        pickle.dump(knn_clf, f)

    # ── Save combined results ─────────────────────────────────────────────
    results_path = "artifacts/intent_classifier_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved classifier results to {results_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n=== INTENT CLASSIFIER RESULTS ===")
    print(f"{'System':<30} {'Macro F1':>10} {'Weighted F1':>12} {'Brier':>8}")
    print("-" * 65)
    for r in all_results:
        print(
            f"{r['system']:<30} {r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f} "
            f"{r.get('brier_score', 0.0):>8.4f}"
        )


if __name__ == "__main__":
    main()
