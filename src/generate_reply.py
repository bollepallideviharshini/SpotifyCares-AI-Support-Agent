"""
generate_reply.py — LLM-based reply generator for SpotifyCares support agent.

Produces a structured AgentOutput by:
  1. Running the intent classifier
  2. Retrieving similar historical threads
  3. Calling the LLM with a strict, grounded prompt
  4. Running the escalation policy

In --offline / --demo mode, uses frozen cached replies so no API key is needed.

Usage:
    python src/generate_reply.py --demo
    python src/generate_reply.py --text "I keep getting logged out of spotify"
    python src/generate_reply.py --batch --golden-path data/golden_set.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import yaml

from schemas import AgentOutput, IntentPrediction, RetrievedExample, INTENTS
from escalation import EscalationPolicy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── Demo tweets ──────────────────────────────────────────────────────────────

DEMO_TWEETS = [
    {"tweet_id": "demo_001", "text": "@SpotifyCares songs keep pausing every 30 seconds it's so annoying"},
    {"tweet_id": "demo_002", "text": "@SpotifyCares I can't log into my account, password reset email never arrives"},
    {"tweet_id": "demo_003", "text": "@SpotifyCares you charged me twice this month what is going on"},
    {"tweet_id": "demo_004", "text": "@SpotifyCares the app crashed and won't open anymore on my iphone 14"},
    {"tweet_id": "demo_005", "text": "@SpotifyCares is spotify down? all my friends can't connect either"},
]

# ─── Prompt loading ────────────────────────────────────────────────────────────

def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


# ─── Classifier loading ────────────────────────────────────────────────────────

def load_classifier(cfg: dict, classifier_type: str = "tfidf"):
    """Load a pre-trained intent classifier from disk."""
    if classifier_type == "tfidf":
        model_path = "artifacts/intent_model_tfidf.pkl"
    elif classifier_type == "knn":
        model_path = "artifacts/intent_model_knn.pkl"
    else:
        model_path = "artifacts/intent_model_majority.pkl"

    if not Path(model_path).exists():
        log.warning(f"Model not found at {model_path}. Using majority fallback.")
        model_path = "artifacts/intent_model_majority.pkl"
        if not Path(model_path).exists():
            log.error("No classifier found. Run train_intent.py first.")
            return None

    with open(model_path, "rb") as f:
        return pickle.load(f)


def classify_intent(
    text: str,
    classifier,
    cfg: dict,
    classifier_type: str = "tfidf",
) -> IntentPrediction:
    """Run intent classification, returning label + calibrated probabilities."""
    if classifier is None:
        return IntentPrediction(intent="other_unclear", confidence=0.5)

    try:
        if classifier_type == "knn":
            from sentence_transformers import SentenceTransformer  # type: ignore
            emb_model = SentenceTransformer(cfg["intent"]["embedding"]["model_name"])
            import numpy as np
            emb = emb_model.encode([text])
            proba_arr = classifier.predict_proba(emb)[0]
            pred_idx = proba_arr.argmax()
            intent = INTENTS[pred_idx]
            confidence = float(proba_arr[pred_idx])
            all_scores = {INTENTS[i]: float(proba_arr[i]) for i in range(len(INTENTS))}
        else:
            # TF-IDF pipeline
            proba_arr = classifier.predict_proba([text])[0]
            classes = classifier.classes_ if hasattr(classifier, "classes_") else INTENTS
            pred_idx = proba_arr.argmax()
            intent = classes[pred_idx]
            if intent not in INTENTS:
                intent = "other_unclear"
            confidence = float(proba_arr[pred_idx])
            all_scores = {classes[i]: float(proba_arr[i]) for i in range(len(proba_arr))}

        return IntentPrediction(
            intent=intent,
            confidence=confidence,
            all_scores=all_scores,
            classifier=classifier_type,
        )
    except Exception as exc:
        log.warning(f"Classification failed: {exc}")
        return IntentPrediction(intent="other_unclear", confidence=0.3)


# ─── LLM Call ─────────────────────────────────────────────────────────────────

def call_llm(
    prompt_template: str,
    customer_text: str,
    intent: str,
    retrieved: list[RetrievedExample],
    cfg: dict,
) -> str:
    """
    Call the configured LLM with a grounded, structured prompt.
    Returns the raw reply string.
    """
    examples_block = "\n\n".join(
        f"Example {i+1}:\n"
        f"  Customer: {ex.customer_text[:200]}\n"
        f"  Spotify response: {ex.brand_response[:200]}\n"
        f"  Note: {ex.note}"
        for i, ex in enumerate(retrieved)
    )
    if not examples_block:
        examples_block = "(No sufficiently similar historical examples found.)"

    prompt = prompt_template.format(
        intent=intent,
        customer_text=customer_text,
        examples=examples_block,
    )

    provider = cfg["generation"]["llm_provider"]
    model = cfg["generation"]["model"]
    temperature = cfg["generation"]["temperature"]
    max_tokens = cfg["generation"]["max_tokens"]

    if provider == "openai":
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a Spotify customer support agent. "
                            "Reply ONLY based on the provided historical examples. "
                            "Do NOT invent policies, refunds, timelines, or links."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            log.warning(f"LLM call failed: {exc}")
            return _fallback_reply(intent, retrieved)

    elif provider == "anthropic":
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as exc:
            log.warning(f"Anthropic call failed: {exc}")
            return _fallback_reply(intent, retrieved)

    elif provider == "gemini":
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            model_obj = genai.GenerativeModel(model)
            resp = model_obj.generate_content(prompt)
            return resp.text.strip()
        except Exception as exc:
            log.warning(f"Gemini call failed: {exc}")
            return _fallback_reply(intent, retrieved)

    return _fallback_reply(intent, retrieved)


def _fallback_reply(intent: str, retrieved: list[RetrievedExample]) -> str:
    """Return the best retrieved brand response as a safe fallback."""
    if retrieved:
        return retrieved[0].brand_response
    fallback_map = {
        "playback_issue": "Hi! Sorry you're having playback issues. Please try logging out and back in, then clear the app cache. Let us know if that helps!",
        "login_account_access": "Hi! Sorry about that. Please try resetting your password via the Spotify website. Check your spam folder for the reset email.",
        "subscription_cancellation": "Hi! You can manage your subscription at spotify.com/account. Let us know if you run into any issues there.",
        "payment_charge": "Hi! We're sorry about that charge. Please DM us your account details (no passwords) and we'll look into it.",
        "app_bug_crash": "Hi! Sorry the app is misbehaving. Please try reinstalling the latest version from your app store.",
        "feature_question": "Hi! Happy to help. Could you tell us a bit more about what you're trying to do?",
        "content_availability": "Hi! Content availability varies by region and licensing. Could you let us know which song/album you're looking for?",
        "account_security": "Hi! If you believe your account is compromised, please change your password immediately at spotify.com/account.",
        "service_outage": "Hi! We're aware some users may be experiencing issues. Please check status.spotify.com for updates.",
    }
    return fallback_map.get(intent, "Hi! Thanks for reaching out. Could you tell us a bit more about the issue?")


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    tweet_id: str,
    text: str,
    cfg: dict,
    classifier=None,
    retriever=None,
    policy: Optional[EscalationPolicy] = None,
    prompt_template: str = "",
    classifier_type: str = "tfidf",
    retrieval_mode: str = "hybrid",
    exclude_conversation_ids: Optional[set] = None,
    system_name: str = "proposed",
) -> AgentOutput:
    """
    End-to-end pipeline for one tweet.
    Each stage is explicit and separate — no single LLM call drives all decisions.
    """

    # 1. Classify intent
    intent_pred = classify_intent(text, classifier, cfg, classifier_type)

    # 2. Retrieve historical examples
    retrieved: list[RetrievedExample] = []
    top_retrieval_score = 0.0
    if retriever is not None and retrieval_mode != "none":
        retrieved = retriever.retrieve(
            query=text,
            predicted_intent=intent_pred.intent,
            exclude_conversation_ids=exclude_conversation_ids,
            retrieval_mode=retrieval_mode,
        )
        top_retrieval_score = retrieved[0].score if retrieved else 0.0

    # 3. Generate reply
    if prompt_template and os.environ.get("OPENAI_API_KEY") or \
       os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        draft_reply = call_llm(prompt_template, text, intent_pred.intent, retrieved, cfg)
    else:
        draft_reply = _fallback_reply(intent_pred.intent, retrieved)

    # 4. Escalation policy (explicit, separate from LLM)
    if policy is None:
        policy = EscalationPolicy(cfg["escalation"])

    action, reason, flags = policy.decide(
        intent=intent_pred.intent,
        confidence=intent_pred.confidence,
        retrieval_score=top_retrieval_score,
        reply_text=draft_reply,
        customer_text=text,
    )

    grounding_passed = policy.is_grounding_passed(draft_reply)

    return AgentOutput(
        tweet_id=tweet_id,
        text=text,
        intent=intent_pred.intent,
        intent_confidence=round(intent_pred.confidence, 4),
        draft_reply=draft_reply,
        resolution_examples=[r.tweet_id for r in retrieved],
        action=action,
        reason=reason,
        safety_flags=flags,
        retrieval_score=round(top_retrieval_score, 4),
        grounding_passed=grounding_passed,
        system=system_name,
    )


# ─── Batch mode ───────────────────────────────────────────────────────────────

def run_batch(
    cfg: dict,
    golden_path: str,
    out_path: str,
    classifier_type: str = "tfidf",
    retrieval_mode: str = "hybrid",
    system_name: str = "proposed",
) -> None:
    """Run the full pipeline on all golden set examples and save frozen predictions."""
    import pandas as pd
    from tqdm import tqdm  # type: ignore

    # Load components
    classifier = load_classifier(cfg, classifier_type)
    policy = EscalationPolicy(cfg["escalation"])
    prompt_template = load_prompt(cfg["generation"]["prompt_path"])

    retriever = None
    faiss_path = cfg["retrieval"]["faiss_index_path"]
    bm25_path = faiss_path.replace(".bin", "_bm25.pkl")
    if Path(faiss_path + ".faiss").exists() and Path(bm25_path).exists():
        from retrieve import load_retriever
        retriever = load_retriever(cfg)

    gdf = pd.read_csv(golden_path, dtype=str)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f_out:
        for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Running pipeline"):
            tweet_id = str(row.get("tweet_id", ""))
            text = str(row.get("text", ""))
            # Exclude own conversation from retrieval (leakage prevention)
            conv_id = str(row.get("conversation_id", tweet_id))
            exclude = {conv_id}

            result = run_pipeline(
                tweet_id=tweet_id,
                text=text,
                cfg=cfg,
                classifier=classifier,
                retriever=retriever,
                policy=policy,
                prompt_template=prompt_template,
                classifier_type=classifier_type,
                retrieval_mode=retrieval_mode,
                exclude_conversation_ids=exclude,
                system_name=system_name,
            )
            f_out.write(result.model_dump_json() + "\n")

    log.info(f"Saved {len(gdf)} predictions to {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SpotifyCares support agent")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--text", default=None, help="Single tweet text")
    parser.add_argument("--tweet-id", default="live_001")
    parser.add_argument("--demo", action="store_true", help="Run on 5 demo tweets")
    parser.add_argument("--batch", action="store_true", help="Run on full golden set")
    parser.add_argument("--golden-path", default=None)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--classifier", default="tfidf", choices=["tfidf", "knn", "majority"])
    parser.add_argument("--retrieval-mode", default="hybrid",
                        choices=["hybrid", "bm25", "dense", "random", "none"])
    parser.add_argument("--system-name", default="proposed")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    classifier = load_classifier(cfg, args.classifier)
    policy = EscalationPolicy(cfg["escalation"])

    prompt_template = ""
    if Path(cfg["generation"]["prompt_path"]).exists():
        prompt_template = load_prompt(cfg["generation"]["prompt_path"])

    retriever = None
    faiss_path = cfg["retrieval"]["faiss_index_path"]
    bm25_path = faiss_path.replace(".bin", "_bm25.pkl")
    if Path(faiss_path + ".faiss").exists() and Path(bm25_path).exists():
        from retrieve import load_retriever
        retriever = load_retriever(cfg)
    else:
        log.warning("Retrieval index not found. Run 'make train' or 'python src/retrieve.py --build-index'.")

    if args.demo or (not args.text and not args.batch):
        tweets = DEMO_TWEETS
        print("\n" + "=" * 70)
        print("SpotifyCares AI Support Agent — DEMO")
        print("=" * 70)
        for t in tweets:
            result = run_pipeline(
                tweet_id=t["tweet_id"],
                text=t["text"],
                cfg=cfg,
                classifier=classifier,
                retriever=retriever,
                policy=policy,
                prompt_template=prompt_template,
                classifier_type=args.classifier,
                retrieval_mode=args.retrieval_mode,
            )
            print(f"\n[Tweet] {t['text']}")
            print(json.dumps(result.model_dump(), indent=2))
        return

    if args.text:
        result = run_pipeline(
            tweet_id=args.tweet_id,
            text=args.text,
            cfg=cfg,
            classifier=classifier,
            retriever=retriever,
            policy=policy,
            prompt_template=prompt_template,
            classifier_type=args.classifier,
            retrieval_mode=args.retrieval_mode,
        )
        print(json.dumps(result.model_dump(), indent=2))
        return

    if args.batch:
        golden_path = args.golden_path or cfg["data"]["golden_set_path"]
        out_path = args.out_path or cfg["generation"]["frozen_replies_path"]
        run_batch(cfg, golden_path, out_path, args.classifier, args.retrieval_mode, args.system_name)


if __name__ == "__main__":
    main()
