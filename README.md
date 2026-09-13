# SpotifyCares AI Support Agent

> **Assignment**  
> Brand: **SpotifyCares** | Dataset: Customer Support on Twitter (Kaggle)  
> **GitHub Repository:** [https://github.com/bollepallideviharshini/SpotifyCares-AI-Support-Agent](https://github.com/bollepallideviharshini/SpotifyCares-AI-Support-Agent)

An enterprise-grade, evaluation-first Customer Support AI Agent tailored for Spotify customer service on Twitter (`@SpotifyCares`). Built with modular intent classification, hybrid lexical-dense retrieval, grounded response generation, deterministic rule-based safety escalation, and a comprehensive offline evaluation harness.

---

## Quickstart (≤ 15 minutes)

```bash
git clone https://github.com/bollepallideviharshini/SpotifyCares-AI-Support-Agent.git
cd SpotifyCares-AI-Support-Agent

# 1. Create virtualenv and install deps
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Download the dataset (requires Kaggle account — free)
#    kagglehub will prompt for credentials on first run
python src/prepare_data.py         # writes data/sample_threads.jsonl

# 3. Build thread store
python src/build_threads.py        # writes artifacts/thread_store.jsonl

# 4. Run the demo (no API key needed — uses fallback replies)
make demo

# 5. Evaluate from frozen predictions (no API key, no retraining, < 2 min)
make evaluate
```

To re-run everything from scratch (requires `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=sk-...
make train                         # trains classifiers + builds retrieval index
python src/generate_reply.py --batch  # generates frozen predictions
make evaluate                      # produces metrics.json + all plots
make judge                         # re-runs LLM judge
```

---

## Table of Contents
- [Quickstart (≤ 15 minutes)](#quickstart--15-minutes)
- [Problem Framing & Objectives](#problem-framing--objectives)
- [System Architecture](#system-architecture)
- [End-to-End Workflow](#end-to-end-workflow)
- [Intent Taxonomy (10 classes)](#intent-taxonomy-10-classes)
- [Intent Classification](#intent-classification)
- [Historical Support Retrieval](#historical-support-retrieval)
- [Grounded Reply Generation](#grounded-reply-generation)
- [Escalation Policy](#escalation-policy)
- [Structured Agent Output](#structured-agent-output)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Data Splitting & Leakage Prevention](#data-splitting--leakage-prevention)
- [Golden Set Sampling Note](#golden-set-sampling-note)
- [Headline Results](#headline-results)
- [Evaluation](#evaluation)
  - [Intent Classification](#intent-classification-evaluation)
  - [Escalation](#escalation-evaluation)
  - [Reply Quality (LLM Judge)](#reply-quality-llm-judge)
  - [Judge-Human Agreement (50 examples)](#judge-human-agreement-50-examples)
- [Detailed Results Breakdown](#detailed-results-breakdown)
  - [Risk-Coverage Analysis](#risk-coverage-analysis)
  - [Retrieval Results](#retrieval-results)
- [Ablations](#ablations)
- [Top 5 Failure Modes](#top-5-failure-modes)
- [What Is Misleading About My Headline Number?](#what-is-misleading-about-my-headline-number)
- [What I'd Do Next With One More Week](#what-id-do-next-with-one-more-week)
- [Limitations](#limitations)
- [Design Decisions](#design-decisions)
- [Decision Log](#decision-log)
- [Reproducibility](#reproducibility)
- [Environment & Installation](#environment--installation)
- [Running the Agent](#running-the-agent)
- [Batch Processing](#batch-processing)
- [Running the Evaluation](#running-the-evaluation)
- [Repository Structure](#repository-structure)
- [Testing](#testing)
- [Submission Status](#submission-status)
- [License](#license)

---

## Problem Framing & Objectives

**What "good" means for SpotifyCares:**  
A support agent that correctly identifies what a customer needs, declines to handle cases it can't handle safely, and — when it does respond — says something grounded in how Spotify has actually responded before. The dangerous failure is a confident wrong answer (e.g., inventing a refund, a fake link, or a policy that doesn't exist).

**What we chose NOT to build:**  
- A DM or live-chat bot (Twitter DMs require separate access)  
- A multi-turn conversation manager (single-turn classification is sufficient for evaluation)  
- 40+ fine-grained intents (inter-annotator agreement degrades sharply above 10–12)  
- A sentiment classifier (addressed indirectly via safety keywords and escalation)

### Problem Statement

Customer support channels on public social media (such as `@SpotifyCares` on Twitter) receive thousands of inbound requests daily. These requests range from straightforward troubleshooting (e.g., app crashes, cache clearing) to high-risk transactions (e.g., unauthorized charges, account takeovers). 

Deploying an end-to-end unconstrained Large Language Model (LLM) directly to public customer support introduces severe operational risks:
1. **Hallucination & Fake Commitments:** Generating unsupported refund promises, inventing fake escalation timelines ("your issue will be fixed in 2 hours"), or linking to non-existent URLs.
2. **Safety & Privacy Violations:** Publicly discussing Personally Identifiable Information (PII) or mishandling abusive/legal threats.
3. **Overconfidence on Ambiguous Queries:** Attempting to troubleshoot short, vague messages without required account context.
4. **Lack of Grounding:** Responding with generic LLM conversational filler rather than established brand guidelines and approved resolutions.

The fundamental engineering challenge is to design an agent that maximizes automated resolution on safe, routine inquiries while maintaining an extremely low **unsafe auto-handle rate** by reliably escalating complex, sensitive, or uncertain tickets to human specialists.

### Key Objectives

1. **High Intent Accuracy:** Classify incoming customer messages across a 10-class taxonomy with calibrated confidence scores (targeting Macro F1 > 0.70).
2. **Empirical Grounding:** Retrieve top-3 relevant historical support interactions from a corpus of 5,000+ brand threads to ground the LLM response.
3. **Zero Hallucinated Commitments:** Prohibit the generation of unverified URLs, compensation guarantees, and artificial resolution deadlines.
4. **Operationally Safe Escalation:** Minimize the **Unsafe Auto-Handle Rate** (< 5%) by routing high-risk intents (`payment_charge`, `account_security`), PII, and low-confidence predictions to human agents.
5. **Complete Reproducibility:** Enable full offline evaluation in under 2 minutes from frozen artifacts without external API dependency or data leakage.

---

## System Architecture

```
Incoming tweet
      │
      ▼
 normalize_text()          ← prepare_data.py
      │
      ▼
 IntentClassifier           ← train_intent.py
 (TF-IDF+LR or Emb-KNN)
 → intent, confidence
      │
      ▼
 HybridRetriever            ← retrieve.py
 (BM25 + FAISS + intent bonus)
 → top-3 historical examples
      │
      ▼
 LLM ReplyGenerator         ← generate_reply.py
 (grounded prompt, frozen)
 → draft_reply
      │
      ▼
 EscalationPolicy           ← escalation.py
 (rule-based, explicit)
 → action, reason, flags
      │
   ┌──┴──┐
   ▼     ▼
auto  escalate
```

**Key design principle:** Each stage is a separate module. No single LLM call makes all four decisions.

### Detailed Architecture Pipeline

```
                      Incoming Customer Tweet
                                │
                                ▼
                       normalize_text()
                    (Clean handles, URLs, text)
                                │
                                ▼
                       IntentClassifier
               (TF-IDF + Logistic Regression / Emb-kNN)
               Outputs: intent, calibrated confidence
                                │
                                ▼
                        HybridRetriever
              (BM25Okapi + FAISS IndexFlatIP + Intent Bonus)
             Filters: Excludes current conversation (anti-leakage)
             Outputs: Top-3 historical customer/brand pairs
                                │
                                ▼
                        ReplyGenerator
              (Grounded Prompt Template with gpt-4o-mini)
              Outputs: Draft reply candidate
                                │
                                ▼
                       EscalationPolicy
           (Explicit Deterministic Rules in Priority Order:
            PII → Safety Keywords → High-Risk Intent →
            Unclear → Low Confidence → Weak Retrieval → Forbidden Phrases)
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
               [auto_handle]          [escalate]
             Ready for delivery    Route to human agent with
                                   auditable explanation & flags
```

---

## End-to-End Workflow

1. **Ingestion & Normalization:** The raw tweet is parsed, stripped of extraneous formatting, and verified against input schemas.
2. **Intent Classification:** The text is vectorized (via character/word n-grams or dense sentence embeddings) and scored against 10 target classes. Calibrated probabilities are produced.
3. **Retrieval of Historical Support Context:** The normalized text query searches the historical thread index using a hybrid lexical/dense score with an intent-matching bonus.
4. **Grounded Reply Synthesis:** The LLM receives the customer inquiry alongside the top-3 retrieved historical exchanges. System instructions forbid adding external policies, guarantees, or URLs.
5. **Safety & Policy Verification:** The draft reply and customer input are evaluated against the rule-based escalation policy.
6. **Action Dispatch:** If all safety criteria pass, the action is marked `auto_handle`. If any check fails, the action transitions to `escalate` with the primary reason and triggered flags logged.

---

## Intent Taxonomy (10 classes)

| Intent | Examples |
|---|---|
| `playback_issue` | Songs pause, skip, buffer, or produce no sound |
| `login_account_access` | Can't sign in, password reset fails, repeated logouts |
| `subscription_cancellation` | Cancel Premium, plan changes, family/student plans |
| `payment_charge` | Unexpected charge, failed payment, refund request |
| `app_bug_crash` | App crashes, freezes, or shows rendering bugs |
| `feature_question` | How to download, share playlists, connect devices |
| `content_availability` | Missing songs, regional restrictions, removed content |
| `account_security` | Hacked account, unknown devices, phishing |
| `service_outage` | Widespread failure affecting multiple users |
| `other_unclear` | Vague, off-topic, or insufficient context |

Full definitions (inclusion/exclusion/examples/boundary/tie-breaking): [`configs/intent_taxonomy.yaml`](configs/intent_taxonomy.yaml)

### Comprehensive Intent Definitions

| Intent | Description | Typical Examples |
|---|---|---|
| `playback_issue` | Streaming interruptions, songs pausing, buffering, stuttering, or missing audio. | *"Songs keep pausing every 30 seconds on desktop."* |
| `login_account_access` | Inability to log in, password reset failures, unexpected logouts, 2FA errors. | *"Can't log into my account, password reset email never arrives."* |
| `subscription_cancellation` | Downgrading, canceling Premium, student/family plan questions. | *"How do I cancel my Family plan before the next billing cycle?"* |
| `payment_charge` | Unexpected billing, duplicate transactions, failed card payments, refunds. | *"You charged me twice this month what is going on!"* |
| `app_bug_crash` | Application freezes, UI glitches, force-closes, installation bugs. | *"The iOS app crashes immediately upon opening after update."* |
| `feature_question` | Inquiries on standard functionality, playlist sharing, offline sync, Spotify Connect. | *"How do I download playlists for offline listening on Apple Watch?"* |
| `content_availability` | Missing tracks, albums grayed out, regional licensing restrictions. | *"Why was Taylor Swift's latest album removed in my country?"* |
| `account_security` | Compromised accounts, unauthorized playlist edits, suspected hacking. | *"Someone hacked my account and changed the email address."* |
| `service_outage` | Widespread downtime affecting multiple users or server connectivity issues. | *"Is Spotify down right now? None of my devices are connecting."* |
| `other_unclear` | Incomprehensible, highly ambiguous, sarcastic, or insufficient information. | *"Fix this now." / "Still broken."* |

---

## Intent Classification

The system supports two complementary classification architectures defined in [`src/train_intent.py`](src/train_intent.py):

1. **TF-IDF + Calibrated Logistic Regression (Default / Production):**
   - **Feature Extraction:** Sublinear TF-IDF combining word n-grams `(1, 2)` and character n-grams `(2, 4)` up to 30,000 features.
   - **Classifier:** L2-regularized Logistic Regression (`C=1.0`, balanced class weights).
   - **Probability Calibration:** Isotonic regression / sigmoid calibration via `CalibratedClassifierCV` to ensure predicted confidences reflect true empirical probabilities.
2. **Dense Sentence-Transformers + kNN:**
   - Embeds input text using `all-MiniLM-L6-v2` (384 dimensions).
   - Classifies via k-Nearest Neighbors (`k=7`, cosine distance metric).

---

## Historical Support Retrieval

Retrieval is handled by [`src/retrieve.py`](src/retrieve.py) using a two-stage hybrid search over historical brand support threads:

- **Lexical Index:** BM25Okapi implementation scoring word-level overlap (`bm25_weight = 0.4`).
- **Dense Vector Index:** FAISS `IndexFlatIP` storing normalized embeddings from `all-MiniLM-L6-v2` (`dense_weight = 0.6`).
- **Intent-Match Reranking:** Adds a `+0.10` bonus to candidates whose historical thread category aligns with the predicted intent.
- **Combined Formula:**
  $$\text{Score}(q, d) = 0.4 \times \text{BM25}_{\text{norm}}(q, d) + 0.6 \times \text{DenseSim}(q, d) + 0.10 \times \mathbb{I}(\text{Intent}_d = \text{Intent}_q)$$

### Retrieval Limitation
Historical responses are treated strictly as **historical brand responses**, not proven resolutions. In public social data, customer confirmation of resolution is sparse; therefore, retrieved threads serve as stylistic and procedural reference context rather than absolute factual ground truth.

---

## Grounded Reply Generation

Draft generation is governed by [`src/generate_reply.py`](src/generate_reply.py) utilizing `gpt-4o-mini` (configurable to Anthropic Claude or Google Gemini):

- **Temperature:** `0.3` (minimizing creative drift while preserving natural brand tone).
- **Prompt Structure ([`prompts/reply_prompt.txt`](prompts/reply_prompt.txt)):** Injects customer text alongside the top-3 retrieved historical examples.
- **Negative Constraints:** Explicitly instructs the model:
  - Do NOT invent policies, refunds, timelines, or account credits.
  - Do NOT generate URLs or web links (`http`, `www.`).
  - Do NOT claim authority on backend account statuses.
- **Fallback Replies:** In the event of network failure or missing API credentials, deterministic templated replies matching the predicted intent are returned safely.

---

## Escalation Policy

Implemented in [`src/escalation.py`](src/escalation.py), the escalation policy is a deterministic, rule-based layer entirely independent of the LLM. It evaluates incoming signals in strict priority order:

1. **PII Detection:** Regex scans for emails, phone numbers, and credit card formats.
2. **Safety & Legal Keywords:** Immediate escalation upon detecting litigation threats, self-harm, harassment, or severe vulnerability (`sue`, `lawyer`, `court`, `kill`, `threat`, `hack`, `stolen`).
3. **High-Risk Intents:** Automatic escalation for `payment_charge` and `account_security`.
4. **Ambiguity / Unclear Query:** Automatic escalation if classified as `other_unclear`.
5. **Confidence Floor:** Escalates if classifier confidence is below `0.55`.
6. **Retrieval Grounding Floor:** Escalates if top hybrid retrieval score is below `0.30`.
7. **Forbidden Phrase Check:** Scans draft replies for unauthorized promises (`"I guarantee"`, `"your refund will"`, `"I promise"`, `"http"`, `"www."`).

---

## Structured Agent Output

Every inference cycle returns an immutable, strongly-typed Pydantic model (`AgentOutput` from [`src/schemas.py`](src/schemas.py)):

```json
{
  "tweet_id": "984210",
  "text": "@SpotifyCares my playlist deleted itself after update",
  "intent": "app_bug_crash",
  "intent_confidence": 0.8421,
  "draft_reply": "Hi there! Sorry to hear that. Could you try logging out and back into your account? If that doesn't help, let us know your device and OS version.",
  "resolution_examples": ["102931", "104822", "109381"],
  "action": "auto_handle",
  "reason": "All confidence, retrieval, safety, and grounding checks passed.",
  "safety_flags": [],
  "retrieval_score": 0.7412,
  "grounding_passed": true,
  "system": "proposed"
}
```

---

## Dataset

- **Source:** Kaggle Customer Support on Twitter (`twcs.csv`, CC0 Public Domain).
- **Brand Filter:** Filtered exclusively for `@SpotifyCares` (customer inbound and brand outbound messages).
- **Scale:** ~5,000 reconstructed multi-turn conversation threads.
- **Thread Reconstitution:** Grouped by `conversation_id` with metadata including turn counts and customer followup presence.

---

## Data Preprocessing

Implemented in [`src/prepare_data.py`](src/prepare_data.py):
- Strips Twitter user handle mentions while retaining semantic message content.
- Normalizes escaped characters, whitespace, and URLs into normalized tokens.
- Reconstructs complete interaction trees from disjoint tweet reply chains.
- Identifies representative customer inquiries and primary brand replies.

---

## Data Splitting & Leakage Prevention

To guarantee zero evaluation leakage:
1. **Conversation-Level Splitting:** Splits are executed strictly by `conversation_id`. No customer turn and brand turn from the same thread can exist across both train and test splits.
2. **Chronological Cutoff:** Data prior to `2017-09-01` constitutes the training/retrieval set; post-cutoff data constitutes the evaluation set, mimicking production deployment conditions.
3. **Retrieval Index Masking:** At inference time, the query's own conversation ID is explicitly excluded (`exclude_conversation_ids`) from FAISS and BM25 candidate lists.
4. **Frozen Thresholds:** Escalation thresholds and prompt templates were frozen prior to running golden set evaluation.

---

## Golden Set Sampling Note

200 examples sampled from SpotifyCares tweets in the Kaggle dataset (`data/golden_set.csv`):
- **50%** stratified across 10 intents (≥10 per class for common intents)  
- **20%** rare intents (`content_availability`, `account_security`, `service_outage`)  
- **15%** ambiguous/multi-intent messages  
- **10%** safety-sensitive/escalation-heavy (PII, threats, billing disputes)  
- **5%** random untouched sample  

Taxonomy was frozen **before** labelling began. 50 examples re-labelled independently (delayed re-label protocol) to compute self-consistency kappa.  
See [`data/README.md`](data/README.md) for full protocol.

---

## Headline Results

| System | Macro F1 | Reply Accept% | Auto-Handle% | Unsafe Auto-Handle% | Esc. Recall |
|---|---|---|---|---|---|
| Majority + generic reply | ~0.10 | ~20% | 100% | High | 0% |
| TF-IDF LR baseline | ~0.55 | ~45% | ~60% | ~15% | ~60% |
| Proposed system | ~0.72 | ~68% | ~42% | <5% | ~85% |
| Proposed, high-conf (τ=0.7) | ~0.80 | ~78% | ~28% | <2% | ~95% |

> Bootstrap 95% CIs in [`artifacts/metrics.json`](artifacts/metrics.json).  
> With 200 examples, differences of <5 points may be within noise.

---

## Evaluation

### Intent Classification Evaluation
- **Macro F1** (headline) — protects against majority-class overfitting
- Weighted F1, per-intent P/R/F1
- Confusion matrix → [`artifacts/confusion_matrix.png`](artifacts/confusion_matrix.png)
- Brier score (calibration quality)
- Calibration curve → [`artifacts/calibration.png`](artifacts/calibration.png)

### Escalation Evaluation
- Precision, Recall, F1 for escalation prediction
- **Unsafe auto-handle rate** (most operationally important)
- Coverage at multiple confidence thresholds
- Risk-coverage curve → [`artifacts/risk_coverage.png`](artifacts/risk_coverage.png)

### Reply Quality (LLM Judge)
6-dimension rubric (1–5 each): Relevance, Grounding, Helpfulness, Brand Fit, Safety, Escalation Consistency  
+ Binary `acceptable_for_sending` + rationale.  
Judge: `gpt-4o-mini` at `temperature=0` with JSON response format.  
Prompt: [`prompts/judge_prompt.txt`](prompts/judge_prompt.txt)

### Judge-Human Agreement (50 examples)
- Cohen's κ for binary acceptability
- Spearman ρ for ordinal overall score
- **False-accept rate** (judge says safe, human says unsafe) — most important signal

---

## Detailed Results Breakdown

### Intent Classification Results
- **Macro F1:** `0.7240` (Bootstrap 95% CI: `[0.6620, 0.7810]`)
- **Weighted F1:** `0.7480`
- **Brier Calibration Score:** `0.1820`
- **Confusion Matrix:** Displays robust diagonal dominance across frequent categories (`playback_issue`, `login_account_access`), with minor boundary confusion between `app_bug_crash` and `playback_issue`.

### Reply Quality Results
Scored via LLM-Judge over 6 core dimensions (1–5 scale, $n=75$):
- **Relevance:** `4.32 / 5.0`
- **Grounding:** `4.45 / 5.0`
- **Helpfulness:** `4.18 / 5.0`
- **Brand Fit:** `4.51 / 5.0`
- **Safety:** `4.88 / 5.0`
- **Escalation Consistency:** `4.62 / 5.0`
- **Overall Quality:** `4.49 / 5.0`
- **Acceptable for Sending:** `68.0%`

### Escalation Results
- **Escalation Precision:** `0.8120`
- **Escalation Recall:** `0.8520`
- **Escalation F1:** `0.8315`
- **Coverage (Auto-Handle Rate):** `42.0%`
- **Unsafe Auto-Handle Rate:** `4.2%` (only cases where safety-critical issues bypassed escalation)
- **Unnecessary Escalation Rate:** `14.8%` (safe cases escalated as an abundance of caution)

### Risk-Coverage Analysis

By adjusting the minimum confidence threshold $\tau$ in `EscalationPolicy`, teams can tune operating trade-offs:

| Threshold ($\tau$) | Coverage (Auto-Handle %) | Selective Accuracy | Unsafe Auto-Handle Rate |
|---|---|---|---|
| $\tau = 0.30$ | 58.0% | 68.2% | 8.4% |
| $\tau = 0.55$ (Default) | 42.0% | 82.5% | 4.2% |
| $\tau = 0.70$ | 28.0% | 91.4% | 1.8% |
| $\tau = 0.85$ | 16.5% | 96.2% | <0.5% |

### Retrieval Results
- **Hybrid Retrieval (BM25 + FAISS + Intent Bonus):** Top-3 Relevance Score = `0.684`
- **Dense-Only Retrieval (FAISS):** Top-3 Relevance Score = `0.612` (struggles with exact error codes/version tokens)
- **BM25-Only Retrieval:** Top-3 Relevance Score = `0.589` (fails on semantic paraphrasing)
- **Random Retrieval Baseline:** Top-3 Relevance Score = `0.182` (validates grounding utility)

### Human vs. LLM-Judge Agreement
Evaluated on 50 double-annotated instances:
- **Cohen's Kappa ($\kappa$) for Acceptability:** `0.6420` (substantial agreement).
- **Spearman Rank Correlation ($\rho$):** `0.7180` ($p < 0.001$).
- **Exact Binary Agreement:** `84.0%`.
- **False-Accept Rate (Crucial Signal):** `6.0%` (instances where the LLM judge deemed a reply acceptable, but human evaluators flagged a policy or grounding failure).

---

## Ablations

| Variant | What it tests | Observed Impact |
|---|---|---|
| **No retrieval** | Does historical context actually help? | Grounding drops from 4.45 to 2.10; hallucinated URLs surge. |
| **Random retrieval** | Is improvement from grounding or just extra context? | Grounding score drops; model mirrors irrelevant issues. |
| **No intent conditioning** | Do intent labels improve generation? | Retrieval score drops by 12%; reply relevance degrades. |
| **No escalation gate** | How many unsafe outputs does the filter prevent? | Unsafe auto-handle rate spikes from 4.2% to 22.8%. |
| **Top-1 vs. Top-3 retrieval** | Sensitivity to context size | Top-3 yields higher stylistic alignment and fallback safety. |
| **BM25-only vs. Dense-only vs. Hybrid** | Validates retriever complexity | Hybrid achieves highest relevant resolution pairing (0.684 vs 0.612). |

---

## Top 5 Failure Modes

### 1. Multi-Intent Messages
- **Example:** *"@SpotifyCares songs keep pausing AND my login keeps failing — two problems"*  
- **Expected:** `playback_issue` + escalate for complexity  
- **System:** Picks one intent, misses the other  
- **Frequency:** ~8% of golden set (secondary_intent populated)  
- **Hypothesis:** Single-label taxonomy forces an arbitrary choice  
- **Mitigation:** Add a multi-intent detection flag; escalate when secondary_intent confidence is high

### 2. False Historical Resolution
- **Example:** Agent retrieves a "try reinstalling" response and presents it as a known fix for account suspension  
- **Expected:** Escalate (account-specific, no clear resolution signal)  
- **System:** Auto-handles, drafts a reinstall suggestion  
- **Frequency:** ~6% of auto-handled cases  
- **Hypothesis:** Thread store has no resolution signal beyond "brand replied once"  
- **Mitigation:** Weight retrieval by `has_followup=False` (customer didn't need to reply again)

### 3. Entity-Sensitive Mismatch
- **Example:** iOS-specific fix applied to Android user; UK-specific plan advice given to US user  
- **Expected:** Escalate or caveat the reply  
- **System:** Returns confident reply with wrong platform/region  
- **Frequency:** ~5% of golden set  
- **Hypothesis:** Retriever finds lexically similar tweets but ignores entity context (OS, region, plan)  
- **Mitigation:** Add entity extraction step; filter retrieved examples by detected platform/region

### 4. Confident Unsupported Promise
- **Example:** Generated reply says *"your account will be restored within 48 hours"*  
- **Expected:** Escalate (account-specific, invented timeline)  
- **System:** Auto-handles with invented promise  
- **Frequency:** ~4% of auto-handled cases  
- **Hypothesis:** LLM tends to be helpful even when told not to invent timelines  
- **Mitigation:** Grounding check catches "48 hours" via forbidden phrases list; improve coverage

### 5. Implicit Context
- **Example:** *"@SpotifyCares still broken"* (refers to a deleted prior tweet)  
- **Expected:** `other_unclear`, escalate  
- **System:** Sometimes forced into a playback or login classification based on word overlap  
- **Frequency:** ~7% of golden set (ambiguity=3)  
- **Hypothesis:** No access to prior conversation context; single tweet is insufficient signal  
- **Mitigation:** `other_unclear` classifier should flag short, pronoun-heavy messages; enforce escalation

---

## What Is Misleading About My Headline Number?

Our **68% reply-acceptance rate** overstates real deployment quality for several reasons:

1. **Sampling bias in the golden set.** All 200 examples come from historical public Twitter conversations. This excludes: screenshots referenced in tweets, deleted context tweets, private DM transitions, and queries that were never publicly posted. Real traffic is harder.

2. **The LLM judge is not independent ground truth.** The same provider (OpenAI) generates replies and judges them. We measured a false-accept rate of ~6% vs. human reviewers — meaning the judge approves some replies that humans would reject as unsafe.

3. **Selective quality is purchased by lower coverage.** The 68% acceptance rate covers only the ~42% of cases the system auto-handles. The other ~58% are escalated. If you auto-handle everything, quality drops sharply.

4. **Historical replies are not proven resolutions.** We retrieved and grounded on brand responses that may not have resolved the original issue. The thread quality signal (`has_followup`) is a weak proxy.

5. **Thresholds were frozen, not tuned on the golden set.** This is good for honesty, but it means the system may not be at its optimal operating point.

6. **200 examples produce wide confidence intervals.** A 5-point macro F1 difference between systems is likely within the bootstrap 95% CI. Do not interpret small gaps as meaningful.

7. **Taxonomy subjectivity.** Intent labels for ambiguous messages depend on the annotator's interpretation of the taxonomy. Cohen's κ for intent reflects this — even careful labellers disagree ~20–30% of the time on boundary cases.

8. **Temporal policy drift.** Spotify's support policies and product features in 2017 (dataset period) differ from today. Historical replies may suggest outdated procedures.

---

## What I'd Do Next With One More Week

1. **True multi-label classification** — train a multi-label head and report per-label Average Precision (AP) to natively handle compound inquiries.
2. **Entity extraction & pre-filtering** — detect OS (iOS/Android/Windows/Mac), tier (Free/Premium), and region; use as hard filters during retrieval.
3. **Customer resolution signal weighting** — use `has_followup=False` and customer thank-you sentiment as a proxy; weight retrieval toward threads where the customer didn't need to reply again.
4. **Human annotation expansion** — annotate 50 more independent examples across edge cases to further improve inter-annotator kappa estimates.
5. **Embedding fine-tuning** — contrastively fine-tune the sentence-transformer on `(intent, tweet)` pairs from the pseudo-labelled training set.
6. **Adversarial red-teaming & test cases** — deliberately craft adversarial tweets that trick each failure mode and measure robustness against jailbreaks and prompt-injection.

---

## Limitations

- **Single-Turn Focus:** Does not manage multi-turn conversational state or authentication tokens.
- **Historical Drift:** Trained on 2017 Twitter data; UI flows and Spotify policy features may differ from modern apps.
- **Twitter-Specific Syntax:** Tailored to public short-text constraints (280 characters).

---

## Design Decisions

1. **Brand Specialization:** Restricted exclusively to `@SpotifyCares` to prevent tone and vocabulary contamination.
2. **Conversation-Level Splitting:** Prevented data leakage across turns.
3. **Chronological Cutoff:** Enforced time-based splits to reflect deployment reality.
4. **Decoupled Escalation Module:** Removed escalation decision logic from the LLM prompt to eliminate self-serving generation bias.
5. **Prioritizing Safety over Automation:** Optimized for lowest unsafe auto-handle rate rather than maximal automation volume.

---

## Decision Log

All 15 non-obvious engineering decisions, rationale, and consequences are fully documented in [`decision_log.md`](decision_log.md).

---

## Reproducibility

- **Fixed Random Seeds:** `seed = 42` across all scripts.
- **Deterministic Artifacts:** Pre-computed embeddings, trained models, frozen predictions, and judge outputs are versioned in `artifacts/`.
- **Zero-API Evaluation:** `make evaluate` runs offline in under 2 minutes.

---

## Environment & Installation

```
Python 3.10+
OPENAI_API_KEY=sk-...   (optional — only needed for live generation + judge)
```

See [`requirements.txt`](requirements.txt) for exact package versions.

### Installation

```bash
# Clone the repository
git clone https://github.com/bollepallideviharshini/SpotifyCares-AI-Support-Agent.git
cd SpotifyCares-AI-Support-Agent

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK corpora
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
```

### Environment Configuration

Create a `.env` file in the root directory (see [`.env.example`](.env.example)):

```bash
# LLM Provider Keys (Optional for offline evaluation, required for live generation)
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-gemini-api-key
```

### Dataset Setup

```bash
# Downloads and prepares SpotifyCares threads from Kaggle
python src/prepare_data.py

# Builds the multi-turn thread store
python src/build_threads.py
```

---

## Running the Agent

### Interactive / Single Query
```bash
python src/generate_reply.py --text "@SpotifyCares my music stops playing when my phone locks"
```

### Quick Demo (No API key required)
```bash
make demo
```

---

## Batch Processing

Run inference across the entire golden evaluation dataset and export frozen predictions:

```bash
python src/generate_reply.py --batch --golden-path data/golden_set.csv --out-path artifacts/predictions.jsonl
```

---

## Running the Evaluation

### Offline Evaluation (< 2 minutes, No API Key)
```bash
make evaluate
```

### Re-Run LLM-as-a-Judge Evaluation (Requires OpenAI Key)
```bash
make judge
```

---

## Repository Structure

```
SpotifyCares-AI-Support-Agent/
├── README.md                    ← Comprehensive project documentation and evaluation report
├── requirements.txt             ← Production and evaluation dependencies
├── Makefile                     ← Developer automation targets (demo, train, evaluate, judge, test)
├── decision_log.md              ← 15 non-obvious engineering decisions & trade-offs
├── conftest.py                  ← Pytest configuration
├── configs/
│   ├── experiment.yaml          ← All hyperparameters + paths + frozen thresholds
│   └── intent_taxonomy.yaml    ← 10 intents with full definitions and boundary guidelines
├── data/
│   ├── README.md               ← Sampling + labelling protocol
│   ├── golden_set.csv          ← 200 hand-labelled evaluation benchmark
│   └── sample_threads.jsonl    ← Generated reconstructed conversation threads
├── src/
│   ├── schemas.py              ← Strongly typed Pydantic models for all I/O
│   ├── prepare_data.py         ← Filter + normalize + sample dataset
│   ├── build_threads.py        ← Enrich threads with resolution signals
│   ├── train_intent.py         ← Train 3 classifiers + evaluate
│   ├── retrieve.py             ← BM25 + FAISS hybrid retriever with intent bonus
│   ├── generate_reply.py       ← Grounded LLM reply generator + full pipeline
│   ├── escalation.py           ← Rule-based deterministic escalation policy
│   └── evaluate.py             ← Full evaluation harness with bootstrap CIs
├── prompts/
│   ├── reply_prompt.txt        ← Frozen reply generation prompt template
│   └── judge_prompt.txt        ← LLM judge rubric prompt
├── tests/
│   ├── test_thread_building.py ← Multi-turn reconstruction unit tests
│   ├── test_leakage.py         ← Split & retrieval leakage validation tests
│   ├── test_escalation.py      ← Rule-based policy logic tests
│   └── test_output_schema.py   ← Pydantic schema validation tests
└── artifacts/
    ├── predictions.jsonl        ← Frozen system outputs
    ├── judge_scores.json        ← Frozen judge scores
    ├── metrics.json             ← Final evaluation benchmark metrics
    ├── confusion_matrix.png     ← Normalized intent confusion matrix
    ├── risk_coverage.png        ← Selective risk vs. coverage curve
    └── calibration.png          ← Probability calibration curve
```

---

## Testing

Execute the comprehensive test suite verifying schema integrity, escalation boundaries, thread building, and zero evaluation leakage:

```bash
make test
# Or directly via pytest
pytest tests/ -v --tb=short
```

---

## Submission Status

- **Status:** Complete & Verified
- **Deliverables:** Working pipeline, offline evaluation harness, frozen artifacts, unit tests, decision log, and comprehensive documentation.

---

## License

- **Code:** [MIT License](LICENSE)
- **Dataset:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) licensed under [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).
