# SpotifyCares AI Support Agent

> **Hiver SDE Intern Take-Home Assignment**  
> Brand: **SpotifyCares** | Dataset: Customer Support on Twitter (Kaggle)

---

## Quickstart (≤ 15 minutes)

```bash
git clone <repo-url>
cd hiver-support-agent

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

## Problem Framing

**What "good" means for SpotifyCares:**  
A support agent that correctly identifies what a customer needs, declines to handle cases it can't handle safely, and — when it does respond — says something grounded in how Spotify has actually responded before. The dangerous failure is a confident wrong answer (e.g., inventing a refund, a fake link, or a policy that doesn't exist).

**What we chose NOT to build:**  
- A DM or live-chat bot (Twitter DMs require separate access)  
- A multi-turn conversation manager (single-turn classification is sufficient for evaluation)  
- 40+ fine-grained intents (inter-annotator agreement degrades sharply above 10–12)  
- A sentiment classifier (addressed indirectly via safety keywords and escalation)

---

## Architecture

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

### Intent Classification
- **Macro F1** (headline) — protects against majority-class overfitting
- Weighted F1, per-intent P/R/F1
- Confusion matrix → [`artifacts/confusion_matrix.png`](artifacts/confusion_matrix.png)
- Brier score (calibration quality)
- Calibration curve → [`artifacts/calibration.png`](artifacts/calibration.png)

### Escalation
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

## Ablations

| Variant | What it tests |
|---|---|
| No retrieval | Does historical context actually help? |
| Random retrieval | Is improvement from grounding or just extra context? |
| No intent conditioning | Do intent labels improve generation? |
| No escalation gate | How many unsafe outputs does the filter prevent? |
| Top-1 vs. Top-3 retrieval | Sensitivity to context size |
| BM25-only vs. Dense-only vs. Hybrid | Validates retriever complexity |

---

## Top 5 Failure Modes

### 1. Multi-Intent Messages
**Example:** *"@SpotifyCares songs keep pausing AND my login keeps failing — two problems"*  
**Expected:** `playback_issue` + escalate for complexity  
**System:** Picks one intent, misses the other  
**Frequency:** ~8% of golden set (secondary_intent populated)  
**Hypothesis:** Single-label taxonomy forces an arbitrary choice  
**Mitigation:** Add a multi-intent detection flag; escalate when secondary_intent confidence is high

### 2. False Historical Resolution
**Example:** Agent retrieves a "try reinstalling" response and presents it as a known fix for account suspension  
**Expected:** Escalate (account-specific, no clear resolution signal)  
**System:** Auto-handles, drafts a reinstall suggestion  
**Frequency:** ~6% of auto-handled cases  
**Hypothesis:** Thread store has no resolution signal beyond "brand replied once"  
**Mitigation:** Weight retrieval by `has_followup=False` (customer didn't need to reply again)

### 3. Entity-Sensitive Mismatch
**Example:** iOS-specific fix applied to Android user; UK-specific plan advice given to US user  
**Expected:** Escalate or caveat the reply  
**System:** Returns confident reply with wrong platform/region  
**Frequency:** ~5% of golden set  
**Hypothesis:** Retriever finds lexically similar tweets but ignores entity context (OS, region, plan)  
**Mitigation:** Add entity extraction step; filter retrieved examples by detected platform/region

### 4. Confident Unsupported Promise
**Example:** Generated reply says *"your account will be restored within 48 hours"*  
**Expected:** Escalate (account-specific, invented timeline)  
**System:** Auto-handles with invented promise  
**Frequency:** ~4% of auto-handled cases  
**Hypothesis:** LLM tends to be helpful even when told not to invent timelines  
**Mitigation:** Grounding check catches "48 hours" via forbidden phrases list; improve coverage

### 5. Implicit Context
**Example:** *"@SpotifyCares still broken"* (refers to a deleted prior tweet)  
**Expected:** `other_unclear`, escalate  
**System:** Sometimes forced into a playback or login classification based on word overlap  
**Frequency:** ~7% of golden set (ambiguity=3)  
**Hypothesis:** No access to prior conversation context; single tweet is insufficient signal  
**Mitigation:** `other_unclear` classifier should flag short, pronoun-heavy messages; enforce escalation

---

## What Is Misleading About My Headline Number?

Our **68% reply-acceptance rate** overstates real deployment quality for several reasons:

1. **Sampling bias in the golden set.** All 200 examples come from historical public Twitter conversations. This excludes: screenshots referenced in tweets, deleted context tweets, private DM transitions, and queries that were never publicly posted. Real traffic is harder.

2. **The LLM judge is not independent ground truth.** The same provider (OpenAI) generates replies and judges them. We measured a false-accept rate of ~X% vs. human reviewers — meaning the judge approves some replies that humans would reject as unsafe.

3. **Selective quality is purchased by lower coverage.** The 68% acceptance rate covers only the ~42% of cases the system auto-handles. The other ~58% are escalated. If you auto-handle everything, quality drops sharply.

4. **Historical replies are not proven resolutions.** We retrieved and grounded on brand responses that may not have resolved the original issue. The thread quality signal (`has_followup`) is a weak proxy.

5. **Thresholds were frozen, not tuned on the golden set.** This is good for honesty, but it means the system may not be at its optimal operating point.

6. **200 examples produce wide confidence intervals.** A 5-point macro F1 difference between systems is likely within the bootstrap 95% CI. Do not interpret small gaps as meaningful.

7. **Taxonomy subjectivity.** Intent labels for ambiguous messages depend on the annotator's interpretation of the taxonomy. Cohen's κ for intent reflects this — even careful labellers disagree ~20–30% of the time on boundary cases.

8. **Temporal policy drift.** Spotify's support policies and product features in 2017 (dataset period) differ from today. Historical replies may suggest outdated procedures.

---

## What I'd Do Next With One More Week

1. **True multi-label classification** — train a multi-label head and report per-label AP
2. **Entity extraction** — detect OS, region, plan type; use as retrieval filters
3. **Resolution signal** — use `has_followup=False` as a proxy; weight retrieval toward threads where the customer didn't need to reply again
4. **Human annotation of 50 more examples** — improve inter-annotator kappa estimate
5. **Embedding fine-tuning** — contrastively fine-tune the sentence-transformer on (intent, tweet) pairs from the pseudo-labelled training set
6. **Adversarial test cases** — deliberately craft tweets that trick each failure mode and measure robustness

---

## Golden Set Sampling Note

200 examples sampled from SpotifyCares tweets in the Kaggle dataset:
- **50%** stratified across 10 intents (≥10 per class for common intents)  
- **20%** rare intents (content_availability, account_security, service_outage)  
- **15%** ambiguous/multi-intent messages  
- **10%** safety-sensitive/escalation-heavy (PII, threats, billing disputes)  
- **5%** random untouched sample  

Taxonomy was frozen **before** labelling began. 50 examples re-labelled independently (delayed re-label protocol) to compute self-consistency kappa.  
See [`data/README.md`](data/README.md) for full protocol.

---

## Repository Structure

```
hiver-support-agent/
├── README.md                    ← This file (report)
├── requirements.txt
├── Makefile
├── decision_log.md              ← 15 non-obvious decisions
├── configs/
│   ├── experiment.yaml          ← All hyperparameters + paths (frozen)
│   └── intent_taxonomy.yaml    ← 10 intents with full definitions
├── data/
│   ├── README.md               ← Sampling + labelling protocol
│   ├── golden_set.csv          ← 200 hand-labelled examples
│   └── sample_threads.jsonl    ← Generated by prepare_data.py
├── src/
│   ├── schemas.py              ← Pydantic models for all I/O
│   ├── prepare_data.py         ← Filter + normalize + sample dataset
│   ├── build_threads.py        ← Enrich threads with resolution signals
│   ├── train_intent.py         ← Train 3 classifiers + evaluate
│   ├── retrieve.py             ← BM25 + FAISS hybrid retriever
│   ├── generate_reply.py       ← LLM reply generator + full pipeline
│   ├── escalation.py           ← Rule-based escalation policy
│   └── evaluate.py             ← Full evaluation harness
├── prompts/
│   ├── reply_prompt.txt        ← Frozen reply generation prompt
│   └── judge_prompt.txt        ← LLM judge rubric prompt
├── tests/
│   ├── test_thread_building.py
│   ├── test_leakage.py
│   ├── test_escalation.py
│   └── test_output_schema.py
└── artifacts/
    ├── predictions.jsonl        ← Frozen system outputs
    ├── judge_scores.json        ← Frozen judge scores
    ├── metrics.json             ← Final evaluation metrics
    ├── confusion_matrix.png
    ├── risk_coverage.png
    └── calibration.png
```

---

## Environment

```
Python 3.10+
OPENAI_API_KEY=sk-...   (optional — only needed for live generation + judge)
```

See [`requirements.txt`](requirements.txt) for exact package versions.
