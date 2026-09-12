# Decision Log

15 non-obvious decisions made during this project, with rationale.

---

1. **Selected SpotifyCares as the single brand.**  
   Mixing brands would contaminate the tone, policy, and vocabulary of both the retrieval corpus and reply style constraints. SpotifyCares has high tweet volume, understandable intents, and lower stakes than banking or airlines — enabling more recoverable auto-handle decisions.

2. **Used conversation-level splitting, not tweet-level splitting.**  
   Splitting by individual tweet allows multiple turns from the same thread to appear in both train and test, causing leakage. A retriever could return the agent's own previous response as an "example." Conversation-level splitting by `conversation_id` prevents this entirely.

3. **Chose a chronological split (cutoff: 2017-09-01) over random split.**  
   Random splits overstate real-world performance because training data can contain messages posted *after* the test messages. A chronological split approximates the actual deployment condition where the agent answers future tweets using past history only.

4. **Derived the intent taxonomy from a 500-thread development sample, then froze it before labelling the golden set.**  
   Deriving intents after seeing every golden example would let the taxonomy overfit to the evaluation data. The taxonomy was frozen at the start of annotation and not changed during labelling.

5. **Retained an explicit `other_unclear` intent and treated it as a real class, not a catch-all.**  
   Discarding unclear messages from evaluation makes macro F1 look better than reality. A real deployment sees vague messages constantly. By including `other_unclear` as a first-class intent, the system learns to identify when it cannot help rather than hallucinating a wrong confident classification.

6. **Used a single primary intent + optional secondary intent per golden example.**  
   Forcing multi-label classification on 10 classes with 200 examples would produce a sparse, poorly-calibrated multi-label model. A single primary label with a recorded secondary intent captures the ambiguity for failure analysis without requiring a more complex model.

7. **Treated historical brand replies as "historical_response", never "proven_resolution".**  
   Twitter threads rarely show whether a customer's issue was actually fixed. The final brand tweet is evidence of what was said, not proof that it worked. Calling retrieved examples "resolutions" would be misleading to both the system and any human reviewer.

8. **Excluded evaluated threads from the retrieval index.**  
   If a test example's conversation ID existed in the FAISS/BM25 index, the retriever could return the test tweet itself or a sibling tweet as a "similar example." This would inflate retrieval quality and contaminate generation. The `exclude_conversation_ids` parameter enforces this at every inference call.

9. **Separated the escalation policy into a dedicated module (`escalation.py`), not embedded in the LLM prompt.**  
   An LLM asked to simultaneously generate a reply and decide escalation has a conflict of interest: it tends to always attempt a reply. A rule-based policy layer makes escalation auditable, reproducible, and independent of the LLM's confidence in its own output.

10. **Optimised for unsafe auto-handle rate, not raw automation rate.**  
    A system that escalates everything has 0% unsafe auto-handle rate but 0% automation value. The operationally meaningful constraint is: *among cases the system chooses to handle automatically, how many were genuinely unsafe?* Tuning thresholds to minimise unsafe auto-handle rate at acceptable coverage is more honest than maximising coverage.

11. **Validated hybrid retrieval against BM25-only and dense-only ablations.**  
    Claiming "hybrid retrieval improves reply quality" without measuring BM25-only and dense-only conditions is unverifiable. The ablation study also includes a random-retrieval control: if random examples perform nearly as well as nearest examples, the retriever is not actually helping grounding.

12. **Used a different model for the LLM judge than for reply generation (when possible).**  
    Using the same model as both generator and judge introduces self-consistency bias — the model rates its own outputs more favourably. The judge runs at temperature=0 with a structured JSON rubric, not a free-text rating, to reduce this bias.

13. **Froze all prompts and escalation thresholds before running the final golden-set evaluation.**  
    Tuning prompts or thresholds after seeing evaluation results would be an implicit form of test-set overfitting. All thresholds in `configs/experiment.yaml` were fixed before any golden-set metrics were computed.

14. **Added bootstrap 95% confidence intervals to all headline metrics.**  
    With only 200 test examples, a 5-point macro F1 difference is easily within sampling noise. Reporting raw point estimates without intervals is misleading. Bootstrap CIs make it honest that many differences between systems may not be statistically meaningful.

15. **Committed frozen predictions (`artifacts/predictions.jsonl`) and judge scores (`artifacts/judge_scores.json`) to the repository.**  
    Reproducibility requires that `make evaluate` produces the same metrics table without re-running LLM calls. Cached artifacts allow reviewers to verify results independently even if they lack an API key or if the LLM model has drifted since the submission.
