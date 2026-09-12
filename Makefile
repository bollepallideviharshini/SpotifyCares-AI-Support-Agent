.PHONY: demo evaluate train judge test setup clean

# ─── Environment ──────────────────────────────────────────────────────────────
setup:
	python -m pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# ─── Data ─────────────────────────────────────────────────────────────────────
data:
	python src/prepare_data.py
	python src/build_threads.py

# ─── Training ─────────────────────────────────────────────────────────────────
train:
	python src/train_intent.py
	python src/retrieve.py --build-index

# ─── Demo ─────────────────────────────────────────────────────────────────────
# Runs the full agent on 5 curated test tweets and prints JSON.
# Works without API key in --offline mode (uses cached frozen replies).
demo:
	python src/generate_reply.py --demo

# ─── Evaluation ───────────────────────────────────────────────────────────────
# Produces metrics table from FROZEN predictions only — no API calls, no retraining.
# Runs in < 2 minutes on any machine.
evaluate:
	python src/evaluate.py --from-cache

# ─── LLM Judge ────────────────────────────────────────────────────────────────
# Re-runs the LLM judge on frozen replies (requires OPENAI_API_KEY).
judge:
	python src/evaluate.py --run-judge

# ─── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

# ─── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
