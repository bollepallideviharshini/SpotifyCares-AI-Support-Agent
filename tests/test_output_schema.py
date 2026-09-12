"""
test_output_schema.py — Tests that the AgentOutput schema is correctly populated
and that run_pipeline produces valid structured outputs.
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schemas import AgentOutput, INTENTS, IntentPrediction, RetrievedExample
from pydantic import ValidationError


class TestAgentOutputSchema:
    def test_valid_auto_handle(self):
        output = AgentOutput(
            tweet_id="t001",
            text="songs keep pausing",
            intent="playback_issue",
            intent_confidence=0.87,
            draft_reply="Hi! Try restarting the app.",
            resolution_examples=["t_ref_001", "t_ref_002"],
            action="auto_handle",
            reason="All checks passed.",
            retrieval_score=0.72,
            grounding_passed=True,
        )
        assert output.action == "auto_handle"
        assert output.intent == "playback_issue"
        assert output.intent_confidence == 0.87

    def test_valid_escalate(self):
        output = AgentOutput(
            tweet_id="t002",
            text="you charged me twice",
            intent="payment_charge",
            intent_confidence=0.91,
            draft_reply="Sorry about that.",
            resolution_examples=[],
            action="escalate",
            reason="High-risk intent: payment_charge.",
            safety_flags=["high_risk_intent"],
            retrieval_score=0.65,
            grounding_passed=True,
        )
        assert output.action == "escalate"
        assert "high_risk_intent" in output.safety_flags

    def test_invalid_intent_rejected(self):
        with pytest.raises(ValidationError):
            AgentOutput(
                tweet_id="t003",
                text="test",
                intent="made_up_intent",  # not in taxonomy
                intent_confidence=0.80,
                draft_reply="test reply",
                resolution_examples=[],
                action="auto_handle",
                reason="test",
            )

    def test_invalid_action_rejected(self):
        with pytest.raises(ValidationError):
            AgentOutput(
                tweet_id="t004",
                text="test",
                intent="playback_issue",
                intent_confidence=0.80,
                draft_reply="test reply",
                resolution_examples=[],
                action="maybe_handle",  # not a valid action
                reason="test",
            )

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            AgentOutput(
                tweet_id="t005",
                text="test",
                intent="playback_issue",
                intent_confidence=1.5,  # > 1.0
                draft_reply="test reply",
                resolution_examples=[],
                action="auto_handle",
                reason="test",
            )

    def test_json_serialization_round_trip(self):
        output = AgentOutput(
            tweet_id="t006",
            text="I can't log in",
            intent="login_account_access",
            intent_confidence=0.75,
            draft_reply="Hi! Try resetting your password.",
            resolution_examples=["r001"],
            action="auto_handle",
            reason="Checks passed.",
        )
        serialized = output.model_dump_json()
        parsed = json.loads(serialized)
        restored = AgentOutput(**parsed)
        assert restored.tweet_id == "t006"
        assert restored.intent == "login_account_access"

    def test_all_intents_are_valid(self):
        """Every intent in INTENTS must be accepted by the schema."""
        for intent in INTENTS:
            output = AgentOutput(
                tweet_id="t_test",
                text="test message",
                intent=intent,
                intent_confidence=0.8,
                draft_reply="test reply",
                resolution_examples=[],
                action="auto_handle",
                reason="test",
            )
            assert output.intent == intent


class TestIntentPredictionSchema:
    def test_valid_prediction(self):
        pred = IntentPrediction(
            intent="app_bug_crash",
            confidence=0.82,
            all_scores={i: 0.1 for i in INTENTS},
            classifier="tfidf_lr",
        )
        assert pred.intent == "app_bug_crash"

    def test_scores_must_roughly_sum_to_one(self):
        # This should pass — approximately sums to 1
        scores = {i: 1.0 / len(INTENTS) for i in INTENTS}
        pred = IntentPrediction(intent="other_unclear", confidence=0.5, all_scores=scores)
        assert pred is not None

    def test_scores_far_from_one_rejected(self):
        with pytest.raises((ValidationError, AssertionError)):
            scores = {i: 0.01 for i in INTENTS}  # sum << 1
            IntentPrediction(intent="other_unclear", confidence=0.5, all_scores=scores)


class TestRetrievedExampleSchema:
    def test_valid_example(self):
        ex = RetrievedExample(
            tweet_id="r001",
            conversation_id="c001",
            customer_text="songs keep stopping",
            brand_response="Hi! Try restarting the app.",
            score=0.75,
            intent="playback_issue",
            note="historical_response",
        )
        assert ex.note == "historical_response"
        assert ex.score == 0.75

    def test_note_is_historical_response_by_default(self):
        ex = RetrievedExample(
            tweet_id="r002",
            conversation_id="c002",
            customer_text="test",
            brand_response="test reply",
            score=0.5,
        )
        assert ex.note == "historical_response"
        # NEVER "proven_resolution"
        assert "proven" not in ex.note
