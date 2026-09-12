"""
test_escalation.py — Unit tests for the escalation policy module.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from escalation import EscalationPolicy

# Minimal config matching experiment.yaml escalation section
ESCALATION_CFG = {
    "min_intent_confidence": 0.55,
    "min_retrieval_score": 0.30,
    "high_risk_intents": ["payment_charge", "account_security"],
    "pii_patterns": [
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    ],
    "safety_keywords": ["sue", "lawyer", "kill", "hack", "stolen"],
    "forbidden_reply_phrases": ["I guarantee", "your refund will", "http", "www."],
}


@pytest.fixture
def policy():
    return EscalationPolicy(config=ESCALATION_CFG)


class TestEscalationPolicy:
    def test_low_confidence_escalates(self, policy):
        action, reason, flags = policy.decide(
            intent="playback_issue",
            confidence=0.40,  # below 0.55 threshold
            retrieval_score=0.70,
            reply_text="Hi! Try restarting the app.",
            customer_text="songs keep stopping",
        )
        assert action == "escalate"
        assert "low_confidence" in flags

    def test_high_confidence_auto_handles(self, policy):
        action, reason, flags = policy.decide(
            intent="playback_issue",
            confidence=0.88,
            retrieval_score=0.72,
            reply_text="Hi! Try logging out and back in.",
            customer_text="songs keep stopping",
        )
        assert action == "auto_handle"
        assert flags == []

    def test_payment_intent_always_escalates(self, policy):
        action, reason, flags = policy.decide(
            intent="payment_charge",
            confidence=0.95,  # even with high confidence
            retrieval_score=0.80,
            reply_text="Sorry about the charge.",
            customer_text="you charged me twice",
        )
        assert action == "escalate"
        assert "high_risk_intent" in flags

    def test_account_security_escalates(self, policy):
        action, reason, flags = policy.decide(
            intent="account_security",
            confidence=0.91,
            retrieval_score=0.75,
            reply_text="Please change your password.",
            customer_text="someone hacked my account",
        )
        assert action == "escalate"
        assert "high_risk_intent" in flags

    def test_pii_email_escalates(self, policy):
        action, reason, flags = policy.decide(
            intent="login_account_access",
            confidence=0.80,
            retrieval_score=0.65,
            reply_text="Hi! Try resetting your password.",
            customer_text="my email is john.doe@example.com I can't login",
        )
        assert action == "escalate"
        assert "pii" in flags

    def test_pii_phone_escalates(self, policy):
        action, reason, flags = policy.decide(
            intent="login_account_access",
            confidence=0.80,
            retrieval_score=0.65,
            reply_text="Hi! Try resetting your password.",
            customer_text="call me at 555-123-4567 to help me",
        )
        assert action == "escalate"
        assert "pii" in flags

    def test_safety_keyword_escalates(self, policy):
        action, reason, flags = policy.decide(
            intent="other_unclear",
            confidence=0.75,
            retrieval_score=0.60,
            reply_text="Hi, we're sorry to hear that.",
            customer_text="I will sue spotify if this isn't fixed",
        )
        assert action == "escalate"
        assert "safety_keywords" in flags

    def test_weak_retrieval_escalates(self, policy):
        action, reason, flags = policy.decide(
            intent="feature_question",
            confidence=0.80,
            retrieval_score=0.15,  # below 0.30 threshold
            reply_text="Hi! Here's how to do that.",
            customer_text="how do I share a playlist",
        )
        assert action == "escalate"
        assert "weak_retrieval" in flags

    def test_forbidden_phrase_in_reply_escalates(self, policy):
        action, reason, flags = policy.decide(
            intent="subscription_cancellation",
            confidence=0.82,
            retrieval_score=0.68,
            reply_text="I guarantee your refund will arrive soon.",  # forbidden
            customer_text="I want to cancel my subscription",
        )
        assert action == "escalate"
        assert "forbidden_phrase" in flags

    def test_other_unclear_escalates(self, policy):
        action, reason, flags = policy.decide(
            intent="other_unclear",
            confidence=0.60,
            retrieval_score=0.50,
            reply_text="Hi! Can you share more details?",
            customer_text="lol",
        )
        assert action == "escalate"
        assert "other_unclear" in flags

    def test_grounding_check_detects_url(self, policy):
        bad_reply = "Please visit http://fake-spotify.com/help for assistance"
        assert not policy.is_grounding_passed(bad_reply)

    def test_grounding_check_clean_reply(self, policy):
        good_reply = "Hi! Try logging out and back in. Let us know if that helps!"
        assert policy.is_grounding_passed(good_reply)

    def test_reason_is_non_empty_on_escalate(self, policy):
        action, reason, flags = policy.decide(
            intent="payment_charge",
            confidence=0.90,
            retrieval_score=0.70,
            reply_text="Good reply.",
            customer_text="charged twice",
        )
        assert len(reason) > 10

    def test_reason_mentions_auto_handle_when_passing(self, policy):
        action, reason, flags = policy.decide(
            intent="playback_issue",
            confidence=0.90,
            retrieval_score=0.70,
            reply_text="Hi! Try restarting the app.",
            customer_text="songs stop playing",
        )
        assert action == "auto_handle"
        assert "passed" in reason.lower() or "check" in reason.lower()
