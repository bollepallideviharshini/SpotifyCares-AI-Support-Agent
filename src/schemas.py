"""
schemas.py — Pydantic data models for the SpotifyCares support agent pipeline.

Every module in the pipeline uses these models for structured I/O so that
individual components can be swapped or tested in isolation.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Intent taxonomy ──────────────────────────────────────────────────────────

INTENTS = [
    "playback_issue",
    "login_account_access",
    "subscription_cancellation",
    "payment_charge",
    "app_bug_crash",
    "feature_question",
    "content_availability",
    "account_security",
    "service_outage",
    "other_unclear",
]

IntentLabel = Literal[
    "playback_issue",
    "login_account_access",
    "subscription_cancellation",
    "payment_charge",
    "app_bug_crash",
    "feature_question",
    "content_availability",
    "account_security",
    "service_outage",
    "other_unclear",
]

ActionLabel = Literal["auto_handle", "escalate"]


# ─── Tweet / Thread models ─────────────────────────────────────────────────────

class RawTweet(BaseModel):
    tweet_id: str
    author_id: str
    inbound: bool                      # True = customer, False = brand
    created_at: str
    text: str
    response_tweet_id: Optional[str] = None
    in_response_to_tweet_id: Optional[str] = None


class Thread(BaseModel):
    """A reconstructed multi-turn conversation thread."""
    conversation_id: str
    tweet_ids: List[str]
    customer_messages: List[str]
    brand_responses: List[str]
    thread_quality: int = Field(
        ge=0,
        description="Number of brand turns in thread (proxy for completeness)",
    )
    has_followup: bool = Field(
        default=False,
        description="True if customer replied after brand response (possible resolution signal)",
    )
    created_at: Optional[str] = None
    searchable_text: str = Field(
        default="",
        description="Concatenated customer text used for retrieval indexing",
    )


# ─── Classifier outputs ────────────────────────────────────────────────────────

class IntentPrediction(BaseModel):
    intent: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    all_scores: dict[str, float] = Field(default_factory=dict)
    classifier: str = Field(
        default="tfidf_lr",
        description="Which classifier produced this prediction",
    )

    @field_validator("all_scores")
    @classmethod
    def scores_sum_to_one(cls, v: dict) -> dict:
        if v:
            total = sum(v.values())
            assert abs(total - 1.0) < 0.05, f"Scores should sum to ~1, got {total}"
        return v


# ─── Retrieval outputs ─────────────────────────────────────────────────────────

class RetrievedExample(BaseModel):
    tweet_id: str
    conversation_id: str
    customer_text: str
    brand_response: str
    score: float = Field(ge=0.0, description="Combined hybrid retrieval score")
    intent: Optional[str] = None
    note: str = Field(
        default="historical_response",
        description="Always 'historical_response' — never 'proven_resolution'",
    )


# ─── Agent output ──────────────────────────────────────────────────────────────

class AgentOutput(BaseModel):
    """The structured output produced by the full pipeline for one customer tweet."""

    tweet_id: str
    text: str
    intent: IntentLabel
    intent_confidence: float = Field(ge=0.0, le=1.0)
    draft_reply: str
    resolution_examples: List[str] = Field(
        description="Tweet IDs of retrieved historical examples used for grounding",
    )
    action: ActionLabel
    reason: str = Field(
        description="Human-readable explanation of why action was chosen",
    )
    safety_flags: List[str] = Field(
        default_factory=list,
        description="Any safety signals detected (PII, threats, forbidden phrases)",
    )
    retrieval_score: float = Field(
        default=0.0,
        description="Top retrieval score — low score contributes to escalation",
    )
    grounding_passed: bool = Field(
        default=True,
        description="False if reply contains forbidden phrases",
    )
    system: str = Field(
        default="proposed",
        description="Which system generated this output (for evaluation)",
    )


# ─── Evaluation models ─────────────────────────────────────────────────────────

class GoldenExample(BaseModel):
    tweet_id: str
    text: str
    intent: IntentLabel
    secondary_intent: Optional[str] = None
    reply_reference: Optional[str] = None
    should_escalate: bool
    escalation_reason: Optional[str] = None
    risk_tags: str = ""
    ambiguity: int = Field(ge=1, le=3, description="1=clear, 2=borderline, 3=ambiguous")
    annotator: str = "primary"
    notes: str = ""


class JudgeScore(BaseModel):
    """LLM-judge evaluation for a single draft reply."""
    tweet_id: str
    system: str
    relevance: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    helpfulness: int = Field(ge=1, le=5)
    brand_fit: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    escalation_consistency: int = Field(ge=1, le=5)
    acceptable_for_sending: bool
    rationale: str
    overall: float = Field(default=0.0)

    def model_post_init(self, __context):
        self.overall = round(
            (self.relevance + self.grounding + self.helpfulness
             + self.brand_fit + self.safety + self.escalation_consistency) / 6,
            2,
        )


class EvalRecord(BaseModel):
    """Combined prediction + ground-truth record for evaluation."""
    tweet_id: str
    text: str
    true_intent: IntentLabel
    true_escalate: bool
    pred_intent: IntentLabel
    pred_confidence: float
    pred_escalate: bool
    pred_action: ActionLabel
    draft_reply: str
    retrieval_score: float
    grounding_passed: bool
    safety_flags: List[str] = Field(default_factory=list)
    system: str = "proposed"
    judge_score: Optional[JudgeScore] = None
