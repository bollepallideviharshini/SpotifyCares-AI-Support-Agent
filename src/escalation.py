"""
escalation.py — Explicit, rule-based escalation policy layer.

This module is intentionally SEPARATE from the LLM reply generator.
No single unrestricted LLM call makes the escalation decision.

Rules are evaluated in order; the first match sets the action to "escalate".
Each rule also provides a human-readable reason.

Usage:
    from src.escalation import EscalationPolicy
    policy = EscalationPolicy(cfg["escalation"])
    action, reason, flags = policy.decide(
        intent, confidence, retrieval_score, reply_text, customer_text
    )
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

log = logging.getLogger(__name__)

ActionLabel = Literal["auto_handle", "escalate"]


@dataclass
class EscalationRule:
    name: str
    description: str
    reason: str  # surfaced to the user/agent


@dataclass
class EscalationPolicy:
    """
    Applies a deterministic set of escalation rules.

    All thresholds come from configs/experiment.yaml → escalation section.
    Thresholds were frozen before golden-set evaluation.
    """

    config: dict

    # Compiled regex patterns (initialised in __post_init__)
    _pii_patterns: list[re.Pattern] = field(default_factory=list, init=False)
    _safety_keywords: list[str] = field(default_factory=list, init=False)
    _forbidden_phrases: list[str] = field(default_factory=list, init=False)
    _high_risk_intents: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        cfg = self.config
        self._pii_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in cfg.get("pii_patterns", [])
        ]
        self._safety_keywords = [
            kw.strip().lower()
            for kw in cfg.get("safety_keywords", [])
        ]
        self._forbidden_phrases = [
            p.strip().lower()
            for p in cfg.get("forbidden_reply_phrases", [])
        ]
        self._high_risk_intents = set(cfg.get("high_risk_intents", []))

    # ─── Individual rule checks ───────────────────────────────────────────

    def _check_confidence(self, confidence: float) -> Optional[str]:
        threshold = self.config.get("min_intent_confidence", 0.55)
        if confidence < threshold:
            return (
                f"Intent confidence {confidence:.2f} is below threshold {threshold:.2f}. "
                "Classification is uncertain — human review needed."
            )
        return None

    def _check_high_risk_intent(self, intent: str) -> Optional[str]:
        if intent in self._high_risk_intents:
            return (
                f"Intent '{intent}' is in the high-risk category "
                "(payment/security). Requires human verification."
            )
        return None

    def _check_retrieval_score(self, retrieval_score: float) -> Optional[str]:
        threshold = self.config.get("min_retrieval_score", 0.30)
        if retrieval_score < threshold:
            return (
                f"Retrieval score {retrieval_score:.2f} is below threshold {threshold:.2f}. "
                "No sufficiently similar historical example found — reply may lack grounding."
            )
        return None

    def _check_pii(self, text: str) -> Optional[str]:
        for pattern in self._pii_patterns:
            if pattern.search(text):
                return (
                    "Personally identifiable information detected in customer message. "
                    "Must be handled by a human agent through a private channel."
                )
        return None

    def _check_safety_keywords(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for kw in self._safety_keywords:
            if kw in text_lower:
                return (
                    f"Safety-sensitive keyword '{kw}' detected. "
                    "Requires immediate human review."
                )
        return None

    def _check_forbidden_reply_phrases(self, reply_text: str) -> Optional[str]:
        reply_lower = reply_text.lower()
        for phrase in self._forbidden_phrases:
            if phrase in reply_lower:
                return (
                    f"Draft reply contains forbidden phrase '{phrase}'. "
                    "Reply failed grounding check — escalating for safety."
                )
        return None

    def _check_other_unclear(self, intent: str) -> Optional[str]:
        if intent == "other_unclear":
            return (
                "Intent classified as 'other_unclear'. "
                "Message is too ambiguous to auto-handle safely."
            )
        return None

    # ─── Main decision method ──────────────────────────────────────────────

    def decide(
        self,
        intent: str,
        confidence: float,
        retrieval_score: float,
        reply_text: str,
        customer_text: str,
    ) -> tuple[ActionLabel, str, list[str]]:
        """
        Evaluate all escalation rules in priority order.

        Returns:
            action: "auto_handle" or "escalate"
            reason: human-readable explanation
            flags:  list of triggered rule names (for logging/auditing)
        """
        flags: list[str] = []

        # Rules ordered from highest-priority (safety) to lowest.
        checks = [
            ("pii",              self._check_pii(customer_text)),
            ("safety_keywords",  self._check_safety_keywords(customer_text)),
            ("high_risk_intent", self._check_high_risk_intent(intent)),
            ("other_unclear",    self._check_other_unclear(intent)),
            ("low_confidence",   self._check_confidence(confidence)),
            ("weak_retrieval",   self._check_retrieval_score(retrieval_score)),
            ("forbidden_phrase", self._check_forbidden_reply_phrases(reply_text)),
        ]

        triggered_reason: Optional[str] = None
        for rule_name, reason in checks:
            if reason is not None:
                flags.append(rule_name)
                if triggered_reason is None:
                    # First triggered rule wins for the primary reason
                    triggered_reason = reason

        if triggered_reason:
            log.debug(f"Escalating ({', '.join(flags)}): {triggered_reason[:60]}")
            return "escalate", triggered_reason, flags

        return (
            "auto_handle",
            "All confidence, retrieval, safety, and grounding checks passed.",
            [],
        )

    def is_grounding_passed(self, reply_text: str) -> bool:
        """Returns False if any forbidden phrase is found in the reply."""
        reply_lower = reply_text.lower()
        return not any(p in reply_lower for p in self._forbidden_phrases)
