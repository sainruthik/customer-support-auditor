"""Complaint analyzer: OpenAI multi-agent pipeline with rule-based fallback."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from backend.agents_pipeline import run_agent_pipeline

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

REQUIRED_FIELDS = {
    "customer_sentiment",
    "topic",
    "priority",
    "problem_resolved",
    "needs_followup",
    "summary",
}

ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}
ALLOWED_TOPICS = {
    "billing_issue",
    "delayed_delivery",
    "damaged_product",
    "technical_support",
    "refund_request",
    "account_issue",
    "general",
}
ALLOWED_PRIORITIES = {"low", "medium", "high"}


def _rule_based_analyze(text: str) -> dict:
    """Rule-based backup analyzer used when pipeline output is invalid/unavailable."""
    normalized = text.lower().strip()

    topic_rules = {
        "billing_issue": [
            "bill",
            "billing",
            "charged",
            "charge",
            "invoice",
            "payment",
            "fee",
            "deducted",
            "billed",
            "card",
        ],
        "delayed_delivery": [
            "delivery",
            "delayed",
            "late",
            "shipping",
            "shipment",
            "tracking",
            "in transit",
            "not arrived",
            "arived",
            "trakcing",
            "oder",
        ],
        "damaged_product": ["damaged", "broken", "defective", "cracked", "not working", "not as described"],
        "refund_request": ["refund", "money back", "return my money", "reimburse", "canceled order", "cancelled order"],
        "account_issue": [
            "account",
            "acount",
            "login",
            "log in",
            "password",
            "locked out",
            "sign in",
            "verification",
            "two-factor",
        ],
    }

    topic = "general"
    for label, keywords in topic_rules.items():
        if any(keyword in normalized for keyword in keywords):
            topic = label
            break

    negative_words = [
        "angry",
        "upset",
        "frustrated",
        "terrible",
        "awful",
        "bad",
        "unacceptable",
        "disappointed",
        "worst",
        "hate",
        "ridiculous",
        "not okay",
        "tired of chasing",
        "scripted responses",
        "no real help",
    ]
    positive_words = [
        "thank you",
        "thanks",
        "appreciate",
        "grateful",
        "great service",
        "happy",
        "satisfied",
        "resolved quickly",
    ]

    has_negative = any(w in normalized for w in negative_words)
    has_positive = any(w in normalized for w in positive_words)

    if has_negative:
        customer_sentiment = "negative"
    elif has_positive:
        customer_sentiment = "positive"
    else:
        customer_sentiment = "neutral"

    unresolved_phrases = [
        "not resolved",
        "still waiting",
        "no response",
        "hasn't been fixed",
        "has not been fixed",
        "still not working",
        "issue remains",
    ]
    resolved_phrases = ["resolved", "fixed", "solved", "issue closed"]

    has_unresolved = any(p in normalized for p in unresolved_phrases)
    has_resolved = any(p in normalized for p in resolved_phrases)

    needs_followup = has_unresolved
    problem_resolved = has_resolved and not has_unresolved

    urgent_words = ["urgent", "asap", "immediately", "right now", "critical", "emergency", "need help asap"]
    has_urgent = any(w in normalized for w in urgent_words)

    if has_urgent or (has_unresolved and has_negative):
        priority = "high"
    elif problem_resolved and has_positive:
        priority = "low"
    else:
        priority = "medium"

    max_len = 120
    clean_text = " ".join(text.split())
    summary = clean_text if len(clean_text) <= max_len else clean_text[: max_len - 3].rstrip() + "..."

    return {
        "customer_sentiment": customer_sentiment,
        "topic": topic,
        "priority": priority,
        "problem_resolved": problem_resolved,
        "needs_followup": needs_followup,
        "summary": summary,
    }


def _normalize_and_validate(payload: dict[str, Any]) -> dict:
    """Validate required fields, enums, booleans, and summary text."""
    if not REQUIRED_FIELDS.issubset(payload.keys()):
        missing = REQUIRED_FIELDS - set(payload.keys())
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    customer_sentiment = str(payload["customer_sentiment"]).strip().lower()
    topic = str(payload["topic"]).strip().lower()
    priority = str(payload["priority"]).strip().lower()
    summary = str(payload["summary"]).strip()

    if customer_sentiment not in ALLOWED_SENTIMENTS:
        raise ValueError("customer_sentiment is invalid.")
    if topic not in ALLOWED_TOPICS:
        raise ValueError("topic is invalid.")
    if priority not in ALLOWED_PRIORITIES:
        raise ValueError("priority is invalid.")

    problem_resolved = payload["problem_resolved"]
    needs_followup = payload["needs_followup"]
    if not isinstance(problem_resolved, bool):
        raise ValueError("problem_resolved must be boolean.")
    if not isinstance(needs_followup, bool):
        raise ValueError("needs_followup must be boolean.")

    if not summary:
        raise ValueError("summary must be non-empty.")

    return {
        "customer_sentiment": customer_sentiment,
        "topic": topic,
        "priority": priority,
        "problem_resolved": problem_resolved,
        "needs_followup": needs_followup,
        "summary": summary,
    }


def _analyze_with_openai(text: str) -> dict:
    """Run OpenAI pipeline and validate output against the API contract."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")
    raw = run_agent_pipeline(text)
    return _normalize_and_validate(raw)


def analyze_complaint(text: str) -> dict:
    """
    Analyze complaint text using the OpenAI multi-agent pipeline.

    Fallback behavior — rule-based analyzer is used when:
    - OPENAI_API_KEY is absent
    - Pipeline raises any exception
    - Pipeline output fails validation
    """
    try:
        return _analyze_with_openai(text)
    except Exception:
        return _rule_based_analyze(text)
