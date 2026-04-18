"""Complaint analyzer with OpenAI-first logic and rule-based fallback."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

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
    "refund_request",
    "account_issue",
    "general",
}
ALLOWED_PRIORITIES = {"low", "medium", "high"}


def _rule_based_analyze(text: str) -> dict:
    """Rule-based backup analyzer used when LLM output is invalid/unavailable."""
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


def _extract_json_block(raw_text: str) -> str:
    """Extract JSON object text from a model response."""
    raw_text = raw_text.strip()
    if raw_text.startswith("{") and raw_text.endswith("}"):
        return raw_text

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return raw_text[start : end + 1]


def _normalize_and_validate(payload: dict[str, Any]) -> dict:
    """Validate required fields, enums, booleans, and summary text."""
    if not REQUIRED_FIELDS.issubset(payload.keys()):
        missing = REQUIRED_FIELDS - set(payload.keys())
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    customer_sentiment = str(payload["customer_sentiment"]).strip().lower()
    topic = str(payload["topic"]).strip().lower()
    priority = str(payload["priority"]).strip().lower()
    summary = str(payload["summary"]).strip()

    # Enforce enum values exactly as expected by the API contract.
    if customer_sentiment not in ALLOWED_SENTIMENTS:
        raise ValueError("customer_sentiment is invalid.")
    if topic not in ALLOWED_TOPICS:
        raise ValueError("topic is invalid.")
    if priority not in ALLOWED_PRIORITIES:
        raise ValueError("priority is invalid.")

    # LLM output must contain true booleans for these flags.
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


def _extract_text_from_openai_response(payload: dict[str, Any]) -> str:
    """Extract generated text from OpenAI Responses API JSON payload."""
    direct_text = payload.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    output_items = payload.get("output", [])
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content", [])
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                text_value = content.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    return text_value.strip()

    raise ValueError("Could not extract model text from OpenAI response.")


def _analyze_with_openai(text: str) -> dict:
    """Call OpenAI Responses API and return validated structured analysis."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")

    # Prompt forces a strict JSON object so parsing is deterministic.
    prompt = f"""
You are a customer support complaint classifier.
Return strict JSON only. No markdown. No explanation.

Required JSON schema:
{{
  "customer_sentiment": "negative|neutral|positive",
  "topic": "billing_issue|delayed_delivery|damaged_product|refund_request|account_issue|general",
  "priority": "low|medium|high",
  "problem_resolved": true or false,
  "needs_followup": true or false,
  "summary": "short summary (max 120 chars)"
}}

Complaint text:
\"\"\"{text}\"\"\"
""".strip()

    response = httpx.post(
        f"{OPENAI_BASE_URL}/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "input": prompt,
            "temperature": 0,
        },
        timeout=20,
    )
    response.raise_for_status()

    # Parse model text, then extract/validate the JSON object.
    wrapper = response.json()
    model_text = _extract_text_from_openai_response(wrapper)
    parsed = json.loads(_extract_json_block(model_text))
    return _normalize_and_validate(parsed)


def analyze_complaint(text: str) -> dict:
    """
    Analyze complaint text using OpenAI structured extraction.

    Fallback behavior:
    - If OpenAI is unavailable
    - If response is not valid JSON
    - If required fields are missing/invalid
    then use the local rule-based analyzer.
    """
    try:
        return _analyze_with_openai(text)
    except Exception:
        # Keep the API stable even when model/network/JSON parsing fails.
        return _rule_based_analyze(text)
