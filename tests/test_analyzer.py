"""Unit tests for backend/analyzer.py."""

from __future__ import annotations

import pytest

from backend.analyzer import (
    _normalize_and_validate,
    _rule_based_analyze,
    analyze_complaint,
)


# ── _rule_based_analyze: topic detection ─────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected_topic",
    [
        ("I was billed twice this month", "billing_issue"),
        ("My invoice shows an incorrect charge", "billing_issue"),
        ("The delivery is late and still not arrived", "delayed_delivery"),
        ("Tracking says in transit for 3 weeks", "delayed_delivery"),
        ("The item arrived cracked and broken", "damaged_product"),
        ("Product is defective and not working", "damaged_product"),
        ("I want a refund for my canceled order", "refund_request"),
        ("Please return my money", "refund_request"),
        ("I cannot log in to my account", "account_issue"),
        ("My acount is locked out", "account_issue"),
        ("I have a general question about my order", "general"),
    ],
)
def test_rule_based_topic(text, expected_topic):
    result = _rule_based_analyze(text)
    assert result["topic"] == expected_topic


# ── _rule_based_analyze: sentiment detection ─────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected_sentiment",
    [
        ("This is terrible and unacceptable service", "negative"),
        ("I am frustrated and disappointed with this", "negative"),
        ("Thank you for resolving this so quickly", "positive"),
        ("I appreciate your great service", "positive"),
        ("I need to update my shipping address", "neutral"),
    ],
)
def test_rule_based_sentiment(text, expected_sentiment):
    result = _rule_based_analyze(text)
    assert result["customer_sentiment"] == expected_sentiment


# ── _rule_based_analyze: priority detection ───────────────────────────────────

@pytest.mark.unit
def test_rule_based_priority_high_urgent():
    result = _rule_based_analyze("I need help asap, this is urgent and still not working")
    assert result["priority"] == "high"


@pytest.mark.unit
def test_rule_based_priority_high_unresolved_negative():
    result = _rule_based_analyze("This is terrible and still waiting, issue not resolved")
    assert result["priority"] == "high"


@pytest.mark.unit
def test_rule_based_priority_low_resolved_positive():
    result = _rule_based_analyze("Issue resolved quickly, I appreciate your great service")
    assert result["priority"] == "low"


@pytest.mark.unit
def test_rule_based_priority_medium_default():
    result = _rule_based_analyze("Please update my shipping address to 123 Main St")
    assert result["priority"] == "medium"


# ── _rule_based_analyze: resolution flags ────────────────────────────────────

@pytest.mark.unit
def test_rule_based_resolved_true():
    result = _rule_based_analyze("The issue was resolved and fixed, thank you")
    assert result["problem_resolved"] is True
    assert result["needs_followup"] is False


@pytest.mark.unit
def test_rule_based_unresolved_sets_needs_followup():
    result = _rule_based_analyze("Still waiting for a response, not resolved yet")
    assert result["needs_followup"] is True
    assert result["problem_resolved"] is False


@pytest.mark.unit
def test_rule_based_unresolved_overrides_resolved():
    result = _rule_based_analyze("It was resolved but still waiting for confirmation")
    assert result["problem_resolved"] is False
    assert result["needs_followup"] is True


# ── _rule_based_analyze: summary truncation ───────────────────────────────────

@pytest.mark.unit
def test_rule_based_summary_short_text_unchanged():
    text = "Short complaint text."
    result = _rule_based_analyze(text)
    assert result["summary"] == text


@pytest.mark.unit
def test_rule_based_summary_truncated_at_120():
    text = "A" * 130
    result = _rule_based_analyze(text)
    assert len(result["summary"]) <= 120
    assert result["summary"].endswith("...")


# ── _normalize_and_validate ───────────────────────────────────────────────────

VALID_PAYLOAD = {
    "customer_sentiment": "negative",
    "topic": "billing_issue",
    "priority": "high",
    "problem_resolved": False,
    "needs_followup": True,
    "summary": "Customer was double-billed.",
}


@pytest.mark.unit
def test_normalize_happy_path():
    result = _normalize_and_validate(VALID_PAYLOAD)
    assert result["customer_sentiment"] == "negative"
    assert result["topic"] == "billing_issue"
    assert result["priority"] == "high"
    assert result["problem_resolved"] is False
    assert result["needs_followup"] is True


@pytest.mark.unit
def test_normalize_missing_field_raises():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "summary"}
    with pytest.raises(ValueError, match="Missing required fields"):
        _normalize_and_validate(payload)


@pytest.mark.unit
def test_normalize_invalid_sentiment_raises():
    payload = {**VALID_PAYLOAD, "customer_sentiment": "angry"}
    with pytest.raises(ValueError, match="customer_sentiment is invalid"):
        _normalize_and_validate(payload)


@pytest.mark.unit
def test_normalize_invalid_topic_raises():
    payload = {**VALID_PAYLOAD, "topic": "unknown_topic"}
    with pytest.raises(ValueError, match="topic is invalid"):
        _normalize_and_validate(payload)


@pytest.mark.unit
def test_normalize_invalid_priority_raises():
    payload = {**VALID_PAYLOAD, "priority": "critical"}
    with pytest.raises(ValueError, match="priority is invalid"):
        _normalize_and_validate(payload)


@pytest.mark.unit
def test_normalize_non_bool_problem_resolved_raises():
    payload = {**VALID_PAYLOAD, "problem_resolved": "false"}
    with pytest.raises(ValueError, match="problem_resolved must be boolean"):
        _normalize_and_validate(payload)


@pytest.mark.unit
def test_normalize_non_bool_needs_followup_raises():
    payload = {**VALID_PAYLOAD, "needs_followup": 0}
    with pytest.raises(ValueError, match="needs_followup must be boolean"):
        _normalize_and_validate(payload)


@pytest.mark.unit
def test_normalize_empty_summary_raises():
    payload = {**VALID_PAYLOAD, "summary": "   "}
    with pytest.raises(ValueError, match="summary must be non-empty"):
        _normalize_and_validate(payload)


@pytest.mark.unit
def test_normalize_strips_whitespace_and_lowercases():
    payload = {**VALID_PAYLOAD, "customer_sentiment": "  Negative  ", "topic": "  Billing_Issue  ", "priority": "HIGH"}
    result = _normalize_and_validate(payload)
    assert result["customer_sentiment"] == "negative"
    assert result["topic"] == "billing_issue"
    assert result["priority"] == "high"


@pytest.mark.unit
def test_normalize_accepts_technical_support_topic():
    payload = {**VALID_PAYLOAD, "topic": "technical_support"}
    result = _normalize_and_validate(payload)
    assert result["topic"] == "technical_support"


# ── analyze_complaint: fallback behavior ─────────────────────────────────────

@pytest.mark.unit
def test_analyze_complaint_no_api_key_uses_fallback(monkeypatch):
    import backend.analyzer as mod
    monkeypatch.setattr(mod, "OPENAI_API_KEY", "")
    text = "My billing is wrong"
    result = analyze_complaint(text)
    expected = _rule_based_analyze(text)
    assert result == expected


@pytest.mark.unit
def test_analyze_complaint_crew_raises_uses_fallback(monkeypatch):
    import backend.analyzer as mod
    monkeypatch.setattr(mod, "OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr(mod, "run_agent_pipeline", lambda _: (_ for _ in ()).throw(RuntimeError("crew failed")))
    text = "My billing is wrong"
    result = analyze_complaint(text)
    expected = _rule_based_analyze(text)
    assert result == expected


@pytest.mark.unit
def test_analyze_complaint_crew_success(monkeypatch):
    import backend.analyzer as mod
    monkeypatch.setattr(mod, "OPENAI_API_KEY", "sk-fake")
    fake_output = {
        "customer_sentiment": "negative",
        "topic": "billing_issue",
        "priority": "high",
        "problem_resolved": False,
        "needs_followup": True,
        "summary": "Customer was charged twice.",
    }
    monkeypatch.setattr(mod, "run_agent_pipeline", lambda _: fake_output)
    result = analyze_complaint("I was charged twice")
    assert result == fake_output


@pytest.mark.unit
def test_analyze_complaint_output_has_required_fields(monkeypatch):
    import backend.analyzer as mod
    monkeypatch.setattr(mod, "OPENAI_API_KEY", "")
    result = analyze_complaint("Some complaint text")
    from backend.analyzer import REQUIRED_FIELDS
    assert REQUIRED_FIELDS.issubset(result.keys())
