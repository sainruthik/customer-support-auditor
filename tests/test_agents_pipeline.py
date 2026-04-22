"""Unit tests for backend/agents_pipeline.py (OpenAI client mocked)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.agents_pipeline import (
    _priority_agent,
    _sentiment_agent,
    _summary_agent,
    _topic_agent,
    run_agent_pipeline,
)


def _mock_client(*json_payloads: dict) -> MagicMock:
    """Return a mock OpenAI client whose chat.completions.create yields
    sequential canned JSON strings, one per call."""
    client = MagicMock()
    responses = iter(json.dumps(p) for p in json_payloads)

    def _create(**_kwargs):
        msg = MagicMock()
        msg.choices = [MagicMock()]
        msg.choices[0].message.content = next(responses)
        return msg

    client.chat.completions.create.side_effect = _create
    return client


# ── _sentiment_agent ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_sentiment_agent_parses_valid_json():
    client = _mock_client({"sentiment": "negative", "confidence": 0.9})
    result = _sentiment_agent(client, "I was billed twice.")
    assert result == {"sentiment": "negative", "confidence": 0.9}


@pytest.mark.unit
def test_sentiment_agent_accepts_all_valid_labels():
    for label in ("positive", "neutral", "negative"):
        client = _mock_client({"sentiment": label, "confidence": 0.8})
        result = _sentiment_agent(client, "some complaint")
        assert result["sentiment"] == label


@pytest.mark.unit
def test_sentiment_agent_rejects_invalid_label():
    client = _mock_client({"sentiment": "angry", "confidence": 0.5})
    with pytest.raises(ValueError, match="sentiment agent returned invalid label"):
        _sentiment_agent(client, "whatever")


# ── _topic_agent ──────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_topic_agent_accepts_known_label():
    client = _mock_client({"topic": "billing_issue"})
    result = _topic_agent(client, "I was overcharged.")
    assert result["topic"] == "billing_issue"


@pytest.mark.unit
def test_topic_agent_accepts_technical_support():
    client = _mock_client({"topic": "technical_support"})
    result = _topic_agent(client, "App keeps crashing.")
    assert result["topic"] == "technical_support"


@pytest.mark.unit
def test_topic_agent_rejects_unknown_label():
    client = _mock_client({"topic": "spaceship"})
    with pytest.raises(ValueError, match="topic agent returned invalid label"):
        _topic_agent(client, "whatever")


# ── _priority_agent ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_priority_agent_accepts_valid_priority():
    client = _mock_client({"priority": "high", "reasoning": "Urgent billing."})
    result = _priority_agent(client, "x", "negative", "billing_issue")
    assert result["priority"] == "high"


@pytest.mark.unit
def test_priority_agent_rejects_invalid_priority():
    client = _mock_client({"priority": "urgent", "reasoning": "nope"})
    with pytest.raises(ValueError, match="priority agent returned invalid label"):
        _priority_agent(client, "x", "negative", "billing_issue")


# ── _summary_agent ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_summary_agent_returns_structured_payload():
    client = _mock_client({
        "summary": "Customer double-billed; unresolved.",
        "recommended_action": "Refund.",
        "problem_resolved": False,
        "needs_followup": True,
    })
    result = _summary_agent(client, "x", "negative", "billing_issue", "high")
    assert result["problem_resolved"] is False
    assert result["needs_followup"] is True


@pytest.mark.unit
def test_summary_agent_rejects_string_resolved_flag():
    client = _mock_client({
        "summary": "ok",
        "recommended_action": "ok",
        "problem_resolved": "false",
        "needs_followup": True,
    })
    with pytest.raises(ValueError, match="problem_resolved"):
        _summary_agent(client, "x", "negative", "billing_issue", "high")


@pytest.mark.unit
def test_summary_agent_rejects_string_followup_flag():
    client = _mock_client({
        "summary": "ok",
        "recommended_action": "ok",
        "problem_resolved": False,
        "needs_followup": "yes",
    })
    with pytest.raises(ValueError, match="needs_followup"):
        _summary_agent(client, "x", "negative", "billing_issue", "high")


# ── run_agent_pipeline ────────────────────────────────────────────────────────

@pytest.mark.unit
def test_run_agent_pipeline_returns_legacy_shape():
    fake_client = _mock_client(
        {"sentiment": "negative", "confidence": 0.95},
        {"topic": "billing_issue"},
        {"priority": "high", "reasoning": "Unresolved billing."},
        {
            "summary": "Customer was double-billed and issue is unresolved.",
            "recommended_action": "Refund duplicate charge.",
            "problem_resolved": False,
            "needs_followup": True,
        },
    )
    with patch("backend.agents_pipeline.OpenAI", return_value=fake_client):
        result = run_agent_pipeline("I was billed twice this month.")

    assert set(result.keys()) == {
        "customer_sentiment", "topic", "priority",
        "problem_resolved", "needs_followup", "summary",
    }


@pytest.mark.unit
def test_run_agent_pipeline_correct_values():
    fake_client = _mock_client(
        {"sentiment": "negative", "confidence": 0.95},
        {"topic": "billing_issue"},
        {"priority": "high", "reasoning": "Unresolved billing."},
        {
            "summary": "Customer was double-billed and issue is unresolved.",
            "recommended_action": "Refund duplicate charge.",
            "problem_resolved": False,
            "needs_followup": True,
        },
    )
    with patch("backend.agents_pipeline.OpenAI", return_value=fake_client):
        result = run_agent_pipeline("I was billed twice this month.")

    assert result["customer_sentiment"] == "negative"
    assert result["topic"] == "billing_issue"
    assert result["priority"] == "high"
    assert result["problem_resolved"] is False
    assert result["needs_followup"] is True
    assert "double-billed" in result["summary"]


@pytest.mark.unit
def test_run_agent_pipeline_truncates_long_summary():
    fake_client = _mock_client(
        {"sentiment": "neutral", "confidence": 0.5},
        {"topic": "general"},
        {"priority": "low", "reasoning": "Low urgency."},
        {
            "summary": "A" * 130,
            "recommended_action": "No action.",
            "problem_resolved": True,
            "needs_followup": False,
        },
    )
    with patch("backend.agents_pipeline.OpenAI", return_value=fake_client):
        result = run_agent_pipeline("Some complaint.")

    assert len(result["summary"]) <= 120
    assert result["summary"].endswith("...")


@pytest.mark.unit
def test_run_agent_pipeline_propagates_openai_errors():
    """Pipeline surfaces SDK errors — analyze_complaint handles fallback."""
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("api down")

    with patch("backend.agents_pipeline.OpenAI", return_value=fake_client):
        with pytest.raises(RuntimeError, match="api down"):
            run_agent_pipeline("any text")
