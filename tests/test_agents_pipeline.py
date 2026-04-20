"""Unit tests for backend/agents_pipeline.py (CrewAI mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.agents_pipeline import (
    PriorityOutput,
    SentimentOutput,
    SummaryOutput,
    TopicOutput,
    run_agent_pipeline,
)


def _make_task_output(pydantic_model):
    """Build a mock TaskOutput whose .pydantic attribute returns the given model."""
    mock = MagicMock()
    mock.pydantic = pydantic_model
    return mock


def _fake_kickoff_side_effect(sentiment_task, topic_task, priority_task, summary_task):
    """Assign mock outputs to each task in place, simulating crew.kickoff()."""
    sentiment_task.output = _make_task_output(
        SentimentOutput(sentiment="negative", confidence=0.95)
    )
    topic_task.output = _make_task_output(TopicOutput(topic="billing_issue"))
    priority_task.output = _make_task_output(
        PriorityOutput(priority="high", reasoning="Unresolved billing issue with negative sentiment.")
    )
    summary_task.output = _make_task_output(
        SummaryOutput(
            summary="Customer was double-billed and issue is unresolved.",
            recommended_action="Refund duplicate charge immediately.",
            problem_resolved=False,
            needs_followup=True,
        )
    )


@pytest.mark.unit
def test_run_agent_pipeline_returns_legacy_shape(monkeypatch):
    """Pipeline returns dict with exactly the six required keys."""
    _tasks = {}

    class FakeCrew:
        def __init__(self, agents, tasks, **kwargs):
            _tasks["sentiment"] = tasks[0]
            _tasks["topic"] = tasks[1]
            _tasks["priority"] = tasks[2]
            _tasks["summary"] = tasks[3]

        def kickoff(self):
            _fake_kickoff_side_effect(
                _tasks["sentiment"],
                _tasks["topic"],
                _tasks["priority"],
                _tasks["summary"],
            )

    with patch("backend.agents_pipeline.Crew", FakeCrew):
        result = run_agent_pipeline("I was billed twice this month.")

    assert set(result.keys()) == {
        "customer_sentiment",
        "topic",
        "priority",
        "problem_resolved",
        "needs_followup",
        "summary",
    }


@pytest.mark.unit
def test_run_agent_pipeline_correct_values(monkeypatch):
    """Pipeline maps agent outputs to correct field values."""
    _tasks = {}

    class FakeCrew:
        def __init__(self, agents, tasks, **kwargs):
            _tasks["sentiment"] = tasks[0]
            _tasks["topic"] = tasks[1]
            _tasks["priority"] = tasks[2]
            _tasks["summary"] = tasks[3]

        def kickoff(self):
            _fake_kickoff_side_effect(
                _tasks["sentiment"],
                _tasks["topic"],
                _tasks["priority"],
                _tasks["summary"],
            )

    with patch("backend.agents_pipeline.Crew", FakeCrew):
        result = run_agent_pipeline("I was billed twice this month.")

    assert result["customer_sentiment"] == "negative"
    assert result["topic"] == "billing_issue"
    assert result["priority"] == "high"
    assert result["problem_resolved"] is False
    assert result["needs_followup"] is True
    assert "double-billed" in result["summary"]


@pytest.mark.unit
def test_run_agent_pipeline_truncates_long_summary():
    """Summary exceeding 120 chars is truncated and ends with '...'."""
    _tasks = {}

    class FakeCrew:
        def __init__(self, agents, tasks, **kwargs):
            _tasks["sentiment"] = tasks[0]
            _tasks["topic"] = tasks[1]
            _tasks["priority"] = tasks[2]
            _tasks["summary"] = tasks[3]

        def kickoff(self):
            _tasks["sentiment"].output = _make_task_output(
                SentimentOutput(sentiment="neutral", confidence=0.5)
            )
            _tasks["topic"].output = _make_task_output(TopicOutput(topic="general"))
            _tasks["priority"].output = _make_task_output(
                PriorityOutput(priority="low", reasoning="Low urgency.")
            )
            _tasks["summary"].output = _make_task_output(
                SummaryOutput(
                    summary="A" * 130,
                    recommended_action="No action.",
                    problem_resolved=True,
                    needs_followup=False,
                )
            )

    with patch("backend.agents_pipeline.Crew", FakeCrew):
        result = run_agent_pipeline("Some complaint.")

    assert len(result["summary"]) <= 120
    assert result["summary"].endswith("...")


@pytest.mark.unit
def test_run_agent_pipeline_raises_when_pydantic_output_none():
    """ValueError raised when any task returns no structured output."""
    _tasks = {}

    class FakeCrew:
        def __init__(self, agents, tasks, **kwargs):
            _tasks["sentiment"] = tasks[0]
            _tasks["topic"] = tasks[1]
            _tasks["priority"] = tasks[2]
            _tasks["summary"] = tasks[3]

        def kickoff(self):
            _tasks["sentiment"].output = _make_task_output(None)
            _tasks["topic"].output = _make_task_output(TopicOutput(topic="general"))
            _tasks["priority"].output = _make_task_output(
                PriorityOutput(priority="low", reasoning="ok")
            )
            _tasks["summary"].output = _make_task_output(
                SummaryOutput(
                    summary="ok",
                    recommended_action="ok",
                    problem_resolved=False,
                    needs_followup=False,
                )
            )

    with patch("backend.agents_pipeline.Crew", FakeCrew):
        with pytest.raises(ValueError, match="no structured output"):
            run_agent_pipeline("Some complaint.")
