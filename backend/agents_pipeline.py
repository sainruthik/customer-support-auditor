"""CrewAI sequential multi-agent pipeline for complaint analysis."""

from __future__ import annotations

import os
from typing import Literal

from crewai import Agent, Crew, LLM, Process, Task
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


class SentimentOutput(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)


class TopicOutput(BaseModel):
    topic: Literal[
        "billing_issue",
        "delayed_delivery",
        "damaged_product",
        "technical_support",
        "refund_request",
        "account_issue",
        "general",
    ]


class PriorityOutput(BaseModel):
    priority: Literal["low", "medium", "high"]
    reasoning: str


class SummaryOutput(BaseModel):
    summary: str
    recommended_action: str
    problem_resolved: bool
    needs_followup: bool


def run_agent_pipeline(text: str) -> dict:
    """Run 4-agent sequential pipeline and return legacy output dict."""
    llm = LLM(model=_MODEL, temperature=0, max_tokens=512)

    sentiment_agent = Agent(
        role="Sentiment Analyst",
        goal="Determine the emotional tone of a customer complaint.",
        backstory="Expert at reading emotional cues in customer support messages.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    topic_agent = Agent(
        role="Topic Specialist",
        goal="Classify the main issue category of a customer complaint.",
        backstory="Expert at routing customer complaints to the right support queue.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    priority_agent = Agent(
        role="Priority Assessor",
        goal="Assess the urgency of a complaint given its sentiment and topic.",
        backstory="Expert triage agent for customer support tickets.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    summary_agent = Agent(
        role="Report Writer",
        goal="Write a concise actionable summary and determine resolution status.",
        backstory="Expert at distilling support tickets into clear, actionable briefs.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    sentiment_task = Task(
        description=(
            "Analyze the sentiment of the following customer complaint.\n\n"
            f"Complaint:\n{text}"
        ),
        expected_output="Sentiment (positive/negative/neutral) and confidence score 0-1.",
        agent=sentiment_agent,
        output_pydantic=SentimentOutput,
    )
    topic_task = Task(
        description=(
            "Classify the topic of the following customer complaint.\n\n"
            f"Complaint:\n{text}\n\n"
            "Choose exactly one topic from: billing_issue, delayed_delivery, "
            "damaged_product, technical_support, refund_request, account_issue, general."
        ),
        expected_output="One topic label from the allowed list.",
        agent=topic_agent,
        output_pydantic=TopicOutput,
    )
    priority_task = Task(
        description=(
            "Assess the priority of the following customer complaint.\n\n"
            f"Complaint:\n{text}\n\n"
            "Use the sentiment and topic from previous agents."
        ),
        expected_output="Priority (low/medium/high) and one-sentence reasoning.",
        agent=priority_agent,
        context=[sentiment_task, topic_task],
        output_pydantic=PriorityOutput,
    )
    summary_task = Task(
        description=(
            "Write a concise summary (max 120 chars) for the following complaint "
            "and determine resolution status.\n\n"
            f"Complaint:\n{text}\n\n"
            "Use all prior agent outputs. Set problem_resolved=true only if the "
            "text explicitly states the issue is already resolved. "
            "Set needs_followup=true if action is still required."
        ),
        expected_output=(
            "One-sentence summary, recommended action, and boolean resolution flags."
        ),
        agent=summary_agent,
        context=[sentiment_task, topic_task, priority_task],
        output_pydantic=SummaryOutput,
    )

    crew = Crew(
        agents=[sentiment_agent, topic_agent, priority_agent, summary_agent],
        tasks=[sentiment_task, topic_task, priority_task, summary_task],
        process=Process.sequential,
        verbose=False,
    )
    crew.kickoff()

    s_out: SentimentOutput = sentiment_task.output.pydantic
    t_out: TopicOutput = topic_task.output.pydantic
    p_out: PriorityOutput = priority_task.output.pydantic
    r_out: SummaryOutput = summary_task.output.pydantic

    if any(o is None for o in (s_out, t_out, p_out, r_out)):
        raise ValueError("One or more agent tasks returned no structured output.")

    summary = r_out.summary.strip()
    if len(summary) > 120:
        summary = summary[:117].rstrip() + "..."

    return {
        "customer_sentiment": s_out.sentiment,
        "topic": t_out.topic,
        "priority": p_out.priority,
        "problem_resolved": r_out.problem_resolved,
        "needs_followup": r_out.needs_followup,
        "summary": summary,
    }
