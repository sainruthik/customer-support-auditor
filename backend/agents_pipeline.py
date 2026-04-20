"""Sequential multi-agent pipeline using direct OpenAI API calls."""

from __future__ import annotations

import json
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _chat(client: OpenAI, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=_MODEL,
        temperature=0,
        max_tokens=256,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def _sentiment_agent(client: OpenAI, text: str) -> dict:
    system = (
        "You are a sentiment analyst. Analyze the customer complaint and respond with "
        "a JSON object containing exactly: "
        '{"sentiment": "<positive|negative|neutral>", "confidence": <0.0-1.0>}'
    )
    raw = _chat(client, system, text)
    data = json.loads(raw)
    assert data["sentiment"] in {"positive", "negative", "neutral"}
    return data


def _topic_agent(client: OpenAI, text: str) -> dict:
    system = (
        "You are a complaint topic classifier. Classify the complaint and respond with "
        "a JSON object containing exactly: "
        '{"topic": "<label>"} '
        "where label is one of: billing_issue, delayed_delivery, damaged_product, "
        "technical_support, refund_request, account_issue, general."
    )
    raw = _chat(client, system, text)
    data = json.loads(raw)
    assert data["topic"] in {
        "billing_issue", "delayed_delivery", "damaged_product",
        "technical_support", "refund_request", "account_issue", "general",
    }
    return data


def _priority_agent(client: OpenAI, text: str, sentiment: str, topic: str) -> dict:
    system = (
        "You are a priority assessor for customer support tickets. "
        "Given the complaint, its sentiment, and topic, respond with a JSON object: "
        '{"priority": "<low|medium|high>", "reasoning": "<one sentence>"}'
    )
    user = f"Complaint: {text}\nSentiment: {sentiment}\nTopic: {topic}"
    raw = _chat(client, system, user)
    data = json.loads(raw)
    assert data["priority"] in {"low", "medium", "high"}
    return data


def _summary_agent(client: OpenAI, text: str, sentiment: str, topic: str, priority: str) -> dict:
    system = (
        "You are a support ticket summarizer. Given the complaint and analysis, respond with a JSON object: "
        '{"summary": "<max 120 chars>", "recommended_action": "<one sentence>", '
        '"problem_resolved": <true|false>, "needs_followup": <true|false>} '
        "Set problem_resolved=true only if the text explicitly states the issue is already resolved. "
        "Set needs_followup=true if action is still required."
    )
    user = (
        f"Complaint: {text}\n"
        f"Sentiment: {sentiment}\nTopic: {topic}\nPriority: {priority}"
    )
    raw = _chat(client, system, user)
    data = json.loads(raw)
    assert isinstance(data["problem_resolved"], bool)
    assert isinstance(data["needs_followup"], bool)
    return data


def run_agent_pipeline(text: str) -> dict:
    """Run 4-agent sequential pipeline and return analysis dict."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    s = _sentiment_agent(client, text)
    t = _topic_agent(client, text)
    p = _priority_agent(client, text, s["sentiment"], t["topic"])
    r = _summary_agent(client, text, s["sentiment"], t["topic"], p["priority"])

    summary = r["summary"].strip()
    if len(summary) > 120:
        summary = summary[:117].rstrip() + "..."

    return {
        "customer_sentiment": s["sentiment"],
        "topic": t["topic"],
        "priority": p["priority"],
        "problem_resolved": r["problem_resolved"],
        "needs_followup": r["needs_followup"],
        "summary": summary,
    }
