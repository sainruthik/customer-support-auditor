"""Shared Pydantic schemas for API responses."""

from datetime import datetime

from pydantic import BaseModel


class ComplaintResponse(BaseModel):
    """Serialized complaint object returned by API endpoints."""

    id: int
    complaint_id: str
    customer_id: str
    complaint_text: str
    channel: str
    created_at: datetime
    customer_sentiment: str
    topic: str
    priority: str
    problem_resolved: bool
    needs_followup: bool
    summary: str


class MetricsResponse(BaseModel):
    """Aggregated complaint metrics for dashboard use."""

    total: int
    sentiment: dict[str, int]
    topics: dict[str, int]
    priority: dict[str, int]
    unresolved: int
    complaints_by_day: dict[str, int]
    negative_percentage: float
    unresolved_percentage: float
    top_topic: str | None
