import asyncio
import csv
import io
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from .analyzer import analyze_complaint
from .database import get_db
from .models import Complaint
from shared.schemas import ComplaintResponse, MetricsResponse

# Create a router so routes can be included in main.py.
router = APIRouter()


class AnalyzeTextRequest(BaseModel):
    """Request body for complaint text analysis."""

    complaint_text: str


def _parse_created_at(value: str | None) -> datetime:
    """Parse CSV timestamp if present; otherwise use current timezone.utc time."""
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.strip())
        # Keep timezone-aware timestamps for consistency in API responses.
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _serialize_complaint(complaint: Complaint) -> ComplaintResponse:
    """Convert ORM model to a typed response object."""
    return ComplaintResponse(
        id=complaint.id,
        complaint_id=complaint.complaint_id,
        customer_id=complaint.customer_id,
        complaint_text=complaint.complaint_text,
        channel=complaint.channel,
        created_at=complaint.created_at,
        customer_sentiment=complaint.customer_sentiment,
        topic=complaint.topic,
        priority=complaint.priority,
        problem_resolved=complaint.problem_resolved,
        needs_followup=complaint.needs_followup,
        summary=complaint.summary,
    )


@router.post("/analyze-text", response_model=ComplaintResponse)
def analyze_text(payload: AnalyzeTextRequest, db: Session = Depends(get_db)) -> ComplaintResponse:
    """
    Analyze incoming complaint text, save it, and return saved data.
    """
    # Run rule-based analysis first.
    analysis = analyze_complaint(payload.complaint_text)

    # Build ORM object. Some fields are placeholders until extra inputs are added.
    complaint = Complaint(
        complaint_id=str(uuid4()),
        customer_id="unknown",
        complaint_text=payload.complaint_text,
        channel="unknown",
        created_at=datetime.now(timezone.utc),
        customer_sentiment=analysis["customer_sentiment"],
        topic=analysis["topic"],
        priority=analysis["priority"],
        problem_resolved=analysis["problem_resolved"],
        needs_followup=analysis["needs_followup"],
        summary=analysis["summary"],
    )

    # Save and reload so DB-generated fields (like id) are available.
    db.add(complaint)
    try:
        db.commit()
        db.refresh(complaint)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to save complaint.") from exc

    return _serialize_complaint(complaint)


@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict[str, int]:
    """
    Upload a CSV, analyze each complaint_text row, and save valid records.

    Expected CSV column:
    - complaint_text (required)
    Optional columns:
    - complaint_id, customer_id, channel, created_at
    """
    try:
        content = await file.read()
        text_content = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.") from exc

    reader = csv.DictReader(io.StringIO(text_content))
    if not reader.fieldnames or "complaint_text" not in reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV must include a complaint_text column.")

    rows = [row for row in reader if (row.get("complaint_text") or "").strip()]

    # Run all LLM calls concurrently in the thread pool so the event loop
    # is not blocked by synchronous httpx calls inside analyze_complaint.
    analyses = await asyncio.gather(
        *[asyncio.to_thread(analyze_complaint, row["complaint_text"].strip()) for row in rows],
        return_exceptions=True,
    )

    processed = 0
    failed = 0

    for row, analysis in zip(rows, analyses):
        if isinstance(analysis, Exception):
            failed += 1
            continue

        complaint_text = row["complaint_text"].strip()
        complaint = Complaint(
            complaint_id=(row.get("complaint_id") or "").strip() or str(uuid4()),
            customer_id=(row.get("customer_id") or "").strip() or "unknown",
            complaint_text=complaint_text,
            channel=(row.get("channel") or "").strip() or "unknown",
            created_at=_parse_created_at(row.get("created_at")),
            customer_sentiment=analysis["customer_sentiment"],
            topic=analysis["topic"],
            priority=analysis["priority"],
            problem_resolved=analysis["problem_resolved"],
            needs_followup=analysis["needs_followup"],
            summary=analysis["summary"],
        )

        db.add(complaint)
        try:
            db.commit()
            processed += 1
        except Exception:
            db.rollback()
            failed += 1

    return {"processed": processed, "failed": failed}


@router.delete("/complaints", status_code=200)
def delete_all_complaints(db: Session = Depends(get_db)) -> dict[str, int]:
    """Delete all complaints from the database."""
    deleted = db.query(Complaint).delete()
    db.commit()
    return {"deleted": deleted}


@router.get("/complaints", response_model=list[ComplaintResponse])
def get_complaints(
    response: Response,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[ComplaintResponse]:
    """Return complaints ordered by most recent first, with pagination."""
    total = db.query(func.count(Complaint.id)).scalar() or 0
    complaints = (
        db.query(Complaint)
        .order_by(desc(Complaint.created_at))
        .offset(skip)
        .limit(limit)
        .all()
    )
    response.headers["X-Total-Count"] = str(total)
    return [_serialize_complaint(complaint) for complaint in complaints]


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(db: Session = Depends(get_db)) -> MetricsResponse:
    """Return aggregate complaint metrics for dashboard cards/charts."""
    total = db.query(func.count(Complaint.id)).scalar() or 0

    sentiment_rows = (
        db.query(Complaint.customer_sentiment, func.count(Complaint.id))
        .group_by(Complaint.customer_sentiment)
        .all()
    )
    topic_rows = db.query(Complaint.topic, func.count(Complaint.id)).group_by(Complaint.topic).all()
    priority_rows = db.query(Complaint.priority, func.count(Complaint.id)).group_by(Complaint.priority).all()
    day_rows = (
        db.query(func.date(Complaint.created_at), func.count(Complaint.id))
        .group_by(func.date(Complaint.created_at))
        .order_by(func.date(Complaint.created_at))
        .all()
    )

    unresolved = db.query(func.count(Complaint.id)).filter(Complaint.needs_followup.is_(True)).scalar() or 0
    negative = (
        db.query(func.count(Complaint.id))
        .filter(Complaint.customer_sentiment == "negative")
        .scalar()
        or 0
    )

    negative_percentage = round((negative / total) * 100, 2) if total else 0.0
    unresolved_percentage = round((unresolved / total) * 100, 2) if total else 0.0

    top_topic = None
    top_topic_count = 0
    if topic_rows:
        top_topic, top_topic_count = max(topic_rows, key=lambda row: row[1])

    return MetricsResponse(
        total=total,
        sentiment={key or "unknown": count for key, count in sentiment_rows},
        topics={key or "unknown": count for key, count in topic_rows},
        priority={key or "unknown": count for key, count in priority_rows},
        unresolved=unresolved,
        complaints_by_day={str(day): count for day, count in day_rows if day is not None},
        negative_percentage=negative_percentage,
        unresolved_percentage=unresolved_percentage,
        top_topic=top_topic if top_topic_count > 0 else None,
    )


@router.get("/alerts")
def get_alerts(db: Session = Depends(get_db)) -> dict:
    """Return simple threshold-based operational alerts."""
    total = db.query(func.count(Complaint.id)).scalar() or 0
    negative_count = (
        db.query(func.count(Complaint.id))
        .filter(Complaint.customer_sentiment == "negative")
        .scalar()
        or 0
    )
    unresolved_count = (
        db.query(func.count(Complaint.id))
        .filter(Complaint.needs_followup.is_(True))
        .scalar()
        or 0
    )
    topic_rows = db.query(Complaint.topic, func.count(Complaint.id)).group_by(Complaint.topic).all()

    negative_percentage = round((negative_count / total) * 100, 2) if total else 0.0
    unresolved_percentage = round((unresolved_count / total) * 100, 2) if total else 0.0

    top_topic = None
    top_topic_ratio = 0.0
    if topic_rows and total:
        top_topic, top_topic_count = max(topic_rows, key=lambda row: row[1])
        top_topic_ratio = top_topic_count / total

    alerts: list[str] = []
    if negative_percentage > 50:
        alerts.append("High negative sentiment detected")
    if unresolved_percentage > 30:
        alerts.append("Too many unresolved complaints")
    # "Dominates" means one topic accounts for more than half of all complaints.
    if top_topic and top_topic_ratio > 0.5:
        alerts.append(f"Topic dominance detected: {top_topic}")

    return {
        "alerts": alerts,
        "negative_percentage": negative_percentage,
        "unresolved_percentage": unresolved_percentage,
        "top_topic": top_topic,
    }
