"""Seed complaint data from CSV into the local database."""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.analyzer import analyze_complaint
from backend.database import SessionLocal, create_tables
from backend.models import Complaint


def _parse_datetime(value: str) -> datetime:
    """Parse CSV datetime values like '2024-11-28 19:50:26'."""
    return datetime.fromisoformat(value.strip())


def seed_from_csv(csv_path: Path) -> tuple[int, int]:
    """
    Load complaints from CSV and persist analyzed records.

    Returns:
    - inserted_count
    - updated_count
    """
    create_tables()
    inserted_count = 0
    updated_count = 0

    db = SessionLocal()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                complaint_text = (row.get("complaint_text") or "").strip()
                if not complaint_text:
                    continue

                analysis = analyze_complaint(complaint_text)
                complaint_id = (row.get("complaint_id") or "").strip()

                existing = db.query(Complaint).filter(Complaint.complaint_id == complaint_id).first()

                if existing:
                    existing.customer_id = (row.get("customer_id") or "unknown").strip() or "unknown"
                    existing.complaint_text = complaint_text
                    existing.channel = (row.get("channel") or "unknown").strip() or "unknown"
                    existing.created_at = _parse_datetime(row.get("created_at") or datetime.now().isoformat())
                    existing.customer_sentiment = analysis["customer_sentiment"]
                    existing.topic = analysis["topic"]
                    existing.priority = analysis["priority"]
                    existing.problem_resolved = analysis["problem_resolved"]
                    existing.needs_followup = analysis["needs_followup"]
                    existing.summary = analysis["summary"]
                    updated_count += 1
                else:
                    db.add(
                        Complaint(
                            complaint_id=complaint_id,
                            customer_id=(row.get("customer_id") or "unknown").strip() or "unknown",
                            complaint_text=complaint_text,
                            channel=(row.get("channel") or "unknown").strip() or "unknown",
                            created_at=_parse_datetime(row.get("created_at") or datetime.now().isoformat()),
                            customer_sentiment=analysis["customer_sentiment"],
                            topic=analysis["topic"],
                            priority=analysis["priority"],
                            problem_resolved=analysis["problem_resolved"],
                            needs_followup=analysis["needs_followup"],
                            summary=analysis["summary"],
                        )
                    )
                    inserted_count += 1

        db.commit()
        return inserted_count, updated_count
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    csv_file = PROJECT_ROOT / "data" / "complaints.csv"
    inserted, updated = seed_from_csv(csv_file)
    print(f"Seeding completed. Inserted: {inserted}, Updated: {updated}")
