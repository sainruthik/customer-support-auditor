"""Integration tests for all FastAPI routes using TestClient + in-memory SQLite."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

UTC = timezone.utc

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_csv(rows: list[dict]) -> bytes:
    """Serialise a list of dicts to CSV bytes."""
    if not rows:
        return b"complaint_text\n"
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _seed_complaint(client, text: str = "My package never arrived") -> dict:
    resp = client.post("/analyze-text", json={"complaint_text": text})
    assert resp.status_code == 200
    return resp.json()


def _seed_n(client, n: int) -> list[dict]:
    return [_seed_complaint(client, f"Complaint number {i}: billing issue") for i in range(n)]


# ── POST /analyze-text ────────────────────────────────────────────────────────

@pytest.mark.integration
def test_analyze_text_returns_complaint_response(client):
    resp = client.post("/analyze-text", json={"complaint_text": "My order was delayed by two weeks."})
    assert resp.status_code == 200
    data = resp.json()
    assert data["complaint_text"] == "My order was delayed by two weeks."
    assert data["customer_sentiment"] in {"positive", "neutral", "negative"}
    assert data["topic"] in {"billing_issue", "delayed_delivery", "damaged_product", "refund_request", "account_issue", "general"}
    assert data["priority"] in {"low", "medium", "high"}
    assert isinstance(data["problem_resolved"], bool)
    assert isinstance(data["needs_followup"], bool)
    assert data["summary"]
    assert data["complaint_id"]
    assert data["id"]


@pytest.mark.integration
def test_analyze_text_persists_to_db(client):
    client.post("/analyze-text", json={"complaint_text": "Damaged item received."})
    resp = client.get("/complaints?limit=1")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.integration
def test_analyze_text_missing_body_returns_422(client):
    resp = client.post("/analyze-text", json={})
    assert resp.status_code == 422


# ── POST /upload-csv ──────────────────────────────────────────────────────────

@pytest.mark.integration
def test_upload_csv_all_optional_columns(client):
    rows = [
        {
            "complaint_id": "C001",
            "customer_id": "U001",
            "complaint_text": "Received wrong item",
            "channel": "email",
            "created_at": "2025-04-01T10:00:00",
        }
    ]
    resp = client.post("/upload-csv", files={"file": ("test.csv", _make_csv(rows), "text/csv")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] == 1
    assert data["failed"] == 0


@pytest.mark.integration
def test_upload_csv_complaint_text_only(client):
    rows = [{"complaint_text": "My refund has not arrived"}]
    resp = client.post("/upload-csv", files={"file": ("test.csv", _make_csv(rows), "text/csv")})
    assert resp.status_code == 200
    assert resp.json()["processed"] == 1


@pytest.mark.integration
def test_upload_csv_skips_blank_rows(client):
    csv_bytes = b"complaint_text\n\n   \nValid complaint here\n"
    resp = client.post("/upload-csv", files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert resp.status_code == 200
    assert resp.json()["processed"] == 1


@pytest.mark.integration
def test_upload_csv_missing_complaint_text_column_returns_400(client):
    csv_bytes = b"customer_id,channel\nU001,email\n"
    resp = client.post("/upload-csv", files={"file": ("bad.csv", csv_bytes, "text/csv")})
    assert resp.status_code == 400
    assert "complaint_text" in resp.json()["detail"]


@pytest.mark.integration
def test_upload_csv_non_utf8_returns_400(client):
    bad_bytes = b"\xff\xfeNot valid UTF-8"
    resp = client.post("/upload-csv", files={"file": ("bad.csv", bad_bytes, "text/csv")})
    assert resp.status_code == 400


@pytest.mark.integration
def test_upload_csv_multiple_rows(client):
    rows = [{"complaint_text": f"Complaint {i}"} for i in range(5)]
    resp = client.post("/upload-csv", files={"file": ("bulk.csv", _make_csv(rows), "text/csv")})
    assert resp.status_code == 200
    assert resp.json()["processed"] == 5


# ── GET /complaints ───────────────────────────────────────────────────────────

@pytest.mark.integration
def test_complaints_empty_db_returns_empty_list(client):
    resp = client.get("/complaints")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.integration
def test_complaints_default_limit_25(client):
    _seed_n(client, 30)
    resp = client.get("/complaints")
    assert resp.status_code == 200
    assert len(resp.json()) == 25


@pytest.mark.integration
def test_complaints_skip_returns_remaining(client):
    _seed_n(client, 30)
    resp = client.get("/complaints?skip=25&limit=25")
    assert resp.status_code == 200
    assert len(resp.json()) == 5


@pytest.mark.integration
def test_complaints_total_count_header(client):
    _seed_n(client, 10)
    resp = client.get("/complaints?limit=5")
    assert resp.status_code == 200
    assert resp.headers.get("x-total-count") == "10"


@pytest.mark.integration
def test_complaints_ordered_most_recent_first(client):
    _seed_n(client, 3)
    resp = client.get("/complaints?limit=10")
    rows = resp.json()
    dates = [row["created_at"] for row in rows]
    assert dates == sorted(dates, reverse=True)


@pytest.mark.integration
def test_complaints_limit_zero_returns_422(client):
    resp = client.get("/complaints?limit=0")
    assert resp.status_code == 422


@pytest.mark.integration
def test_complaints_negative_skip_returns_422(client):
    resp = client.get("/complaints?skip=-1")
    assert resp.status_code == 422


@pytest.mark.integration
def test_complaints_limit_over_200_returns_422(client):
    resp = client.get("/complaints?limit=201")
    assert resp.status_code == 422


# ── GET /metrics ──────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_metrics_empty_db_returns_zeros(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["unresolved"] == 0
    assert data["negative_percentage"] == 0.0
    assert data["sentiment"] == {}
    assert data["topics"] == {}
    assert data["priority"] == {}
    assert data["complaints_by_day"] == {}
    assert data["top_topic"] is None


@pytest.mark.integration
def test_metrics_totals_after_seeding(client):
    _seed_n(client, 5)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert sum(data["sentiment"].values()) == 5
    assert sum(data["topics"].values()) == 5
    assert sum(data["priority"].values()) == 5


@pytest.mark.integration
def test_metrics_complaints_by_day_populated(client):
    _seed_n(client, 3)
    resp = client.get("/metrics")
    data = resp.json()
    assert len(data["complaints_by_day"]) >= 1
    assert all(isinstance(v, int) for v in data["complaints_by_day"].values())


@pytest.mark.integration
def test_metrics_top_topic_set(client):
    for _ in range(4):
        client.post("/analyze-text", json={"complaint_text": "billing issue again"})
    resp = client.get("/metrics")
    data = resp.json()
    assert data["top_topic"] is not None


# ── GET /alerts ───────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_alerts_empty_db_no_alerts(client):
    resp = client.get("/alerts")
    assert resp.status_code == 200
    assert resp.json()["alerts"] == []


@pytest.mark.integration
def test_alerts_high_negative_triggers_alert(client):
    for _ in range(8):
        client.post("/analyze-text", json={"complaint_text": "This is terrible and unacceptable, I am frustrated"})
    for _ in range(2):
        client.post("/analyze-text", json={"complaint_text": "Thank you, great service, I appreciate it"})
    resp = client.get("/alerts")
    alerts = resp.json()["alerts"]
    assert any("negative" in a.lower() for a in alerts)


@pytest.mark.integration
def test_alerts_unresolved_triggers_alert(client):
    for _ in range(7):
        client.post("/analyze-text", json={"complaint_text": "Still waiting, no response, not resolved"})
    for _ in range(3):
        client.post("/analyze-text", json={"complaint_text": "Issue resolved and fixed"})
    resp = client.get("/alerts")
    alerts = resp.json()["alerts"]
    assert any("unresolved" in a.lower() for a in alerts)


@pytest.mark.integration
def test_alerts_topic_dominance_triggers_alert(client):
    for _ in range(9):
        client.post("/analyze-text", json={"complaint_text": "My bill is wrong and I was overcharged"})
    client.post("/analyze-text", json={"complaint_text": "General question about my account"})
    resp = client.get("/alerts")
    alerts = resp.json()["alerts"]
    assert any("dominance" in a.lower() or "billing" in a.lower() for a in alerts)
