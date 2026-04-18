"""Typed HTTP wrappers for the Customer Support Ticket Auditor backend API."""

from __future__ import annotations

import json
import os
from urllib import error, request

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def fetch_json(path: str) -> dict | list:
    with request.urlopen(f"{API_BASE_URL}{path}") as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{API_BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_complaints(skip: int = 0, limit: int = 25) -> tuple[list[dict], int | None]:
    """Return (rows, total) where total comes from X-Total-Count header."""
    url = f"{API_BASE_URL}/complaints?skip={skip}&limit={limit}"
    req = request.Request(url)
    with request.urlopen(req) as response:
        rows: list[dict] = json.loads(response.read().decode("utf-8"))
        raw_total = response.headers.get("X-Total-Count")
        total = int(raw_total) if raw_total is not None else None
    return rows, total


def upload_csv(file_bytes: bytes, filename: str) -> dict:
    """POST multipart CSV to /upload-csv; returns {processed, failed}."""
    resp = requests.post(
        f"{API_BASE_URL}/upload-csv",
        files={"file": (filename, file_bytes, "text/csv")},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()
