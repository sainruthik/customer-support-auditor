"""Shared pytest fixtures: in-memory SQLite DB, TestClient, OpenAI patch."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, get_db
from backend.main import app


@pytest.fixture()
def client(monkeypatch):
    """TestClient backed by an isolated in-memory SQLite database.

    Also patches _analyze_with_openai to raise so tests never hit the real API
    and deterministically exercise the rule-based fallback path.
    """
    import backend.analyzer as analyzer_module

    monkeypatch.setattr(
        analyzer_module,
        "_analyze_with_crew",
        lambda _text: (_ for _ in ()).throw(RuntimeError("patched")),
    )

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine)
