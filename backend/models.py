from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from .database import Base


class Complaint(Base):
    """Represents a customer support complaint record."""

    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(String, unique=True, index=True)
    customer_id = Column(String)
    complaint_text = Column(Text)
    channel = Column(String)
    created_at = Column(DateTime)

    customer_sentiment = Column(String)
    topic = Column(String)
    priority = Column(String)
    problem_resolved = Column(Boolean)
    needs_followup = Column(Boolean)
    summary = Column(Text)
