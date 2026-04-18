from fastapi import FastAPI
from .routes import router
from .database import create_tables


app = FastAPI(
    title="Customer Support Ticket Auditor",
    description="API for analyzing and tracking customer complaints",
)

app.include_router(router)

create_tables()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {"service": "customer-support-auditor", "status": "ok"}

