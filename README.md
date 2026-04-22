# Customer Support Ticket Auditor

Analyze customer complaint text with an LLM, store structured results, and visualize trends in a dashboard.

## Run with Docker

The fastest way to run the full stack locally.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Compose v2)

### Steps

1. Copy the env template and add your OpenAI key:
   ```bash
   cp .env.example .env
   # edit .env — set OPENAI_API_KEY=sk-...
   ```

2. Build and start both services:
   ```bash
   docker-compose up --build
   ```

   - Frontend: http://localhost:8501
   - Backend API docs: http://localhost:8000/docs

3. The frontend waits for the backend healthcheck to pass before starting, so
   the dashboard loads cleanly on the first visit.

### Data Persistence

SQLite data lives in a named Docker volume (`sqlite_data`). It survives
`docker-compose down` and `docker-compose up` restarts.

```bash
# Stop containers, keep data
docker-compose down

# Stop containers AND wipe the database
docker-compose down -v
```

### Seeding Sample Data (Docker)

```bash
docker-compose exec backend python scripts/seed_data.py
```

### Deploy to Render

`render.yaml` at the project root is a Render blueprint that deploys both
services as Docker web services.

1. Push this repo to GitHub.
2. In the [Render dashboard](https://render.com), choose **New → Blueprint** and
   point it at your repo.
3. Set `OPENAI_API_KEY` in the Render environment dashboard for both services
   (marked `sync: false` in `render.yaml` — never committed).

> **Note:** The backend uses a Render Disk for SQLite persistence (1 GB,
> `starter` plan required). For free-tier deploys, data resets on each redeploy.

---

## Tech Stack

- FastAPI + SQLAlchemy (SQLite)
- OpenAI Responses API with rule-based fallback
- Pydantic v2
- Streamlit + Plotly + Pandas

## Project Structure

```
backend/
  main.py          FastAPI app entrypoint
  database.py      SQLAlchemy engine / session
  models.py        Complaint ORM model
  analyzer.py      OpenAI-first analyzer with rule-based fallback
  routes.py        API routes
shared/
  schemas.py       Pydantic response schemas
frontend/
  app.py           Streamlit dashboard
  api_client.py    Typed HTTP wrappers for the backend API
scripts/
  seed_data.py     CSV seed helper
data/
  complaints.csv   Sample complaints
tests/
  conftest.py      Fixtures (in-memory SQLite, TestClient)
  test_analyzer.py Unit tests for analyzer logic
  test_routes.py   Integration tests for all API endpoints
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. Install runtime dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install dev dependencies (for tests):
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Copy `.env.example` to `.env` and fill in your values:
   ```
   DATABASE_URL=sqlite:///./test.db
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o-mini           # optional, default shown
   OPENAI_BASE_URL=https://api.openai.com/v1  # optional, default shown
   ```

   If `OPENAI_API_KEY` is missing or the API call fails for any reason, the
   analyzer automatically falls back to a local rule-based classifier — no
   internet access required for basic operation.

## Run Backend

```bash
uvicorn backend.main:app --reload
```

## Run Frontend

```bash
streamlit run frontend/app.py
```

## Seed Sample Data

```bash
python scripts/seed_data.py
```

## Run Tests

```bash
pytest --cov=backend --cov=shared --cov-report=term-missing
```

The suite uses an in-memory SQLite database and patches out the OpenAI call so
no API key is needed.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API status |
| `GET` | `/health` | Health check |
| `POST` | `/analyze-text` | Analyze a single complaint and save it |
| `POST` | `/upload-csv` | Bulk upload complaints via CSV file |
| `GET` | `/complaints` | List complaints (paginated) |
| `GET` | `/metrics` | Aggregated counts and `complaints_by_day` |
| `GET` | `/alerts` | Threshold-based operational alerts |

### POST /analyze-text

```json
{ "complaint_text": "My order never arrived after 3 weeks." }
```

Returns a full `ComplaintResponse` object.

### POST /upload-csv

Multipart form upload. Required column: `complaint_text`.
Optional columns: `complaint_id`, `customer_id`, `channel`, `created_at`.

Returns `{ "processed": N, "failed": M }`.

### GET /complaints

Query params:
- `skip` — offset (default `0`, min `0`)
- `limit` — page size (default `25`, min `1`, max `200`)

Response header `X-Total-Count` contains the total row count for pagination UI.

### GET /metrics

Returns:
```json
{
  "total": 120,
  "sentiment": { "negative": 80, "neutral": 25, "positive": 15 },
  "topics": { "billing_issue": 40, ... },
  "priority": { "high": 30, ... },
  "unresolved": 55,
  "complaints_by_day": { "2025-04-10": 12, ... },
  "negative_percentage": 66.67,
  "unresolved_percentage": 45.83,
  "top_topic": "billing_issue"
}
```

### GET /alerts

Returns alert strings when thresholds are crossed:
- Negative sentiment > 50 %
- Unresolved complaints > 30 %
- Single topic accounts for > 50 % of all complaints

## Analyzer Behavior

Each complaint is classified into:

| Field | Values |
|-------|--------|
| `customer_sentiment` | `positive` / `neutral` / `negative` |
| `topic` | `billing_issue` / `delayed_delivery` / `damaged_product` / `technical_support` / `refund_request` / `account_issue` / `general` |
| `priority` | `low` / `medium` / `high` |
| `problem_resolved` | `true` / `false` |
| `needs_followup` | `true` / `false` |
| `summary` | max 120-char summary |

The OpenAI call uses temperature `0` and requires strict JSON. If the response
is missing fields, contains invalid enum values, or the API is unreachable, the
rule-based fallback is used transparently.
