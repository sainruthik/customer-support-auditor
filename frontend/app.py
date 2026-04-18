"""Streamlit dashboard for the Customer Support Ticket Auditor."""

from __future__ import annotations

from urllib import error

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import fetch_complaints, fetch_json, post_json, upload_csv

st.set_page_config(page_title="Customer Support Ticket Auditor", layout="wide")
st.title("Customer Support Ticket Auditor")

# ── Fetch metrics (required for all chart sections) ──────────────────────────
try:
    metrics = fetch_json("/metrics")
except error.URLError:
    st.error("Cannot connect to backend API. Start FastAPI first: uvicorn backend.main:app --reload")
    st.stop()

# ── Top-level KPI row ─────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Complaints", metrics.get("total", 0))
kpi2.metric("Unresolved", metrics.get("unresolved", 0))
kpi3.metric("Negative %", f"{metrics.get('negative_percentage', 0):.1f}%")
kpi4.metric("Top Topic", metrics.get("top_topic") or "—")

st.divider()

# ── Time-series chart ─────────────────────────────────────────────────────────
st.subheader("Complaint Volume Over Time")
day_data = metrics.get("complaints_by_day", {})
if day_data:
    day_df = pd.DataFrame(
        [{"date": d, "count": c} for d, c in day_data.items()]
    )
    day_df["date"] = pd.to_datetime(day_df["date"])
    day_df = day_df.sort_values("date")
    st.plotly_chart(
        px.line(day_df, x="date", y="count", markers=True, labels={"count": "Complaints", "date": "Date"}),
        use_container_width=True,
    )
else:
    st.info("No time-series data yet.")

st.divider()

# ── Distribution charts ───────────────────────────────────────────────────────
chart_col1, chart_col2, chart_col3 = st.columns(3)

sentiment_df = pd.DataFrame(
    [{"sentiment": k, "count": v} for k, v in metrics.get("sentiment", {}).items()]
)
topic_df = pd.DataFrame(
    [{"topic": k, "count": v} for k, v in metrics.get("topics", {}).items()]
)
priority_df = pd.DataFrame(
    [{"priority": k, "count": v} for k, v in metrics.get("priority", {}).items()]
)

with chart_col1:
    st.subheader("Sentiment Distribution")
    if not sentiment_df.empty:
        st.plotly_chart(px.pie(sentiment_df, names="sentiment", values="count"), use_container_width=True)
    else:
        st.info("No data yet.")

with chart_col2:
    st.subheader("Top Complaint Topics")
    if not topic_df.empty:
        st.plotly_chart(
            px.bar(topic_df.sort_values("count", ascending=False), x="topic", y="count"),
            use_container_width=True,
        )
    else:
        st.info("No data yet.")

with chart_col3:
    st.subheader("Priority Breakdown")
    if not priority_df.empty:
        st.plotly_chart(px.bar(priority_df, x="priority", y="count"), use_container_width=True)
    else:
        st.info("No data yet.")

st.divider()

# ── Alerts ────────────────────────────────────────────────────────────────────
try:
    alerts_payload = fetch_json("/alerts")
    alerts = alerts_payload.get("alerts", [])
    if alerts:
        st.subheader("Alerts")
        for alert in alerts:
            st.warning(alert)
except error.URLError:
    pass

# ── Complaints table with pagination ─────────────────────────────────────────
st.subheader("Recent Complaints")

st.session_state.setdefault("page", 0)
st.session_state.setdefault("page_size", 25)

# Page size selector — reset page when size changes.
prev_size = st.session_state.get("_prev_page_size", st.session_state["page_size"])
page_size = st.selectbox("Rows per page", [25, 50, 100], index=[25, 50, 100].index(st.session_state["page_size"]))
if page_size != prev_size:
    st.session_state["page"] = 0
st.session_state["page_size"] = page_size
st.session_state["_prev_page_size"] = page_size

page = st.session_state["page"]
skip = page * page_size

try:
    complaints, total = fetch_complaints(skip=skip, limit=page_size)
except error.URLError:
    st.error("Could not fetch complaints.")
    complaints, total = [], None

complaints_df = pd.DataFrame(complaints)
if not complaints_df.empty:
    display_cols = ["created_at", "complaint_id", "topic", "customer_sentiment", "priority", "needs_followup", "summary"]
    existing_cols = [c for c in display_cols if c in complaints_df.columns]
    st.dataframe(complaints_df[existing_cols], use_container_width=True)
else:
    st.info("No complaints on this page.")

# Prev / page label / Next
nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
with nav_left:
    if st.button("← Prev", disabled=(page == 0)):
        st.session_state["page"] -= 1
        st.rerun()
with nav_mid:
    if total is not None:
        total_pages = max(1, -(-total // page_size))  # ceil division
        st.caption(f"Page {page + 1} of {total_pages}  ({total} total)")
    else:
        st.caption(f"Page {page + 1}")
with nav_right:
    at_last = len(complaints) < page_size or (total is not None and (page + 1) * page_size >= total)
    if st.button("Next →", disabled=at_last):
        st.session_state["page"] += 1
        st.rerun()

st.divider()

# ── Analyze new complaint ─────────────────────────────────────────────────────
st.subheader("Analyze New Complaint")
new_text = st.text_area("Complaint text", placeholder="Enter customer complaint text here...")
if st.button("Analyze and Save"):
    if not new_text.strip():
        st.error("Please enter complaint text.")
    else:
        try:
            saved = post_json("/analyze-text", {"complaint_text": new_text})
            st.success(f"Saved complaint {saved.get('complaint_id')}")
            st.session_state["page"] = 0
            st.rerun()
        except error.HTTPError as exc:
            st.error(f"Request failed: {exc.read().decode('utf-8')}")
        except error.URLError:
            st.error("Backend API is not reachable.")

st.divider()

# ── Upload CSV ────────────────────────────────────────────────────────────────
st.subheader("Upload Complaints CSV")
st.caption("Required column: `complaint_text`. Optional: `complaint_id`, `customer_id`, `channel`, `created_at`. Large files may take a minute while complaints are analyzed.")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded is not None:
    if st.button("Upload and Analyze"):
        with st.spinner("Uploading and analyzing complaints — this may take a moment..."):
            try:
                result = upload_csv(uploaded.read(), uploaded.name)
                processed = result.get("processed", 0)
                failed = result.get("failed", 0)
                st.success(f"Done — {processed} processed, {failed} failed.")
                st.session_state["page"] = 0
                st.rerun()
            except Exception as exc:
                st.error(f"Upload failed: {exc}")
