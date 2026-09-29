"""
Reusable UI Components for Retail Customer Intelligence Streamlit App.

This module provides consistent, reusable UI components for the dashboard.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .app_formatting import auto_format, format_currency, format_probability, format_percent


# =============================================================================
# DESIGN TOKENS
# =============================================================================

ACTION_COLORS = {
    "protect_value": "#218739",      # green
    "accelerate_purchase": "#315efb", # blue
    "reactivate": "#a56600",          # amber
    "cross_sell": "#7c3aed",          # purple
    "nurture": "#0ea5e9",             # light blue
    "monitor": "#697386",             # gray
}

SEGMENT_PALETTE = [
    "#315efb", "#218739", "#a56600", "#7c3aed",
    "#0ea5e9", "#e11d48", "#f97316", "#84cc16",
    "#ec4899", "#6366f1", "#14b8a6", "#f43f5e",
]

STATUS_COLORS = {
    "ready": "#218739",
    "incomplete": "#a56600",
    "stale": "#e11d48",
    "validation_failed": "#e11d48",
    "unavailable": "#697386",
    "unknown": "#697386",
}


# =============================================================================
# KPI CARD
# =============================================================================

def render_kpi_card(
    label: str,
    value: Any,
    help_text: Optional[str] = None,
    formatter: str = "auto",
    column_name: str = "",
) -> None:
    """Render a single KPI metric card.
    
    Args:
        label: Metric label
        value: Metric value
        help_text: Optional tooltip text
        formatter: "auto", "currency", "probability", "percent", "ratio", "count"
        column_name: Column name for auto-formatting
    """
    if formatter == "auto":
        formatted = auto_format(value, column_name)
    elif formatter == "currency":
        formatted = format_currency(value)
    elif formatter == "probability":
        formatted = format_probability(value)
    elif formatter == "percent":
        formatted = format_percent(value)
    elif formatter == "ratio":
        formatted = format_ratio(value)
    elif formatter == "count":
        formatted = format_count(value)
    else:
        formatted = str(value) if value is not None else "—"
    
    st.metric(label, formatted, help=help_text)


def format_ratio(value: Any, decimals: int = 2) -> str:
    from .app_formatting import format_ratio
    return format_ratio(value, decimals)


def format_count(value: Any) -> str:
    from .app_formatting import format_count
    return format_count(value)


# =============================================================================
# SCIENCE CARD
# =============================================================================

def render_science_card(title: str, body: str, icon: Optional[str] = None) -> None:
    """Render an expandable science context card.
    
    Args:
        title: Card title
        body: Markdown body content
        icon: Optional icon (not used, kept for compatibility)
    """
    st.markdown(
        f"""
        <div class='science-card'>
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# STATUS CHIP
# =============================================================================

def render_status_chip(label: str, status: str, custom_color: Optional[str] = None) -> None:
    """Render a status indicator chip.
    
    Args:
        label: Chip label
        status: Status key (ready, incomplete, stale, validation_failed, unavailable, unknown)
        custom_color: Optional custom hex color
    """
    color = custom_color or STATUS_COLORS.get(status, "#697386")
    st.markdown(
        f"""
        <span style='
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: {color}15;
            color: {color};
            font-size: 0.76rem;
            font-weight: 700;
            margin-right: 6px;
            margin-bottom: 6px;
        '>
            {label}: {status.replace('_', ' ').title()}
        </span>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# EVIDENCE TABLE
# =============================================================================

def render_evidence_table(rows: List[Dict[str, Any]]) -> None:
    """Render a formatted evidence/metrics table.
    
    Args:
        rows: List of dicts with 'Metric' and 'Value' keys
    """
    if not rows:
        return
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# =============================================================================
# MISSING DATA PLACEHOLDER
# =============================================================================

def render_missing(message: str = "Required output is not available yet.") -> None:
    """Render a consistent missing data placeholder."""
    st.info(message)


# =============================================================================
# DOWNLOAD BUTTON
# =============================================================================

def render_download_button(
    df: pd.DataFrame,
    filename: str,
    label: str = "Download CSV",
    help_text: Optional[str] = None,
) -> None:
    """Render a CSV download button for a DataFrame.
    
    Args:
        df: DataFrame to export
        filename: Output filename
        label: Button label
        help_text: Optional tooltip
    """
    if df is None or df.empty:
        st.button(label, disabled=True, help="No data to download")
        return
    
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        help=help_text,
    )


# =============================================================================
# CUSTOMER SELECTOR
# =============================================================================

def render_customer_selector(
    customer_df: pd.DataFrame,
    id_column: str = "Customer ID",
    default_customer: Optional[int] = None,
    key_prefix: str = "customer_selector",
) -> Optional[int]:
    """Render customer search-first selector with filtered results.
    
    Args:
        customer_df: DataFrame with customer data
        id_column: Name of customer ID column
        default_customer: Optional default customer ID (for deep linking)
        key_prefix: Unique key prefix for Streamlit widgets
    
    Returns:
        Selected customer ID or None
    """
    if customer_df is None or customer_df.empty:
        st.warning("No customer data available.")
        return None
    
    # Ensure ID column is numeric
    ids = pd.to_numeric(customer_df[id_column], errors="coerce").dropna().astype(int).tolist()
    
    if not ids:
        st.warning("No valid customer IDs found.")
        return None
    
    # Search input - primary interaction
    search_key = f"{key_prefix}_search"
    search = st.text_input(
        "Search customer ID",
        value=str(default_customer) if default_customer else "",
        placeholder="Type customer ID (e.g. 12345) or leave blank to browse",
        key=search_key,
        help="Enter a customer ID directly, or leave empty to select from all customers",
    )
    
    # Filter IDs based on search
    if search.strip().isdigit():
        search_id = int(search.strip())
        if search_id in ids:
            filtered_ids = [search_id]
        else:
            filtered_ids = []
            st.info(f"No customer found with ID {search_id}")
    else:
        filtered_ids = ids
    
    # Show count of filtered results
    if search.strip():
        st.caption(f"Showing {len(filtered_ids)} of {len(ids)} customers")
    
    # Selectbox on filtered results (or all if no search)
    select_key = f"{key_prefix}_select"
    
    if not filtered_ids:
        st.selectbox(
            "Customer",
            options=[],
            disabled=True,
            placeholder="No matching customers",
            key=select_key,
        )
        return None
    
    # Determine default index
    default_idx = 0
    if default_customer and default_customer in filtered_ids:
        default_idx = filtered_ids.index(default_customer)
    
    selected_id = st.selectbox(
        "Customer",
        options=filtered_ids,
        index=default_idx,
        format_func=lambda x: f"Customer {x}",
        key=select_key,
    )
    
    return int(selected_id) if selected_id is not None else None


# =============================================================================
# ACTION SUMMARY TABLE
# =============================================================================

def render_action_summary_table(action_summary: pd.DataFrame) -> None:
    """Render formatted action summary table with color coding.
    
    Args:
        action_summary: DataFrame with action summary
    """
    if action_summary is None or action_summary.empty:
        render_missing("Action summary not available.")
        return
    
    # Find action column
    action_col = None
    for col in ["final_action", "recommended_action", "action"]:
        if col in action_summary.columns:
            action_col = col
            break
    
    customer_col = None
    for col in ["customers", "customer_count", "count"]:
        if col in action_summary.columns:
            customer_col = col
            break
    
    if not action_col or not customer_col:
        st.dataframe(action_summary, use_container_width=True, hide_index=True)
        return
    
    # Create display dataframe
    display = action_summary[[action_col, customer_col]].copy()
    display.columns = ["Recommended Action", "Customers"]
    display["Customers"] = display["Customers"].apply(lambda x: f"{x:,}")
    
    # Add color coding via styler
    def color_action(action):
        color = ACTION_COLORS.get(str(action).lower(), "#697386")
        return f"background-color: {color}15; color: {color}; font-weight: 600"
    
    styled = display.style.applymap(
        color_action,
        subset=["Recommended Action"]
    )
    
    st.dataframe(styled, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE HERO
# =============================================================================

def render_page_hero(
    kicker: str,
    title: str,
    description: str,
) -> None:
    """Render consistent page hero section.
    
    Args:
        kicker: Section label (uppercase, small)
        title: Page title
        description: One-sentence description
    """
    st.markdown(
        f"""
        <div class='hero'>
            <div class='kicker'>{kicker}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION LABEL
# =============================================================================

def render_section_label(label: str) -> None:
    """Render a section label."""
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)


# =============================================================================
# CUSTOMER HEADER
# =============================================================================

def render_customer_header(
    customer_id: int,
    segment: Optional[str] = None,
    action: Optional[str] = None,
) -> None:
    """Render customer profile header."""
    segment_display = segment if segment else "Unknown segment"
    action_display = action if action else "No action policy available"
    
    st.markdown(
        f"""
        <div class='customer-header'>
            <div class='kicker'>Customer profile</div>
            <h2>Customer {customer_id}</h2>
            <p>{segment_display} · {action_display}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# CUSTOMER METRIC ROW
# =============================================================================

def render_customer_metric_row(metrics: List[tuple[str, Any, str]]) -> None:
    """Render a row of customer metric cards.
    
    Args:
        metrics: List of (label, value, formatter) tuples
    """
    cols = st.columns(len(metrics))
    
    for col, (label, value, formatter) in zip(cols, metrics):
        with col:
            if formatter == "currency":
                formatted = format_currency(value)
            elif formatter == "probability":
                formatted = format_probability(value)
            elif formatter == "percent":
                formatted = format_percent(value)
            elif formatter == "count":
                formatted = format_count(value)
            else:
                formatted = auto_format(value)
            
            st.metric(label, formatted)


# =============================================================================
# HERO COPY MAP
# =============================================================================

HERO_COPY = {
    "Executive": (
        "Executive view",
        "A decision-oriented summary of customer economics, retention, risk, and opportunity.",
    ),
    "Customer 360": (
        "Customer 360",
        "Move from portfolio-level metrics to an individual customer and inspect the evidence behind the models.",
    ),
    "Segmentation": (
        "Behavioral segmentation",
        "A multi-dimensional view of customer behavior beyond conventional RFM.",
    ),
    "Cohorts": (
        "Cohort intelligence",
        "Read acquisition quality through retention, revenue persistence, and reactivation over customer age.",
    ),
    "CLV": (
        "Predicted Future Net Revenue (CLV Proxy)",
        "Inspect value distributions, uncertainty, and the economic concentration of the customer base.",
    ),
    "Retention & Next Purchase": (
        "Retention and next purchase",
        "Translate customer behavior into forward-looking probability signals with clear model diagnostics.",
    ),
    "Recommendations": (
        "Product recommendations",
        "Co-purchase based product recommendations with transparent scoring and fallback logic.",
    ),
    "Decision Engine": (
        "Commercial decision engine",
        "Combine value, risk, propensity, and behavioral context into a transparent next-best-action policy.",
    ),
    "Methodology": (
        "The science",
        "Understand how the analytical layers connect, what each model is answering, and where causal claims stop.",
    ),
}