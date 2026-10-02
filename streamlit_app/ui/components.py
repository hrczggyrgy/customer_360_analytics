"""
Reusable UI Components for Retail Customer Intelligence Streamlit App.

Centralized, theme-aware components that replace ad-hoc HTML/CSS.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .theme import (
    get_status_color,
    get_action_color,
    get_segment_color,
    apply_plotly_theme,
    PLOTLY_CONFIG,
    STATUS_COLORS,
    ACTION_COLORS,
    SEGMENT_PALETTE,
    get_color_tokens,
)
from ..app_formatting import (
    format_currency,
    format_probability,
    format_percent,
    format_ratio,
    format_count,
    format_date,
    format_month,
    format_duration_months,
    format_days,
    format_score,
    auto_format,
    infer_semantic_type,
    SemanticType,
)


# =============================================================================
# KPI CARD SYSTEM
# =============================================================================

def render_kpi_card(
    label: str,
    value: Any,
    help_text: Optional[str] = None,
    formatter: str = "auto",
    column_name: str = "",
    delta: Optional[str] = None,
    delta_color: str = "normal",
) -> None:
    """
    Render a styled KPI metric card using native st.metric with consistent formatting.
    
    Args:
        label: Metric label
        value: Metric value
        help_text: Optional tooltip text
        formatter: "auto", "currency", "probability", "percent", "ratio", "count", "score", "days"
        column_name: Column name for auto-formatting
        delta: Optional delta indicator
        delta_color: "normal", "inverse", "off"
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
    elif formatter == "score":
        formatted = format_score(value)
    elif formatter == "days":
        formatted = format_days(value)
    else:
        formatted = str(value) if value is not None else "—"
    
    st.metric(label, formatted, delta=delta, delta_color=delta_color, help=help_text)


def render_kpi_row(metrics: List[Dict[str, Any]], max_cols: int = 4) -> None:
    """
    Render a responsive grid of KPI cards with proper row wrapping.
    
    Args:
        metrics: List of dicts with keys: label, value, formatter, help, delta, delta_color
        max_cols: Maximum columns per row (responsive)
    """
    n = len(metrics)
    if n == 0:
        return
    
    for start in range(0, n, max_cols):
        row_metrics = metrics[start:start + max_cols]
        cols = st.columns(len(row_metrics))
        
        for col, metric in zip(cols, row_metrics):
            with col:
                render_kpi_card(
                    label=metric.get("label", ""),
                    value=metric.get("value"),
                    help_text=metric.get("help"),
                    formatter=metric.get("formatter", "auto"),
                    column_name=metric.get("column_name", ""),
                    delta=metric.get("delta"),
                    delta_color=metric.get("delta_color", "normal"),
                )


# =============================================================================
# SCIENCE CARD
# =============================================================================

def render_science_card(
    title: str,
    body: str,
    icon: Optional[str] = None,
    expanded: bool = False,
) -> None:
    """
    Render a science context card using native Streamlit components.
    
    Args:
        title: Card title
        body: Markdown body content
        icon: Optional icon (emoji or text)
        expanded: Whether to use expander for long content
    """
    if expanded:
        with st.expander(title, expanded=False):
            st.markdown(body)
    else:
        st.markdown(
            f"""
            <div style='
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: var(--radius-lg);
                padding: 18px 20px;
                height: 100%;
                box-shadow: var(--shadow-sm);
            '>
                <h4 style='margin: 0 0 8px 0; color: var(--text-primary); font-size: var(--text-base); font-weight: 600;'>
                    {icon + ' ' if icon else ''}{title}
                </h4>
                <p style='margin: 0; color: var(--text-secondary); line-height: 1.55; font-size: var(--text-sm);'>
                    {body}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# STATUS CHIP
# =============================================================================

def render_status_chip(
    label: str,
    status: str,
    custom_color: Optional[str] = None,
    show_icon: bool = True,
) -> None:
    """
    Render a status indicator chip using centralized status colors.
    
    Args:
        label: Chip label
        status: Status key (ready, incomplete, stale, validation_failed, unavailable, unknown)
        custom_color: Optional custom hex color
        show_icon: Whether to show status icon
    """
    color = custom_color or get_status_color(status)
    
    # Status icons
    icons = {
        "ready": "✓",
        "incomplete": "○",
        "stale": "⟳",
        "validation_failed": "✗",
        "unavailable": "−",
        "unknown": "?",
    }
    icon = icons.get(status, "•") if show_icon else ""
    
    st.markdown(
        f"""
        <span style='
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 5px 10px;
            border-radius: var(--radius-full);
            background: {color}1A;
            color: {color};
            font-size: var(--text-xs);
            font-weight: 700;
            margin-right: 6px;
            margin-bottom: 6px;
        '>
            {icon} {label}: {status.replace('_', ' ').title()}
        </span>
        """,
        unsafe_allow_html=True,
    )


def render_status_row(statuses: Dict[str, Any]) -> None:
    """Render a row of status chips for pipeline modules."""
    for module, status in statuses.items():
        render_status_chip(
            label=module.replace('_', ' ').title(),
            status=status.overall_status,
        )
        if status.freshness_note:
            st.caption(f"  {status.freshness_note}")


# =============================================================================
# EVIDENCE TABLE
# =============================================================================

def render_evidence_table(rows: List[Dict[str, Any]]) -> None:
    """Render a formatted evidence/metrics table with semantic formatting."""
    if not rows:
        return
    
    df = pd.DataFrame(rows)
    
    # Apply semantic formatting to value column if present
    if "Value" in df.columns:
        df["Value"] = df["Value"].apply(lambda x: x if isinstance(x, str) else str(x))
    
    st.dataframe(df, use_container_width=True, hide_index=True)


# =============================================================================
# MISSING DATA PLACEHOLDER
# =============================================================================

def render_missing(
    message: str = "Required output is not available yet.",
    action: Optional[str] = None,
    script: Optional[str] = None,
) -> None:
    """
    Render a consistent missing data placeholder with actionable guidance.
    
    Args:
        message: What is missing
        action: What to do (e.g., "Run the pipeline")
        script: Script to run (e.g., "customer_360.py")
    """
    if action and script:
        st.info(f"**{message}**  \n{action}: `{script}`")
    elif action:
        st.info(f"**{message}**  \n{action}")
    else:
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
    """Render a CSV download button for a DataFrame."""
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
    """
    Render customer search-first selector with filtered results.
    
    Args:
        customer_df: DataFrame with customer data
        id_column: Name of customer ID column
        default_customer: Optional default customer ID (for deep linking)
        key_prefix: Unique key prefix for Streamlit widgets
    
    Returns:
        Selected customer ID or None
    """
    if customer_df is None or customer_df.empty:
        render_missing("No customer data available.")
        return None
    
    # Ensure ID column is numeric
    ids = pd.to_numeric(customer_df[id_column], errors="coerce").dropna().astype(int).tolist()
    
    if not ids:
        render_missing("No valid customer IDs found.")
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
    """Render formatted action summary table with color coding."""
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
    
    # Use column config for styling
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Recommended Action": st.column_config.TextColumn(
                "Recommended Action",
                help="Recommended commercial action",
            ),
            "Customers": st.column_config.NumberColumn(
                "Customers",
                format="%,d",
                help="Number of customers",
            ),
        },
    )


# =============================================================================
# PAGE HERO
# =============================================================================

HERO_COPY = {
    "Strategy": (
        "Customer Strategy",
        "Understand portfolio health, value concentration, customer movement, and commercial opportunity.",
    ),
    "Customers": (
        "Customer 360",
        "Move from portfolio-level metrics to an individual customer and inspect the evidence behind the models.",
    ),
    "Value & Retention": (
        "Value & Retention",
        "Where is future value concentrated, and where is economically meaningful risk concentrated?",
    ),
    "Segments": (
        "Customer Segments",
        "A commercial segmentation workspace: portfolio map, scorecard, composition, movement, and RFM benchmark.",
    ),
    "Products & Baskets": (
        "Products & Baskets",
        "What products, combinations, and customer affinities represent commercial opportunity?",
    ),
    "Personalisation": (
        "Personalisation",
        "Which recommendations are most relevant, and how well do the methods perform out of sample?",
    ),
    "Activation": (
        "Activation",
        "Given limited capacity, where is analytical attention allocated?",
    ),
    "Science & Governance": (
        "Science & Governance",
        "Data quality, feature governance, model performance, calibration, lineage, and methodology.",
    ),
}


def render_page_hero(
    kicker: str,
    title: str,
    description: str,
    freshness: Optional[str] = None,
    model_version: Optional[str] = None,
) -> None:
    """
    Render consistent page hero section with optional metadata.
    
    Args:
        kicker: Section label (uppercase, small)
        title: Page title
        description: One-sentence description
        freshness: Optional data freshness note
        model_version: Optional model version
    """
    meta_parts = []
    if freshness:
        meta_parts.append(f"Data: {freshness}")
    if model_version:
        meta_parts.append(f"Model: {model_version}")
    
    meta_html = f"<div style='color: var(--text-muted); font-size: var(--text-xs); margin-top: 8px;'>{' · '.join(meta_parts)}</div>" if meta_parts else ""
    
    st.markdown(
        f"""
        <div class='hero'>
            <div class='kicker'>{kicker}</div>
            <h1>{title}</h1>
            <p>{description}</p>
            {meta_html}
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
    clv: Optional[float] = None,
    churn_risk: Optional[float] = None,
) -> None:
    """
    Render customer profile header with key metrics.
    
    Args:
        customer_id: Customer identifier
        segment: Behavioral segment
        action: Recommended action
        clv: Customer lifetime value
        churn_risk: Churn probability
    """
    segment_display = segment if segment else "Unknown segment"
    action_display = action if action else "No action policy available"
    
    # Build metric badges
    badges = []
    if clv is not None:
        badges.append(f"CLV: {format_currency(clv)}")
    if churn_risk is not None:
        badges.append(f"Churn risk: {format_probability(churn_risk)}")
    
    badge_html = " · ".join(badges) if badges else ""
    badge_html = f"<div style='color: var(--text-muted); font-size: var(--text-sm); margin-top: 8px;'>{badge_html}</div>" if badge_html else ""
    
    st.markdown(
        f"""
        <div class='customer-header'>
            <div class='kicker'>Customer profile</div>
            <h2>Customer {customer_id}</h2>
            <p>{segment_display} · {action_display}</p>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# CUSTOMER METRIC ROW
# =============================================================================

def render_customer_metric_row(metrics: List[tuple[str, Any, str]]) -> None:
    """
    Render a row of customer metric cards.
    
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
            elif formatter == "score":
                formatted = format_score(value)
            elif formatter == "days":
                formatted = format_days(value)
            elif formatter == "ratio":
                formatted = format_ratio(value)
            else:
                formatted = auto_format(value)
            
            st.metric(label, formatted)


# =============================================================================
# FORMATTED DATAFRAME
# =============================================================================

def render_formatted_dataframe(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    column_config: Optional[Dict] = None,
    **kwargs,
) -> None:
    """
    Render a DataFrame with semantic column formatting.
    
    Args:
        df: DataFrame to render
        columns: Columns to display (default: all)
        column_config: Optional Streamlit column config overrides
        **kwargs: Additional st.dataframe arguments
    """
    if df is None or df.empty:
        render_missing("No data to display.")
        return
    
    display_df = df[columns] if columns else df.copy()
    
    # Build column config with semantic formatting
    config = {}
    for col in display_df.columns:
        sem_type = infer_semantic_type(col)
        
        if sem_type == SemanticType.CURRENCY:
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="£%.0f",
            )
        elif sem_type == SemanticType.PROBABILITY:
            display_df[col] = display_df[col] * 100
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%.1f%%",
            )
        elif sem_type == SemanticType.PERCENTAGE:
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%.1f%%",
            )
        elif sem_type == SemanticType.RATIO:
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%.2fx",
            )
        elif sem_type == SemanticType.COUNT:
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%,d",
            )
        elif sem_type == SemanticType.SCORE:
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%.2f",
            )
        elif sem_type == SemanticType.DURATION_DAYS:
            config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title(),
            )
        elif sem_type == SemanticType.DURATION_MONTHS:
            config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title(),
            )
        elif sem_type == SemanticType.DATE:
            config[col] = st.column_config.DateColumn(
                col.replace("_", " ").title(),
                format="YYYY-MM-DD",
            )
        elif sem_type == SemanticType.MONTH:
            config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title(),
            )
        else:
            config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title(),
            )
    
    # Override with user-provided config
    if column_config:
        config.update(column_config)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=config,
        **kwargs,
    )


# =============================================================================
# EMPTY STATE
# =============================================================================

def render_empty_state(
    title: str,
    message: str,
    action: Optional[str] = None,
    script: Optional[str] = None,
    icon: str = "📭",
) -> None:
    """Render a consistent empty state with actionable guidance."""
    st.markdown(
        f"""
        <div style='
            text-align: center;
            padding: 48px 24px;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-xl);
            margin: 16px 0;
        '>
            <div style='font-size: 48px; margin-bottom: 16px;'>{icon}</div>
            <h3 style='margin: 0 0 8px 0; color: var(--text-primary);'>{title}</h3>
            <p style='margin: 0 0 16px 0; color: var(--text-secondary);'>{message}</p>
            {f'<code style="background: var(--primary-soft); color: var(--primary); padding: 8px 12px; border-radius: var(--radius-md); font-size: var(--text-sm);">{script}</code>' if script else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if action and script:
        st.caption(f"{action}: `{script}`")


# =============================================================================
# ACTION BADGE
# =============================================================================

def render_action_badge(action: str, size: str = "sm") -> str:
    """Render an action badge as HTML string for use in tables."""
    color = get_action_color(action)
    
    padding = "4px 8px" if size == "sm" else "6px 12px"
    font_size = "var(--text-xs)" if size == "sm" else "var(--text-sm)"
    
    return (
        f"<span style='"
        f"display: inline-block;"
        f"padding: {padding};"
        f"border-radius: var(--radius-full);"
        f"background: {color}1A;"
        f"color: {color};"
        f"font-size: {font_size};"
        f"font-weight: 700;"
        f"text-transform: capitalize;"
        f"'>{action.replace('_', ' ')}</span>"
    )


# =============================================================================
# SEGMENT BADGE
# =============================================================================

def render_segment_badge(segment: str, index: int = 0, size: str = "sm") -> str:
    """Render a segment badge as HTML string."""
    color = get_segment_color(index)
    
    padding = "4px 8px" if size == "sm" else "6px 12px"
    font_size = "var(--text-xs)" if size == "sm" else "var(--text-sm)"
    
    return (
        f"<span style='"
        f"display: inline-block;"
        f"padding: {padding};"
        f"border-radius: var(--radius-full);"
        f"background: {color}1A;"
        f"color: {color};"
        f"font-size: {font_size};"
        f"font-weight: 700;"
        f"'>{segment}</span>"
    )


# =============================================================================
# RESPONSIVE COLUMNS
# =============================================================================

def responsive_columns(n: int, max_cols: int = 4, min_width: int = 280) -> List:
    """
    Create responsive column layout.
    
    Args:
        n: Number of items
        max_cols: Maximum columns
        min_width: Minimum column width in pixels (approximate)
    
    Returns:
        List of column objects
    """
    cols_count = min(max_cols, n)
    return st.columns(cols_count)


# =============================================================================
# INSIGHT PANEL
# =============================================================================

def render_insight(
    label: str,
    headline: str,
    detail: str,
    evidence: Optional[str] = None,
    severity: Optional[str] = None,
) -> None:
    """
    Render a strategic insight card.
    
    Args:
        label: Insight category label (e.g., "KEY INSIGHT", "OPPORTUNITY", "RISK")
        headline: One-sentence summary
        detail: Supporting detail with computed values
        evidence: Optional metric basis explanation
        severity: Optional severity indicator ("high", "medium", "low", "info")
    """
    severity_colors = {
        "high": "var(--danger)",
        "medium": "var(--warning)",
        "low": "var(--info)",
        "info": "var(--primary)",
    }
    severity_bg = {
        "high": "var(--danger-soft)",
        "medium": "var(--warning-soft)",
        "low": "var(--info-soft)",
        "info": "var(--primary-soft)",
    }
    
    color = severity_colors.get(severity, "var(--primary)")
    bg = severity_bg.get(severity, "var(--primary-soft)")
    
    label_html = f"<div style='color:{color};font-weight:700;font-size:var(--text-xs);letter-spacing:0.04em;text-transform:uppercase;margin-bottom:4px;'>{label}</div>"
    headline_html = f"<div style='color:var(--text-primary);font-size:var(--text-base);font-weight:600;line-height:1.4;margin-bottom:6px;'>{headline}</div>"
    detail_html = f"<div style='color:var(--text-secondary);font-size:var(--text-sm);line-height:1.5;margin-bottom:8px;'>{detail}</div>"
    
    evidence_html = ""
    if evidence:
        evidence_html = f"<div style='color:var(--text-muted);font-size:var(--text-xs);font-style:italic;border-top:1px solid var(--border);padding-top:8px;'>Evidence: {evidence}</div>"
    
    st.markdown(
        f"""
        <div style='
            background: {bg};
            border-left: 4px solid {color};
            border-radius: var(--radius-lg);
            padding: 16px 20px;
            margin: 8px 0;
        '>
            {label_html}
            {headline_html}
            {detail_html}
            {evidence_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_insight_row(insights: List[Dict[str, Any]], max_cols: int = 2) -> None:
    """Render multiple insights in a responsive grid."""
    n = len(insights)
    if n == 0:
        return
    
    for start in range(0, n, max_cols):
        row_insights = insights[start:start + max_cols]
        cols = st.columns(len(row_insights))
        
        for col, insight in zip(cols, row_insights):
            with col:
                render_insight(
                    label=insight.get("label", "INSIGHT"),
                    headline=insight.get("headline", ""),
                    detail=insight.get("detail", ""),
                    evidence=insight.get("evidence"),
                    severity=insight.get("severity"),
                )


# =============================================================================
# KPI STRIP
# =============================================================================

def render_kpi_strip(metrics: List[Dict[str, Any]]) -> None:
    """Render a compact horizontal strip of KPIs."""
    if not metrics:
        return
    
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            render_kpi_card(
                label=metric.get("label", ""),
                value=metric.get("value"),
                help_text=metric.get("help"),
                formatter=metric.get("formatter", "auto"),
                column_name=metric.get("column_name", ""),
                delta=metric.get("delta"),
                delta_color=metric.get("delta_color", "normal"),
            )


# =============================================================================
# METRIC CONTEXT
# =============================================================================

def render_metric_context(
    label: str,
    value: Any,
    benchmark: Any,
    formatter: str = "auto",
    column_name: str = "",
    higher_is_better: bool = True,
) -> None:
    """Render a metric with peer/benchmark context."""
    formatted_value = auto_format(value, column_name) if formatter == "auto" else (
        format_currency(value) if formatter == "currency" else
        format_probability(value) if formatter == "probability" else
        format_percent(value) if formatter == "percent" else
        format_ratio(value) if formatter == "ratio" else
        format_count(value) if formatter == "count" else
        format_score(value) if formatter == "score" else
        format_days(value) if formatter == "days" else
        str(value) if value is not None else "—"
    )
    
    formatted_benchmark = auto_format(benchmark, column_name) if formatter == "auto" else (
        format_currency(benchmark) if formatter == "currency" else
        format_probability(benchmark) if formatter == "probability" else
        format_percent(benchmark) if formatter == "percent" else
        format_ratio(benchmark) if formatter == "ratio" else
        format_count(benchmark) if formatter == "count" else
        format_score(benchmark) if formatter == "score" else
        format_days(benchmark) if formatter == "days" else
        str(benchmark) if benchmark is not None else "—"
    )
    
    try:
        v = float(value) if value is not None else 0
        b = float(benchmark) if benchmark is not None else 0
        favorable = (v >= b) if higher_is_better else (v <= b)
        color = "var(--success)" if favorable else "var(--danger)"
        icon = "▲" if favorable else "▼"
    except (TypeError, ValueError):
        color = "var(--text-muted)"
        icon = "•"
    
    st.markdown(
        f"""
        <div style='
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 14px 18px;
        '>
            <div style='color:var(--text-secondary);font-size:var(--text-xs);font-weight:600;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:4px;'>{label}</div>
            <div style='
                display:flex;
                align-items:baseline;
                gap:8px;
                color:var(--text-primary);
                font-size:var(--text-xl);
                font-weight:700;
            '>
                {formatted_value}
                <span style='
                    color:{color};
                    font-size:var(--text-sm);
                    font-weight:600;
                '>{icon} vs {formatted_benchmark}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PEER BENCHMARK
# =============================================================================

def render_peer_benchmark(
    metrics: List[Dict[str, Any]],
    title: str = "Customer vs Segment Peers",
) -> None:
    """
    Render a peer benchmark comparison.
    
    Args:
        metrics: List of dicts with keys: label, value, peer_median, higher_is_better, formatter, column_name
        title: Section title
    """
    st.markdown(f"<div class='section-label'>{title}</div>", unsafe_allow_html=True)
    
    for metric in metrics:
        render_metric_context(
            label=metric.get("label", ""),
            value=metric.get("value"),
            benchmark=metric.get("peer_median"),
            formatter=metric.get("formatter", "auto"),
            column_name=metric.get("column_name", ""),
            higher_is_better=metric.get("higher_is_better", True),
        )


# =============================================================================
# AUDIENCE TABLE
# =============================================================================

def render_audience_table(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    column_config: Optional[Dict] = None,
    download_filename: Optional[str] = None,
    download_label: str = "Export audience",
    max_rows: int = 500,
    **kwargs,
) -> None:
    """
    Render an actionable audience table with export capability.
    
    Args:
        df: DataFrame with audience data
        columns: Columns to display
        column_config: Streamlit column config
        download_filename: Filename for CSV export
        download_label: Label for download button
        max_rows: Maximum rows to display (for performance)
        **kwargs: Additional st.dataframe arguments
    """
    if df is None or df.empty:
        render_missing("No audience data to display.")
        return
    
    display_df = df[columns] if columns else df.copy()
    
    if len(display_df) > max_rows:
        st.caption(f"Showing {max_rows:,} of {len(display_df):,} rows. Use export for full data.")
        display_df = display_df.head(max_rows)
    
    config = {}
    for col in display_df.columns:
        sem_type = infer_semantic_type(col)
        
        if sem_type == SemanticType.CURRENCY:
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="£%.0f",
            )
        elif sem_type == SemanticType.PROBABILITY:
            display_df[col] = display_df[col] * 100
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%.1f%%",
            )
        elif sem_type == SemanticType.PERCENTAGE:
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%.1f%%",
            )
        elif sem_type == SemanticType.RATIO:
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%.2fx",
            )
        elif sem_type == SemanticType.COUNT:
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%,d",
            )
        elif sem_type == SemanticType.SCORE:
            config[col] = st.column_config.NumberColumn(
                col.replace("_", " ").title(),
                format="%.2f",
            )
        elif sem_type == SemanticType.DURATION_DAYS:
            config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title(),
            )
        elif sem_type == SemanticType.DURATION_MONTHS:
            config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title(),
            )
        elif sem_type == SemanticType.DATE:
            config[col] = st.column_config.DateColumn(
                col.replace("_", " ").title(),
                format="YYYY-MM-DD",
            )
        elif sem_type == SemanticType.MONTH:
            config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title(),
            )
        else:
            config[col] = st.column_config.TextColumn(
                col.replace("_", " ").title(),
            )
    
    if column_config:
        config.update(column_config)
    
    action_cols = [c for c in display_df.columns if any(kw in c.lower() for kw in ["action", "reason", "priority", "tier"])]
    for col in action_cols:
        if col in config:
            pass
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=config,
        **kwargs,
    )
    
    if download_filename:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=download_label,
            data=csv,
            file_name=download_filename,
            mime="text/csv",
            help=f"Download full audience ({len(df):,} rows) as CSV",
        )


# =============================================================================
# DISTRIBUTION SUMMARY
# =============================================================================

def render_distribution_summary(
    values: np.ndarray,
    label: str,
    formatter: str = "auto",
    column_name: str = "",
) -> None:
    """Render a compact distribution summary (min, p25, median, p75, max, mean)."""
    clean = values[~np.isnan(values)] if isinstance(values, np.ndarray) else np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))])
    
    if len(clean) == 0:
        render_missing(f"No {label} data available.")
        return
    
    stats = {
        "Min": np.min(clean),
        "P25": np.percentile(clean, 25),
        "Median": np.median(clean),
        "P75": np.percentile(clean, 75),
        "Max": np.max(clean),
        "Mean": np.mean(clean),
    }
    
    cols = st.columns(len(stats))
    for col, (stat_label, stat_value) in zip(cols, stats.items()):
        with col:
            if formatter == "auto":
                formatted = auto_format(stat_value, column_name)
            elif formatter == "currency":
                formatted = format_currency(stat_value)
            elif formatter == "probability":
                formatted = format_probability(stat_value)
            elif formatter == "percent":
                formatted = format_percent(stat_value)
            elif formatter == "ratio":
                formatted = format_ratio(stat_value)
            elif formatter == "count":
                formatted = format_count(stat_value)
            elif formatter == "score":
                formatted = format_score(stat_value)
            elif formatter == "days":
                formatted = format_days(stat_value)
            else:
                formatted = str(stat_value)
            
            st.metric(stat_label, formatted, help=f"{label} distribution")


# =============================================================================
# SCIENTIFIC NOTE
# =============================================================================

def render_scientific_note(note: str, title: str = "Scientific note") -> None:
    """Render a scientific methodology note."""
    st.markdown(
        f"""
        <div style='
            background: var(--info-soft);
            border: 1px solid var(--info);
            border-radius: var(--radius-lg);
            padding: 14px 18px;
            margin: 12px 0;
        '>
            <div style='color:var(--info);font-weight:700;font-size:var(--text-xs);letter-spacing:0.04em;text-transform:uppercase;margin-bottom:4px;'>{title}</div>
            <div style='color:var(--text-secondary);font-size:var(--text-sm);line-height:1.55;'>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> str:
    """Render the global sidebar with pipeline status."""
    import streamlit as st
    from ..app_data import get_registry
    from .theme import get_status_color
    
    st.sidebar.markdown("### Retail Customer Intelligence")
    
    registry = get_registry()
    statuses = registry.get_all_module_statuses()
    
    st.sidebar.markdown("---")
    
    # Module status
    with st.sidebar.expander("Pipeline status", expanded=True):
        for module, status in statuses.items():
            color = get_status_color(status.overall_status)
            icons = {
                "ready": "✓",
                "incomplete": "○",
                "stale": "⟳",
                "validation_failed": "✗",
                "unavailable": "−",
                "unknown": "?",
            }
            icon = icons.get(status.overall_status, "•")
            chip = (
                f"<span style='display:inline-flex;align-items:center;gap:4px;"
                f"padding:5px 10px;border-radius:999px;"
                f"background:{color}1A;color:{color};"
                f"font-size:0.76rem;font-weight:700;margin:2px;'>"
                f"{icon} {module.replace('_', ' ').title()}: {status.overall_status.title()}"
                f"</span>"
            )
            st.sidebar.markdown(chip, unsafe_allow_html=True)
            
            if status.freshness_note:
                st.sidebar.caption(f"  {status.freshness_note}")
    
    st.sidebar.markdown("---")
    
    # Page navigation
    pages = [
        "Executive",
        "Customer 360",
        "Segmentation",
        "Cohorts",
        "Predictive Value",
        "Retention & Next Purchase",
        "Recommendations",
        "Decision Engine",
        "Methodology",
    ]
    
    page = st.sidebar.radio("Navigate", pages, key="nav_page")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Observational retail analytics. Decision scores are not causal uplift estimates."
    )
    
    return page


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "render_kpi_card",
    "render_kpi_row",
    "render_science_card",
    "render_status_chip",
    "render_status_row",
    "render_evidence_table",
    "render_missing",
    "render_empty_state",
    "render_download_button",
    "render_customer_selector",
    "render_action_summary_table",
    "render_page_hero",
    "render_section_label",
    "render_customer_header",
    "render_customer_metric_row",
    "render_formatted_dataframe",
    "render_action_badge",
    "render_segment_badge",
    "render_sidebar",
    "responsive_columns",
    "HERO_COPY",
    "STATUS_COLORS",
    "ACTION_COLORS",
    "SEGMENT_PALETTE",
    "PLOTLY_CONFIG",
    "render_insight",
    "render_insight_row",
    "render_kpi_strip",
    "render_metric_context",
    "render_peer_benchmark",
    "render_audience_table",
    "render_distribution_summary",
    "render_scientific_note",
]