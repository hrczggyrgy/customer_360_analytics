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
    Render a responsive row of KPI cards.
    
    Args:
        metrics: List of dicts with keys: label, value, formatter, help, delta, delta_color
        max_cols: Maximum columns per row (responsive)
    """
    n = len(metrics)
    if n == 0:
        return
    
    # Responsive: use min(max_cols, n) columns
    cols = st.columns(min(max_cols, n))
    
    for i, metric in enumerate(metrics):
        with cols[i % len(cols)]:
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
    "Predictive Value": (
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
]