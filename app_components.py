"""
Reusable UI components for the Retail Customer Intelligence application.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from app_config import MODULE_LABELS
from app_data import ArtifactRegistry, ModuleStatus, get_registry
from app_formatting import (
    format_clv,
    format_churn_risk,
    format_count,
    format_currency,
    format_date,
    format_next_purchase,
    format_nrr_index,
    format_probability,
    format_retention,
)


# =============================================================================
# STATUS INDICATORS
# =============================================================================

STATUS_COLORS = {
    "ready": "#218739",        # success green
    "incomplete": "#a56600",    # warning amber
    "stale": "#a56600",         # warning amber
    "validation_failed": "#c53d32",  # danger red
    "unavailable": "#697386",   # muted gray
    "unknown": "#697386",       # muted gray
}

STATUS_LABELS = {
    "ready": "Ready",
    "incomplete": "Incomplete",
    "stale": "Stale",
    "validation_failed": "Validation Failed",
    "unavailable": "Unavailable",
    "unknown": "Unknown",
}


def render_status_chip(status: str, label: str = "", size: str = "normal") -> None:
    """Render a status indicator chip."""
    color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
    text = STATUS_LABELS.get(status, status.title())
    if label:
        text = f"{label}: {text}"

    font_size = "0.76rem" if size == "normal" else "0.68rem"
    padding = "5px 10px" if size == "normal" else "3px 8px"

    st.markdown(
        f"""
        <span style="
            display: inline-block;
            padding: {padding};
            border-radius: 999px;
            background: {color}15;
            color: {color};
            font-size: {font_size};
            font-weight: 700;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid {color}40;
        ">
            {text}
        </span>
        """,
        unsafe_allow_html=True,
    )


def render_module_status_grid(statuses: Dict[str, ModuleStatus], columns: int = 4) -> None:
    """Render a grid of module status chips."""
    cols = st.columns(columns)
    items = list(statuses.items())

    for i, (module, status) in enumerate(items):
        with cols[i % columns]:
            render_status_chip(status.overall_status, status.label, size="small")


# =============================================================================
# KPI CARDS
# =============================================================================

def render_kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
) -> None:
    """Render a styled KPI metric card."""
    st.metric(label, value, delta=delta, delta_color=delta_color, help=help_text)


def render_kpi_row(
    kpis: List[Dict[str, Any]],
    columns: int = 5,
) -> None:
    """Render a row of KPI cards.

    Each kpi dict should have: label, value, delta (optional), delta_color (optional), help (optional)
    """
    cols = st.columns(columns)
    for i, kpi in enumerate(kpis):
        with cols[i % columns]:
            render_kpi_card(
                label=kpi.get("label", ""),
                value=kpi.get("value", "—"),
                delta=kpi.get("delta"),
                delta_color=kpi.get("delta_color", "normal"),
                help_text=kpi.get("help"),
            )


# =============================================================================
# SECTION HEADER
# =============================================================================

def render_section_label(label: str) -> None:
    """Render a section label (uppercase, muted)."""
    st.markdown(
        f"""
        <div style="
            color: #697386;
            text-transform: uppercase;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin: 22px 0 8px 0;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_hero(title: str, description: str, kicker: str = "") -> None:
    """Render the page hero section."""
    kicker_html = f"<div style='color: #315efb; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 8px;'>{kicker}</div>" if kicker else ""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #ffffff 0%, #f6f8ff 100%);
            border: 1px solid #e4e7ec;
            border-radius: 20px;
            padding: 28px 32px;
            margin-bottom: 20px;
        ">
            {kicker_html}
            <h1 style="margin: 0 0 8px 0; color: #172033; font-size: 2.25rem; line-height: 1.1; letter-spacing: -0.03em;">
                {title}
            </h1>
            <p style="margin: 0; color: #697386; max-width: 980px; font-size: 1rem; line-height: 1.6;">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# SCIENCE CARDS
# =============================================================================

def render_science_card(title: str, body: str, icon: str = "") -> None:
    """Render a science/info card."""
    icon_html = f"<span style='margin-right: 8px;'>{icon}</span>" if icon else ""
    st.markdown(
        f"""
        <div style="
            background: #ffffff;
            border: 1px solid #e4e7ec;
            border-radius: 14px;
            padding: 18px 20px;
            height: 100%;
        ">
            <h4 style="margin: 0 0 8px 0; color: #172033; font-size: 1rem;">{icon_html}{title}</h4>
            <p style="margin: 0; color: #697386; line-height: 1.55; font-size: 0.92rem;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_method_step(number: str, title: str, body: str) -> None:
    """Render a methodology step."""
    st.markdown(
        f"""
        <div style="border-left: 3px solid #315efb; padding: 9px 0 9px 14px; margin: 8px 0;">
            <strong style="color: #172033;">{number} · {title}</strong>
            <div style="color: #697386; font-size: 0.85rem; line-height: 1.5; margin-top: 4px;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# CUSTOMER HEADER
# =============================================================================

def render_customer_header(
    customer_id: int,
    segment: Optional[str] = None,
    action: Optional[str] = None,
    lifecycle_state: Optional[str] = None,
    cohort: Optional[str] = None,
) -> None:
    """Render the customer profile header."""
    parts = []
    if segment:
        parts.append(str(segment))
    if lifecycle_state:
        parts.append(str(lifecycle_state))
    if cohort:
        parts.append(f"Cohort: {cohort}")

    subtitle = " · ".join(parts) if parts else "No segment or lifecycle info"
    if action:
        subtitle += f" · Action: {action}"

    st.markdown(
        f"""
        <div style="
            background: #fff;
            border: 1px solid #e4e7ec;
            border-radius: 16px;
            padding: 20px 22px;
            margin-bottom: 16px;
        ">
            <div style="color: #315efb; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 4px;">
                Customer profile
            </div>
            <h2 style="margin: 0; color: #172033;">Customer {customer_id}</h2>
            <p style="margin: 5px 0 0 0; color: #697386;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# EVIDENCE TABLE
# =============================================================================

def render_evidence_table(rows: List[Dict[str, str]]) -> None:
    """Render a formatted evidence table."""
    if not rows:
        st.info("No evidence available.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def build_evidence_rows(
    row: pd.Series,
    field_map: List[tuple],
) -> List[Dict[str, str]]:
    """Build evidence rows from a customer row using field mapping.

    field_map: List of (label, candidates, formatter) where formatter is a formatting function.
    """
    from app_formatting import (
        format_currency, format_probability, format_count,
        format_score, format_date, format_duration_months
    )

    def find_value(candidates):
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                return row[c]
        return None

    evidence = []
    for label, candidates, fmt_func in field_map:
        value = find_value(candidates)
        if value is None:
            continue
        try:
            formatted = fmt_func(value)
        except Exception:
            formatted = str(value)
        evidence.append({"Metric": label, "Value": formatted})

    return evidence


# =============================================================================
# DATA FRESHNESS INDICATOR
# =============================================================================

def render_data_freshness(registry: ArtifactRegistry) -> None:
    """Render a data freshness indicator in the sidebar."""
    st.sidebar.markdown("### Data Freshness")

    for module in ["customer_360", "segmentation", "cohorts", "clv", "churn", "decision_engine"]:
        status = registry.get_module_status(module)
        if status and status.primary_artifact:
            artifact = status.primary_artifact
            if artifact.generated_at:
                age_days = (datetime.now() - artifact.generated_at).days
                if age_days == 0:
                    freshness = "Today"
                elif age_days == 1:
                    freshness = "Yesterday"
                elif age_days < 7:
                    freshness = f"{age_days}d ago"
                else:
                    freshness = f"{age_days}d ago ⚠️"
            elif artifact.modified_at:
                age_days = (datetime.now() - artifact.modified_at).days
                freshness = f"{age_days}d ago (file mtime)"
            else:
                freshness = "Unknown"

            label = MODULE_LABELS.get(module, module)
            st.sidebar.markdown(f"**{label}**: {freshness}")
        else:
            label = MODULE_LABELS.get(module, module)
            st.sidebar.markdown(f"**{label}**: Not generated")


# =============================================================================
# RUN CONSISTENCY WARNING
# =============================================================================

def render_run_consistency_warning(registry: ArtifactRegistry) -> None:
    """Render a warning if modules come from different runs."""
    statuses = registry.get_all_module_statuses()

    run_ids = {}
    for module, status in statuses.items():
        if status.primary_artifact and status.primary_artifact.run_id:
            run_ids.setdefault(status.primary_artifact.run_id, []).append(module)

    if len(run_ids) > 1:
        st.warning(
            f"⚠️ **Analytical outputs were generated from {len(run_ids)} different runs.** "
            f"Cross-model comparisons may not be consistent. "
            f"Run groups: {', '.join([f'{rid} ({len(mods)} modules)' for rid, mods in run_ids.items()])}"
        )
    elif len(run_ids) == 1:
        run_id = list(run_ids.keys())[0]
        modules = run_ids[run_id]
        st.success(f"✅ All {len(modules)} modules share run ID: `{run_id}`")


# =============================================================================
# DOWNLOAD BUTTON
# =============================================================================

def render_download_button(
    df: pd.DataFrame,
    filename: str,
    label: str = "Download CSV",
    help_text: Optional[str] = None,
) -> None:
    """Render a download button for a DataFrame."""
    csv = df.to_csv(index=False)
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        help=help_text,
        use_container_width=True,
    )


# =============================================================================
# EXPANDER WITH SCIENTIFIC CONTEXT
# =============================================================================

def render_scientific_context(title: str, content: str, expanded: bool = False) -> None:
    """Render an expander with scientific context/methodology."""
    with st.expander(f"📖 {title}", expanded=expanded):
        st.markdown(content)


# =============================================================================
# MISSING DATA PLACEHOLDER
# =============================================================================

def render_missing_data(message: str = "Required output is not available yet.", action: Optional[str] = None) -> None:
    """Render a consistent missing data placeholder."""
    if action:
        st.info(f"{message} {action}")
    else:
        st.info(message)


# =============================================================================
# TABS WITH EMPTY STATE HANDLING
# =============================================================================

def render_tabbed_content(
    tabs_config: List[Dict[str, Any]],
    default_tab: int = 0,
) -> int:
    """Render tabbed content with consistent empty state handling.

    tabs_config: List of {label, content_fn, empty_message}
    Returns the selected tab index.
    """
    tab_labels = [t["label"] for t in tabs_config]
    tabs = st.tabs(tab_labels)

    for i, (tab, config) in enumerate(zip(tabs, tabs_config)):
        with tab:
            try:
                config["content_fn"]()
            except Exception as e:
                st.error(f"Error rendering {config['label']}: {e}")
                if config.get("empty_message"):
                    render_missing_data(config["empty_message"])

    return default_tab


# =============================================================================
# CUSTOMER SEARCH / SELECTOR
# =============================================================================

def render_customer_selector(
    customer_df: pd.DataFrame,
    key_prefix: str = "customer",
    default_customer: Optional[int] = None,
) -> Optional[int]:
    """Render a customer search/selector widget.

    Returns the selected customer ID or None.
    """
    if customer_df is None or customer_df.empty:
        st.warning("No customer data available.")
        return None

    id_col = "Customer ID"
    if id_col not in customer_df.columns:
        for c in ["customer_id", "CustomerID"]:
            if c in customer_df.columns:
                id_col = c
                break

    ids = customer_df[id_col].dropna().astype(int).tolist()

    default_index = 0
    if default_customer is not None and default_customer in ids:
        default_index = ids.index(default_customer)

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        selected_id = st.selectbox(
            "Select Customer",
            options=ids,
            format_func=lambda x: f"Customer {x}",
            index=default_index,
            key=f"{key_prefix}_select",
        )

    with col2:
        search = st.text_input(
            "Or search by ID",
            value="",
            placeholder="e.g. 12345",
            key=f"{key_prefix}_search",
        )

        if search.strip().isdigit():
            search_id = int(search.strip())
            if search_id in set(ids):
                selected_id = search_id

    return selected_id


# =============================================================================
# FILTER BAR
# =============================================================================

def render_filter_bar(
    filters: Dict[str, Any],
    key_prefix: str = "filter",
) -> Dict[str, Any]:
    """Render a horizontal filter bar and return updated filter values."""
    cols = st.columns(len(filters))
    updated = {}

    for i, (name, config) in enumerate(filters.items()):
        with cols[i]:
            filter_type = config.get("type", "select")

            if filter_type == "select":
                options = config.get("options", [])
                default = config.get("default", options[0] if options else None)
                updated[name] = st.selectbox(
                    config.get("label", name),
                    options=options,
                    index=options.index(default) if default in options else 0,
                    key=f"{key_prefix}_{name}",
                )
            elif filter_type == "multiselect":
                options = config.get("options", [])
                default = config.get("default", [])
                updated[name] = st.multiselect(
                    config.get("label", name),
                    options=options,
                    default=default,
                    key=f"{key_prefix}_{name}",
                )
            elif filter_type == "slider":
                min_val = config.get("min", 0)
                max_val = config.get("max", 100)
                default = config.get("default", (min_val, max_val))
                updated[name] = st.slider(
                    config.get("label", name),
                    min_value=min_val,
                    max_value=max_val,
                    value=default,
                    key=f"{key_prefix}_{name}",
                )
            elif filter_type == "number":
                updated[name] = st.number_input(
                    config.get("label", name),
                    min_value=config.get("min"),
                    max_value=config.get("max"),
                    value=config.get("default", 0),
                    key=f"{key_prefix}_{name}",
                )

    return updated