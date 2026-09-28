#!/usr/bin/env python3
"""
Retail Customer Intelligence — Streamlit app (Modular Architecture).

A clean decision-science interface for the project's analytical workflow:

    Data -> Customer 360 -> Segmentation -> Cohorts -> Predictive Value
         -> Retention & Next Purchase -> Recommendations -> Decision Engine
         -> Methodology

The app uses a config-driven artifact registry for robust, version-aware data loading.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from app_config import get_config, MODULE_LABELS, get_project_root
from app_data import get_registry, reset_registry
from app_components import (
    render_page_hero,
    render_run_consistency_warning,
    render_data_freshness,
)
from pages import (
    render_executive_page,
    render_customer_360_page,
    render_segmentation_page,
    render_cohorts_page,
    render_predictive_page,
    render_retention_page,
    render_recommendations_page,
    render_decision_page,
    render_methodology_page,
)


# =============================================================================
# APP CONFIG
# =============================================================================

st.set_page_config(
    page_title="Retail Customer Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load configuration
config = get_config()

# Apply theme CSS
st.markdown(
    f"""
    <style>
    :root {{
        --ink: #172033;
        --muted: #697386;
        --line: #e4e7ec;
        --panel: #ffffff;
        --panel-soft: #f7f8fa;
        --accent: #315efb;
        --accent-soft: #eef2ff;
        --success: #218739;
        --warning: #a56600;
        --danger: #c53d32;
    }}

    .stApp {{
        background: #fbfcfe;
    }}

    .block-container {{
        max-width: 1480px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }}

    [data-testid="stSidebar"] {{
        background: #f7f8fa;
        border-right: 1px solid var(--line);
    }}

    [data-testid="stMetric"] {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(23, 32, 51, 0.03);
    }}

    .hero {{
        background: linear-gradient(135deg, #ffffff 0%, #f6f8ff 100%);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }}

    .hero h1 {{
        margin: 0 0 8px 0;
        color: var(--ink);
        font-size: 2.25rem;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }}

    .hero p {{
        margin: 0;
        color: var(--muted);
        max-width: 980px;
        font-size: 1rem;
        line-height: 1.6;
    }}

    .section-label {{
        color: var(--muted);
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin: 22px 0 8px 0;
    }}

    .science-card {{
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
    }}

    .science-card h4 {{
        margin: 0 0 8px 0;
        color: var(--ink);
        font-size: 1rem;
    }}

    .science-card p {{
        margin: 0;
        color: var(--muted);
        line-height: 1.55;
        font-size: 0.92rem;
    }}

    .status-chip {{
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }}

    .method-step {{
        border-left: 3px solid var(--accent);
        padding: 9px 0 9px 14px;
        margin: 8px 0;
    }}

    .method-step strong {{
        color: var(--ink);
    }}

    .small-note {{
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.45;
    }}

    .kicker {{
        color: var(--accent);
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}

    .customer-header {{
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }}

    .customer-header h2 {{
        margin: 0;
        color: var(--ink);
    }}

    .customer-header p {{
        margin: 5px 0 0 0;
        color: var(--muted);
    }}

    div[data-testid="stDataFrame"] {{
        border-radius: 12px;
        overflow: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the application sidebar with navigation and project status."""
    registry = get_registry()

    st.sidebar.markdown("### Retail Customer Intelligence")

    if st.sidebar.button("Refresh Project Data", use_container_width=True):
        reset_registry()
        st.cache_data.clear()
        st.rerun()

    statuses = registry.get_all_module_statuses()

    available_count = sum(1 for s in statuses.values() if s.overall_status == "ready")

    st.sidebar.markdown(f"**Analytical modules ready:** {available_count} / {len(MODULE_LABELS)}")

    # Pipeline status
    with st.sidebar.expander("Pipeline Status", expanded=True):
        for module, label in MODULE_LABELS.items():
            status = statuses.get(module)
            if status:
                status_text = status.overall_status.title()
                freshness = status.freshness_note
                run_consistency = status.run_consistency
            else:
                status_text = "Unknown"
                freshness = ""
                run_consistency = "unknown"

            # Color code
            color_map = {
                "ready": "#218739",
                "incomplete": "#a56600",
                "stale": "#a56600",
                "validation_failed": "#c53d32",
                "unavailable": "#697386",
            }
            color = color_map.get(status.overall_status if status else "unknown", "#697386")

            st.sidebar.markdown(
                f"""
                <div style="margin: 4px 0;">
                    <span style="
                        display: inline-block;
                        padding: 3px 8px;
                        border-radius: 999px;
                        background: {color}15;
                        color: {color};
                        font-size: 0.7rem;
                        font-weight: 700;
                        margin-right: 6px;
                    ">
                        {status_text}
                    </span>
                    <span style="font-size: 0.85rem; color: #172033;">{label}</span>
                    {f'<br><span style="font-size: 0.7rem; color: #697386;">{freshness}</span>' if freshness else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Data freshness
    with st.sidebar.expander("Data Freshness", expanded=False):
        render_data_freshness(registry)

    # Navigation
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

    page = st.sidebar.radio("Navigate", pages, key="page_nav")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Observational retail analytics. Decision scores are not causal uplift estimates. "
        "Predicted future net revenue is a discounted revenue proxy (no margin data)."
    )

    return page


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize registry
    registry = get_registry()
    registry.discover()

    # Render sidebar and get selected page
    page = render_sidebar()

    # Render run consistency warning at top of main area
    render_run_consistency_warning(registry)

    # Page routing
    page_map = {
        "Executive": render_executive_page,
        "Customer 360": render_customer_360_page,
        "Segmentation": render_segmentation_page,
        "Cohorts": render_cohorts_page,
        "Predictive Value": render_predictive_page,
        "Retention & Next Purchase": render_retention_page,
        "Recommendations": render_recommendations_page,
        "Decision Engine": render_decision_page,
        "Methodology": render_methodology_page,
    }

    render_func = page_map.get(page)
    if render_func:
        try:
            render_func()
        except Exception as e:
            st.error(f"Error rendering {page}: {e}")
            logging.exception(f"Page render error: {page}")
    else:
        st.error(f"Unknown page: {page}")

    # Footer
    st.markdown(
        """
        <div style='margin-top: 36px; padding-top: 14px; border-top: 1px solid #e4e7ec;'>
            <div style='color: #697386; font-size: 0.85rem;'>
                Retail Customer Intelligence · Online Retail II · Analytical Decision Support
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()