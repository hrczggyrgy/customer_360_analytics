#!/usr/bin/env python3
"""
Retail Customer Intelligence — Streamlit App.

A clean decision-science interface for the project's analytical workflow:

    Data -> Customer 360 -> Segmentation -> Cohorts -> CLV
         -> Churn / Next Purchase -> Recommendations -> Decision Engine

The app uses a config-driven ArtifactRegistry for robust artifact discovery
and consistent scientific labeling.
"""

from __future__ import annotations

import streamlit as st

from .app_components import (
    render_page_hero,
    render_section_label,
    render_status_chip,
    render_customer_header,
    render_customer_metric_row,
    render_customer_selector,
    render_kpi_card,
    render_science_card,
    render_missing,
    render_action_summary_table,
    render_evidence_table,
    HERO_COPY,
)
from .app_data import get_registry, format_freshness
from .app_formatting import (
    format_currency,
    format_probability,
    format_percent,
    format_ratio,
    format_count,
    format_month,
    format_duration_months,
    format_score,
    auto_format,
)
from .app_charts import (
    base_layout,
    plot_missing,
    plot_histogram_with_marginal,
    plot_horizontal_bar,
    plot_scatter,
    plot_heatmap,
    plot_retention_matrix,
    plot_decay_curve,
    plot_uncertainty_band,
    plot_pca_scatter,
    plot_concentration_curve,
    plot_segment_heatmap,
    plot_action_allocation,
    plot_clv_by_segment,
    plot_decision_scatter,
    plot_priority_distribution,
    plot_expected_value_by_action,
    plot_clv_vs_churn_by_action,
)

# Page modules
from .pages import (
    executive,
    customer_360,
    segmentation,
    cohorts,
    predictive,
    retention,
    recommendations,
    decision_engine,
    methodology,
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

# Load CSS
st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #697386;
        --line: #e4e7ec;
        --panel: #ffffff;
        --panel-soft: #f7f8fa;
        --accent: #315efb;
        --accent-soft: #eef2ff;
        --success: #218739;
        --warning: #a56600;
        --danger: #e11d48;
    }

    .stApp {
        background: #fbfcfe;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #f7f8fa;
        border-right: 1px solid var(--line);
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(23, 32, 51, 0.03);
    }

    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #f6f8ff 100%);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }

    .hero h1 {
        margin: 0 0 8px 0;
        color: var(--ink);
        font-size: 2.25rem;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }

    .hero p {
        margin: 0;
        color: var(--muted);
        max-width: 980px;
        font-size: 1rem;
        line-height: 1.6;
    }

    .section-label {
        color: var(--muted);
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin: 22px 0 8px 0;
    }

    .science-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
    }

    .science-card h4 {
        margin: 0 0 8px 0;
        color: var(--ink);
        font-size: 1rem;
    }

    .science-card p {
        margin: 0;
        color: var(--muted);
        line-height: 1.55;
        font-size: 0.92rem;
    }

    .status-chip {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .kicker {
        color: var(--accent);
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .customer-header {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }

    .customer-header h2 {
        margin: 0;
        color: var(--ink);
    }

    .customer-header p {
        margin: 5px 0 0 0;
        color: var(--muted);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the global sidebar with pipeline status."""
    st.sidebar.markdown("### Retail Customer Intelligence")
    
    registry = get_registry()
    statuses = registry.get_all_module_statuses()
    
    st.sidebar.markdown("---")
    
    # Module status
    with st.sidebar.expander("Pipeline status", expanded=True):
        for module, status in statuses.items():
            chip = (
                f"<span style='display:inline-block;padding:5px 10px;border-radius:999px;"
                f"background:{status.overall_status}15;color:{status.overall_status};"
                f"font-size:0.76rem;font-weight:700;margin:2px;'>"
                f"{module.replace('_', ' ').title()}: {status.overall_status.title()}"
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
# MAIN
# =============================================================================

def main():
    """Main app entry point."""
    # Page config is set at module level
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Get hero copy
    kicker, description = HERO_COPY.get(page, ("", ""))
    
    # Render hero
    render_page_hero(kicker, "Retail Customer Intelligence", description)
    
    # Route to page module
    page_modules = {
        "Executive": executive,
        "Customer 360": customer_360,
        "Segmentation": segmentation,
        "Cohorts": cohorts,
        "Predictive Value": predictive,
        "Retention & Next Purchase": retention,
        "Recommendations": recommendations,
        "Decision Engine": decision_engine,
        "Methodology": methodology,
    }
    
    page_module = page_modules.get(page)
    if page_module:
        page_module.render()
    else:
        st.error(f"Page module not found: {page}")


if __name__ == "__main__":
    main()