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

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path for both `python -m streamlit run` and `streamlit run`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import new UI design system
from streamlit_app.ui import (
    inject_global_css,
    get_current_theme,
    render_sidebar,
    render_page_hero,
    HERO_COPY,
)

from streamlit_app.app_data import get_registry

# Page modules
from streamlit_app.pages import (
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

# Inject global CSS with theme support
inject_global_css()


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> str:
    """Render the global sidebar with pipeline status."""
    st.sidebar.markdown("### Retail Customer Intelligence")
    
    registry = get_registry()
    statuses = registry.get_all_module_statuses()
    
    st.sidebar.markdown("---")
    
    # Module status
    with st.sidebar.expander("Pipeline status", expanded=True):
        for module, status in statuses.items():
            from streamlit_app.ui import get_status_color
            color = get_status_color(status.overall_status)
            chip = (
                f"<span style='display:inline-flex;align-items:center;gap:4px;"
                f"padding:5px 10px;border-radius:999px;"
                f"background:{color}1A;color:{color};"
                f"font-size:0.76rem;font-weight:700;margin:2px;'>"
                f"{'✓' if status.overall_status == 'ready' else '○' if status.overall_status == 'incomplete' else '⟳' if status.overall_status == 'stale' else '✗' if status.overall_status == 'validation_failed' else '−' if status.overall_status == 'unavailable' else '?'}"
                f" {module.replace('_', ' ').title()}: {status.overall_status.title()}"
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

def main() -> None:
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