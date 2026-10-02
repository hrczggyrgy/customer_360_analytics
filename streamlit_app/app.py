#!/usr/bin/env python3
"""
Retail Customer Intelligence — Streamlit App.

A strategic customer intelligence cockpit for retail decision-making.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path for both `python -m streamlit run` and `streamlit run`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import UI design system
from streamlit_app.ui import (
    inject_global_css,
    render_sidebar,
    HERO_COPY,
)

from streamlit_app.app_data import get_registry

# Page modules - new workspace structure
from streamlit_app.pages import (
    strategy,
    customers,
    value_retention,
    segments,
    products_baskets,
    personalisation,
    activation,
    science,
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
# GLOBAL SCOPE / FILTER STATE
# =============================================================================

def init_global_scope() -> None:
    """Initialize global scope state in session."""
    if "global_scope" not in st.session_state:
        st.session_state.global_scope = {
            "as_of_date": None,
            "country": "All",
            "segment": "All",
            "lifecycle": "All",
            "value_band": "All",
            "risk_band": "All",
            "rfm_segment": "All",
        }
    
    if "scope_dirty" not in st.session_state:
        st.session_state.scope_dirty = False


def render_global_filter_bar() -> None:
    """Render the global filter bar at the top of the app."""
    registry = get_registry()
    combined = registry.load_dataframe("customer_360")
    
    # Get available filter options from data
    countries = ["All"]
    segments_list = ["All"]
    lifecycles = ["All"]
    value_bands = ["All"]
    risk_bands = ["All"]
    rfm_segments = ["All"]
    
    if combined is not None:
        if "primary_country" in combined.columns:
            countries = ["All"] + sorted(combined["primary_country"].dropna().unique().tolist())
        if "segment_name" in combined.columns:
            segments_list = ["All"] + sorted([s for s in combined["segment_name"].dropna().unique() if s and s != "unknown"])
        if "lifecycle_state" in combined.columns:
            lifecycles = ["All"] + sorted(combined["lifecycle_state"].dropna().unique().tolist())
        if "value_tier" in combined.columns:
            value_bands = ["All"] + sorted(combined["value_tier"].dropna().unique().tolist())
        if "action_priority_tier" in combined.columns:
            risk_bands = ["All"] + sorted(combined["action_priority_tier"].dropna().unique().tolist())
    
    # Render filter bar
    with st.container():
        cols = st.columns([1.5, 1, 1, 1, 1, 1, 1, 0.5])
        
        scope = st.session_state.global_scope
        
        with cols[0]:
            scope["as_of_date"] = st.date_input(
                "As of",
                value=scope["as_of_date"],
                key="scope_as_of_date",
                help="Analysis reference date",
            )
        
        with cols[1]:
            scope["country"] = st.selectbox(
                "Country",
                options=countries,
                index=countries.index(scope["country"]) if scope["country"] in countries else 0,
                key="scope_country",
            )
        
        with cols[2]:
            scope["segment"] = st.selectbox(
                "Segment",
                options=segments_list,
                index=segments_list.index(scope["segment"]) if scope["segment"] in segments_list else 0,
                key="scope_segment",
            )
        
        with cols[3]:
            scope["lifecycle"] = st.selectbox(
                "Lifecycle",
                options=lifecycles,
                index=lifecycles.index(scope["lifecycle"]) if scope["lifecycle"] in lifecycles else 0,
                key="scope_lifecycle",
            )
        
        with cols[4]:
            scope["value_band"] = st.selectbox(
                "Value band",
                options=value_bands,
                index=value_bands.index(scope["value_band"]) if scope["value_band"] in value_bands else 0,
                key="scope_value_band",
            )
        
        with cols[5]:
            scope["risk_band"] = st.selectbox(
                "Risk band",
                options=risk_bands,
                index=risk_bands.index(scope["risk_band"]) if scope["risk_band"] in risk_bands else 0,
                key="scope_risk_band",
            )
        
        with cols[6]:
            scope["rfm_segment"] = st.selectbox(
                "RFM segment",
                options=rfm_segments,
                index=rfm_segments.index(scope["rfm_segment"]) if scope["rfm_segment"] in rfm_segments else 0,
                key="scope_rfm_segment",
            )
        
        with cols[7]:
            st.write("")
            if st.button("Clear", key="scope_clear", help="Clear all filters"):
                for key in scope:
                    if key == "as_of_date":
                        scope[key] = None
                    else:
                        scope[key] = "All"
                st.session_state.scope_dirty = True
                st.rerun()
    
    # Show active scope summary
    active_filters = []
    if scope["as_of_date"]:
        active_filters.append(f"As of: {scope['as_of_date']}")
    if scope["country"] != "All":
        active_filters.append(f"Country: {scope['country']}")
    if scope["segment"] != "All":
        active_filters.append(f"Segment: {scope['segment']}")
    if scope["lifecycle"] != "All":
        active_filters.append(f"Lifecycle: {scope['lifecycle']}")
    if scope["value_band"] != "All":
        active_filters.append(f"Value: {scope['value_band']}")
    if scope["risk_band"] != "All":
        active_filters.append(f"Risk: {scope['risk_band']}")
    if scope["rfm_segment"] != "All":
        active_filters.append(f"RFM: {scope['rfm_segment']}")
    
    scope_text = " · ".join(active_filters) if active_filters else "All customers · All segments · All lifecycles"
    st.caption(f"Scope: {scope_text}")
    
    st.markdown("---")


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> str:
    """Render the global sidebar with workspace navigation."""
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
    
    # Workspace navigation
    workspaces = [
        "Strategy",
        "Customers",
        "Value & Retention",
        "Segments",
        "Products & Baskets",
        "Personalisation",
        "Activation",
        "Science & Governance",
    ]
    
    page = st.sidebar.radio("Workspace", workspaces, key="nav_workspace")
    
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
    init_global_scope()
    
    # Render sidebar and get selected workspace
    workspace = render_sidebar()
    
    # Render global filter bar
    render_global_filter_bar()
    
    # Get hero copy
    kicker, description = HERO_COPY.get(workspace, ("", ""))
    
    # Render hero
    from streamlit_app.ui import render_page_hero
    render_page_hero(kicker, "Retail Customer Intelligence", description)
    
    # Route to workspace module
    workspace_modules = {
        "Strategy": strategy,
        "Customers": customers,
        "Value & Retention": value_retention,
        "Segments": segments,
        "Products & Baskets": products_baskets,
        "Personalisation": personalisation,
        "Activation": activation,
        "Science & Governance": science,
    }
    
    workspace_module = workspace_modules.get(workspace)
    if workspace_module:
        workspace_module.render()
    else:
        st.error(f"Workspace module not found: {workspace}")


if __name__ == "__main__":
    main()