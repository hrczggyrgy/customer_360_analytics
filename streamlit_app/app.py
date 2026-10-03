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

from streamlit_app.ui.scope import render_global_filter_bar, render_sidebar

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
# MAIN
# =============================================================================

def main() -> None:
    """Main app entry point."""
    from streamlit_app.ui.scope import get_scope
    get_scope()
    
    workspace = render_sidebar()
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