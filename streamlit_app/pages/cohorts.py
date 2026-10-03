"""
Cohorts page for Retail Customer Intelligence.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..ui import (
    render_section_label,
    render_science_card,
    render_kpi_card,
    render_kpi_row,
    render_missing,
)
from ..ui.scope import apply_scope_to_dataframe
from ..app_data import get_registry
from ..app_formatting import format_count, format_percent
from ..ui.charts import (
    plot_missing,
    plot_retention_matrix,
    plot_decay_curve,
    PLOTLY_CONFIG,
)


def render() -> None:
    """Render the Cohorts page."""
    registry = get_registry()
    
    logo_file = registry.load_dataframe("cohorts", "matrix_logo_retention.csv")
    nrr_file = registry.load_dataframe("cohorts", "matrix_net_revenue_retention.csv")
    decay = registry.load_dataframe("cohorts", "retention_decay_curve.csv")
    scorecard = registry.load_dataframe("cohorts", "cohort_scorecard.csv")
    
    if all(x is None for x in [logo_file, nrr_file, decay, scorecard]):
        from ..ui import render_empty_state
        render_empty_state(
            "No cohort data",
            "No cohort outputs found.",
            "Run the pipeline",
            "python scripts/cohort_analysis.py",
        )
        st.stop()
    
    # KPIs
    retention_3 = None
    if decay is not None:
        age_col = "age_month" if "age_month" in decay.columns else None
        logo_col = "weighted_logo_retention" if "weighted_logo_retention" in decay.columns else "logo_retention"
        if age_col and logo_col and logo_col in decay.columns:
            match = decay[pd.to_numeric(decay[age_col], errors="coerce") == 3]
            if not match.empty:
                retention_3 = float(pd.to_numeric(match[logo_col], errors="coerce").iloc[0])
    
    nrr_6 = None
    if decay is not None:
        age_col = "age_month" if "age_month" in decay.columns else None
        nrr_col = "weighted_net_revenue_retention" if "weighted_net_revenue_retention" in decay.columns else "net_revenue_retention"
        if age_col and nrr_col and nrr_col in decay.columns:
            match = decay[pd.to_numeric(decay[age_col], errors="coerce") == 6]
            if not match.empty:
                nrr_6 = float(pd.to_numeric(match[nrr_col], errors="coerce").iloc[0])
    
    max_age = None
    if decay is not None:
        age_col = "age_month" if "age_month" in decay.columns else None
        if age_col:
            max_age = int(pd.to_numeric(decay[age_col], errors="coerce").max())
    
    n_cohorts = scorecard["cohort_month"].nunique() if scorecard is not None and "cohort_month" in scorecard.columns else None
    
    render_kpi_row([
        {"label": "Acquisition cohorts", "value": n_cohorts, "formatter": "count"},
        {"label": "Month-3 logo retention", "value": retention_3, "formatter": "percent"},
        {"label": "Month-6 revenue retention", "value": nrr_6, "formatter": "percent"},
        {"label": "Observed age horizon", "value": max_age, "formatter": "count"},
    ])
    
    tabs = st.tabs(["Logo retention", "Revenue retention", "Maturity curve", "Cohort scorecard"])
    
    with tabs[0]:
        if logo_file is not None:
            fig = plot_retention_matrix(logo_file, title="Logo retention cohort matrix")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            render_missing()
    
    with tabs[1]:
        if nrr_file is not None:
            fig = plot_retention_matrix(
                nrr_file,
                title="Net revenue retention cohort matrix",
                color_scale="Viridis",
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            render_missing()
    
    with tabs[2]:
        if decay is not None:
            age_col = "age_month" if "age_month" in decay.columns else None
            logo_col = "weighted_logo_retention" if "weighted_logo_retention" in decay.columns else "logo_retention"
            nrr_col = "weighted_net_revenue_retention" if "weighted_net_revenue_retention" in decay.columns else "net_revenue_retention"
            
            if age_col:
                fig = plot_decay_curve(
                    decay,
                    age_col=age_col,
                    logo_col=logo_col if logo_col in decay.columns else None,
                    nrr_col=nrr_col if nrr_col in decay.columns else None,
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                
                render_science_card(
                    "Why the curve matters",
                    "The cohort matrix shows individual acquisition vintages; "
                    "the maturity curve aggregates customers at the same age. "
                    "This separates lifecycle decay from calendar-time effects "
                    "and prevents newer cohorts from being treated as if they "
                    "had already had time to mature."
                )
        else:
            render_missing()
    
    with tabs[3]:
        if scorecard is not None:
            st.dataframe(scorecard, use_container_width=True, hide_index=True)
        else:
            render_missing()