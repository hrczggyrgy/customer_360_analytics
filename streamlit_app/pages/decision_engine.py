"""
Decision Engine page for Retail Customer Intelligence.
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
    render_action_summary_table,
    render_formatted_dataframe,
)
from ..app_data import get_registry
from ..app_formatting import format_currency, format_probability, format_count
from ..ui.charts import (
    plot_missing,
    plot_action_allocation,
    plot_decision_scatter,
    plot_priority_distribution,
    plot_expected_value_by_action,
    plot_clv_vs_churn_by_action,
    PLOTLY_CONFIG,
)


def render() -> None:
    """Render the Decision Engine page."""
    registry = get_registry()
    
    decision = registry.load_dataframe("decision_engine")
    action_summary = registry.load_dataframe("decision_engine", "action_summary.csv")
    
    if decision is None:
        from ..ui import render_empty_state
        render_empty_state(
            "No decision engine data",
            "No decision-engine output found.",
            "Run the pipeline",
            "python scripts/decision_engine.py",
        )
        st.stop()
    
    action_col = "recommended_action_capped" if "recommended_action_capped" in decision.columns else None
    for c in ["final_action", "recommended_action", "action"]:
        if c in decision.columns:
            action_col = c
            break
    
    priority_col = "priority_score" if "priority_score" in decision.columns else None
    decision_clv = "clv_mean" if "clv_mean" in decision.columns else None
    for c in ["clv", "predicted_clv", "customer_clv"]:
        if c in decision.columns:
            decision_clv = c
            break
    d_churn = "churn_probability" if "churn_probability" in decision.columns else None
    for c in ["churn_prob", "prob_churn"]:
        if c in decision.columns:
            d_churn = c
            break
    d_next = "next_purchase_probability" if "next_purchase_probability" in decision.columns else None
    for c in ["next_purchase_probability_30d", "purchase_probability_30d"]:
        if c in decision.columns:
            d_next = c
            break
    d_id = "Customer ID" if "Customer ID" in decision.columns else None
    
    if not action_col:
        st.error("Decision file found but no action field was detected.")
        st.stop()
    
    # KPIs
    active_mask = ~decision[action_col].astype(str).str.startswith("monitor", na=False)
    median_priority = pd.to_numeric(decision[priority_col], errors="coerce").median() if priority_col else None
    total_clv = pd.to_numeric(decision[decision_clv], errors="coerce").sum() if decision_clv else None
    
    render_kpi_row([
        {"label": "Customers scored", "value": len(decision), "formatter": "count"},
        {"label": "Active policy rows", "value": int(active_mask.sum()), "formatter": "count"},
        {"label": "Median priority", "value": median_priority, "formatter": "score", "column_name": "priority_score"},
        {"label": "CLV represented", "value": total_clv, "formatter": "currency"},
    ])
    
    render_section_label("Action allocation")
    
    left, right = st.columns([0.95, 1.05])
    
    with left:
        if action_col:
            action_counts = decision[action_col].value_counts().reset_index()
            action_counts.columns = [action_col, "count"]
            fig = plot_action_allocation(action_counts, action_col=action_col, count_col="count")
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            render_missing("Action column not found.")
    
    with right:
        if decision_clv and d_churn and priority_col:
            fig = plot_clv_vs_churn_by_action(decision, clv_col=decision_clv, churn_col=d_churn, action_col=action_col)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            render_missing("CLV, churn, and priority fields are needed for the decision scatter.")
    
    render_section_label("Top commercial targets")
    
    top_n = st.slider("Customers", min_value=10, max_value=200, value=50, step=10, key="decision_top_n")
    
    if priority_col:
        targets = decision.copy()
        targets[priority_col] = pd.to_numeric(targets[priority_col], errors="coerce")
        targets = targets.sort_values(priority_col, ascending=False).head(top_n)
    else:
        targets = decision.head(top_n)
    
    preferred_columns = []
    for c in [d_id, action_col, "priority_tier", priority_col, decision_clv, d_churn, d_next,
              "reactivation_probability", "decision_confidence", "action_reason"]:
        if c and c in targets.columns:
            preferred_columns.append(c)
    
    display = targets[preferred_columns].copy()
    
    # Use formatted dataframe
    render_formatted_dataframe(display)
    
    render_science_card(
        "What this engine does — and does not do",
        "It estimates **expected commercial opportunity** from the "
        "analytical signals already produced in the project. It then applies a "
        "transparent policy with capacity constraints. It does not infer an "
        "incremental treatment effect from observational purchase history. "
        "The next portfolio upgrade is an experiment or quasi-experiment that "
        "measures actual incremental response to each intervention."
    )
    
    if action_summary is not None:
        with st.expander("View exported action summary"):
            render_action_summary_table(action_summary)