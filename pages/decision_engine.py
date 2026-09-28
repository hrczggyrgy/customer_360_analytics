"""
Decision Engine page — Commercial prioritization with capacity-aware allocation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_components import (
    render_page_hero,
    render_section_label,
    render_kpi_row,
    render_science_card,
    render_missing_data,
    render_download_button,
)
from app_charts import (
    plot_action_allocation,
    plot_decision_scatter,
    render_chart_responsive,
)
from app_data import get_registry
from app_formatting import format_count, format_clv, format_score, format_churn_risk


def render_decision_page() -> None:
    """Render the Decision Engine page."""
    registry = get_registry()

    decision = registry.load_dataframe("decision_engine", "customer_decision_scores")
    action_summary = registry.load_dataframe("decision_engine", "action_summary")
    model_card = registry.load_model_card("decision_engine")

    if decision is None:
        st.warning("No decision-engine output found. Run decision_engine.py first.")
        st.stop()

    # Find key columns
    action_col = None
    for c in ["final_action", "recommended_action", "action"]:
        if c in decision.columns:
            action_col = c
            break

    priority_col = "priority_score" if "priority_score" in decision.columns else None
    decision_clv = None
    for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
        if c in decision.columns:
            decision_clv = c
            break

    d_churn = None
    for c in ["churn_probability", "churn_prob", "prob_churn"]:
        if c in decision.columns:
            d_churn = c
            break

    d_next = None
    for c in ["next_purchase_probability_30d", "next_purchase_probability", "purchase_probability_30d"]:
        if c in decision.columns:
            d_next = c
            break

    d_id = "Customer ID" if "Customer ID" in decision.columns else "customer_id"

    if not action_col:
        st.error("Decision file found but no action field was detected.")
        st.stop()

    # Page hero
    render_page_hero(
        title="Commercial Decision Engine",
        description="Combine value, risk, propensity, behavioral context, and capacity into a transparent next-best-action policy.",
        kicker="DECISION INTELLIGENCE",
    )

    # ---- TOP KPIs ----
    n_customers = len(decision)
    active_actions = ~decision[action_col].astype(str).str.startswith("monitor", na=False)
    n_active = int(active_actions.sum())

    median_priority = None
    if priority_col:
        median_priority = pd.to_numeric(decision[priority_col], errors="coerce").median()

    clv_represented = None
    if decision_clv:
        clv_represented = pd.to_numeric(decision[decision_clv], errors="coerce").sum()

    kpis = [
        {"label": "Customers Scored", "value": format_count(n_customers)},
        {"label": "Eligible for Active Action", "value": format_count(n_active),
         "help": "Customers assigned a non-monitor action"},
        {"label": "Median Priority Score", "value": format_score(median_priority) if median_priority is not None else "—"},
        {"label": "Total Predicted Future Net Revenue Represented", "value": format_clv(clv_represented) if clv_represented is not None else "—"},
    ]
    render_kpi_row(kpis, columns=4)

    # ---- CAPACITY VISUALIZATION ----
    if model_card and "capacity_constraints" in model_card:
        st.markdown("#### Capacity Allocation")
        capacity = model_card["capacity_constraints"]

        cap_data = []
        for action, cap in capacity.items():
            if action == "total":
                continue
            allocated = int((decision[action_col] == action).sum()) if action_col in decision.columns else 0
            cap_data.append({
                "Action": action.replace("_", " ").title(),
                "Capacity": cap,
                "Allocated": allocated,
                "Utilization": f"{allocated/cap*100:.0f}%" if cap > 0 else "N/A",
            })

        if cap_data:
            cap_df = pd.DataFrame(cap_data)
            st.dataframe(cap_df, use_container_width=True, hide_index=True)

    # ---- ACTION ALLOCATION ----
    st.markdown("#### Action Allocation")
    action_counts = (
        decision.groupby(action_col, dropna=False)
        .size()
        .reset_index(name="customers")
        .sort_values("customers", ascending=True)
    )

    left, right = st.columns([0.95, 1.05])

    with left:
        fig = plot_action_allocation(
            action_counts,
            action_col=action_col,
            count_col="customers",
            title="Customer Allocation by Recommended Action",
            height=470,
        )
        render_chart_responsive(fig, "dec_allocation")

    with right:
        if decision_clv and d_churn and priority_col:
            plot_df = decision.copy()
            plot_df[decision_clv] = pd.to_numeric(plot_df[decision_clv], errors="coerce")
            plot_df[d_churn] = pd.to_numeric(plot_df[d_churn], errors="coerce")
            plot_df[priority_col] = pd.to_numeric(plot_df[priority_col], errors="coerce")
            plot_df = plot_df.dropna(subset=[decision_clv, d_churn, priority_col])

            if len(plot_df) > 12000:
                plot_df = plot_df.sample(12000, random_state=42)

            fig = plot_decision_scatter(
                plot_df,
                x_col=decision_clv,
                y_col=d_churn,
                color_col=action_col,
                size_col=priority_col,
                title="Predicted Future Net Revenue, Risk, and Policy Assignment",
                height=470,
                x_log=True,
                y_pct=True,
            )
            render_chart_responsive(fig, "dec_scatter")
        else:
            render_missing_data("CLV, churn, and priority fields needed for decision scatter.")

    # ---- DECISION TARGET TABLE ----
    st.markdown("#### Decision Targets")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        action_filter = st.multiselect(
            "Action",
            options=decision[action_col].dropna().unique().tolist(),
            default=decision[action_col].dropna().unique().tolist(),
        )
    with col2:
        priority_range = st.slider(
            "Priority Score Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05,
        ) if priority_col else (0.0, 1.0)
    with col3:
        top_n = st.slider("Top N Customers", 10, 200, 50, 10)

    # Apply filters
    filtered = decision.copy()
    if action_filter:
        filtered = filtered[filtered[action_col].isin(action_filter)]
    if priority_col:
        filtered[priority_col] = pd.to_numeric(filtered[priority_col], errors="coerce")
        filtered = filtered[
            (filtered[priority_col] >= priority_range[0]) &
            (filtered[priority_col] <= priority_range[1])
        ]

    # Sort by priority
    if priority_col:
        targets = filtered.sort_values(priority_col, ascending=False).head(top_n)
    else:
        targets = filtered.head(top_n)

    # Preferred columns
    preferred_columns = [
        c for c in [
            d_id,
            action_col,
            "priority_tier",
            priority_col,
            decision_clv,
            d_churn,
            d_next,
            "reactivation_probability",
            "decision_confidence",
            "action_reason",
        ]
        if c and c in targets.columns
    ]

    display = targets[preferred_columns].copy()

    # Format for display
    for column in display.columns:
        if column in {d_churn, d_next, "reactivation_probability", "decision_confidence"}:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda x: format_churn_risk(x) if pd.notna(x) else "—"
            )
        elif column == decision_clv:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda x: format_clv(x) if pd.notna(x) else "—"
            )
        elif column == priority_col:
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda x: format_score(x, 1) if pd.notna(x) else "—"
            )

    st.dataframe(display, use_container_width=True, hide_index=True)

    render_download_button(
        display,
        "decision_targets.csv",
        "Download Filtered Targets (CSV)",
    )

    # ---- ACTION SUMMARY ----
    if action_summary is not None:
        with st.expander("View Exported Action Summary"):
            st.dataframe(action_summary, use_container_width=True, hide_index=True)

    # ---- DECISION SCIENCE ----
    st.markdown("---")
    render_science_card(
        "What This Engine Does — And Does Not Do",
        "It estimates **expected commercial opportunity** from the analytical signals already produced in the project. "
        "It then applies a transparent policy with capacity constraints. "
        "It does not infer an incremental treatment effect from observational purchase history. "
        "The next portfolio upgrade is an experiment or quasi-experiment that measures actual incremental response to each intervention.",
    )

    render_science_card(
        "Policy Transparency",
        f"**Actions**: {', '.join(decision[action_col].dropna().unique())}\n\n"
        f"**Capacity Constraints**: Total={model_card.get('capacity_constraints', {}).get('total', 'N/A') if model_card else 'N/A'}\n\n"
        f"**Confidence Threshold**: {model_card.get('confidence_threshold', 'N/A') if model_card else 'N/A'}\n\n"
        "Customers are scored on a priority function combining predicted future net revenue, "
        "inactivity risk, purchase propensity, segment context, and model confidence. "
        "Actions are assigned by priority tier subject to per-action capacity caps.",
    )


if __name__ == "__main__":
    render_decision_page()