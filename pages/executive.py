"""
Executive page — Portfolio-level summary of customer economics, retention, risk, and opportunity.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from app_components import (
    render_kpi_row,
    render_page_hero,
    render_science_card,
    render_section_label,
    render_missing_data,
)
from app_charts import (
    plot_concentration_curve,
    plot_action_allocation,
    render_chart_responsive,
)
from app_data import get_registry
from app_formatting import (
    format_count, format_clv, format_churn_risk, format_next_purchase,
    format_retention,
)


def render_executive_page() -> None:
    """Render the Executive dashboard page."""
    registry = get_registry()

    # Load data
    customer_profile = registry.load_dataframe("customer_360", "customer_360_current")
    if customer_profile is None or customer_profile.empty:
        customer_profile = registry.load_dataframe("customer_360", "customer_360")

    segment_profiles = registry.load_dataframe("segmentation", "segment_profiles")
    cohort_decay = registry.load_dataframe("cohorts", "retention_decay_curve")
    churn_data = registry.load_dataframe("churn", "customer_churn_next_purchase")
    next_purchase_data = registry.load_dataframe("churn", "customer_churn_next_purchase")  # same file
    action_summary = registry.load_dataframe("decision_engine", "action_summary")
    clv_data = registry.load_dataframe("clv", "clv_customer_predictions")

    # Page hero
    render_page_hero(
        title="Executive View",
        description="A decision-oriented summary of customer economics, retention, risk, and commercial opportunity.",
        kicker="CUSTOMER INTELLIGENCE",
    )

    # ---- KPI ROW ----
    n_customers = len(customer_profile) if customer_profile is not None else None

    # Portfolio CLV
    clv_total = None
    if clv_data is not None:
        clv_col = "clv_mean" if "clv_mean" in clv_data.columns else "clv"
        if clv_col in clv_data.columns:
            clv_total = pd.to_numeric(clv_data[clv_col], errors="coerce").sum()
    elif customer_profile is not None:
        # Fallback to customer_360 CLV
        clv_col = None
        for c in ["clv", "predicted_clv", "customer_clv", "clv_mean"]:
            if c in customer_profile.columns:
                clv_col = c
                break
        if clv_col:
            clv_total = pd.to_numeric(customer_profile[clv_col], errors="coerce").sum()

    # Mean churn risk
    churn_mean = None
    if churn_data is not None:
        for c in ["churn_probability", "churn_prob", "prob_churn"]:
            if c in churn_data.columns:
                churn_mean = pd.to_numeric(churn_data[c], errors="coerce").mean()
                break

    # Mean next-purchase probability
    np_mean = None
    if next_purchase_data is not None:
        for c in ["next_purchase_probability_30d", "next_purchase_probability", "purchase_probability_30d"]:
            if c in next_purchase_data.columns:
                np_mean = pd.to_numeric(next_purchase_data[c], errors="coerce").mean()
                break

    # Month-3 retention
    retention_3 = None
    if cohort_decay is not None:
        age_col = "age_month" if "age_month" in cohort_decay.columns else None
        ret_col = None
        for c in ["weighted_logo_retention", "logo_retention", "logo_retention_pct"]:
            if c in cohort_decay.columns:
                ret_col = c
                break
        if age_col and ret_col:
            match = cohort_decay.loc[
                pd.to_numeric(cohort_decay[age_col], errors="coerce") == 3,
                ret_col
            ]
            if not match.empty:
                retention_3 = float(pd.to_numeric(match, errors="coerce").iloc[0])

    kpis = [
        {"label": "Customers", "value": format_count(n_customers)},
        {"label": "Portfolio Predicted Future Net Revenue", "value": format_clv(clv_total),
         "help": "Sum of predicted future net revenue (CLV proxy) across all customers"},
        {"label": "Mean Next-Month Inactivity Risk", "value": format_churn_risk(churn_mean),
         "help": "Average predicted probability of no purchase in the next month"},
        {"label": "Mean 30-Day Purchase Propensity", "value": format_next_purchase(np_mean),
         "help": "Average predicted probability of a purchase within 30 days"},
        {"label": "Month-3 Logo Retention", "value": format_retention(retention_3) if retention_3 is not None else "—",
         "help": "Weighted average logo retention at 3 months since acquisition"},
    ]
    render_kpi_row(kpis, columns=5)

    # ---- PORTFOLIO SIGNALS ----
    render_section_label("Portfolio Signals")

    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown("#### Customer Value Concentration")
        if clv_data is not None:
            clv_col = "clv_mean" if "clv_mean" in clv_data.columns else "clv"
            if clv_col in clv_data.columns:
                clv_values = pd.to_numeric(clv_data[clv_col], errors="coerce").dropna()
                clv_values = clv_values[clv_values > 0]
                if len(clv_values) > 0:
                    fig = plot_concentration_curve(
                        clv_values.tolist(),
                        title="How concentrated is predicted future value?",
                        value_label="Predicted Future Net Revenue",
                        reference_lines=[0.5, 0.8],
                    )
                    render_chart_responsive(fig, "exec_concentration")
                else:
                    render_missing_data("CLV values are empty or non-positive.")
            else:
                render_missing_data("CLV field not recognized in output.")
        elif customer_profile is not None:
            # Try customer_360 CLV
            clv_col = None
            for c in ["clv", "predicted_clv", "customer_clv", "clv_mean"]:
                if c in customer_profile.columns:
                    clv_col = c
                    break
            if clv_col:
                clv_values = pd.to_numeric(customer_profile[clv_col], errors="coerce").dropna()
                clv_values = clv_values[clv_values > 0]
                if len(clv_values) > 0:
                    fig = plot_concentration_curve(
                        clv_values.tolist(),
                        title="How concentrated is predicted future value?",
                        value_label="Predicted Future Net Revenue",
                        reference_lines=[0.5, 0.8],
                    )
                    render_chart_responsive(fig, "exec_concentration")
                else:
                    render_missing_data("CLV values are empty or non-positive.")
            else:
                render_missing_data("Run clv_analysis.py to populate the executive value view.")
        else:
            render_missing_data("Run customer_360.py or clv_analysis.py to populate the executive value view.")

    with right:
        st.markdown("#### Decision Allocation")
        if action_summary is not None:
            action_col = None
            for c in ["final_action", "recommended_action", "action"]:
                if c in action_summary.columns:
                    action_col = c
                    break
            customer_col = None
            for c in ["customers", "customer_count", "n_customers"]:
                if c in action_summary.columns:
                    customer_col = c
                    break

            if action_col and customer_col:
                plot_df = action_summary[[action_col, customer_col]].copy()
                plot_df[customer_col] = pd.to_numeric(plot_df[customer_col], errors="coerce")
                plot_df = plot_df.sort_values(customer_col, ascending=True)

                fig = plot_action_allocation(plot_df, action_col, customer_col,
                                            title="Customers by Recommended Action")
                render_chart_responsive(fig, "exec_allocation")
            else:
                render_missing_data("Action summary columns not recognized.")
        else:
            render_missing_data("Run decision_engine.py to populate the commercial action layer.")

    # ---- WHAT THE PORTFOLIO IS ANSWERING ----
    render_section_label("What the Portfolio Is Answering")

    cards = st.columns(4)
    card_data = [
        (
            "Who behaves differently?",
            "Behavioral segmentation compresses customer heterogeneity into interpretable behavioral states rather than relying only on recency, frequency, and monetary value.",
        ),
        (
            "Which cohorts persist?",
            "Cohort analysis separates acquisition quality from customer age and shows whether revenue and customer activity decay over time.",
        ),
        (
            "Who is worth retaining?",
            "Predictive future net revenue combines expected future economics with uncertainty, allowing value to be discussed as a distribution rather than a single deterministic number.",
        ),
        (
            "What should happen next?",
            "Retention risk, purchase propensity, behavioral state, and value are combined in the decision layer. The policy is observational until experimentally validated.",
        ),
    ]

    for col, (title, body) in zip(cards, card_data):
        with col:
            render_science_card(title, body)

    # ---- SCIENTIFIC BOUNDARY ----
    st.markdown("---")
    render_science_card(
        "Analytical Boundary",
        "The platform forecasts and prioritizes customer behavior from transaction history. "
        "Observational scores are not causal treatment effects. "
        "Predicted future net revenue is a discounted revenue proxy, not economic CLV (no margin data). "
        "Churn risk represents next-month inactivity probability, not a business-defined churn event.",
        icon="🔬"
    )


if __name__ == "__main__":
    render_executive_page()