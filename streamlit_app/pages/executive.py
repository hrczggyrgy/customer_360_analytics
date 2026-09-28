"""
Executive page for Retail Customer Intelligence.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..app_components import (
    render_section_label,
    render_science_card,
    render_kpi_card,
    render_action_summary_table,
)
from ..app_data import get_registry
from ..app_formatting import (
    format_currency,
    format_probability,
    format_percent,
    auto_format,
)
from ..app_charts import (
    base_layout,
    plot_missing,
    plot_concentration_curve,
    plot_action_allocation,
    PLOTLY_CONFIG,
)


def render():
    """Render the Executive page."""
    registry = get_registry()
    
    # Load data
    combined = registry.load_dataframe("customer_360")
    segments = registry.load_dataframe("segmentation")
    segment_profiles = registry.load_dataframe("segmentation", "segment_profiles.csv")
    cohort_decay = registry.load_dataframe("cohorts", "retention_decay_curve.csv")
    churn = registry.load_dataframe("churn")
    next_purchase = registry.load_dataframe("churn")  # Same file
    action_summary = registry.load_dataframe("decision_engine", "action_summary.csv")
    
    # KPI extraction
    n_customers = len(combined) if combined is not None else None
    
    clv_total = None
    if combined is not None:
        clv_col = None
        for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
            if c in combined.columns:
                clv_col = c
                break
        if clv_col:
            clv_total = pd.to_numeric(combined[clv_col], errors="coerce").sum()
    
    if clv_total is None and segments is not None:
        clv_col = None
        for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
            if c in segments.columns:
                clv_col = c
                break
        if clv_col:
            clv_total = pd.to_numeric(segments[clv_col], errors="coerce").sum()
    
    churn_mean = None
    if combined is not None:
        churn_col = None
        for c in ["churn_probability", "churn_prob", "prob_churn"]:
            if c in combined.columns:
                churn_col = c
                break
        if churn_col:
            churn_mean = pd.to_numeric(combined[churn_col], errors="coerce").mean()
    
    if churn_mean is None and churn is not None:
        churn_col = None
        for c in ["churn_probability", "churn_prob", "prob_churn"]:
            if c in churn.columns:
                churn_col = c
                break
        if churn_col:
            churn_mean = pd.to_numeric(churn[churn_col], errors="coerce").mean()
    
    np_mean = None
    if combined is not None:
        np_col = None
        for c in ["next_purchase_probability", "next_purchase_probability_30d", "purchase_probability_30d"]:
            if c in combined.columns:
                np_col = c
                break
        if np_col:
            np_mean = pd.to_numeric(combined[np_col], errors="coerce").mean()
    
    if np_mean is None and next_purchase is not None:
        np_col = None
        for c in ["next_purchase_probability", "next_purchase_probability_30d", "purchase_probability_30d"]:
            if c in next_purchase.columns:
                np_col = c
                break
        if np_col:
            np_mean = pd.to_numeric(next_purchase[np_col], errors="coerce").mean()
    
    retention_3 = None
    if cohort_decay is not None:
        age_col = None
        for c in ["age_month", "age"]:
            if c in cohort_decay.columns:
                age_col = c
                break
        ret_col = None
        for c in ["weighted_logo_retention", "logo_retention"]:
            if c in cohort_decay.columns:
                ret_col = c
                break
        if age_col and ret_col:
            tmp = cohort_decay.copy()
            ages = pd.to_numeric(tmp[age_col], errors="coerce")
            vals = pd.to_numeric(tmp[ret_col], errors="coerce")
            match = tmp.loc[ages.eq(3), ret_col]
            if not match.empty:
                retention_3 = float(pd.to_numeric(match, errors="coerce").iloc[0])
    
    # KPIs
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        render_kpi_card("Customers", n_customers, formatter="count")
    
    with kpi_cols[1]:
        render_kpi_card("Portfolio CLV", clv_total, formatter="currency")
    
    with kpi_cols[2]:
        render_kpi_card("Mean churn risk", churn_mean, formatter="probability")
    
    with kpi_cols[3]:
        render_kpi_card("Mean next-purchase probability", np_mean, formatter="probability")
    
    with kpi_cols[4]:
        render_kpi_card("Month-3 retention", retention_3, formatter="percent")
    
    st.markdown("")
    render_section_label("Portfolio signals")
    
    left, right = st.columns([1.05, 0.95])
    
    with left:
        st.markdown("#### Customer value concentration")
        
        if combined is not None:
            clv_col = None
            for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
                if c in combined.columns:
                    clv_col = c
                    break
            if clv_col:
                plot_df = combined[["Customer ID", clv_col]].copy()
                plot_df[clv_col] = pd.to_numeric(plot_df[clv_col], errors="coerce")
                plot_df = plot_df.dropna(subset=[clv_col])
                
                if not plot_df.empty:
                    plot_df["rank"] = plot_df[clv_col].rank(pct=True)
                    plot_df = plot_df.sort_values(clv_col, ascending=False)
                    total_positive_clv = max(float(plot_df[clv_col].clip(lower=0).sum()), 1e-9)
                    plot_df["cumulative_clv_share"] = (
                        plot_df[clv_col].clip(lower=0).cumsum() / total_positive_clv
                    )
                    
                    sample = plot_df.iloc[::max(1, len(plot_df) // 1000)].copy()
                    
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=sample["rank"],
                        y=sample["cumulative_clv_share"],
                        mode="lines",
                        line=dict(width=2),
                        name="Cumulative CLV share",
                        hovertemplate=(
                            "Customer percentile: %{x:.0%}<br>"
                            "Cumulative CLV share: %{y:.0%}<extra></extra>"
                        ),
                    ))
                    fig.add_hline(y=0.50, line_dash="dot", annotation_text="50% of value")
                    fig.update_xaxes(tickformat=".0%", title="Customer percentile by CLV")
                    fig.update_yaxes(tickformat=".0%", title="Cumulative CLV share")
                    st.plotly_chart(
                        base_layout(fig, title="How concentrated is customer value?"),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )
                else:
                    plot_missing("CLV values are empty.")
            else:
                plot_missing("A CLV field was not detected.")
        else:
            plot_missing("Run customer_360.py or clv_analysis.py to populate the executive value view.")
    
    with right:
        st.markdown("#### Decision allocation")
        
        if action_summary is not None:
            action_col = None
            for c in ["final_action", "recommended_action", "action"]:
                if c in action_summary.columns:
                    action_col = c
                    break
            customer_col = None
            for c in ["customers", "customer_count"]:
                if c in action_summary.columns:
                    customer_col = c
                    break
            
            if action_col and customer_col:
                from ..app_charts import plot_action_allocation
                fig = plot_action_allocation(
                    action_summary[[action_col, customer_col]].rename(
                        columns={action_col: "action", customer_col: "count"}
                    ),
                    action_col="action",
                    count_col="count",
                    title="Customers by recommended action",
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                plot_missing("Action summary columns were not recognized.")
        else:
            plot_missing("Run decision_engine.py to populate the commercial action layer.")
    
    # Science cards
    st.markdown("")
    render_section_label("What the portfolio is answering")
    
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
            "Predicted Future Net Revenue (CLV Proxy) combines expected future economics with uncertainty, allowing value to be discussed as a distribution rather than a single deterministic number.",
        ),
        (
            "What should happen next?",
            "Retention, purchase propensity, behavioral state, and value are combined in the decision layer. The policy is observational until experimentally validated.",
        ),
    ]
    
    for col, (title, body) in zip(cards, card_data):
        with col:
            render_science_card(title, body)