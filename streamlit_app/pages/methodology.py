"""
Methodology page for Retail Customer Intelligence.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..ui import (
    render_section_label,
    render_science_card,
    HERO_COPY,
)
from ..app_data import get_registry


def render() -> None:
    """Render the Methodology page."""
    registry = get_registry()
    
    st.markdown("### Analytical workflow")
    
    steps = [
        (
            "01",
            "Data quality",
            "Standardize transactions, identify reversals/cancellations, validate identifiers, and preserve a clear observation window.",
        ),
        (
            "02",
            "Customer 360",
            "Create one reusable analytical customer mart containing economics, cadence, assortment, price behavior, temporal behavior, lifecycle, and returns.",
        ),
        (
            "03",
            "Behavioral segmentation",
            "Robustly transform heterogeneous customer features, balance behavioral blocks, denoise with PCA, and cluster in latent behavioral space.",
        ),
        (
            "04",
            "Cohort analysis",
            "Measure customer retention and economic decay by acquisition cohort age, keeping return-only months visible to avoid understating commercial friction.",
        ),
        (
            "05",
            "Predicted Future Net Revenue (CLV Proxy)",
            "Project customer economics into the future and retain an uncertainty representation rather than reducing value to one deterministic number.",
        ),
        (
            "06",
            "Retention and purchase propensity",
            "Model forward-looking customer state using temporal splits so future purchase behavior does not leak into model development.",
        ),
        (
            "07",
            "Product recommendations",
            "Generate co-purchase based product recommendations with transparent scoring and popularity fallback.",
        ),
        (
            "08",
            "Commercial decision engine",
            "Combine value, risk, propensity, behavioral context, confidence, and capacity into a transparent next-best-action policy.",
        ),
        (
            "09",
            "Experimentation",
            "The causal layer comes last: actual interventions need randomized or otherwise defensible treatment/control data before incremental impact is claimed.",
        ),
    ]
    
    for number_, title, body in steps:
        st.markdown(
            f"""
            <div style='border-left: 3px solid var(--primary); padding: 9px 0 9px 14px; margin: 8px 0;'>
                <strong style='color: var(--text-primary);'>{number_} · {title}</strong>
                <div style='color: var(--text-secondary); font-size: 0.85rem; line-height: 1.45; margin-top: 4px;'>{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.markdown("")
    render_section_label("Model responsibilities")
    
    cols = st.columns(3)
    
    with cols[0]:
        render_science_card(
            "Segmentation answers: who is different?",
            "It discovers stable behavioral structure across customer economics, "
            "cadence, assortment, price, temporal preferences, lifecycle, and "
            "return behavior."
        )
    
    with cols[1]:
        render_science_card(
            "CLV answers: who is economically valuable?",
            "It forecasts future commercial value. It should be judged using "
            "temporal holdouts and uncertainty, not just in-sample fit."
        )
    
    with cols[2]:
        render_science_card(
            "Decisioning answers: who should receive attention?",
            "It prioritizes customers for actions under commercial constraints. "
            "The policy becomes causal only after intervention effects are measured."
        )
    
    render_section_label("Project maturity checklist")
    
    statuses = registry.get_all_module_statuses()
    
    readiness = []
    for label, key in [
        ("Data quality", "data_quality"),
        ("Customer 360", "customer_360"),
        ("Behavioral segmentation", "segmentation"),
        ("Cohort retention", "cohorts"),
        ("CLV Proxy", "clv"),
        ("Churn / survival", "churn"),
        ("Next purchase", "churn"),
        ("Reactivation", "reactivation"),
        ("Product analytics", "product_analytics"),
        ("Recommendations", "recommendations"),
        ("Decision engine", "decision_engine"),
    ]:
        status = statuses.get(key)
        if status:
            readiness.append({
                "Layer": label,
                "Status": status.overall_status.title(),
                "Freshness": status.freshness_note,
            })
        else:
            readiness.append({
                "Layer": label,
                "Status": "Unknown",
                "Freshness": "N/A",
            })
    
    st.dataframe(pd.DataFrame(readiness), use_container_width=True, hide_index=True)
    
    render_science_card(
        "Important scientific boundary",
        "Transaction data is excellent for understanding observed customer "
        "behavior and forecasting future behavior. It is not, by itself, enough "
        "to establish that a campaign, discount, reminder, or recommendation "
        "caused an incremental outcome. The portfolio should explicitly preserve "
        "this distinction and introduce experimentation for the final causal layer."
    )