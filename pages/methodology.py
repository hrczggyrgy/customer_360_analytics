"""
Methodology page — Dynamic documentation pulled from model cards, config, and implementation status.
"""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from app_components import (
    render_page_hero,
    render_section_label,
    render_science_card,
    render_method_step,
    render_missing_data,
)
from app_data import get_registry
from app_config import get_config


def render_methodology_page() -> None:
    """Render the dynamic Methodology page."""
    registry = get_registry()
    config = get_config()

    # Load all model cards
    model_cards = {}
    for module in ["segmentation", "cohorts", "clv", "churn", "reactivation", "recommendations", "decision_engine"]:
        card = registry.load_model_card(module)
        if card:
            model_cards[module] = card

    # Load implementation status
    impl_status = {}
    try:
        with open("IMPLEMENTATION_STATUS.json", "r") as f:
            impl_status = json.load(f)
    except Exception:
        pass

    # Page hero
    render_page_hero(
        title="The Science",
        description="Understand how the analytical layers connect, what each model is answering, and where causal claims stop.",
        kicker="METHODOLOGY",
    )

    # ---- ANALYTICAL WORKFLOW ----
    st.markdown("### Analytical Workflow")

    steps = [
        (
            "01",
            "Data Quality & Canonical Ingestion",
            "Standardize transactions, identify reversals/cancellations, validate identifiers, and preserve a clear observation window. "
            "Invoice preserved as string to retain cancellation codes (C...).",
        ),
        (
            "02",
            "Canonical Customer-Month Panel",
            "Dense calendar grid per customer (cohort month → last observed month) with zeros for inactive months. "
            "Foundation for all temporal modeling.",
        ),
        (
            "03",
            "Point-in-Time Customer 360",
            "Two outputs: (1) Descriptive current-state mart (lifetime totals, current recency, product affinity) — valid for profiling. "
            "(2) Point-in-time feature engine (`features_at_date()`) — leakage-safe features for historical ML.",
        ),
        (
            "04",
            "Behavioral Segmentation",
            "Robust feature engineering: winsorize → Yeo-Johnson → RobustScaler → block balancing → PCA → HDBSCAN. "
            "12 clusters + noise (16.6%). Bootstrap stability (10 repeats, ARI 0.97–0.99).",
        ),
        (
            "05",
            "Cohort Analysis",
            "25 acquisition cohorts (Dec 2009 – Dec 2011). Logo, gross/net revenue retention matrices. "
            "Maturity tracking with explicit right-censoring. NRR > 100% documented as retail revenue index.",
        ),
        (
            "06",
            "Predictive Future Net Revenue (CLV Proxy)",
            "Dynamic probabilistic discounted net-revenue forecast. Discrete-time hazard + conditional spend + Empirical Bayes + Monte Carlo. "
            "Margin scenarios (10%/20%/30%/40%). Temporal backtest framework. NOT economic CLV (no margin data).",
        ),
        (
            "07",
            "Retention & Next Purchase",
            "Discrete-time survival (next-month inactivity risk) + next-purchase (7/30/60 days). "
            "Temporal train/val/test splits. Platt calibration. Recursive survival forecasting.",
        ),
        (
            "08",
            "Reactivation Model",
            "Inactive customers (threshold: 3 months) → purchase within 3 months. "
            "Features: recency, historical cadence, volatility, prior reactivations, revenue, CLV, segment, returns.",
        ),
        (
            "09",
            "Product Intelligence",
            "Product-level metrics: revenue, units, penetration, repeat rate, return rate, velocity. "
            "Behavioral roles: acquisition, repeat, basket builder, retention, niche high-value, volatile. "
            "Co-purchase matrix (629k pairs) for affinity.",
        ),
        (
            "10",
            "Recommendation Engine",
            "Co-purchase affinity + customer history + segment affinity + popularity fallback. "
            "Vectorized implementation for 5,878 customers. Output: Customer ID, product, score, reason, support, lift.",
        ),
        (
            "11",
            "Decision Engine",
            "Multi-signal scoring: predicted future net revenue + inactivity risk + purchase propensity + segment + confidence. "
            "Capacity-aware allocation (total 1,000, per-action caps). Observational policy — not causal.",
        ),
        (
            "12",
            "Experimentation (Future)",
            "The causal layer comes last: actual interventions need randomized or otherwise defensible treatment/control data "
            "before incremental impact is claimed.",
        ),
    ]

    for number, title, body in steps:
        render_method_step(number, title, body)

    # ---- MODEL RESPONSIBILITIES ----
    render_section_label("Model Responsibilities")

    cols = st.columns(3)

    with cols[0]:
        render_science_card(
            "Segmentation answers: Who is different?",
            "It discovers stable behavioral structure across customer economics, "
            "cadence, assortment, pricing, temporal preferences, lifecycle, and return behavior. "
            "12 clusters + noise (16.6%). Stability validated via bootstrap ARI (0.97–0.99).",
        )

    with cols[1]:
        render_science_card(
            "Predictive Value answers: Who is economically valuable?",
            "It forecasts future commercial value as discounted net revenue. "
            "Judged using temporal holdouts and uncertainty, not just in-sample fit. "
            "Margin scenarios (10–40%) allow conversion to economic CLV when margin data available.",
        )

    with cols[2]:
        render_science_card(
            "Decisioning answers: Who should receive attention?",
            "It prioritizes customers for actions under commercial constraints. "
            "The policy becomes causal only after intervention effects are measured. "
            "Current policy: observational next-best-action with capacity constraints.",
        )

    # ---- DYNAMIC MODULE DETAILS ----
    render_section_label("Module Details (from Model Cards)")

    module_tabs = st.tabs([
        "Data Quality",
        "Segmentation",
        "Cohorts",
        "Predictive Value",
        "Retention",
        "Reactivation",
        "Recommendations",
        "Decision Engine",
    ])

    with module_tabs[0]:
        render_module_detail("data_quality", registry, model_cards)

    with module_tabs[1]:
        render_module_detail("segmentation", registry, model_cards)

    with module_tabs[2]:
        render_module_detail("cohorts", registry, model_cards)

    with module_tabs[3]:
        render_module_detail("clv", registry, model_cards)

    with module_tabs[4]:
        render_module_detail("churn", registry, model_cards)

    with module_tabs[5]:
        render_module_detail("reactivation", registry, model_cards)

    with module_tabs[6]:
        render_module_detail("recommendations", registry, model_cards)

    with module_tabs[7]:
        render_module_detail("decision_engine", registry, model_cards)

    # ---- GLOSSARY ----
    render_section_label("Glossary")

    glossary = [
        ("Cohort", "Group of customers acquired in the same calendar month (first clean sale)."),
        ("Logo Retention", "Fraction of cohort customers who make ≥1 purchase in a given age month."),
        ("Revenue Retention Index", "Cohort revenue at age / cohort revenue at age 0. Can exceed 100% (retail revenue index)."),
        ("Point-in-Time Feature", "Feature computed using only information available at a specific prediction date — leakage-safe for ML."),
        ("Hazard", "Discrete-time probability of an event (purchase) in the next period, conditional on survival to current period."),
        ("Survival Probability", "Probability of remaining active through period t: S(t) = ∏(1 - h_k)."),
        ("CLV Proxy", "Discounted expected future net revenue. Not economic CLV (no margin, no probabilistic lifetime model)."),
        ("Predictive Interval", "Model uncertainty range from Monte Carlo simulation (p10–p90). Not a calibrated confidence interval."),
        ("Recommendation Score", "Composite of co-purchase affinity and product popularity. Observational association, not causal."),
        ("Observational Decision Score", "Priority score combining value, risk, propensity, and confidence. Not an incremental treatment effect."),
    ]

    for term, definition in glossary:
        st.markdown(f"**{term}**: {definition}")

    # ---- PROJECT MATURITY ----
    render_section_label("Project Maturity")

    if impl_status:
        phases = impl_status.get("phases", {})
        maturity_data = []
        for phase_name, phase_info in phases.items():
            if isinstance(phase_info, dict):
                status = phase_info.get("status", "UNKNOWN")
                evidence = phase_info.get("evidence", "")
                maturity_data.append({
                    "Layer": phase_name.replace("_", " ").title(),
                    "Status": status,
                    "Evidence": evidence[:100] + "..." if len(evidence) > 100 else evidence,
                })

        if maturity_data:
            st.dataframe(pd.DataFrame(maturity_data), use_container_width=True, hide_index=True)

    # ---- SCIENTIFIC BOUNDARY ----
    render_section_label("Important Scientific Boundary")

    render_science_card(
        "Observational ≠ Causal",
        "Transaction data is excellent for understanding observed customer behavior and forecasting future behavior. "
        "It is not, by itself, enough to establish that a campaign, discount, reminder, or recommendation "
        "caused an incremental outcome. The portfolio explicitly preserves this distinction and introduces "
        "experimentation for the final causal layer.",
    )


def render_module_detail(module: str, registry, model_cards: dict) -> None:
    """Render dynamic details for a module."""
    card = model_cards.get(module)

    if not card:
        render_missing_data(f"No model card available for {module}.")
        return

    # Basic info
    st.markdown(f"**Model**: {card.get('model_name', module)}")
    st.markdown(f"**Methodology**: {card.get('methodology', 'Not specified')}")

    # Metrics
    if "metrics" in card:
        st.markdown("**Validation Metrics**")
        metrics = card["metrics"]
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, dict):
                    st.markdown(f"*{k}*")
                    cols = st.columns(min(4, len(v)))
                    for i, (mk, mv) in enumerate(v.items()):
                        with cols[i % len(cols)]:
                            if isinstance(mv, float):
                                st.metric(mk.replace("_", " ").title(), f"{mv:.3f}")
                            else:
                                st.metric(mk.replace("_", " ").title(), str(mv))
                else:
                    st.metric(k.replace("_", " ").title(), str(v))

    # Features
    if "features" in card:
        with st.expander("Features Used"):
            features = card["features"]
            if isinstance(features, list):
                st.write(", ".join(str(f) for f in features))
            else:
                st.json(features)

    # Limitations
    if "known_limitations" in card:
        with st.expander("Known Limitations"):
            for lim in card["known_limitations"]:
                st.markdown(f"• {lim}")
    elif "limitations" in card:
        with st.expander("Limitations"):
            for lim in card["limitations"]:
                st.markdown(f"• {lim}")

    # Config parameters
    config = get_config()
    if module == "segmentation":
        with st.expander("Segmentation Config"):
            st.json(config.model_segmentation)
    elif module == "clv":
        with st.expander("CLV Config"):
            st.json(config.model_clv)
    elif module == "churn":
        with st.expander("Churn/Next-Purchase Config"):
            st.json(config.model_churn)
    elif module == "decision_engine":
        with st.expander("Decision Engine Config"):
            st.json(config.model_decision_engine)


if __name__ == "__main__":
    render_methodology_page()