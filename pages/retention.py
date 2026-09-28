"""
Retention & Next Purchase page — Forward-looking probability signals with model diagnostics.
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
)
from app_charts import (
    plot_distribution_histogram,
    plot_scatter_risk_propensity,
    plot_retention_decay,
    render_chart_responsive,
)
from app_data import get_registry
from app_formatting import (
    format_count, format_churn_risk, format_next_purchase, format_probability,
    format_retention,
)


def render_retention_page() -> None:
    """Render the Retention & Next Purchase page."""
    registry = get_registry()

    churn = registry.load_dataframe("churn", "customer_churn_next_purchase")
    model_card = registry.load_model_card("churn")
    survival_importance = registry.load_dataframe("churn", "survival_feature_importance")

    if churn is None:
        st.warning("No churn/next-purchase outputs found. Run churn_next_purchase.py first.")
        st.stop()

    # Page hero
    render_page_hero(
        title="Retention & Next Purchase",
        description="Translate customer behavior into forward-looking probability signals with clear model diagnostics.",
        kicker="PREDICTIVE SIGNALS",
    )

    # ---- TABS ----
    tabs = st.tabs([
        "Risk Profile",
        "Purchase Propensity",
        "Risk vs Propensity",
        "Survival Curves",
        "Model Diagnostics",
    ])

    with tabs[0]:
        render_risk_profile_tab(churn)

    with tabs[1]:
        render_purchase_propensity_tab(churn)

    with tabs[2]:
        render_risk_vs_propensity_tab(churn)

    with tabs[3]:
        render_survival_tab(churn)

    with tabs[4]:
        render_diagnostics_tab(model_card, survival_importance)


def render_risk_profile_tab(churn: pd.DataFrame) -> None:
    """Render the risk profile tab."""
    st.markdown("#### Next-Month Inactivity Risk Distribution")

    ccol = None
    for c in ["churn_probability", "churn_prob", "prob_churn"]:
        if c in churn.columns:
            ccol = c
            break

    if ccol:
        values = pd.to_numeric(churn[ccol], errors="coerce").dropna()

        k = st.columns(4)
        with k[0]:
            st.metric("Customers Scored", format_count(len(values)))
        with k[1]:
            st.metric("Median Risk", format_churn_risk(values.median()))
        with k[2]:
            st.metric("High Risk (≥70%)", format_churn_risk((values >= 0.70).mean()))
        with k[3]:
            st.metric("Very High Risk (≥85%)", format_churn_risk((values >= 0.85).mean()))

        fig = plot_distribution_histogram(
            values.tolist(),
            title="Predicted Probability of Next-Month Inactivity",
            height=480,
            x_title="Next-Month Inactivity Risk",
            nbins=40,
        )
        fig.update_xaxes(tickformat=".0%")
        render_chart_responsive(fig, "ret_risk_dist")

        render_science_card(
            "Why This Is Not 'Churn'",
            "This model predicts the **probability of no purchase in the next calendar month**. "
            "It is a discrete-time hazard model, not a business-defined churn event. "
            "A survival formulation models the timing of continued activity and accounts for the fact that "
            "newer customers have had less time to experience an observed lapse. "
            "This is preferable to defining churn with one arbitrary recency threshold and calling that label ground truth.",
        )
    else:
        render_missing_data("Churn output found but no probability field was recognized.")


def render_purchase_propensity_tab(churn: pd.DataFrame) -> None:
    """Render the purchase propensity tab."""
    st.markdown("#### Next-Purchase Probability")

    # Support multiple horizons
    horizons = {
        "7-day": ["next_purchase_probability_7d", "purchase_probability_7d"],
        "30-day": ["next_purchase_probability_30d", "next_purchase_probability", "purchase_probability_30d"],
        "60-day": ["next_purchase_probability_60d", "purchase_probability_60d"],
    }

    horizon = st.selectbox("Prediction Horizon", options=list(horizons.keys()), index=1)

    pcol = None
    for c in horizons[horizon]:
        if c in churn.columns:
            pcol = c
            break

    if pcol:
        values = pd.to_numeric(churn[pcol], errors="coerce").dropna()

        k = st.columns(4)
        with k[0]:
            st.metric("Customers Scored", format_count(len(values)))
        with k[1]:
            st.metric(f"Median {horizon} Propensity", format_next_purchase(values.median()))
        with k[2]:
            st.metric(f"High Propensity (≥70%)", format_next_purchase((values >= 0.70).mean()))
        with k[3]:
            st.metric(f"Low Propensity (<30%)", format_next_purchase((values < 0.30).mean()))

        fig = plot_distribution_histogram(
            values.tolist(),
            title=f"Predicted Probability of Purchase Within {horizon.title()}",
            height=480,
            x_title=f"{horizon.title()} Purchase Probability",
            nbins=40,
        )
        fig.update_xaxes(tickformat=".0%")
        render_chart_responsive(fig, f"ret_propensity_{horizon}")
    else:
        render_missing_data(f"No {horizon} purchase probability field recognized.")


def render_risk_vs_propensity_tab(churn: pd.DataFrame) -> None:
    """Render risk vs propensity scatter."""
    st.markdown("#### Risk vs Propensity")

    ccol = None
    for c in ["churn_probability", "churn_prob", "prob_churn"]:
        if c in churn.columns:
            ccol = c
            break

    pcol = None
    for c in ["next_purchase_probability_30d", "next_purchase_probability", "purchase_probability_30d"]:
        if c in churn.columns:
            pcol = c
            break

    if ccol and pcol:
        churn_vals = pd.to_numeric(churn[ccol], errors="coerce")
        purch_vals = pd.to_numeric(churn[pcol], errors="coerce")

        valid = pd.DataFrame({ccol: churn_vals, pcol: purch_vals}).dropna()

        if len(valid) > 12000:
            valid = valid.sample(12000, random_state=42)

        fig = plot_scatter_risk_propensity(
            churn_prob=valid[ccol].values,
            purchase_prob=valid[pcol].values,
            title="Customer Risk vs Near-Term Purchase Propensity",
            height=560,
        )
        render_chart_responsive(fig, "ret_risk_prop")

        st.caption(
            "The most commercially interesting customers are not necessarily the highest-risk or "
            "highest-propensity customers in isolation. The decision layer combines these signals "
            "with value and behavioral context."
        )
    else:
        render_missing_data("Both churn and next-purchase probabilities needed for this view.")


def render_survival_tab(churn: pd.DataFrame) -> None:
    """Render survival curves."""
    st.markdown("#### Survival Probabilities")

    # Check for survival columns
    surv_cols = {
        "3-Month": ["survival_3m", "survival_3_month", "S_3"],
        "6-Month": ["survival_6m", "survival_6_month", "S_6"],
        "12-Month": ["survival_12m", "survival_12_month", "S_12"],
    }

    available = {}
    for label, candidates in surv_cols.items():
        for c in candidates:
            if c in churn.columns:
                available[label] = c
                break

    if available:
        # Distribution of survival probabilities
        surv_data = {}
        for label, col in available.items():
            vals = pd.to_numeric(churn[col], errors="coerce").dropna()
            if len(vals) > 0:
                surv_data[label] = vals

        if surv_data:
            # Show median survival probabilities
            cols = st.columns(len(surv_data))
            for i, (label, col) in enumerate(available.items()):
                vals = pd.to_numeric(churn[col], errors="coerce").dropna()
                with cols[i]:
                    st.metric(f"{label} Survival (median)", format_retention(vals.median()))

            # Survival decay curve (aggregate)
            st.markdown("#### Aggregate Survival Decay")
            ages = [3, 6, 12]
            medians = [surv_data.get(f"{a}-Month", pd.Series()).median() for a in ages]
            medians = [m for m in medians if not np.isnan(m)]

            if len(medians) == len(ages):
                fig = plot_retention_decay(
                    ages=ages,
                    logo_retention=None,
                    nrr=medians,
                    grr=None,
                    title="Model-Derived Survival Decay",
                    height=500,
                )
                fig.update_yaxes(title="Survival Probability", tickformat=".0%")
                render_chart_responsive(fig, "ret_survival")

            render_science_card(
                "Survival Model",
                "Survival probabilities are derived from the discrete-time hazard model via recursive forecasting: "
                "S(t) = ∏(1 - h_k) for k=1 to t. "
                "This is a **model-derived future activity estimate**, not a Kaplan-Meier estimator. "
                "It accounts for the fact that customers who have already survived longer have different hazard profiles.",
            )
        else:
            render_missing_data("Survival columns found but contain no valid data.")
    else:
        render_missing_data("Survival outputs not available in churn predictions.")


def render_diagnostics_tab(model_card: dict, survival_importance: pd.DataFrame) -> None:
    """Render model diagnostics from model_card."""
    st.markdown("#### Model Diagnostics")

    if model_card:
        # Churn diagnostics
        if "churn" in model_card:
            churn_metrics = model_card["churn"]
            st.markdown("**Next-Month Inactivity Risk Model**")

            if "validation" in churn_metrics:
                v = churn_metrics["validation"]
                cols = st.columns(4)
                cols[0].metric("ROC-AUC", f"{v.get('roc_auc', 0):.3f}")
                cols[1].metric("PR-AUC", f"{v.get('pr_auc', 0):.3f}")
                cols[2].metric("Brier Score", f"{v.get('brier', 0):.3f}")
                cols[3].metric("Log Loss", f"{v.get('log_loss', 0):.3f}")

            if "test" in churn_metrics:
                t = churn_metrics["test"]
                st.markdown("**Test Set Performance**")
                cols = st.columns(4)
                cols[0].metric("ROC-AUC", f"{t.get('roc_auc', 0):.3f}")
                cols[1].metric("PR-AUC", f"{t.get('pr_auc', 0):.3f}")
                cols[2].metric("Brier Score", f"{t.get('brier', 0):.3f}")
                cols[3].metric("Log Loss", f"{t.get('log_loss', 0):.3f}")

        # Next-purchase diagnostics
        if "next_purchase" in model_card:
            np_metrics = model_card["next_purchase"]
            st.markdown("**Next-Purchase Models**")
            for horizon in ["7", "30", "60"]:
                if horizon in np_metrics:
                    m = np_metrics[horizon]
                    cols = st.columns(4)
                    cols[0].metric(f"{horizon}d ROC-AUC", f"{m.get('roc_auc', 0):.3f}")
                    cols[1].metric(f"{horizon}d PR-AUC", f"{m.get('pr_auc', 0):.3f}")
                    cols[2].metric(f"{horizon}d Brier", f"{m.get('brier', 0):.3f}")
                    cols[3].metric(f"{horizon}d Log Loss", f"{m.get('log_loss', 0):.3f}")

        # Features
        if "features" in model_card.get("churn", {}):
            with st.expander("Features Used"):
                st.write(model_card["churn"]["features"])

    if survival_importance is not None and not survival_importance.empty:
        with st.expander("Feature Importance (Survival)"):
            st.dataframe(survival_importance, use_container_width=True, hide_index=True)

    render_science_card(
        "Validation Principle",
        "Predictive retention and purchase models are evaluated with **time-based backtesting**. "
        "A random train/test split can leak future customer behavior into the training population "
        "and make model performance look artificially strong. "
        "The validation uses temporal splits: Train → Validation (2 months) → Test (3 months).",
    )


if __name__ == "__main__":
    render_retention_page()