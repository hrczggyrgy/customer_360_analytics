"""
Predictive Value (CLV) page — Customer lifetime value proxy distributions, uncertainty, and validation.
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
    plot_distribution_histogram,
    plot_uncertainty_band,
    plot_horizontal_bar,
    render_chart_responsive,
)
from app_data import get_registry
from app_formatting import format_clv, format_count, format_currency, format_score


def render_predictive_page() -> None:
    """Render the Predictive Value (CLV) page."""
    registry = get_registry()

    clv = registry.load_dataframe("clv", "clv_customer_predictions")
    if clv is None:
        clv = registry.load_dataframe("clv", "customer_clv")

    model_card = registry.load_model_card("clv")
    validation_preds = registry.load_dataframe("clv", "validation_predictions")
    feature_importance = registry.load_dataframe("clv", "feature_importance")
    monthly_summary = registry.load_dataframe("clv", "clv_monthly_summary")

    if clv is None or clv.empty:
        st.warning("No CLV output was detected. Run clv_analysis.py first.")
        st.stop()

    # Find CLV column
    clv_col = None
    for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
        if c in clv.columns:
            clv_col = c
            break

    if clv_col is None:
        st.error("CLV file found, but no CLV field was recognized.")
        st.stop()

    clv_values = pd.to_numeric(clv[clv_col], errors="coerce")
    clean = clv_values.dropna()
    clean_positive = clean[clean > 0]

    lower_col = None
    for c in ["clv_lower", "clv_p10", "clv_lower_bound"]:
        if c in clv.columns:
            lower_col = c
            break

    upper_col = None
    for c in ["clv_upper", "clv_p90", "clv_upper_bound"]:
        if c in clv.columns:
            upper_col = c
            break

    # Page hero
    render_page_hero(
        title="Predictive Customer Value",
        description="Inspect predicted future net revenue distributions, uncertainty, and model validation. "
                    "This is a discounted net-revenue CLV proxy — not economic CLV (no margin data).",
        kicker="PREDICTIVE VALUE",
    )

    # ---- TOP KPIs ----
    kpis = [
        {"label": "Customers Scored", "value": format_count(clean.size)},
        {"label": "Median Predicted Future Net Revenue", "value": format_clv(clean.median())},
        {"label": "Mean Predicted Future Net Revenue", "value": format_clv(clean.mean())},
        {"label": "Total Predicted Future Net Revenue", "value": format_clv(clean.sum())},
        {"label": "90th Percentile", "value": format_clv(clean.quantile(0.90))},
    ]
    render_kpi_row(kpis, columns=5)

    # ---- VISUALS ----
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("#### Predicted Future Net Revenue Distribution")

        # Log-scale option
        use_log = st.checkbox("Log scale (recommended for heavy-tailed distribution)", value=True)

        fig = plot_distribution_histogram(
            clean_positive.tolist(),
            title="Customer Predicted Future Net Revenue Distribution",
            height=500,
            x_title="Predicted Future Net Revenue",
            nbins=60,
            log_x=use_log,
            show_box=True,
        )
        render_chart_responsive(fig, "clv_dist")

    with right:
        st.markdown("#### Value Uncertainty (p10–p90)")

        if lower_col and upper_col:
            lower = pd.to_numeric(clv[lower_col], errors="coerce")
            upper = pd.to_numeric(clv[upper_col], errors="coerce")

            valid = pd.DataFrame({
                "clv": clv_values,
                "lower": lower,
                "upper": upper,
            }).dropna()

            valid = valid.sort_values("clv").reset_index(drop=True)

            if len(valid) > 1000:
                valid = valid.iloc[::max(1, len(valid) // 1000)]

            fig = plot_uncertainty_band(
                x=np.arange(len(valid)),
                y=valid["clv"].values,
                lower=valid["lower"].values,
                upper=valid["upper"].values,
                title="Expected Value with Predictive Interval",
                height=500,
                x_title="Customers ordered by Predicted Future Net Revenue",
                y_title="Predicted Future Net Revenue",
            )
            render_chart_responsive(fig, "clv_uncertainty")
        else:
            render_missing_data("No lower/upper uncertainty fields detected in CLV output.")

    # ---- CLV BY SEGMENT ----
    seg_col = None
    for c in ["segment_name", "segment", "cluster"]:
        if c in clv.columns:
            seg_col = c
            break

    if seg_col:
        st.markdown("#### Predicted Future Net Revenue by Behavioral Segment")

        tmp = clv.copy()
        tmp[clv_col] = pd.to_numeric(tmp[clv_col], errors="coerce")
        profile = (
            tmp.dropna(subset=[clv_col])
            .groupby(seg_col)
            .agg(
                customers=(clv_col, "size"),
                total_clv=(clv_col, "sum"),
                median_clv=(clv_col, "median"),
                mean_clv=(clv_col, "mean"),
            )
            .reset_index()
            .sort_values("total_clv", ascending=False)
        )

        fig = plot_horizontal_bar(
            profile,
            x_col="total_clv",
            y_col=seg_col,
            title="Where is Predicted Future Value Concentrated?",
            height=420,
        )
        render_chart_responsive(fig, "clv_by_seg")

    # ---- MARGIN SCENARIOS ----
    if model_card and "margin_scenarios" in model_card:
        st.markdown("#### Margin Scenario Analysis")

        margin_data = model_card["margin_scenarios"]
        scenario_names = list(margin_data.keys())

        selected_scenario = st.selectbox(
            "Margin Scenario",
            options=scenario_names,
            format_func=lambda x: x.replace("_", " ").title(),
        )

        if selected_scenario:
            scenario_clv = margin_data[selected_scenario]
            scenario_df = pd.DataFrame(scenario_clv)

            if not scenario_df.empty:
                st.markdown(f"**{selected_scenario.replace('_', ' ').title()}**")
                st.dataframe(
                    scenario_df[["Customer ID", "clv_mean", "clv_median", "clv_p10", "clv_p90"]].head(20),
                    use_container_width=True,
                    hide_index=True,
                )

                render_science_card(
                    "Margin Scenarios",
                    "The Online Retail II dataset does not contain cost-of-goods-sold (COGS) data. "
                    "CLV is computed as discounted net revenue. "
                    "Apply a realistic contribution margin (e.g., 20–40%) to estimate economic CLV: "
                    "**Economic CLV ≈ Predicted Future Net Revenue × Contribution Margin**. "
                    "The margin scenarios show the effect of different margin assumptions.",
                )

    # ---- MODEL DIAGNOSTICS ----
    if model_card:
        with st.expander("Model Diagnostics"):
            metrics = model_card.get("metrics", {})

            if "purchase_model" in metrics:
                st.markdown("**Purchase Probability Model**")
                pm = metrics["purchase_model"]
                cols = st.columns(4)
                cols[0].metric("ROC-AUC", f"{pm.get('roc_auc', 0):.3f}")
                cols[1].metric("PR-AUC", f"{pm.get('average_precision', 0):.3f}")
                cols[2].metric("Brier Score", f"{pm.get('brier_score', 0):.3f}")
                cols[3].metric("Val. Rows", format_count(pm.get('validation_rows', 0)))

            if "conditional_spend_model" in metrics:
                st.markdown("**Conditional Spend Model (Active Months Only)**")
                sm = metrics["conditional_spend_model"]
                cols = st.columns(4)
                cols[0].metric("MAE", format_currency(sm.get('mae', 0)))
                cols[1].metric("RMSE", format_currency(sm.get('rmse', 0)))
                cols[2].metric("R²", f"{sm.get('r2', 0):.3f}")
                cols[3].metric("Val. Active Rows", format_count(sm.get('validation_active_rows', 0)))

            if "validation" in metrics:
                st.markdown("**Validation Window**")
                val = metrics["validation"]
                st.write(f"Validation Start: {val.get('validation_start', 'N/A')}")
                st.write(f"Validation Months: {val.get('validation_months', 'N/A')}")

            if "backtest_results" in metrics and metrics["backtest_results"]:
                st.markdown("**Rolling Origin Backtests**")
                st.json(metrics["backtest_results"])

    # ---- FEATURE IMPORTANCE ----
    if feature_importance is not None and not feature_importance.empty:
        with st.expander("Feature Importance"):
            st.dataframe(feature_importance, use_container_width=True, hide_index=True)

    # ---- VALIDATION PREDICTIONS ----
    if validation_preds is not None and not validation_preds.empty:
        with st.expander("Validation Predictions (Sample)"):
            st.dataframe(validation_preds.head(100), use_container_width=True, hide_index=True)

    # ---- SCIENTIFIC EXPLANATION ----
    st.markdown("---")
    render_science_card(
        "What Is Being Estimated?",
        "**Predicted Future Net Revenue** — The discounted sum of expected future net revenue over the forecast horizon (default 24 months). "
        "This is a *proxy* for Customer Lifetime Value. True economic CLV would require contribution margin data, "
        "which is not available in the Online Retail II dataset.",
    )

    render_science_card(
        "How Far Ahead?",
        f"Forecast horizon: **{model_card.get('methodology', '24 months')}** (configurable). "
        "The model projects month-by-month purchase probability and conditional spend, "
        "then discounts at 10% annual rate (≈0.8% monthly).",
    )

    render_science_card(
        "What Does Uncertainty Mean?",
        "The p10–p90 interval comes from Monte Carlo simulation (default 200 paths). "
        "It reflects *model uncertainty* — variability in predicted purchase/spend paths — "
        "not parameter estimation uncertainty. **Empirical coverage has not been independently verified.**",
    )

    render_science_card(
        "What Is Not Included?",
        "• No contribution margin / COGS (margin scenarios are assumptions)\n"
        "• No probabilistic lifetime model (BG/NBD, Gamma-Gamma)\n"
        "• No churn process separate from purchase probability\n"
        "• No causal treatment effects — this is observational forecasting",
    )


if __name__ == "__main__":
    render_predictive_page()