"""
Customer 360 page — Search-first individual customer deep dive.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

import pandas as pd
import streamlit as st

from app_components import (
    render_page_hero,
    render_customer_header,
    render_customer_selector,
    render_evidence_table,
    build_evidence_rows,
    render_missing_data,
    render_section_label,
    render_science_card,
    render_download_button,
)
from app_charts import (
    plot_time_series,
    render_chart_responsive,
)
from app_data import get_registry, get_profile_adapter
from app_formatting import (
    format_clv, format_churn_risk, format_next_purchase, format_currency,
    format_count, format_date, format_duration_months, format_probability,
    format_percent, auto_format,
)


def render_customer_360_page() -> None:
    """Render the Customer 360 page."""
    registry = get_registry()
    adapter = get_profile_adapter(registry)

    # Build unified customer profile
    customer_df = adapter.build_profile()

    if customer_df is None or customer_df.empty:
        st.warning("No customer-level table was detected. Run customer_360.py first.")
        st.stop()

    # Data source info
    source_info = adapter.get_source_info()
    source_labels = ", ".join(source_info.keys())
    st.markdown(f"Data sources: **{source_labels}** — `{len(customer_df):,}` customers")

    # Customer selector
    selected_id = render_customer_selector(customer_df, key_prefix="c360")

    if selected_id is None:
        st.stop()

    # Get customer row
    row = customer_df[customer_df["Customer ID"] == selected_id]
    if row.empty:
        st.warning("Customer not found.")
        st.stop()

    row = row.iloc[0]

    # Extract key fields for header
    segment = None
    for c in ["segment_name", "segment", "cluster"]:
        if c in row.index and pd.notna(row[c]):
            segment = str(row[c])
            break

    action = None
    for c in ["final_action", "recommended_action"]:
        if c in row.index and pd.notna(row[c]):
            action = str(row[c])
            break

    lifecycle_state = None
    for c in ["lifecycle_state", "customer_state"]:
        if c in row.index and pd.notna(row[c]):
            lifecycle_state = str(row[c])
            break

    cohort = None
    for c in ["cohort_month", "cohort"]:
        if c in row.index and pd.notna(row[c]):
            cohort = format_date(row[c])
            break

    # Page hero with customer context
    render_page_hero(
        title=f"Customer {selected_id}",
        description="Explore the current economic, behavioral, lifecycle and predictive state of an individual customer.",
        kicker="CUSTOMER INTELLIGENCE",
    )

    # Customer header
    render_customer_header(selected_id, segment, action, lifecycle_state, cohort)

    # ---- PRIMARY KPIs ----
    clv_val = None
    for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
        if c in row.index and pd.notna(row[c]):
            clv_val = row[c]
            break

    churn_val = None
    for c in ["churn_probability", "churn_prob", "prob_churn"]:
        if c in row.index and pd.notna(row[c]):
            churn_val = row[c]
            break

    np_val = None
    for c in ["next_purchase_probability_30d", "next_purchase_probability", "purchase_probability_30d"]:
        if c in row.index and pd.notna(row[c]):
            np_val = row[c]
            break

    revenue_val = None
    for c in ["lifetime_net_revenue", "net_revenue", "revenue", "lifetime_gross_revenue"]:
        if c in row.index and pd.notna(row[c]):
            revenue_val = row[c]
            break

    orders_val = None
    for c in ["lifetime_orders", "orders", "invoice_count"]:
        if c in row.index and pd.notna(row[c]):
            orders_val = row[c]
            break

    cards = st.columns(5)
    with cards[0]:
        st.metric("Predicted Future Net Revenue", format_clv(clv_val))
    with cards[1]:
        st.metric("Next-Month Inactivity Risk", format_churn_risk(churn_val))
    with cards[2]:
        st.metric("30-Day Purchase Propensity", format_next_purchase(np_val))
    with cards[3]:
        st.metric("Historical Net Revenue", format_currency(revenue_val))
    with cards[4]:
        st.metric("Historical Orders", format_count(orders_val))

    # Load monthly history
    monthly = registry.load_dataframe("customer_360", "customer_month_events")
    if monthly is None:
        monthly = registry.load_dataframe("data_quality", "customer_month")

    # ---- TABS ----
    tabs = st.tabs([
        "Overview",
        "Economics",
        "Behavior",
        "Lifecycle",
        "Risk & Propensity",
        "Predictive Value",
        "Products",
        "Decision",
    ])

    with tabs[0]:
        render_overview_tab(row, monthly, selected_id)

    with tabs[1]:
        render_economics_tab(row, monthly, selected_id)

    with tabs[2]:
        render_behavior_tab(row)

    with tabs[3]:
        render_lifecycle_tab(row, monthly, selected_id)

    with tabs[4]:
        render_risk_propensity_tab(row)

    with tabs[5]:
        render_predictive_value_tab(row)

    with tabs[6]:
        render_products_tab(row, registry, selected_id)

    with tabs[7]:
        render_decision_tab(row)

    # ---- DOWNLOAD ----
    st.markdown("---")
    customer_data = customer_df[customer_df["Customer ID"] == selected_id]
    render_download_button(
        customer_data,
        f"customer_{selected_id}_profile.csv",
        "Download Customer Profile (CSV)",
        "Download the full customer profile as CSV",
    )


def render_overview_tab(row: pd.Series, monthly: Optional[pd.DataFrame], customer_id: int) -> None:
    """Render the Overview tab with trajectory and key evidence."""
    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown("#### Purchase Trajectory")
        if monthly is not None:
            m_id_col = "Customer ID"
            month_col = "calendar_month" if "calendar_month" in monthly.columns else "month"
            revenue_col = "net_revenue" if "net_revenue" in monthly.columns else "gross_revenue"
            order_col = "orders" if "orders" in monthly.columns else "invoice_count"

            if m_id_col in monthly.columns and month_col in monthly.columns and revenue_col in monthly.columns:
                hist = monthly[
                    pd.to_numeric(monthly[m_id_col], errors="coerce").round() == customer_id
                ].copy()

                if not hist.empty:
                    hist[month_col] = pd.to_datetime(hist[month_col], errors="coerce")
                    hist[revenue_col] = pd.to_numeric(hist[revenue_col], errors="coerce")
                    hist = hist.dropna(subset=[month_col]).sort_values(month_col)

                    fig = plot_time_series(
                        hist,
                        date_col=month_col,
                        value_cols=[revenue_col],
                        title="Observed Monthly Net Revenue",
                        height=460,
                        secondary_y_cols=[order_col] if order_col in hist.columns else None,
                    )
                    render_chart_responsive(fig, f"cust_traj_{customer_id}")
                else:
                    render_missing_data("No monthly event history found for this customer.")
            else:
                render_missing_data("Customer-month columns not recognized.")
        else:
            render_missing_data("Run cohort_analysis.py or customer_360.py to populate monthly customer history.")

    with right:
        st.markdown("#### Model Evidence")

        field_map = [
            ("Behavioral segment", ["segment_name", "segment", "cluster"], lambda v: str(v)),
            ("Predicted Future Net Revenue", ["clv_mean", "clv", "predicted_clv"], format_clv),
            ("CLV p10", ["clv_p10"], format_clv),
            ("CLV p90", ["clv_p90"], format_clv),
            ("Next-month inactivity risk", ["churn_probability", "churn_prob"], format_churn_risk),
            ("30-day purchase propensity", ["next_purchase_probability_30d", "next_purchase_probability"], format_next_purchase),
            ("7-day purchase propensity", ["next_purchase_probability_7d"], format_next_purchase),
            ("60-day purchase propensity", ["next_purchase_probability_60d"], format_next_purchase),
            ("Reactivation probability", ["reactivation_probability"], format_probability),
            ("Expected days to next purchase", ["expected_days_to_next_purchase"], format_duration_months),
            ("Recency (days)", ["recency_days"], format_count),
            ("Decision confidence", ["decision_confidence"], format_probability),
        ]

        evidence_rows = build_evidence_rows(row, field_map)

        if evidence_rows:
            render_evidence_table(evidence_rows)
        else:
            render_missing_data("No model evidence available for this customer.")

        render_science_card(
            "How to read this customer",
            "The dashboard intentionally separates **value**, **risk**, and **propensity**. "
            "A high predicted future net revenue customer is not automatically a good intervention target; "
            "the action layer looks for a commercially meaningful signal and a sufficiently strong model-confidence context.",
        )


def render_economics_tab(row: pd.Series, monthly: Optional[pd.DataFrame], customer_id: int) -> None:
    """Render the Economics tab."""
    st.markdown("#### Historical Economics")

    # Key economic metrics
    econ_fields = [
        ("Lifetime Gross Revenue", ["lifetime_gross_revenue", "gross_revenue"], format_currency),
        ("Lifetime Net Revenue", ["lifetime_net_revenue", "net_revenue"], format_currency),
        ("Lifetime Orders", ["lifetime_orders", "orders", "invoice_count"], format_count),
        ("Average Order Value", ["historical_avg_order_value", "avg_order_value", "aov"], format_currency),
        ("Lifetime Units", ["lifetime_units", "units"], format_count),
        ("Lifetime Return Value", ["lifetime_return_value", "return_value"], format_currency),
        ("Return Rate", ["lifetime_return_rate", "return_rate"], lambda v: format_percent(v, already_percentage=True)),
        ("First Purchase", ["first_purchase_date", "first_order_date"], format_date),
        ("Last Purchase", ["last_purchase_date", "last_order_date"], format_date),
        ("Tenure (days)", ["tenure_days"], format_count),
        ("Active Months", ["active_month_count", "active_months"], format_count),
    ]

    evidence = build_evidence_rows(row, econ_fields)
    if evidence:
        render_evidence_table(evidence)

    # Monthly trajectory
    if monthly is not None:
        m_id_col = "Customer ID"
        month_col = "calendar_month" if "calendar_month" in monthly.columns else "month"
        revenue_col = "net_revenue" if "net_revenue" in monthly.columns else "gross_revenue"
        order_col = "orders" if "orders" in monthly.columns else "invoice_count"
        return_col = "return_value" if "return_value" in monthly.columns else None

        if m_id_col in monthly.columns:
            hist = monthly[
                pd.to_numeric(monthly[m_id_col], errors="coerce").round() == customer_id
            ].copy()

            if not hist.empty:
                hist[month_col] = pd.to_datetime(hist[month_col], errors="coerce")
                hist = hist.dropna(subset=[month_col]).sort_values(month_col)

                st.markdown("#### Monthly Revenue & Orders")
                value_cols = [revenue_col]
                if order_col in hist.columns:
                    value_cols.append(order_col)
                if return_col and return_col in hist.columns:
                    value_cols.append(return_col)

                fig = plot_time_series(
                    hist,
                    date_col=month_col,
                    value_cols=value_cols,
                    title="Monthly Economics",
                    height=460,
                    secondary_y_cols=[order_col] if order_col in hist.columns else None,
                )
                render_chart_responsive(fig, f"cust_econ_{customer_id}")


def render_behavior_tab(row: pd.Series) -> None:
    """Render the Behavior tab."""
    st.markdown("#### Behavioral Profile")

    behavior_fields = [
        ("Purchase Cadence (median days)", ["median_interpurchase_days", "interpurchase_days_median"], format_duration_months),
        ("Cadence Volatility (CV)", ["interpurchase_cv", "cadence_cv"], lambda v: format_score(v, 2)),
        ("Active Months", ["active_month_count", "active_months"], format_count),
        ("Assortment Breadth (unique products)", ["unique_products", "product_breadth"], format_count),
        ("Product Concentration (HHI)", ["product_revenue_hhi", "revenue_hhi"], lambda v: format_score(v, 3)),
        ("Repeat Product Ratio", ["repeat_product_ratio", "repeat_ratio"], lambda v: format_percent(v, already_percentage=True)),
        ("Mean Unit Price", ["mean_unit_price", "avg_price"], format_currency),
        ("Price Volatility (CV)", ["unit_price_cv", "price_cv"], lambda v: format_score(v, 2)),
        ("Premium Price Share", ["premium_price_line_share", "premium_share"], lambda v: format_percent(v, already_percentage=True)),
        ("Hour Entropy", ["hour_entropy"], lambda v: format_score(v, 2)),
        ("Weekend Order Share", ["weekend_order_share"], lambda v: format_percent(v, already_percentage=True)),
        ("Month Entropy", ["month_entropy"], lambda v: format_score(v, 2)),
    ]

    evidence = build_evidence_rows(row, behavior_fields)
    if evidence:
        render_evidence_table(evidence)

    # Technical features expander
    with st.expander("Show Technical Features (All 100+)"):
        # Show all numeric columns
        numeric_cols = row.index[row.apply(lambda x: isinstance(x, (int, float)) and not pd.isna(x))]
        tech_df = pd.DataFrame({"Feature": numeric_cols, "Value": [row[c] for c in numeric_cols]})
        tech_df["Value"] = tech_df.apply(lambda r: auto_format(r["Value"], r["Feature"]), axis=1)
        st.dataframe(tech_df, use_container_width=True, hide_index=True)


def render_lifecycle_tab(row: pd.Series, monthly: Optional[pd.DataFrame], customer_id: int) -> None:
    """Render the Lifecycle tab."""
    st.markdown("#### Lifecycle State")

    lifecycle_fields = [
        ("Lifecycle State", ["lifecycle_state", "customer_state"], lambda v: str(v)),
        ("Cohort Month", ["cohort_month", "cohort"], format_month),
        ("Tenure (days)", ["tenure_days"], format_count),
        ("Recency (days)", ["recency_days", "recency"], format_count),
        ("Months Since First Purchase", ["months_since_first_purchase", "age_month"], format_count),
        ("Reactivation Count", ["reactivation_count", "n_reactivations"], format_count),
        ("Has Reactivated", ["has_reactivated"], lambda v: "Yes" if v else "No"),
        ("Churn Transition Count", ["churn_transition_count"], format_count),
    ]

    evidence = build_evidence_rows(row, lifecycle_fields)
    if evidence:
        render_evidence_table(evidence)

    # Lifecycle timeline from monthly data
    if monthly is not None:
        m_id_col = "Customer ID"
        month_col = "calendar_month" if "calendar_month" in monthly.columns else "month"
        active_col = "active_flag" if "active_flag" in monthly.columns else None

        if m_id_col in monthly.columns and active_col:
            hist = monthly[
                pd.to_numeric(monthly[m_id_col], errors="coerce").round() == customer_id
            ].copy()

            if not hist.empty:
                hist[month_col] = pd.to_datetime(hist[month_col], errors="coerce")
                hist = hist.dropna(subset=[month_col]).sort_values(month_col)

                st.markdown("#### Activity Timeline")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hist[month_col],
                    y=hist[active_col],
                    name="Active Month",
                    marker_color="#315efb",
                    hovertemplate="%{x|%Y-%m}<br>Active: %{y}<extra></extra>",
                ))
                fig.update_xaxes(title="Calendar Month")
                fig.update_yaxes(title="Active (1/0)", range=[-0.1, 1.1])
                render_chart_responsive(fig, f"cust_lifecycle_{customer_id}")


def render_risk_propensity_tab(row: pd.Series) -> None:
    """Render the Risk & Propensity tab."""
    st.markdown("#### Predictive Risk & Propensity Signals")

    risk_fields = [
        ("Next-Month Inactivity Risk", ["churn_probability", "churn_prob", "prob_churn"], format_churn_risk),
        ("7-Day Purchase Propensity", ["next_purchase_probability_7d"], format_next_purchase),
        ("30-Day Purchase Propensity", ["next_purchase_probability_30d", "next_purchase_probability", "purchase_probability_30d"], format_next_purchase),
        ("60-Day Purchase Propensity", ["next_purchase_probability_60d"], format_next_purchase),
        ("3-Month Survival", ["survival_3m", "survival_3_month"], format_probability),
        ("6-Month Survival", ["survival_6m", "survival_6_month"], format_probability),
        ("12-Month Survival", ["survival_12m", "survival_12_month"], format_probability),
        ("Expected Remaining Active Months", ["expected_remaining_active_months"], format_duration_months),
        ("Reactivation Probability", ["reactivation_probability"], format_probability),
        ("Recency Pressure", ["recency_pressure"], lambda v: format_score(v, 2)),
    ]

    evidence = build_evidence_rows(row, risk_fields)
    if evidence:
        render_evidence_table(evidence)

    # Visual risk profile
    st.markdown("#### Risk Profile Visualization")
    risk_probs = []
    risk_labels = []

    for label, candidates, _ in risk_fields:
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                try:
                    val = float(row[c])
                    if 0 <= val <= 10:  # probabilities or survival
                        risk_probs.append(val if val <= 1 else val / 100)
                        risk_labels.append(label)
                except Exception:
                    pass
                break

    if risk_probs:
        import plotly.graph_objects as go
        fig = go.Figure()
        colors = []
        for p in risk_probs:
            if p >= 0.70:
                colors.append("#c53d32")
            elif p >= 0.40:
                colors.append("#a56600")
            else:
                colors.append("#218739")

        fig.add_trace(go.Bar(
            x=risk_labels,
            y=risk_probs,
            marker_color=colors,
            orientation="v",
            hovertemplate="%{x}: %{y:.0%}<extra></extra>",
        ))
        fig.update_yaxes(title="Probability", tickformat=".0%", range=[0, 1])
        fig.update_xaxes(tickangle=45)
        render_chart_responsive(fig, f"cust_risk_profile_{row['Customer ID']}")


def render_predictive_value_tab(row: pd.Series) -> None:
    """Render the Predictive Value (CLV) tab."""
    st.markdown("#### Predicted Future Net Revenue (CLV Proxy)")

    clv_fields = [
        ("Predicted Future Net Revenue (mean)", ["clv_mean", "clv", "predicted_clv"], format_clv),
        ("Predicted Future Net Revenue (median)", ["clv_median"], format_clv),
        ("Lower Bound (p10)", ["clv_p10", "clv_lower"], format_clv),
        ("Upper Bound (p90)", ["clv_p90", "clv_upper"], format_clv),
        ("Standard Deviation", ["clv_std"], format_currency),
        ("Forecast Horizon", ["clv_horizon_months", "horizon_months"], lambda v: f"{int(v)} months"),
        ("Margin Scenario", ["margin_scenario", "margin_rate"], lambda v: f"{float(v)*100:.0f}%" if v else "100% (revenue only)"),
    ]

    evidence = build_evidence_rows(row, clv_fields)
    if evidence:
        render_evidence_table(evidence)

    render_science_card(
        "Interpretation",
        "This is a **predictive value estimate**, not observed revenue. "
        "It represents the discounted sum of expected future net revenue over the forecast horizon. "
        "The uncertainty band (p10–p90) reflects model uncertainty from Monte Carlo simulation. "
        "The Online Retail II dataset does not contain cost-of-goods-sold data; "
        "the default margin scenario assumes 100% (revenue only). "
        "Apply a realistic contribution margin (e.g., 20–40%) to estimate economic CLV.",
    )


def render_products_tab(row: pd.Series, registry, customer_id: int) -> None:
    """Render the Products tab with recommendations."""
    st.markdown("#### Product Affinity & Recommendations")

    # Check for recommendations
    recs = registry.load_dataframe("recommendations", "recommendations")
    if recs is not None and not recs.empty:
        cust_recs = recs[recs["Customer ID"] == customer_id]
        if not cust_recs.empty:
            st.markdown("**Top Recommended Products**")
            display_cols = ["recommended_product", "score", "reason", "support", "lift"]
            display_cols = [c for c in display_cols if c in cust_recs.columns]

            # Add product details if available
            product_metrics = registry.load_dataframe("product_analytics", "product_metrics")
            if product_metrics is not None:
                prod_info = product_metrics[["StockCode", "Description", "product_role", "avg_price"]].copy()
                prod_info = prod_info.rename(columns={"StockCode": "recommended_product"})
                cust_recs = cust_recs.merge(prod_info, on="recommended_product", how="left")
                display_cols = ["recommended_product", "Description", "product_role", "avg_price"] + display_cols

            st.dataframe(cust_recs[display_cols].head(10), use_container_width=True, hide_index=True)

            render_science_card(
                "Recommendation Basis",
                "Recommendations are based on **observed co-purchase affinity** from historical transactions. "
                "A high score indicates products frequently purchased together with this customer's historical basket. "
                "This is not a guarantee of future purchase — it reflects historical association patterns.",
            )
        else:
            st.info("No specific recommendations generated for this customer.")
    else:
        st.info("Recommendation engine output not available. Run recommendation_engine.py to generate.")

    # Historical product affinity from customer_360
    affinity_cols = [c for c in row.index if c.startswith("share_top_product") or c.startswith("top_product")]
    if affinity_cols:
        st.markdown("#### Historical Product Affinity (Top Products)")
        affinity_data = []
        for c in affinity_cols:
            val = row[c]
            if pd.notna(val):
                product_code = c.replace("share_top_product_pool_", "").replace("top_product_", "")
                affinity_data.append({"Product": product_code, "Revenue Share": format_percent(val, already_percentage=True)})

        if affinity_data:
            st.dataframe(pd.DataFrame(affinity_data), use_container_width=True, hide_index=True)


def render_decision_tab(row: pd.Series) -> None:
    """Render the Decision tab."""
    st.markdown("#### Decision Recommendation")

    decision_fields = [
        ("Recommended Action", ["final_action", "recommended_action"], lambda v: str(v)),
        ("Priority Score", ["priority_score"], lambda v: format_score(v, 1)),
        ("Priority Tier", ["priority_tier", "tier"], lambda v: str(v)),
        ("Decision Confidence", ["decision_confidence"], format_probability),
        ("Action Reason", ["action_reason", "reason"], lambda v: str(v)),
        ("Expected Value Proxy", ["expected_value_proxy", "ev_proxy"], format_currency),
    ]

    evidence = build_evidence_rows(row, decision_fields)
    if evidence:
        render_evidence_table(evidence)

    render_science_card(
        "Why This Action?",
        "The decision engine combines predicted future net revenue, inactivity risk, purchase propensity, "
        "behavioral segment, and model confidence into a transparent prioritization policy. "
        "Capacity constraints are applied per action type. "
        "**This is an observational prioritization — not a causal treatment effect estimate.**",
    )


# Import at module level
import plotly.graph_objects as go