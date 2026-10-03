"""
Customer 360 page for Retail Customer Intelligence.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..ui import (
    render_customer_header,
    render_customer_metric_row,
    render_customer_selector,
    render_science_card,
    render_evidence_table,
    render_missing,
    render_empty_state,
)
from ..app_data import get_registry
from ..app_formatting import (
    format_currency,
    format_probability,
    format_percent,
    format_count,
    format_days,
    auto_format,
)
from ..ui.charts import (
    base_layout,
    plot_missing,
    plot_dual_axis_line_bar,
    PLOTLY_CONFIG,
)


def render() -> None:
    """Render the Customer 360 page."""
    registry = get_registry()
    
    # Load data
    customer_df = registry.load_dataframe("customer_360")
    # Use canonical customer-month panel from customer_360 output
    monthly = registry.load_dataframe("customer_360", "customer_month.parquet")
    
    if customer_df is None or customer_df.empty:
        render_empty_state(
            "No customer data",
            "No customer-level table was detected.",
            "Run the pipeline",
            "python scripts/customer_360.py",
        )
        st.stop()
    
    id_col = "Customer ID"
    
    # Check for deep linking via query params
    query_params = st.query_params
    default_customer = None
    if "customer_id" in query_params:
        try:
            default_customer = int(query_params["customer_id"])
        except (ValueError, TypeError):
            pass
    
    st.caption(f"Data source: **Customer 360** — `{len(customer_df):,}` customers")
    
    # Customer selector
    selected_id = render_customer_selector(customer_df, id_col, default_customer=default_customer)
    
    if selected_id is None:
        st.stop()
    
    row = customer_df[customer_df[id_col].astype(int) == int(selected_id)]
    
    if row.empty:
        render_missing("Customer not found.")
        st.stop()
    
    row = row.iloc[0]
    
    # Extract key values - use safe null-aware extraction
    def safe_get(row, *keys):
        """Get first non-null value from row for given keys."""
        for k in keys:
            if k in row and pd.notna(row[k]):
                return row[k]
        return None
    
    segment_value = safe_get(row, "segment_name", "segment")
    clv_value = safe_get(row, "clv_mean", "clv", "predicted_clv", "customer_clv")
    churn_value = safe_get(row, "churn_probability", "churn_prob", "prob_churn")
    purchase_value = safe_get(row, "next_purchase_probability", "next_purchase_probability_30d", "purchase_probability_30d")
    action_value = safe_get(row, "recommended_action_capped", "final_action", "recommended_action")
    
    # Customer header with key metrics
    render_customer_header(
        customer_id=int(selected_id),
        segment=segment_value,
        action=action_value,
        clv=pd.to_numeric(clv_value, errors="coerce") if clv_value is not None else None,
        churn_risk=pd.to_numeric(churn_value, errors="coerce") if churn_value is not None else None,
    )
    
    # Metric cards
    revenue_value = safe_get(row, "net_revenue", "revenue", "gross_revenue")
    orders_value = safe_get(row, "orders", "invoice_count")
    
    render_customer_metric_row([
        ("Predicted Future Value", clv_value, "currency"),
        ("Inactivity Risk", churn_value, "probability"),
        ("30d Purchase Propensity", purchase_value, "probability"),
        ("Revenue", revenue_value, "currency"),
        ("Orders", orders_value, "count"),
    ])
    
    left, right = st.columns([1.2, 0.8])
    
    with left:
        st.markdown("#### Customer purchase trajectory")
        
        if monthly is not None:
            m_id_col = "Customer ID"
            month_col = "calendar_month"
            revenue_col = "net_revenue"
            order_col = "orders"
            
            if all(c in monthly.columns for c in [m_id_col, month_col, revenue_col]):
                hist = monthly[
                    pd.to_numeric(monthly[m_id_col], errors="coerce").round() == int(selected_id)
                ].copy()
                
                if not hist.empty:
                    hist[month_col] = pd.to_datetime(hist[month_col], errors="coerce")
                    hist[revenue_col] = pd.to_numeric(hist[revenue_col], errors="coerce")
                    hist = hist.dropna(subset=[month_col]).sort_values(month_col)
                    
                    fig = plot_dual_axis_line_bar(
                        hist,
                        x_col=month_col,
                        bar_col=revenue_col,
                        line_col=order_col,
                        title="Observed monthly customer economics",
                        bar_name="Net revenue",
                        line_name="Orders",
                    )
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    plot_missing("No monthly event history was found for this customer.")
            else:
                plot_missing("Customer-month columns were not recognized.")
        else:
            plot_missing("Run customer_360.py to populate monthly customer history.")
    
    with right:
        st.markdown("#### Model evidence")
        
        evidence_rows = []
        
        field_map = [
            ("Behavioral segment", ["segment_name", "segment"]),
            ("CLV", ["clv_mean", "clv", "predicted_clv", "customer_clv"]),
            ("CLV lower", ["clv_lower", "clv_p10", "clv_lower_bound"]),
            ("CLV upper", ["clv_upper", "clv_p90", "clv_upper_bound"]),
            ("Inactivity Risk", ["churn_probability", "churn_prob", "prob_churn"]),
            ("Next-purchase probability", ["next_purchase_probability", "next_purchase_probability_30d", "purchase_probability_30d"]),
            ("Reactivation probability", ["reactivation_probability"]),
            ("Expected days to next purchase", ["expected_days_to_next_purchase"]),
            ("Recency pressure", ["recency_pressure"]),
            ("Policy Score", ["decision_confidence"]),
        ]
        
        for label, candidates in field_map:
            value = None
            for c in candidates:
                if c in row and pd.notna(row[c]):
                    value = row[c]
                    break
            if value is None:
                continue
            
            if "probability" in label.lower():
                display = format_probability(value)
            elif "score" in label.lower():
                display = format_score(value)
            elif "clv" in label.lower():
                display = format_currency(value)
            elif "days" in label.lower():
                display = format_days(value)
            else:
                display = auto_format(value)
            
            evidence_rows.append({"Metric": label, "Value": display})
        
        if evidence_rows:
            render_evidence_table(evidence_rows)
        
        render_science_card(
            "How to read this customer",
            "The dashboard intentionally separates **value**, **risk**, and **propensity**. "
            "A high-value customer is not automatically a good intervention target; "
            "the action layer looks for a commercially meaningful signal and a "
            "sufficiently strong model-confidence context."
        )