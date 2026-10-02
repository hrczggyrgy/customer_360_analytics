"""
Activation workspace for Retail Customer Intelligence.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ..ui import (
    render_section_label,
    render_science_card,
    render_kpi_row,
    render_missing,
    render_audience_table,
    apply_global_scope,
)
from ..app_data import get_registry
from ..app_formatting import format_currency, format_count, format_probability, format_score
from ..ui.charts import (
    plot_action_allocation,
    plot_clv_vs_churn_by_action,
    PLOTLY_CONFIG,
)


def render() -> None:
    """Render the Activation workspace."""
    registry = get_registry()
    
    # Load data
    decision = registry.load_dataframe("decision_engine")
    action_summary = registry.load_dataframe("decision_engine", "action_summary.csv")
    combined = registry.load_dataframe("customer_360")
    
    if decision is None:
        render_missing("Run decision_engine.py to populate this workspace.")
        st.stop()
    
    # Apply global scope
    if combined is not None:
        combined = apply_global_scope(combined)
    
    # =============================================================================
    # COLUMN DETECTION
    # =============================================================================
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
    d_next = "next_purchase_30d_probability" if "next_purchase_30d_probability" in decision.columns else None
    for c in ["next_purchase_probability", "next_purchase_probability_30d", "purchase_probability_30d"]:
        if c in decision.columns:
            d_next = c
            break
    d_id = "Customer ID" if "Customer ID" in decision.columns else None
    d_confidence = "decision_confidence" if "decision_confidence" in decision.columns else None
    d_reason = "action_reason" if "action_reason" in decision.columns else None
    d_eligible = "eligible" if "eligible" in decision.columns else None
    d_suppressed = "suppressed" if "suppressed" in decision.columns else None
    d_suppression_reason = "suppression_reason" if "suppression_reason" in decision.columns else None
    
    if not action_col:
        st.error("Decision file found but no action field was detected.")
        st.stop()
    
    # =============================================================================
    # SECTION 1: CAPACITY BOARD
    # =============================================================================
    render_section_label("Capacity board")
    
    # Get capacity from action_summary or infer
    total_capacity = len(decision)
    active_mask = ~decision[action_col].astype(str).str.startswith("monitor", na=False)
    allocated = int(active_mask.sum())
    remaining = total_capacity - allocated
    utilization = allocated / total_capacity * 100 if total_capacity > 0 else 0
    
    cap_cols = st.columns(4)
    with cap_cols[0]:
        st.metric("Total Capacity", f"{total_capacity:,}")
    with cap_cols[1]:
        st.metric("Allocated", f"{allocated:,}")
    with cap_cols[2]:
        st.metric("Remaining", f"{remaining:,}")
    with cap_cols[3]:
        st.metric("Utilization", f"{utilization:.0f}%")
    
    # Progress bar
    st.progress(utilization / 100)
    
    # =============================================================================
    # SECTION 2: ACTION ALLOCATION
    # =============================================================================
    render_section_label("Action allocation")
    
    left, right = st.columns([1, 1])
    
    with left:
        # Action allocation chart
        action_counts = decision[action_col].value_counts().reset_index()
        action_counts.columns = [action_col, "count"]
        
        fig = plot_action_allocation(
            action_counts,
            action_col=action_col,
            count_col="count",
            title="Customers by recommended action",
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    with right:
        # CLV vs Churn by action
        if decision_clv and d_churn:
            fig = plot_clv_vs_churn_by_action(
                decision,
                clv_col=decision_clv,
                churn_col=d_churn,
                action_col=action_col,
                title="Value vs Risk by Recommended Action",
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            render_missing("CLV and churn fields needed for decision scatter.")
    
    # Action capacity detail
    if action_summary is not None:
        st.markdown("#### Action capacity detail")
        render_audience_table(
            action_summary,
            download_filename="action_summary.csv",
            download_label="Export action summary",
        )
    
    # =============================================================================
    # SECTION 3: OPPORTUNITY MATRIX
    # =============================================================================
    render_section_label("Opportunity matrix — Value vs Risk vs Propensity")
    
    if decision_clv and d_churn and d_next:
        plot_df = decision.dropna(subset=[decision_clv, d_churn, d_next, action_col]).copy()
        plot_df[decision_clv] = pd.to_numeric(plot_df[decision_clv], errors="coerce")
        plot_df[d_churn] = pd.to_numeric(plot_df[d_churn], errors="coerce")
        plot_df[d_next] = pd.to_numeric(plot_df[d_next], errors="coerce")
        plot_df = plot_df.dropna(subset=[decision_clv, d_churn, d_next])
        
        if len(plot_df) > 8000:
            plot_df = plot_df.sample(8000, random_state=42)
        
        from ..ui.charts import plot_scatter
        
        fig = plot_scatter(
            plot_df,
            x_col=decision_clv,
            y_col=d_churn,
            size_col=d_next,
            color_col=action_col,
            title="Action Allocation: Value vs Risk (bubble size = Purchase Propensity)",
            labels={decision_clv: "Predicted Future Value", d_churn: "Inactivity Risk", d_next: "30d Purchase Propensity"},
            log_x=True,
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Quadrant summary
        x_med = plot_df[decision_clv].median()
        y_med = plot_df[d_churn].median()
        
        st.markdown("#### Quadrant allocation")
        quad_cols = st.columns(4)
        quadrants = {
            "Protect (High value, High risk)": plot_df[(plot_df[decision_clv] >= x_med) & (plot_df[d_churn] >= y_med)],
            "Grow (High value, Low risk)": plot_df[(plot_df[decision_clv] >= x_med) & (plot_df[d_churn] < y_med)],
            "Reactivate (Low value, High risk)": plot_df[(plot_df[decision_clv] < x_med) & (plot_df[d_churn] >= y_med)],
            "Accelerate (Low value, Low risk)": plot_df[(plot_df[decision_clv] < x_med) & (plot_df[d_churn] < y_med)],
        }
        
        for (label, qdf), col in zip(quadrants.items(), quad_cols):
            with col:
                n = len(qdf)
                val = qdf[decision_clv].sum()
                prop = qdf[d_next].mean()
                st.metric(label, f"{n:,}", f"£{val:,.0f}")
                st.caption(f"Avg propensity: {prop:.1%}")
    else:
        render_missing("CLV, churn, and next-purchase columns required.")
    
    # =============================================================================
    # SECTION 4: PRIORITY QUEUE
    # =============================================================================
    render_section_label("Priority queue — Operational target list")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top_n = st.slider("Top N customers", 10, 500, 100, 10, key="activation_top_n")
    
    with col2:
        action_filter = st.selectbox(
            "Action",
            ["All"] + sorted(decision[action_col].unique().tolist()),
            key="activation_action_filter",
        )
    
    with col3:
        segment_filter = "All"
        if combined is not None and "segment_name" in combined.columns:
            segments = ["All"] + sorted(combined["segment_name"].dropna().unique().tolist())
            segment_filter = st.selectbox("Segment", segments, key="activation_segment_filter")
    
    with col4:
        risk_filter = st.selectbox(
            "Risk band",
            ["All", "High (≥70%)", "Medium (30-70%)", "Low (<30%)"],
            key="activation_risk_filter",
        )
    
    # Apply filters
    filtered = decision.copy()
    
    if action_filter != "All":
        filtered = filtered[filtered[action_col] == action_filter]
    
    if segment_filter != "All" and combined is not None:
        seg_customers = combined[combined["segment_name"] == segment_filter]["Customer ID"].unique()
        filtered = filtered[filtered[d_id].isin(seg_customers)]
    
    if risk_filter != "All" and d_churn:
        churn_vals = pd.to_numeric(filtered[d_churn], errors="coerce")
        if risk_filter == "High (≥70%)":
            filtered = filtered[churn_vals >= 0.7]
        elif risk_filter == "Medium (30-70%)":
            filtered = filtered[(churn_vals >= 0.3) & (churn_vals < 0.7)]
        elif risk_filter == "Low (<30%)":
            filtered = filtered[churn_vals < 0.3]
    
    # Sort by priority
    if priority_col:
        filtered[priority_col] = pd.to_numeric(filtered[priority_col], errors="coerce")
        filtered = filtered.sort_values(priority_col, ascending=False).head(top_n)
    else:
        filtered = filtered.head(top_n)
    
    # Display columns
    preferred_columns = []
    for c in [d_id, action_col, "priority_tier", priority_col, decision_clv, d_churn, d_next,
              "reactivation_probability", d_confidence, d_reason]:
        if c and c in filtered.columns:
            preferred_columns.append(c)
    
    display = filtered[preferred_columns].copy()
    
    # Format for display
    for col in display.columns:
        if col in [decision_clv]:
            display[col] = display[col].apply(format_currency)
        elif col in [d_churn, d_next, "reactivation_probability", d_confidence]:
            display[col] = display[col].apply(format_probability)
        elif col == priority_col:
            display[col] = display[col].apply(format_score)
    
    render_audience_table(
        display,
        columns=preferred_columns,
        download_filename="activation_priority_queue.csv",
        download_label="Export priority queue",
    )
    
    # =============================================================================
    # SECTION 5: ELIGIBILITY / SUPPRESSION
    # =============================================================================
    render_section_label("Eligibility & suppression")
    
    if d_eligible or d_suppressed:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            eligible_count = int(decision[d_eligible].sum()) if d_eligible else "N/A"
            st.metric("Eligible", f"{eligible_count:,}")
        
        with col2:
            suppressed_count = int(decision[d_suppressed].sum()) if d_suppressed else "N/A"
            st.metric("Suppressed", f"{suppressed_count:,}")
        
        with col3:
            capacity_excluded = len(decision) - eligible_count if d_eligible else "N/A"
            st.metric("Capacity excluded", f"{capacity_excluded:,}")
        
        if d_suppressed and d_suppression_reason:
            st.markdown("#### Suppression reasons")
            suppressed_df = decision[decision[d_suppressed] == True]
            if not suppressed_df.empty:
                reason_counts = suppressed_df[d_suppression_reason].value_counts().reset_index()
                reason_counts.columns = ["Suppression Reason", "Count"]
                st.dataframe(reason_counts, use_container_width=True, hide_index=True)
    else:
        render_missing("Eligibility/suppression fields not available in decision output.")
    
    # =============================================================================
    # SECTION 6: EXPORT
    # =============================================================================
    render_section_label("Export actionable audience")
    
    st.markdown("Export the filtered priority audience for campaign execution.")
    
    export_cols = st.multiselect(
        "Columns to include",
        options=decision.columns.tolist(),
        default=preferred_columns,
        key="activation_export_cols",
    )
    
    if export_cols:
        export_df = filtered[export_cols].copy()
        
        # Format for CSV
        for col in export_df.columns:
            if col in [decision_clv]:
                export_df[col] = pd.to_numeric(export_df[col], errors="coerce")
            elif col in [d_churn, d_next, "reactivation_probability", d_confidence]:
                export_df[col] = pd.to_numeric(export_df[col], errors="coerce")
        
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="activation_audience.csv",
            mime="text/csv",
            help=f"Export {len(export_df):,} customers for activation",
        )
        
        st.caption(f"Preview: {len(export_df):,} rows × {len(export_cols)} columns")
    
    render_science_card(
        "What this engine does — and does not do",
        "It estimates **expected commercial opportunity** from the "
        "analytical signals already produced in the project. It then applies a "
        "transparent policy with capacity constraints. It does not infer an "
        "incremental treatment effect from observational purchase history. "
        "The next portfolio upgrade is an experiment or quasi-experiment that "
        "measures actual incremental response to each intervention."
    )