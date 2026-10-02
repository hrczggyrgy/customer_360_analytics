"""
Value & Retention workspace for Retail Customer Intelligence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ..ui import (
    render_section_label,
    render_science_card,
    render_kpi_row,
    render_insight,
    render_insight_row,
    render_missing,
    apply_global_scope,
)
from ..app_data import get_registry
from ..app_formatting import format_currency, format_probability, format_percent, format_count
from ..ui.charts import (
    plot_histogram_with_marginal,
    plot_concentration_curve,
    plot_uncertainty_band,
    plot_scatter_with_quadrants,
    plot_clv_by_segment,
    PLOTLY_CONFIG,
    apply_plotly_theme,
)


def render() -> None:
    """Render the Value & Retention workspace."""
    registry = get_registry()
    
    # Load data
    clv = registry.load_dataframe("clv")
    churn = registry.load_dataframe("churn")
    next_purchase = registry.load_dataframe("churn")  # Same file
    combined = registry.load_dataframe("customer_360")
    decision = registry.load_dataframe("decision_engine")
    
    # Fallback to combined customer_360
    if clv is None:
        combined_all = registry.load_dataframe("customer_360")
        if combined_all is not None:
            clv_col = None
            for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
                if c in combined_all.columns:
                    clv_col = c
                    break
            if clv_col:
                clv = combined_all.copy()
    
    if churn is None and combined is not None and "churn_probability" in combined.columns:
        churn = combined
    if next_purchase is None and combined is not None and "next_purchase_probability" in combined.columns:
        next_purchase = combined
    
    if clv is None and churn is None:
        render_missing("Run clv_analysis.py and churn_next_purchase.py to populate this workspace.")
        st.stop()
    
    # Apply global scope
    if combined is not None:
        combined = apply_global_scope(combined)
    
    # =============================================================================
    # SECTION 1: VALUE DISTRIBUTION
    # =============================================================================
    render_section_label("Value distribution")
    
    if clv is not None:
        ccol = None
        for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
            if c in clv.columns:
                ccol = c
                break
        
        if ccol:
            clv_values = pd.to_numeric(clv[ccol], errors="coerce").dropna()
            
            render_kpi_row([
                {"label": "Customers scored", "value": len(clv_values), "formatter": "count"},
                {"label": "Median CLV Proxy", "value": clv_values.median(), "formatter": "currency"},
                {"label": "Mean CLV Proxy", "value": clv_values.mean(), "formatter": "currency"},
                {"label": "Total CLV Proxy", "value": clv_values.sum(), "formatter": "currency"},
                {"label": "90th percentile", "value": clv_values.quantile(0.90), "formatter": "currency"},
            ])
            
            left, right = st.columns([1.2, 0.8])
            
            with left:
                fig = plot_histogram_with_marginal(
                    clv_values.values,
                    "Predicted Future Value",
                    "CLV Proxy distribution",
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            with right:
                # Uncertainty band
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
                
                if lower_col and upper_col:
                    lower = pd.to_numeric(clv[lower_col], errors="coerce")
                    upper = pd.to_numeric(clv[upper_col], errors="coerce")
                    
                    valid = pd.DataFrame({
                        "clv": clv_values,
                        "lower": lower,
                        "upper": upper,
                    }).dropna().sort_values("clv").reset_index(drop=True)
                    
                    if len(valid) > 1000:
                        valid = valid.iloc[::max(1, len(valid) // 1000)]
                    
                    fig = plot_uncertainty_band(
                        x_vals=np.arange(len(valid)),
                        expected=valid["clv"].values,
                        lower=valid["lower"].values,
                        upper=valid["upper"].values,
                        title="Expected value with uncertainty (p10–p90)",
                    )
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    render_missing("No uncertainty bounds available.")
        else:
            render_missing("CLV column not recognized.")
    else:
        render_missing("CLV data not available.")
    
    # =============================================================================
    # SECTION 2: VALUE CONCENTRATION
    # =============================================================================
    render_section_label("Value concentration")
    
    if clv is not None and ccol:
        clv_values = pd.to_numeric(clv[ccol], errors="coerce").dropna()
        
        fig = plot_concentration_curve(
            clv_values.values,
            title="Where is customer value concentrated?",
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Top decile/quintile callouts
        total = clv_values.sum()
        sorted_vals = clv_values.sort_values(ascending=False)
        
        top_10 = sorted_vals.head(int(len(sorted_vals) * 0.1)).sum() / total * 100
        top_20 = sorted_vals.head(int(len(sorted_vals) * 0.2)).sum() / total * 100
        top_40 = sorted_vals.head(int(len(sorted_vals) * 0.4)).sum() / total * 100
        
        render_insight(
            label="VALUE CONCENTRATION",
            headline=f"Top 20% of customers account for {top_20:.0f}% of predicted future value",
            detail=f"Top 10% → {top_10:.0f}%  |  Top 40% → {top_40:.0f}%",
            evidence=f"CLV proxy distribution across {len(clv_values):,} customers",
            severity="high",
        )
    
    # =============================================================================
    # SECTION 3: VALUE-AT-RISK MATRIX
    # =============================================================================
    render_section_label("Value at risk — Value vs Inactivity Risk")
    
    if combined is not None:
        # Find columns
        clv_col = None
        for c in ["clv_mean", "clv", "predicted_clv", "customer_clv"]:
            if c in combined.columns:
                clv_col = c
                break
        
        churn_col = None
        for c in ["churn_probability", "churn_prob", "prob_churn"]:
            if c in combined.columns:
                churn_col = c
                break
        
        seg_col = "segment_name" if "segment_name" in combined.columns else ("segment" if "segment" in combined.columns else None)
        lifecycle_col = "lifecycle_state" if "lifecycle_state" in combined.columns else None
        
        if clv_col and churn_col:
            plot_df = combined.dropna(subset=[clv_col, churn_col]).copy()
            plot_df[clv_col] = pd.to_numeric(plot_df[clv_col], errors="coerce")
            plot_df[churn_col] = pd.to_numeric(plot_df[churn_col], errors="coerce")
            plot_df = plot_df.dropna(subset=[clv_col, churn_col])
            
            # Sample for performance
            if len(plot_df) > 8000:
                plot_df = plot_df.sample(8000, random_state=42)
            
            # Determine color column
            color_col = seg_col if seg_col else (lifecycle_col if lifecycle_col else None)
            
            if color_col:
                fig = plot_scatter_with_quadrants(
                    plot_df,
                    x_col=clv_col,
                    y_col=churn_col,
                    color_col=color_col,
                    x_threshold=plot_df[clv_col].median(),
                    y_threshold=plot_df[churn_col].median(),
                    title="Customer Value vs Inactivity Risk (quadrants = median splits)",
                    labels={clv_col: "Predicted Future Value", churn_col: "Inactivity Risk"},
                )
            else:
                fig = plot_scatter(
                    plot_df,
                    x_col=clv_col,
                    y_col=churn_col,
                    title="Customer Value vs Inactivity Risk",
                    labels={clv_col: "Predicted Future Value", churn_col: "Inactivity Risk"},
                    opacity=0.4,
                )
                # Add median lines manually
                fig.add_vline(x=plot_df[clv_col].median(), line_dash="dot", line_color="rgba(100,100,100,0.5)")
                fig.add_hline(y=plot_df[churn_col].median(), line_dash="dot", line_color="rgba(100,100,100,0.5)")
            
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Quadrant interpretation
            x_med = plot_df[clv_col].median()
            y_med = plot_df[churn_col].median()
            
            hv_hr = plot_df[(plot_df[clv_col] >= x_med) & (plot_df[churn_col] >= y_med)]
            hv_lr = plot_df[(plot_df[clv_col] >= x_med) & (plot_df[churn_col] < y_med)]
            lv_hr = plot_df[(plot_df[clv_col] < x_med) & (plot_df[churn_col] >= y_med)]
            lv_lr = plot_df[(plot_df[clv_col] < x_med) & (plot_df[churn_col] < y_med)]
            
            quad_cols = st.columns(4)
            quad_data = [
                ("Value Protection", hv_hr, "High value + High risk → Priority retention"),
                ("Healthy Value Base", hv_lr, "High value + Low risk → Nurture & grow"),
                ("Lower Priority Retention", lv_hr, "Low value + High risk → Monitor"),
                ("Nurture / Grow", lv_lr, "Low value + Low risk → Develop"),
            ]
            
            for (label, qdf, desc), col in zip(quad_data, quad_cols):
                with col:
                    n = len(qdf)
                    val = qdf[clv_col].sum()
                    st.metric(label, f"{n:,}", f"£{val:,.0f}")
                    st.caption(desc)
        else:
            render_missing("CLV and churn columns required for value-at-risk matrix.")
    else:
        render_missing("Combined customer data required.")
    
    # =============================================================================
    # SECTION 4: RISK BY VALUE DECILE
    # =============================================================================
    render_section_label("Risk by value decile")
    
    if combined is not None and clv_col and churn_col:
        plot_df = combined.dropna(subset=[clv_col, churn_col]).copy()
        plot_df[clv_col] = pd.to_numeric(plot_df[clv_col], errors="coerce")
        plot_df[churn_col] = pd.to_numeric(plot_df[churn_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[clv_col, churn_col])
        
        # Create deciles
        plot_df["value_decile"] = pd.qcut(plot_df[clv_col], 10, labels=[f"D{i+1}" for i in range(10)], duplicates="drop")
        
        decile_stats = plot_df.groupby("value_decile", observed=True).agg(
            customers=(clv_col, "size"),
            median_clv=(clv_col, "median"),
            mean_clv=(clv_col, "mean"),
            median_risk=(churn_col, "median"),
            mean_risk=(churn_col, "mean"),
            high_risk_share=(churn_col, lambda x: (x >= 0.7).mean()),
        ).reset_index()
        
        # Bar chart
        from ..ui.charts import plot_horizontal_bar
        fig = plot_horizontal_bar(
            decile_stats,
            x_col="mean_risk",
            y_col="value_decile",
            title="Mean inactivity risk by value decile",
            labels={"mean_risk": "Mean inactivity risk", "value_decile": "Value decile (D1=lowest, D10=highest)"},
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Table
        display = decile_stats.copy()
        display["mean_risk"] = display["mean_risk"].apply(format_probability)
        display["median_risk"] = display["median_risk"].apply(format_probability)
        display["mean_clv"] = display["mean_clv"].apply(format_currency)
        display["high_risk_share"] = display["high_risk_share"].apply(lambda x: f"{x:.0%}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        
        # Insight
        # Check if risk increases with value
        corr = plot_df[clv_col].corr(plot_df[churn_col])
        if corr > 0.1:
            render_insight(
                label="RISK-VALUE CORRELATION",
                headline=f"Inactivity risk increases with predicted value (r={corr:.2f})",
                detail="Higher-value customers show elevated inactivity risk — value protection is critical",
                evidence=f"Pearson correlation between CLV proxy and churn probability across {len(plot_df):,} customers",
                severity="high",
            )
        elif corr < -0.1:
            render_insight(
                label="RISK-VALUE CORRELATION",
                headline=f"Inactivity risk decreases with predicted value (r={corr:.2f})",
                detail="Higher-value customers are more stable — retention focus on mid-value segments",
                evidence=f"Pearson correlation between CLV proxy and churn probability across {len(plot_df):,} customers",
                severity="medium",
            )
    else:
        render_missing("Combined CLV and churn data required.")
    
    # =============================================================================
    # SECTION 5: SURVIVAL
    # =============================================================================
    render_section_label("Survival — Probability of remaining commercially active")
    
    if churn is not None:
        # Survival columns
        survival_cols = []
        for c in ["survival_3m", "survival_6m", "survival_12m"]:
            if c in churn.columns:
                survival_cols.append(c)
        
        if survival_cols:
            survival_vals = {}
            for col in survival_cols:
                vals = pd.to_numeric(churn[col], errors="coerce").dropna()
                survival_vals[col] = vals
            
            if survival_vals:
                # Summary metrics
                surv_cols = st.columns(len(survival_cols))
                for (col, vals), col_obj in zip(survival_vals.items(), surv_cols):
                    with col_obj:
                        st.metric(
                            col.replace("survival_", "Survival ").replace("m", "m"),
                            format_probability(vals.median()),
                        )
                
                # Survival curves by segment if available
                seg_col = "segment_name" if "segment_name" in churn.columns else ("segment" if "segment" in churn.columns else None)
                
                if seg_col:
                    st.markdown("#### Survival by behavioral segment")
                    
                    # Average survival per segment
                    seg_survival = churn.groupby(seg_col)[survival_cols].apply(
                        lambda x: x.apply(pd.to_numeric, errors="coerce").mean()
                    ).reset_index()
                    
                    from ..ui.charts import plot_line
                    
                    # Melt for line chart
                    melt_df = seg_survival.melt(
                        id_vars=seg_col,
                        value_vars=survival_cols,
                        var_name="Horizon",
                        value_name="Survival Probability",
                    )
                    melt_df["Horizon"] = melt_df["Horizon"].str.replace("survival_", "").str.replace("m", "m")
                    
                    fig = plot_line(
                        melt_df,
                        x_col="Horizon",
                        y_col="Survival Probability",
                        color_col=seg_col,
                        title="Survival probability by segment",
                        y_format=".0%",
                    )
                    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            render_missing("Survival probability columns not found in churn output.")
    else:
        render_missing("Churn/survival data not available.")
    
    # =============================================================================
    # SECTION 6: MODEL QUALITY (collapsible)
    # =============================================================================
    with st.expander("Model quality & calibration"):
        st.markdown("**Churn / Inactivity Risk Model**")
        
        registry = get_registry()
        churn_card = registry.load_model_card("churn")
        
        if churn_card and "metrics" in churn_card:
            metrics = churn_card["metrics"]
            metric_cols = st.columns(4)
            for (name, val), col in zip(metrics.items(), metric_cols):
                with col:
                    if isinstance(val, float):
                        st.metric(name.replace("_", " ").title(), f"{val:.3f}")
                    else:
                        st.metric(name.replace("_", " ").title(), str(val))
        else:
            st.caption("Model card not available or missing metrics.")
        
        st.markdown("**Next Purchase Model**")
        next_card = registry.load_model_card("churn")  # Same file may have both
        
        if next_card and "next_purchase_metrics" in next_card:
            metrics = next_card["next_purchase_metrics"]
            metric_cols = st.columns(4)
            for (name, val), col in zip(metrics.items(), metric_cols):
                with col:
                    if isinstance(val, float):
                        st.metric(name.replace("_", " ").title(), f"{val:.3f}")
                    else:
                        st.metric(name.replace("_", " ").title(), str(val))
        else:
            st.caption("Next purchase model metrics not available in model card.")
        
        render_science_card(
            "Validation principle",
            "Predictive retention and purchase models are evaluated with "
            "time-based backtesting. A random train/test split can leak future "
            "customer behavior into the training population and make model "
            "performance look artificially strong."
        )