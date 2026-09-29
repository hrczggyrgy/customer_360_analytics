"""
Centralized Plotly Charts for Retail Customer Intelligence Streamlit App.

All charts use the centralized theme system for consistent styling across light/dark modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Dict, Any

from .theme import (
    apply_plotly_theme,
    get_plotly_theme,
    PLOTLY_CONFIG,
    SEGMENT_PALETTE,
    ACTION_COLORS,
    get_segment_color,
    get_action_color,
    get_current_theme,
    SEED,
    get_color_tokens,
    TYPOGRAPHY,
)

# Set random seed for reproducible sampling
np.random.seed(SEED)


# =============================================================================
# BASE CHART FUNCTIONS
# =============================================================================

def plot_missing(message: str = "Required output is not available yet.") -> None:
    """Render a missing data placeholder."""
    import streamlit as st
    st.info(message)


# =============================================================================
# HISTOGRAM
# =============================================================================

def plot_histogram(
    data: np.ndarray,
    x_label: str,
    title: str,
    nbins: int = 40,
    height: int = 480,
    x_format: Optional[str] = None,
) -> go.Figure:
    """Create a consistent histogram."""
    fig = px.histogram(
        x=data,
        nbins=nbins,
        labels={"x": x_label},
    )
    
    if x_format:
        fig.update_xaxes(tickformat=x_format)
    
    return apply_plotly_theme(fig, height=height, title=title)


def plot_histogram_with_marginal(
    data: np.ndarray,
    x_label: str,
    title: str,
    nbins: int = 60,
    height: int = 500,
) -> go.Figure:
    """Create histogram with box marginal."""
    fig = px.histogram(
        x=data,
        nbins=nbins,
        marginal="box",
        labels={"x": x_label},
    )
    fig.update_xaxes(rangemode="tozero")
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# BAR CHARTS
# =============================================================================

def plot_horizontal_bar(
    df,
    x_col: str,
    y_col: str,
    title: str,
    height: int = 470,
    labels: Optional[dict] = None,
    orientation: str = "h",
    color_col: Optional[str] = None,
    color_map: Optional[dict] = None,
) -> go.Figure:
    """Create a consistent horizontal bar chart."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation=orientation,
        labels=labels or {},
        color=color_col,
        color_discrete_map=color_map,
    )
    return apply_plotly_theme(fig, height=height, title=title)


def plot_stacked_bar(
    df,
    x_col: str,
    y_col: str,
    color_col: str,
    title: str,
    height: int = 470,
    labels: Optional[dict] = None,
) -> go.Figure:
    """Create a stacked/grouped bar chart."""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        labels=labels or {},
    )
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# SCATTER PLOTS
# =============================================================================

def plot_scatter(
    df,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    title: str = "",
    height: int = 560,
    labels: Optional[dict] = None,
    opacity: float = 0.45,
    max_points: int = 12000,
    x_format: Optional[str] = None,
    y_format: Optional[str] = None,
    log_x: bool = False,
    color_map: Optional[dict] = None,
) -> go.Figure:
    """Create a consistent scatter plot with optional sampling."""
    plot_df = df
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=SEED)
    
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        opacity=opacity,
        labels=labels or {},
        color_discrete_map=color_map,
    )
    
    if log_x:
        fig.update_xaxes(type="log")
    
    if x_format:
        fig.update_xaxes(tickformat=x_format)
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    
    return apply_plotly_theme(fig, height=height, title=title)


def plot_scatter_with_quadrants(
    df,
    x_col: str,
    y_col: str,
    color_col: str,
    x_threshold: float,
    y_threshold: float,
    title: str,
    height: int = 560,
    labels: Optional[dict] = None,
    color_map: Optional[dict] = None,
) -> go.Figure:
    """Create scatter plot with quadrant reference lines."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        opacity=0.40,
        labels=labels or {},
        color_discrete_map=color_map,
    )
    
    fig.add_vline(x=x_threshold, line_dash="dot", line_color="rgba(100,100,100,0.5)")
    fig.add_hline(y=y_threshold, line_dash="dot", line_color="rgba(100,100,100,0.5)")
    
    if labels:
        if x_col in labels:
            fig.update_xaxes(title=labels[x_col])
        if y_col in labels:
            fig.update_yaxes(title=labels[y_col])
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# HEATMAPS
# =============================================================================

def plot_heatmap(
    data,
    title: str,
    height: int = 500,
    labels: Optional[dict] = None,
    color_scale: str = "Blues",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> go.Figure:
    """Create a consistent heatmap."""
    fig = px.imshow(
        data,
        aspect="auto",
        labels=labels or {},
        color_continuous_scale=color_scale,
        zmin=zmin,
        zmax=zmax,
    )
    return apply_plotly_theme(fig, height=height, title=title)


def plot_segment_heatmap(
    df,
    segment_col: str,
    metric_cols: List[str],
    title: str,
    height: int = 560,
) -> go.Figure:
    """Create a segment profile heatmap."""
    heat = df.set_index(segment_col)[metric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    heat = heat.replace([np.inf, -np.inf], np.nan)
    
    fig = px.imshow(
        heat,
        aspect="auto",
        labels={"x": "Behavioral feature", "y": "Segment", "color": "Score"},
        color_continuous_scale="RdBu",
    )
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# LINE CHARTS
# =============================================================================

def plot_line(
    df,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    height: int = 500,
    labels: Optional[dict] = None,
    y_format: Optional[str] = None,
) -> go.Figure:
    """Create a consistent line chart."""
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        labels=labels or {},
    )
    
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    
    return apply_plotly_theme(fig, height=height, title=title)


def plot_dual_axis_line_bar(
    df,
    x_col: str,
    bar_col: str,
    line_col: str,
    title: str,
    height: int = 460,
    bar_name: str = "Revenue",
    line_name: str = "Orders",
    bar_format: str = "£,.2f",
    line_format: str = ",.0f",
) -> go.Figure:
    """Create a dual-axis chart with bars and line."""
    tokens = get_plotly_theme()["layout"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df[x_col],
        y=df[bar_col],
        name=bar_name,
        hovertemplate=f"%{{x|%Y-%m}}<br>{bar_name}: %{{y:{bar_format}}}<extra></extra>",
        marker_color=tokens["colorway"][0],
    ))
    
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[line_col],
        mode="lines+markers",
        name=line_name,
        yaxis="y2",
        line=dict(color=tokens["colorway"][1], width=3),
        marker=dict(color=tokens["colorway"][1], size=8),
    ))
    
    fig.update_layout(
        yaxis=dict(
            title=bar_name,
            gridcolor=tokens["xaxis"]["gridcolor"],
            linecolor=tokens["xaxis"]["linecolor"],
            tickfont=tokens["xaxis"]["tickfont"],
            titlefont=tokens["xaxis"]["titlefont"],
        ),
        yaxis2=dict(
            title=line_name,
            overlaying="y",
            side="right",
            showgrid=False,
            linecolor=tokens["xaxis"]["linecolor"],
            tickfont=tokens["xaxis"]["tickfont"],
            titlefont=tokens["xaxis"]["titlefont"],
        ),
    )
    
    fig.update_xaxes(
        title="Calendar month",
        gridcolor=tokens["xaxis"]["gridcolor"],
        linecolor=tokens["xaxis"]["linecolor"],
        tickfont=tokens["xaxis"]["tickfont"],
        titlefont=tokens["xaxis"]["titlefont"],
    )
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# RETENTION MATRIX
# =============================================================================

def plot_retention_matrix(
    matrix,
    title: str,
    height: int = 500,
    zmin: float = 0,
    zmax: float = 1,
    color_scale: str = "Blues",
    max_cols: int = 36,
) -> go.Figure:
    """Plot a cohort retention matrix heatmap."""
    heat = matrix.iloc[:, :max_cols]
    
    fig = px.imshow(
        heat,
        aspect="auto",
        zmin=zmin,
        zmax=zmax,
        labels={
            "x": "Months since acquisition",
            "y": "Acquisition cohort",
            "color": "Retention",
        },
        color_continuous_scale=color_scale,
    )
    
    # Format colorbar as percentage
    fig.update_coloraxes(colorbar=dict(tickformat=".0%"))
    
    return apply_plotly_theme(fig, height=height, title=title)


def plot_decay_curve(
    decay_df,
    age_col: str,
    logo_col: Optional[str] = None,
    nrr_col: Optional[str] = None,
    title: str = "Maturity-aware retention decay",
    height: int = 500,
) -> go.Figure:
    """Plot logo retention and NRR decay curves."""
    tokens = get_plotly_theme()["layout"]
    
    fig = go.Figure()
    
    if logo_col and logo_col in decay_df.columns:
        fig.add_trace(go.Scatter(
            x=decay_df[age_col],
            y=decay_df[logo_col],
            mode="lines+markers",
            name="Logo retention",
            line=dict(color=tokens["colorway"][0], width=3),
            marker=dict(size=8),
        ))
    
    if nrr_col and nrr_col in decay_df.columns:
        fig.add_trace(go.Scatter(
            x=decay_df[age_col],
            y=decay_df[nrr_col],
            mode="lines+markers",
            name="Net revenue retention",
            line=dict(color=tokens["colorway"][1], width=3),
            marker=dict(size=8),
        ))
    
    fig.update_yaxes(tickformat=".0%", title="Retention")
    fig.update_xaxes(title="Months since acquisition")
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# UNCERTAINTY BANDS
# =============================================================================

def plot_uncertainty_band(
    x_vals: np.ndarray,
    expected: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str,
    height: int = 500,
    x_title: str = "Customers ordered by value",
    y_title: str = "Value",
    max_points: int = 1000,
) -> go.Figure:
    """Plot expected values with uncertainty ribbon."""
    tokens = get_plotly_theme()["layout"]
    
    # Sample if too many points
    if len(x_vals) > max_points:
        idx = np.linspace(0, len(x_vals) - 1, max_points, dtype=int)
        x_vals = x_vals[idx]
        expected = expected[idx]
        lower = lower[idx]
        upper = upper[idx]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=expected,
        mode="lines",
        name="Expected value",
        line=dict(color=tokens["colorway"][0], width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_vals, x_vals[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        line=dict(width=0),
        name="Uncertainty range (p10-p90)",
        opacity=0.25,
        fillcolor=tokens["colorway"][0],
        hoverinfo="skip",
    ))
    
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# PCA SCATTER
# =============================================================================

def plot_pca_scatter(
    pca_df,
    pc1_col: str,
    pc2_col: str,
    color_col: Optional[str] = None,
    title: str = "Customer behavioral space after PCA",
    height: int = 560,
    max_points: int = 12000,
) -> go.Figure:
    """Create a PCA scatter plot."""
    plot_df = pca_df
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=SEED)
    
    fig = px.scatter(
        plot_df,
        x=pc1_col,
        y=pc2_col,
        color=color_col,
        opacity=0.55,
        labels={pc1_col: "PC1", pc2_col: "PC2"},
        hover_data=[c for c in ["Customer ID", color_col, "segment_confidence"] 
                    if c and c in plot_df.columns],
        color_discrete_sequence=SEGMENT_PALETTE,
    )
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# CONCENTRATION CURVE
# =============================================================================

def plot_concentration_curve(
    values: np.ndarray,
    title: str = "How concentrated is customer value?",
    height: int = 500,
) -> go.Figure:
    """Plot a concentration/Lorenz curve."""
    tokens = get_plotly_theme()["layout"]
    
    values = np.array(values)
    values = values[values > 0]
    values = np.sort(values)[::-1]
    
    cumsum = np.cumsum(values)
    total = cumsum[-1] if len(cumsum) > 0 else 1
    cumshare = cumsum / total
    
    ranks = np.arange(1, len(values) + 1) / len(values)
    
    # Sample for performance
    if len(values) > 1000:
        idx = np.linspace(0, len(values) - 1, 1000, dtype=int)
        ranks = ranks[idx]
        cumshare = cumshare[idx]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ranks,
        y=cumshare,
        mode="lines",
        line=dict(color=tokens["colorway"][0], width=2),
        name="Cumulative value share",
        hovertemplate=(
            "Customer percentile: %{x:.0%}<br>"
            "Cumulative share: %{y:.0%}<extra></extra>"
        ),
    ))
    
    fig.add_hline(y=0.50, line_dash="dot", line_color="rgba(100,100,100,0.5)", annotation_text="50% of value")
    fig.update_xaxes(tickformat=".0%", title="Customer percentile")
    fig.update_yaxes(tickformat=".0%", title="Cumulative share")
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# ACTION ALLOCATION BAR
# =============================================================================

def plot_action_allocation(
    action_counts: pd.DataFrame,
    action_col: str,
    count_col: str,
    title: str = "Customer allocation",
    height: int = 470,
) -> go.Figure:
    """Plot action allocation horizontal bar chart."""
    action_counts = action_counts.sort_values(count_col, ascending=True)
    
    # Build color map
    color_map = {}
    for action in action_counts[action_col].unique():
        color_map[action] = get_action_color(str(action))
    
    fig = px.bar(
        action_counts,
        x=count_col,
        y=action_col,
        orientation="h",
        labels={count_col: "Customers", action_col: "Action"},
        color=action_col,
        color_discrete_map=color_map,
    )
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# CLV BY SEGMENT
# =============================================================================

def plot_clv_by_segment(
    segment_profile: pd.DataFrame,
    segment_col: str,
    clv_col: str,
    count_col: str = "customers",
    median_col: str = "median_clv",
    title: str = "Where is customer value concentrated?",
    height: int = 470,
) -> go.Figure:
    """Plot CLV by segment as horizontal bar chart."""
    tokens = get_plotly_theme()["layout"]
    
    fig = px.bar(
        segment_profile,
        x=clv_col,
        y=segment_col,
        orientation="h",
        labels={clv_col: "Total CLV", segment_col: "Segment"},
        hover_data=[count_col, median_col],
        color=segment_col,
        color_discrete_sequence=SEGMENT_PALETTE,
    )
    
    fig.update_xaxes(tickformat="£,.0f")
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# DECISION SCATTER
# =============================================================================

def plot_decision_scatter(
    df,
    x_col: str,
    y_col: str,
    color_col: str,
    size_col: str,
    title: str,
    height: int = 470,
    x_log: bool = False,
    y_format: Optional[str] = None,
) -> go.Figure:
    """Create decision engine scatter plot with action coloring."""
    color_map = {}
    for action in df[color_col].unique():
        color_map[action] = get_action_color(str(action))
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        size_max=18,
        opacity=0.45,
        labels={x_col: x_col, y_col: y_col, color_col: "Action"},
        color_discrete_map=color_map,
    )
    
    if x_log:
        fig.update_xaxes(type="log", title=f"{x_col} (log scale)")
    else:
        fig.update_xaxes(title=x_col)
    
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# PRIORITY DISTRIBUTION
# =============================================================================

def plot_priority_distribution(
    df,
    priority_col: str,
    action_col: str,
    title: str = "Priority Score Distribution by Action",
    height: int = 470,
) -> go.Figure:
    """Plot priority score distribution by action."""
    tokens = get_plotly_theme()["layout"]
    
    fig = go.Figure()
    
    for action in df[action_col].unique():
        subset = df[df[action_col] == action]
        fig.add_trace(go.Histogram(
            x=subset[priority_col],
            name=str(action),
            opacity=0.5,
            nbinsx=20,
            marker_color=get_action_color(str(action)),
        ))
    
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Priority Score")
    fig.update_yaxes(title="Density")
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# EXPECTED VALUE BY ACTION
# =============================================================================

def plot_expected_value_by_action(
    df,
    action_col: str,
    value_col: str,
    title: str = "Average Expected Value by Recommended Action",
    height: int = 470,
) -> go.Figure:
    """Plot expected value by action."""
    ev_by_action = df.groupby(action_col)[value_col].mean().sort_values()
    
    colors = [get_action_color(a) for a in ev_by_action.index]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ev_by_action.index,
        x=ev_by_action.values,
        orientation="h",
        marker_color=colors,
    ))
    
    fig.update_xaxes(title="Expected Value Proxy", tickformat="£,.0f")
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# CLV VS CHURN BY ACTION
# =============================================================================

def plot_clv_vs_churn_by_action(
    df,
    clv_col: str,
    churn_col: str,
    action_col: str,
    title: str = "Customer Value vs Churn Risk by Recommended Action",
    height: int = 470,
) -> go.Figure:
    """Plot CLV vs Churn colored by action."""
    tokens = get_plotly_theme()["layout"]
    
    fig = go.Figure()
    
    for action in df[action_col].unique():
        subset = df[df[action_col] == action]
        fig.add_trace(go.Scatter(
            x=subset[clv_col],
            y=subset[churn_col],
            mode="markers",
            marker=dict(
                size=20,
                color=get_action_color(str(action)),
                opacity=0.5,
                line=dict(width=1, color="rgba(255,255,255,0.5)"),
            ),
            name=str(action),
        ))
    
    fig.update_xaxes(type="log", title="CLV (log scale)", tickformat="£,.0f")
    fig.update_yaxes(tickformat=".0%", title="Churn Probability")
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# BASE LAYOUT
# =============================================================================

def base_layout(
    fig: go.Figure,
    height: int = 420,
    title: Optional[str] = None,
) -> go.Figure:
    """Apply consistent base layout to a Plotly figure."""
    tokens = get_color_tokens()
    
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family=TYPOGRAPHY.font_sans,
            color=tokens.chart_text,
        ),
        title=(
            dict(
                text=title,
                x=0,
                xanchor="left",
                font=dict(size=17, color=tokens.text_primary),
            )
            if title
            else None
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12, color=tokens.text_secondary),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            gridcolor=tokens.chart_grid,
            linecolor=tokens.chart_axis,
            tickcolor=tokens.chart_axis,
            tickfont=dict(color=tokens.text_secondary, size=11),
            titlefont=dict(color=tokens.text_secondary, size=12),
            zerolinecolor=tokens.border_subtle,
        ),
        yaxis=dict(
            gridcolor=tokens.chart_grid,
            linecolor=tokens.chart_axis,
            tickcolor=tokens.chart_axis,
            tickfont=dict(color=tokens.text_secondary, size=11),
            titlefont=dict(color=tokens.text_secondary, size=12),
            zerolinecolor=tokens.border_subtle,
        ),
        colorway=SEGMENT_PALETTE,
        hoverlabel=dict(
            bgcolor=tokens.surface,
            bordercolor=tokens.border,
            font=dict(color=tokens.text_primary, size=12, family=TYPOGRAPHY.font_sans),
        ),
    )
    return fig


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "base_layout",
    "plot_missing",
    "plot_histogram",
    "plot_histogram_with_marginal",
    "plot_horizontal_bar",
    "plot_stacked_bar",
    "plot_scatter",
    "plot_scatter_with_quadrants",
    "plot_heatmap",
    "plot_segment_heatmap",
    "plot_line",
    "plot_dual_axis_line_bar",
    "plot_retention_matrix",
    "plot_decay_curve",
    "plot_uncertainty_band",
    "plot_pca_scatter",
    "plot_concentration_curve",
    "plot_action_allocation",
    "plot_clv_by_segment",
    "plot_decision_scatter",
    "plot_priority_distribution",
    "plot_expected_value_by_action",
    "plot_clv_vs_churn_by_action",
    "PLOTLY_CONFIG",
    "apply_plotly_theme",
]