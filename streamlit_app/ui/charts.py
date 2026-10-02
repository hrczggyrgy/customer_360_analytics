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
            title_font=tokens["xaxis"]["titlefont"],
        ),
        yaxis2=dict(
            title=line_name,
            overlaying="y",
            side="right",
            showgrid=False,
            linecolor=tokens["xaxis"]["linecolor"],
            tickfont=tokens["xaxis"]["tickfont"],
            title_font=tokens["xaxis"]["titlefont"],
        ),
    )
    
    fig.update_xaxes(
        title="Calendar month",
        gridcolor=tokens["xaxis"]["gridcolor"],
        linecolor=tokens["xaxis"]["linecolor"],
        tickfont=tokens["xaxis"]["tickfont"],
        title_font=tokens["xaxis"]["titlefont"],
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
# OPPORTUNITY MATRIX
# =============================================================================

def plot_opportunity_matrix(
    df,
    x_col: str,
    y_col: str,
    size_col: str,
    color_col: str,
    x_label: str = "Next-purchase propensity",
    y_label: str = "Inactivity risk",
    title: str = "Opportunity Matrix: Propensity vs Risk",
    height: int = 560,
    x_threshold: Optional[float] = None,
    y_threshold: Optional[float] = None,
    x_threshold_label: Optional[str] = None,
    y_threshold_label: Optional[str] = None,
    quadrant_labels: Optional[Dict[str, str]] = None,
    max_points: int = 8000,
    log_x: bool = False,
    log_y: bool = False,
    hover_cols: Optional[List[str]] = None,
) -> go.Figure:
    """
    Create an opportunity matrix scatter plot with quadrant interpretation.
    
    Args:
        df: DataFrame with customer/segment data
        x_col: Column for x-axis (propensity)
        y_col: Column for y-axis (risk)
        size_col: Column for bubble size (value)
        color_col: Column for color (action/segment/lifecycle)
        x_label: X-axis label
        y_label: Y-axis label
        title: Chart title
        height: Chart height
        x_threshold: Vertical quadrant threshold (quantile or absolute)
        y_threshold: Horizontal quadrant threshold (quantile or absolute)
        x_threshold_label: Label for x threshold
        y_threshold_label: Label for y threshold
        quadrant_labels: Dict mapping quadrant names to labels
        max_points: Maximum points to plot (sampled if exceeded)
        log_x: Use log scale for x-axis
        log_y: Use log scale for y-axis
        hover_cols: Additional columns to show on hover
    """
    tokens = get_plotly_theme()["layout"]
    
    plot_df = df.copy()
    
    # Sample if too many points
    if len(plot_df) > max_points:
        plot_df = plot_df.sample(max_points, random_state=SEED)
    
    # Compute thresholds if not provided (use medians as defaults)
    if x_threshold is None:
        x_threshold = plot_df[x_col].median()
    if y_threshold is None:
        y_threshold = plot_df[y_col].median()
    
    # Default quadrant labels
    if quadrant_labels is None:
        quadrant_labels = {
            "top_right": "PROTECT<br>High value, high risk",
            "top_left": "NURTURE<br>Lower value, high risk",
            "bottom_right": "GROW<br>High value, low risk",
            "bottom_left": "ACCELERATE<br>Lower value, low risk",
        }
    
    # Build color map for discrete color_col
    if color_col in plot_df.columns:
        color_map = {}
        for val in plot_df[color_col].unique():
            color_map[str(val)] = get_action_color(str(val))
    else:
        color_map = None
    
    # Hover data
    hover_data = {}
    if hover_cols:
        for col in hover_cols:
            if col in plot_df.columns:
                hover_data[col] = True
    
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        opacity=0.55,
        size_max=30,
        labels={x_col: x_label, y_col: y_label, color_col: color_col.replace("_", " ").title()},
        color_discrete_map=color_map,
        hover_data=hover_data,
    )
    
    # Add quadrant reference lines
    fig.add_vline(
        x=x_threshold,
        line_dash="dot",
        line_color="rgba(100,100,100,0.5)",
        annotation_text=x_threshold_label or f"P50: {x_threshold:.2f}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=y_threshold,
        line_dash="dot",
        line_color="rgba(100,100,100,0.5)",
        annotation_text=y_threshold_label or f"P50: {y_threshold:.2f}",
        annotation_position="top left",
    )
    
    x_min, x_max = plot_df[x_col].min(), plot_df[x_col].max()
    y_min, y_max = plot_df[y_col].min(), plot_df[y_col].max()
    
    annotations = [
        dict(
            x=x_max * 0.85 if not log_x else 10 ** (np.log10(x_max) * 0.85),
            y=y_max * 0.85 if not log_y else 10 ** (np.log10(y_max) * 0.85),
            text=quadrant_labels.get("top_right", "PROTECT"),
            showarrow=False,
            font=dict(size=11, color="rgba(100,100,100,0.7)"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(100,100,100,0.3)",
            borderwidth=1,
            borderpad=4,
        ),
        dict(
            x=x_min * 1.1 if not log_x else 10 ** (np.log10(x_min) * 1.1),
            y=y_max * 0.85 if not log_y else 10 ** (np.log10(y_max) * 0.85),
            text=quadrant_labels.get("top_left", "NURTURE"),
            showarrow=False,
            font=dict(size=11, color="rgba(100,100,100,0.7)"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(100,100,100,0.3)",
            borderwidth=1,
            borderpad=4,
        ),
        dict(
            x=x_max * 0.85 if not log_x else 10 ** (np.log10(x_max) * 0.85),
            y=y_min * 1.1 if not log_y else 10 ** (np.log10(y_min) * 1.1),
            text=quadrant_labels.get("bottom_right", "GROW"),
            showarrow=False,
            font=dict(size=11, color="rgba(100,100,100,0.7)"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(100,100,100,0.3)",
            borderwidth=1,
            borderpad=4,
        ),
        dict(
            x=x_min * 1.1 if not log_x else 10 ** (np.log10(x_min) * 1.1),
            y=y_min * 1.1 if not log_y else 10 ** (np.log10(y_min) * 1.1),
            text=quadrant_labels.get("bottom_left", "ACCELERATE"),
            showarrow=False,
            font=dict(size=11, color="rgba(100,100,100,0.7)"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(100,100,100,0.3)",
            borderwidth=1,
            borderpad=4,
        ),
    ]
    
    fig.update_layout(annotations=annotations)
    
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# TRANSITION MATRIX
# =============================================================================

def plot_transition_matrix(
    transition_df: pd.DataFrame,
    from_col: str,
    to_col: str,
    count_col: str,
    title: str = "Segment Transition Matrix",
    height: int = 500,
    normalize: str = "count",  # "count", "row_pct", "col_pct"
    color_scale: str = "Blues",
) -> go.Figure:
    """
    Create a transition matrix heatmap.
    
    Args:
        transition_df: DataFrame with from_state, to_state, count columns
        from_col: Column for from-state
        to_col: Column for to-state
        count_col: Column for count
        title: Chart title
        height: Chart height
        normalize: "count", "row_pct" (outflow share), "col_pct" (inflow share)
        color_scale: Color scale
    """
    tokens = get_plotly_theme()["layout"]
    
    # Pivot to matrix
    matrix = transition_df.pivot_table(
        index=from_col,
        columns=to_col,
        values=count_col,
        aggfunc="sum",
        fill_value=0,
    )
    
    if normalize == "row_pct":
        matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0) * 100
        z_label = "Share of from-state (%)"
        z_format = ".1f"
    elif normalize == "col_pct":
        matrix = matrix.div(matrix.sum(axis=0), axis=1).fillna(0) * 100
        z_label = "Share of to-state (%)"
        z_format = ".1f"
    else:
        z_label = "Customers"
        z_format = ",d"
    
    fig = px.imshow(
        matrix,
        aspect="auto",
        labels={"x": "Current state", "y": "Prior state", "color": z_label},
        color_continuous_scale=color_scale,
    )
    
    # Add text annotations
    for i, from_state in enumerate(matrix.index):
        for j, to_state in enumerate(matrix.columns):
            val = matrix.iloc[i, j]
            if val > 0:
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{val:{z_format}}",
                    showarrow=False,
                    font=dict(
                        size=10,
                        color="white" if val > matrix.values.max() * 0.6 else "black",
                    ),
                )
    
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=TYPOGRAPHY.font_sans, color=tokens.chart_text, size=13),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=17, color=tokens.text_primary)) if title else None,
        xaxis=dict(
            side="top",
            tickfont=dict(color=tokens.text_secondary, size=11),
            title_font=dict(color=tokens.text_secondary, size=12),
        ),
        yaxis=dict(
            tickfont=dict(color=tokens.text_secondary, size=11),
            title_font=dict(color=tokens.text_secondary, size=12),
        ),
        coloraxis_colorbar=dict(
            title=z_label,
            tickfont=dict(color=tokens.text_secondary, size=11),
            title_font=dict(color=tokens.text_secondary, size=12),
        ),
    )
    
    return fig


# =============================================================================
# VALUE CONCENTRATION CURVE (enhanced)
# =============================================================================

def plot_value_concentration_curve(
    values: np.ndarray,
    title: str = "Where is customer value concentrated?",
    height: int = 500,
    annotate_thresholds: Optional[List[float]] = None,
    label_prefix: str = "Top",
) -> go.Figure:
    """
    Plot a concentration/Lorenz curve with threshold annotations.
    
    Args:
        values: Array of values (e.g., CLV per customer)
        title: Chart title
        height: Chart height
        annotate_thresholds: List of percentiles to annotate (e.g., [0.1, 0.2, 0.4])
        label_prefix: Prefix for annotations (e.g., "Top")
    """
    tokens = get_plotly_theme()["layout"]
    
    if annotate_thresholds is None:
        annotate_thresholds = [0.1, 0.2, 0.4]
    
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
    
    # Lorenz curve
    fig.add_trace(go.Scatter(
        x=ranks,
        y=cumshare,
        mode="lines",
        line=dict(color=tokens["colorway"][0], width=3),
        name="Cumulative value share",
        hovertemplate=(
            "Customer percentile: %{x:.0%}<br>"
            "Cumulative share: %{y:.0%}<extra></extra>"
        ),
    ))
    
    # Equality line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color="rgba(100,100,100,0.5)", width=1, dash="dash"),
        name="Perfect equality",
        hoverinfo="skip",
    ))
    
    # Annotate thresholds
    annotations = []
    for threshold in annotate_thresholds:
        # Find closest point
        idx = np.abs(ranks - threshold).argmin()
        x_val = ranks[idx]
        y_val = cumshare[idx]
        
        annotations.append(dict(
            x=x_val,
            y=y_val,
            xref="x",
            yref="y",
            text=f"{label_prefix} {threshold:.0%} → {y_val:.0%} of value",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=tokens["colorway"][0],
            ax=40,
            ay=-40,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=tokens["colorway"][0],
            borderwidth=1,
            font=dict(size=11, color=tokens.text_primary),
        ))
    
    fig.update_layout(annotations=annotations)
    
    fig.update_xaxes(tickformat=".0%", title="Customer percentile")
    fig.update_yaxes(tickformat=".0%", title="Cumulative share of value")
    
    return apply_plotly_theme(fig, height=height, title=title)


# =============================================================================
# PEER BENCHMARK
# =============================================================================

def plot_peer_benchmark(
    customer_values: Dict[str, float],
    peer_distributions: Dict[str, np.ndarray],
    metric_labels: Optional[Dict[str, str]] = None,
    higher_is_better: Optional[Dict[str, bool]] = None,
    title: str = "Customer vs Peer Group",
    height: int = 500,
) -> go.Figure:
    """
    Create a percentile benchmark visualization comparing a customer to peers.
    
    Args:
        customer_values: Dict of metric_name -> customer value
        peer_distributions: Dict of metric_name -> peer array
        metric_labels: Optional dict mapping metric_name to display label
        higher_is_better: Optional dict mapping metric_name to True/False (default True)
        title: Chart title
        height: Chart height
    """
    tokens = get_plotly_theme()["layout"]
    
    if metric_labels is None:
        metric_labels = {}
    if higher_is_better is None:
        higher_is_better = {}
    
    metrics = list(customer_values.keys())
    n_metrics = len(metrics)
    
    # Calculate percentiles
    percentiles = {}
    for metric in metrics:
        peer_vals = peer_distributions.get(metric, np.array([]))
        customer_val = customer_values[metric]
        
        if len(peer_vals) > 0:
            peer_vals = peer_vals[~np.isnan(peer_vals)]
            if len(peer_vals) > 0:
                pct = (peer_vals <= customer_val).mean() * 100
                if not higher_is_better.get(metric, True):
                    pct = 100 - pct
                percentiles[metric] = pct
            else:
                percentiles[metric] = 50
        else:
            percentiles[metric] = 50
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    for i, metric in enumerate(metrics):
        pct = percentiles[metric]
        label = metric_labels.get(metric, metric.replace("_", " ").title())
        
        # Color based on percentile
        if pct >= 80:
            color = tokens.success
        elif pct >= 60:
            color = tokens.primary
        elif pct >= 40:
            color = tokens.warning
        else:
            color = tokens.danger
        
        fig.add_trace(go.Bar(
            y=[label],
            x=[pct],
            orientation="h",
            marker_color=color,
            name=label,
            hovertemplate=(
                f"{label}<br>"
                f"Percentile: %{{x:.0f}}%<br>"
                f"Customer value: {customer_values[metric]:.2f}<br>"
                f"Peer median: {np.nanmedian(peer_distributions.get(metric, [0])):.2f}<extra></extra>"
            ),
            showlegend=False,
        ))
        
        # Add median reference line
        peer_median = np.nanmedian(peer_distributions.get(metric, [0]))
        customer_val = customer_values[metric]
        
    # Add 50th percentile reference
    fig.add_vline(x=50, line_dash="dot", line_color="rgba(100,100,100,0.5)", annotation_text="Peer median")
    
    fig.update_xaxes(title="Percentile vs peer group", range=[0, 100], ticksuffix="th")
    fig.update_yaxes(autorange="reversed")
    
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
            title_font=dict(color=tokens.text_secondary, size=12),
            zerolinecolor=tokens.border_subtle,
        ),
        yaxis=dict(
            gridcolor=tokens.chart_grid,
            linecolor=tokens.chart_axis,
            tickcolor=tokens.chart_axis,
            tickfont=dict(color=tokens.text_secondary, size=11),
            title_font=dict(color=tokens.text_secondary, size=12),
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
    "plot_opportunity_matrix",
    "plot_transition_matrix",
    "plot_value_concentration_curve",
    "plot_peer_benchmark",
    "PLOTLY_CONFIG",
    "apply_plotly_theme",
]