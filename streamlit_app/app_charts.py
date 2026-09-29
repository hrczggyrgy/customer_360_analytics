"""
Compatibility shim for legacy app_charts imports.

This module re-exports from the new centralized ui package.
All new code should import from streamlit_app.ui.charts directly.
"""

from __future__ import annotations

from .ui.charts import (
    plot_missing,
    plot_histogram,
    plot_histogram_with_marginal,
    plot_horizontal_bar,
    plot_stacked_bar,
    plot_scatter,
    plot_scatter_with_quadrants,
    plot_heatmap,
    plot_segment_heatmap,
    plot_line,
    plot_dual_axis_line_bar,
    plot_retention_matrix,
    plot_decay_curve,
    plot_uncertainty_band,
    plot_pca_scatter,
    plot_concentration_curve,
    plot_action_allocation,
    plot_clv_by_segment,
    plot_decision_scatter,
    plot_priority_distribution,
    plot_expected_value_by_action,
    plot_clv_vs_churn_by_action,
    PLOTLY_CONFIG,
    ACTION_COLORS,
)

__all__ = [
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
    "ACTION_COLORS",
]