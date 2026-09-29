"""
Tests for UI Contracts - Theme, Charts, and Data Contracts
"""
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit_app.ui.tokens import (
    LIGHT_TOKENS,
    DARK_TOKENS,
    STATUS_COLORS,
    ACTION_COLORS,
    SEGMENT_PALETTE,
    PLOTLY_CONFIG,
    ColorTokens,
    generate_css_variables,
    get_theme_tokens,
)
from streamlit_app.ui.theme import (
    get_current_theme,
    get_color_tokens,
    apply_plotly_theme,
    get_plotly_theme,
    get_status_color,
    get_action_color,
    get_segment_color,
)
from streamlit_app.ui.charts import (
    apply_plotly_theme as charts_apply_plotly_theme,
    plot_histogram,
    plot_scatter,
    plot_scatter_with_quadrants,
    plot_action_allocation,
)
from streamlit_app.ui.components import (
    render_formatted_dataframe,
)
from streamlit_app.app_data import ArtifactRegistry, EXPECTED_ARTIFACTS


class TestThemeTokens:
    """Tests for design tokens completeness."""

    def test_light_tokens_complete(self):
        assert isinstance(LIGHT_TOKENS, ColorTokens)
        assert LIGHT_TOKENS.background is not None
        assert LIGHT_TOKENS.surface is not None
        assert LIGHT_TOKENS.text_primary is not None
        assert LIGHT_TOKENS.primary is not None

    def test_dark_tokens_complete(self):
        assert isinstance(DARK_TOKENS, ColorTokens)
        assert DARK_TOKENS.background is not None
        assert DARK_TOKENS.surface is not None
        assert DARK_TOKENS.text_primary is not None
        assert DARK_TOKENS.primary is not None

    def test_light_and_dark_differ(self):
        assert LIGHT_TOKENS.background != DARK_TOKENS.background
        assert LIGHT_TOKENS.text_primary != DARK_TOKENS.text_primary
        assert LIGHT_TOKENS.primary != DARK_TOKENS.primary

    def test_css_variable_generation(self):
        css = generate_css_variables(LIGHT_TOKENS)
        assert "--background" in css
        assert "--primary" in css
        assert "--success" in css

    def test_get_theme_tokens(self):
        light = get_theme_tokens("light")
        dark = get_theme_tokens("dark")
        assert light.background == LIGHT_TOKENS.background
        assert dark.background == DARK_TOKENS.background


class TestThemeDetection:
    """Tests for theme detection and application."""

    @pytest.fixture(autouse=True)
    def reset_theme_cache(self):
        """Reset theme cache before each test."""
        import streamlit_app.ui.theme as theme_module
        theme_module._THEME_CACHE = None
        yield
        theme_module._THEME_CACHE = None

    def test_get_current_theme_light(self):
        with patch("streamlit_app.ui.theme.st.get_option", return_value="light"):
            assert get_current_theme() == "light"

    def test_get_current_theme_dark(self):
        with patch("streamlit_app.ui.theme.st.get_option", return_value="dark"):
            assert get_current_theme() == "dark"

    def test_get_current_theme_unknown_fallback(self):
        with patch("streamlit_app.ui.theme.st.get_option", return_value="unknown"):
            assert get_current_theme() == "light"


class TestColorAccessors:
    """Tests for color accessor functions."""

    def test_get_status_color(self):
        for status in STATUS_COLORS:
            color = get_status_color(status)
            assert color == STATUS_COLORS[status]
        assert get_status_color("unknown") == "#697386"

    def test_get_action_color(self):
        for action in ACTION_COLORS:
            color = get_action_color(action)
            assert color == ACTION_COLORS[action]
        assert get_action_color("unknown") == "#64748b"

    def test_get_segment_color(self):
        color = get_segment_color(0)
        assert color == SEGMENT_PALETTE[0]
        color = get_segment_color(15)  # Wraps around
        assert color == SEGMENT_PALETTE[15 % len(SEGMENT_PALETTE)]


class TestPlotlyTheme:
    """Tests for Plotly theme application."""

    def test_apply_plotly_theme_returns_figure(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))
        result = apply_plotly_theme(fig, height=400, title="Test")
        assert result is fig
        assert result.layout.height == 400
        assert result.layout.title.text == "Test"

    def test_apply_plotly_theme_sets_colors(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))
        result = apply_plotly_theme(fig)
        assert result.layout.colorway is not None

    def test_get_plotly_theme(self):
        theme = get_plotly_theme()
        assert "layout" in theme
        assert theme["layout"]["paper_bgcolor"] == "rgba(0,0,0,0)"

    def test_get_plotly_theme_has_font(self):
        theme = get_plotly_theme()
        assert "font" in theme["layout"]
        assert "family" in theme["layout"]["font"]


class TestCharts:
    """Tests for chart functions."""

    def test_plot_histogram(self):
        data = np.random.randn(100)
        fig = plot_histogram(data, "Value", "Test Histogram")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_plot_scatter(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 4, 2]})
        fig = plot_scatter(df, "x", "y")
        assert isinstance(fig, go.Figure)

    def test_plot_scatter_with_quadrants(self):
        df = pd.DataFrame({"x": [0.2, 0.8], "y": [0.2, 0.8], "color": ["A", "B"]})
        fig = plot_scatter_with_quadrants(df, "x", "y", "color", 0.5, 0.5, "Test")
        assert isinstance(fig, go.Figure)
        shapes = [s for s in fig.layout.shapes if s.type == "line"]
        assert len(shapes) >= 2

    def test_plot_action_allocation(self):
        df = pd.DataFrame({
            "action": ["protect_value", "nurture", "monitor"],
            "customers": [100, 200, 50],
        })
        fig = plot_action_allocation(df, "action", "customers")
        assert isinstance(fig, go.Figure)
        trace = fig.data[0]
        assert trace.marker.color in ACTION_COLORS.values()


class TestFormattedDataFrameContracts:
    """Tests for formatted dataframe semantic contracts."""

    def test_probability_column_converts_to_percentage(self):
        from streamlit_app.ui.components import infer_semantic_type
        from streamlit_app.app_formatting import SemanticType

        assert infer_semantic_type("churn_probability") == SemanticType.PROBABILITY

    def test_currency_column_formats(self):
        from streamlit_app.ui.components import infer_semantic_type
        from streamlit_app.app_formatting import SemanticType

        assert infer_semantic_type("revenue") == SemanticType.CURRENCY

    def test_score_column_formats(self):
        from streamlit_app.ui.components import infer_semantic_type
        from streamlit_app.app_formatting import SemanticType

        assert infer_semantic_type("decision_confidence") == SemanticType.SCORE
        assert infer_semantic_type("priority_score") == SemanticType.SCORE

    def test_duration_days_column_formats(self):
        from streamlit_app.ui.components import infer_semantic_type
        from streamlit_app.app_formatting import SemanticType

        assert infer_semantic_type("expected_days_to_next_purchase") == SemanticType.DURATION_DAYS


class TestArtifactRegistryContracts:
    """Tests for ArtifactRegistry contracts."""

    def test_expected_artifacts_defined(self):
        assert len(EXPECTED_ARTIFACTS) == 10
        expected_modules = [
            "data_quality", "customer_360", "segmentation",
            "cohorts", "clv", "churn", "recommendations",
            "decision_engine", "product_analytics", "reactivation"
        ]
        for module in expected_modules:
            assert module in EXPECTED_ARTIFACTS

    def test_artifact_schema_has_required_fields(self):
        for module, spec in EXPECTED_ARTIFACTS.items():
            assert "required_columns" in spec
            assert "min_rows" in spec
            assert isinstance(spec["required_columns"], list)
            assert isinstance(spec["min_rows"], int)
            assert spec["min_rows"] >= 0

    def test_registry_initialization(self):
        registry = ArtifactRegistry()
        assert registry.project_root is not None

    def test_get_all_module_statuses_returns_status_dict(self):
        registry = ArtifactRegistry()
        statuses = registry.get_all_module_statuses()
        assert isinstance(statuses, dict)
        assert len(statuses) > 0
        for module, status in statuses.items():
            assert hasattr(status, "overall_status")
            assert hasattr(status, "freshness_note")
            assert hasattr(status, "run_consistency")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])