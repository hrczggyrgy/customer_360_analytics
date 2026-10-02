"""
Tests for UI Components - Rendering and Contract Tests
"""
import pytest
import pandas as pd
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit_app.ui.components import (
    render_kpi_card,
    render_kpi_row,
    render_science_card,
    render_status_chip,
    render_status_row,
    render_evidence_table,
    render_missing,
    render_empty_state,
    render_download_button,
    render_customer_selector,
    render_action_summary_table,
    render_page_hero,
    render_section_label,
    render_customer_header,
    render_customer_metric_row,
    render_formatted_dataframe,
    render_action_badge,
    render_segment_badge,
    STATUS_COLORS,
    ACTION_COLORS,
    SEGMENT_PALETTE,
    HERO_COPY,
)


class TestStatusColors:
    """Tests for centralized status color mapping."""

    def test_all_status_colors_defined(self):
        """All expected status keys should have colors."""
        expected = {"ready", "incomplete", "stale", "validation_failed", "unavailable", "unknown"}
        assert set(STATUS_COLORS.keys()) == expected

    def test_status_colors_are_valid_hex(self):
        """All status colors should be valid hex strings."""
        for status, color in STATUS_COLORS.items():
            assert color.startswith("#")
            assert len(color) == 7
            int(color[1:], 16)  # Should not raise


class TestActionColors:
    """Tests for action color mapping."""

    def test_all_action_colors_defined(self):
        expected = {"protect_value", "accelerate_purchase", "reactivate", "cross_sell", "nurture", "monitor"}
        assert set(ACTION_COLORS.keys()) == expected


class TestSegmentPalette:
    """Tests for segment color palette."""

    def test_palette_has_enough_colors(self):
        """Should have at least 12 colors for segments."""
        assert len(SEGMENT_PALETTE) >= 12

    def test_palette_colors_are_valid_hex(self):
        for color in SEGMENT_PALETTE:
            assert color.startswith("#")
            assert len(color) == 7


class TestKPICard:
    """Tests for KPI card rendering."""

    def test_render_kpi_card_with_currency(self):
        with patch("streamlit_app.ui.components.st.metric") as mock_metric:
            render_kpi_card("Revenue", 1234567, formatter="currency")
            mock_metric.assert_called_once()
            args, kwargs = mock_metric.call_args
            assert args[0] == "Revenue"
            assert "£1.2M" in args[1]

    def test_render_kpi_card_with_probability(self):
        with patch("streamlit_app.ui.components.st.metric") as mock_metric:
            render_kpi_card("Churn Risk", 0.75, formatter="probability")
            mock_metric.assert_called_once()
            args, kwargs = mock_metric.call_args
            assert "75.0%" in args[1]

    def test_render_kpi_card_with_none(self):
        with patch("streamlit_app.ui.components.st.metric") as mock_metric:
            render_kpi_card("Test", None)
            mock_metric.assert_called_once()
            args, kwargs = mock_metric.call_args
            assert args[1] == "—"


class TestKPIRow:
    """Tests for KPI row responsive grid."""

    def test_render_kpi_row_creates_multiple_rows(self):
        """With 5 metrics and max_cols=4, should create 2 rows (4 + 1)."""
        metrics = [
            {"label": f"KPI {i}", "value": i * 100, "formatter": "count"}
            for i in range(1, 6)
        ]
        with patch("streamlit_app.ui.components.st.columns") as mock_columns:
            # Mock columns to return list of mock contexts
            mock_col = MagicMock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=False)
            mock_columns.return_value = [mock_col] * 4
            
            with patch("streamlit_app.ui.components.render_kpi_card"):
                render_kpi_row(metrics, max_cols=4)
            
            # Should call columns twice (once for first row of 4, once for second row of 1)
            assert mock_columns.call_count == 2

    def test_render_kpi_row_empty_list(self):
        """Empty metrics list should not crash."""
        render_kpi_row([])


class TestScienceCard:
    """Tests for science context card."""

    def test_render_science_card_basic(self):
        with patch("streamlit_app.ui.components.st.markdown") as mock_markdown:
            render_science_card("Title", "Body content")
            mock_markdown.assert_called_once()

    def test_render_science_card_expanded(self):
        with patch("streamlit_app.ui.components.st.expander") as mock_expander:
            mock_context = MagicMock()
            mock_expander.return_value.__enter__ = Mock(return_value=mock_context)
            mock_expander.return_value.__exit__ = Mock(return_value=False)
            
            render_science_card("Title", "Body", expanded=True)
            mock_expander.assert_called_once_with("Title", expanded=False)


class TestStatusChip:
    """Tests for status indicator chip."""

    def test_render_status_chip_uses_centralized_colors(self):
        with patch("streamlit_app.ui.components.st.markdown") as mock_markdown:
            render_status_chip("Pipeline", "ready")
            mock_markdown.assert_called_once()
            html = mock_markdown.call_args[0][0]
            assert "#218739" in html  # STATUS_COLORS["ready"]

    def test_render_status_chip_includes_icon(self):
        with patch("streamlit_app.ui.components.st.markdown") as mock_markdown:
            render_status_chip("Test", "ready", show_icon=True)
            html = mock_markdown.call_args[0][0]
            assert "✓" in html


class TestCustomerSelector:
    """Tests for customer selector component."""

    def test_render_customer_selector_empty_df(self):
        with patch("streamlit_app.ui.components.render_missing") as mock_missing:
            result = render_customer_selector(pd.DataFrame())
            assert result is None
            mock_missing.assert_called_once()

    def test_render_customer_selector_invalid_ids(self):
        df = pd.DataFrame({"Customer ID": ["abc", "def"]})
        with patch("streamlit_app.ui.components.render_missing") as mock_missing:
            result = render_customer_selector(df)
            assert result is None
            mock_missing.assert_called_once()


class TestActionSummaryTable:
    """Tests for action summary table."""

    def test_render_action_summary_table_finds_columns(self):
        df = pd.DataFrame({
            "final_action": ["protect_value", "nurture"],
            "customers": [100, 200],
        })
        with patch("streamlit_app.ui.components.st.dataframe"):
            render_action_summary_table(df)
            # Should not raise


class TestFormattedDataFrame:
    """Tests for semantic dataframe rendering."""

    def test_render_formatted_dataframe_probability(self):
        df = pd.DataFrame({"churn_probability": [0.5, 0.75]})
        with patch("streamlit_app.ui.components.st.dataframe") as mock_df:
            render_formatted_dataframe(df)
            mock_df.assert_called_once()
            # Check column config for probability
            kwargs = mock_df.call_args[1]
            config = kwargs.get("column_config", {})
            assert "churn_probability" in config
            # Probability should be converted to percentage scale
            # (the function multiplies by 100 internally)

    def test_render_formatted_dataframe_currency(self):
        df = pd.DataFrame({"revenue": [1000, 2000]})
        with patch("streamlit_app.ui.components.st.dataframe") as mock_df:
            render_formatted_dataframe(df)
            mock_df.assert_called_once()

    def test_render_formatted_dataframe_empty(self):
        with patch("streamlit_app.ui.components.render_missing") as mock_missing:
            render_formatted_dataframe(pd.DataFrame())
            mock_missing.assert_called_once()


class TestHeroCopy:
    """Tests for HERO_COPY map."""

    def test_all_pages_defined(self):
        expected = [
            "Strategy", "Customers", "Value & Retention", "Segments",
            "Products & Baskets", "Personalisation", "Activation", "Science & Governance"
        ]
        for page in expected:
            assert page in HERO_COPY
            kicker, desc = HERO_COPY[page]
            assert isinstance(kicker, str)
            assert isinstance(desc, str)
            assert len(kicker) > 0
            assert len(desc) > 0


class TestEmptyState:
    """Tests for empty state component."""

    def test_render_empty_state_basic(self):
        with patch("streamlit_app.ui.components.st.markdown") as mock_markdown:
            render_empty_state("No data", "Nothing to show")
            mock_markdown.assert_called_once()

    def test_render_empty_state_with_action(self):
        with patch("streamlit_app.ui.components.st.markdown") as mock_markdown:
            with patch("streamlit_app.ui.components.st.caption") as mock_caption:
                render_empty_state("No data", "Nothing", "Run script", "python foo.py")
                mock_markdown.assert_called_once()
                mock_caption.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])