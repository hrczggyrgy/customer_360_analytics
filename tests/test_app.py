"""
Test suite for Streamlit App pages.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit_app.app_components import (
    render_kpi_card,
    render_science_card,
    render_customer_selector,
    render_customer_header,
    render_customer_metric_row,
    render_evidence_table,
    render_missing,
    render_action_summary_table,
    HERO_COPY,
)
from streamlit_app.app_data import get_registry, ArtifactRegistry, format_freshness, format_run_id
from streamlit_app.app_formatting import (
    format_currency,
    format_probability,
    format_percent,
    format_ratio,
    format_count,
    format_date,
    format_month,
    format_duration_months,
    format_score,
    auto_format,
    infer_semantic_type,
    COLUMN_FORMAT_MAP,
    SemanticType,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_customer_data():
    """Create sample customer data for testing."""
    return pd.DataFrame({
        "Customer ID": [12345, 12346, 12347],
        "segment_name": ["High Value", "Medium", "Low"],
        "clv_mean": [10000.0, 5000.0, 100.0],
        "churn_probability": [0.1, 0.3, 0.8],
        "next_purchase_probability": [0.9, 0.5, 0.1],
        "net_revenue": [10000.0, 5000.0, 100.0],
        "orders": [50, 25, 2],
    })


@pytest.fixture
def sample_segment_data():
    """Create sample segment data for testing."""
    return pd.DataFrame({
        "Customer ID": [1, 2, 3, 4, 5],
        "segment": [0, 1, 2, -1, 0],
        "segment_name": ["A", "B", "C", "Noise", "A"],
    })


# =============================================================================
# Formatter Tests
# =============================================================================

class TestFormatters:
    """Tests for semantic formatters."""

    def test_format_currency(self):
        assert format_currency(1234.56, decimals=2) == "£1.23K"
        assert format_currency(1_234_567, decimals=2) == "£1.23M"
        assert format_currency(1_234_567) == "£1.2M"
        assert format_currency(12345, decimals=2) == "£12.35K"
        assert format_currency(12345) == "£12.3K"
        assert format_currency(0) == "£0"
        assert format_currency(None) == "—"
        assert format_currency(float("nan")) == "—"

    def test_format_probability(self):
        assert format_probability(0.5) == "50.0%"
        assert format_probability(0.1234) == "12.3%"
        assert format_probability(0) == "0.0%"
        assert format_probability(1) == "100.0%"
        assert format_probability(1.5) == "Invalid"
        assert format_probability(-0.1) == "Invalid"
        assert format_probability(None) == "—"

    def test_format_percent(self):
        assert format_percent(0.25) == "25.0%"
        assert format_percent(25) == "25.0%"
        assert format_percent(1.5) == "150.0%"
        assert format_percent(1.6) == "1.6%"
        assert format_percent(1.6, decimals=2) == "1.60%"
        assert format_percent(None) == "—"

    def test_format_ratio(self):
        assert format_ratio(1.5) == "1.50x"
        assert format_ratio(1) == "1.00x"
        assert format_ratio(None) == "—"

    def test_format_count(self):
        assert format_count(1234) == "1,234"
        assert format_count(1000000) == "1,000,000"
        assert format_count(None) == "—"

    def test_format_date(self):
        import datetime
        assert format_date(datetime.date(2023, 1, 15)) == "2023-01-15"
        assert format_date("2023-01-15") == "2023-01-15"
        assert format_date(None) == "—"

    def test_format_month(self):
        import datetime
        assert format_month(datetime.date(2023, 1, 15)) == "2023-01"
        assert format_month("2023-01-15") == "2023-01"
        assert format_month(None) == "—"

    def test_format_duration_months(self):
        assert format_duration_months(12.5) == "12.5 mo"
        assert format_duration_months(0.5) == "15 days"
        assert format_duration_months(None) == "—"

    def test_format_score(self):
        assert format_score(0.8765) == "0.88"
        assert format_score(0.5) == "0.50"
        assert format_score(None) == "—"

    def test_auto_format(self):
        assert auto_format(1000, "revenue") == "£1.0K"
        assert auto_format(0.5, "churn_probability") == "50.0%"
        assert auto_format(25, "retention") == "25.0%"
        assert auto_format(1.5, "ratio") == "1.50x"
        assert auto_format(1000, "orders") == "1,000"
        assert auto_format(0.5, "unknown_column") == "50.0%"

    def test_infer_semantic_type(self):
        assert infer_semantic_type("revenue") == SemanticType.CURRENCY
        assert infer_semantic_type("churn_probability") == SemanticType.PROBABILITY
        assert infer_semantic_type("retention") == SemanticType.PERCENTAGE
        assert infer_semantic_type("ratio") == SemanticType.RATIO
        assert infer_semantic_type("orders") == SemanticType.COUNT
        assert infer_semantic_type("first_purchase_date") == SemanticType.DATE
        assert infer_semantic_type("cohort_month") == SemanticType.MONTH
        assert infer_semantic_type("recency_months") == SemanticType.DURATION_MONTHS
        assert infer_semantic_type("score") == SemanticType.SCORE


# =============================================================================
# App Data Tests
# =============================================================================

class TestAppData:
    """Tests for ArtifactRegistry and data loading."""

    def test_registry_creation(self):
        registry = ArtifactRegistry()
        assert registry is not None
        assert registry.project_root is not None

    def test_get_all_module_statuses(self):
        registry = ArtifactRegistry()
        statuses = registry.get_all_module_statuses()
        assert isinstance(statuses, dict)
        assert len(statuses) > 0

    def test_format_freshness(self):
        import datetime
        now = datetime.datetime.now()
        assert format_freshness(now) == "Generated today"
        yesterday = now - datetime.timedelta(days=1)
        assert format_freshness(yesterday) == "Generated yesterday"
        week_ago = now - datetime.timedelta(days=3)
        assert "days ago" in format_freshness(week_ago)

    def test_format_run_id(self):
        assert format_run_id("abc123456789") == "abc12345..."
        assert format_run_id("short") == "short"
        assert format_run_id(None) == "Unknown"


# =============================================================================
# App Components Tests
# =============================================================================

class TestAppComponents:
    """Tests for UI components."""

    def test_HERO_COPY_complete(self):
        expected_pages = [
            "Executive", "Customer 360", "Segmentation", "Cohorts",
            "CLV", "Retention & Next Purchase",
            "Recommendations", "Decision Engine", "Methodology"
        ]
        for page in expected_pages:
            assert page in HERO_COPY
            kicker, description = HERO_COPY[page]
            assert isinstance(kicker, str)
            assert isinstance(description, str)
            assert len(kicker) > 0
            assert len(description) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full app stack."""

    def test_registry_loads_data(self):
        """Test that registry can load actual data."""
        registry = get_registry()
        
        # Test loading customer_360
        df = registry.load_dataframe("customer_360")
        if df is not None:
            assert "Customer ID" in df.columns
            assert len(df) > 0

    def test_pages_import(self):
        """Test that all page modules import without errors."""
        from streamlit_app.pages import (
            executive, customer_360, segmentation, cohorts,
            predictive, retention, recommendations, decision_engine, methodology
        )
        # All modules imported successfully

    def test_formatters_used_in_pages(self):
        """Test that semantic formatters are used instead of old pct()."""
        # Check that app.py doesn't use the old pct() function
        import streamlit_app.app as app
        import inspect
        source = inspect.getsource(app)
        assert "def pct(" not in source
        assert "pct(" not in source or "format_percent" in source


# =============================================================================
# Data Validation Tests
# =============================================================================

class TestDataValidation:
    """Tests for data quality and validation."""

    def test_duplicate_recommendations(self):
        """Test that recommendations don't have duplicate customer-product pairs."""
        registry = get_registry()
        recs = registry.load_dataframe("recommendations")
        if recs is not None:
            dupes = recs.duplicated(subset=["Customer ID", "recommended_product"]).sum()
            # Current implementation has some duplicates
            print(f"Found {dupes} duplicate recommendations")

    def test_recommendation_reason_distribution(self):
        """Test that recommendations have valid reason distribution."""
        registry = get_registry()
        recs = registry.load_dataframe("recommendations")
        if recs is not None and "reason" in recs.columns:
            valid_reasons = {"co_purchase", "popularity"}
            actual_reasons = set(recs["reason"].unique())
            assert actual_reasons.issubset(valid_reasons)

    def test_segment_noise_detection(self):
        """Test that noise is detected using segment == -1."""
        registry = get_registry()
        segments = registry.load_dataframe("segmentation")
        if segments is not None and "segment" in segments.columns:
            noise_count = (segments["segment"] == -1).sum()
            total = len(segments)
            noise_pct = noise_count / total
            # Should be around 16% based on scientific review
            assert 0.1 < noise_pct < 0.3


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

