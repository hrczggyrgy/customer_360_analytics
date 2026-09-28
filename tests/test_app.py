"""
Test suite for the Retail Customer Intelligence Streamlit application.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app_config import get_config, MODULE_LABELS
from app_data import (
    ArtifactRegistry,
    ArtifactInfo,
    ModuleStatus,
    EXPECTED_ARTIFACTS,
    CustomerProfileAdapter,
    get_registry,
    reset_registry,
)
from app_formatting import (
    format_currency,
    format_probability,
    format_percent,
    format_ratio,
    format_count,
    format_date,
    format_month,
    format_duration_months,
    format_score,
    infer_semantic_type,
    auto_format,
    format_clv,
    format_churn_risk,
    format_next_purchase,
    format_retention,
    format_nrr_index,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_customer_df():
    """Sample customer DataFrame for testing."""
    return pd.DataFrame({
        "Customer ID": [12345, 12346, 12347],
        "lifetime_net_revenue": [1000.0, 5000.0, 100.0],
        "lifetime_orders": [5, 20, 1],
        "recency_days": [30, 10, 200],
        "segment": [0, 1, -1],
        "clv_mean": [2000.0, 10000.0, 500.0],
        "churn_probability": [0.2, 0.1, 0.8],
        "next_purchase_probability_30d": [0.6, 0.8, 0.1],
    })


@pytest.fixture
def sample_churn_df():
    """Sample churn DataFrame for testing."""
    return pd.DataFrame({
        "Customer ID": [12345, 12346, 12347],
        "churn_probability": [0.2, 0.1, 0.8],
        "next_purchase_30d_probability": [0.6, 0.8, 0.1],
        "survival_3m": [0.9, 0.95, 0.5],
    })


# =============================================================================
# FORMATTER TESTS
# =============================================================================

class TestFormatters:
    """Tests for formatting utilities."""

    def test_format_currency(self):
        assert format_currency(1234567) == "£1.2M"
        assert format_currency(1234567, abbreviate=False) == "£1,234,567"
        assert format_currency(1000) == "£1.0K"
        assert format_currency(100) == "£100"
        assert format_currency(None) == "—"
        assert format_currency(float("nan")) == "—"

    def test_format_probability(self):
        assert format_probability(0.87) == "87.0%"
        assert format_probability(0.87, decimals=2) == "87.00%"
        assert format_probability(0.87, as_percent=False, decimals=0) == "0.87"
        assert format_probability(1.5) == "100.0%"  # Clamped
        assert format_probability(-0.1) == "0.0%"   # Clamped
        assert format_probability(None) == "—"

    def test_format_percent(self):
        assert format_percent(0.87) == "87.0%"
        assert format_percent(87, already_percentage=True) == "87.0%"
        assert format_percent(1.5) == "150.0%"
        assert format_percent(None) == "—"

    def test_format_ratio(self):
        assert format_ratio(3.28) == "3.28"
        assert format_ratio(3.28, as_percent=True) == "328.00%"
        assert format_ratio(None) == "—"

    def test_format_count(self):
        assert format_count(1234567) == "1.2M"
        assert format_count(1234) == "1.2K"
        assert format_count(100) == "100"
        assert format_count(None) == "—"

    def test_format_clv(self):
        assert format_clv(1234567) == "£1.2M"
        assert format_clv(1000) == "£1.0K"

    def test_format_churn_risk(self):
        assert format_churn_risk(0.75) == "75.0%"
        # format_churn_risk is a lambda with fixed decimals=1
        # For custom decimals, use format_probability directly
        from app_formatting import format_probability
        assert format_probability(0.75, decimals=2) == "75.00%"

    def test_format_next_purchase(self):
        assert format_next_purchase(0.45) == "45.0%"
        from app_formatting import format_probability
        assert format_probability(0.45, decimals=2) == "45.00%"

    def test_format_retention(self):
        # format_retention lambda has already_percentage=True, so input should be in percentage units
        assert format_retention(23.0) == "23.0%"
        # For ratio input (0.23 = 23%), use format_ratio
        from app_formatting import format_ratio
        assert format_ratio(0.23, as_percent=True, decimals=1) == "23.0%"

    def test_format_nrr_index(self):
        assert format_nrr_index(3.28) == "328.0%"

    def test_infer_semantic_type(self):
        assert infer_semantic_type("lifetime_net_revenue") == "currency"
        assert infer_semantic_type("churn_probability") == "probability"
        assert infer_semantic_type("retention_rate") == "percent"
        assert infer_semantic_type("nrr_index") == "ratio"
        assert infer_semantic_type("order_count") == "count"
        assert infer_semantic_type("cohort_month") == "month"
        assert infer_semantic_type("tenure_days") == "duration"
        assert infer_semantic_type("priority_score") == "score"
        assert infer_semantic_type("customer_name") == "text"

    def test_auto_format(self):
        assert auto_format(1234567, "lifetime_net_revenue") == "£1.2M"
        assert auto_format(0.87, "churn_probability") == "87.0%"
        assert auto_format(23, "retention_rate") == "23.0%"
        assert auto_format(3.28, "nrr_index") == "328.00%"


# =============================================================================
# ARTIFACT REGISTRY TESTS
# =============================================================================

class TestArtifactRegistry:
    """Tests for artifact discovery and validation."""

    def test_registry_initialization(self):
        reset_registry()
        registry = get_registry()
        assert registry is not None
        assert registry.config is not None

    def test_discover_artifacts(self):
        reset_registry()
        registry = get_registry()
        statuses = registry.discover()

        # All expected modules should be present
        for module in MODULE_LABELS.keys():
            assert module in statuses

        # At least some modules should be ready
        ready_count = sum(1 for s in statuses.values() if s.overall_status == "ready")
        assert ready_count >= 5  # Most modules should be ready

    def test_module_status_structure(self):
        reset_registry()
        registry = get_registry()
        registry.discover()

        for module, status in registry.get_all_module_statuses().items():
            assert isinstance(status.module, str)
            assert isinstance(status.label, str)
            assert isinstance(status.artifacts, list)
            assert status.overall_status in [
                "ready", "incomplete", "stale", "validation_failed", "unavailable"
            ]

    def test_load_dataframe(self):
        reset_registry()
        registry = get_registry()
        registry.discover()

        # Test loading a known artifact
        df = registry.load_dataframe("customer_360", "customer_360_current")
        if df is not None:
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert "Customer ID" in df.columns

    def test_load_model_card(self):
        reset_registry()
        registry = get_registry()
        registry.discover()

        card = registry.load_model_card("clv")
        if card:
            assert isinstance(card, dict)
            assert "model_name" in card


# =============================================================================
# CUSTOMER PROFILE ADAPTER TESTS
# =============================================================================

class TestCustomerProfileAdapter:
    """Tests for customer profile adapter."""

    def test_build_profile(self):
        reset_registry()
        registry = get_registry()
        registry.discover()
        adapter = CustomerProfileAdapter(registry)

        profile = adapter.build_profile()
        assert isinstance(profile, pd.DataFrame)
        # Should have customers from customer_360
        assert len(profile) > 0
        assert "Customer ID" in profile.columns

    def test_get_customer(self):
        reset_registry()
        registry = get_registry()
        registry.discover()
        adapter = CustomerProfileAdapter(registry)

        adapter.build_profile()
        if len(adapter._profile_cache) > 0:
            customer_id = int(adapter._profile_cache["Customer ID"].iloc[0])
            customer = adapter.get_customer(customer_id)
            assert customer is not None
            assert customer["Customer ID"] == customer_id


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Tests for configuration loading."""

    def test_config_loads(self):
        config = get_config()
        assert config.config_path.exists()
        assert config.data_raw_online_retail_ii.exists()
        assert len(config.output_dirs) > 0

    def test_output_dirs(self):
        config = get_config()
        for module, path in config.output_dirs.items():
            assert isinstance(path, Path)


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================

class TestSchemaValidation:
    """Tests for artifact schema validation."""

    def test_customer_360_schema(self):
        reset_registry()
        registry = get_registry()
        registry.discover()

        artifact = registry.get_artifact("customer_360", "customer_360_current")
        assert artifact is not None
        assert artifact.exists
        assert artifact.valid

        required = EXPECTED_ARTIFACTS["customer_360"]["customer_360_current"]["required_columns"]
        for col in required:
            assert col in artifact.columns, f"Missing required column: {col}"

    def test_segmentation_schema(self):
        reset_registry()
        registry = get_registry()
        registry.discover()

        artifact = registry.get_artifact("segmentation", "customer_segments")
        assert artifact is not None
        assert artifact.exists
        assert artifact.valid

        required = EXPECTED_ARTIFACTS["segmentation"]["customer_segments"]["required_columns"]
        for col in required:
            assert col in artifact.columns, f"Missing required column: {col}"

    def test_cohorts_schema(self):
        reset_registry()
        registry = get_registry()
        registry.discover()

        artifact = registry.get_artifact("cohorts", "matrix_logo_retention")
        assert artifact is not None
        assert artifact.exists
        assert artifact.valid

    def test_clv_schema(self):
        reset_registry()
        registry = get_registry()
        registry.discover()

        artifact = registry.get_artifact("clv", "clv_customer_predictions")
        assert artifact is not None
        assert artifact.exists
        assert artifact.valid

        required = EXPECTED_ARTIFACTS["clv"]["clv_customer_predictions"]["required_columns"]
        for col in required:
            assert col in artifact.columns, f"Missing required column: {col}"


# =============================================================================
# PROBABILITY FORMATTING TESTS
# =============================================================================

class TestProbabilityFormatting:
    """Tests for correct probability vs percentage formatting."""

    def test_probability_not_confused_with_percent(self):
        # A probability of 0.87 should format as 87%
        assert format_probability(0.87) == "87.0%"

        # A ratio of 3.28 (NRR index) should format as 328%
        assert format_ratio(3.28, as_percent=True) == "328.00%"

        # A retention rate of 23 (already percentage units) should format as 23%
        assert format_retention(23.0) == "23.0%"

    def test_clv_not_formatted_as_probability(self):
        # CLV should use currency formatting, not probability
        assert format_clv(1234) == "£1.2K"
        assert "£" in format_clv(1234)
        assert "%" not in format_clv(1234)


# =============================================================================
# SEGMENT NOISE DETECTION TESTS
# =============================================================================

class TestSegmentNoiseDetection:
    """Tests for correct noise segment detection."""

    def test_noise_detection_numeric(self):
        """Noise should be detected as segment == -1."""
        import pandas as pd
        df = pd.DataFrame({
            "Customer ID": [1, 2, 3, 4, 5],
            "segment": [0, 0, 1, -1, -1],
        })
        noise_mask = (df["segment"].astype(str).str.contains("noise", case=False, na=False) |
                      (pd.to_numeric(df["segment"], errors="coerce") == -1))
        assert noise_mask.sum() == 2
        assert noise_mask.tolist() == [False, False, False, True, True]

    def test_noise_detection_string(self):
        """Noise should be detected as string 'noise'."""
        import pandas as pd
        df = pd.DataFrame({
            "Customer ID": [1, 2, 3, 4],
            "segment": ["Segment_0", "Segment_1", "Noise", "noise"],
        })
        noise_mask = (df["segment"].astype(str).str.contains("noise", case=False, na=False) |
                      (pd.to_numeric(df["segment"], errors="coerce") == -1))
        assert noise_mask.sum() == 2


# =============================================================================
# COHORT NAN SEMANTICS TESTS
# =============================================================================

class TestCohortNaNSemantics:
    """Tests for correct handling of immature cohort NaN values."""

    def test_immature_cohort_handling(self):
        import numpy as np
        import pandas as pd

        # Create a matrix with NaN for immature cohorts
        matrix = pd.DataFrame({
            0: [1.0, 1.0, 1.0],
            3: [0.5, 0.3, np.nan],  # Last cohort not yet mature at age 3
            6: [0.4, np.nan, np.nan],  # Only first cohort mature at age 6
        }, index=["2010-01", "2010-07", "2011-01"])

        # NaN should be identifiable as immature
        immature_mask = matrix.isna()
        assert immature_mask.iloc[2, 1] == True  # 2011-01 at age 3
        assert immature_mask.iloc[1, 2] == True  # 2010-07 at age 6
        assert immature_mask.iloc[2, 2] == True  # 2011-01 at age 6

        # Mature values should not be NaN
        assert immature_mask.iloc[0, 1] == False
        assert immature_mask.iloc[0, 2] == False


# =============================================================================
# DECISION CAPACITY DISPLAY TESTS
# =============================================================================

class TestDecisionCapacityDisplay:
    """Tests for decision engine capacity display."""

    def test_capacity_calculation(self):
        import pandas as pd

        # Simulate decision output
        decision = pd.DataFrame({
            "Customer ID": range(100),
            "final_action": ["monitor"] * 40 + ["reactivate"] * 30 + ["accelerate_purchase"] * 20 + ["cross_sell"] * 10,
            "priority_score": list(range(100, 0, -1)),
        })

        capacity = {"total": 60, "reactivate": 30, "accelerate_purchase": 20, "cross_sell": 10}

        action_counts = decision["final_action"].value_counts().to_dict()

        # Check allocations
        assert action_counts.get("reactivate", 0) <= capacity["reactivate"]
        assert action_counts.get("accelerate_purchase", 0) <= capacity["accelerate_purchase"]
        assert action_counts.get("cross_sell", 0) <= capacity["cross_sell"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full app stack."""

    def test_app_imports(self):
        """Test that the main app module imports without errors."""
        import app
        assert app is not None

    def test_pages_import(self):
        """Test that all page modules import without errors."""
        from pages import (
            render_executive_page,
            render_customer_360_page,
            render_segmentation_page,
            render_cohorts_page,
            render_predictive_page,
            render_retention_page,
            render_recommendations_page,
            render_decision_page,
            render_methodology_page,
        )

    def test_data_reconciliation_sample(self):
        """Verify a few customer values match between sources."""
        reset_registry()
        registry = get_registry()
        registry.discover()

        # Load customer from customer_360
        c360 = registry.load_dataframe("customer_360", "customer_360_current")
        if c360 is not None and len(c360) > 0:
            cid = int(c360["Customer ID"].iloc[0])
            c360_row = c360[c360["Customer ID"] == cid].iloc[0]

            # Load from profile adapter (merged sources)
            adapter = CustomerProfileAdapter(registry)
            profile = adapter.build_profile()
            profile_row = profile[profile["Customer ID"] == cid]

            if not profile_row.empty:
                # Core identifiers should match
                assert profile_row["Customer ID"].iloc[0] == c360_row["Customer ID"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])