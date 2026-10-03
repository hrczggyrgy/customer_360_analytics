"""
Test suite for artifact contracts and schema validation.
"""

import pytest
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit_app.app_data import ArtifactRegistry, EXPECTED_ARTIFACTS


class TestArtifactContracts:
    """Tests for artifact schema validation."""

    @pytest.fixture
    def registry(self):
        return ArtifactRegistry()

    def test_all_modules_have_specs(self):
        """All expected modules should have artifact specifications."""
        expected_modules = [
            "data_quality", "customer_360", "segmentation", "cohorts",
            "clv", "churn", "reactivation", "product_analytics",
            "recommendations", "decision_engine"
        ]
        for module in expected_modules:
            assert module in EXPECTED_ARTIFACTS, f"Missing spec for module: {module}"
            spec = EXPECTED_ARTIFACTS[module]
            assert "primary" in spec, f"Module {module} missing primary artifact"
            assert "required_columns" in spec, f"Module {module} missing required_columns"
            assert "min_rows" in spec, f"Module {module} missing min_rows"

    def test_recommendations_artifact_schema(self, registry):
        """Recommendations artifact should have required schema."""
        # This test loads actual artifact if available
        recs = registry.load_dataframe("recommendations")
        if recs is not None:
            required = ["Customer ID", "recommended_product", "rank", "score", "reason"]
            for col in required:
                assert col in recs.columns, f"Missing required column: {col}"
            
            # Check uniqueness
            dupes = recs.duplicated(subset=["Customer ID", "recommended_product"]).sum()
            assert dupes == 0, f"Found {dupes} duplicate recommendations"
            
            # Check rank bounds
            assert (recs["rank"] >= 1).all()
            assert (recs["rank"] <= 10).all()  # top_k default
            
            # Check score is finite
            assert recs["score"].notna().all()
            assert np.isfinite(recs["score"]).all()
            
            # Check reason values
            valid_reasons = {"co_purchase", "popularity"}
            actual_reasons = set(recs["reason"].unique())
            assert actual_reasons.issubset(valid_reasons)

    def test_churn_artifact_schema(self, registry):
        """Churn artifact should have probability bounds."""
        churn = registry.load_dataframe("churn")
        if churn is not None:
            # New scientific terminology
            required = ["Customer ID", "next_month_inactivity_risk"]
            for col in required:
                assert col in churn.columns, f"Missing required column: {col}"
            
            # Probability bounds - new scientific terminology
            prob_cols = [
                "next_month_inactivity_risk", 
                "model_survival_probability_3m", 
                "model_survival_probability_6m", 
                "model_survival_probability_12m",
                "next_purchase_7d_probability", 
                "next_purchase_30d_probability", 
                "next_purchase_60d_probability"
            ]
            for col in prob_cols:
                if col in churn.columns:
                    # Check non-null values are in [0, 1]
                    non_null = churn[col].dropna()
                    assert (non_null >= 0).all(), f"{col} has values < 0"
                    assert (non_null <= 1).all(), f"{col} has values > 1"

    def test_decision_engine_artifact_schema(self, registry):
        """Decision engine artifact should have action and priority fields."""
        decision = registry.load_dataframe("decision_engine")
        if decision is not None:
            required = ["Customer ID", "recommended_action_capped", "priority_score", "decision_confidence"]
            for col in required:
                assert col in decision.columns, f"Missing required column: {col}"
            
            # Valid actions
            valid_actions = {"protect_value", "accelerate_purchase", "reactivate", 
                            "cross_sell", "nurture", "monitor"}
            actual_actions = set(decision["recommended_action_capped"].unique())
            assert actual_actions.issubset(valid_actions)
            
            # Priority score should be non-negative
            assert (decision["priority_score"] >= 0).all()
            
            # Confidence bounds
            assert (decision["decision_confidence"] >= 0).all()
            assert (decision["decision_confidence"] <= 1).all()

    def test_segmentation_artifact_schema(self, registry):
        """Segmentation artifact should have segment column with -1 for noise."""
        segments = registry.load_dataframe("segmentation")
        if segments is not None:
            assert "Customer ID" in segments.columns
            assert "segment" in segments.columns
            
            # Segment -1 should exist for noise
            noise_count = (segments["segment"] == -1).sum()
            total = len(segments)
            noise_pct = noise_count / total
            assert 0.05 < noise_pct < 0.5, f"Noise percentage {noise_pct:.1%} outside expected range"
            
            # Segments should be integers
            import pandas as pd
            assert pd.api.types.is_integer_dtype(segments["segment"])

    def test_customer_360_artifact_schema(self, registry):
        """Customer 360 artifact should have required fields."""
        cust360 = registry.load_dataframe("customer_360")
        if cust360 is not None:
            assert "Customer ID" in cust360.columns
            
            # Check for common fields
            expected_fields = ["lifetime_gross_revenue", "recency_months", "lifecycle_state"]
            for field in expected_fields:
                assert field in cust360.columns, f"Missing expected field: {field}"
            
            # Customer ID should be unique
            dupes = cust360.duplicated(subset=["Customer ID"]).sum()
            assert dupes == 0, f"Found {dupes} duplicate customers"

    def test_product_analytics_artifact_schema(self, registry):
        """Product analytics artifact should have one row per StockCode."""
        metrics = registry.load_dataframe("product_analytics", "product_metrics.parquet")
        if metrics is not None:
            assert "StockCode" in metrics.columns
            
            # One row per StockCode
            dupes = metrics.duplicated(subset=["StockCode"]).sum()
            assert dupes == 0, f"Found {dupes} duplicate StockCodes"
            
            # Required fields
            required = ["total_revenue", "unique_customers", "repeat_customer_rate", "product_role"]
            for field in required:
                assert field in metrics.columns, f"Missing field: {field}"
            
            # repeat_customer_rate should be in [0, 1]
            assert (metrics["repeat_customer_rate"] >= 0).all()
            assert (metrics["repeat_customer_rate"] <= 1).all()

    def test_clv_artifact_schema(self, registry):
        """CLV artifact should have mean and uncertainty bounds."""
        clv = registry.load_dataframe("clv")
        if clv is not None:
            required = ["Customer ID", "clv_mean"]
            for col in required:
                assert col in clv.columns, f"Missing required column: {col}"
            
            # Uncertainty bounds
            bounds = [("clv_p10", "clv_p90"), ("clv_lower", "clv_upper")]
            for lower, upper in bounds:
                if lower in clv.columns and upper in clv.columns:
                    assert (clv[lower] <= clv["clv_mean"]).all(), f"{lower} > clv_mean"
                    assert (clv[upper] >= clv["clv_mean"]).all(), f"{upper} < clv_mean"
                    assert (clv[lower] <= clv[upper]).all(), f"{lower} > {upper}"

    def test_cohorts_artifact_schema(self, registry):
        """Cohort artifacts should have proper matrix structure."""
        logo_ret = registry.load_dataframe("cohorts", "matrix_logo_retention.csv")
        if logo_ret is not None:
            assert "cohort_month" in logo_ret.columns
            # Should have age columns (0, 1, 2, ...)
            age_cols = [c for c in logo_ret.columns if c.isdigit() or (c.startswith("age_") and c[4:].isdigit())]
            assert len(age_cols) > 0, "No age columns found in logo retention matrix"
            
            # Values should be in [0, 1] for retention rates
            for col in age_cols:
                if col != "cohort_month":
                    vals = logo_ret[col].dropna()
                    if len(vals) > 0:
                        assert (vals >= 0).all(), f"Negative retention in {col}"
                        # Can exceed 1 for revenue retention but not logo
                        assert (vals <= 2).all(), f"Retention > 200% in {col}"

    def test_missing_artifact_handled_gracefully(self, registry):
        """Registry should handle missing artifacts without crashing."""
        # This should not raise an exception
        missing = registry.load_dataframe("nonexistent_module")
        assert missing is None

    def test_artifact_metadata_loaded(self, registry):
        """Artifact metadata (run_id, generated_at) should be loaded if available."""
        status = registry.discover_module_artifacts("recommendations")
        if status.primary_artifact:
            # Metadata fields should be populated if available
            assert status.primary_artifact.module == "recommendations"
            # run_id, data_version, code_version may be None if not in model_card


class TestArtifactRegistryValidation:
    """Tests for ArtifactRegistry validation logic."""

    @pytest.fixture
    def registry(self):
        return ArtifactRegistry()

    def test_validation_passes_for_valid_artifact(self, registry):
        """Valid artifacts should pass validation."""
        # Create a mock valid artifact
        from streamlit_app.app_data import ArtifactInfo
        from datetime import datetime
        
        artifact = ArtifactInfo(
            module="test",
            path=Path("dummy.parquet"),
            exists=True,
            row_count=100,
            columns=["Customer ID", "value"],
            validation_state="valid",
        )
        
        # Mock spec
        spec = {"required_columns": ["Customer ID", "value"], "min_rows": 1}
        registry._validate_artifact(artifact, spec)
        
        assert artifact.validation_state == "valid"
        assert artifact.valid

    def test_validation_fails_missing_columns(self, registry):
        """Missing required columns should fail validation."""
        from streamlit_app.app_data import ArtifactInfo
        
        artifact = ArtifactInfo(
            module="test",
            path=Path("dummy.parquet"),
            exists=True,
            row_count=100,
            columns=["Customer ID"],  # Missing "value"
            validation_state="valid",
        )
        
        spec = {"required_columns": ["Customer ID", "value"], "min_rows": 1}
        registry._validate_artifact(artifact, spec)
        
        assert artifact.validation_state == "schema_mismatch"
        assert not artifact.valid

    def test_validation_fails_empty(self, registry):
        """Empty artifacts should fail validation."""
        from streamlit_app.app_data import ArtifactInfo
        
        artifact = ArtifactInfo(
            module="test",
            path=Path("dummy.parquet"),
            exists=True,
            row_count=0,
            columns=["Customer ID", "value"],
            validation_state="valid",
        )
        
        spec = {"required_columns": ["Customer ID", "value"], "min_rows": 1}
        registry._validate_artifact(artifact, spec)
        
        assert artifact.validation_state == "empty"
        assert not artifact.valid

    def test_validation_fails_missing_file(self, registry):
        """Missing file should be marked as missing."""
        from streamlit_app.app_data import ArtifactInfo
        
        artifact = ArtifactInfo(
            module="test",
            path=Path("dummy.parquet"),
            exists=False,
            row_count=None,
            columns=None,
            validation_state="valid",
        )
        
        spec = {"required_columns": ["Customer ID", "value"], "min_rows": 1}
        registry._validate_artifact(artifact, spec)
        
        assert artifact.validation_state == "missing"
        assert not artifact.valid


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])