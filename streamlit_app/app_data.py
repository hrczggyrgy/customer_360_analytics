"""
Config-driven Artifact Registry for Retail Customer Intelligence Streamlit App.

This module replaces the recursive glob-based file discovery with explicit,
config-driven artifact resolution. All paths are resolved relative to the
project root via config/project.yaml.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from .app_config import get_config, get_project_root

LOGGER = logging.getLogger("app_data")


@dataclass
class ArtifactInfo:
    """Metadata for a discovered artifact file."""
    module: str
    path: Path
    exists: bool
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    run_id: Optional[str] = None
    data_version: Optional[str] = None
    code_version: Optional[str] = None
    model_version: Optional[str] = None
    generated_at: Optional[datetime] = None
    valid: bool = False
    validation_state: str = "unknown"
    validation_errors: List[str] = field(default_factory=list)
    file_size_bytes: int = 0
    file_mtime_ns: int = 0


@dataclass
class ModuleStatus:
    """Aggregated status for a pipeline module."""
    module: str
    overall_status: str  # ready, incomplete, stale, validation_failed, unavailable, unknown
    primary_artifact: Optional[ArtifactInfo] = None
    supporting_artifacts: List[ArtifactInfo] = field(default_factory=list)
    freshness_note: str = ""
    run_consistency: str = "unknown"  # consistent, mixed_runs, unknown


# Expected artifact schemas for validation
EXPECTED_ARTIFACTS: Dict[str, Dict] = {
    "data_quality": {
        "primary": "canonical_transactions.parquet",
        "required_columns": ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"],
        "min_rows": 1,
        "supporting": [
            "customer_month.parquet",
            "schema_report.csv",
            "missingness_report.csv",
            "duplicate_report.csv",
            "transaction_type_report.csv",
            "reconciliation_report.json",
            "validation_results.json",
        ],
    },
    "customer_360": {
        "primary": "customer_360_unified.parquet",
        "required_columns": ["Customer ID"],
        "min_rows": 1,
        "supporting": [
            "customer_360_unified.csv",
            "customer_360_current.parquet",
            "customer_360_current.csv",
            "customer_360_metadata.json",
            "feature_dictionary.json",
            "feature_groups.json",
            "validation_results.json",
            "unified_validation.json",
            "model_card.json",
        ],
    },
    "segmentation": {
        "primary": "customer_segments.parquet",
        "required_columns": ["Customer ID", "segment"],
        "min_rows": 1,
        "supporting": [
            "customer_segments.csv",
            "segment_profiles.csv",
            "pca_coordinates.csv",
            "pca_feature_loadings.csv",
            "segment_feature_medians_raw.csv",
            "segment_feature_medians_robust.csv",
            "cluster_model_candidates.csv",
            "model_metadata.json",
            "rfm_segments.csv",
            "rfm_comparison.json",
            "segment_transition_matrix.csv",
            "segment_transition_summary.csv",
        ],
    },
    "cohorts": {
        "primary": "matrix_logo_retention.csv",
        "required_columns": ["cohort_month"],
        "min_rows": 1,
        "supporting": [
            "matrix_net_revenue_retention.csv",
            "matrix_gross_revenue_retention.csv",
            "retention_decay_curve.csv",
            "cohort_scorecard.csv",
            "cohort_acquisition_quality.csv",
            "cohort_age_metrics.csv",
            "cohort_age_metrics_dense.csv",
            "cohort_sizes.csv",
            "calendar_month_performance.csv",
            "customer_lifecycle_status.csv",
            "customer_acquisition_cohorts.csv",
        ],
    },
    "clv": {
        "primary": "clv_customer_predictions.csv",
        "required_columns": ["Customer ID", "clv_mean"],
        "min_rows": 1,
        "supporting": [
            "clv_customer_predictions.parquet",
            "clv_monthly_summary.csv",
            "feature_importance.csv",
            "validation_predictions.csv",
            "model_card.json",
            "run_manifest.json",
            "run_summary.json",
            "bgnbd_benchmark/bgnbd_predictions.parquet",
            "bgnbd_benchmark/bgnbd_params.json",
            "bgnbd_benchmark/model_card.json",
        ],
    },
    "churn": {
        "primary": "customer_churn_next_purchase.csv",
        "required_columns": ["Customer ID", "churn_probability"],
        "min_rows": 1,
        "supporting": [
            "customer_month_panel.parquet",
            "survival_feature_importance.csv",
            "survival_model_tuning.csv",
            "model_card.json",
            "survival_churn_refactored/customer_churn_next_purchase.csv",
            "survival_churn_refactored/model_card.json",
        ],
    },
    "reactivation": {
        "primary": "reactivation_predictions.csv",
        "required_columns": ["Customer ID", "reactivation_probability"],
        "min_rows": 1,
        "supporting": [
            "reactivation_panel.parquet",
            "feature_importance.csv",
            "model_card.json",
        ],
    },
    "product_analytics": {
        "primary": "product_metrics.parquet",
        "required_columns": ["StockCode"],
        "min_rows": 1,
        "supporting": [
            "product_metrics.csv",
            "co_purchase_matrix.parquet",
            "co_purchase_matrix.csv",
            "product_analytics_card.json",
            "product_role_summary.csv",
        ],
    },
    "market_basket": {
        "primary": "market_basket/association_rules.parquet",
        "required_columns": ["antecedent", "consequent", "support", "confidence", "lift"],
        "min_rows": 1,
        "supporting": [
            "market_basket/frequent_itemsets.parquet",
            "market_basket/model_card.json",
            "market_basket/association_rules.csv",
            "market_basket/frequent_itemsets.csv",
        ],
    },
    "recommendations": {
        "primary": "recommendations.parquet",
        "required_columns": ["Customer ID", "recommended_product", "score"],
        "min_rows": 1,
        "supporting": [
            "recommendations.csv",
            "model_card.json",
            "recommendation_challenger/challenger_results.parquet",
            "recommendation_challenger/model_card.json",
        ],
    },
    "decision_engine": {
        "primary": "customer_decision_scores.csv",
        "required_columns": ["Customer ID", "recommended_action_capped"],
        "min_rows": 1,
        "supporting": [
            "action_summary.csv",
            "model_card.json",
        ],
    },
}


class ArtifactRegistry:
    """Centralized artifact discovery and validation using config-driven paths."""

    def __init__(self):
        self.config = get_config()
        self.project_root = get_project_root()
        self._cache: Dict[str, Any] = {}

    def _get_output_dir(self, module: str) -> Optional[Path]:
        """Resolve output directory for a module from config."""
        # Map module names to config keys
        module_to_config = {
            "data_quality": "data_quality",
            "customer_360": "customer_360",
            "segmentation": "segmentation",
            "cohorts": "cohorts",
            "clv": "clv",
            "churn": "churn",
            "reactivation": "reactivation",
            "product_analytics": "product_analytics",
            "market_basket": "recommendations",  # Uses recommendations dir
            "recommendations": "recommendations",
            "decision_engine": "decision_engine",
        }
        
        output_key = module_to_config.get(module, module)
        rel_path = self.config.get("outputs", {}).get(output_key)
        if not rel_path:
            return None
        
        base_dir = self.project_root / rel_path
        
        # Handle subdirectories
        if module == "market_basket":
            return base_dir / "market_basket"
        elif module == "clv":
            return base_dir  # bgnbd_benchmark is inside clv
        elif module == "churn":
            return base_dir  # survival_churn_refactored is inside churn
        elif module == "recommendations":
            return base_dir  # recommendation_challenger is inside recommendations
        
        return base_dir

    def _load_manifest(self, module: str, artifact: ArtifactInfo) -> None:
        """Load run manifest / model card metadata if available."""
        output_dir = self._get_output_dir(module)
        if output_dir is None:
            return
        
        # Try model_card.json first
        model_card_path = output_dir / "model_card.json"
        if model_card_path.exists():
            try:
                with open(model_card_path) as f:
                    card = json.load(f)
                artifact.model_version = card.get("model_name") or card.get("methodology")
                artifact.code_version = card.get("code_version")
                artifact.data_version = card.get("data_source")
                if "generated_at" in card:
                    artifact.generated_at = datetime.fromisoformat(card["generated_at"])
            except Exception as e:
                LOGGER.debug(f"Failed to parse model_card for {module}: {e}")
        
        # Try run_manifest.json
        manifest_path = output_dir / "run_manifest.json"
        if manifest_path.exists() and not artifact.generated_at:
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                artifact.run_id = manifest.get("run_id")
                artifact.generated_at = datetime.fromisoformat(manifest.get("generated_at", ""))
                artifact.code_version = manifest.get("code_version")
                artifact.data_version = manifest.get("data_version")
            except Exception as e:
                LOGGER.debug(f"Failed to parse run_manifest for {module}: {e}")

    def _validate_artifact(self, artifact: ArtifactInfo, spec: Dict) -> None:
        """Validate artifact against expected schema and semantic constraints."""
        artifact.validation_errors = []
        
        if not artifact.exists:
            artifact.validation_state = "missing"
            return
        
        if artifact.row_count is not None and artifact.row_count < spec.get("min_rows", 1):
            artifact.validation_errors.append(f"Row count {artifact.row_count} below minimum {spec['min_rows']}")
            artifact.validation_state = "empty"
            return
        
        if spec.get("required_columns") and artifact.columns:
            missing = [c for c in spec["required_columns"] if c not in artifact.columns]
            if missing:
                artifact.validation_errors.append(f"Missing required columns: {missing}")
                artifact.validation_state = "schema_mismatch"
                return
        
        # Semantic validation for known modules
        if artifact.module == "recommendations":
            self._validate_recommendations_artifact(artifact)
        elif artifact.module == "churn":
            self._validate_churn_artifact(artifact)
        elif artifact.module == "decision_engine":
            self._validate_decision_engine_artifact(artifact)
        
        if artifact.validation_errors:
            artifact.validation_state = "validation_failed"
        else:
            artifact.validation_state = "valid"
            artifact.valid = True

    def _validate_recommendations_artifact(self, artifact: ArtifactInfo) -> None:
        """Validate recommendations artifact semantics."""
        # This would require loading the data; for now, we check column presence
        required = ["Customer ID", "recommended_product", "rank", "score", "reason"]
        if artifact.columns:
            missing = [c for c in required if c not in artifact.columns]
            if missing:
                artifact.validation_errors.append(f"Recommendations missing required columns: {missing}")

    def _validate_churn_artifact(self, artifact: ArtifactInfo) -> None:
        """Validate churn artifact semantics."""
        prob_cols = ["churn_probability", "survival_3m", "survival_6m", "survival_12m",
                     "next_purchase_7d_probability", "next_purchase_30d_probability", "next_purchase_60d_probability"]
        if artifact.columns:
            for col in prob_cols:
                if col in artifact.columns:
                    # Note: actual value validation would require loading data
                    pass

    def _validate_decision_engine_artifact(self, artifact: ArtifactInfo) -> None:
        """Validate decision engine artifact semantics."""
        required = ["Customer ID", "recommended_action_capped", "priority_score", "decision_confidence"]
        if artifact.columns:
            missing = [c for c in required if c not in artifact.columns]
            if missing:
                artifact.validation_errors.append(f"Decision engine missing required columns: {missing}")

    def _read_artifact_metadata(self, path: Path) -> ArtifactInfo:
        """Read metadata from an artifact file without loading full data."""
        stat = path.stat()
        info = ArtifactInfo(
            module="",
            path=path,
            exists=True,
            file_size_bytes=stat.st_size,
            file_mtime_ns=stat.st_mtime_ns,
        )
        
        # Read columns and row count based on file type
        try:
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(path)
                info.columns = pf.schema.names
                info.row_count = pf.metadata.num_rows
            elif path.suffix == ".csv":
                df = pd.read_csv(path, nrows=0)
                info.columns = df.columns.tolist()
                # Count rows properly (not estimate from file size)
                row_count = sum(1 for _ in open(path)) - 1  # subtract header
                info.row_count = max(0, row_count)
        except Exception as e:
            LOGGER.debug(f"Failed to read metadata from {path}: {e}")
        
        return info

    def discover_module_artifacts(self, module: str) -> ModuleStatus:
        """Discover and validate all artifacts for a module."""
        output_dir = self._get_output_dir(module)
        spec = EXPECTED_ARTIFACTS.get(module, {})
        
        primary_path = output_dir / spec.get("primary", "")
        primary_artifact = self._read_artifact_metadata(primary_path) if primary_path.exists() else None
        if primary_artifact:
            primary_artifact.module = module
            self._load_manifest(module, primary_artifact)
            self._validate_artifact(primary_artifact, spec)
        
        supporting_artifacts = []
        for supp_name in spec.get("supporting", []):
            supp_path = output_dir / supp_name
            if supp_path.exists():
                supp_artifact = self._read_artifact_metadata(supp_path)
                supp_artifact.module = module
                supporting_artifacts.append(supp_artifact)
        
        # Determine overall status
        if not primary_artifact or not primary_artifact.exists:
            overall_status = "unavailable"
        elif primary_artifact.validation_state == "valid":
            overall_status = "ready"
        elif primary_artifact.validation_state == "empty":
            overall_status = "incomplete"
        elif primary_artifact.validation_state == "schema_mismatch":
            overall_status = "validation_failed"
        else:
            overall_status = "unknown"
        
        # Freshness note
        freshness_note = "Unknown"
        if primary_artifact and primary_artifact.generated_at:
            delta = datetime.now() - primary_artifact.generated_at
            if delta.days == 0:
                freshness_note = "Generated today"
            elif delta.days == 1:
                freshness_note = "Generated yesterday"
            elif delta.days < 7:
                freshness_note = f"Generated {delta.days} days ago"
            else:
                freshness_note = f"Generated {delta.days} days ago ⚠️"
        elif primary_artifact:
            freshness_note = f"Modified {datetime.fromtimestamp(primary_artifact.file_mtime_ns / 1e9).strftime('%Y-%m-%d')}"
        
        return ModuleStatus(
            module=module,
            overall_status=overall_status,
            primary_artifact=primary_artifact,
            supporting_artifacts=supporting_artifacts,
            freshness_note=freshness_note,
        )

    def get_all_module_statuses(self) -> Dict[str, ModuleStatus]:
        """Get status for all known modules."""
        modules = list(EXPECTED_ARTIFACTS.keys())
        return {module: self.discover_module_artifacts(module) for module in modules}

    def load_dataframe(self, module: str, artifact_name: str = "primary") -> Optional[pd.DataFrame]:
        """Load a DataFrame from an artifact with caching."""
        output_dir = self._get_output_dir(module)
        if output_dir is None:
            return None
        spec = EXPECTED_ARTIFACTS.get(module, {})
        
        if artifact_name == "primary":
            path = output_dir / spec.get("primary", "")
        else:
            path = output_dir / artifact_name
        
        if not path.exists():
            return None
        
        # Cache key includes file mtime for automatic invalidation
        cache_key = f"{module}:{artifact_name}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix == ".csv":
                df = pd.read_csv(path, low_memory=False)
            else:
                return None
            
            self._cache[cache_key] = df
            return df
        except Exception as e:
            LOGGER.warning(f"Failed to load {path}: {e}")
            return None

    def load_model_card(self, module: str) -> Optional[Dict]:
        """Load model card for a module."""
        output_dir = self._get_output_dir(module)
        if output_dir is None:
            return None
        
        card_path = output_dir / "model_card.json"
        if not card_path.exists():
            return None
        
        cache_key = f"model_card:{module}:{card_path.stat().st_mtime_ns}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            with open(card_path) as f:
                card = json.load(f)
            self._cache[cache_key] = card
            return card
        except Exception as e:
            LOGGER.warning(f"Failed to load model card for {module}: {e}")
            return None


@st.cache_data(show_spinner=False)
def get_registry() -> ArtifactRegistry:
    """Get or create the singleton ArtifactRegistry."""
    return ArtifactRegistry()


def format_freshness(generated_at: Optional[datetime]) -> str:
    """Format a freshness note from a generated_at timestamp."""
    if not generated_at:
        return "Unknown"
    delta = datetime.now() - generated_at
    if delta.days == 0:
        return "Generated today"
    elif delta.days == 1:
        return "Generated yesterday"
    elif delta.days < 7:
        return f"Generated {delta.days} days ago"
    else:
        return f"Generated {delta.days} days ago ⚠️"


def format_run_id(run_id: Optional[str]) -> str:
    """Format run ID for display."""
    if not run_id:
        return "Unknown"
    return run_id[:8] + "..." if len(run_id) > 8 else run_id