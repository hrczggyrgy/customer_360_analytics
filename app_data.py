"""
Artifact registry and data loading layer for the Retail Customer Intelligence application.

This module replaces the fragile recursive file discovery with a version-aware,
config-driven artifact registry that tracks run metadata, validates schemas,
and provides consistent data access across all pages.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from app_config import get_config, get_output_dir, MODULE_KEYS, MODULE_LABELS

LOGGER = logging.getLogger("app_data")


# =============================================================================
# ARTIFACT METADATA
# =============================================================================

@dataclass
class ArtifactInfo:
    """Metadata about a discovered analytical artifact."""
    module: str
    path: Path
    exists: bool
    file_name: str
    file_size_bytes: int
    modified_at: Optional[datetime]
    row_count: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    run_id: Optional[str] = None
    data_version: Optional[str] = None
    code_version: Optional[str] = None
    model_version: Optional[str] = None
    generated_at: Optional[datetime] = None
    valid: bool = False
    validation_state: str = "unknown"  # "valid", "invalid", "schema_mismatch", "empty", "unknown"
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "path": str(self.path),
            "file_name": self.file_name,
            "exists": self.exists,
            "file_size_bytes": self.file_size_bytes,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "row_count": self.row_count,
            "columns": self.columns,
            "run_id": self.run_id,
            "data_version": self.data_version,
            "code_version": self.code_version,
            "model_version": self.model_version,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "valid": self.valid,
            "validation_state": self.validation_state,
            "validation_errors": self.validation_errors,
        }


@dataclass
class ModuleStatus:
    """Aggregated status for a pipeline module."""
    module: str
    label: str
    artifacts: List[ArtifactInfo]
    overall_status: str  # "ready", "incomplete", "stale", "validation_failed", "unavailable"
    primary_artifact: Optional[ArtifactInfo] = None
    freshness_note: str = ""
    run_consistency: str = "unknown"  # "consistent", "mixed_runs", "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "label": self.label,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "overall_status": self.overall_status,
            "primary_artifact": self.primary_artifact.to_dict() if self.primary_artifact else None,
            "freshness_note": self.freshness_note,
            "run_consistency": self.run_consistency,
        }


# =============================================================================
# EXPECTED ARTIFACT CONTRACTS
# =============================================================================

EXPECTED_ARTIFACTS = {
    "data_quality": {
        "canonical_transactions": {
            "patterns": ["canonical_transactions.parquet", "canonical_transactions.csv"],
            "required_columns": ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country",
                                "transaction_type", "is_clean_sale", "gross_merchandise_revenue"],
            "min_rows": 10000,
        },
        "customer_month": {
            "patterns": ["customer_month.parquet"],
            "required_columns": ["Customer ID", "calendar_month", "orders", "net_revenue", "active"],
            "min_rows": 1000,
        },
        "validation_results": {
            "patterns": ["validation_results.json"],
            "required_columns": [],
            "min_rows": 0,
        },
    },
    "customer_360": {
        "customer_360_current": {
            "patterns": ["customer_360_current.parquet", "customer_360_current.csv", "customer_360.parquet", "customer_360.csv"],
            "required_columns": ["Customer ID", "lifetime_net_revenue", "lifetime_orders", "recency_days"],
            "min_rows": 1000,
        },
        "feature_dictionary": {
            "patterns": ["feature_dictionary.json", "feature_dictionary.csv"],
            "required_columns": [],
            "min_rows": 0,
        },
        "customer_month_events": {
            "patterns": ["customer_month_events.parquet", "customer_month_events.csv"],
            "required_columns": ["Customer ID", "calendar_month"],
            "min_rows": 1000,
        },
    },
    "segmentation": {
        "customer_segments": {
            "patterns": ["customer_segments.parquet", "customer_segments.csv"],
            "required_columns": ["Customer ID", "segment"],
            "min_rows": 1000,
        },
        "segment_profiles": {
            "patterns": ["segment_profiles.csv"],
            "required_columns": ["cluster"],
            "min_rows": 1,
        },
        "pca_coordinates": {
            "patterns": ["pca_coordinates.csv", "pca_coordinates.parquet"],
            "required_columns": ["Customer ID", "PC1", "PC2"],
            "min_rows": 1000,
        },
        "model_metadata": {
            "patterns": ["model_metadata.json", "cluster_model_candidates.csv"],
            "required_columns": [],
            "min_rows": 0,
        },
    },
    "cohorts": {
        "matrix_logo_retention": {
            "patterns": ["matrix_logo_retention.csv"],
            "required_columns": ["cohort_month"],
            "min_rows": 1,
        },
        "matrix_net_revenue_retention": {
            "patterns": ["matrix_net_revenue_retention.csv"],
            "required_columns": ["cohort_month"],
            "min_rows": 1,
        },
        "matrix_gross_revenue_retention": {
            "patterns": ["matrix_gross_revenue_retention.csv"],
            "required_columns": ["cohort_month"],
            "min_rows": 1,
        },
        "retention_decay_curve": {
            "patterns": ["retention_decay_curve.csv"],
            "required_columns": ["age_month"],
            "min_rows": 1,
        },
        "cohort_scorecard": {
            "patterns": ["cohort_scorecard.csv"],
            "required_columns": ["cohort_month"],
            "min_rows": 1,
        },
        "cohort_acquisition_quality": {
            "patterns": ["cohort_acquisition_quality.csv"],
            "required_columns": ["cohort_month"],
            "min_rows": 1,
        },
        "customer_lifecycle_status": {
            "patterns": ["customer_lifecycle_status.csv"],
            "required_columns": ["Customer ID", "cohort_month"],
            "min_rows": 1000,
        },
    },
    "clv": {
        "clv_customer_predictions": {
            "patterns": ["clv_customer_predictions.csv", "customer_clv.csv", "customer_clv.parquet"],
            "required_columns": ["Customer ID", "clv_mean", "clv_median", "clv_p10", "clv_p90"],
            "min_rows": 1000,
        },
        "model_card": {
            "patterns": ["model_card.json"],
            "required_columns": [],
            "min_rows": 0,
        },
        "validation_predictions": {
            "patterns": ["validation_predictions.csv"],
            "required_columns": ["calendar_month", "actual_purchase", "predicted_purchase_probability"],
            "min_rows": 100,
        },
        "feature_importance": {
            "patterns": ["feature_importance.csv"],
            "required_columns": ["feature", "importance"],
            "min_rows": 1,
        },
    },
    "churn": {
        "customer_churn_next_purchase": {
            "patterns": ["customer_churn_next_purchase.csv", "churn_predictions.csv"],
            "required_columns": ["Customer ID", "churn_probability", "next_purchase_30d_probability"],
            "min_rows": 1000,
        },
        "model_card": {
            "patterns": ["model_card.json"],
            "required_columns": [],
            "min_rows": 0,
        },
        "survival_feature_importance": {
            "patterns": ["survival_feature_importance.csv"],
            "required_columns": ["feature", "importance_mean"],
            "min_rows": 1,
        },
    },
    "reactivation": {
        "reactivation_predictions": {
            "patterns": ["reactivation_predictions.csv"],
            "required_columns": ["Customer ID", "reactivation_probability"],
            "min_rows": 100,
        },
        "model_card": {
            "patterns": ["model_card.json"],
            "required_columns": [],
            "min_rows": 0,
        },
    },
    "product_analytics": {
        "product_metrics": {
            "patterns": ["product_metrics.parquet", "product_metrics.csv"],
            "required_columns": ["StockCode", "total_revenue", "total_units", "unique_customers"],
            "min_rows": 100,
        },
        "co_purchase_matrix": {
            "patterns": ["co_purchase_matrix.parquet", "co_purchase_matrix.csv"],
            "required_columns": ["product_a", "product_b", "cooccurrence"],
            "min_rows": 100,
        },
        "product_role_summary": {
            "patterns": ["product_role_summary.csv"],
            "required_columns": ["product_role"],
            "min_rows": 1,
        },
        "product_analytics_card": {
            "patterns": ["product_analytics_card.json"],
            "required_columns": [],
            "min_rows": 0,
        },
    },
    "recommendations": {
        "recommendations": {
            "patterns": ["recommendations.parquet", "recommendations.csv"],
            "required_columns": ["Customer ID", "recommended_product", "score", "reason"],
            "min_rows": 100,
        },
        "model_card": {
            "patterns": ["model_card.json"],
            "required_columns": [],
            "min_rows": 0,
        },
    },
    "decision_engine": {
        "customer_decision_scores": {
            "patterns": ["customer_decision_scores.csv", "customer_decision_scores.parquet"],
            "required_columns": ["Customer ID", "recommended_action_capped", "priority_score"],
            "min_rows": 100,
        },
        "action_summary": {
            "patterns": ["action_summary.csv"],
            "required_columns": ["final_action", "customers"],
            "min_rows": 1,
        },
        "model_card": {
            "patterns": ["model_card.json"],
            "required_columns": [],
            "min_rows": 0,
        },
    },
}


# =============================================================================
# ARTIFACT REGISTRY
# =============================================================================

class ArtifactRegistry:
    """Centralized artifact discovery, validation, and loading."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = get_config(config_path)
        self.project_root = Path(__file__).resolve().parent
        self._artifact_cache: Dict[str, Dict[str, ArtifactInfo]] = {}
        self._module_status_cache: Dict[str, ModuleStatus] = {}
        self._dataframe_cache: Dict[str, pd.DataFrame] = {}
        self._run_manifest: Optional[Dict[str, Any]] = None
        self._discovered = False

    def discover(self, force_refresh: bool = False) -> Dict[str, ModuleStatus]:
        """Discover and validate all artifacts across modules."""
        if self._discovered and not force_refresh:
            return self._module_status_cache

        self._artifact_cache = {}
        self._module_status_cache = {}
        self._run_manifest = self._load_run_manifest()

        for module in MODULE_KEYS:
            artifacts = self._discover_module_artifacts(module)
            self._artifact_cache[module] = artifacts
            self._module_status_cache[module] = self._compute_module_status(module, artifacts)

        self._check_cross_module_consistency()
        self._discovered = True
        return self._module_status_cache

    def _load_run_manifest(self) -> Optional[Dict[str, Any]]:
        """Load the consolidated run manifest if available."""
        manifest_paths = [
            self.project_root / "xlsx_output" / "run_manifest.json",
            self.project_root / "data_quality_output" / "run_manifest.json",
            self.project_root / "customer_360_output" / "run_manifest.json",
        ]
        for p in manifest_paths:
            if p.exists():
                try:
                    with open(p, "r") as f:
                        return json.load(f)
                except Exception as e:
                    LOGGER.warning(f"Failed to load run manifest from {p}: {e}")
        return None

    def _discover_module_artifacts(self, module: str) -> Dict[str, ArtifactInfo]:
        """Discover artifacts for a specific module."""
        output_dir = get_output_dir(module)
        expected = EXPECTED_ARTIFACTS.get(module, {})
        artifacts = {}

        for artifact_key, spec in expected.items():
            artifact_info = self._find_and_validate_artifact(
                module, artifact_key, output_dir, spec
            )
            artifacts[artifact_key] = artifact_info

        return artifacts

    def _find_and_validate_artifact(
        self,
        module: str,
        artifact_key: str,
        output_dir: Path,
        spec: Dict[str, Any],
    ) -> ArtifactInfo:
        """Find and validate a single artifact."""
        # Find the file
        file_path = None
        for pattern in spec["patterns"]:
            candidates = list(output_dir.glob(pattern))
            if not candidates and output_dir.parent != output_dir:
                # Also search recursively one level down
                candidates = list(output_dir.glob(f"**/{pattern}"))
            if candidates:
                # Prefer most recently modified
                file_path = max(candidates, key=lambda p: p.stat().st_mtime)
                break

        if file_path is None:
            return ArtifactInfo(
                module=module,
                path=output_dir / spec["patterns"][0],
                exists=False,
                file_name=spec["patterns"][0],
                file_size_bytes=0,
                modified_at=None,
                validation_state="missing",
            )

        stat = file_path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime)
        file_size = stat.st_size

        # Load and validate
        row_count = None
        columns = []
        run_id = None
        data_version = None
        code_version = None
        model_version = None
        generated_at = None
        valid = False
        validation_state = "unknown"
        validation_errors = []

        try:
            if file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path)
                row_count = len(df)
                columns = df.columns.tolist()
            elif file_path.suffix == ".csv":
                df = pd.read_csv(file_path, nrows=1)
                columns = df.columns.tolist()
                # Get full row count efficiently
                df_full = pd.read_csv(file_path)
                row_count = len(df_full)
            elif file_path.suffix == ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # Extract metadata from model card / run manifest
                    run_id = data.get("run_id") or data.get("model_name")
                    data_version = data.get("data_version")
                    code_version = data.get("code_version")
                    model_version = data.get("model_version")
                    generated_at_str = data.get("generated_at") or data.get("timestamp")
                    if generated_at_str:
                        try:
                            generated_at = datetime.fromisoformat(generated_at_str.replace("Z", "+00:00"))
                        except Exception:
                            pass
                row_count = 1  # JSON is a single record
                columns = list(data.keys()) if isinstance(data, dict) else []

            # Validate schema
            required_cols = spec.get("required_columns", [])
            min_rows = spec.get("min_rows", 0)

            missing_cols = [c for c in required_cols if c not in columns]
            if missing_cols:
                validation_errors.append(f"Missing required columns: {missing_cols}")
                validation_state = "schema_mismatch"
            elif row_count is not None and row_count < min_rows:
                validation_errors.append(f"Row count {row_count} below minimum {min_rows}")
                validation_state = "empty" if row_count == 0 else "incomplete"
            else:
                validation_state = "valid"
                valid = True

        except Exception as e:
            validation_errors.append(f"Failed to load/validate: {e}")
            validation_state = "invalid"

        return ArtifactInfo(
            module=module,
            path=file_path,
            exists=True,
            file_name=file_path.name,
            file_size_bytes=file_size,
            modified_at=modified_at,
            row_count=row_count,
            columns=columns,
            run_id=run_id,
            data_version=data_version,
            code_version=code_version,
            model_version=model_version,
            generated_at=generated_at,
            valid=valid,
            validation_state=validation_state,
            validation_errors=validation_errors,
        )

    def _compute_module_status(self, module: str, artifacts: Dict[str, ArtifactInfo]) -> ModuleStatus:
        """Compute overall status for a module."""
        label = MODULE_LABELS.get(module, module)

        # Find primary artifact (first expected artifact)
        expected_keys = list(EXPECTED_ARTIFACTS.get(module, {}).keys())
        primary_artifact = None
        if expected_keys:
            primary_artifact = artifacts.get(expected_keys[0])

        # Determine overall status
        valid_count = sum(1 for a in artifacts.values() if a.valid)
        total_expected = len(expected_keys)
        missing_count = sum(1 for a in artifacts.values() if not a.exists)
        invalid_count = sum(1 for a in artifacts.values() if a.exists and not a.valid)

        if total_expected == 0:
            overall_status = "unavailable"
        elif valid_count == total_expected:
            overall_status = "ready"
        elif valid_count == 0 and missing_count == total_expected:
            overall_status = "unavailable"
        elif missing_count > 0 and valid_count == 0:
            overall_status = "incomplete"
        elif invalid_count > 0:
            overall_status = "validation_failed"
        elif valid_count < total_expected:
            overall_status = "incomplete"
        else:
            overall_status = "ready"

        # Freshness note
        freshness_note = ""
        if primary_artifact and primary_artifact.modified_at:
            age_days = (datetime.now() - primary_artifact.modified_at).days
            if age_days == 0:
                freshness_note = "Generated today"
            elif age_days == 1:
                freshness_note = "Generated yesterday"
            elif age_days < 7:
                freshness_note = f"Generated {age_days} days ago"
            else:
                freshness_note = f"Generated {age_days} days ago (may be stale)"

        return ModuleStatus(
            module=module,
            label=label,
            artifacts=list(artifacts.values()),
            overall_status=overall_status,
            primary_artifact=primary_artifact,
            freshness_note=freshness_note,
            run_consistency="unknown",  # Will be updated by cross-module check
        )

    def _check_cross_module_consistency(self) -> None:
        """Check if artifacts come from consistent runs."""
        run_ids = []
        for module, artifacts in self._artifact_cache.items():
            for artifact in artifacts.values():
                if artifact.run_id:
                    run_ids.append((module, artifact.run_id))

        # Group by run_id
        run_groups: Dict[str, List[str]] = {}
        for module, run_id in run_ids:
            run_groups.setdefault(run_id, []).append(module)

        # Determine consistency
        for module in self._module_status_cache:
            status = self._module_status_cache[module]
            modules_in_same_run = set()
            for run_id, modules in run_groups.items():
                if module in modules:
                    modules_in_same_run.update(modules)

            if len(run_groups) == 1:
                status.run_consistency = "consistent"
            elif len(run_groups) == 0:
                status.run_consistency = "unknown"
            elif len(modules_in_same_run) > 1:
                status.run_consistency = f"mixed_with_{len(modules_in_same_run)-1}_other"
            else:
                status.run_consistency = "isolated"

    # -------------------------------------------------------------------------
    # PUBLIC DATA ACCESS
    # -------------------------------------------------------------------------

    def get_artifact(self, module: str, artifact_key: str) -> Optional[ArtifactInfo]:
        """Get artifact info by module and key."""
        if not self._discovered:
            self.discover()
        return self._artifact_cache.get(module, {}).get(artifact_key)

    def get_module_status(self, module: str) -> Optional[ModuleStatus]:
        """Get module status."""
        if not self._discovered:
            self.discover()
        return self._module_status_cache.get(module)

    def get_all_module_statuses(self) -> Dict[str, ModuleStatus]:
        """Get all module statuses."""
        if not self._discovered:
            self.discover()
        return self._module_status_cache

    def load_dataframe(self, module: str, artifact_key: str) -> Optional[pd.DataFrame]:
        """Load a dataframe with caching."""
        cache_key = f"{module}/{artifact_key}"

        if cache_key in self._dataframe_cache:
            return self._dataframe_cache[cache_key]

        artifact = self.get_artifact(module, artifact_key)
        if not artifact or not artifact.exists:
            return None

        try:
            if artifact.path.suffix == ".parquet":
                df = pd.read_parquet(artifact.path)
            elif artifact.path.suffix == ".csv":
                df = pd.read_csv(artifact.path, low_memory=False)
            elif artifact.path.suffix == ".json":
                with open(artifact.path, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
            else:
                return None

            self._dataframe_cache[cache_key] = df
            return df
        except Exception as e:
            LOGGER.error(f"Failed to load {artifact.path}: {e}")
            return None

    def load_model_card(self, module: str) -> Optional[Dict[str, Any]]:
        """Load model card for a module."""
        artifact = self.get_artifact(module, "model_card")
        if artifact and artifact.exists:
            try:
                with open(artifact.path, "r") as f:
                    return json.load(f)
            except Exception as e:
                LOGGER.error(f"Failed to load model card for {module}: {e}")
        return None

    def get_run_id(self, module: str) -> Optional[str]:
        """Get the run ID for a module's primary artifact."""
        artifact = self.get_artifact(module, list(EXPECTED_ARTIFACTS.get(module, {}).keys())[0]) if EXPECTED_ARTIFACTS.get(module) else None
        return artifact.run_id if artifact else None

    def get_generated_at(self, module: str) -> Optional[datetime]:
        """Get the generation timestamp for a module's primary artifact."""
        artifact = self.get_artifact(module, list(EXPECTED_ARTIFACTS.get(module, {}).keys())[0]) if EXPECTED_ARTIFACTS.get(module) else None
        return artifact.generated_at if artifact else artifact.modified_at if artifact else None

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._dataframe_cache.clear()
        self._discovered = False


# =============================================================================
# CUSTOMER PROFILE ADAPTER
# =============================================================================

class CustomerProfileAdapter:
    """Unified customer profile access combining all available sources."""

    def __init__(self, registry: ArtifactRegistry):
        self.registry = registry
        self._profile_cache: Optional[pd.DataFrame] = None
        self._profile_sources: List[str] = []

    def build_profile(self, force_refresh: bool = False) -> pd.DataFrame:
        """Build the unified customer profile by merging all available sources."""
        if self._profile_cache is not None and not force_refresh:
            return self._profile_cache

        # Priority order: Customer 360 (descriptive) as base, then enrich
        base_df = self.registry.load_dataframe("customer_360", "customer_360_current")
        if base_df is None:
            base_df = self.registry.load_dataframe("customer_360", "customer_360")

        if base_df is None or base_df.empty:
            # Fallback to segmentation
            base_df = self.registry.load_dataframe("segmentation", "customer_segments")

        if base_df is None or base_df.empty:
            self._profile_cache = pd.DataFrame()
            self._profile_sources = []
            return self._profile_cache

        # Ensure Customer ID is standardized
        base_df = base_df.copy()
        id_col = self._find_id_column(base_df)
        if id_col:
            base_df["Customer ID"] = pd.to_numeric(base_df[id_col], errors="coerce").round()
            base_df = base_df.dropna(subset=["Customer ID"])
            base_df["Customer ID"] = base_df["Customer ID"].astype("Int64")

        self._profile_sources = ["customer_360"]

        # Enrichment sources in priority order
        enrichment_sources = [
            ("segmentation", "customer_segments", ["segment", "segment_name", "segment_confidence"]),
            ("clv", "clv_customer_predictions", ["clv_mean", "clv_median", "clv_p10", "clv_p90", "clv_std"]),
            ("churn", "customer_churn_next_purchase", ["churn_probability", "next_purchase_probability_30d",
                                                        "next_purchase_probability_7d", "next_purchase_probability_60d",
                                                        "survival_3m", "survival_6m", "survival_12m"]),
            ("reactivation", "reactivation_predictions", ["reactivation_probability"]),
            ("decision_engine", "customer_decision_scores", ["final_action", "priority_score",
                                                              "decision_confidence", "action_reason"]),
        ]

        for module, artifact_key, preferred_cols in enrichment_sources:
            df = self.registry.load_dataframe(module, artifact_key)
            if df is not None and not df.empty:
                id_col = self._find_id_column(df)
                if id_col:
                    df = df.copy()
                    df["Customer ID"] = pd.to_numeric(df[id_col], errors="coerce").round()
                    df = df.dropna(subset=["Customer ID"])
                    df["Customer ID"] = df["Customer ID"].astype("Int64")

                    # Keep only preferred columns that exist
                    keep_cols = ["Customer ID"] + [c for c in preferred_cols if c in df.columns]
                    if len(keep_cols) > 1:
                        df = df[keep_cols].drop_duplicates(subset=["Customer ID"], keep="first")
                        base_df = base_df.merge(df, on="Customer ID", how="left")
                        self._profile_sources.append(module)

        self._profile_cache = base_df
        return self._profile_cache

    def _find_id_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in ["Customer ID", "customer_id", "CustomerID"]:
            if col in df.columns:
                return col
        return None

    def get_customer(self, customer_id: int) -> Optional[pd.Series]:
        """Get a single customer's profile."""
        profile = self.build_profile()
        if profile.empty:
            return None
        matches = profile[profile["Customer ID"] == customer_id]
        if matches.empty:
            return None
        return matches.iloc[0]

    def get_all_customers(self) -> pd.DataFrame:
        """Get all customer profiles."""
        return self.build_profile()

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about the data sources used."""
        sources = {}
        for module in self._profile_sources:
            artifact = self.registry.get_artifact(module, list(EXPECTED_ARTIFACTS.get(module, {}).keys())[0]) if EXPECTED_ARTIFACTS.get(module) else None
            sources[module] = {
                "run_id": artifact.run_id if artifact else None,
                "generated_at": artifact.generated_at.isoformat() if artifact and artifact.generated_at else None,
                "modified_at": artifact.modified_at.isoformat() if artifact and artifact.modified_at else None,
                "row_count": artifact.row_count if artifact else None,
            }
        return sources


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_registry_instance: Optional[ArtifactRegistry] = None
_profile_adapter_instance: Optional[CustomerProfileAdapter] = None


def get_registry(config_path: Optional[Path] = None) -> ArtifactRegistry:
    """Get the artifact registry singleton."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ArtifactRegistry(config_path)
    return _registry_instance


def get_profile_adapter(registry: Optional[ArtifactRegistry] = None) -> CustomerProfileAdapter:
    """Get the customer profile adapter singleton."""
    global _profile_adapter_instance
    if _profile_adapter_instance is None:
        _profile_adapter_instance = CustomerProfileAdapter(registry or get_registry())
    return _profile_adapter_instance


def reset_registry() -> None:
    """Reset the registry singleton (mainly for testing)."""
    global _registry_instance, _profile_adapter_instance
    _registry_instance = None
    _profile_adapter_instance = None