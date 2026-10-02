"""
Configuration loader and resolver for the Retail Intelligence pipeline.

Single source of truth for all paths and model parameters.
All scripts must use this for path resolution.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

LOGGER = logging.getLogger("retail_ds.config")


@dataclass
class ProjectConfig:
    """Resolved project configuration with absolute paths."""

    # Raw data
    raw_data_path: Path

    # Output directories (absolute)
    data_quality_dir: Path
    customer_360_dir: Path
    segmentation_dir: Path
    cohorts_dir: Path
    clv_dir: Path
    churn_dir: Path
    reactivation_dir: Path
    product_analytics_dir: Path
    recommendations_dir: Path
    decision_engine_dir: Path

    # Model parameters
    seed: int = 42

    # Segmentation
    segmentation_min_customers: int = 150
    segmentation_pca_variance: float = 0.90
    segmentation_max_pca_components: int = 12
    segmentation_stability_repeats: int = 10
    segmentation_hdbscan_min_cluster_size: int = 59
    segmentation_hdbscan_min_samples: int = 59

    # CLV
    clv_horizon_months: int = 24
    clv_simulations: int = 200
    clv_annual_discount_rate: float = 0.10
    clv_margin_rate: float = 1.0
    clv_validation_months: int = 3
    clv_min_active_history_months: int = 2
    clv_margin_scenarios: list = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])

    # Churn / Next Purchase
    churn_test_months: int = 3
    churn_validation_months: int = 2
    churn_horizon_months: int = 12
    churn_bootstrap: int = 2
    churn_windows: list = field(default_factory=lambda: [1, 3, 6, 12])

    # Reactivation
    reactivation_horizon_months: int = 3
    reactivation_inactive_threshold_months: int = 3
    reactivation_validation_months: int = 2

    # Product Analytics
    product_analytics_min_sales: int = 10
    product_analytics_co_purchase_min_support: int = 3
    product_analytics_max_pairs_per_invoice: int = 50

    # Recommendations
    recommendations_top_k: int = 10
    recommendations_min_support: int = 3

    # Decision Engine
    decision_engine_capacity_total: int = 1000
    decision_engine_capacity_reactivate: int = 300
    decision_engine_capacity_accelerate: int = 400
    decision_engine_capacity_cross_sell: int = 300
    decision_engine_capacity_nurture: int = 500
    decision_engine_confidence_threshold: float = 0.5

    # Config metadata
    config_path: Path = field(default_factory=Path)
    config_hash: str = ""
    project_root: Path = field(default_factory=Path)

    def get_output_dir(self, module: str) -> Path:
        """Get output directory for a module by name."""
        module_map = {
            "data_quality": self.data_quality_dir,
            "customer_360": self.customer_360_dir,
            "segmentation": self.segmentation_dir,
            "cohorts": self.cohorts_dir,
            "clv": self.clv_dir,
            "churn": self.churn_dir,
            "reactivation": self.reactivation_dir,
            "product_analytics": self.product_analytics_dir,
            "recommendations": self.recommendations_dir,
            "decision_engine": self.decision_engine_dir,
        }
        if module not in module_map:
            raise ValueError(f"Unknown module: {module}. Valid: {list(module_map.keys())}")
        return module_map[module]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (paths as strings)."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, list):
                result[key] = value
            else:
                result[key] = value
        return result


def load_config(config_path: Path, project_root: Optional[Path] = None) -> ProjectConfig:
    """
    Load and resolve configuration from YAML file.

    All relative paths in config are resolved relative to project_root.
    If project_root is not provided, it's inferred from config_path.
    """
    config_path = Path(config_path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if project_root is None:
        # Infer project root: parent of config directory
        project_root = config_path.parent.parent

    project_root = Path(project_root).expanduser().resolve()

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Compute config hash for reproducibility tracking
    config_hash = hashlib.sha256(json.dumps(raw_config, sort_keys=True).encode()).hexdigest()[:12]

    # Resolve paths
    data_cfg = raw_config.get("data", {})
    outputs_cfg = raw_config.get("outputs", {})
    model_cfg = raw_config.get("model", {})

    def resolve_rel(rel_path: str) -> Path:
        """Resolve relative path against project root."""
        return (project_root / rel_path).expanduser().resolve()

    # Raw data
    raw_data_rel = data_cfg.get("raw", {}).get("online_retail_ii", "./data_xslx/online_retail_II.xlsx")
    raw_data_path = resolve_rel(raw_data_rel)

    # Output directories
    def get_output(key: str, default: str) -> Path:
        return resolve_rel(outputs_cfg.get(key, default))

    config = ProjectConfig(
        raw_data_path=raw_data_path,
        data_quality_dir=get_output("data_quality", "./data_quality_output"),
        customer_360_dir=get_output("customer_360", "./customer_360_output"),
        segmentation_dir=get_output("segmentation", "./online_retail_segmentation"),
        cohorts_dir=get_output("cohorts", "./cohort_analysis_output"),
        clv_dir=get_output("clv", "./clv_analysis_output"),
        churn_dir=get_output("churn", "./churn_next_purchase_output"),
        reactivation_dir=get_output("reactivation", "./reactivation_output"),
        product_analytics_dir=get_output("product_analytics", "./product_analytics_output"),
        recommendations_dir=get_output("recommendations", "./recommendation_output"),
        decision_engine_dir=get_output("decision_engine", "./decision_engine_output"),
        seed=model_cfg.get("seed", 42),
        # Segmentation
        segmentation_min_customers=model_cfg.get("segmentation", {}).get("min_customers", 150),
        segmentation_pca_variance=model_cfg.get("segmentation", {}).get("pca_variance", 0.90),
        segmentation_max_pca_components=model_cfg.get("segmentation", {}).get("max_pca_components", 12),
        segmentation_stability_repeats=model_cfg.get("segmentation", {}).get("stability_repeats", 10),
        segmentation_hdbscan_min_cluster_size=model_cfg.get("segmentation", {}).get("hdbscan_min_cluster_size", 59),
        segmentation_hdbscan_min_samples=model_cfg.get("segmentation", {}).get("hdbscan_min_samples", 59),
        # CLV
        clv_horizon_months=model_cfg.get("clv", {}).get("horizon_months", 24),
        clv_simulations=model_cfg.get("clv", {}).get("simulations", 200),
        clv_annual_discount_rate=model_cfg.get("clv", {}).get("annual_discount_rate", 0.10),
        clv_margin_rate=model_cfg.get("clv", {}).get("margin_rate", 1.0),
        clv_validation_months=model_cfg.get("clv", {}).get("validation_months", 3),
        clv_min_active_history_months=model_cfg.get("clv", {}).get("min_active_history_months", 2),
        clv_margin_scenarios=model_cfg.get("clv", {}).get("margin_scenarios", [0.1, 0.2, 0.3, 0.4]),
        # Churn
        churn_test_months=model_cfg.get("churn", {}).get("test_months", 3),
        churn_validation_months=model_cfg.get("churn", {}).get("validation_months", 2),
        churn_horizon_months=model_cfg.get("churn", {}).get("horizon_months", 12),
        churn_bootstrap=model_cfg.get("churn", {}).get("bootstrap", 2),
        churn_windows=model_cfg.get("churn", {}).get("windows", [1, 3, 6, 12]),
        # Reactivation
        reactivation_horizon_months=model_cfg.get("reactivation", {}).get("horizon_months", 3),
        reactivation_inactive_threshold_months=model_cfg.get("reactivation", {}).get("inactive_threshold_months", 3),
        reactivation_validation_months=model_cfg.get("reactivation", {}).get("validation_months", 2),
        # Product Analytics
        product_analytics_min_sales=model_cfg.get("product_analytics", {}).get("min_sales", 10),
        product_analytics_co_purchase_min_support=model_cfg.get("product_analytics", {}).get("co_purchase_min_support", 3),
        product_analytics_max_pairs_per_invoice=model_cfg.get("product_analytics", {}).get("max_pairs_per_invoice", 50),
        # Recommendations
        recommendations_top_k=model_cfg.get("recommendations", {}).get("top_k", 10),
        recommendations_min_support=model_cfg.get("recommendations", {}).get("min_support", 3),
        # Decision Engine
        decision_engine_capacity_total=model_cfg.get("decision_engine", {}).get("capacity_total", 1000),
        decision_engine_capacity_reactivate=model_cfg.get("decision_engine", {}).get("capacity_reactivate", 300),
        decision_engine_capacity_accelerate=model_cfg.get("decision_engine", {}).get("capacity_accelerate", 400),
        decision_engine_capacity_cross_sell=model_cfg.get("decision_engine", {}).get("capacity_cross_sell", 300),
        decision_engine_capacity_nurture=model_cfg.get("decision_engine", {}).get("capacity_nurture", 500),
        decision_engine_confidence_threshold=model_cfg.get("decision_engine", {}).get("confidence_threshold", 0.5),
        # Metadata
        config_path=config_path,
        config_hash=config_hash,
        project_root=project_root,
    )

    # Ensure output directories exist
    for attr_name in [
        "data_quality_dir", "customer_360_dir", "segmentation_dir", "cohorts_dir",
        "clv_dir", "churn_dir", "reactivation_dir", "product_analytics_dir",
        "recommendations_dir", "decision_engine_dir",
    ]:
        getattr(config, attr_name).mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loaded config from {config_path} (hash: {config_hash})")
    LOGGER.info(f"Project root: {project_root}")
    LOGGER.info(f"Raw data: {raw_data_path}")

    return config


def add_config_args(parser) -> None:
    """Add standard config arguments to an ArgumentParser."""
    parser.add_argument(
        "--config",
        default="config/project.yaml",
        help="Path to project configuration YAML (relative to project root or absolute).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID for lineage tracking.",
    )


def resolve_config_path(config_arg: str, project_root: Path) -> Path:
    """Resolve config path relative to project root if not absolute."""
    path = Path(config_arg)
    if not path.is_absolute():
        path = project_root / path
    return path.expanduser().resolve()


def get_project_root() -> Path:
    """Get project root from the location of this module."""
    return Path(__file__).resolve().parent.parent


# Global config cache (for singleton pattern in scripts)
_config_cache: Optional[ProjectConfig] = None


def get_config(config_path: Optional[str] = None, project_root: Optional[Path] = None) -> ProjectConfig:
    """Get or create the global config singleton."""
    global _config_cache
    if _config_cache is None:
        if config_path is None:
            config_path = "config/project.yaml"
        if project_root is None:
            project_root = get_project_root()
        resolved_path = resolve_config_path(config_path, project_root)
        _config_cache = load_config(resolved_path, project_root)
    return _config_cache


def reset_config() -> None:
    """Reset the global config cache (for testing)."""
    global _config_cache
    _config_cache = None