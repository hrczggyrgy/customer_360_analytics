#!/usr/bin/env python3
"""
Data Quality Analysis — canonical ingestion, transaction classification, and reconciliation.

This script:
1. Loads raw data with Invoice preserved as string (fixes the critical ingestion bug)
2. Classifies transactions into explicit types (sale, return, cancellation, etc.)
3. Computes canonical financial measures
4. Runs comprehensive data quality validations
5. Outputs canonical transactions + data quality reports

Example:
    python data_quality.py \
        --input ./data_xslx/online_retail_II.xlsx \
        --output-dir ./data_quality_output
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import polars as pl

from retail_ds.io import load_raw_transactions, load_canonical_transactions
from retail_ds.cleaning import clean_transactions, add_calendar_fields
from retail_ds.transactions import classify_transactions, compute_financial_measures, get_transaction_type_report
from retail_ds.validation import (
    run_all_validations,
    assert_validations_pass,
    validate_invoice_string_preservation,
    validate_financial_reconciliation,
)
from retail_ds.customer_month import build_customer_month_panel
from retail_ds.config import ProjectConfig, load_config, add_config_args


SEED = 42


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("data_quality")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data quality analysis for Online Retail II.")
    add_config_args(parser)
    parser.add_argument("--input", default=None, help="Input .xlsx/.csv/.parquet file (overrides config).")
    parser.add_argument("--output-dir", default=None, help="Output directory (overrides config).")
    parser.add_argument("--sheet", default=None, help="Optional Excel sheet name.")
    parser.add_argument("--fail-on-warning", action="store_true", help="Treat validation warnings as failures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).expanduser().resolve()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(config_path, project_root)

    input_path = Path(args.input).expanduser().resolve() if args.input else config.raw_data_path
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.data_quality_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Load raw data with Invoice preserved as string
    # -------------------------------------------------------------------------
    LOGGER.info("Loading raw transactions with Invoice as string...")
    raw = load_raw_transactions(input_path, args.sheet)

    # Verify Invoice is string
    invoice_check = validate_invoice_string_preservation(raw)
    LOGGER.info(str(invoice_check))
    if not invoice_check.passed:
        raise ValueError("Invoice ingestion failed - not preserved as string")

    # -------------------------------------------------------------------------
    # 2. Clean transactions
    # -------------------------------------------------------------------------
    LOGGER.info("Cleaning transactions...")
    tx = clean_transactions(raw)
    tx = add_calendar_fields(tx)

    # -------------------------------------------------------------------------
    # 3. Classify transactions into explicit types
    # -------------------------------------------------------------------------
    LOGGER.info("Classifying transactions...")
    tx = classify_transactions(tx)

    # -------------------------------------------------------------------------
    # 4. Compute financial measures
    # -------------------------------------------------------------------------
    LOGGER.info("Computing financial measures...")
    tx = compute_financial_measures(tx)

    # -------------------------------------------------------------------------
    # 5. Run validations
    # -------------------------------------------------------------------------
    LOGGER.info("Running data quality validations...")
    # Build customer-month for cohort validation
    _, customer_month_dense, _ = build_customer_month_panel(tx)

    validation_results = run_all_validations(
        tx=tx,
        customer_month_dense=customer_month_dense,
    )

    # Also run financial reconciliation
    fin_check = validate_financial_reconciliation(tx)
    validation_results.append(fin_check)
    LOGGER.info(str(fin_check))

    # Assert all validations pass
    assert_validations_pass(validation_results, fail_on_warning=args.fail_on_warning)

    # -------------------------------------------------------------------------
    # 6. Generate data quality artifacts
    # -------------------------------------------------------------------------
    LOGGER.info("Generating data quality artifacts...")

    # Schema report
    schema_report = pl.DataFrame({
        "column": list(tx.schema.keys()),
        "dtype": [str(d) for d in tx.schema.values()],
    })
    schema_report.write_csv(output_dir / "schema_report.csv")

    # Missingness report
    missing_data = []
    for col in tx.columns:
        null_count = tx.select(pl.col(col).is_null().sum()).item()
        missing_data.append({"column": col, "null_count": null_count, "null_pct": null_count / tx.height * 100})
    missing_report = pl.DataFrame(missing_data)
    missing_report.write_csv(output_dir / "missingness_report.csv")

    # Duplicate report
    dup_rows = tx.unique().height
    duplicate_report = pl.DataFrame({
        "metric": ["total_rows", "unique_rows", "duplicate_rows", "duplicate_pct"],
        "value": [float(tx.height), float(dup_rows), float(tx.height - dup_rows), float((tx.height - dup_rows) / tx.height * 100)],
    })
    duplicate_report.write_csv(output_dir / "duplicate_report.csv")

    # Transaction type report
    type_report = get_transaction_type_report(tx)
    type_report.write_csv(output_dir / "transaction_type_report.csv")

    # Financial reconciliation report
    gross_revenue = tx.filter(pl.col("is_sale") & pl.col("is_positive_price")).select(pl.col("line_value").sum()).item()
    returns_value = tx.filter(pl.col("is_return")).select(pl.col("return_value").sum()).item()
    cancellation_value = tx.filter(pl.col("is_cancellation")).select(pl.col("cancellation_value").sum()).item()
    net_revenue = gross_revenue - returns_value - cancellation_value

    reconciliation_report = {
        "gross_merchandise_revenue": gross_revenue,
        "returns_value": returns_value,
        "cancellation_value": cancellation_value,
        "net_merchandise_revenue": net_revenue,
        "reconciliation_check": {
            "gross_matches": abs(gross_revenue - tx.select(pl.col("gross_merchandise_revenue").sum()).item()) < 0.01,
            "net_matches": abs(net_revenue - tx.select(pl.col("net_merchandise_revenue").sum()).item()) < 0.01,
        },
    }
    with open(output_dir / "reconciliation_report.json", "w") as f:
        json.dump(reconciliation_report, f, indent=2, default=str)

    # -------------------------------------------------------------------------
    # 7. Output canonical transactions
    # -------------------------------------------------------------------------
    LOGGER.info("Writing canonical transactions...")
    canonical_cols = [
        "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
        "Customer ID", "Country", "calendar_date", "calendar_month", "calendar_week",
        "year", "month", "weekday", "hour",
        "is_cancellation_invoice", "is_positive_quantity", "is_negative_quantity",
        "is_positive_price", "is_zero_or_negative_price", "line_value",
        "is_clean_sale", "is_return_or_cancellation", "gross_sale_value",
        "return_value", "return_units",
        "transaction_type", "is_sale", "is_return", "is_cancellation",
        "is_discount", "is_postage", "is_fee", "is_voucher", "is_manual_adjustment", "is_other",
        "gross_merchandise_revenue", "cancellation_value", "net_merchandise_revenue",
    ]

    # Select only existing columns
    existing_cols = [c for c in canonical_cols if c in tx.columns]
    canonical = tx.select(existing_cols)

    canonical.write_parquet(output_dir / "canonical_transactions.parquet")
    canonical.write_csv(output_dir / "canonical_transactions.csv")

    # Customer-month panel
    customer_month_dense.write_parquet(output_dir / "customer_month.parquet")

    # Validation results
    val_output = [{"check": r.check_name, "passed": r.passed, "message": r.message, "severity": r.severity, "details": r.details} for r in validation_results]
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(val_output, f, indent=2, default=str)

    import pandas as pd
    manifest = {
        "run_id": getattr(args, "run_id", None),
        "config_hash": config.config_hash,
        "data_version": config.config_hash,
        "code_version": "v1",
        "generated_at": pd.Timestamp.now().isoformat(),
        "observation_start": customer_month_dense.select(pl.col("calendar_month").min()).item(),
        "observation_end": customer_month_dense.select(pl.col("calendar_month").max()).item(),
        "files": [
            "canonical_transactions.parquet",
            "canonical_transactions.csv",
            "customer_month.parquet",
            "schema_report.csv",
            "missingness_report.csv",
            "duplicate_report.csv",
            "transaction_type_report.csv",
            "reconciliation_report.json",
            "validation_results.json",
        ],
    }
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    LOGGER.info("Data quality analysis complete. Outputs in %s", output_dir)
    LOGGER.info("Canonical transactions: %s rows", f"{canonical.height:,}")
    LOGGER.info("Gross revenue: £%.2f, Returns: £%.2f, Cancellations: £%.2f, Net: £%.2f",
                gross_revenue, returns_value, cancellation_value, net_revenue)


if __name__ == "__main__":
    main()