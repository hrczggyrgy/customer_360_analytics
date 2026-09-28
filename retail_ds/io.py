"""
IO utilities for loading raw transaction data with type safety.

Critical design decision: Invoice must be loaded as Utf8/string from the source
to preserve cancellation identifiers (C12345, etc.). Polars Excel reader infers
Invoice as Int64 by default, which silently converts C-prefixed invoices to null.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import polars as pl

SEED = 42

LOGGER = logging.getLogger("retail_ds.io")


def normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize column names to canonical schema (case-insensitive)."""
    aliases = {
        "invoiceno": "Invoice",
        "invoice": "Invoice",
        "invoice no": "Invoice",
        "stockcode": "StockCode",
        "stock code": "StockCode",
        "description": "Description",
        "quantity": "Quantity",
        "invoicedate": "InvoiceDate",
        "invoice date": "InvoiceDate",
        "unitprice": "Price",
        "price": "Price",
        "customerid": "Customer ID",
        "customer id": "Customer ID",
        "country": "Country",
    }

    rename_map = {}
    for column in df.columns:
        normalized = str(column).strip().lower().replace("_", " ")
        rename_map[column] = aliases.get(normalized, column)

    df = df.rename(rename_map)

    required = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns}")

    if "Description" not in df.columns:
        df = df.with_columns(pl.lit("").alias("Description"))
    if "Country" not in df.columns:
        df = df.with_columns(pl.lit("Unknown").alias("Country"))

    return df


def load_raw_transactions(
    path: Path,
    sheet: Optional[str] = None,
) -> pl.DataFrame:
    """
    Load raw transactions from Excel/CSV/Parquet with Invoice preserved as string.

    This is the SINGLE canonical ingestion function. All scripts must use this.
    """
    LOGGER.info("Loading %s", path)

    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        # CRITICAL: Read Excel with pandas first to preserve Invoice as string,
        # then convert to Polars. Polars read_excel infers Invoice as Int64.
        # Read ALL columns as object/string to prevent type inference issues,
        # then let cleaning layer handle proper type casting.
        import pandas as pd

        try:
            if sheet:
                pdf = pd.read_excel(path, sheet_name=sheet, dtype=str)
            else:
                # Read all sheets with string dtype
                loaded = pd.read_excel(path, sheet_name=None, dtype=str)
                if isinstance(loaded, dict):
                    frames = [f for f in loaded.values() if len(f) > 0]
                    if not frames:
                        raise ValueError("Workbook contains no non-empty sheets.")
                    pdf = pd.concat(frames, ignore_index=True)
                else:
                    pdf = loaded
        except Exception as exc:
            LOGGER.warning("Pandas Excel read failed (%s). Falling back.", exc)
            # Fallback without dtype override
            if sheet:
                pdf = pd.read_excel(path, sheet_name=sheet)
            else:
                loaded = pd.read_excel(path, sheet_name=None)
                if isinstance(loaded, dict):
                    frames = [f for f in loaded.values() if len(f) > 0]
                    pdf = pd.concat(frames, ignore_index=True)
                else:
                    pdf = loaded
            # Force all columns to string
            for col in pdf.columns:
                pdf[col] = pdf[col].astype(str)

        df = pl.from_pandas(pdf)

    elif suffix == ".csv":
        # CRITICAL: Use schema_overrides to force Invoice as Utf8
        df = pl.read_csv(
            path,
            infer_schema_length=20_000,
            try_parse_dates=True,
            ignore_errors=True,
            schema_overrides={"Invoice": pl.Utf8},
        )

    elif suffix in {".parquet", ".pq"}:
        df = pl.read_parquet(path)

    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    return normalize_columns(df)


def load_canonical_transactions(path: Path) -> pl.DataFrame:
    """Load pre-computed canonical transactions from parquet."""
    return pl.read_parquet(path)