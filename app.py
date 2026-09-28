#!/usr/bin/env python3
"""
Retail Customer Intelligence — Streamlit app.

A clean decision-science interface for the project's analytical workflow:

    Data -> Customer 360 -> Segmentation -> Cohorts -> CLV
         -> Churn / Next Purchase -> Decision Engine

The app is intentionally resilient to partial project runs. It discovers
outputs recursively from the project directory and renders each module only
when the required data is available.

Run from the project directory:
    streamlit run app.py

Optional:
    streamlit run app.py --server.headless true

Recommended packages:
    pip install streamlit plotly pandas numpy openpyxl pyarrow
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# APP CONFIG
# =============================================================================

st.set_page_config(
    page_title="Retail Customer Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

SEED = 42
APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_DIR = APP_DIR


# =============================================================================
# DESIGN SYSTEM
# =============================================================================

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #697386;
        --line: #e4e7ec;
        --panel: #ffffff;
        --panel-soft: #f7f8fa;
        --accent: #315efb;
        --accent-soft: #eef2ff;
        --success: #218739;
        --warning: #a56600;
        --danger: #c53d32;
    }

    .stApp {
        background: #fbfcfe;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.25rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #f7f8fa;
        border-right: 1px solid var(--line);
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(23, 32, 51, 0.03);
    }

    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #f6f8ff 100%);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }

    .hero h1 {
        margin: 0 0 8px 0;
        color: var(--ink);
        font-size: 2.25rem;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }

    .hero p {
        margin: 0;
        color: var(--muted);
        max-width: 980px;
        font-size: 1rem;
        line-height: 1.6;
    }

    .section-label {
        color: var(--muted);
        text-transform: uppercase;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin: 22px 0 8px 0;
    }

    .science-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px 20px;
        height: 100%;
    }

    .science-card h4 {
        margin: 0 0 8px 0;
        color: var(--ink);
        font-size: 1rem;
    }

    .science-card p {
        margin: 0;
        color: var(--muted);
        line-height: 1.55;
        font-size: 0.92rem;
    }

    .status-chip {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.76rem;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .method-step {
        border-left: 3px solid var(--accent);
        padding: 9px 0 9px 14px;
        margin: 8px 0;
    }

    .method-step strong {
        color: var(--ink);
    }

    .small-note {
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.45;
    }

    .kicker {
        color: var(--accent);
        font-weight: 700;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .customer-header {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }

    .customer-header h2 {
        margin: 0;
        color: var(--ink);
    }

    .customer-header p {
        margin: 5px 0 0 0;
        color: var(--muted);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

FILE_PATTERNS = {
    "customer_360": [
        "customer_360.csv",
        "customer_360.parquet",
        "customer360.csv",
        "customer360.parquet",
        "*customer_360*.csv",
        "*customer_360*.parquet",
    ],
    "segments": [
        "customer_segments.csv",
        "customer_segments.parquet",
        "*customer_segments*.csv",
        "*customer_segments*.parquet",
    ],
    "segment_profiles": [
        "segment_profiles.csv",
        "segment_profiles.parquet",
        "*segment_profiles*.csv",
    ],
    "pca": [
        "pca_coordinates.csv",
        "pca_coordinates.parquet",
        "*pca_coordinates*.csv",
    ],
    "cohort_matrix_logo": [
        "matrix_logo_retention.csv",
        "*matrix_logo_retention*.csv",
    ],
    "cohort_matrix_nrr": [
        "matrix_net_revenue_retention.csv",
        "*matrix_net_revenue_retention*.csv",
    ],
    "cohort_decay": [
        "retention_decay_curve.csv",
        "*retention_decay_curve*.csv",
    ],
    "cohort_scorecard": [
        "cohort_scorecard.csv",
        "*cohort_scorecard*.csv",
    ],
    "clv": [
        "clv_customer_predictions.csv",
        "customer_clv.csv",
        "clv_predictions.csv",
        "*clv*.csv",
        "*clv*.parquet",
    ],
    "churn": [
        "churn_predictions.csv",
        "customer_churn_predictions.csv",
        "retention_predictions.csv",
        "*churn*.csv",
        "*survival*.csv",
        "*retention*.csv",
        "*churn*.parquet",
    ],
    "next_purchase": [
        "next_purchase_predictions.csv",
        "customer_next_purchase_predictions.csv",
        "*next_purchase*.csv",
        "*next*purch*.csv",
        "*next_purchase*.parquet",
    ],
    "decision": [
        "customer_decision_scores.csv",
        "customer_decision_scores.parquet",
        "*decision_scores*.csv",
        "*decision_scores*.parquet",
    ],
    "action_summary": [
        "action_summary.csv",
        "*action_summary*.csv",
    ],
    "customer_month": [
        "customer_month_events.csv",
        "*customer_month_events*.csv",
    ],
    "lifecycle": [
        "customer_lifecycle_status.csv",
        "*customer_lifecycle*.csv",
    ],
}


def normalize_key(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    return text


def humanize(value: object) -> str:
    if value is None:
        return "—"
    text = str(value)
    return text.replace("_", " ").replace("—", "—").title()


def money(value: object, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{max(1, decimals)}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.{max(1, decimals)}f}K"
    return f"{value:,.{decimals}f}"


def pct(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) <= 1.5:
        value *= 100
    return f"{value:.{decimals}f}%"


def number(value: object, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def safe_series(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[pd.Series]:
    columns = {normalize_key(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in columns:
            return pd.to_numeric(
                df[columns[key]],
                errors="coerce",
            )
    return None


def find_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    columns = {normalize_key(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in columns:
            return columns[key]
    return None


def value_at_row(
    row: pd.Series,
    candidates: Sequence[str],
):
    columns = {normalize_key(c): c for c in row.index}
    for candidate in candidates:
        key = normalize_key(candidate)
        if key in columns:
            return row[columns[key]]
    return None


def to_numeric_series(
    df: pd.DataFrame,
    column: Optional[str],
) -> pd.Series:
    if column is None:
        return pd.Series(
            np.nan,
            index=df.index,
        )
    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def unique_sorted_values(
    df: pd.DataFrame,
    column: str,
) -> List:
    values = df[column].dropna().unique().tolist()
    try:
        return sorted(values)
    except Exception:
        return sorted(
            values,
            key=lambda x: str(x),
        )


# =============================================================================
# FILE DISCOVERY + LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def discover_files(
    root: str,
) -> Dict[str, Optional[str]]:
    base = Path(root).expanduser().resolve()

    if not base.exists():
        return {
            key: None
            for key in FILE_PATTERNS
        }

    result: Dict[str, Optional[str]] = {}

    for key, patterns in FILE_PATTERNS.items():
        candidates: List[Path] = []

        for pattern in patterns:
            candidates.extend(
                base.glob(pattern)
            )
            if "*" in pattern and base != base.parent:
                candidates.extend(
                    base.glob(f"**/{pattern}")
                )

        unique = sorted(
            set(
                p.resolve()
                for p in candidates
                if p.is_file()
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        result[key] = (
            str(unique[0])
            if unique
            else None
        )

    return result


@st.cache_data(show_spinner=False)
def load_table(
    path_str: str,
    mtime_ns: int,
) -> pd.DataFrame:
    path = Path(path_str)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path,
            low_memory=False,
        )

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(
            path
        )

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(
            path,
            sheet_name=0,
        )

    raise ValueError(
        f"Unsupported file type: {path}"
    )


def get_table(
    files: Dict[str, Optional[str]],
    key: str,
) -> Optional[pd.DataFrame]:
    path_str = files.get(key)
    if not path_str:
        return None

    path = Path(path_str)

    try:
        return load_table(
            path_str,
            path.stat().st_mtime_ns,
        )
    except Exception as exc:
        st.warning(
            f"Could not read {path.name}: {exc}"
        )
        return None


def file_label(
    files: Dict[str, Optional[str]],
    key: str,
) -> str:
    path_str = files.get(key)
    if not path_str:
        return "Not found"
    return Path(path_str).name


# =============================================================================
# DATASET ADAPTERS
# =============================================================================

@st.cache_data(show_spinner=False)
def combine_customer_sources(
    root: str,
    files_json: str,
) -> pd.DataFrame:
    files = json_to_files(files_json)

    sources = []

    for key in [
        "customer_360",
        "segments",
        "clv",
        "churn",
        "next_purchase",
        "decision",
        "lifecycle",
    ]:
        path_str = files.get(key)
        if not path_str:
            continue

        try:
            path = Path(path_str)
            df = load_table(
                path_str,
                path.stat().st_mtime_ns,
            )
        except Exception:
            continue

        customer_col = find_column(
            df,
            [
                "Customer ID",
                "customer_id",
                "CustomerID",
            ],
        )
        if customer_col is None:
            continue

        df = df.copy()
        df["Customer ID"] = pd.to_numeric(
            df[customer_col],
            errors="coerce",
        ).round()
        df = df.dropna(
            subset=["Customer ID"]
        )
        df["Customer ID"] = df[
            "Customer ID"
        ].astype("Int64")

        # Keep first row/customer and suffix only when needed.
        df = (
            df.sort_index()
            .drop_duplicates(
                subset=["Customer ID"],
                keep="first",
            )
        )

        if not sources:
            sources.append(
                (key, df)
            )
            continue

        base = sources[0][1]
        overlap = {
            c
            for c in df.columns
            if c != "Customer ID"
            and c in base.columns
        }

        keep = [
            "Customer ID"
        ] + [
            c
            for c in df.columns
            if c != "Customer ID"
            and c not in overlap
        ]

        if len(keep) > 1:
            sources.append(
                (key, df[keep])
            )

    if not sources:
        return pd.DataFrame()

    merged = sources[0][1].copy()

    for _, df in sources[1:]:
        merged = merged.merge(
            df,
            on="Customer ID",
            how="outer",
        )

    return merged


def json_to_files(
    files_json: str,
) -> Dict[str, Optional[str]]:
    import json
    parsed = json.loads(files_json)
    return parsed


def customer_source(
    files: Dict[str, Optional[str]],
) -> Tuple[Optional[pd.DataFrame], str]:
    priority = [
        ("decision", "Decision Engine"),
        ("customer_360", "Customer 360"),
        ("segments", "Segmentation"),
    ]

    for key, label in priority:
        df = get_table(
            files,
            key,
        )
        if df is not None:
            customer_col = find_column(
                df,
                [
                    "Customer ID",
                    "customer_id",
                    "CustomerID",
                ],
            )
            if customer_col:
                df = df.copy()
                df["Customer ID"] = pd.to_numeric(
                    df[customer_col],
                    errors="coerce",
                ).round()
                df = df.dropna(
                    subset=["Customer ID"]
                )
                df["Customer ID"] = df[
                    "Customer ID"
                ].astype("Int64")
                return df, label

    return None, "None"


# =============================================================================
# CHART HELPERS
# =============================================================================

PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
    ],
}


def base_layout(
    fig: go.Figure,
    height: int = 420,
    title: Optional[str] = None,
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(
            l=10,
            r=10,
            t=50 if title else 20,
            b=10,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, ui-sans-serif, system-ui, sans-serif",
            color="#172033",
        ),
        title=(
            dict(
                text=title,
                x=0,
                xanchor="left",
                font=dict(
                    size=17,
                    color="#172033",
                ),
            )
            if title
            else None
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )
    return fig


def plot_missing(
    message: str = "Required output is not available yet.",
) -> None:
    st.info(
        message
    )


def segment_column(
    df: pd.DataFrame,
) -> Optional[str]:
    return find_column(
        df,
        [
            "segment_name",
            "segment",
            "cluster",
        ],
    )


def clv_column(
    df: pd.DataFrame,
) -> Optional[str]:
    return find_column(
        df,
        [
            "clv",
            "predicted_clv",
            "customer_clv",
            "clv_mean",
            "observed_clv",
        ],
    )


def churn_column(
    df: pd.DataFrame,
) -> Optional[str]:
    return find_column(
        df,
        [
            "churn_probability",
            "churn_prob",
            "prob_churn",
        ],
    )


def next_purchase_column(
    df: pd.DataFrame,
) -> Optional[str]:
    return find_column(
        df,
        [
            "next_purchase_probability",
            "next_purchase_probability_30d",
            "purchase_probability_30d",
        ],
    )


def revenue_column(
    df: pd.DataFrame,
) -> Optional[str]:
    return find_column(
        df,
        [
            "net_revenue",
            "revenue",
            "gross_revenue",
        ],
    )


def retention_matrix_from_file(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    cohort_col = find_column(
        df,
        [
            "cohort_month",
            "cohort",
        ],
    )

    if cohort_col:
        df = df.set_index(
            cohort_col
        )

    # If already wide, use numeric columns.
    numeric_cols = []
    for col in df.columns:
        if str(col).lower().startswith("age"):
            numeric_cols.append(col)
        elif str(col).isdigit():
            numeric_cols.append(col)

    if numeric_cols:
        out = df[numeric_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )
        out.columns = [
            int(str(c).replace("age_", ""))
            if str(c).replace("age_", "").isdigit()
            else c
            for c in out.columns
        ]
        return out

    return df.select_dtypes(
        include=["number"]
    )


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown(
    "### Retail Customer Intelligence"
)

project_dir = st.sidebar.text_input(
    "Project data directory",
    value=str(DEFAULT_PROJECT_DIR),
    help=(
        "Point this to the directory containing the outputs generated by your "
        "Python scripts."
    ),
)

if st.sidebar.button(
    "Refresh project data",
    use_container_width=True,
):
    st.cache_data.clear()
    st.rerun()

files = discover_files(
    project_dir
)

available_count = sum(
    bool(v)
    for v in files.values()
)

st.sidebar.markdown(
    f"**Detected analytical outputs:** {available_count}"
)

with st.sidebar.expander(
    "Pipeline status",
    expanded=False,
):
    pipeline_labels = [
        ("customer_360", "Customer 360"),
        ("segments", "Segmentation"),
        ("segment_profiles", "Segment profiles"),
        ("cohort_matrix_logo", "Cohort retention"),
        ("clv", "CLV"),
        ("churn", "Churn / survival"),
        ("next_purchase", "Next purchase"),
        ("decision", "Decision engine"),
    ]

    for key, label in pipeline_labels:
        status = "Ready" if files.get(key) else "Pending"
        chip = (
            "<span class='status-chip'>"
            + f"{label}: {status}"
            + "</span>"
        )
        st.markdown(
            chip,
            unsafe_allow_html=True,
        )

pages = [
    "Executive",
    "Customer 360",
    "Segmentation",
    "Cohorts",
    "CLV",
    "Retention & Next Purchase",
    "Decision Engine",
    "Methodology",
]

page = st.sidebar.radio(
    "Navigate",
    pages,
)

st.sidebar.markdown(
    "---"
)

st.sidebar.caption(
    "Observational retail analytics. Decision scores are not causal uplift estimates."
)


# =============================================================================
# HERO
# =============================================================================

hero_copy = {
    "Executive": (
        "Executive view",
        "A decision-oriented summary of customer economics, retention, risk, and opportunity."
    ),
    "Customer 360": (
        "Customer 360",
        "Move from portfolio-level metrics to an individual customer and inspect the evidence behind the models."
    ),
    "Segmentation": (
        "Behavioral segmentation",
        "A multi-dimensional view of customer behavior beyond conventional RFM."
    ),
    "Cohorts": (
        "Cohort intelligence",
        "Read acquisition quality through retention, revenue persistence, and reactivation over customer age."
    ),
    "CLV": (
        "Customer lifetime value",
        "Inspect value distributions, uncertainty, and the economic concentration of the customer base."
    ),
    "Retention & Next Purchase": (
        "Retention and next purchase",
        "Translate customer behavior into forward-looking probability signals with clear model diagnostics."
    ),
    "Decision Engine": (
        "Commercial decision engine",
        "Combine value, risk, propensity, and behavioral context into a transparent next-best-action policy."
    ),
    "Methodology": (
        "The science",
        "Understand how the analytical layers connect, what each model is answering, and where causal claims stop."
    ),
}

kicker, description = hero_copy[page]

st.markdown(
    f"""
    <div class='hero'>
        <div class='kicker'>{kicker}</div>
        <h1>Retail Customer Intelligence</h1>
        <p>{description}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# EXECUTIVE
# =============================================================================

if page == "Executive":

    combined, combined_source = customer_source(
        files
    )

    segments = get_table(
        files,
        "segments",
    )

    segment_profiles = get_table(
        files,
        "segment_profiles",
    )

    cohort_decay = get_table(
        files,
        "cohort_decay",
    )

    churn = get_table(
        files,
        "churn",
    )

    next_purchase = get_table(
        files,
        "next_purchase",
    )

    action_summary = get_table(
        files,
        "action_summary",
    )

    # KPI extraction.
    n_customers = (
        len(combined)
        if combined is not None
        else None
    )

    clv_total = None
    if combined is not None:
        ccol = clv_column(combined)
        if ccol:
            clv_total = pd.to_numeric(
                combined[ccol],
                errors="coerce",
            ).sum()

    if clv_total is None and segments is not None:
        ccol = clv_column(segments)
        if ccol:
            clv_total = pd.to_numeric(
                segments[ccol],
                errors="coerce",
            ).sum()

    churn_mean = None
    if combined is not None:
        col = churn_column(combined)
        if col:
            churn_mean = pd.to_numeric(
                combined[col],
                errors="coerce",
            ).mean()
    if churn_mean is None and churn is not None:
        col = churn_column(churn)
        if col:
            churn_mean = pd.to_numeric(
                churn[col],
                errors="coerce",
            ).mean()

    np_mean = None
    if combined is not None:
        col = next_purchase_column(combined)
        if col:
            np_mean = pd.to_numeric(
                combined[col],
                errors="coerce",
            ).mean()
    if np_mean is None and next_purchase is not None:
        col = next_purchase_column(next_purchase)
        if col:
            np_mean = pd.to_numeric(
                next_purchase[col],
                errors="coerce",
            ).mean()

    retention_3 = None
    if cohort_decay is not None:
        age_col = find_column(
            cohort_decay,
            ["age_month"],
        )
        ret_col = find_column(
            cohort_decay,
            ["weighted_logo_retention", "logo_retention"],
        )
        if age_col and ret_col:
            tmp = cohort_decay.copy()
            ages = pd.to_numeric(
                tmp[age_col],
                errors="coerce",
            )
            vals = pd.to_numeric(
                tmp[ret_col],
                errors="coerce",
            )
            match = tmp.loc[
                ages.eq(3),
                ret_col,
            ]
            if not match.empty:
                retention_3 = float(
                    pd.to_numeric(
                        match,
                        errors="coerce",
                    ).iloc[0]
                )

    kpi_cols = st.columns(5)

    with kpi_cols[0]:
        st.metric(
            "Customers",
            number(n_customers),
        )

    with kpi_cols[1]:
        st.metric(
            "Portfolio CLV",
            money(clv_total),
        )

    with kpi_cols[2]:
        st.metric(
            "Mean churn risk",
            pct(churn_mean),
        )

    with kpi_cols[3]:
        st.metric(
            "Mean next-purchase probability",
            pct(np_mean),
        )

    with kpi_cols[4]:
        st.metric(
            "Month-3 retention",
            pct(retention_3),
        )

    st.markdown(
        "<div class='section-label'>Portfolio signals</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(
        [1.05, 0.95]
    )

    with left:
        st.markdown(
            "#### Customer value concentration"
        )

        if combined is not None:
            ccol = clv_column(combined)
            if ccol:
                plot_df = combined[[
                    "Customer ID",
                    ccol,
                ]].copy()
                plot_df[ccol] = pd.to_numeric(
                    plot_df[ccol],
                    errors="coerce",
                )
                plot_df = plot_df.dropna(
                    subset=[ccol]
                )

                if not plot_df.empty:
                    plot_df["rank"] = plot_df[ccol].rank(
                        pct=True
                    )
                    plot_df = plot_df.sort_values(
                        ccol,
                        ascending=False,
                    )
                    total_positive_clv = max(
                        float(plot_df[ccol].clip(lower=0).sum()),
                        1e-9,
                    )
                    plot_df["cumulative_clv_share"] = (
                        plot_df[ccol].clip(lower=0).cumsum()
                        / total_positive_clv
                    )

                    sample = plot_df.iloc[
                        ::max(1, len(plot_df) // 1000)
                    ].copy()

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=sample["rank"],
                            y=sample["cumulative_clv_share"],
                            mode="lines",
                            line=dict(width=2),
                            name="Cumulative CLV share",
                            hovertemplate=(
                                "Customer percentile: %{x:.0%}<br>"
                                "Cumulative CLV share: %{y:.0%}<extra></extra>"
                            ),
                        )
                    )
                    fig.add_hline(
                        y=0.50,
                        line_dash="dot",
                        annotation_text="50% of value",
                    )
                    fig.update_xaxes(
                        tickformat=".0%",
                        title="Customer percentile by CLV",
                    )
                    fig.update_yaxes(
                        tickformat=".0%",
                        title="Cumulative CLV share",
                    )
                    st.plotly_chart(
                        base_layout(
                            fig,
                            title="How concentrated is customer value?",
                        ),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )
                else:
                    plot_missing("CLV values are empty.")
            else:
                plot_missing("A CLV field was not detected.")
        else:
            plot_missing(
                "Run customer_360.py or clv_analysis.py to populate the executive value view."
            )

    with right:
        st.markdown(
            "#### Decision allocation"
        )

        if action_summary is not None:
            action_col = find_column(
                action_summary,
                [
                    "final_action",
                    "recommended_action",
                    "action",
                ],
            )
            customer_col = find_column(
                action_summary,
                [
                    "customers",
                    "customer_count",
                ],
            )

            if action_col and customer_col:
                plot_df = action_summary[[
                    action_col,
                    customer_col,
                ]].copy()
                plot_df[customer_col] = pd.to_numeric(
                    plot_df[customer_col],
                    errors="coerce",
                )
                plot_df = plot_df.sort_values(
                    customer_col,
                    ascending=True,
                )

                fig = px.bar(
                    plot_df,
                    x=customer_col,
                    y=action_col,
                    orientation="h",
                    labels={
                        customer_col: "Customers",
                        action_col: "Action",
                    },
                )
                st.plotly_chart(
                    base_layout(
                        fig,
                        title="Customers by recommended action",
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            else:
                plot_missing(
                    "Action summary columns were not recognized."
                )
        else:
            plot_missing(
                "Run decision_engine.py to populate the commercial action layer."
            )

    st.markdown(
        "<div class='section-label'>What the portfolio is answering</div>",
        unsafe_allow_html=True,
    )

    cards = st.columns(4)
    card_data = [
        (
            "Who behaves differently?",
            "Behavioral segmentation compresses customer heterogeneity into interpretable behavioral states rather than relying only on recency, frequency, and monetary value.",
        ),
        (
            "Which cohorts persist?",
            "Cohort analysis separates acquisition quality from customer age and shows whether revenue and customer activity decay over time.",
        ),
        (
            "Who is worth retaining?",
            "CLV combines expected future economics with uncertainty, allowing value to be discussed as a distribution rather than a single deterministic number.",
        ),
        (
            "What should happen next?",
            "Retention, purchase propensity, behavioral state, and value are combined in the decision layer. The policy is observational until experimentally validated.",
        ),
    ]

    for col, (title, body) in zip(
        cards,
        card_data,
    ):
        with col:
            st.markdown(
                f"""
                <div class='science-card'>
                    <h4>{title}</h4>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =============================================================================
# CUSTOMER 360
# =============================================================================

elif page == "Customer 360":

    customer_df, customer_source_name = customer_source(
        files
    )

    monthly = get_table(
        files,
        "customer_month",
    )

    if customer_df is None or customer_df.empty:
        st.warning(
            "No customer-level table was detected. Run customer_360.py first."
        )
        st.stop()

    id_col = "Customer ID"

    st.markdown(
        f"Data source: **{customer_source_name}** — `{len(customer_df):,}` customers",
    )

    ids = customer_df[id_col].dropna().astype(int).tolist()

    col1, col2 = st.columns(
        [0.55, 0.45]
    )

    with col1:
        selected_id = st.selectbox(
            "Customer",
            options=ids,
            format_func=lambda x: f"Customer {x}",
        )

    with col2:
        search = st.text_input(
            "Or search customer ID",
            value="",
            placeholder="e.g. 12345",
        )

        if search.strip().isdigit():
            search_id = int(search.strip())
            if search_id in set(ids):
                selected_id = search_id

    row = customer_df[
        customer_df[id_col].astype(int)
        == int(selected_id)
    ]

    if row.empty:
        st.warning("Customer not found.")
        st.stop()

    row = row.iloc[0]

    segment_value = value_at_row(
        row,
        ["segment_name", "segment", "cluster"],
    )
    clv_value = value_at_row(
        row,
        ["clv", "predicted_clv", "customer_clv", "clv_mean"],
    )
    churn_value = value_at_row(
        row,
        ["churn_probability", "churn_prob", "prob_churn"],
    )
    purchase_value = value_at_row(
        row,
        [
            "next_purchase_probability",
            "next_purchase_probability_30d",
            "purchase_probability_30d",
        ],
    )
    action_value = value_at_row(
        row,
        [
            "final_action",
            "recommended_action",
        ],
    )

    st.markdown(
        f"""
        <div class='customer-header'>
            <div class='kicker'>Customer profile</div>
            <h2>Customer {int(selected_id)}</h2>
            <p>{humanize(segment_value)} · {humanize(action_value) if action_value is not None else 'No action policy available'}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(5)

    metric_map = [
        ("CLV", money(clv_value)),
        ("Churn risk", pct(churn_value)),
        ("Next purchase", pct(purchase_value)),
        (
            "Revenue",
            money(
                value_at_row(
                    row,
                    [
                        "net_revenue",
                        "revenue",
                    ],
                )
            ),
        ),
        (
            "Orders",
            number(
                value_at_row(
                    row,
                    [
                        "orders",
                        "invoice_count",
                    ],
                )
            ),
        ),
    ]

    for col, (label, value) in zip(
        cards,
        metric_map,
    ):
        with col:
            st.metric(
                label,
                value,
            )

    left, right = st.columns(
        [1.2, 0.8]
    )

    with left:
        st.markdown(
            "#### Customer purchase trajectory"
        )

        if monthly is not None:
            m_id_col = find_column(
                monthly,
                ["Customer ID", "customer_id"],
            )
            month_col = find_column(
                monthly,
                ["calendar_month", "month", "date"],
            )
            revenue_col = find_column(
                monthly,
                [
                    "net_revenue",
                    "gross_revenue",
                    "revenue",
                ],
            )
            order_col = find_column(
                monthly,
                ["orders", "invoice_count"],
            )

            if m_id_col and month_col and revenue_col:
                hist = monthly[
                    pd.to_numeric(
                        monthly[m_id_col],
                        errors="coerce",
                    ).round()
                    == int(selected_id)
                ].copy()

                if not hist.empty:
                    hist[month_col] = pd.to_datetime(
                        hist[month_col],
                        errors="coerce",
                    )
                    hist[revenue_col] = pd.to_numeric(
                        hist[revenue_col],
                        errors="coerce",
                    )
                    hist = hist.dropna(
                        subset=[month_col]
                    ).sort_values(
                        month_col
                    )

                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=hist[month_col],
                            y=hist[revenue_col],
                            name="Net revenue",
                            hovertemplate=(
                                "%{x|%Y-%m}<br>"
                                "Net revenue: %{y:,.2f}<extra></extra>"
                            ),
                        )
                    )

                    if order_col:
                        orders = pd.to_numeric(
                            hist[order_col],
                            errors="coerce",
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=hist[month_col],
                                y=orders,
                                mode="lines+markers",
                                name="Orders",
                                yaxis="y2",
                            )
                        )
                        fig.update_layout(
                            yaxis2=dict(
                                title="Orders",
                                overlaying="y",
                                side="right",
                                showgrid=False,
                            )
                        )

                    fig.update_xaxes(
                        title="Calendar month"
                    )
                    fig.update_yaxes(
                        title="Net revenue"
                    )

                    st.plotly_chart(
                        base_layout(
                            fig,
                            title="Observed monthly customer economics",
                            height=460,
                        ),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )
                else:
                    plot_missing(
                        "No monthly event history was found for this customer."
                    )
            else:
                plot_missing(
                    "Customer-month columns were not recognized."
                )
        else:
            plot_missing(
                "Run cohort_analysis.py to populate monthly customer history."
            )

    with right:
        st.markdown(
            "#### Model evidence"
        )

        evidence_rows = []

        field_map = [
            ("Behavioral segment", ["segment_name", "segment"]),
            ("CLV", ["clv", "predicted_clv", "clv_mean"]),
            ("CLV lower", ["clv_lower"]),
            ("CLV upper", ["clv_upper"]),
            ("Churn probability", ["churn_probability", "churn_prob"]),
            ("Next-purchase probability", ["next_purchase_probability", "next_purchase_probability_30d"]),
            ("Reactivation probability", ["reactivation_probability"]),
            ("Expected days to next purchase", ["expected_days_to_next_purchase"]),
            ("Recency pressure", ["recency_pressure"]),
            ("Decision confidence", ["decision_confidence"]),
        ]

        for label, candidates in field_map:
            value = value_at_row(
                row,
                candidates,
            )
            if value is None or pd.isna(value):
                continue

            if "probability" in label.lower() or "confidence" in label.lower():
                display = pct(value)
            elif "clv" in label.lower():
                display = money(value)
            elif "days" in label.lower():
                display = number(value, 1)
            else:
                display = humanize(value)

            evidence_rows.append(
                {
                    "Metric": label,
                    "Value": display,
                }
            )

        if evidence_rows:
            st.dataframe(
                pd.DataFrame(evidence_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            """
            <div class='science-card'>
                <h4>How to read this customer</h4>
                <p>
                The dashboard intentionally separates <strong>value</strong>,
                <strong>risk</strong>, and <strong>propensity</strong>.
                A high-CLV customer is not automatically a good intervention target;
                the action layer looks for a commercially meaningful signal and a
                sufficiently strong model-confidence context.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# SEGMENTATION
# =============================================================================

elif page == "Segmentation":

    segments = get_table(
        files,
        "segments",
    )
    profiles = get_table(
        files,
        "segment_profiles",
    )
    pca = get_table(
        files,
        "pca",
    )

    if segments is None and profiles is None and pca is None:
        st.warning(
            "No segmentation outputs found. Run customer_segmentation.py first."
        )
        st.stop()

    if segments is not None:
        seg_col = segment_column(
            segments
        )
        if seg_col:
            seg_summary = (
                segments.groupby(
                    seg_col,
                    dropna=False,
                )
                .size()
                .reset_index(
                    name="customers"
                )
            )

            seg_cols = st.columns(4)
            with seg_cols[0]:
                st.metric(
                    "Customers",
                    number(len(segments)),
                )
            with seg_cols[1]:
                st.metric(
                    "Segments",
                    number(
                        seg_summary[
                            seg_col
                        ].nunique()
                    ),
                )
            with seg_cols[2]:
                st.metric(
                    "Largest segment",
                    number(
                        seg_summary["customers"].max()
                    ),
                )
            with seg_cols[3]:
                noise_mask = (
                    segments[seg_col]
                    .astype(str)
                    .str.contains(
                        "noise",
                        case=False,
                        na=False,
                    )
                )
                st.metric(
                    "Noise / low-density",
                    pct(
                        noise_mask.mean()
                    ),
                )

            st.markdown(
                "#### Segment size"
            )
            seg_summary = seg_summary.sort_values(
                "customers",
                ascending=True,
            )
            fig = px.bar(
                seg_summary,
                x="customers",
                y=seg_col,
                orientation="h",
                labels={
                    "customers": "Customers",
                    seg_col: "Segment",
                },
            )
            st.plotly_chart(
                base_layout(
                    fig,
                    title="Customer distribution across behavioral segments",
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

    left, right = st.columns(
        [1.1, 0.9]
    )

    with left:
        st.markdown(
            "#### Latent behavioral map"
        )

        if pca is not None:
            pc1 = find_column(
                pca,
                ["PC1", "pc1"],
            )
            pc2 = find_column(
                pca,
                ["PC2", "pc2"],
            )
            ps = segment_column(
                pca
            )

            if pc1 and pc2:
                plot_df = pca.copy()
                if len(plot_df) > 12_000:
                    plot_df = plot_df.sample(
                        12_000,
                        random_state=SEED,
                    )

                fig = px.scatter(
                    plot_df,
                    x=pc1,
                    y=pc2,
                    color=ps if ps else None,
                    hover_data=[
                        c
                        for c in [
                            "Customer ID",
                            ps,
                            "segment_confidence",
                        ]
                        if c and c in plot_df.columns
                    ],
                    opacity=0.55,
                    labels={
                        pc1: "PC1",
                        pc2: "PC2",
                    },
                )
                st.plotly_chart(
                    base_layout(
                        fig,
                        title="Customer behavioral space after PCA",
                        height=560,
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            else:
                plot_missing(
                    "PCA output was found but PC1 / PC2 were not detected."
                )
        else:
            plot_missing(
                "PCA coordinates are not available."
            )

    with right:
        st.markdown(
            "#### Behavioral views"
        )

        if profiles is not None:
            p = profiles.copy()
            seg = find_column(
                p,
                ["cluster", "segment"],
            )
            exclude = {
                seg,
                "customers",
                "confidence_mean",
                "share",
            }
            numeric = p.select_dtypes(
                include=["number"]
            ).columns.tolist()
            metric_cols = [
                c for c in numeric
                if c not in exclude
            ]

            if seg and metric_cols:
                choices = metric_cols
                selected = st.multiselect(
                    "Behavioral dimensions",
                    choices,
                    default=choices[:min(8, len(choices))],
                )

                if selected:
                    heat = p.set_index(
                        seg
                    )[selected].apply(
                        pd.to_numeric,
                        errors="coerce",
                    )
                    heat = heat.replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )

                    fig = px.imshow(
                        heat,
                        aspect="auto",
                        labels={
                            "x": "Behavioral feature",
                            "y": "Segment",
                            "color": "Score",
                        },
                    )
                    st.plotly_chart(
                        base_layout(
                            fig,
                            title="Segment behavioral profile",
                            height=560,
                        ),
                        use_container_width=True,
                        config=PLOTLY_CONFIG,
                    )

            st.caption(
                "The feature blocks are balanced before PCA so that economic intensity, cadence, assortment, price, temporal behavior, lifecycle, and returns each contribute to the latent space rather than letting one block dominate."
            )
        else:
            plot_missing(
                "segment_profiles.csv is not available."
            )

    if profiles is not None:
        st.markdown(
            "#### Segment profile table"
        )
        st.dataframe(
            profiles,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# COHORTS
# =============================================================================

elif page == "Cohorts":

    logo_file = get_table(
        files,
        "cohort_matrix_logo",
    )
    nrr_file = get_table(
        files,
        "cohort_matrix_nrr",
    )
    decay = get_table(
        files,
        "cohort_decay",
    )
    scorecard = get_table(
        files,
        "cohort_scorecard",
    )

    if all(
        x is None
        for x in [
            logo_file,
            nrr_file,
            decay,
            scorecard,
        ]
    ):
        st.warning(
            "No cohort outputs found. Run cohort_analysis.py first."
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if scorecard is not None:
            n_col = find_column(
                scorecard,
                ["cohort_month"],
            )
            st.metric(
                "Acquisition cohorts",
                number(
                    scorecard[n_col].nunique()
                    if n_col else len(scorecard)
                ),
            )
        else:
            st.metric(
                "Acquisition cohorts",
                "—",
            )

    with c2:
        st.metric(
            "Month-3 logo retention",
            "See curve",
        )

    with c3:
        st.metric(
            "Month-6 revenue retention",
            "See curve",
        )

    with c4:
        if decay is not None:
            age_col = find_column(
                decay,
                ["age_month"],
            )
            st.metric(
                "Observed age horizon",
                number(
                    pd.to_numeric(
                        decay[age_col],
                        errors="coerce",
                    ).max()
                    if age_col
                    else None
                ),
            )
        else:
            st.metric(
                "Observed age horizon",
                "—",
            )

    tabs = st.tabs(
        [
            "Logo retention",
            "Revenue retention",
            "Maturity curve",
            "Cohort scorecard",
        ]
    )

    with tabs[0]:
        if logo_file is not None:
            heat = retention_matrix_from_file(
                logo_file
            )
            heat = heat.iloc[:, :36]
            fig = px.imshow(
                heat,
                aspect="auto",
                zmin=0,
                zmax=1,
                labels={
                    "x": "Months since acquisition",
                    "y": "Acquisition cohort",
                    "color": "Retention",
                },
                color_continuous_scale="Blues",
            )
            st.plotly_chart(
                base_layout(
                    fig,
                    title="Logo retention cohort matrix",
                    height=max(
                        500,
                        220 + 24 * min(18, len(heat)),
                    ),
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            plot_missing()

    with tabs[1]:
        if nrr_file is not None:
            heat = retention_matrix_from_file(
                nrr_file
            )
            heat = heat.iloc[:, :36]
            fig = px.imshow(
                heat,
                aspect="auto",
                labels={
                    "x": "Months since acquisition",
                    "y": "Acquisition cohort",
                    "color": "Net revenue retention",
                },
                color_continuous_scale="Viridis",
            )
            st.plotly_chart(
                base_layout(
                    fig,
                    title="Net revenue retention cohort matrix",
                    height=max(
                        500,
                        220 + 24 * min(18, len(heat)),
                    ),
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            plot_missing()

    with tabs[2]:
        if decay is not None:
            age = find_column(
                decay,
                ["age_month"],
            )
            logo = find_column(
                decay,
                ["weighted_logo_retention", "logo_retention"],
            )
            nrr = find_column(
                decay,
                [
                    "weighted_net_revenue_retention",
                    "net_revenue_retention",
                ],
            )

            if age:
                fig = go.Figure()
                if logo:
                    fig.add_trace(
                        go.Scatter(
                            x=decay[age],
                            y=decay[logo],
                            mode="lines+markers",
                            name="Logo retention",
                        )
                    )
                if nrr:
                    fig.add_trace(
                        go.Scatter(
                            x=decay[age],
                            y=decay[nrr],
                            mode="lines+markers",
                            name="Net revenue retention",
                        )
                    )
                fig.update_yaxes(
                    tickformat=".0%",
                    title="Retention",
                )
                fig.update_xaxes(
                    title="Months since acquisition"
                )
                st.plotly_chart(
                    base_layout(
                        fig,
                        title="Maturity-aware retention decay",
                        height=500,
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                st.markdown(
                    """
                    <div class='science-card'>
                        <h4>Why the curve matters</h4>
                        <p>
                        The cohort matrix shows individual acquisition vintages;
                        the maturity curve aggregates customers at the same age.
                        This separates lifecycle decay from calendar-time effects
                        and prevents newer cohorts from being treated as if they
                        had already had time to mature.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            plot_missing()

    with tabs[3]:
        if scorecard is not None:
            st.dataframe(
                scorecard,
                use_container_width=True,
                hide_index=True,
            )
        else:
            plot_missing()


# =============================================================================
# CLV
# =============================================================================

elif page == "CLV":

    clv = get_table(
        files,
        "clv",
    )

    combined, _ = customer_source(
        files
    )

    if clv is None and combined is not None:
        ccol = clv_column(combined)
        if ccol:
            clv = combined.copy()

    if clv is None or clv.empty:
        st.warning(
            "No CLV output was detected. Run clv_analysis.py first."
        )
        st.stop()

    ccol = clv_column(
        clv
    )

    if ccol is None:
        st.error(
            "CLV file found, but no CLV field was recognized."
        )
        st.stop()

    clv_values = pd.to_numeric(
        clv[ccol],
        errors="coerce",
    )

    lower_col = find_column(
        clv,
        [
            "clv_lower",
            "clv_p10",
            "clv_lower_bound",
        ],
    )
    upper_col = find_column(
        clv,
        [
            "clv_upper",
            "clv_p90",
            "clv_upper_bound",
        ],
    )

    clean = clv_values.dropna()

    kpis = st.columns(5)
    with kpis[0]:
        st.metric(
            "Customers",
            number(clean.size),
        )
    with kpis[1]:
        st.metric(
            "Median CLV",
            money(clean.median()),
        )
    with kpis[2]:
        st.metric(
            "Mean CLV",
            money(clean.mean()),
        )
    with kpis[3]:
        st.metric(
            "Total CLV",
            money(clean.sum()),
        )
    with kpis[4]:
        st.metric(
            "90th percentile",
            money(clean.quantile(0.90)),
        )

    left, right = st.columns(
        [1.15, 0.85]
    )

    with left:
        st.markdown(
            "#### CLV distribution"
        )
        plot_df = pd.DataFrame(
            {
                "CLV": clean,
            }
        )

        fig = px.histogram(
            plot_df,
            x="CLV",
            nbins=60,
            marginal="box",
            labels={
                "CLV": "Customer lifetime value"
            },
        )
        fig.update_xaxes(
            rangemode="tozero"
        )
        st.plotly_chart(
            base_layout(
                fig,
                title="Customer value distribution",
                height=500,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with right:
        st.markdown(
            "#### Value uncertainty"
        )

        if lower_col and upper_col:
            lower = pd.to_numeric(
                clv[lower_col],
                errors="coerce",
            )
            upper = pd.to_numeric(
                clv[upper_col],
                errors="coerce",
            )

            valid = pd.DataFrame(
                {
                    "clv": clv_values,
                    "lower": lower,
                    "upper": upper,
                }
            ).dropna()

            valid = valid.sort_values(
                "clv"
            ).reset_index(
                drop=True
            )

            if len(valid) > 1000:
                valid = valid.iloc[
                    ::max(1, len(valid) // 1000)
                ]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(valid)),
                    y=valid["clv"],
                    mode="lines",
                    name="Expected CLV",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=np.r_[
                        np.arange(len(valid)),
                        np.arange(len(valid))[::-1],
                    ],
                    y=np.r_[
                        valid["upper"].to_numpy(),
                        valid["lower"].to_numpy()[::-1],
                    ],
                    fill="toself",
                    line=dict(width=0),
                    name="Uncertainty range",
                    opacity=0.25,
                )
            )
            fig.update_xaxes(
                title="Customers ordered by CLV"
            )
            fig.update_yaxes(
                title="CLV"
            )
            st.plotly_chart(
                base_layout(
                    fig,
                    title="Expected value with uncertainty",
                    height=500,
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            st.info(
                "No lower/upper CLV uncertainty fields were detected in the output."
            )

    seg_col = segment_column(
        clv
    )
    if seg_col:
        st.markdown(
            "#### CLV by behavioral segment"
        )

        tmp = clv.copy()
        tmp[ccol] = pd.to_numeric(
            tmp[ccol],
            errors="coerce",
        )
        profile = (
            tmp.dropna(
                subset=[ccol]
            )
            .groupby(seg_col)
            .agg(
                customers=(ccol, "size"),
                total_clv=(ccol, "sum"),
                median_clv=(ccol, "median"),
            )
            .reset_index()
            .sort_values(
                "total_clv",
                ascending=False,
            )
        )

        fig = px.bar(
            profile,
            x="total_clv",
            y=seg_col,
            orientation="h",
            labels={
                "total_clv": "Total CLV",
                seg_col: "Segment",
            },
            hover_data=[
                "customers",
                "median_clv",
            ],
        )
        st.plotly_chart(
            base_layout(
                fig,
                title="Where is customer value concentrated?",
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    st.markdown(
        """
        <div class='science-card'>
            <h4>How to interpret CLV scientifically</h4>
            <p>
            CLV is a forward economic estimate, not a historical revenue total.
            Its practical value comes from combining expected future activity,
            expected spend, persistence, returns, and model uncertainty. For
            portfolio decisions, the uncertainty band should be treated as part
            of the signal rather than ignored.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# RETENTION & NEXT PURCHASE
# =============================================================================

elif page == "Retention & Next Purchase":

    churn = get_table(
        files,
        "churn",
    )
    next_purchase = get_table(
        files,
        "next_purchase",
    )

    combined, _ = customer_source(
        files
    )

    if churn is None and combined is not None and churn_column(combined):
        churn = combined
    if next_purchase is None and combined is not None and next_purchase_column(combined):
        next_purchase = combined

    if churn is None and next_purchase is None:
        st.warning(
            "No churn or next-purchase outputs found. Run churn_next_purchase.py first."
        )
        st.stop()

    tabs = st.tabs(
        [
            "Risk profile",
            "Purchase propensity",
            "Risk vs propensity",
            "Model diagnostics",
        ]
    )

    with tabs[0]:
        if churn is not None:
            ccol = churn_column(churn)
            if ccol:
                values = pd.to_numeric(
                    churn[ccol],
                    errors="coerce",
                ).dropna()

                k = st.columns(4)
                with k[0]:
                    st.metric(
                        "Customers scored",
                        number(len(values)),
                    )
                with k[1]:
                    st.metric(
                        "Median churn risk",
                        pct(values.median()),
                    )
                with k[2]:
                    st.metric(
                        "High risk",
                        pct(
                            (values >= 0.70).mean()
                        ),
                    )
                with k[3]:
                    st.metric(
                        "Very high risk",
                        pct(
                            (values >= 0.85).mean()
                        ),
                    )

                fig = px.histogram(
                    pd.DataFrame(
                        {
                            "Churn probability": values
                        }
                    ),
                    x="Churn probability",
                    nbins=40,
                )
                fig.update_xaxes(
                    tickformat=".0%"
                )
                st.plotly_chart(
                    base_layout(
                        fig,
                        title="Predicted probability of churn / inactivity",
                        height=480,
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                st.markdown(
                    """
                    <div class='science-card'>
                        <h4>Why survival modeling is different from a churn label</h4>
                        <p>
                        A survival formulation models the timing of continued activity
                        and accounts for the fact that newer customers have had less
                        time to experience an observed lapse. This is preferable to
                        defining churn with one arbitrary recency threshold and calling
                        that label ground truth.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                plot_missing(
                    "Churn output found but no probability field was recognized."
                )
        else:
            plot_missing()

    with tabs[1]:
        if next_purchase is not None:
            pcol = next_purchase_column(
                next_purchase
            )
            if pcol:
                values = pd.to_numeric(
                    next_purchase[pcol],
                    errors="coerce",
                ).dropna()

                k = st.columns(4)
                with k[0]:
                    st.metric(
                        "Customers scored",
                        number(len(values)),
                    )
                with k[1]:
                    st.metric(
                        "Median probability",
                        pct(values.median()),
                    )
                with k[2]:
                    st.metric(
                        "High propensity",
                        pct(
                            (values >= 0.70).mean()
                        ),
                    )
                with k[3]:
                    st.metric(
                        "Low propensity",
                        pct(
                            (values < 0.30).mean()
                        ),
                    )

                fig = px.histogram(
                    pd.DataFrame(
                        {
                            "Next-purchase probability": values
                        }
                    ),
                    x="Next-purchase probability",
                    nbins=40,
                )
                fig.update_xaxes(
                    tickformat=".0%"
                )
                st.plotly_chart(
                    base_layout(
                        fig,
                        title="Predicted probability of the next purchase event",
                        height=480,
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            else:
                plot_missing(
                    "Next-purchase output found but no probability field was recognized."
                )
        else:
            plot_missing()

    with tabs[2]:
        if churn is not None and next_purchase is not None:
            ccol = churn_column(churn)
            pcol = next_purchase_column(next_purchase)

            id1 = find_column(
                churn,
                ["Customer ID", "customer_id"],
            )
            id2 = find_column(
                next_purchase,
                ["Customer ID", "customer_id"],
            )

            if ccol and pcol and id1 and id2:
                left = churn[[
                    id1,
                    ccol,
                ]].copy()
                right = next_purchase[[
                    id2,
                    pcol,
                ]].copy()
                left["Customer ID"] = pd.to_numeric(
                    left[id1],
                    errors="coerce",
                ).round()
                right["Customer ID"] = pd.to_numeric(
                    right[id2],
                    errors="coerce",
                ).round()

                merged = left[[
                    "Customer ID",
                    ccol,
                ]].merge(
                    right[[
                        "Customer ID",
                        pcol,
                    ]],
                    on="Customer ID",
                    how="inner",
                )

                merged[ccol] = pd.to_numeric(
                    merged[ccol],
                    errors="coerce",
                )
                merged[pcol] = pd.to_numeric(
                    merged[pcol],
                    errors="coerce",
                )
                merged = merged.dropna(
                    subset=[ccol, pcol]
                )

                if len(merged) > 12_000:
                    merged_plot = merged.sample(
                        12_000,
                        random_state=SEED,
                    )
                else:
                    merged_plot = merged

                fig = px.scatter(
                    merged_plot,
                    x=ccol,
                    y=pcol,
                    opacity=0.40,
                    labels={
                        ccol: "Churn probability",
                        pcol: "Next-purchase probability",
                    },
                    hover_data=["Customer ID"],
                )
                fig.update_xaxes(
                    tickformat=".0%"
                )
                fig.update_yaxes(
                    tickformat=".0%"
                )
                fig.add_vline(
                    x=0.70,
                    line_dash="dot",
                )
                fig.add_hline(
                    y=0.50,
                    line_dash="dot",
                )
                st.plotly_chart(
                    base_layout(
                        fig,
                        title="Customer risk versus near-term purchase propensity",
                        height=560,
                    ),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )

                st.caption(
                    "The most commercially interesting customers are not necessarily the highest-risk or highest-propensity customers in isolation. The decision layer combines these signals with value and behavioral context."
                )
            else:
                plot_missing(
                    "Customer ID and probability columns could not be matched across the two outputs."
                )
        else:
            plot_missing(
                "Both churn and next-purchase outputs are needed for this view."
            )

    with tabs[3]:
        diagnostics = []

        for label, df in [
            ("Churn", churn),
            ("Next purchase", next_purchase),
        ]:
            if df is None:
                diagnostics.append(
                    {
                        "Model": label,
                        "Status": "Not available",
                        "Rows": 0,
                        "Detected metrics": 0,
                    }
                )
                continue

            metric_candidates = [
                c for c in df.columns
                if any(
                    token in normalize_key(c)
                    for token in [
                        "auc",
                        "logloss",
                        "brier",
                        "precision",
                        "recall",
                        "f1",
                        "accuracy",
                        "calibration",
                        "rmse",
                        "mae",
                    ]
                )
            ]

            diagnostics.append(
                {
                    "Model": label,
                    "Status": "Available",
                    "Rows": len(df),
                    "Detected metrics": len(metric_candidates),
                }
            )

        st.dataframe(
            pd.DataFrame(diagnostics),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            <div class='science-card'>
                <h4>Validation principle</h4>
                <p>
                Predictive retention and purchase models should be evaluated with
                time-based backtesting. A random train/test split can leak future
                customer behavior into the training population and make model
                performance look artificially strong.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# DECISION ENGINE
# =============================================================================

elif page == "Decision Engine":

    decision = get_table(
        files,
        "decision",
    )
    action_summary = get_table(
        files,
        "action_summary",
    )

    if decision is None:
        st.warning(
            "No decision-engine output found. Run decision_engine.py first."
        )
        st.stop()

    action_col = find_column(
        decision,
        [
            "final_action",
            "recommended_action",
        ],
    )
    priority_col = find_column(
        decision,
        [
            "priority_score",
        ],
    )
    decision_clv = clv_column(
        decision
    )
    d_churn = churn_column(
        decision
    )
    d_next = next_purchase_column(
        decision
    )
    d_id = find_column(
        decision,
        ["Customer ID", "customer_id"],
    )

    if not action_col:
        st.error(
            "Decision file found but no action field was detected."
        )
        st.stop()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric(
            "Customers scored",
            number(len(decision)),
        )

    with kpi2:
        active_actions = ~decision[action_col].astype(str).str.startswith(
            "monitor",
            na=False,
        )
        st.metric(
            "Active policy rows",
            number(active_actions.sum()),
        )

    with kpi3:
        if priority_col:
            st.metric(
                "Median priority",
                number(
                    pd.to_numeric(
                        decision[priority_col],
                        errors="coerce",
                    ).median(),
                    1,
                ),
            )
        else:
            st.metric(
                "Median priority",
                "—",
            )

    with kpi4:
        if decision_clv:
            st.metric(
                "CLV represented",
                money(
                    pd.to_numeric(
                        decision[decision_clv],
                        errors="coerce",
                    ).sum()
                ),
            )
        else:
            st.metric(
                "CLV represented",
                "—",
            )

    st.markdown(
        "#### Action allocation"
    )

    action_counts = (
        decision.groupby(
            action_col,
            dropna=False,
        )
        .size()
        .reset_index(
            name="customers"
        )
        .sort_values(
            "customers",
            ascending=True,
        )
    )

    left, right = st.columns(
        [0.95, 1.05]
    )

    with left:
        fig = px.bar(
            action_counts,
            x="customers",
            y=action_col,
            orientation="h",
            labels={
                "customers": "Customers",
                action_col: "Recommended action",
            },
        )
        st.plotly_chart(
            base_layout(
                fig,
                title="Customer allocation",
                height=470,
            ),
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

    with right:
        if decision_clv and d_churn and priority_col:
            plot_df = decision.copy()
            plot_df[decision_clv] = pd.to_numeric(
                plot_df[decision_clv],
                errors="coerce",
            )
            plot_df[d_churn] = pd.to_numeric(
                plot_df[d_churn],
                errors="coerce",
            )
            plot_df[priority_col] = pd.to_numeric(
                plot_df[priority_col],
                errors="coerce",
            )
            plot_df = plot_df.dropna(
                subset=[
                    decision_clv,
                    d_churn,
                    priority_col,
                ]
            )

            if len(plot_df) > 12_000:
                plot_df = plot_df.sample(
                    12_000,
                    random_state=SEED,
                )

            fig = px.scatter(
                plot_df,
                x=decision_clv,
                y=d_churn,
                color=action_col,
                size=priority_col,
                size_max=18,
                opacity=0.45,
                labels={
                    decision_clv: "CLV",
                    d_churn: "Churn probability",
                    action_col: "Action",
                },
            )
            fig.update_xaxes(
                type="log",
                title="CLV (log scale)",
            )
            fig.update_yaxes(
                tickformat=".0%"
            )
            st.plotly_chart(
                base_layout(
                    fig,
                    title="Economic value, risk, and policy assignment",
                    height=470,
                ),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        else:
            plot_missing(
                "CLV, churn, and priority fields are needed for the decision scatter."
            )

    st.markdown(
        "#### Top commercial targets"
    )

    top_n = st.slider(
        "Customers",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

    if priority_col:
        targets = decision.copy()
        targets[priority_col] = pd.to_numeric(
            targets[priority_col],
            errors="coerce",
        )
        targets = targets.sort_values(
            priority_col,
            ascending=False,
        ).head(
            top_n
        )
    else:
        targets = decision.head(
            top_n
        )

    preferred_columns = [
        c for c in [
            d_id,
            action_col,
            "priority_tier",
            priority_col,
            decision_clv,
            d_churn,
            d_next,
            "reactivation_probability",
            "decision_confidence",
            "action_reason",
        ]
        if c and c in targets.columns
    ]

    display = targets[preferred_columns].copy()

    for column in display.columns:
        if column in {
            d_churn,
            d_next,
            "reactivation_probability",
            "decision_confidence",
        }:
            display[column] = pd.to_numeric(
                display[column],
                errors="coerce",
            ).map(
                lambda x: pct(x)
                if pd.notna(x)
                else "—"
            )
        elif column == decision_clv:
            display[column] = pd.to_numeric(
                display[column],
                errors="coerce",
            ).map(
                lambda x: money(x)
                if pd.notna(x)
                else "—"
            )
        elif column == priority_col:
            display[column] = pd.to_numeric(
                display[column],
                errors="coerce",
            ).map(
                lambda x: f"{x:.1f}"
                if pd.notna(x)
                else "—"
            )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "#### Decision science",
    )

    st.markdown(
        """
        <div class='science-card'>
            <h4>What this engine does — and does not do</h4>
            <p>
            It estimates <strong>expected commercial opportunity</strong> from the
            analytical signals already produced in the project. It then applies a
            transparent policy with capacity constraints. It does not infer an
            incremental treatment effect from observational purchase history.
            The next portfolio upgrade is an experiment or quasi-experiment that
            measures actual incremental response to each intervention.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if action_summary is not None:
        with st.expander(
            "View exported action summary"
        ):
            st.dataframe(
                action_summary,
                use_container_width=True,
                hide_index=True,
            )


# =============================================================================
# METHODOLOGY
# =============================================================================

else:

    st.markdown(
        "### Analytical workflow"
    )

    steps = [
        (
            "01",
            "Data quality",
            "Standardize transactions, identify reversals/cancellations, validate identifiers, and preserve a clear observation window.",
        ),
        (
            "02",
            "Customer 360",
            "Create one reusable analytical customer mart containing economics, cadence, assortment, price behavior, temporal behavior, lifecycle, and returns.",
        ),
        (
            "03",
            "Behavioral segmentation",
            "Robustly transform heterogeneous customer features, balance behavioral blocks, denoise with PCA, and cluster in latent behavioral space.",
        ),
        (
            "04",
            "Cohort analysis",
            "Measure customer retention and economic decay by acquisition cohort age, keeping return-only months visible to avoid understating commercial friction.",
        ),
        (
            "05",
            "CLV",
            "Project customer economics into the future and retain an uncertainty representation rather than reducing value to one deterministic number.",
        ),
        (
            "06",
            "Retention and purchase propensity",
            "Model forward-looking customer state using temporal splits so future purchase behavior does not leak into model development.",
        ),
        (
            "07",
            "Decision engine",
            "Combine value, risk, propensity, behavioral context, confidence, and capacity into a transparent commercial prioritization policy.",
        ),
        (
            "08",
            "Experimentation",
            "The causal layer comes last: actual interventions need randomized or otherwise defensible treatment/control data before incremental impact is claimed.",
        ),
    ]

    for number_, title, body in steps:
        st.markdown(
            f"""
            <div class='method-step'>
                <strong>{number_} · {title}</strong>
                <div class='small-note'>{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-label'>Model responsibilities</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)

    with cols[0]:
        st.markdown(
            """
            <div class='science-card'>
                <h4>Segmentation answers: who is different?</h4>
                <p>
                It discovers stable behavioral structure across customer economics,
                cadence, assortment, price, temporal preferences, lifecycle, and
                return behavior.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        st.markdown(
            """
            <div class='science-card'>
                <h4>CLV answers: who is economically valuable?</h4>
                <p>
                It forecasts future commercial value. It should be judged using
                temporal holdouts and uncertainty, not just in-sample fit.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[2]:
        st.markdown(
            """
            <div class='science-card'>
                <h4>Decisioning answers: who should receive attention?</h4>
                <p>
                It prioritizes customers for actions under commercial constraints.
                The policy becomes causal only after intervention effects are measured.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-label'>Project maturity checklist</div>",
        unsafe_allow_html=True,
    )

    readiness = []
    status_map = [
        ("Customer 360", "customer_360"),
        ("Behavioral segmentation", "segments"),
        ("Cohort retention", "cohort_matrix_logo"),
        ("CLV", "clv"),
        ("Churn / survival", "churn"),
        ("Next purchase", "next_purchase"),
        ("Decision engine", "decision"),
    ]

    for label, key in status_map:
        readiness.append(
            {
                "Layer": label,
                "Status": "Ready" if files.get(key) else "Pending",
                "Detected output": file_label(
                    files,
                    key,
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(readiness),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class='science-card'>
            <h4>Important scientific boundary</h4>
            <p>
            Transaction data is excellent for understanding observed customer
            behavior and forecasting future behavior. It is not, by itself, enough
            to establish that a campaign, discount, reminder, or recommendation
            caused an incremental outcome. The portfolio should explicitly preserve
            this distinction and introduce experimentation for the final causal layer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    """
    <div style='margin-top: 36px; padding-top: 14px; border-top: 1px solid #e4e7ec;'>
        <div class='small-note'>
            Retail Customer Intelligence · Online Retail II · Analytical decision support
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
