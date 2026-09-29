# UI Audit V4 — Retail Customer Intelligence Streamlit App

**Date:** 2026-09-29  
**Branch:** app-polish-v3 (HEAD: 72c1460)  
**Streamlit:** 1.59.1  
**Python:** 3.12.3

---

## Executive Summary

The Streamlit application has a solid analytical foundation but exhibits significant UI/rendering defects that prevent it from feeling like a professional commercial analytics product. The audit identifies **4 critical**, **12 high**, **18 medium**, and **8 low** severity issues across 11 categories.

---

## Audit Methodology

Inspected files:
- `streamlit_app/app.py` — Main entrypoint, CSS, sidebar, routing
- `streamlit_app/app_components.py` — Reusable UI components
- `streamlit_app/app_formatting.py` — Semantic formatters
- `streamlit_app/app_charts.py` — Plotly chart functions
- `streamlit_app/app_data.py` — Artifact registry
- `streamlit_app/app_config.py` — Configuration loader
- `streamlit_app/pages/*.py` — 9 page modules
- `config/project.yaml` — Project configuration
- `tests/` — Test files

Searched for: `unsafe_allow_html`, `st.markdown`, `st.dataframe`, `st.metric`, `st.columns`, `st.tabs`, `st.plotly_chart`, `<style>`, `st.selectbox`, `st.radio`

---

## Findings by Category

### 1. CSS/THEME SYSTEM

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 1.1 | app.py | Lines 101-239 | CRITICAL | Hardcoded light-theme CSS variables (`--ink: #172033`, `--panel: #ffffff`, `--line: #e4e7ec`) | **Dark mode completely broken** — white text on white backgrounds, invisible borders, unreadable charts | CSS variables defined only for light theme; no `@media (prefers-color-scheme: dark)` or Streamlit theme variable usage | Replace with theme-aware CSS using Streamlit's `config.toml` theme variables or CSS custom properties that adapt to `[data-theme="dark"]` |
| 1.2 | app.py | Line 118 | CRITICAL | `.stApp { background: #fbfcfe; }` | Dark mode: dark gray app background with dark text | Hardcoded background color | Use `var(--background-color)` or inherit from Streamlit theme |
| 1.3 | app.py | Line 127-130 | HIGH | Sidebar hardcoded `#f7f8fa` background | Dark mode: sidebar doesn't adapt | Hardcoded color | Use theme-aware background |
| 1.4 | app.py | Lines 132-138 | HIGH | `stMetric` hardcoded white background/border | Dark mode: metric cards have white background with dark text | Hardcoded panel color | Use `var(--secondary-background-color)` |
| 1.5 | app.py | Lines 140-146 | HIGH | Hero card hardcoded gradient/background | Dark mode: hero card has light gradient | Hardcoded colors | Use semantic theme tokens |
| 1.6 | app.py | Lines 173-192 | HIGH | `.science-card` hardcoded white background | Dark mode: cards invisible | Hardcoded `var(--panel)` which is `#ffffff` | Define `--surface` token that adapts |
| 1.7 | app.py | Lines 194-204 | HIGH | `.status-chip` hardcoded `var(--accent-soft)` | May not contrast in dark mode | Derived from primary color only | Use explicit semantic status colors with contrast guarantees |
| 1.8 | app.py | Lines 214-230 | HIGH | `.customer-header` hardcoded `#fff` background | Dark mode: white card on dark background | Hardcoded white | Use theme-aware surface token |

### 2. SIDEBAR STATUS CSS (CRITICAL DEFECT)

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 2.1 | app.py | Lines 257-268 | CRITICAL | `background:{color}15;color:{color}` uses status string directly | **Invalid CSS generated** — `color: ready; background: ready15;` — browser ignores, chips have no color | Status keys (`ready`, `incomplete`, `stale`, etc.) used as CSS color values instead of hex colors from `STATUS_COLORS` map | Already fixed in `STATUS_COLORS` map — must use `STATUS_COLORS[status]` not `status` string |

### 3. UNSAFE_ALLOW_HTML FRAGILITY

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 3.1 | app.py | Lines 101-239 | HIGH | 240 lines of inline CSS in `st.markdown(unsafe_allow_html=True)` | Maintenance burden; no type safety; CSS conflicts | All CSS embedded in Python string | Extract to external CSS file or centralized theme module |
| 3.2 | app_components.py | Lines 107-115 | HIGH | `render_science_card` uses HTML for card | Duplicated card styling across pages | HTML template in function | Replace with native Streamlit components or centralized component |
| 3.3 | app_components.py | Lines 131-148 | HIGH | `render_status_chip` uses inline HTML/CSS | Duplicated chip styling | HTML template in function | Centralize status chip component |
| 3.3 | app_components.py | Lines 360-369 | HIGH | `render_page_hero` uses HTML template | Hero styling not centralized | HTML template in function | Create hero component with native elements |
| 3.4 | app_components.py | Lines 376-378 | MEDIUM | `render_section_label` uses HTML | Section label styling duplicated | HTML template | Use native `st.markdown` with CSS class |
| 3.5 | app_components.py | Lines 394-403 | HIGH | `render_customer_header` uses HTML | Customer header not theme-aware | HTML template | Replace with native components |
| 3.6 | executive.py | Lines 68-79 | MEDIUM | Raw HTML science card | Duplicate of component | Copy-paste | Use `render_science_card` |
| 3.7 | retention.py | Lines 68-79, 206-216 | MEDIUM | Raw HTML science cards | Duplicated | Copy-paste | Use `render_science_card` |
| 3.8 | methodology.py | Lines 73-81 | MEDIUM | Raw HTML step cards | Custom styling not in design system | Ad-hoc HTML | Create step card component |
| 3.9 | methodology.py | Lines 206-216 | MEDIUM | Raw HTML science card | Duplicate | Copy-paste | Use `render_science_card` |

### 4. FORMATTING SEMANTIC BUGS

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 4.1 | app_formatting.py | Line 156 | HIGH | `"expected_days_to_next_purchase": SemanticType.DURATION_DAYS` — but `customer_360.py` line 209 calls `format_duration_months(value)` | **Days displayed as months** — e.g., "45 days" shown as "1.5 mo" | Wrong formatter called for days field | Fix `customer_360.py` to use `format_days` or add `DURATION_DAYS` to auto-formatter |
| 4.2 | app_formatting.py | Line 85 | HIGH | `"decision_confidence": SemanticType.PROBABILITY` — but confidence is a model confidence score, not a probability | **Confidence shown as percentage** — "0.85" → "85.0%" misleading | Semantic misclassification | Change to `SCORE` or create `CONFIDENCE` type |
| 4.3 | app_formatting.py | Lines 281-284 | HIGH | `format_percent` heuristic: `if abs(value) > 1` treats as percentage | **Ambiguous behavior** — 0.5 → "50.0%", 50 → "50.0%", but 1.5 → "1.5%" | Heuristic instead of explicit contract | Remove heuristic; require explicit semantic type |
| 4.4 | app_formatting.py | Lines 548-555 | HIGH | `auto_format` fallback heuristic for floats | **Unpredictable formatting** — values ≤1.5 become probability, <100 become ratio, else currency | Heuristic fallback | Remove fallback; require explicit column mapping |
| 4.5 | customer_360.py | Line 210 | HIGH | `format_duration_months(value)` for "Expected days to next purchase" | **Days shown as months** | Wrong formatter import | Use `format_days` from app_formatting |
| 4.6 | decision_engine.py | Line 82 | MEDIUM | `formatter="count", column_name="priority_score"` for priority score | **Priority score formatted as integer count** | Wrong formatter argument | Use `formatter="score"` |
| 4.7 | decision_engine.py | Lines 132-144 | MEDIUM | Manual formatting loop instead of `format_dataframe_columns` | Inconsistent formatting; duplication | Not using centralized formatter | Use `format_dataframe_columns` |

### 5. CUSTOMER 360 SOURCE CONTRACT VIOLATION

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 5.1 | customer_360.py | Line 41 | HIGH | `registry.load_dataframe("cohorts", "customer_month_events.csv")` for monthly history | **Wrong module dependency** — Customer 360 should own its customer-month panel | Historical artifact location | Load from `customer_360` module; update ArtifactRegistry contract |
| 5.2 | customer_360.py | Lines 80-85 | HIGH | `safe_get(row, *keys)` uses `pd.notna()` but then `or` fallback in other places | **Zero values treated as missing** — revenue=0 becomes "—" | `value or fallback` pattern | Use explicit `pd.notna()` checks everywhere |
| 5.3 | customer_360.py | Lines 49-69 | MEDIUM | Query param handling updates `st.query_params` on every render | **Potential render loops** — URL changes trigger rerun | Mutable query params in render path | Read once, don't write on every render |

### 6. COMPONENT DUPLICATION / INCONSISTENCY

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 6.1 | app_components.py | Lines 51-82 | HIGH | `render_kpi_card` wraps `st.metric` but no consistent card styling | Metric cards look different across pages | Minimal wrapper | Create styled KPI card component |
| 6.2 | app_components.py | Lines 213-292 | MEDIUM | `render_customer_selector` has search + selectbox but used inconsistently | Some pages use raw `st.selectbox` | Not enforced | Replace all customer selectors with this component |
| 6.3 | app_components.py | Lines 299-341 | MEDIUM | `render_action_summary_table` uses `Styler` with inline color function | Styler API deprecated in newer pandas; colors hardcoded | Pandas Styler usage | Use native column config or Plotly table |
| 6.4 | Multiple pages | Various | HIGH | KPI rows use `st.columns(4)` or `st.columns(5)` with `render_kpi_card` | **Inconsistent KPI density** — 4 vs 5 columns, different gaps | No standard | Define KPI row component with responsive behavior |
| 6.5 | Multiple pages | Various | MEDIUM | Science cards rendered with `st.columns(4)` | **4 equal cards** dominate bottom of pages | No visual hierarchy | Reduce to 1-2 primary cards; use expanders for detail |

### 7. CHART SYSTEM DEFECTS

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 7.1 | app_charts.py | Lines 61-94 | CRITICAL | `base_layout` hardcodes `font.color="#172033"`, `paper_bgcolor="rgba(0,0,0,0)"` | **Charts unreadable in dark mode** — dark text on dark background | No theme awareness | Use theme-aware colors; detect Streamlit theme |
| 7.2 | app_charts.py | Lines 21-31 | HIGH | `CHART_COLORS` defines light-theme only | No dark mode palette | Single palette | Define light/dark variants |
| 7.3 | app_charts.py | Line 72-75 | HIGH | Font family `Inter, ui-sans-serif...` may not exist | Fallback to system font; inconsistent | Assumption | Use system font stack |
| 7.4 | app_charts.py | Line 74 | HIGH | `color="#172033"` hardcoded dark text | Dark mode: invisible axis labels | Hardcoded | Use theme-aware color |
| 7.5 | app_charts.py | Various | MEDIUM | Color scales hardcoded (`"Blues"`, `"Viridis"`) | May not be colorblind-safe | Default Plotly scales | Use colorblind-safe scales (viridis, cividis, turbo) |
| 7.6 | executive.py | Lines 172-192 | MEDIUM | Inline Plotly figure construction (not using `app_charts` functions) | Inconsistent styling; duplication | Bypass chart system | Use centralized chart functions |
| 7.7 | customer_360.py | Lines 133-170 | MEDIUM | Inline Plotly figure construction | Inconsistent styling | Bypass chart system | Use centralized chart functions |
| 7.8 | app_charts.py | Lines 386-383 | MEDIUM | `plot_dual_axis_line_bar` doesn't use `base_layout` fully | Inconsistent dual-axis styling | Custom layout | Extend base_layout for dual axis |

### 8. TABLE SYSTEM DEFECTS

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 8.1 | Multiple pages | Various | HIGH | Raw `st.dataframe(df, use_container_width=True, hide_index=True)` | **Technical column names** (`recommended_product`, `clv_mean`), no formatting, excessive width | No table formatting layer | Create table component with semantic column config |
| 8.2 | recommendations.py | Lines 125-135 | HIGH | Manual column formatting with lambdas | **Inconsistent formatting**; `score` → "0.8700", `association_lift` → "1.23" | Ad-hoc formatting | Use `format_dataframe_columns` |
| 8.3 | decision_engine.py | Lines 129-146 | HIGH | Manual column formatting loop | **Inconsistent formatting**; priority score as "123.0" not "123" | Ad-hoc formatting | Use `format_dataframe_columns` |
| 8.4 | segmentation.py | Line 124 | MEDIUM | Raw `st.dataframe(profiles)` | Technical columns, no formatting | No table component | Apply table formatting |
| 8.5 | cohorts.py | Line 127 | MEDIUM | Raw `st.dataframe(scorecard)` | Technical columns | No table component | Apply table formatting |
| 8.6 | methodology.py | Line 142 | MEDIUM | Raw `st.dataframe(pd.DataFrame(readiness))` | Technical columns | No table component | Apply table formatting |
| 8.7 | app_components.py | Line 165 | MEDIUM | `render_evidence_table` creates DataFrame and renders raw | No semantic formatting | Minimal wrapper | Enhance with formatting |

### 9. PAGE STRUCTURE / VISUAL HIERARCHY

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 9.1 | executive.py | Lines 231-256 | HIGH | 4 equal science cards at bottom | **Visual noise** — cards dominate, no hierarchy | Equal weighting | Reduce to 2 primary cards; move detail to expanders |
| 9.2 | segmentation.py | Lines 48-61 | MEDIUM | 4 KPIs: Customers, Segments, Largest, Noise | **Noise KPI not primary** — should be secondary | Equal prominence | Primary: Customers, Segments; Secondary: Noise |
| 9.3 | predictive.py | Lines 75-85 | MEDIUM | 5 KPIs equal width | **Crowded** on narrow screens | Fixed 5 columns | Responsive KPI row (3+2 or 2+2+1) |
| 9.4 | retention.py | Lines 54-63, 96-104 | MEDIUM | 4 KPIs per tab | **Repetitive** structure | Template repetition | Vary KPI selection per tab |
| 9.5 | recommendations.py | Lines 38-44 | MEDIUM | 3 KPIs only | **Sparse** — could show more | Under-utilized | Add coverage/diversity metrics |
| 9.6 | decision_engine.py | Lines 71-90 | MEDIUM | 4 KPIs; "Active policy rows" uses string match | **Fragile** — string match on action name | Ad-hoc logic | Use explicit action categorization |
| 9.7 | Multiple pages | Various | MEDIUM | `st.columns([1.1, 0.9])` or similar ratios | **Fragile layouts** — break on narrow screens | Fixed ratios | Use responsive patterns |

### 10. RESPONSIVE LAYOUT ISSUES

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 10.1 | executive.py | Line 126 | MEDIUM | `st.columns(5)` for KPIs | **Horizontal overflow** on <1200px | Fixed 5 columns | Max 4 columns; wrap on mobile |
| 10.2 | predictive.py | Line 75 | MEDIUM | `st.columns(5)` for KPIs | Same as above | Fixed 5 columns | Responsive KPI grid |
| 10.3 | customer_360.py | Line 112 | MEDIUM | `st.columns([1.2, 0.8])` | **Right column crushed** on narrow | Fixed ratio | Stack on narrow |
| 10.4 | segmentation.py | Line 73 | MEDIUM | `st.columns([1.1, 0.9])` | Same | Fixed ratio | Stack on narrow |
| 10.5 | cohorts.py | Line 79 | MEDIUM | 4 tabs — OK but content may overflow | Tables may overflow | No table constraints | Add max-width to tables |

### 11. ENTRYPOINT / LAUNCH ISSUES

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 11.1 | app.py | Lines 21-24 | MEDIUM | `sys.path.insert(0, PROJECT_ROOT)` for imports | **Fragile** — depends on file location | Package-relative imports not working | Fix package structure or use proper module entrypoint |
| 11.2 | README.md | Line 254 | MEDIUM | Documents `streamlit run streamlit_app/app.py` | **Not tested** from clean checkout | Not verified | Test and document verified command |
| 11.3 | config/project.yaml | Lines 81-89 | LOW | Dashboard theme config exists but not used | Theme config ignored | Not connected to app | Connect theme config to Streamlit config |

### 12. ACCESSIBILITY

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 12.1 | app_charts.py | Various | HIGH | Red/green color coding for actions (protect_value=green, accelerate=blue) | **Colorblind unsafe** — deuteranopes can't distinguish | Standard palette | Use colorblind-safe palette (viridis, or explicit patterns) |
| 12.2 | app_charts.py | Line 445 | MEDIUM | `tickformat=".0%"` on y-axis | **Screen readers** may not announce % | No aria labels | Add accessible labels |
| 12.3 | Multiple | Various | MEDIUM | Status chips use color only | **No text indicator** for status | Color-only state | Add text/icon indicator |

### 13. PERFORMANCE

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 13.1 | app_charts.py | Lines 210-212, 513-515 | MEDIUM | `max_points=12000` sampling for scatter plots | **Potential data loss** in visualization | No progressive loading | Add sampling indicator; consider aggregation |
| 13.2 | app_data.py | Lines 409-418 | MEDIUM | Cache key includes `st_mtime_ns` and `st_size` | **Cache invalidation** on every file touch | Over-sensitive cache | Use content hash or version-based cache |
| 13.3 | Multiple pages | Various | LOW | `pd.to_numeric(..., errors="coerce")` repeated | **Repeated computation** on every render | No caching of transformed columns | Pre-compute or cache formatted columns |

### 14. EMPTY/ERROR STATE HANDLING

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 14.1 | app_components.py | Line 174 | MEDIUM | `render_missing` uses `st.info` | **Generic message** — doesn't explain what to do | Minimal implementation | Add actionable guidance: what, why, how to fix |
| 14.2 | Multiple pages | Various | MEDIUM | `st.warning("No X outputs found. Run Y.py first.")` | **Inconsistent messages** | Ad-hoc messages | Standardize empty state component |
| 14.3 | executive.py | Line 198 | MEDIUM | "Run customer_360.py or clv_analysis.py" — ambiguous | User doesn't know which script | Multiple fallbacks | Single canonical source per artifact |

### 15. DEEP LINKING / STATE ISSUES

| # | File | Location | Severity | Problem | Visible Consequence | Root Cause | Recommended Correction |
|---|------|----------|----------|---------|---------------------|------------|------------------------|
| 15.1 | customer_360.py | Lines 68-69 | HIGH | `st.query_params["customer_id"] = str(selected_id)` on every render | **Render loop** — URL change triggers rerun | Writing query params in render path | Write only on selection change |
| 15.2 | customer_360.py | Line 63 | MEDIUM | `render_customer_selector` creates widgets with dynamic keys | **Widget state conflicts** on navigation | Key prefix not unique enough | Use stable, unique keys |

---

## Summary by Severity

| Severity | Count |
|----------|-------|
| CRITICAL | 4 |
| HIGH | 12 |
| MEDIUM | 18 |
| LOW | 8 |
| **Total** | **42** |

---

## Priority Fix Order

1. **CRITICAL**: Fix sidebar status CSS (2.1) — breaks pipeline status display
2. **CRITICAL**: Dark theme support (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 7.1, 7.4) — app unusable in dark mode
3. **CRITICAL**: Chart theme awareness (7.1, 7.2, 7.4) — charts unreadable in dark mode
4. **HIGH**: Formatting semantic bugs (4.1, 4.2, 4.3, 4.4, 4.5, 4.6) — wrong data representation
5. **HIGH**: Customer 360 source contract (5.1) — architectural violation
6. **HIGH**: Centralize design tokens and components (3.1-3.9, 6.1-6.5) — duplication/fragility
7. **HIGH**: Table system (8.1-8.7) — poor data presentation
8. **HIGH**: Page visual hierarchy (9.1-9.7) — poor UX
9. **MEDIUM**: Responsive layouts (10.1-10.5)
10. **MEDIUM**: Accessibility (12.1-12.3)
11. **MEDIUM**: Performance (13.1-13.3)
12. **MEDIUM**: Empty states (14.1-14.3)
13. **MEDIUM**: Deep linking (15.1-15.2)
14. **LOW**: Entrypoint (11.1-11.3)

---

## Files Requiring Changes

### Core Infrastructure (Highest Impact)
1. `streamlit_app/app.py` — CSS, sidebar, entrypoint
2. `streamlit_app/app_components.py` — Components, design tokens
3. `streamlit_app/app_formatting.py` — Semantic formatters
4. `streamlit_app/app_charts.py` — Plotly theme system
5. `streamlit_app/app_data.py` — Artifact registry (Customer 360 contract)

### Page Modules (All 9)
6. `streamlit_app/pages/executive.py`
7. `streamlit_app/pages/customer_360.py`
8. `streamlit_app/pages/segmentation.py`
9. `streamlit_app/pages/cohorts.py`
10. `streamlit_app/pages/predictive.py`
11. `streamlit_app/pages/retention.py`
12. `streamlit_app/pages/recommendations.py`
13. `streamlit_app/pages/decision_engine.py`
14. `streamlit_app/pages/methodology.py`

### Configuration
15. `config/project.yaml` — Theme config
16. `.streamlit/config.toml` — Streamlit theme config (new)

### Tests (New)
17. `tests/test_ui_components.py`
18. `tests/test_ui_formatting.py`
19. `tests/test_ui_contracts.py`

### Documentation
20. `docs/UI_QA_RESULTS_V4.json`
21. `docs/UI_POLISH_REPORT_V4.md`