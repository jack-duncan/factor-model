# Factor Lens

Interactive Fama-French factor model dashboard for US equities, built with Streamlit. Construct portfolios from a top-3000 market-cap universe (2015–present) and analyze factor exposures, rolling betas, return attribution, variance decomposition, and beta-neutral optimization.

## Features

- **Portfolio builder** — search and select tickers (up to 20) from the top-3000 universe; equal, market-cap, shares, or dollar-amount weighting (long or short)
- **Random portfolio generator** — generate a random portfolio of n stocks restricted to tickers with data in the selected date range; shares mode anchors equal-dollar positions to the largest-mcap stock at 10 shares; dollar mode draws from N(μ=$100, σ=$30)
- **Long-short support** — negative shares or dollar amounts treated as short positions; gross/net exposure displayed separately
- **Sector classification** — SIC-based broad sector shown in holdings table (Tech, Finance, Manufacturing, etc.)
- **Factor model equation** — fitted OLS equation with coefficients displayed above the tabs
- **Factor exposures** — beta heatmap across all selected stocks and the portfolio, with significance stars and R²
- **Rolling exposures** — time-varying betas via RollingOLS; estimation window in sidebar, two-ended display range slider in tab; includes idiosyncratic return series
- **Return attribution** — stacked area chart decomposing returns into factor contributions + idiosyncratic residual; consistent colors with variance decomposition
- **Variance decomposition** — donut chart showing how much return variance each factor explains (idio = 1 − R², exact)
- **Factor correlation** — correlation matrix across all active factors
- **Beta-neutral optimizer** — Treynor-Black QP anchored to current portfolio weights: maximizes α/σ²ₑ subject to β_mkt = 0; optimized weight shown as a column in the holdings table alongside current weight, with return metrics below

## Factor Model

**Fama-French factors (active)**
| Factor | Description |
|--------|-------------|
| MKT-RF | Market excess return |
| SMB | Small minus big (size) |
| HML | High minus low (value) |
| RMW | Robust minus weak (profitability) |
| CMA | Conservative minus aggressive (investment) |
| Momentum | Winners minus losers (UMD) |

**Constructed factors (disabled — available in `src/factors.py` for future use)**
| Factor | Characteristic | Source |
|--------|---------------|--------|
| Volatility | Trailing 12-month return std dev | CRSP |
| Liquidity | Trailing 12-month avg dollar volume | CRSP |
| Log Size | Log market cap | CRSP |
| Leverage | (LT debt + ST debt) / total assets | Compustat |
| Growth | Year-over-year sales growth | Compustat |
| Value | Book equity / market cap | Compustat |

To re-enable: set `ALL_FACTORS = FACTORS_PREBUILT + FACTORS_CONSTRUCTED` in `src/config.py` and restore the `build_all_factor_returns` call in `refresh_data()`.

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/), WRDS account

```bash
# Install dependencies
uv sync

# Add WRDS credentials
cp .env.example .env
# Edit .env and set WRDS_USERNAME=your_username
```

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

On first launch, click **Refresh Data from WRDS** at the bottom of the sidebar to pull CRSP and Fama-French data (2015–present). Subsequent runs use the local parquet cache and only pull new data incrementally.

## Controls

| Control | Location | Effect |
|---------|----------|--------|
| Date range | Sidebar | Filters all data; bounds regression and charts |
| Ticker search | Sidebar | Add up to 20 stocks to the portfolio |
| Random generator | Sidebar | Generate n random stocks with auto-calculated position sizes |
| Weighting | Sidebar | Equal / market-cap / shares / dollar-amount (all support short positions) |
| Factor toggles | Sidebar | Include/exclude factors from all regressions |
| Estimation window | Sidebar | Months of data used per rolling OLS estimate |
| Refresh Data | Sidebar (bottom) | Pull new data from WRDS incrementally |
| Display range | Rolling tab | Two-ended date slider to zoom into any sub-period |

## Project Structure

```
src/
  config.py         — constants: factors, dates, universe size, SIC sector mapping
  data_loader.py    — WRDS pulls, incremental refresh, parquet caching
  universe.py       — top-3000 market-cap universe construction with sector labels
  factors.py        — quintile long-short factor return construction (disabled)
  factor_model.py   — OLS, RollingOLS, attribution, variance decomposition
  portfolio.py      — portfolio returns, per-stock regressions, beta-neutral optimizer
  visualization.py  — Plotly chart builders with consistent color palette
app.py              — Streamlit dashboard
MATH.md             — mathematical reference for all models and formulas
notebooks/          — exploratory analysis (not production code)
data/
  raw/              — cached CRSP, FF parquet files (gitignored)
  processed/        — universe, factor returns (gitignored)
```

## Data Sources

- **CRSP Monthly Stock File** — returns, prices, shares, market cap, SIC codes (via `crsp.msf` + `crsp.msenames`)
- **Fama-French Factors** — FF3+momentum from `ff.factors_monthly`, FF5 from `ff.fivefactors_monthly`, joined on date
