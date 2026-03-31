# Factor Lens

Interactive Fama-French factor model dashboard for US equities, built with Streamlit. Construct portfolios from a top-1000 market-cap universe (2015–2024) and analyze factor exposures, rolling betas, return attribution, and variance decomposition.

## Features

- **Portfolio builder** — search and select tickers (up to 20) from the top-1000 universe; equal, market-cap, or custom shares weighting
- **Factor model equation** — displays the fitted OLS equation with coefficients above the tabs
- **Factor exposures** — beta heatmap across all selected stocks and the portfolio, with significance stars and R²
- **Rolling exposures** — time-varying betas via RollingOLS with configurable estimation window; display range controlled by a two-ended date slider
- **Return attribution** — stacked decomposition of returns into factor contributions + idiosyncratic residual
- **Variance decomposition** — donut chart showing how much variance each factor explains
- **Factor correlation** — correlation matrix across all active factors

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

On first launch, click **Refresh Data from WRDS** in the sidebar to pull CRSP and Fama-French data (2015–present). Subsequent runs use the local parquet cache and only pull new data incrementally.

## Controls

| Control | Location | Effect |
|---------|----------|--------|
| Date range | Sidebar | Filters all data; bounds regression and charts |
| Ticker search | Sidebar | Add up to 20 stocks to the portfolio |
| Weighting | Sidebar | Equal / market-cap / custom shares |
| Factor toggles | Sidebar | Include/exclude factors from all regressions |
| Estimation window | Sidebar | Months of data used per rolling OLS estimate |
| Display range | Rolling tab | Two-ended slider to zoom into any sub-period |

## Project Structure

```
src/
  config.py         — constants: factors, dates, universe size, paths
  data_loader.py    — WRDS pulls, incremental refresh, parquet caching
  universe.py       — top-1000 market-cap universe construction
  factors.py        — quintile long-short factor return construction (disabled)
  factor_model.py   — OLS, RollingOLS, attribution, variance decomposition
  portfolio.py      — portfolio returns and per-stock regressions
  visualization.py  — Plotly chart builders
app.py              — Streamlit dashboard
notebooks/          — exploratory analysis (not production code)
data/
  raw/              — cached CRSP, FF parquet files (gitignored)
  processed/        — universe, factor returns (gitignored)
```

## Data Sources

- **CRSP Monthly Stock File** — returns, prices, shares, market cap (via `crsp.msf` + `crsp.msenames`)
- **Fama-French Factors** — FF3+momentum from `ff.factors_monthly`, FF5 from `ff.fivefactors_monthly`, joined on date
