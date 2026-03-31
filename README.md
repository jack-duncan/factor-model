# Factor Lens

Interactive multi-factor model dashboard for US equities, built with Streamlit. Construct portfolios from a top-1000 market-cap universe (2015–present) and analyze factor exposures, rolling betas, return attribution, and variance decomposition.

## Features

- **Portfolio builder** — search and select any tickers from the top-1000 universe, equal or market-cap weighted
- **Factor exposures** — beta heatmap across all selected stocks and the portfolio, with significance stars
- **Rolling exposures** — time-varying betas via RollingOLS (configurable window)
- **Return attribution** — stacked decomposition of returns into factor contributions + residual
- **Variance decomposition** — donut chart showing how much variance each factor explains
- **Factor correlation** — correlation matrix across all active factors

## Factor Model

**Pre-built (Fama-French library)**
| Factor | Description |
|--------|-------------|
| MKT-RF | Market excess return |
| SMB | Small minus big (size) |
| HML | High minus low (value) |
| RMW | Robust minus weak (profitability) |
| CMA | Conservative minus aggressive (investment) |
| Momentum | Winners minus losers |

**Constructed (quintile long-short spreads)**
| Factor | Characteristic | Source |
|--------|---------------|--------|
| Volatility | Trailing 12-month return std dev | CRSP |
| Liquidity | Trailing 12-month avg dollar volume | CRSP |
| Log Size | Log market cap | CRSP |
| Leverage | (LT debt + ST debt) / total assets | Compustat |
| Growth | Year-over-year sales growth | Compustat |
| Value | Book equity / market cap | Compustat |

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

On first launch, click **Refresh Data from WRDS** in the sidebar to pull data (CRSP, Fama-French, Compustat). Subsequent runs use the local parquet cache and only pull new data incrementally.

## Project Structure

```
src/
  config.py         — constants: factors, dates, universe size, paths
  data_loader.py    — WRDS pulls, incremental refresh, parquet caching
  universe.py       — top-1000 market-cap universe construction
  factors.py        — quintile long-short factor return construction
  factor_model.py   — OLS, RollingOLS, attribution, variance decomposition
  portfolio.py      — portfolio returns and per-stock regressions
  visualization.py  — Plotly chart builders
app.py              — Streamlit dashboard
notebooks/          — exploratory analysis (not production code)
data/
  raw/              — cached CRSP, FF, Compustat parquet files (gitignored)
  processed/        — universe, factor returns (gitignored)
```

## Data Sources

- **CRSP Monthly Stock File** — returns, prices, shares, market cap
- **Fama-French Factors** — via WRDS `ff.factors_monthly` and `ff.fivefactors_monthly`
- **Compustat Annual Fundamentals** — leverage, growth, book value (optional; app degrades gracefully without access)
- **CRSP-Compustat Link** — maps CRSP permnos to Compustat gvkeys
