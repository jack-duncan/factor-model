# Factor Lens – Project Guide

## What this project is
An interactive Streamlit dashboard for multi-factor model analysis on US equities.
Users build portfolios by selecting tickers from a top-1000 market-cap universe (2015–present),
then view factor exposures, rolling betas, return attribution, and variance decomposition.

## Tech stack
- **Python 3.12+**, managed with **uv** (virtual env in `.venv/`)
- **Streamlit** – app entry point is `app.py`
- **WRDS** – academic financial data (CRSP, Fama-French, Compustat). Requires credentials in `.env`
- **statsmodels** – OLS and RollingOLS for factor decomposition
- **Plotly** – interactive charts
- **pandas / numpy** – data manipulation

## Project structure
```
src/
  config.py         – constants: factor names, date ranges, universe size, paths
  data_loader.py    – WRDS pulls + incremental refresh + parquet caching
  universe.py       – top-N universe construction by market cap
  factors.py        – constructed factor returns (quintile long-short spreads)
  factor_model.py   – full-sample & rolling OLS, attribution, variance decomposition
  portfolio.py      – portfolio return computation, per-stock regressions
  visualization.py  – Plotly chart builders (heatmap, rolling, attribution, risk pie, correlation)
app.py              – Streamlit dashboard (sidebar controls + 4 tabs)
notebooks/          – scratch exploration, not production code
```

## Key conventions
- All module imports use `from src.<module> import ...` (package-style).
- Configuration lives in `src/config.py`. Do not hardcode magic numbers elsewhere.
- Data is cached as parquet in `data/raw/` (CRSP, FF, Compustat) and `data/processed/` (universe, factor returns). These dirs are gitignored.
- WRDS queries use raw psycopg2 via `query()` in `data_loader.py` — this bypasses the pandas 2.3 / SQLAlchemy 1.4 incompatibility in the wrds library.
- Date alignment between CRSP and FF uses `dt.to_period('M')`.

## Factor model
- **Pre-built factors** (from FF library): MKT-RF, SMB, HML, RMW, CMA, Momentum
- **Constructed factors** (quintile spread returns): Volatility, Liquidity, Log Size, Leverage, Growth, Value
- Compustat-based factors (Leverage, Growth, Value) use a 6-month lag to avoid look-ahead bias. They are gracefully skipped if Compustat access is unavailable.
- Factor selection is dynamic — the UI lets users toggle any subset of factors.

## Running the app
```bash
source .venv/bin/activate
streamlit run app.py
```

## Common commands
```bash
uv add <package>          # add a dependency
uv sync                   # sync venv with pyproject.toml
streamlit run app.py      # launch dashboard
```

## Data pipeline
1. `refresh_data()` in `data_loader.py` orchestrates the full pipeline
2. Incremental pulls: `_metadata.json` tracks last pull date per dataset
3. `build_universe()` selects top 1000 stocks by market cap each month
4. `build_all_factor_returns()` constructs long-short factor series and merges with FF factors
5. All cached data lives in `data/raw/` and `data/processed/` as parquet files

## Style
- No type annotations on internal helpers unless complex.
- Keep functions short and focused. One regression = one function.
- Prefer pandas vectorized operations over loops.
- Charts go in `visualization.py`, not in `app.py`.
