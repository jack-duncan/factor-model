# Factor Lens – Project Guide

## What this project is
An interactive Streamlit dashboard for Fama-French factor model analysis on US equities.
Users select stocks or build portfolios from a top-500 market-cap universe, then view
rolling factor exposures, return attribution, and portfolio-level analytics.

## Tech stack
- **Python 3.12+**, managed with **uv** (virtual env in `.venv/`)
- **Streamlit** – app entry point is `app.py`
- **WRDS** – academic financial data (CRSP, Fama-French). Requires credentials in `.env`
- **statsmodels** – OLS regressions for factor decomposition
- **Plotly** – interactive charts
- **pandas / numpy / scipy** – data manipulation

## Project structure
```
src/
  config.py         – constants: factor names, date ranges, universe size, paths
  data_loader.py    – WRDS pulls + parquet caching in data/raw/
  universe.py       – top-N universe construction by market cap
  factor_model.py   – rolling & full-sample OLS, factor attribution
  portfolio.py      – weighting schemes, portfolio-level exposures
  visualization.py  – Plotly chart builders
app.py              – Streamlit dashboard (imports from src/)
notebooks/          – scratch exploration, not production code
```

## Key conventions
- All module imports use `from src.<module> import ...` (package-style).
- Configuration lives in `src/config.py`. Do not hardcode magic numbers elsewhere.
- Data is cached as parquet in `data/raw/` and `data/processed/`. These dirs are gitignored.
- Functions that are not yet implemented raise `NotImplementedError` as stubs.

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

## Implementation notes
- The Fama-French 5-factor model is the default: Mkt-RF, SMB, HML, RMW, CMA.
- Rolling window default is 60 months, minimum 36 observations.
- Universe rebalances monthly by market cap.
- WRDS connection requires `WRDS_USERNAME` set in `.env` (see `.env.example`).
- Use `load_cached()` / `save_cache()` in `data_loader.py` to avoid repeated WRDS queries.

## When implementing stubs
Each `NotImplementedError` stub has a docstring describing expected inputs/outputs.
Implement them one module at a time in this order:
1. `data_loader.py` – WRDS queries and caching
2. `universe.py` – universe construction
3. `factor_model.py` – OLS regressions
4. `portfolio.py` – weighting and aggregation
5. `visualization.py` – chart builders
6. `app.py` – wire everything together in the dashboard

## Style
- No type annotations on internal helpers unless complex.
- Keep functions short and focused. One regression = one function.
- Prefer pandas vectorized operations over loops.
- Charts go in `visualization.py`, not in `app.py`.
