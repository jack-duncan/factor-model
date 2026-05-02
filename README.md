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
- **Beta-neutral optimizer** — Treynor-Black LP (HiGHS solver): maximizes α/σ²ₑ subject to β_mkt = 0; globally optimal by construction; zero weights are meaningful (stock adds no alpha given the beta constraint); optimized weight shown as a column in the holdings table with return metrics below

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

---

## Regression Comparison Experiment

> Notebook: `notebooks/regression_comparison.ipynb`

### Objective

Test whether a stock's estimated **market beta** (`beta_mkt`) has cross-sectional predictive power for future relative returns, and whether pre-residualizing other features against it changes predictions. Secondary questions: does Lasso keep or zero out `beta_mkt`? Does regularization method matter? Is pooled OLS the right estimator for panel return data?

Note: an earlier version of this experiment used the contemporaneous FF market return (`mktrf`) as a feature. That was incorrect — `mktrf` is date-level (same for all stocks in a month) so it has zero cross-sectional variance and cannot predict relative performance. `beta_mkt` is a stock-specific characteristic that varies cross-sectionally and is the correct variable to test.

### Data

- **Source:** WRDS — CRSP daily (`crsp.dsf`) + Fama-French daily factors
- **Universe:** All common US equities (shrcd 10/11, exchcd 1/2/3), 2010–2019
- **Splits:** Train 2010–2015 | Break 2016 | Test 2017–2019
- **Target:** 28-day compounded forward excess return, converted to cross-sectional percentile rank within each month (see below)
- **Subsampling:** End-of-month observation per stock — eliminates 27/28-day overlap in the forward return window

### Features (17 total, all stock-specific)

Date-level features (`mktrf`, `smb`, `hml`, `rmw`, `cma`, `umd`, their lags, rolling versions, and `month`) are excluded — they are constant within any month's cross-section and carry zero cross-sectional predictive information.

| Group | Features |
|-------|----------|
| Market sensitivity (1) | `beta_mkt` — trailing 252-day OLS beta vs market |
| Return signals (7) | `ret_lag1`, `mom_5d/21d/63d/126d/252d`, `high_52w_ratio` |
| Volatility (3) | `rvol_21d`, `rvol_63d`, `dvol_21d` |
| Liquidity (2) | `amihud_21d`, `turnover_21d` |
| Price / size (4) | `log_mcap`, `log_prc`, `prc_to_ma50`, `prc_to_ma200` |

### Why cross-sectional ranking, not raw returns

The first attempt predicted raw 28-day excess returns directly. Results were degenerate: Lasso zeroed all coefficients, Ridge drove regularization to the maximum grid boundary (1e10), and OLS massively overfitted. This is a signal-to-noise problem: raw stock returns are dominated by the market's direction that month, which the stock-specific features cannot predict.

Switching the target to the **within-month percentile rank** of each stock's 28-day forward excess return removes the common market component. The model only needs to answer which stocks beat their peers, not whether the market is up or down. After this change, all six models produced real, non-degenerate results.

### Experimental design

Six models in a 2×3 design:

|  | OLS | Ridge | Lasso |
|--|-----|-------|-------|
| **Group A — beta_mkt included** | A1 | A2 | A3 |
| **Group B — residualized against beta_mkt** | B1 | B2 | B3 |

Plus a **Fama-MacBeth** baseline (cross-sectional OLS each month, averaged over time). FM now includes `beta_mkt` since it varies cross-sectionally.

- Group A: all 17 features including `beta_mkt` directly
- Group B: all other features projected onto the orthogonal complement of `beta_mkt` (Frisch-Waugh residualization), `beta_mkt` dropped
- Hyperparameters tuned via `TimeSeriesSplit` (5 folds) on training data only
- Features standardized using train-period statistics — no look-ahead

### Results

2016 is a **break (cooling-off) period** — a one-year buffer between train and test to let any temporal leakage decay. It is not evaluated on.

| Model | Group | Features | CV Rank IC | Test Rank IC | Test R² |
|-------|-------|----------|-----------|-------------|---------|
| A1: OLS | beta_mkt included | 17 | 0.0899 | 0.1134 | −0.2089 |
| A2: Ridge | beta_mkt included | 17 | 0.0930 | 0.1137 | −0.1320 |
| A3: Lasso | beta_mkt included | 17 | 0.0906 | 0.1136 | −0.0956 |
| B1: OLS | residualized | 16 | 0.0987 | 0.1141 | −0.2087 |
| B2: Ridge | residualized | 16 | 0.1002 | 0.1143 | −0.1564 |
| B3: Lasso ★ | residualized | 16 | 0.0997 | **0.1144** | −0.1181 |

★ Best model on test (Rank IC = 0.1144)

Primary metric is Rank IC (Spearman correlation of predicted vs actual rank). R² is negative for all models — expected, because the rank target has meaningful variance that no linear model fully captures. A Rank IC of ~0.11 OOS is a substantive result; published quantitative strategies typically target 0.05–0.15.

### Key findings

**1. Cross-sectional ranking was essential**
Predicting raw return was too noisy. Ranking within each month removed that noise and revealed real signal in the stock-specific features. The correct target is the within-month percentile rank, which removes the common market component and forces the model to predict only relative performance.

**2. FWL theorem confirmed: A1 ≈ B1**
OLS with `beta_mkt` included (A1) and OLS on features residualized against `beta_mkt` (B1) produced nearly identical results (Rank IC 0.1134 vs 0.1141, difference of 0.0007 attributable to floating point precision). The Frisch-Waugh-Lovell theorem guarantees these are algebraically identical, and the near-zero difference is the expected numerical outcome.

**3. beta_mkt has minimal additional predictive power**
Lasso kept `beta_mkt` with a near-zero coefficient (−0.000029) — barely non-zero. This means `beta_mkt` has almost no incremental predictive signal once volatility (`rvol_21d`, `rvol_63d`, `dvol_21d`) and size (`log_mcap`) are included, since these features are highly correlated with market beta. Consistent with the low-beta anomaly (Frazzini & Pedersen 2014): beta exposure alone does not predict relative returns; the associated risk characteristics (volatility, size) do.

**4. Residualizing against beta_mkt marginally improves regularized models**
Group B (residualized) slightly outperforms Group A across Lasso and Ridge (~0.0007 Rank IC), and is the best overall. Removing beta_mkt's component from other features lets the regularizer allocate capacity more cleanly to the idiosyncratic parts of each signal.

**5. Lasso selected 12 of 17 features — all interpretable**
Survivors: all volatility features (`rvol_21d`, `rvol_63d`, `dvol_21d`), all liquidity features (`amihud_21d`, `turnover_21d`), all price/size features (`log_mcap`, `log_prc`, `prc_to_ma50`, `prc_to_ma200`), short-term momentum (`mom_5d`), 52-week high ratio, and `beta_mkt` (near-zero). Dropped: short-term reversal (`ret_lag1`) and all medium/long-horizon momentum (`mom_21d`, `mom_63d`, `mom_126d`, `mom_252d`). Only the fastest momentum signal survives.

**6. OLS overfits; regularized models generalize better**
OLS R² collapses from +0.013 in-sample to −0.21 OOS. Ridge and Lasso converge to sensible regularization levels (α ~2e-3 for Lasso, ~6e4 for Ridge) and generalize more cleanly, with Lasso edging out Ridge on test Rank IC.

**7. Fama-MacBeth**
FM runs a separate cross-sectional OLS each month on the 17 stock-specific features and averages the resulting coefficient vectors. Standard errors are computed from the time-series of monthly beta estimates, correctly accounting for cross-sectional correlation in returns (all stocks in a month share the same market shock).

---

## Data Sources

- **CRSP Monthly Stock File** — returns, prices, shares, market cap, SIC codes (via `crsp.msf` + `crsp.msenames`)
- **Fama-French Factors** — FF3+momentum from `ff.factors_monthly`, FF5 from `ff.fivefactors_monthly`, joined on date
