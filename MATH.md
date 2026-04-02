# Factor Model Mathematics

A reference for all the math underlying Factor Lens.

---

## 1. The Factor Model

### 1.1 CAPM (baseline)

The Capital Asset Pricing Model says a stock's expected excess return is proportional to its exposure to the market:

$$r_i - r_f = \alpha_i + \beta_i \cdot (r_m - r_f) + \varepsilon_i$$

| Symbol | Meaning |
|--------|---------|
| $r_i$ | Stock return |
| $r_f$ | Risk-free rate (1-month T-bill) |
| $r_m - r_f$ | Market excess return (MKT-RF) |
| $\beta_i$ | Market sensitivity ("beta") |
| $\alpha_i$ | Intercept — return unexplained by the market |
| $\varepsilon_i$ | Idiosyncratic residual, $E[\varepsilon_i] = 0$ |

### 1.2 Fama-French 3-Factor Model (FF3)

Fama & French (1993) found that two additional factors — size and value — explain cross-sectional returns beyond CAPM:

$$r_i - r_f = \alpha_i + \beta_{mkt} \cdot \text{MKT-RF} + \beta_{smb} \cdot \text{SMB} + \beta_{hml} \cdot \text{HML} + \varepsilon_i$$

| Factor | Construction | Economic intuition |
|--------|-------------|-------------------|
| MKT-RF | Value-weighted market return minus T-bill | Compensation for bearing systematic risk |
| SMB | Small-cap portfolio return minus large-cap | Small firms earn a premium (size effect) |
| HML | High book-to-market minus low book-to-market | Value stocks earn a premium over growth |

### 1.3 Fama-French 5-Factor Model (FF5)

Fama & French (2015) added two factors motivated by the dividend discount model:

$$r_i - r_f = \alpha_i + \beta_{mkt} \cdot \text{MKT-RF} + \beta_{smb} \cdot \text{SMB} + \beta_{hml} \cdot \text{HML} + \beta_{rmw} \cdot \text{RMW} + \beta_{cma} \cdot \text{CMA} + \varepsilon_i$$

| Factor | Construction | Economic intuition |
|--------|-------------|-------------------|
| RMW | Robust profitability minus weak | Profitable firms earn more |
| CMA | Conservative investment minus aggressive | Firms investing less earn more |

### 1.4 FF6: Adding Momentum

Carhart (1997) documented that past winners continue to outperform past losers (UMD = Up Minus Down):

$$r_i - r_f = \alpha_i + \beta_{mkt} \cdot \text{MKT-RF} + \beta_{smb} \cdot \text{SMB} + \beta_{hml} \cdot \text{HML} + \beta_{rmw} \cdot \text{RMW} + \beta_{cma} \cdot \text{CMA} + \beta_{umd} \cdot \text{UMD} + \varepsilon_i$$

This is the model used in Factor Lens (all 6 factors toggleable).

---

## 2. OLS Regression

### 2.1 Setup

In matrix form, with $T$ months of data and $K$ factors:

$$\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}$$

where:
- $\mathbf{y} \in \mathbb{R}^T$ — excess returns
- $\mathbf{X} \in \mathbb{R}^{T \times (K+1)}$ — factor returns, with a leading column of ones (for the intercept $\alpha$)
- $\boldsymbol{\beta} \in \mathbb{R}^{K+1}$ — coefficients to estimate

### 2.2 Ordinary Least Squares Estimator

OLS minimizes the sum of squared residuals:

$$\hat{\boldsymbol{\beta}} = \underset{\boldsymbol{\beta}}{\arg\min} \; \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

### 2.3 Fitted values and residuals

$$\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}, \qquad \hat{\boldsymbol{\varepsilon}} = \mathbf{y} - \hat{\mathbf{y}}$$

### 2.4 Goodness of fit: R²

R² measures the fraction of total variance explained by the factors:

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum_t \hat{\varepsilon}_t^2}{\sum_t (y_t - \bar{y})^2}$$

- $\text{RSS}$ = residual sum of squares (unexplained)
- $\text{TSS}$ = total sum of squares (total variance in $y$, up to a constant)
- $R^2 \in [0, 1]$; higher = factors explain more of the return

### 2.5 Adjusted R²

Penalizes for adding factors that don't improve fit:

$$\bar{R}^2 = 1 - (1 - R^2) \cdot \frac{T - 1}{T - K - 1}$$

Adding a useless factor always increases $R^2$ but may decrease $\bar{R}^2$.

### 2.6 Standard errors and t-statistics

Under classical OLS assumptions, the variance of $\hat{\boldsymbol{\beta}}$ is:

$$\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}, \qquad \hat{\sigma}^2 = \frac{\text{RSS}}{T - K - 1}$$

The t-statistic for each coefficient tests $H_0: \beta_k = 0$:

$$t_k = \frac{\hat{\beta}_k}{\text{se}(\hat{\beta}_k)}$$

Significance stars (used in the beta heatmap):
- `***` — $|t| > 2.576$ ($p < 0.01$)
- `**` — $|t| > 1.960$ ($p < 0.05$)
- `*` — $|t| > 1.645$ ($p < 0.10$)

---

## 3. Rolling OLS

### 3.1 Motivation

Full-sample betas assume exposures are constant over time. Rolling OLS relaxes this by estimating the model over a sliding window of $W$ months:

$$\hat{\boldsymbol{\beta}}_t = (\mathbf{X}_{t-W+1:t}^\top \mathbf{X}_{t-W+1:t})^{-1} \mathbf{X}_{t-W+1:t}^\top \mathbf{y}_{t-W+1:t}$$

Each $\hat{\boldsymbol{\beta}}_t$ uses only the $W$ most recent observations ending at month $t$.

### 3.2 Idiosyncratic return (rolling)

After estimating the rolling model, the idiosyncratic return at each month is the actual return minus the fitted factor contribution:

$$\varepsilon_t = y_t - \hat{\mathbf{x}}_t^\top \hat{\boldsymbol{\beta}}_t$$

This captures the part of the return not explained by systematic factors at that point in time.

### 3.3 Estimation window choice

- **Too short** (e.g., 12 months): betas are noisy, react to outliers
- **Too long** (e.g., 120 months): betas are smooth but may miss structural shifts
- **Rule of thumb**: 36–60 months balances stability and responsiveness; requires at least $K + 1$ observations per window

---

## 4. Portfolio Construction

### 4.1 Portfolio return

The portfolio return each month is the weighted average of individual stock returns:

$$r_p = \sum_{i=1}^{N} w_i r_i$$

where $w_i$ is the weight of stock $i$ and $\sum_i w_i = 1$ for a long-only portfolio.

### 4.2 Equal weighting

$$w_i = \frac{1}{N}$$

Simple, no data required beyond the return series.

### 4.3 Market-cap weighting

$$w_i = \frac{\text{mcap}_i}{\sum_j \text{mcap}_j}$$

Larger stocks get more weight — mirrors how index funds like the S&P 500 are constructed.

### 4.4 Shares weighting (long-short)

With a vector of signed share counts $n_i$ (positive = long, negative = short):

$$\text{position}_i = n_i \cdot |P_i|, \qquad \text{gross} = \sum_i |\text{position}_i|$$

$$w_i = \frac{\text{position}_i}{\text{gross}}$$

- Long positions: $w_i > 0$, contribute positively when the stock rises
- Short positions: $w_i < 0$, contribute negatively when the stock rises (profit when it falls)

Using **gross exposure** as the denominator (not net) prevents the weights from blowing up when longs and shorts roughly cancel.

**Key identity:** $\sum_i |w_i| = 1$ (gross-exposure-weighted), but $\sum_i w_i \neq 1$ for long-short portfolios.

---

## 5. Market Neutrality

### 5.1 Definition

A portfolio is market-neutral if its beta to MKT-RF is zero:

$$\beta_p = \sum_i w_i \beta_i = 0$$

### 5.2 Two-stock case

You're long stock A (beta $\beta_A$) and short stock B (beta $\beta_B$). In dollar terms, let the long position be $\$1$ and the short position be $\$x$:

$$\beta_A \cdot 1 - \beta_B \cdot x = 0 \implies x = \frac{\beta_A}{\beta_B}$$

**In shares:** if you hold $n_A$ shares of A at price $P_A$ and want to short $n_B$ shares of B at price $P_B$:

$$n_B = n_A \cdot \frac{P_A}{P_B} \cdot \frac{\beta_A}{\beta_B}$$

**Example:** Long 10 shares of GM at \$50 (β = 1.2), short TSLA at \$250 (β = 2.0):

$$n_{TSLA} = 10 \cdot \frac{50}{250} \cdot \frac{1.2}{2.0} = 10 \cdot 0.2 \cdot 0.6 = 1.2 \text{ shares}$$

Short 1.2 shares of TSLA to make the pair market-neutral.

### 5.3 General case

For any number of stocks, the portfolio beta is:

$$\beta_p = \mathbf{w}^\top \boldsymbol{\beta}$$

Market neutrality requires $\mathbf{w}^\top \boldsymbol{\beta} = 0$. In practice, this is a constraint in a portfolio optimization problem.

---

## 6. Beta-Neutral Return Optimization

### 6.1 Problem statement

Given a set of stocks with estimated alphas $\hat{\alpha}_i$, market betas $\hat{\beta}_i$, and idiosyncratic variances $\hat{\sigma}^2_{\varepsilon,i}$, find weights $w_i$ that:

$$\max_{w} \quad \sum_i \frac{\hat{\alpha}_i}{\hat{\sigma}^2_{\varepsilon,i}} w_i \qquad \text{(maximize appraisal-ratio-weighted alpha)}$$

$$\text{subject to} \quad \sum_i \hat{\beta}_i \, w_i = 0 \quad \text{(market neutral)}$$

$$w_i \in [-1, 1] \quad \text{(per-stock position limit)}$$

After solving, normalize so $\sum_i |w_i| = 1$ (gross exposure = 1). This preserves the beta-neutrality constraint since it is linear and homogeneous: if $\boldsymbol{\beta}^\top w = 0$ then $\boldsymbol{\beta}^\top (w/c) = 0$ for any scalar $c$.

Net exposure $\sum_i w_i$ is unconstrained — the optimizer may tilt net long or net short depending on where the alpha opportunities lie.

### 6.2 Why $\alpha / \sigma^2_\varepsilon$?

The **appraisal ratio** $\hat{\alpha}_i / \hat{\sigma}_{\varepsilon,i}$ measures how much excess return a stock delivers per unit of idiosyncratic risk. Weighting the objective by $1/\hat{\sigma}^2_{\varepsilon,i}$ (rather than just maximizing raw alpha) means:

- Stocks with high alpha but noisy estimates (large $\hat{\sigma}^2_\varepsilon$) get down-weighted
- Stocks with stable, consistent alpha get up-weighted
- This is the **Treynor-Black model** (1973): the optimal active weight in a stock is proportional to its appraisal ratio

### 6.3 Inputs from the factor model

| Input | Source |
|-------|--------|
| $\hat{\alpha}_i$ | `params["const"]` from per-stock OLS |
| $\hat{\beta}_{mkt,i}$ | `params["mktrf"]` from per-stock OLS |
| $\hat{\sigma}^2_{\varepsilon,i}$ | `mse_resid` from per-stock OLS (= RSS / (T − K − 1)) |

### 6.4 Solution method

The problem is a **quadratic program** (linear objective, linear constraints, box bounds) solved via Sequential Least Squares Programming (SLSQP). With a linear objective and linear constraints it reduces to a linear program, but SLSQP handles the box bounds cleanly.

With only one equality constraint (beta-neutrality), the optimizer has more freedom than a dollar-neutral formulation and will generally find a higher-alpha solution by allowing a net long or short tilt.

The gradient of the objective is constant:

$$\nabla_w \left(\sum_i \frac{\hat{\alpha}_i}{\hat{\sigma}^2_{\varepsilon,i}} w_i\right) = \left(\frac{\hat{\alpha}_1}{\hat{\sigma}^2_{\varepsilon,1}}, \ldots, \frac{\hat{\alpha}_N}{\hat{\sigma}^2_{\varepsilon,N}}\right)$$

### 6.5 Interpretation of results

| Result | Meaning |
|--------|---------|
| $w_i > 0$ | Long — stock has positive appraisal ratio after beta adjustment |
| $w_i < 0$ | Short — stock has negative appraisal ratio; shorting it adds alpha |
| $\sum_i \hat{\beta}_i w_i \approx 0$ | Portfolio has no net market exposure |
| $\sum_i w_i \neq 0$ (generally) | Net long or short tilt — unconstrained |
| $\sum_i \hat{\alpha}_i w_i$ | Expected monthly alpha of the optimized portfolio |

### 6.6 Limitations

- Alphas from a factor model are **backward-looking** — high historical alpha does not guarantee future alpha
- The model ignores **transaction costs**, **liquidity**, and **short-selling constraints**
- Idiosyncratic variances are estimated with error, especially for short histories
- Net exposure is unconstrained — the optimizer will tilt net long or short wherever alpha is highest; add $\sum w_i = 0$ as a second constraint if dollar-neutrality is required

---

## 7. Return Attribution

### 7.1 Decomposing returns into sources

Given full-sample OLS estimates $\hat{\alpha}$, $\hat{\beta}_k$, the realized return in month $t$ is exactly:

$$r_{p,t} - r_{f,t} = \underbrace{\hat{\alpha}}_{\text{alpha}} + \underbrace{\sum_k \hat{\beta}_k \cdot F_{k,t}}_{\text{factor contributions}} + \underbrace{\hat{\varepsilon}_t}_{\text{idiosyncratic}}$$

Each term represents a distinct economic source:

| Term | Interpretation |
|------|---------------|
| $\hat{\alpha}$ | Constant excess return (skill, mispricing, or omitted factor) |
| $\hat{\beta}_k \cdot F_{k,t}$ | Return from exposure to factor $k$ in month $t$ |
| $\hat{\varepsilon}_t$ | Return from stock-specific events (earnings surprises, news) |

The stacked area chart in Factor Lens shows these three components each month.

---

## 8. Variance Decomposition

### 8.1 Total return variance

$$\text{Var}(r_p - r_f) = \text{Var}\!\left(\sum_k \hat{\beta}_k F_k\right) + \text{Var}(\hat{\varepsilon})$$

The cross term is zero by the Frisch-Waugh theorem (OLS residuals are orthogonal to regressors).

### 8.2 Factor variance (full covariance form)

The variance of the factor-driven component is:

$$\text{Var}\!\left(\sum_k \hat{\beta}_k F_k\right) = \boldsymbol{\hat{\beta}}^\top \boldsymbol{\Sigma}_F \boldsymbol{\hat{\beta}}$$

where $\boldsymbol{\Sigma}_F$ is the $K \times K$ covariance matrix of factor returns. **Using only the diagonal** (i.e., treating factors as uncorrelated) understates this when factors co-move — as MKT, SMB, and CMA often do.

### 8.3 Idiosyncratic fraction

The cleanest decomposition uses R² directly:

$$\text{Idio fraction} = 1 - R^2, \qquad \text{Factor fraction} = R^2$$

This is exact by definition: $R^2 = 1 - \text{RSS}/\text{TSS}$.

### 8.4 Attributing R² across individual factors

Since cross-factor covariance terms are ambiguous to assign, Factor Lens uses proportional allocation:

$$\text{fraction}_k = \frac{\hat{\beta}_k^2 \cdot \text{Var}(F_k)}{\sum_j \hat{\beta}_j^2 \cdot \text{Var}(F_j)} \cdot R^2$$

Each factor's share is proportional to its standalone variance contribution ($\hat{\beta}_k^2 \sigma_k^2$), scaled so they sum to $R^2$. Cross-covariance terms are distributed implicitly.

### 8.5 Interpretation

| R² | Interpretation |
|----|---------------|
| 0.85–0.95 | Well-diversified large-cap portfolio; factors dominate |
| 0.50–0.80 | Moderate concentration or mid-cap exposure |
| < 0.30 | Concentrated or speculative micro-cap portfolio; idio dominates |

A low R² does not mean the model is wrong — it means the stocks have high idiosyncratic risk that factors cannot capture.

---

## 9. Constructed Factors (Quintile Spreads)

Factors can also be constructed from stock characteristics using a long-short portfolio:

### 9.1 Procedure (each month)

1. Rank all stocks in the universe by a characteristic (e.g., trailing volatility)
2. Form quintiles (5 equal groups)
3. Long quintile 5 (highest), short quintile 1 (lowest)
4. Factor return = return of long leg minus return of short leg

$$F_{k,t} = \bar{r}_{\text{Q5},t} - \bar{r}_{\text{Q1},t}$$

### 9.2 Characteristics used (available in `src/factors.py`)

| Factor | Characteristic | Source |
|--------|---------------|--------|
| Volatility | Trailing 12m std(ret) | CRSP |
| Liquidity | Trailing 12m avg dollar volume (log) | CRSP |
| Log Size | log(market cap) | CRSP |
| Leverage | (LT debt + ST debt) / total assets | Compustat |
| Growth | Year-over-year sales growth | Compustat |
| Value | Book equity / market cap | Compustat |

### 9.3 Look-ahead bias prevention

Compustat annual reports are only available ~6 months after fiscal year-end. To avoid using future information, fundamentals are lagged by 6 months before computing any characteristic.

---

## 10. Universe Construction

### 10.1 Top-N by market cap

Each month, rank all eligible stocks by market capitalization and keep the top 3,000:

$$\text{Universe}_t = \{i : \text{rank}(\text{mcap}_{i,t}) \leq 3000\}$$

Eligibility filters (from CRSP):
- `shrcd ∈ {10, 11}` — ordinary common shares only (excludes REITs, ADRs, ETFs)
- `exchcd ∈ {1, 2, 3}` — NYSE, AMEX, NASDAQ only

### 10.2 Market cap calculation

$$\text{mcap}_{i,t} = |P_{i,t}| \times \text{shrout}_{i,t}$$

CRSP sometimes reports negative prices (indicating a bid-ask midpoint was used). The absolute value is taken to get a positive market cap.

---

## 11. Key Identities Summary

| Quantity | Formula |
|----------|---------|
| Portfolio beta | $\beta_p = \sum_i w_i \beta_i$ |
| Market-neutral short ratio | $x = \beta_A / \beta_B$ (dollars) |
| OLS estimator | $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| R² | $1 - \text{RSS}/\text{TSS}$ |
| Idio variance fraction | $1 - R^2$ |
| Total factor variance | $\boldsymbol{\hat{\beta}}^\top \boldsymbol{\Sigma}_F \boldsymbol{\hat{\beta}}$ |
| Cumulative return | $\prod_t (1 + r_t) - 1$ |
| Annualized return | $(1 + r_{\text{cum}})^{12/T} - 1$ |
| Annualized volatility | $\sigma_{\text{monthly}} \times \sqrt{12}$ |

---

## References

- Fama, E. F. & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds.* Journal of Financial Economics, 33(1), 3–56.
- Fama, E. F. & French, K. R. (2015). *A five-factor asset pricing model.* Journal of Financial Economics, 116(1), 1–22.
- Carhart, M. M. (1997). *On persistence in mutual fund performance.* Journal of Finance, 52(1), 57–82.
- Treynor, J. & Black, F. (1973). *How to use security analysis to improve portfolio selection.* Journal of Business, 46(1), 66–86.
- Barra, R. (1994). *Multiple-factor models.* BARRA Research.
