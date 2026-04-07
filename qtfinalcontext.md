# FIN 452 — Quantitative Trading Strategy: Final Context File
**Author:** Ken Sucento | **Date:** April 7, 2026
**File:** `quantitative_trading.qmd` | **Output:** Single self-contained HTML (`embed-resources: true`)

---

## Project Summary

A quantitative pairs trading strategy applied to 10 S&P 500 Select Sector SPDR ETFs. The central question: can two sector ETFs that historically move together be traded profitably when their spread temporarily diverges? The answer flows through a four-step filtering pipeline, culminating in a Kalman-filter-driven signal and a full optimization + out-of-sample evaluation.

---

## Universe & Data

- **Tickers:** SPY, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLV, XLU, XLY
- **Source:** `tidyquant::tq_get()` | Adjusted close + open | Jun 2000 – Mar 2026
- **Excluded from clustering:** XLC (only ~6.5 yrs history) and XLRE (only ~9 yrs history)
- **Train period:** Jun 2000 – Dec 2024
- **Test period:** Jan 2025 – Mar 2026 (15 months)
- **Price transforms:** Log returns for clustering; log prices for cointegration, OU, and Kalman
- **Open prices:** Adjusted via `open * (adjusted / close)` ratio to correct for splits/dividends

---

## Four-Step Pipeline

### Step 1 — K-Means Clustering (k = 3)
- Cluster on daily log returns (10 tickers, training period only)
- Preprocessing: `recipes::step_normalize()`, then transpose to get tickers as rows
- `stats::kmeans(centers = 3, nstart = 25, seed = 452)`
- k selected by elbow plot + Ward's dendrogram (`hclust(method = "ward.D2")`)
- Silhouette avg ~0.16 — low but expected (all ETFs share ~1.0 SPY beta)
- Stability check: pre-2020 vs post-2020 clusters compared
- **Clusters:** Defensive (XLP, XLU) | Value/Cyclical (XLE, XLF) | Growth (SPY, XLB, XLI, XLK, XLV, XLY)
- **Why log returns, not prices:** Prices drift over 24 years; returns strip drift and isolate co-movement
- **Why no macro variables:** Already embedded in ETF returns — explicit inclusion double-counts

### Step 2 — Engle-Granger Cointegration (ADF p < 0.05)
- All within-cluster pairs generated (no cross-cluster, no self-pairs)
- OLS regression of `log(P_A) ~ log(P_B)` → estimate static hedge ratio `beta.ols`
- ADF test on OLS residuals (the spread): reject H0 (unit root) at p < 0.05 → cointegrated
- `tseries::adf.test()` used
- **Why cointegration, not correlation:** Correlation ≠ bounded spread. A drifting spread compounds losses forever
- Most failures from Defensive and Value/Cyclical clusters
- Survivors predominantly from Growth cluster (shared technology/large-cap exposure)

### Step 3 — OU Half-Life (≤ 252 days)
- Ornstein-Uhlenbeck process: regress `Δspread_t ~ spread_{t-1}`
- `theta` = coefficient on lagged spread (must be negative for mean reversion)
- `half.life = -log(2) / theta`
- Ceiling of 252 trading days (more generous than typical 60-day equity cutoff — sector ETFs revert slowly by construction)
- Pairs with `theta ≥ 0` or `half.life > 252` discarded
- **Tradeable pairs:** all from Growth cluster; traded simultaneously with equal weight
- Fastest reverting: XLV/XLY (Health Care vs Consumer Discretionary)
- `pairs.tradeable` = all pairs passing HL filter, sorted by half-life ascending

### Step 4 — Kalman Filter + Signal Generation
- **Why Kalman over static OLS beta:** Static beta drifts across business cycles; Kalman updates `β_t` daily
- Calibration follows Yang, Huang & Chen (2023): `Q = α² / (1−α) × H`, where `α = 2 / (n.ma + 1)`
- `H.est = var(vec.A - vec.B)` (observation noise), `Q.est = q * H.est` (process noise)
- Kalman gain: `K = P.pred * B_i / (B_i² * P.pred + H)`
- Spread: `spread_t = log(P_A,t) − β_t × log(P_B,t)`
- Z-score: 20-day rolling mean + SD (`slider::slide_dbl(.before = 19)`)
- **Signals:** Z > +z.entry → short spread (-1) | Z < -z.entry → long spread (+1) | |Z| > z.stop → flat (0)
- **Execution:** MOO (market-on-open) orders next-day open → avoids look-ahead bias on fill price
- Returns computed as CC (close-to-close), CO (close-to-open overnight), OC (open-to-close intraday)

---

## Portfolio Construction

- **Function:** `strategy_pair()` for single pair; `portfolio_pairs()` wraps across all tradeable pairs
- **Allocation:** Equal-weight across all tradeable pairs
- **Rebalancing:** Monthly (end of each calendar month) back to equal weight
- **z.stop = 3** (hard stop, fixed — not optimized)

---

## Optimization (Section 8.7)

- **Parameters:** `z.entry` ∈ [0.50, 2.75] by 0.25 | `n.ma` ∈ [10, 120] by 10
- **Grid:** `expand.grid()` → 110 combinations
- **Sweep:** `purrr::map2()` with `purrr::possibly()` for safe evaluation
- **Metrics per combo:** `RTL::tradeStats()` → Sharpe, CumReturn, DD.Max, Omega, %.Win, %.InMrkt, Ret.Ann, SD.Ann
- **Cache:** `#| cache: true` on the grid sweep chunk — stale cache is the root cause of train/test parameter mismatch (see caching note below)
- **Visualization:** 2D heatmaps (patchwork) + 3D plotly surfaces for Sharpe, CumReturn, DD.Max

### Z-Score Normalization
- Each metric standardized: `value.z = (x − mean) / sd` across the grid
- `res.z` is the long-format dataframe with Z-scores per (z.entry, n.ma, variable)

### Composite Z-Score
- `composite.z = Z(Sharpe) + Z(CumReturn) + Z(DD.Max)` per grid point
- DD.Max Z-score direction is naturally correct: less negative DD.Max → higher Z → higher composite

### Plateau Method (Robust Selection)
- For each grid point, compute average composite of ±1 step neighbors
- `robustness = pmin(composite.z, neighbor.avg)`
- Penalizes isolated spikes; rewards broad elevated regions
- **Key distinction:** The 3D composite surface shows `composite.z` (raw sum of Z-scores). The robust selection table ranks by `robustness`. These two peaks are INTENTIONALLY at different coordinates — that is the entire purpose of the plateau criterion.
- `robust.pick = top 1 row sorted by robustness desc`

### Risk Appetite Filter (Section 8.9)
- `optimal = res %>% filter(z.entry == robust.pick$z.entry, n.ma == robust.pick$n.ma)`
- Secondary floor: Sharpe > 0 and DD.Max > −0.50

---

## Risk / Reward Measures (Section 8.8)

All computed on `result.opt` (full train rerun with selected parameters):
- **Sharpe Ratio:** `SR = mean(ret) / sd(ret)` (rf = 0 assumed), via `RTL::tradeStats()`
- **Omega Ratio:** `PerformanceAnalytics::Omega()`
- **Max Drawdown:** via `RTL::tradeStats()` — DD.Max column
- **SPY correlation:** `PerformanceAnalytics::chart.Correlation()` — near-zero confirms market neutrality
- **%.Win, %.InMrkt, N.Trades:** from `RTL::tradeStats()`

---

## Out-of-Sample Evaluation (Section 8.10)

- Parameters from `optimal` (robust.pick) applied to `log.prices.wide.test` and `log.open.wide.test`
- **No refitting** — all Kalman state, hedge ratios, Z-score parameters derived from test data forward from zero state (not from training state)
- Train vs Test comparison table: `dplyr::bind_rows()` of `stats.train` and `stats.test`

### Actual Out-of-Sample Results (Jan 2025 – Mar 2026)
| Metric | Train | Test |
|---|---|---|
| Sharpe | 0.5336 | 0.2145 |
| CumReturn | 1.6197 | 0.0124 |
| Ret.Ann | 0.0400 | 0.0100 |
| SD.Ann | 0.0750 | 0.0468 |
| DD.Max | −0.1952 | −0.0344 |
| Omega | 0.1666 | 0.0542 |
| %.Win | 0.5106 | 0.4619 |
| %.InMrkt | 0.6611 | 0.6355 |
| DD.Length | 1,545 | 96 |

### Why Degradation Occurred
- Regime change: tariff shocks, rate cuts, sector rotation in 2025–2026
- Short test window: ~3 half-life cycles max per pair — insufficient for statistical confidence
- Parameter fit: optimal params from 24-yr training regime; some fit does not transfer

---

## Known Inconsistency — Cache Warning

The optimization grid chunk has `#| cache: true`. If you see the out-of-sample subtitle showing `n.ma = 40` but the robust table showing `n.ma = 100` (or any mismatch), this means cached results from a prior run are active. **Fix: delete the `quantitative_trading_cache/` folder and fully re-render.** All downstream objects (`composite`, `robust.pick`, `optimal`) chain from `res`, so a stale cache propagates throughout.

---

## Weaknesses (Section 8.11)

| Weakness | Severity |
|---|---|
| Narrow universe (10 large-cap US sectors) | High |
| Fixed train/test split | Medium |
| No regime detection | Medium |
| Look-ahead in data construction | Medium |
| Short out-of-sample window (15 months) | Medium |

---

## Literature Flow

| Paper | Step Supported |
|---|---|
| Cartea, Cucuringu & Jin (2023) — SSRN 4560455 | Step 1: Clustering |
| Yang, Huang & Chen (2023) | Steps 1, 2, 4: Clustering + Kalman blueprint |
| Simonson (2018) — U of Washington | Step 3: OU process for ETF spreads |
| Avellaneda & Lee (2008) — SSRN 1153505 | Step 4: Kalman filter application |
| Engle & Granger (1987) | Step 2: Cointegration / ADF test |
| Kalman (1960) | Step 4: State-space filtering |

---

## Key R Packages

`tidyverse`, `tidyquant`, `tidymodels`, `purrr`, `lubridate`, `plotly`, `patchwork`, `gt`, `cluster`, `factoextra`, `tseries`, `KFAS`, `broom`, `slider`, `TTR`, `PerformanceAnalytics`, `RTL`

---

## Document Structure

1. Mental Model (PDF iframe — `qtmm.pdf`)
2. Data (ETF universe, train/test split)
3. Background Setup (log prices/returns, train/test matrices)
4. Step 1 — Clustering
5. Step 2 — Cointegration Testing
6. Step 3 — OU Half-Life
7. Step 4 — Kalman Filter and Strategy
8. Optimization (8.7 grid, 8.8 heatmaps, 8.9 3D surfaces, Z-score, plateau, risk appetite)
9. Risk / Reward Measures (Sharpe, Omega, DD, SPY correlation)
10. Out-of-Sample Evaluation (8.10 train vs test table + equity curve)
11. Literature Flow (mermaid LR diagram)
12. Weaknesses (gt table)
13. Conclusion
14. Things to Improve on
