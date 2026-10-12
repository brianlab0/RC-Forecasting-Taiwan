# Forecasting Time-Varying Cross-Sectional Regression Coefficients with Machine Learning: Evidence from Taiwan

Complete outputs and code for the regression-coefficient (RC) forecasting experiments, adjusted-return version (12 October 2026).

Every table, figure and audit CSV is produced by the single pipeline in `code/`, from one forecast panel and one monthly return series.

## Repository layout

```
.
├── README.md      this document
├── tables/        Tables 2–7, Appendix Tables A.1–D.1, monthly risk-free rate and TAIEX series
├── figures/       Figure 1 (six panels), Figure 2 (four panels) and the underlying data
├── audit/         monthly audit files (9-1) (9-2) (9-3) (9-5), adjusted-return and hyperparameter audits
└── code/          full pipeline; run_all.sh gives the execution order
```

Audit file (9-4), `monthly_stock_scores_and_holdings.csv`, has about 2.17 million rows and is 770 MB uncompressed, which exceeds GitHub's per-file limit. It is attached as a gzip file on the [Releases](../../releases) page. pandas reads `.csv.gz` directly.

## Contents

1. [What changed in this version](#1-what-changed-in-this-version)
2. [Data sources and return adjustment](#2-data-sources-and-return-adjustment)
3. [Remaining differences from the manuscript](#3-remaining-differences-from-the-manuscript)
4. [Results](#4-results)
5. [Implementation definitions](#5-implementation-definitions)
6. [File map](#6-file-map)
7. [How to reproduce](#7-how-to-reproduce)

## 1. What changed in this version

The previous version used unadjusted month-end closing prices for 964 surviving firms, 333 of which were in fact TPEx (OTC) stocks. This version rebuilds the data following Sections 2.1 and 2.2 of the manuscript:

| Manuscript | This version |
|---|---|
| Split- and distribution-adjusted prices | Returns adjusted for ex-rights, ex-dividends, capital reductions and par-value changes, computed from TWSE daily quotes and official reference prices |
| Common equities listed on the Taiwan Stock Exchange | TWSE-listed common shares only; TPEx stocks, ETFs, TDRs and preferred shares are excluded |
| Includes delisted, merged and suspended equities | Firms on the TWSE list of terminated listings are included |
| Sample January 2000 – December 2024 | Returns start in January 2000 |
| Exclude stocks with fewer than 12 months of return history | Applied |
| Exclude full-cash-delivery and reorganisation stocks | Stocks on the TWSE altered-trading-method list at each month-end are excluded, from December 2004 |
| Valid market capitalisation and trading volume at t−1 | Applied (positive trading volume and market capitalisation in the formation month) |
| Table 1 signal definitions | mom6m uses 5 months and mom12m uses 11 months (see Section 3) |
| Two-digit industry | TWSE two-digit industry codes |
| TAIEX and portfolio returns on the same return basis | Portfolio returns include dividends, so the TAIEX Total Return Index is used throughout |

The data source is official public data, not TEJ.

## 2. Data sources and return adjustment

**Sources**

- Daily quotes (close, price change, volume, trading value): TWSE daily closing quotes, retrieved through FinMind `TaiwanStockPrice`.
- Ex-right and ex-dividend reference prices: TWSE ex-right/ex-dividend calculation results (TWT49U), from 5 May 2003; 17,638 records.
- Capital-reduction resumption reference prices: TWSE TWTAUU, from 1 January 2011; 389 records. Par-value-change reference prices: TWTB8U; 8 records.
- Altered-trading-method (full-cash-delivery) list: TWSE TWT85U, from 24 December 2004, taken on the last trading day of each month; 241 months, 158 distinct stocks.
- Terminated listings: TWSE list of delisted companies. Listing dates and industry codes: TWSE listed-company basic data.
- Shares issued: FinMind `TaiwanStockShareholding` (from February 2004), used for stocks not covered by the original market-capitalisation file. The 631 listed stocks already in that file (`raw_size.xlsx`) keep its values.
- Market: TAIEX Total Return Index (dividends reinvested), TWSE, through FinMind. The price index is kept only as a supplementary column. Risk-free rate: Central Bank of the Republic of China (Taiwan), Financial Statistics Monthly.

**Adjustment method**

- Daily adjusted return = close ÷ reference price − 1. On ordinary days the reference price is the previous close; on ex-right, ex-dividend, capital-reduction and par-value-change days it is the reference price published by the TWSE.
- The monthly return compounds the daily adjusted returns within the month.
- A corporate-action day is identified in two ways: it appears in an official table, or the daily quote reports a zero price change although the close differs from the previous close (the exchange did not compare the close with the previous close).
- When no official reference price exists for a corporate-action day, the return on that single day is set to zero. This covers ex-rights before 5 May 2003, capital reductions before 2011, and events that occurred while a stock traded on TPEx. The error is bounded by the daily price limit and no spurious return is created.
- Counts by period are in `audit/audit_corporate_action_days_summary.csv`: from 2011, 10,948 days with an official reference price and 467 without; May 2003 – 2010, 4,078 and 885; January 2000 – April 2003, all 1,297 days have no official reference price.
- Emerging-board quotes that precede exchange listing (negotiated average prices, off the tick grid) are removed.

**External check.** `audit/audit_adjusted_return_check_vs_yahoo.csv` compares these monthly returns with monthly returns from Yahoo Finance adjusted closes, stock-month by stock-month. Yahoo data are not used in any calculation. Over the out-of-sample period January 2012 – December 2024 (136,745 stock-months), 97.7% differ by less than one percentage point and the rank correlation is 0.9955.

## 3. Remaining differences from the manuscript

1. **No full-cash-delivery list before December 2004.** The TWSE altered-trading-method list starts on 24 December 2004, so the screen is not applied in earlier months. The validation period (2007–2011) and the out-of-sample period (2012–2024) are fully covered. The screen removes 4,446 stock-months over the whole sample.
2. **Delisted firms are incomplete for 2000–2004.** Public quotes for delisted firms mostly start in February 2004. The sample therefore includes delisted firms from 2005 onwards (covering both the validation and out-of-sample periods), but 2000–2004 effectively contains only firms that survived. Of the 1,057 stocks that ever enter the sample, 111 were later delisted.
3. **Return in the delisting month.** If a held stock does not trade at all in the following month (mostly merger delistings), its next-month return is set to zero. This affects 88 of 134,285 eligible out-of-sample stock-months.
4. **Time subscripts in Table 1.** The note to Table 1 defines r(t−k) as the return k months before formation month t, while Equation (5) uses Z(t) to predict r(t+1). Read literally, the formation-month return is never used. The code treats k = 1 as the formation month (the most recent known return): mom1m = r(t); mom6m = months t−4 to t (5 months); mom12m = months t−10 to t (11 months); mom36m = months t−35 to t−12; chmom = R(t−5..t) − R(t−11..t−6).
5. **TAIEX is the total return index.** Stock returns now include dividends, so `TAIEX_return` in every table, figure and audit file is the monthly return of the TAIEX Total Return Index, and win rates and alphas are computed against it. The price index is reported as a supplementary column (last column of Table 5; `vs_price_index` columns in Table D.1). The manuscript should state that TAIEX refers to the total return index.
6. **Shorter RC series before 2003.** With the sample starting in January 2000, the first realized RC has formation month December 2000 (December 2002 for mom36m). Table 2 Panel B has 288 and 264 observations.

## 4. Results

With adjusted returns and the full TWSE sample, the conclusions differ from the previous version:

| Item | Previous version (unadjusted, survivors) | This version |
|---|---|---|
| Tree-model R²OOS | Positive for four signals, 2%–5% | Positive for mom1m, mom6m, mom12m, chmom (0.8%–11.8%); negative for mom36m and indmom |
| LightGBM and GBRT R²OOS | LightGBM: four positive, two negative | Both: four positive, two negative (mom36m, indmom) |
| Highest equal-weighted net Sharpe | AR(1) 0.966, Historical mean 0.954 | LightGBM 0.821, OLS 0.766, GBRT 0.750, NN5 0.743 |
| Equal-weighted benchmark net Sharpe | 0.59–0.97 | 0.29–0.49 |
| Highest value-weighted net Sharpe | NN2 0.618 | NN5 0.707, OLS 0.681, Random Forest 0.660, NN1 0.637 |
| Value-weighted benchmark net Sharpe | 0.29–0.45 | 0.00–0.45 |
| Equal-weighted net Sharpe excluding the smallest 20% | Mostly falls to 0.15–0.45 | Most ML models stay at 0.59–0.81 |
| Oracle net Sharpe | 2.24–2.51 | 2.20–2.31 |

- **Portfolios built from model forecasts outperform the three benchmark forecasts**, for both equal and value weighting, after excluding small and illiquid stocks, and in both halves of the out-of-sample period.
- **Tree models forecast more accurately than OLS on every signal.** All four tree models have a higher R²OOS than OLS for all six signals; OLS is negative for four signals.
- **Portfolio performance of ML and OLS is mixed.** Equal-weighted: LightGBM first, OLS second. Value-weighted: NN5 first, OLS second. "Non-linear models outperform the linear model" holds for forecast accuracy but cannot be claimed for portfolio performance.
- **No model forecasts mom36m** (R²OOS negative for all models).
- **No alpha is statistically significant.** Against the total return index (Table D.1), the largest t-statistic is 1.67 (OLS, equal-weighted). The three benchmarks have negative alphas.
- **Comparison with the market.** The total return index has a Sharpe ratio of 0.912 (0.657 for the price index). No equal- or value-weighted portfolio exceeds 0.912, so "beating the market" cannot be claimed.
- **The selected equal-weighted cutoff is mostly 1%** (about 7–9 stocks), so holdings are concentrated. Table C.1 shows that most ML models have a higher net Sharpe at 5% and 10% (for example, Random Forest at 10%: 0.858).
- **Validation Sharpe ratios are low** (0.17–0.53) because 2007–2011 includes the global financial crisis.
- **Results are sensitive to the sample definition.** Adding the full-cash-delivery screen changed selected cutoffs and model rankings (for example, equal-weighted LightGBM moved from 0.64 to 0.82). Small differences between models should not be over-interpreted.

## 5. Implementation definitions

- **Months.** The out-of-sample period is target (return) months January 2012 – December 2024, 156 months, corresponding to formation months December 2011 – November 2024. The validation period is target months January 2007 – December 2011 (60 months).
- **Eligible stocks.** Listed on the TWSE in the formation month, positive trading volume in that month, positive market capitalisation, at least 12 months of returns, and not on the altered-trading-method list at month-end. Both the cross-sectional regressions for realized RCs and the portfolios use eligible stocks only.
- **Forecast information set** (identical for all ML models, 13 variables): the signal's own realized RC at lags 1, 2, 3, 6 and 12; rolling mean and standard deviation over the past 3, 6 and 12 months; sine and cosine of the target calendar month.
- **Hyperparameters.** One configuration per model, chosen by the highest mean validation R² (relative to the expanding historical mean) across the six signals and fixed before January 2012. Candidate grids and selected values are in `tables/appendix_A1_model_configurations.csv`.
- **AR(1).** Intercept and slope are re-estimated recursively each month on all history available at that time.
- **Portfolio universe.** Eligible stocks with all six signals available: 578 stocks in December 2006, 694 in December 2011, 927 in November 2024.
- **Z-scores.** Each signal is winsorized at the 1st and 99th percentiles and standardized within the portfolio universe.
- **Industry momentum.** Market-capitalisation-weighted average of mom12m across eligible stocks in the same TWSE two-digit industry, using formation-month market capitalisation. Delisted firms are mapped to TWSE codes through FinMind industry names.
- **Number of holdings.** max(1, round(N × k%)). Ties in the score are broken by stock code.
- **Turnover.** Equation (13); w⁻ is the previous portfolio's weight after drifting with the month's returns. There are no holdings before the first validation month, so its turnover is 0.5.
- **Sharpe ratio.** √12 × mean(r − rf) / sd(r − rf), sample standard deviation.
- **Newey–West.** Bartlett kernel, 6 lags, no small-sample correction (statsmodels HAC).
- **TAIEX.** `TAIEX_return` is the monthly return of the total return index; `TAIEX_price_index_return` is the price index, for reference only. Their average monthly returns over January 2012 – December 2024 are 1.147% and 0.845%.
- **Win rate.** Share of out-of-sample months in which the net return exceeds `TAIEX_return`.
- **Risk-free rate.** Bank of Taiwan one-month fixed time-deposit rate (annual %), from the Central Bank's Financial Statistics Monthly, "Interest rates of banks" (open-data file EG2WM01). Return month m uses the rate posted at the end of month m−1, divided by 12.
- **Break-even cost.** Average out-of-sample monthly gross return ÷ average monthly turnover, in basis points.
- **Size screen.** Each formation month the bottom 20% of the universe by market capitalisation is excluded; z-scores, scores, ranks, weights and turnover are then recomputed. Cutoff percentages are unchanged.
- **Liquidity screen.** Each formation month the bottom 20% by average daily trading value in that month is excluded.
- **COVID exclusion.** The 24 monthly returns of January 2020 – December 2021 are removed from the performance statistics only; the model estimation timeline is unchanged.

## 6. File map

| Item | Files |
|---|---|
| Table 2, Table 3 | `tables/table2_summary_statistics.csv`, `tables/table3_correlations.csv` (the data changed, so both tables need updating) |
| [1] Figure 1 | `figures/fig_realized_rc_series_(signal).png` (six files), `figures/fig_realized_rc_series_data.csv` |
| Table 4, Table A.2 | `tables/table4_r2oos.csv`, `tables/appendix_A2_mae.csv` |
| [2] Table 5 | `tables/table5_portfolio_performance.csv`, `tables/table5_TAIEX_reference.csv` (total return index and price index rows) |
| [3] Figure 2 | `figures/fig_cumulative_net_wealth_(a/b/c/d).png`, `figures/fig_cumulative_net_wealth_data.csv` (log y-axis, identical across the four panels; the plotted line is the total return index, and the data file also has a price-index column) |
| [4] Table 6 | `tables/table6_transaction_cost_investability.csv` |
| [5] Table 7 | `tables/table7_subperiod_stability.csv` |
| [6] Table A.1 | `tables/appendix_A1_model_configurations.csv` (for compiling the table) |
| Table B.1, Table C.1 | `tables/appendix_B1_oracle.csv`, `tables/appendix_C1_net_sharpe_all_cutoffs.csv` |
| [7] Table D.1 | `tables/appendix_D1_risk_adjusted_performance.csv`, `tables/risk_free_rate_and_TAIEX_monthly.csv` |
| (9-1) | `audit/monthly_rc_forecast_panel.csv` |
| (9-2) | `audit/monthly_signal_summary.csv` (the strategy has no selected-signal rule; `dominant_signal` is an ex-post summary column) |
| (9-3) | `audit/monthly_portfolio_returns.csv` (four extra columns: `universe` = base / ex_small / ex_illiquid, filter on base to reproduce Table 5; `n_holdings`; `n_eligible`; `TAIEX_price_index_return`) |
| (9-4) | `monthly_stock_scores_and_holdings.csv.gz` on the [Releases](../../releases) page (about 2.17 million rows, 770 MB uncompressed; too large for Excel, use pandas or R) |
| (9-5) | `audit/oracle_monthly_returns.csv` |
| Adjusted-return audit | `audit/audit_corporate_action_days_summary.csv`, `audit/audit_adjusted_return_check_vs_yahoo.csv` |
| Hyperparameter audit | `audit/audit_validation_grid_*.csv` |
| Code | `code/` (`run_all.sh` gives the execution order) |

## 7. How to reproduce

```bash
pip install -r code/requirements.txt
bash code/run_all.sh
```

- Steps prefixed `00` need internet access (TWSE, FinMind, Central Bank open data). The FinMind free tier allows about 300 requests per hour, so a full download takes about 6.5 hours; the downloaders are resumable.
- All other steps run offline and are deterministic (random seed 42); they take about 15 minutes.
- Downloaded raw data are not stored in this repository. The scripts write data to `data/` and outputs to `out/deliverables/`.
- `data/raw_size.xlsx` (month-end market capitalisation for 631 listed stocks) is a pre-existing file that is not in this repository and must be supplied before running.
