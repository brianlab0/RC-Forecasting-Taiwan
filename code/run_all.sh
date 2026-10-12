#!/bin/bash
# Full pipeline (public-data, adjusted-return version). Steps 00* need internet; everything else is offline and deterministic.
set -e; cd "$(dirname "$0")"
python3 00e_fetch_twse_events.py          # TWSE ex-right / capital-reduction / par-change reference prices
python3 00d_fetch_public_daily.py         # official daily quotes + shares issued, all TWSE common stocks incl. delisted (FinMind, ~6.5 h on the free tier)
python3 00f_fetch_yahoo.py                # Yahoo adjusted prices: external check only
python3 01a_adjusted_returns.py           # split- and distribution-adjusted monthly returns
python3 00g_fetch_altered_trading.py      # TWSE full-cash-delivery (altered trading) list at each month-end
python3 01_build_panel.py                 # eligibility filters, six signals, realized RCs, liquidity
python3 00b_market_data.py                # TAIEX, TAIEX total return index, risk-free rate
python3 02_forecast_linear_trees.py       # benchmarks, OLS, tree ensembles
python3 03_forecast_nn.py                 # NN1-NN5
python3 04_portfolios.py                  # portfolios, turnover, costs (all universes)
python3 05_tables_figures.py              # Tables 4-7, appendix tables, figures, audit files 9-1, 9-2, 9-3, 9-5
python3 06_holdings_audit.py              # audit file 9-4
python3 07_appendix_A1.py                 # model configurations
python3 08_tables_2_3.py                  # Tables 2 and 3
python3 09_validate_against_yahoo.py      # external check of the adjusted returns
