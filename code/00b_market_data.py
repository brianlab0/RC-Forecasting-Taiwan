# -*- coding: utf-8 -*-
"""Market-level series: TAIEX monthly return and monthly risk-free rate.

TAIEX   : month-end close of the TWSE Capitalization Weighted Stock Index from official TWSE daily
          quotes (via FinMind TaiwanStockPrice, data_id=TAIEX; includes Saturday make-up sessions).
          It is a price index (the benchmark named in the manuscript). Because stock returns are now
          dividend-adjusted, the TAIEX Total Return Index is added as the like-for-like benchmark.
Risk-free: Bank of Taiwan one-month time-deposit rate (fixed), annual % -- Central Bank of the
          Republic of China (Taiwan), Financial Statistics Monthly, table "Interest rates of banks"
          (open-data file EG2WM01). The rate posted at the end of month m-1 is de-annualised
          (/12) and applied to return month m, so it is known at the formation date.
"""
import json, os, pandas as pd, numpy as np
from config import *
tx = pd.read_csv(os.path.join(DATA, 'taiex_monthend_finmind.csv'), index_col=0)['close']
j = json.load(open(os.path.join(DATA, 'cbc_EG2WM01.json'), encoding='utf-8'))
st = j['data']['structure']
assert st['Table1'][0]['data'] == '台灣銀行' and st['Table2'][0]['data'] == '定期存款-一個月' and st['Table3'][1]['data'] == '固定'
rate = {r[0].replace('M', '/'): float(r[2]) for r in j['data']['dataSets'] if r[2] not in ('-', '')}
rate = pd.Series(rate).sort_index()
months = month_list('2007/01', '2024/12'); prev = month_list('2006/12', '2024/11')
df = pd.DataFrame({'month': months,
                   'TAIEX_close': tx.reindex(months).values,
                   'TAIEX_return': (tx.reindex(months).values / tx.reindex(prev).values - 1),
                   'rf_annual_pct_prev_month_end': rate.reindex(prev).values})
df['risk_free_return'] = df['rf_annual_pct_prev_month_end'] / 100 / 12
# Stock returns are dividend-adjusted, so the market benchmark must be on the same (total-return) basis:
# TAIEX_return = return of the TAIEX Total Return Index (official TWSE series, dividends reinvested; via FinMind
# TaiwanStockTotalReturnIndex). The price-index return is kept only as a supplementary column.
tr = pd.read_csv(os.path.join(DATA, 'pub', 'taiex_total_return_index_daily.csv')); tr['m'] = tr.date.str[:7].str.replace('-', '/')
tr = tr.sort_values('date').groupby('m')['price'].last()
df = df.rename(columns={'TAIEX_close': 'TAIEX_price_index_close', 'TAIEX_return': 'TAIEX_price_index_return'})
df.insert(1, 'TAIEX_total_return_index', tr.reindex(months).values)
df.insert(2, 'TAIEX_return', tr.reindex(months).values / tr.reindex(prev).values - 1)
assert df.notna().all().all()
df.to_csv(os.path.join(DATA, 'market_monthly.csv'), index=False)
print(df.head(3)); print(df.tail(3)); print(df.describe().round(5))
