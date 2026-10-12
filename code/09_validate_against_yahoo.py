# -*- coding: utf-8 -*-
"""External check only: compare our adjusted monthly returns with Yahoo Finance adjusted-close monthly returns
for currently listed stocks. Yahoo data are NOT used anywhere in the construction."""
import pandas as pd, numpy as np, glob, os
from config import *
M = pd.read_parquet(os.path.join(DATA, 'monthly_public.parquet')); rows = []
for fn in glob.glob(os.path.join(DATA, 'pub', 'yahoo', '*.parquet')):
    y = pd.read_parquet(fn)
    if len(y) < 300 or 'Adj Close' not in y: continue
    y['m'] = y.date.str[:7].str.replace('-', '/'); a = y.groupby('m')['Adj Close'].last()
    r = a.pct_change().rename('yahoo_ret').reset_index(); r['code'] = os.path.basename(fn)[:4]; rows.append(r)
X = M.merge(pd.concat(rows), on=['code', 'm']).dropna(subset=['ret', 'yahoo_ret']); X['d'] = (X.ret - X.yahoo_ret).abs(); out = []
for a, b in [('2000/02', '2003/04'), ('2003/05', '2010/12'), ('2011/01', '2025/02'), ('2012/01', '2024/12')]:
    x = X[(X.m >= a) & (X.m <= b)]
    out.append(dict(period=f'{a}-{b}', stock_months=len(x), mean_ours_pct=100 * x.ret.mean(), median_ours_pct=100 * x.ret.median(), median_yahoo_pct=100 * x.yahoo_ret.median(),
                    share_abs_diff_below_1pct=(x.d < 0.01).mean(), share_abs_diff_above_10pct=(x.d > 0.10).mean(), rank_correlation=x.ret.corr(x.yahoo_ret, method='spearman')))
out = pd.DataFrame(out).round(4); out.to_csv(os.path.join(OUT, 'deliverables', 'audit_adjusted_return_check_vs_yahoo.csv'), index=False); print(out.to_string())
