# -*- coding: utf-8 -*-
"""Step 1a. Split- and distribution-adjusted monthly returns from official daily quotes.

Daily adjusted return of stock i on day d:
    r_d = close_d / ref_d - 1,
where ref_d is the previous close on ordinary days and the exchange's reference price on corporate-action days.
Corporate-action days are identified in two ways:
  (a) the stock-day appears in an official TWSE table (ex-right/ex-dividend results from 2003-05-05,
      capital-reduction and par-value-change resumption reference prices from 2011-01-01);
  (b) the daily quote reports a zero price change although the close differs from the previous close, i.e. the
      exchange did not compare the close with the previous close.
Reference price: the official TWSE reference price. When no official reference price exists (ex-rights before
2003-05-05, capital reductions before 2011, events while a stock traded on TPEx) the return of that single day is
set to zero (neutral) and the day is flagged 'unresolved' in the audit file. Because daily price limits are 7%
(10% from 2015-06) on such days, an official reference implying more than +-11% per elapsed trading day is rejected.
A zero reported change after one or more non-trading days with a move within the price limit is treated as an
ordinary return (the exchange resets the reference price of non-traded stocks), not as a corporate action.
Yahoo Finance adjusted prices are NOT used in the construction; they serve only as an external check (09_validate).
Monthly return = compounded daily adjusted returns.

Outputs: data/monthly_public.parquet  (code x month: ret, close, volume, value, ndays)
         data/corporate_action_days.csv  (audit of every corporate-action day and how it was resolved)
"""
import numpy as np, pandas as pd, os, glob, warnings
from config import *
warnings.filterwarnings('ignore')
PUB = os.path.join(DATA, 'pub')
roc = lambda s: f"{int(s[:s.index('年')]) + 1911}-{s[s.index('年') + 1:s.index('月')]}-{s[s.index('月') + 1:s.index('日')]}"
num = lambda s: pd.to_numeric(s.astype(str).str.replace(',', ''), errors='coerce')
ex = pd.read_csv(os.path.join(PUB, 'twse_exright.csv'), dtype=str)
ex = pd.DataFrame({'code': ex['股票代號'].str.strip(), 'date': ex['資料日期'].map(roc), 'ref': num(ex['除權息參考價']), 'pre': num(ex['除權息前收盤價']), 'kind': 'exright'})
off = [ex]
for fn, kind in (('twse_reduction.csv', 'reduction'), ('twse_parchange.csv', 'parchange')):
    d = pd.read_csv(os.path.join(PUB, fn), dtype=str)
    dt = d['恢復買賣日期'].map(lambda s: f"{int(s.split('/')[0]) + 1911}-{s.split('/')[1]}-{s.split('/')[2]}")
    off.append(pd.DataFrame({'code': d['股票代號'].str.strip(), 'date': dt, 'ref': num(d['恢復買賣參考價']), 'pre': num(d['停止買賣前收盤價格']), 'kind': kind}))
off = pd.concat(off).dropna(subset=['ref']); off = off[off.ref > 0].drop_duplicates(['code', 'date'], keep='last')
OFFBY = {c: g[['date', 'ref', 'kind']].sort_values('date').to_numpy().tolist() for c, g in off.groupby('code')}

files = sorted(glob.glob(os.path.join(PUB, 'px', '*.parquet')))
files = [f for f in files if 'close' in pd.read_parquet(f).columns]                                   # skip stocks with no quotes
cnt = pd.concat([pd.read_parquet(f, columns=['date', 'close']).query('close > 0')['date'] for f in files]).value_counts()
CAL = np.array(sorted(cnt[cnt >= 50].index)); CALPOS = {d: i for i, d in enumerate(CAL)}        # market trading calendar

rows, audit, EMERGING = [], [], []
for fn in files:
    c = os.path.basename(fn)[:4]; d = pd.read_parquet(fn)
    if len(d) == 0: continue
    d = d[(d.close > 0) & d.date.isin(CALPOS)].sort_values('date').reset_index(drop=True)
    # drop the emerging-board (negotiated average price) period that precedes exchange listing: closes off the tick grid
    big = d[d[['open', 'max', 'min', 'close']].min(axis=1) >= 10]
    if len(big):
        off4 = (np.round(big[['open', 'max', 'min', 'close']] * 100).astype(int) % 5 != 0).any(axis=1)
        offtick = off4.groupby(big.date.str[:7]).agg(['mean', 'size'])
        bad = offtick[(offtick['mean'] > 0.2) & (offtick['size'] >= 5)]
        if len(bad):
            EMERGING.append((c, bad.index.max())); d = d[d.date.str[:7] > bad.index.max()].reset_index(drop=True)
    if len(d) < 2: continue
    dates = d.date.to_numpy(); close = d.close.to_numpy(float); prev = np.r_[np.nan, close[:-1]]; r = close / prev - 1.0
    pos = np.array([CALPOS[x] for x in dates]); gap = np.r_[False, np.diff(pos) > 1]                 # stock did not trade on the previous market day
    notcmp = (d.spread.to_numpy(float) == 0) & (np.abs(close - prev) > 0.011)                        # exchange did not compare with previous close
    # official events are attached to the first day the stock trades on or after the official date (within 15 market days)
    offday = {}
    for dt, ref, kind in OFFBY.get(c, []):
        j = np.searchsorted(dates, dt)
        late = CALPOS[dates[j]] - np.searchsorted(CAL, dt) if 0 < j < len(dates) else 99
        if late <= 15: offday[j] = (ref, kind, late)
    for i in range(1, len(d)):
        dt = dates[i]; how = None
        if i in offday:
            ri = close[i] / offday[i][0] - 1.0
            if abs(ri) <= 0.11 * (1 + offday[i][2]): how = 'official_' + offday[i][1]
        if how is None and (i in offday or (notcmp[i] and (not gap[i] or abs(r[i]) > 0.11))):
            ri, how = 0.0, 'unresolved'
        if how is not None:
            audit.append((c, dt, prev[i], close[i], r[i], ri, how)); r[i] = ri
    d['r'] = r; d['m'] = d.date.str[:7].str.replace('-', '/')
    g = d.groupby('m').agg(ret=('r', lambda x: np.prod(1.0 + x.dropna()) - 1.0), close=('close', 'last'),
                           volume=('Trading_Volume', 'sum'), value=('Trading_money', 'sum'), ndays=('r', 'size'))
    g.iloc[0, g.columns.get_loc('ret')] = np.nan                      # first calendar month: no full-month return
    g['code'] = c; rows.append(g.reset_index())
M = pd.concat(rows, ignore_index=True); M.to_parquet(os.path.join(DATA, 'monthly_public.parquet'))
A = pd.DataFrame(audit, columns=['code', 'date', 'prev_close', 'close', 'raw_return', 'adjusted_return', 'resolved_by'])
A.to_csv(os.path.join(DATA, 'corporate_action_days.csv'), index=False)
print('stocks', M.code.nunique(), 'stock-months', len(M))
A['era'] = np.where(A.date < '2003-05-05', '1999-2003/04', np.where(A.date < '2011-01-01', '2003/05-2010', '2011-2025'))
print(A.groupby(['era', 'resolved_by']).size().unstack(fill_value=0))
pd.DataFrame(EMERGING, columns=['code', 'last_emerging_month']).to_csv(os.path.join(DATA, 'emerging_period_dropped.csv'), index=False); print('emerging periods dropped:', len(EMERGING))
pd.Series(CAL).to_csv(os.path.join(DATA, 'market_calendar.csv'), index=False, header=['date'])
