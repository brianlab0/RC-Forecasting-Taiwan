# -*- coding: utf-8 -*-
"""Step 1. Stock-level panel, six signals, monthly standardisation and realized RCs (public-data version).

Sample   : common shares listed on the Taiwan Stock Exchange, January 2000 - December 2024, including firms that
           were later delisted (TWSE list of terminated listings). ETFs, TDRs (91xx) and preferred shares excluded.
Returns  : split- and distribution-adjusted monthly returns from 01a_adjusted_returns.py.
Filters at each formation month t (manuscript Section 2.1):
           listed on the TWSE, at least 12 months of return history, positive market capitalisation and positive
           trading volume in month t, and not under an altered trading method (full-cash delivery; covers distressed
           and reorganising firms) at the end of month t (official TWSE list TWT85U, available from 2004/12).
Signals  : Table 1, with k = 1 denoting the most recent completed month (the formation month):
           mom1m = r_t; mom6m = prod_{k=1..5}; mom12m = prod_{k=1..11}; mom36m = prod_{k=13..36};
           chmom = R(1,6) - R(7,12); indmom = market-cap-weighted mom12m within the same two-digit TWSE industry.
Outputs  : data/panel.parquet, data/realized_rc.csv, data/adtv_monthly.parquet
"""
import numpy as np, pandas as pd, os, glob, calendar, warnings
from config import *
warnings.filterwarnings('ignore')
PUB = os.path.join(DATA, 'pub'); months = month_list(); T = len(months); MI = {m: i for i, m in enumerate(months)}
M = pd.read_parquet(os.path.join(DATA, 'monthly_public.parquet')); M = M[M.m.isin(MI)]
codes = sorted(M.code.unique()); N = len(codes); CI = {c: i for i, c in enumerate(codes)}
def mat(col):
    A = np.full((N, T), np.nan); A[M.code.map(CI).to_numpy(), M.m.map(MI).to_numpy()] = M[col].to_numpy(float); return A
R, PX, VOL, VAL = mat('ret'), mat('close'), mat('volume'), mat('value')
quoted = np.isfinite(PX)
first = np.where(quoted.any(1), quoted.argmax(1), T); last = np.where(quoted.any(1), T - 1 - quoted[:, ::-1].argmax(1), -1)
tt = np.arange(T)[None, :]; alive = (tt >= first[:, None]) & (tt <= last[:, None])
R[alive & ~quoted] = 0.0; VOL[alive & ~quoted] = 0.0; VAL[alive & ~quoted] = 0.0     # listed but not traded in the month
R[tt == first[:, None]] = np.nan                                                    # no full-month return in the first month
R[:, :MI['2000/01']] = np.nan                                                       # sample starts January 2000

# ---- listing status ---------------------------------------------------------------------------------------------
L = pd.read_csv(os.path.join(PUB, 'twse_listed_basic.csv'), dtype=str).drop_duplicates('公司代號').set_index('公司代號')
D = pd.read_csv(os.path.join(PUB, 'twse_delisted.csv'), dtype=str).sort_values('ymd').drop_duplicates('code', keep='last').set_index('code')
info = pd.read_csv(os.path.join(PUB, 'finmind_stockinfo.csv'), dtype=str)
listed = np.zeros((N, T), bool); name, industry, status = [], [], []
generic = {'電子工業', '化學生技醫療', '其他', '創新板股票', '創新版股票'}
fm_ind = info.groupby('stock_id')['industry_category'].agg(lambda s: next((x for x in s if x not in generic), s.iloc[0]))
cur_ind = L['產業別'].to_dict()
# map FinMind industry names to the TWSE two-digit industry code using currently listed firms
cur = pd.DataFrame({'code2': pd.Series(cur_ind), 'fm': fm_ind}).dropna()
name2code = cur.groupby('fm')['code2'].agg(lambda s: s.value_counts().index[0]).to_dict()
for c in codes:
    i = CI[c]
    if c in L.index and (c not in D.index or D.loc[c, 'ymd'] > '2025-02-28'):
        ld = L.loc[c, '上市日期']; lm = f'{ld[:4]}/{ld[4:6]}'
        listed[i] = np.array([m >= lm for m in months]); name.append(L.loc[c, '公司簡稱']); industry.append(cur_ind[c]); status.append('listed')
    else:
        dm = D.loc[c, 'ymd'][:7].replace('-', '/') if c in D.index else '9999/99'
        listed[i] = np.array([m <= dm for m in months]); name.append(D.loc[c, 'name'] if c in D.index else c)
        industry.append(name2code.get(fm_ind.get(c), 'X' + c[:2])); status.append('delisted')
name, industry, status = np.array(name), np.array(industry), np.array(status)

# ---- market capitalisation (million NTD) -------------------------------------------------------------------------
MC = np.full((N, T), np.nan)
mc = pd.read_excel(os.path.join(DATA, 'raw_size.xlsx'), sheet_name='市值', header=0)
mcode = mc.iloc[:, 0].astype(str).str.strip(); mmon = list(mc.columns[2:]); mv = mc.iloc[:, 2:].apply(pd.to_numeric, errors='coerce').to_numpy(float)
for r, c in enumerate(mcode):
    if c in CI: MC[CI[c], [MI[m] for m in mmon]] = mv[r]
src = np.where(np.isin(codes, mcode), 'TEJ market cap (raw_size.xlsx)', 'close x shares issued')
for fn in glob.glob(os.path.join(PUB, 'sh', '*.parquet')):
    c = os.path.basename(fn)[:4]
    if c not in CI or c in set(mcode): continue
    s = pd.read_parquet(fn)
    if len(s) == 0: continue
    s['m'] = s.date.str[:7].str.replace('-', '/'); sh = s.sort_values('date').groupby('m')['NumberOfSharesIssued'].last()
    sh = sh.reindex(months).ffill(limit=3)                                          # month-end shares issued
    MC[CI[c]] = PX[CI[c]] * sh.to_numpy(float) / 1e6
MC[~(MC > 0)] = np.nan

# ---- eligibility ------------------------------------------------------------------------------------------------
nhist = np.cumsum(np.isfinite(R), axis=1)
# stocks under an altered trading method (full-cash delivery; includes distressed / reorganising firms) at month-end,
# from the official TWSE list (TWT85U), available from 2004/12
ALT = np.zeros((N, T), bool)
alt = pd.read_csv(os.path.join(PUB, 'twse_altered_trading.csv'), dtype=str).dropna(subset=['code'])
alt = alt[alt.code.isin(CI) & alt.month.isin(MI)]; ALT[alt.code.map(CI).to_numpy(), alt.month.map(MI).to_numpy()] = True
ELIG = listed & quoted & (VOL > 0) & np.isfinite(MC) & (nhist >= 12) & ~ALT
print('eligible stock-months removed by the full-cash-delivery filter:', int((listed & quoted & (VOL > 0) & np.isfinite(MC) & (nhist >= 12) & ALT).sum()))
RN = np.full((N, T), np.nan); RN[:, :-1] = np.where(quoted[:, 1:], R[:, 1:], np.nan)   # next-month return (missing if not traded)

def cumret(a, b):
    if a < 0: return np.full(N, np.nan)
    seg = R[:, a:b + 1]; good = np.isfinite(seg).all(axis=1); out = np.full(N, np.nan)
    out[good] = np.prod(1.0 + seg[good], axis=1) - 1.0
    return out

frames = []
for t in range(T):
    e = ELIG[:, t]
    mom1m = R[:, t]; mom6m = cumret(t - 4, t); mom12m = cumret(t - 10, t)
    mom36m = cumret(t - 35, t - 12); chmom = cumret(t - 5, t) - cumret(t - 11, t - 6)
    d = pd.DataFrame({'ind': industry, 'm12': mom12m, 'cap': MC[:, t]}); v = e & d['m12'].notna().to_numpy()
    indmom = np.full(N, np.nan)
    if v.any():
        g = d[v].groupby('ind').apply(lambda x: np.average(x['m12'], weights=x['cap'])); indmom = d['ind'].map(g).to_numpy(float)
    sig = np.column_stack([mom1m, mom6m, mom12m, mom36m, chmom, indmom]); sig[~e] = np.nan
    f = pd.DataFrame(sig, columns=SIGNALS)
    f.insert(0, 'industry', industry); f.insert(0, 'name', name); f.insert(0, 'code', codes)
    f.insert(0, 'target_month', months[t + 1] if t + 1 < T else None); f.insert(0, 't', t); f.insert(0, 'formation_month', months[t])
    f['mktcap'] = np.where(e, MC[:, t], np.nan); f['ret_next'] = RN[:, t]; f['eligible'] = e; f['status'] = status
    f['has_price'] = quoted[:, t]; f['has_price_next'] = quoted[:, t + 1] if t + 1 < T else False
    frames.append(f[e | quoted[:, t]])
panel = pd.concat(frames, ignore_index=True); panel.to_parquet(os.path.join(DATA, 'panel.parquet'))

ndays = pd.read_csv(os.path.join(DATA, 'market_calendar.csv'))['date'].str[:7].str.replace('-', '/').value_counts()
adtv = pd.DataFrame(VAL.T, index=months, columns=codes).div(ndays.reindex(months), axis=0)
adtv.to_parquet(os.path.join(DATA, 'adtv_monthly.parquet'))

def winsor_z(x):
    lo, hi = np.nanpercentile(x, [1, 99]); xw = np.clip(x, lo, hi); sd = xw.std()
    return (xw - xw.mean()) / sd if sd > 0 else xw * 0.0

rows = []
for t, g in panel[panel.eligible].groupby('t'):
    r = g['ret_next'].to_numpy()
    for s in SIGNALS:
        v = g[s].to_numpy(); ok = np.isfinite(v) & np.isfinite(r); rc, n = np.nan, int(ok.sum())
        if n > 20 and t + 1 < T:
            z = winsor_z(v[ok])
            if z.std() > 0 and r[ok].std() > 0: rc = np.polyfit(z, r[ok], 1)[0]
        rows.append((months[t], months[t + 1] if t + 1 < T else None, t, s, rc, n))
rc = pd.DataFrame(rows, columns=['formation_month', 'target_month', 't', 'signal', 'realized_rc', 'n_stocks'])
rc.to_csv(os.path.join(DATA, 'realized_rc.csv'), index=False)
w = rc.pivot(index='t', columns='signal', values='realized_rc')[SIGNALS]
print(w.describe().T[['count', 'mean', 'std', 'min', 'max']].round(4)); print(w.corr().round(3))
e = panel[panel.eligible]; full = e[SIGNALS].notna().all(axis=1)
print(pd.DataFrame({'eligible': e.groupby('formation_month').size(), 'six_signals': e[full].groupby('formation_month').size(),
                    'delisted_later': e[e.status == 'delisted'].groupby('formation_month').size()}).reindex(['2000/12', '2002/12', '2004/01', '2006/12', '2011/12', '2015/09', '2019/11', '2024/11']))
print('first RC month:', rc.dropna().groupby('signal').formation_month.min().to_dict())
