# -*- coding: utf-8 -*-
"""Step 4. Composite scores, long-only top-k% portfolios, turnover and transaction costs.

For every forecasting method (three RC benchmarks, OLS, four tree ensembles, NN1-NN5) and the
perfect-foresight oracle, at the end of each formation month t:
    Score_i,t = sum_j  beta_hat_{j,t+1|t} * Z_{i,j,t}                      (Eq. 12)
    top 1/3/5/10% of eligible stocks, equal- or value-weighted, held over month t+1
    Turnover_t = 0.5 * sum_i | w_{i,t} - w^-_{i,t} |                       (Eq. 13)
    r_net = r_gross - cost * Turnover_t                                    (Eq. 14)
w^- is the drifted weight of the previous portfolio immediately before rebalancing.
Universes: 'base' (all eligible stocks), 'ex_small' (drop bottom 20% market cap at t),
           'ex_illiquid' (drop bottom 20% average daily trading value during month t).
Output: out/portfolio_returns_all.parquet  (model x universe x weighting x cutoff x month)
"""
import numpy as np, pandas as pd, os, sys, warnings
warnings.filterwarnings('ignore')
from fc_common import *

panel = pd.read_parquet(os.path.join(DATA, 'panel.parquet'))
codes = sorted(panel['code'].unique()); N = len(codes)
T = len(MONTHS)
arr = lambda c: panel.pivot(index='t', columns='code', values=c)[codes].to_numpy(float)
SIG = np.stack([arr(s) for s in SIGNALS], axis=2)                 # (T, N, 6)
CAP, RET = arr('mktcap'), arr('ret_next')
ELIG = np.isfinite(SIG).all(axis=2) & np.isfinite(CAP) & (CAP > 0)
RET0 = np.where(np.isfinite(RET), RET, 0.0)                       # a held stock without a next-month price earns 0
liq_path = os.path.join(DATA, 'adtv_monthly.parquet')
LIQ = None
if os.path.exists(liq_path):
    LIQ = pd.read_parquet(liq_path).reindex(index=MONTHS, columns=codes).to_numpy(float)   # NTD, avg daily value in month t


def winsor_z(x):
    lo, hi = np.percentile(x, [1, 99]); xw = np.clip(x, lo, hi); sd = xw.std()
    return (xw - xw.mean()) / sd if sd > 0 else xw * 0.0


def universe(s, kind):
    e = ELIG[s].copy()
    if kind == 'ex_small':
        e &= CAP[s] > np.quantile(CAP[s][ELIG[s]], 0.20)
    elif kind == 'ex_illiquid':
        v = np.where(np.isfinite(LIQ[s]), LIQ[s], 0.0)
        e &= v > np.quantile(v[ELIG[s]], 0.20)
    return np.where(e)[0]


def zscores(s, idx):
    return np.column_stack([winsor_z(SIG[s, idx, j]) for j in range(6)])


def load_forecasts():
    f = pd.concat([pd.read_parquet(os.path.join(OUT, 'fc_linear_trees.parquet')),
                   pd.read_parquet(os.path.join(OUT, 'fc_nn.parquet'))])
    B = {}
    for m, d in f.groupby('model'):
        b = np.full((T, 6), np.nan); b[d.s.values.astype(int), d.j.values.astype(int)] = d.pred.values; B[m] = b
    Y = load_rc(); B['Oracle'] = Y.copy()
    return B, Y


def target_weights(idx, score, cap, k, weighting):
    n = len(idx); nsel = max(1, int(round(n * k / 100.0)))
    order = np.lexsort((idx, -score))[:nsel]                      # descending score, ties by stock position
    w = np.zeros(N)
    if weighting == 'EW': w[idx[order]] = 1.0 / nsel
    else: c = cap[idx[order]]; w[idx[order]] = c / c.sum()
    return w


def run(kind, models, B):
    S = np.concatenate([VAL_S, OOS_S]); rows = []
    prev = {}
    for s in S:
        idx = universe(s, kind); Z = zscores(s, idx)
        for m in models:
            b = B[m][s]
            if not np.isfinite(b).all(): continue
            score = Z @ b
            for k in CUTOFFS:
                for wt in ('EW', 'VW'):
                    w = target_weights(idx, score, CAP[s], k, wt)
                    key = (m, k, wt); wpre = prev.get(key, np.zeros(N))
                    to = 0.5 * np.abs(w - wpre).sum()
                    g = float(w @ RET0[s])
                    grown = w * (1.0 + RET0[s]); prev[key] = grown / grown.sum()
                    rows.append((kind, m, wt, k, MONTHS[s], MONTHS[s + 1], g, to, int((w > 0).sum()), len(idx),
                                 int(((w > 0) & ~np.isfinite(RET[s])).sum())))
    return pd.DataFrame(rows, columns=['universe', 'model', 'weighting', 'cutoff', 'formation_month', 'return_month',
                                       'gross_return', 'turnover', 'n_holdings', 'n_eligible', 'n_missing_next_return'])


if __name__ == '__main__':
    B, Y = load_forecasts(); models = ALL_MODELS + ['Oracle']
    kinds = ['base', 'ex_small'] + (['ex_illiquid'] if LIQ is not None else [])
    df = pd.concat([run(k, models, B) for k in kinds], ignore_index=True)
    df['transaction_cost'] = COST * df['turnover']; df['net_return'] = df['gross_return'] - df['transaction_cost']
    mk = pd.read_csv(os.path.join(DATA, 'market_monthly.csv')).rename(columns={'month': 'return_month'})
    df = df.merge(mk[['return_month', 'TAIEX_return', 'risk_free_return']], on='return_month', how='left')
    df['sample_period'] = np.where(df.return_month >= OOS_START, 'OOS', 'validation')
    df.to_parquet(os.path.join(OUT, 'portfolio_returns_all.parquet'))
    print(df.groupby('universe').size(), df.n_missing_next_return.sum(), df.n_eligible.describe())
