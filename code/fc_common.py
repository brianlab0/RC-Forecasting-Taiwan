# -*- coding: utf-8 -*-
"""Shared forecasting utilities: realized-RC matrix, information set, evaluation windows."""
import numpy as np, pandas as pd, os
from config import *

MONTHS = month_list()
MIDX = {m: i for i, m in enumerate(MONTHS)}
# formation index s forecasts the RC of target month s+1
VAL_S = np.arange(MIDX[VAL_START] - 1, MIDX[VAL_END])       # formation 2006/12 .. 2011/11
OOS_S = np.arange(MIDX[OOS_START] - 1, MIDX[OOS_END])       # formation 2011/12 .. 2024/11


def load_rc():
    rc = pd.read_csv(os.path.join(DATA, 'realized_rc.csv'))
    Y = rc.pivot(index='t', columns='signal', values='realized_rc')[SIGNALS].reindex(range(len(MONTHS))).to_numpy(float)
    return Y                                                 # Y[s, j] = realized RC, formation s, target s+1


def build_features(y):
    """Row s contains only realized RCs with formation index <= s-1 (target month <= s), i.e.
    information available at the end of formation month s, plus the calendar month of the target."""
    n = len(y); cols = []; names = []
    for lag in AR_LAGS:
        c = np.full(n, np.nan); c[lag:] = y[:-lag]; cols.append(c); names.append(f'lag{lag}')
    sh = pd.Series(np.concatenate([[np.nan], y[:-1]]))
    for w in ROLL_WINDOWS:
        cols.append(sh.rolling(w, min_periods=w).mean().to_numpy()); names.append(f'rollmean{w}')
        cols.append(sh.rolling(w, min_periods=w).std().to_numpy()); names.append(f'rollsd{w}')
    cal = np.array([int(MONTHS[min(s + 1, n - 1)].split('/')[1]) for s in range(n)])   # target calendar month
    cal[-1] = cal[-2] % 12 + 1
    cols += [np.sin(2 * np.pi * cal / 12), np.cos(2 * np.pi * cal / 12)]; names += ['cal_sin', 'cal_cos']
    return np.column_stack(cols), names


def train_rows(y, X, s):
    """Training rows available at origin s: all u < s with observed target and complete features."""
    u = np.arange(s)
    return u[np.isfinite(y[:s]) & np.isfinite(X[:s]).all(axis=1)]


def benchmark_forecasts(y, s):
    h = y[:s]; h = h[np.isfinite(h)]
    hm = h.mean(); last = y[s - 1]
    a, b = h[:-1], h[1:]                                     # AR(1): y_u = a + phi * y_{u-1}, estimated recursively
    phi = np.cov(a, b, ddof=0)[0, 1] / a.var(); c = b.mean() - phi * a.mean()
    return hm, last, c + phi * last


def r2_oos(y, p, hm):
    return 1.0 - np.sum((y - p) ** 2) / np.sum((y - hm) ** 2)
