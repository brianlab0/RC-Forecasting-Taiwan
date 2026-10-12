# -*- coding: utf-8 -*-
"""Step 2. Walk-forward RC forecasts: benchmarks, OLS and the four tree ensembles.

Stage A (validation, target months 2007/01-2011/12): every candidate configuration is
          re-estimated at every forecast origin with an expanding window.
Stage B : the configuration with the highest mean validation R2 (vs. the expanding historical
          mean, averaged over the six signals) is fixed, then used for 2012/01-2024/12.
No out-of-sample observation enters the configuration choice.
"""
import numpy as np, pandas as pd, os, sys, time, itertools, warnings
warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from fc_common import *

GRID = {
    'OLS': [{}],
    'Random Forest': [dict(max_depth=d, min_samples_leaf=l, max_features=f)
                      for d in (2, 3, 4) for l in (20, 10, 5) for f in ('sqrt', 1.0)],
    'XGBoost': [dict(n_estimators=n, learning_rate=lr, max_depth=d)
                for d in (1, 2) for lr in (0.01, 0.05) for n in (50, 100, 200)],
    'LightGBM': [dict(n_estimators=n, learning_rate=lr, max_depth=d)
                 for d in (1, 2) for lr in (0.01, 0.05) for n in (50, 100, 200)],
    'GBRT': [dict(n_estimators=n, learning_rate=lr, max_depth=d)
             for d in (1, 2) for lr in (0.01, 0.05) for n in (50, 100, 200)],
}


def make(name, p):
    if name == 'OLS': return LinearRegression()
    if name == 'Random Forest':
        return RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=1, **p)
    if name == 'XGBoost':
        return XGBRegressor(min_child_weight=5, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
                            random_state=SEED, n_jobs=1, verbosity=0, **p)
    if name == 'LightGBM':
        return LGBMRegressor(num_leaves=2 ** p['max_depth'], min_child_samples=10, reg_lambda=1.0,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=SEED, n_jobs=1, verbosity=-1, **p)
    if name == 'GBRT':
        return GradientBoostingRegressor(min_samples_leaf=10, subsample=0.8, random_state=SEED, **p)


def run(name, ci, p, j, S):
    y = Y[:, j]; X, _ = build_features(y); out = []
    for s in S:
        tr = train_rows(y, X, s)
        if len(tr) < MIN_TRAIN or not np.isfinite(X[s]).all(): out.append((name, ci, j, s, np.nan)); continue
        m = make(name, p); m.fit(X[tr], y[tr]); out.append((name, ci, j, s, float(m.predict(X[s:s + 1])[0])))
    return out


if __name__ == '__main__':
    Y = load_rc(); t0 = time.time()
    # ---- benchmarks (validation + OOS) ----
    rows = []
    for j, sig in enumerate(SIGNALS):
        for s in np.concatenate([VAL_S, OOS_S]):
            for nm, v in zip(BENCH, benchmark_forecasts(Y[:, j], s)): rows.append((nm, 0, j, s, v))
    bench = pd.DataFrame(rows, columns=['model', 'config', 'j', 's', 'pred'])
    # ---- stage A: validation for every configuration ----
    tasks = [(n, ci, p, j, VAL_S) for n in GRID for ci, p in enumerate(GRID[n]) for j in range(6)]
    res = Parallel(n_jobs=9, verbose=1)(delayed(run)(*t) for t in tasks)
    val = pd.DataFrame([r for rr in res for r in rr], columns=['model', 'config', 'j', 's', 'pred'])
    val['y'] = Y[val.s.values, val.j.values]
    hm = bench[bench.model == 'Historical mean'].set_index(['j', 's'])['pred']
    val['hm'] = hm.reindex(pd.MultiIndex.from_arrays([val.j, val.s])).values
    g = val.groupby(['model', 'config', 'j']).apply(lambda d: r2_oos(d.y.values, d.pred.values, d.hm.values)).rename('val_r2').reset_index()
    sel = g.groupby(['model', 'config'])['val_r2'].mean().rename('mean_val_r2').reset_index()
    sel['params'] = [str(GRID[m][c]) for m, c in zip(sel.model, sel.config)]
    sel['selected'] = sel.groupby('model')['mean_val_r2'].transform(lambda x: x == x.max())
    sel = sel.sort_values(['model', 'config']); sel.to_csv(os.path.join(OUT, 'audit_validation_grid_linear_trees.csv'), index=False)
    g.to_csv(os.path.join(OUT, 'audit_validation_grid_linear_trees_by_signal.csv'), index=False)
    best = {m: int(d.loc[d.mean_val_r2.idxmax(), 'config']) for m, d in sel.groupby('model')}
    print(sel[sel.selected].to_string(), f'\nstage A {time.time() - t0:.0f}s', flush=True)
    # ---- stage B: OOS with the fixed validation-selected configuration ----
    tasks = [(n, best[n], GRID[n][best[n]], j, OOS_S) for n in GRID for j in range(6)]
    res = Parallel(n_jobs=9, verbose=1)(delayed(run)(*t) for t in tasks)
    oos = pd.DataFrame([r for rr in res for r in rr], columns=['model', 'config', 'j', 's', 'pred'])
    keep = pd.concat([val[[b == best[m] for m, b in zip(val.model, val.config)]][['model', 'config', 'j', 's', 'pred']], oos, bench])
    keep.to_parquet(os.path.join(OUT, 'fc_linear_trees.parquet'))
    print(f'done {time.time() - t0:.0f}s', flush=True)
