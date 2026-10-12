# -*- coding: utf-8 -*-
"""Step 3. Walk-forward RC forecasts with five feedforward neural networks (NN1-NN5).

Architecture (fixed ex ante, pyramid rule of Gu, Kelly and Xiu 2020):
    NN1 (32) | NN2 (32,16) | NN3 (32,16,8) | NN4 (32,16,8,4) | NN5 (32,16,8,4,2)
    hidden layer = Linear -> BatchNorm -> ReLU, linear output unit.
Training controls: inputs and target standardised with training-window moments; Adam, full batch;
    MSE loss + L1 penalty on weights; early stopping (patience 10, max 200 epochs) on the most
    recent 20% of the training window; forecast = average over 5 independently seeded networks.
Validation choice: L1 penalty in {1e-5,1e-4,1e-3} x learning rate in {0.001,0.01}; the pair with
    the highest mean validation R2 (2007/01-2011/12, six signals) is fixed before 2012/01.
"""
import numpy as np, pandas as pd, os, sys, time, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn
from joblib import Parallel, delayed
from fc_common import *
torch.set_num_threads(1)

ARCH = {'NN1': (32,), 'NN2': (32, 16), 'NN3': (32, 16, 8), 'NN4': (32, 16, 8, 4), 'NN5': (32, 16, 8, 4, 2)}
GRID = [dict(l1=l1, lr=lr) for l1 in (1e-3, 1e-4, 1e-5) for lr in (1e-3, 1e-2)]
N_ENS, MAX_EPOCHS, PATIENCE, ES_FRAC = 5, 200, 10, 0.20


class Net(nn.Module):
    def __init__(self, k, widths):
        super().__init__(); layers = []; prev = k
        for w in widths: layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.ReLU()]; prev = w
        self.hidden = nn.Sequential(*layers); self.out = nn.Linear(prev, 1)

    def forward(self, x): return self.out(self.hidden(x)).squeeze(-1)


def fit_predict(Xtr, ytr, Xte, widths, l1, lr, seed):
    torch.manual_seed(seed)
    n = len(Xtr); nv = max(6, int(round(n * ES_FRAC))); a = np.arange(n - nv); b = np.arange(n - nv, n)
    mx, sx = Xtr[a].mean(0), Xtr[a].std(0) + 1e-8; my, sy = ytr[a].mean(), ytr[a].std() + 1e-12
    f = lambda Z: torch.tensor((Z - mx) / sx, dtype=torch.float32)
    Xa, Xb, Xt = f(Xtr[a]), f(Xtr[b]), f(Xte)
    ya = torch.tensor((ytr[a] - my) / sy, dtype=torch.float32); yb = torch.tensor((ytr[b] - my) / sy, dtype=torch.float32)
    net = Net(Xtr.shape[1], widths); opt = torch.optim.Adam(net.parameters(), lr=lr); mse = nn.MSELoss()
    best, state, bad = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        net.train(); opt.zero_grad()
        pen = sum(p.abs().sum() for nme, p in net.named_parameters() if nme.endswith('weight') and p.dim() > 1)
        (mse(net(Xa), ya) + l1 * pen).backward(); opt.step()
        net.eval()
        with torch.no_grad(): v = mse(net(Xb), yb).item()
        if v < best - 1e-9: best, state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    net.load_state_dict(state); net.eval()
    with torch.no_grad(): return float(net(Xt).item()) * sy + my


def run(name, ci, p, j, S):
    y = Y[:, j]; X, _ = build_features(y); out = []; mi = list(ARCH).index(name)
    for s in S:
        tr = train_rows(y, X, s)
        if len(tr) < MIN_TRAIN or not np.isfinite(X[s]).all(): out.append((name, ci, j, int(s), np.nan)); continue
        ps = [fit_predict(X[tr], y[tr], X[s:s + 1], ARCH[name], p['l1'], p['lr'],
                          SEED + 1000003 * mi + 10007 * j + 101 * int(s) + e) for e in range(N_ENS)]
        out.append((name, ci, j, int(s), float(np.mean(ps))))
    return out


if __name__ == '__main__':
    Y = load_rc(); t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == 'time':
        r = run('NN5', 0, GRID[0], 0, VAL_S[:3]); print(r, time.time() - t0); sys.exit()
    NJ = int(os.environ.get('NJOBS', 9))
    chunks = np.array_split(VAL_S, 4)
    tasks = [(n, ci, p, j, S) for n in ARCH for ci, p in enumerate(GRID) for j in range(6) for S in chunks]
    res = Parallel(n_jobs=NJ, verbose=1)(delayed(run)(*t) for t in tasks)
    val = pd.DataFrame([r for rr in res for r in rr], columns=['model', 'config', 'j', 's', 'pred'])
    val.to_parquet(os.path.join(OUT, 'fc_nn_validation_all_configs.parquet'))
    val['y'] = Y[val.s.values, val.j.values]
    val['hm'] = [benchmark_forecasts(Y[:, j], s)[0] for j, s in zip(val.j, val.s)]
    g = val.groupby(['model', 'config', 'j']).apply(lambda d: r2_oos(d.y.values, d.pred.values, d.hm.values)).rename('val_r2').reset_index()
    sel = g.groupby(['model', 'config'])['val_r2'].mean().rename('mean_val_r2').reset_index()
    sel['params'] = [str(GRID[c]) for c in sel.config]
    sel['selected'] = sel.groupby('model')['mean_val_r2'].transform(lambda x: x == x.max())
    sel.to_csv(os.path.join(OUT, 'audit_validation_grid_nn.csv'), index=False)
    g.to_csv(os.path.join(OUT, 'audit_validation_grid_nn_by_signal.csv'), index=False)
    best = {m: int(d.loc[d.mean_val_r2.idxmax(), 'config']) for m, d in sel.groupby('model')}
    print(sel[sel.selected].to_string(), f'\nstage A {time.time() - t0:.0f}s', flush=True)
    chunks = np.array_split(OOS_S, 8)
    tasks = [(n, best[n], GRID[best[n]], j, S) for n in ARCH for j in range(6) for S in chunks]
    res = Parallel(n_jobs=NJ, verbose=1)(delayed(run)(*t) for t in tasks)
    oos = pd.DataFrame([r for rr in res for r in rr], columns=['model', 'config', 'j', 's', 'pred'])
    keep = pd.concat([val[[c == best[m] for m, c in zip(val.model, val.config)]][['model', 'config', 'j', 's', 'pred']], oos])
    keep.to_parquet(os.path.join(OUT, 'fc_nn.parquet'))
    print(f'done {time.time() - t0:.0f}s', flush=True)
