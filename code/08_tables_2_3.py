# -*- coding: utf-8 -*-
"""Tables 2 and 3: summary statistics and correlations of signal values and realized RCs."""
import numpy as np, pandas as pd, os, sys
from config import *
PANEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'panel.parquet')
RCF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA, 'realized_rc.csv')
DEL = os.path.join(OUT, 'deliverables'); os.makedirs(DEL, exist_ok=True)
p = pd.read_parquet(PANEL); rc = pd.read_csv(RCF)
if 'eligible' in p: p = p[p.eligible]
p = p[p.formation_month <= '2024/11']                                    # formation months whose target month is in the sample
rc = rc[rc.formation_month <= '2024/11']
def desc(x):
    x = pd.Series(x).dropna()
    return dict(obs=len(x), mean=x.mean(), sd=x.std(), skew=x.skew(), kurt=x.kurt(), min=x.min(), p25=x.quantile(.25), median=x.median(), p75=x.quantile(.75), max=x.max())
wins = lambda x: x.clip(*np.nanpercentile(x, [1, 99])) if x.notna().sum() > 20 else x * np.nan
W = p.groupby('t')[SIGNALS].transform(wins)
A = pd.DataFrame({s: desc(W[s]) for s in SIGNALS}).T; A.insert(0, 'panel', 'A: signal values (winsorized 1/99, before Z-score)')
Y = rc.pivot(index='t', columns='signal', values='realized_rc')[SIGNALS]
B = pd.DataFrame({s: desc(Y[s]) for s in SIGNALS}).T; B.insert(0, 'panel', 'B: realized monthly regression coefficients')
t2 = pd.concat([A, B]).rename_axis('signal').reset_index(); t2.to_csv(os.path.join(DEL, 'table2_summary_statistics.csv'), index=False)
Z = W.groupby(p['t']).transform(lambda x: (x - x.mean()) / x.std())
cs = [g.corr() for _, g in Z.groupby(p['t']) if g.notna().all(axis=1).sum() > 20]
CA = sum(c.fillna(0) for c in cs) / sum(c.notna().astype(int) for c in cs); CB = Y.corr()
t3 = pd.concat([CA.assign(panel='A: time-series average of monthly cross-sectional signal correlations'), CB.assign(panel='B: time-series correlations of realized RCs')])
t3.rename_axis('signal').reset_index().to_csv(os.path.join(DEL, 'table3_correlations.csv'), index=False)
pd.set_option('display.width', 250); print(t2.round(3).to_string()); print(CA.round(3)); print(CB.round(3))
