# -*- coding: utf-8 -*-
"""Step 6. (9-4) monthly_stock_scores_and_holdings: one row per formation month x model x eligible stock
(baseline universe). EW/VW target and pre-rebalancing weights refer to each model's
validation-selected cutoff for that weighting scheme."""
import numpy as np, pandas as pd, os, json, gzip, importlib
from fc_common import *
P = importlib.import_module('04_portfolios')
B, Y = P.load_forecasts(); N = P.N; codes = np.array(P.codes)
SEL = json.load(open(os.path.join(OUT, 'selected_cutoffs.json')))
name = pd.read_parquet(os.path.join(DATA, 'panel.parquet'), columns=['code', 'name']).drop_duplicates('code').set_index('code')['name']
out = os.path.join(OUT, 'deliverables', 'monthly_stock_scores_and_holdings.csv.gz')
S = np.concatenate([VAL_S, OOS_S]); prev = {}; first = True; nrows = 0
with gzip.open(out, 'wt', encoding='utf-8', newline='') as fh:
    for s in S:
        idx = P.universe(s, 'base'); Z = P.zscores(s, idx); n = len(idx)
        liq = P.LIQ[s, idx] if P.LIQ is not None else np.full(n, np.nan)
        for m in ALL_MODELS:
            b = B[m][s]; contrib = Z * b; score = contrib.sum(1)
            order = np.lexsort((idx, -score)); rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
            d = pd.DataFrame({'stock_id': codes[idx], 'stock_name': name.reindex(codes[idx]).values, 'formation_month': MONTHS[s], 'return_month': MONTHS[s + 1],
                              'sample_period': 'OOS' if MONTHS[s + 1] >= OOS_START else 'validation', 'model': m})
            for j, sg in enumerate(SIGNALS): d[f'z_{sg}'] = Z[:, j]
            for j, sg in enumerate(SIGNALS): d[f'predicted_rc_{sg}'] = b[j]
            for j, sg in enumerate(SIGNALS): d[f'contribution_{sg}'] = contrib[:, j]
            d['composite_score'] = score; d['score_rank'] = rank; d['score_percentile'] = 100.0 * (1 - (rank - 1) / n)
            d['market_cap_mn_ntd'] = P.CAP[s, idx]; d['avg_daily_trading_value_ntd'] = liq
            for k in CUTOFFS: d[f'top{k}_flag'] = (rank <= max(1, int(round(n * k / 100.0)))).astype(int)
            for wt in ('EW', 'VW'):
                k = SEL[f'{m}|{wt}']; w = P.target_weights(idx, score, P.CAP[s], k, wt)
                wpre = prev.get((m, wt), np.zeros(N))
                d[f'{wt}_selected_cutoff_pct'] = k; d[f'{wt}_target_weight'] = w[idx]; d[f'{wt}_pre_rebalancing_weight'] = wpre[idx]
                grown = w * (1.0 + P.RET0[s]); prev[(m, wt)] = grown / grown.sum()
            d['next_month_return'] = P.RET[s, idx]
            d.to_csv(fh, index=False, header=first, float_format='%.8g'); first = False; nrows += len(d)
print('rows', nrows, 'file', out, round(os.path.getsize(out) / 1e6, 1), 'MB')
