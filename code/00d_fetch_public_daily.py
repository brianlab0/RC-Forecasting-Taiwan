# -*- coding: utf-8 -*-
"""Download official daily quotes (FinMind TaiwanStockPrice = TWSE/TPEx daily quotes) for every TWSE-listed
common stock, including delisted ones, and daily shares outstanding (TaiwanStockShareholding) for stocks whose
market capitalisation is not in data/raw_size.xlsx. Resumable; optional worker split: python 00d... PART NPART."""
import requests, pandas as pd, os, time, sys
PART, NPART = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); PUB = os.path.join(ROOT, 'data', 'pub')
for d in ('px', 'sh'): os.makedirs(os.path.join(PUB, d), exist_ok=True)
L = pd.read_csv(os.path.join(PUB, 'twse_listed_basic.csv'), dtype=str)
D = pd.read_csv(os.path.join(PUB, 'twse_delisted.csv'), dtype=str)
ok = lambda s: s.str.fullmatch(r'[1-9]\d{3}') & ~s.str.startswith('91')            # common shares only (no ETF / TDR / preferred)
cur = sorted(L[ok(L['公司代號']) & (L['上市日期'] <= '20250228')]['公司代號'].unique())
D = D[ok(D['code']) & (D['ymd'] <= '2025-02-28')].sort_values('ymd', ascending=False)
dele = [c for c in D['code'].drop_duplicates() if c not in cur]
have_cap = set(pd.read_excel(os.path.join(ROOT, 'data', 'raw_size.xlsx'), sheet_name='市值', header=0, usecols=[0]).iloc[:, 0].astype(str).str.strip())
jobs = [('px', c) for c in cur + dele] + [('sh', c) for c in cur + dele if c not in have_cap]
URL = 'https://api.finmindtrade.com/api/v4/data'
DS = {'px': ('TaiwanStockPrice', '1998-12-01'), 'sh': ('TaiwanStockShareholding', '2004-02-01')}
print(len(cur), 'current', len(dele), 'delisted', len(jobs), 'jobs', flush=True)
for i, (k, c) in enumerate(jobs):
    if i % NPART != PART: continue
    fn = os.path.join(PUB, k, f'{c}.parquet')
    if os.path.exists(fn): continue
    for attempt in range(40):
        try:
            r = requests.get(URL, params=dict(dataset=DS[k][0], data_id=c, start_date=DS[k][1], end_date='2025-02-28'), timeout=180)
            j = r.json()
            if r.status_code == 200 and j.get('msg') == 'success':
                d = pd.DataFrame(j['data'])
                if k == 'sh' and len(d): d = d[['date', 'stock_id', 'NumberOfSharesIssued']]
                if len(d) == 0: d = pd.DataFrame({'date': pd.Series(dtype=str), 'stock_id': pd.Series(dtype=str)})
                d.to_parquet(fn); print(i, k, c, len(d), flush=True); break
            print(i, k, c, 'retry', r.status_code, str(j)[:100], flush=True); time.sleep(300 if r.status_code == 402 else 60)
        except Exception as e:
            print(i, k, c, 'err', repr(e)[:120], flush=True); time.sleep(60)
    time.sleep(1.0)
print('DONE', flush=True)
