# -*- coding: utf-8 -*-
"""Official TWSE corporate-action reference prices:
  TWT49U  ex-right / ex-dividend calculation results (available from 2003-05-05)
  TWTAUU  capital-reduction resumption reference prices (available from 2011-01-01)
  TWTB8U  par-value-change resumption reference prices
Saved raw to data/pub/twse_exright.csv, twse_reduction.csv, twse_parchange.csv."""
import requests, pandas as pd, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); PUB = os.path.join(os.path.dirname(HERE), 'data', 'pub')
H = {'User-Agent': 'Mozilla/5.0'}
def pull(path, first, name):
    rows = []
    for y in range(int(first[:4]), 2026):
        s = max(first, f'{y}0101'); e = f'{y}1231'
        for attempt in range(6):
            try:
                j = requests.get(f'https://www.twse.com.tw/rwd/zh/{path}?startDate={s}&endDate={e}&response=json', headers=H, timeout=60).json()
                break
            except Exception as ex: print('retry', y, repr(ex)[:80]); time.sleep(20)
        if j.get('data'):
            rows += [dict(zip(j['fields'], r)) for r in j['data']]
        print(name, y, j.get('stat'), len(j.get('data') or []), flush=True); time.sleep(4)
    d = pd.DataFrame(rows); d.to_csv(os.path.join(PUB, name), index=False); print(name, d.shape, list(d.columns))
pull('exRight/TWT49U', '20030505', 'twse_exright.csv')
pull('reducation/TWTAUU', '20110101', 'twse_reduction.csv')
pull('change/TWTB8U', '20110101', 'twse_parchange.csv')
