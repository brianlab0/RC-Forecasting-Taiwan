# -*- coding: utf-8 -*-
"""TWSE list of securities under an altered trading method (變更交易, i.e. full-cash-delivery stocks, which include
firms in financial distress or reorganisation) on the last trading day of every month. Official TWSE report TWT85U,
available from 2004-12-24. Output: data/pub/twse_altered_trading.csv (month, date, code, name)."""
import requests, pandas as pd, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(HERE), 'data'); OUTF = os.path.join(DATA, 'pub', 'twse_altered_trading.csv')
cal = pd.read_csv(os.path.join(DATA, 'market_calendar.csv'))['date']; last = cal.groupby(cal.str[:7]).max()
last = last[(last >= '2004-12-24') & (last <= '2024-12-31')]
done = pd.read_csv(OUTF, dtype=str) if os.path.exists(OUTF) else pd.DataFrame(columns=['month', 'date', 'code', 'name'])
rows = done.values.tolist(); have = set(done['date'])
for d in last:
    if d in have: continue
    for attempt in range(8):
        try:
            j = requests.get(f"https://www.twse.com.tw/exchangeReport/TWT85U?response=json&date={d.replace('-', '')}", headers={'User-Agent': 'Mozilla/5.0'}, timeout=60).json()
            if j.get('stat') == 'OK' or '沒有符合' in str(j.get('stat')): break
            print(d, 'stat', j.get('stat'), flush=True); time.sleep(20)
        except Exception as e: print(d, 'retry', repr(e)[:80], flush=True); time.sleep(30)
    data = j.get('data') or []
    rows += [[d[:7].replace('-', '/'), d, r[0].strip(), r[1].strip()] for r in data] or [[d[:7].replace('-', '/'), d, '', '']]
    pd.DataFrame(rows, columns=['month', 'date', 'code', 'name']).to_csv(OUTF, index=False); print(d, len(data), flush=True); time.sleep(4)
print('DONE', flush=True)
