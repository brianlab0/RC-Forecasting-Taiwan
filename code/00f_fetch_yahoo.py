# -*- coding: utf-8 -*-
"""Yahoo Finance daily adjusted close for currently listed TWSE stocks.
External check only (09_validate_against_yahoo.py); not used in the construction of returns."""
import yfinance as yf, pandas as pd, os, time, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); PUB = os.path.join(os.path.dirname(HERE), 'data', 'pub')
OUT = os.path.join(PUB, 'yahoo'); os.makedirs(OUT, exist_ok=True)
L = pd.read_csv(os.path.join(PUB, 'twse_listed_basic.csv'), dtype=str)
codes = sorted(c for c in L['公司代號'].unique() if c.isdigit() and len(c) == 4 and not c.startswith('91'))
for i, c in enumerate(codes):
    fn = os.path.join(OUT, f'{c}.parquet')
    if os.path.exists(fn): continue
    for attempt in range(4):
        try:
            h = yf.Ticker(c + '.TW').history(start='1999-12-01', end='2025-03-05', auto_adjust=False, actions=True)
            if len(h) == 0 and attempt < 2: time.sleep(5); continue
            h = h.reset_index(); h['date'] = h['Date'].dt.strftime('%Y-%m-%d') if len(h) else []
            keep = [k for k in ['date', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits'] if k in h.columns]
            h[keep].to_parquet(fn); print(i, c, len(h), flush=True); break
        except Exception as e:
            print(i, c, 'err', repr(e)[:100], flush=True); time.sleep(20)
    time.sleep(0.4)
print('DONE', flush=True)
