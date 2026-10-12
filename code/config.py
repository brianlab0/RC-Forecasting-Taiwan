# -*- coding: utf-8 -*-
"""Shared configuration for the RC-forecasting pipeline (single source of truth)."""
import os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, OUT, LOGS = [os.path.join(ROOT, d) for d in ('data', 'out', 'logs')]
SIGNALS = ['mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom', 'indmom']
BENCH = ['Historical mean', 'Last observation', 'AR(1)']
TREES = ['Random Forest', 'XGBoost', 'LightGBM', 'GBRT']
NNS = ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']
ML = ['OLS'] + TREES + NNS
ALL_MODELS = BENCH + ML
CUTOFFS = [1, 3, 5, 10]
COST = 0.00585                      # round-trip cost: 0.1425% x 2 commission + 0.3% sell-side tax
# --- chronology, expressed in TARGET (= return) months ---------------------------------------
VAL_START, VAL_END = '2007/01', '2011/12'     # 60 months: hyperparameter + cutoff validation
OOS_START, OOS_END = '2012/01', '2024/12'     # 156 months: formal out-of-sample period
NW_LAGS = 6
SEED = 42
# --- forecasting information set (identical for OLS, trees and NNs) -----------------------------
AR_LAGS = [1, 2, 3, 6, 12]          # lagged realized RCs
ROLL_WINDOWS = [3, 6, 12]           # rolling mean and rolling SD of lagged realized RCs
MIN_TRAIN = 24                      # minimum number of training rows required to fit a model


def month_list(first='1999/01', last='2025/02'):
    y0, m0 = map(int, first.split('/')); y1, m1 = map(int, last.split('/'))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f'{y}/{m:02d}'); m += 1
        if m == 13: y, m = y + 1, 1
    return out
