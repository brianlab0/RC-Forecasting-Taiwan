# -*- coding: utf-8 -*-
"""Step 5. All tables, figures and audit files, built from the locked forecast panel and the
locked monthly portfolio-return series produced by steps 2-4."""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fc_common import *

DEL = os.path.join(OUT, 'deliverables'); os.makedirs(DEL, exist_ok=True)
Y = load_rc(); T = len(MONTHS)
fc = pd.concat([pd.read_parquet(os.path.join(OUT, 'fc_linear_trees.parquet')), pd.read_parquet(os.path.join(OUT, 'fc_nn.parquet'))])
fc['s'] = fc.s.astype(int); fc['j'] = fc.j.astype(int)
pr = pd.read_parquet(os.path.join(OUT, 'portfolio_returns_all.parquet'))
mk = pd.read_csv(os.path.join(DATA, 'market_monthly.csv'))
pr = pr.merge(mk[['month', 'TAIEX_price_index_return']].rename(columns={'month': 'return_month'}), on='return_month', how='left')
ORDER = {m: i for i, m in enumerate(ALL_MODELS)}

# ------------------------------------------------------------------ helpers
def sharpe(r, rf): x = np.asarray(r) - np.asarray(rf); return np.sqrt(12) * x.mean() / x.std(ddof=1)
def nw_mean(x, lags=NW_LAGS):
    res = sm.OLS(np.asarray(x), np.ones(len(x))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return float(res.params[0]), float(res.tvalues[0])
def nw_alpha(y, x, lags=NW_LAGS):
    res = sm.OLS(np.asarray(y), sm.add_constant(np.asarray(x))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return float(res.params[0]), float(res.tvalues[0]), float(res.params[1])

# ------------------------------------------------------------------ (9-1) forecast panel, Table 4, Table A.2
p = fc.copy()
p['formation_month'] = [MONTHS[s] for s in p.s]; p['target_month'] = [MONTHS[s + 1] for s in p.s]
p['sample_period'] = np.where(p.target_month >= OOS_START, 'OOS', 'validation')
p['signal'] = [SIGNALS[j] for j in p.j]; p['realized_rc'] = Y[p.s.values, p.j.values]
p = p.rename(columns={'pred': 'predicted_rc'})
p['forecast_error'] = p.realized_rc - p.predicted_rc; p['squared_error'] = p.forecast_error ** 2; p['absolute_error'] = p.forecast_error.abs()
p['o'] = p.model.map(ORDER); p = p.sort_values(['o', 'j', 's'])
cols91 = ['formation_month', 'target_month', 'sample_period', 'model', 'signal', 'realized_rc', 'predicted_rc', 'forecast_error', 'squared_error', 'absolute_error']
p[cols91].to_csv(os.path.join(DEL, 'monthly_rc_forecast_panel.csv'), index=False)
oos = p[p.sample_period == 'OOS']
assert oos.groupby(['model', 'signal']).size().eq(156).all() and oos.predicted_rc.notna().all()
sse = oos.pivot_table(index='model', columns='signal', values='squared_error', aggfunc='sum')[SIGNALS]
t4 = (100 * (1 - sse / sse.loc['Historical mean'])).loc[ALL_MODELS]
a2 = (1e3 * oos.pivot_table(index='model', columns='signal', values='absolute_error', aggfunc='mean')[SIGNALS]).loc[ALL_MODELS]
t4.round(3).to_csv(os.path.join(DEL, 'table4_r2oos.csv')); a2.round(3).to_csv(os.path.join(DEL, 'appendix_A2_mae.csv'))
print('TABLE 4 (R2_OOS, %)\n', t4.round(3).to_string()); print('TABLE A.2 (MAE x 1e3)\n', a2.round(3).to_string())

# ------------------------------------------------------------------ (9-2) monthly signal summary
q = p[['formation_month', 'target_month', 'sample_period', 'model', 'signal', 'predicted_rc', 'realized_rc']].copy()
q['abs_predicted_rc_rank'] = q.assign(a=q.predicted_rc.abs()).groupby(['model', 'target_month'])['a'].rank(ascending=False, method='first').astype(int)
q['dominant_signal'] = (q.abs_predicted_rc_rank == 1).astype(int)
q.to_csv(os.path.join(DEL, 'monthly_signal_summary.csv'), index=False)

# ------------------------------------------------------------------ cutoff selection (validation gross Sharpe)
base = pr[(pr.universe == 'base') & (pr.model != 'Oracle')].copy()
val = base[base.sample_period == 'validation']; assert val.groupby(['model', 'weighting', 'cutoff']).size().eq(60).all()
vs = val.groupby(['model', 'weighting', 'cutoff']).apply(lambda d: sharpe(d.gross_return, d.risk_free_return)).rename('val_gross_sharpe').reset_index()
sel = vs.loc[vs.groupby(['model', 'weighting'])['val_gross_sharpe'].idxmax()].set_index(['model', 'weighting'])
SEL = sel['cutoff'].to_dict()
pr['validation_selected_cutoff_flag'] = [int(SEL.get((m, w)) == c) for m, w, c in zip(pr.model, pr.weighting, pr.cutoff)]

# ------------------------------------------------------------------ (9-3) monthly portfolio returns, (9-5) oracle
c93 = ['formation_month', 'return_month', 'sample_period', 'model', 'weighting', 'cutoff', 'validation_selected_cutoff_flag', 'gross_return',
       'turnover', 'transaction_cost', 'net_return', 'TAIEX_return', 'risk_free_return', 'universe', 'n_holdings', 'n_eligible', 'TAIEX_price_index_return']
x = pr[pr.model != 'Oracle'].copy(); x['o'] = x.model.map(ORDER)
x.sort_values(['universe', 'o', 'weighting', 'cutoff', 'return_month'])[c93].to_csv(os.path.join(DEL, 'monthly_portfolio_returns.csv'), index=False)
orc = pr[(pr.model == 'Oracle') & (pr.universe == 'base') & (pr.weighting == 'EW') & (pr.sample_period == 'OOS')].sort_values(['cutoff', 'return_month'])
orc.rename(columns={'return_month': 'month'})[['month', 'cutoff', 'gross_return', 'turnover', 'transaction_cost', 'net_return', 'TAIEX_return', 'risk_free_return', 'formation_month']].to_csv(os.path.join(DEL, 'oracle_monthly_returns.csv'), index=False)
b1 = orc.groupby('cutoff').apply(lambda d: pd.Series({
    'gross_sharpe': sharpe(d.gross_return, d.risk_free_return), 'net_sharpe': sharpe(d.net_return, d.risk_free_return),
    'gross_win_rate_pct': 100 * (d.gross_return > d.TAIEX_return).mean(), 'net_win_rate_pct': 100 * (d.net_return > d.TAIEX_return).mean()}))
b1.round(3).to_csv(os.path.join(DEL, 'appendix_B1_oracle.csv')); print('TABLE B.1\n', b1.round(3))

# ------------------------------------------------------------------ Table 5, C.1, 6, 7, D.1
def stats(d, cost=COST):
    net = d.gross_return - cost * d.turnover; mu, t = nw_mean(net)
    return dict(gross_sharpe=sharpe(d.gross_return, d.risk_free_return), net_sharpe=sharpe(net, d.risk_free_return),
                net_mean_pct=100 * mu, nw_t=t, turnover_pct=100 * d.turnover.mean(), win_pct=100 * (net > d.TAIEX_return).mean(),
                win_px_pct=100 * (net > d.TAIEX_price_index_return).mean())
def cell(universe, m, w, k=None):
    k = SEL[(m, w)] if k is None else k
    d = pr[(pr.universe == universe) & (pr.model == m) & (pr.weighting == w) & (pr.cutoff == k) & (pr.sample_period == 'OOS')].sort_values('return_month')
    assert len(d) == 156; return d

t5, t6, t7, c1, d1 = [], [], [], [], {}
HAS_LIQ = 'ex_illiquid' in set(pr.universe)
for w in ('EW', 'VW'):
    for m in ALL_MODELS:
        d = cell('base', m, w); s = stats(d)
        t5.append(dict(panel=w, model=m, selected_cutoff_pct=SEL[(m, w)], validation_gross_sharpe=sel.loc[(m, w), 'val_gross_sharpe'],
                       oos_gross_sharpe=s['gross_sharpe'], oos_net_sharpe=s['net_sharpe'], oos_net_mean_return_pct=s['net_mean_pct'],
                       nw_t_stat=s['nw_t'], average_turnover_pct=s['turnover_pct'], oos_net_win_rate_pct=s['win_pct'],
                       oos_net_win_rate_vs_price_index_pct=s['win_px_pct']))
        c1.append(dict(panel=w, model=m, selected_cutoff_pct=SEL[(m, w)], **{f'top{k}': stats(cell('base', m, w, k))['net_sharpe'] for k in CUTOFFS}))
        r6 = dict(panel=w, model=m, selected_cutoff_pct=SEL[(m, w)], net_sharpe_58p5bps=s['net_sharpe'], net_sharpe_100bps=stats(d, 0.01)['net_sharpe'],
                  break_even_cost_bps=1e4 * d.gross_return.mean() / d.turnover.mean(),
                  net_sharpe_excl_smallest=stats(cell('ex_small', m, w))['net_sharpe'],
                  net_sharpe_excl_least_liquid=stats(cell('ex_illiquid', m, w))['net_sharpe'] if HAS_LIQ else np.nan,
                  mean_gross_return_pct=100 * d.gross_return.mean(), average_turnover_pct=100 * d.turnover.mean())
        t6.append(r6)
        sub = lambda a: sharpe(d.net_return[a], d.risk_free_return[a])
        rm = d.return_month
        t7.append(dict(panel=w, model=m, selected_cutoff_pct=SEL[(m, w)], full_oos=s['net_sharpe'], first_half_201201_201806=sub(rm <= '2018/06'),
                       second_half_201807_202412=sub(rm >= '2018/07'), excluding_covid_202001_202112=sub((rm < '2020/01') | (rm > '2021/12')),
                       n_full=len(d), n_first=int((rm <= '2018/06').sum()), n_second=int((rm >= '2018/07').sum()), n_ex_covid=int(((rm < '2020/01') | (rm > '2021/12')).sum())))
        a, ta, be = nw_alpha(d.net_return - d.risk_free_return, d.TAIEX_return - d.risk_free_return)
        d1.setdefault(m, {'model': m}).update({f'{w}_selected_cutoff_pct': SEL[(m, w)], f'{w}_net_alpha_pct_per_month': 100 * a, f'{w}_nw_t_stat': ta, f'{w}_market_beta': be})
        a2, ta2, be2 = nw_alpha(d.net_return - d.risk_free_return, d.TAIEX_price_index_return - d.risk_free_return)
        d1[m].update({f'{w}_net_alpha_vs_price_index_pct': 100 * a2, f'{w}_nw_t_stat_vs_price_index': ta2})
t5, t6, t7, c1, d1 = pd.DataFrame(t5), pd.DataFrame(t6), pd.DataFrame(t7), pd.DataFrame(c1), pd.DataFrame(list(d1.values()))
tx = mk[(mk.month >= OOS_START) & (mk.month <= OOS_END)]; mu, tt = nw_mean(tx.TAIEX_return)
taiex = pd.DataFrame([dict(benchmark=nm_, period='2012/01-2024/12', n_months=len(tx), average_monthly_return_pct=100 * nw_mean(tx[c_])[0], nw_t_stat=nw_mean(tx[c_])[1],
                           annualized_sharpe=sharpe(tx[c_], tx.risk_free_return), monthly_sd_pct=100 * tx[c_].std(ddof=1),
                           average_monthly_risk_free_pct=100 * tx.risk_free_return.mean())
                      for nm_, c_ in [('TAIEX Total Return Index (same return basis as the portfolios; used in all tables)', 'TAIEX_return'), ('TAIEX price index (supplementary; excludes dividends)', 'TAIEX_price_index_return')]])
for nm, df in [('table5_portfolio_performance', t5), ('table6_transaction_cost_investability', t6), ('table7_subperiod_stability', t7),
               ('appendix_C1_net_sharpe_all_cutoffs', c1), ('appendix_D1_risk_adjusted_performance', d1), ('table5_TAIEX_reference', taiex)]:
    df.round(4).to_csv(os.path.join(DEL, nm + '.csv'), index=False); print('\n' + nm + '\n', df.round(3).to_string(index=False))
mk[(mk.month >= VAL_START)].rename(columns={'rf_annual_pct_prev_month_end': 'BoT_1M_deposit_rate_annual_pct_prev_month_end'}).to_csv(os.path.join(DEL, 'risk_free_rate_and_TAIEX_monthly.csv'), index=False)

# ------------------------------------------------------------------ Figure 1: realized RC series
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
todate = lambda m: pd.Timestamp(m.replace('/', '-') + '-01') + pd.offsets.MonthEnd(0)
rc = pd.read_csv(os.path.join(DATA, 'realized_rc.csv')).dropna(subset=['realized_rc', 'target_month'])
rc = rc[rc.target_month <= OOS_END]                                  # sample ends December 2024
wide = rc.pivot(index='target_month', columns='signal', values='realized_rc')[SIGNALS]
wide.insert(0, 'formation_month', [MONTHS[MIDX[m] - 1] for m in wide.index]); wide.to_csv(os.path.join(DEL, 'fig_realized_rc_series_data.csv'))
lim = 1.05 * np.nanmax(np.abs(wide[SIGNALS].to_numpy()))
for lab, sgn in zip('abcdef', SIGNALS):
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); s = wide[sgn].dropna()
    ax.plot([todate(m) for m in s.index], s.values, color='#1f3b63', lw=0.9, label='Realized RC')
    ax.axhline(0, color='black', lw=0.7); ax.axvline(pd.Timestamp('2012-01-01'), color='#b22222', lw=1.0, ls='--', label='Start of out-of-sample period (2012/01)')
    ax.set_ylim(-lim, lim); ax.set_xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2025-03-31'))
    ax.set_xlabel('Target month'); ax.set_ylabel('Realized regression coefficient')
    ax.xaxis.set_major_locator(mdates.YearLocator(4)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.text(0.015, 0.95, f'({lab}) {sgn}', transform=ax.transAxes, va='top', ha='left', fontweight='bold')
    ax.legend(loc='lower right', frameon=False, fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(DEL, f'fig_realized_rc_series_{sgn}.png'), dpi=300); plt.close(fig)

# ------------------------------------------------------------------ Figure 2: cumulative net wealth
STYLE = {'OLS': ('#7f7f7f', '-'), 'Random Forest': ('#1f77b4', '-'), 'XGBoost': ('#d62728', '-'), 'LightGBM': ('#2ca02c', '-'), 'GBRT': ('#ff7f0e', '-'),
         'NN1': ('#9467bd', '-'), 'NN2': ('#8c564b', '-'), 'NN3': ('#e377c2', '-'), 'NN4': ('#17becf', '-'), 'NN5': ('#bcbd22', '-'), 'TAIEX': ('#222222', '--')}
months_oos = list(tx.month); W = pd.DataFrame({'month': ['2011/12 (initial)'] + months_oos})
W['TAIEX'] = np.concatenate([[1.0], np.cumprod(1 + tx.TAIEX_return.values)])
W['TAIEX_price_index'] = np.concatenate([[1.0], np.cumprod(1 + tx.TAIEX_price_index_return.values)])
for w in ('EW', 'VW'):
    for m in ML: W[f'{w}_{m}'] = np.concatenate([[1.0], np.cumprod(1 + cell('base', m, w).net_return.values)])
W.to_csv(os.path.join(DEL, 'fig_cumulative_net_wealth_data.csv'), index=False)
dates = [pd.Timestamp('2011-12-31')] + [todate(m) for m in months_oos]
ymax = 1.05 * W.drop(columns='month').to_numpy().max(); ymin = 0.95 * W.drop(columns='month').to_numpy().min()
for lab, w, grp in [('a', 'EW', ['OLS'] + TREES), ('b', 'EW', NNS), ('c', 'VW', ['OLS'] + TREES), ('d', 'VW', NNS)]:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for m in grp: ax.plot(dates, W[f'{w}_{m}'], color=STYLE[m][0], ls=STYLE[m][1], lw=1.3, label=f'{m} (top {SEL[(m, w)]}%)')
    ax.plot(dates, W['TAIEX'], color=STYLE['TAIEX'][0], ls=STYLE['TAIEX'][1], lw=1.5, label='TAIEX (total return index)')
    ax.set_yscale('log'); ax.set_ylim(ymin, ymax)
    ax.set_yticks([t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32) if ymin <= t <= ymax]); ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g')); ax.minorticks_off()
    ax.axhline(1, color='black', lw=0.5); ax.set_xlabel('Month'); ax.set_ylabel('Cumulative net wealth (January 2012 = 1, log scale)')
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.text(0.015, 0.97, f'({lab})', transform=ax.transAxes, va='top', ha='left', fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(0.06, 0.99), frameon=False, fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(DEL, f'fig_cumulative_net_wealth_{lab}.png'), dpi=300); plt.close(fig)
pd.Series({f'{m}|{w}': k for (m, w), k in SEL.items()}).to_json(os.path.join(OUT, 'selected_cutoffs.json'))
print('deliverables written to', DEL)
