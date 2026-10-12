# -*- coding: utf-8 -*-
"""Step 7. Appendix Table A.1 helper: fixed settings, validation grid and validation-selected values."""
import pandas as pd, os
from config import *
a = pd.read_csv(os.path.join(OUT, 'audit_validation_grid_linear_trees.csv')); b = pd.read_csv(os.path.join(OUT, 'audit_validation_grid_nn.csv'))
g = pd.concat([a, b]); s = g[g.selected].drop_duplicates('model').set_index('model')
FIXED = {
 'OLS': ('Linear model', 'Ordinary least squares on all 13 predictors; no regularisation', 'None (no tuning parameter)'),
 'Random Forest': ('Tree ensemble', '500 trees; bootstrap', 'max_depth {2,3,4} x min_samples_leaf {5,10,20} x max_features {sqrt, all}'),
 'XGBoost': ('Tree ensemble', 'min_child_weight 5; L2 penalty 1; row subsample 0.8; column subsample 0.8', 'max_depth {1,2} x learning_rate {0.01,0.05} x n_estimators {50,100,200}'),
 'LightGBM': ('Tree ensemble', 'num_leaves 2^max_depth; min_child_samples 10; L2 penalty 1; row subsample 0.8; column subsample 0.8', 'max_depth {1,2} x learning_rate {0.01,0.05} x n_estimators {50,100,200}'),
 'GBRT': ('Tree ensemble', 'min_samples_leaf 10; row subsample 0.8; squared-error loss', 'max_depth {1,2} x learning_rate {0.01,0.05} x n_estimators {50,100,200}'),
}
W = {'NN1': '32', 'NN2': '32-16', 'NN3': '32-16-8', 'NN4': '32-16-8-4', 'NN5': '32-16-8-4-2'}
for k, v in W.items():
    FIXED[k] = ('Feedforward neural network', f'Hidden layers {v}; ReLU; batch normalisation; Adam, full batch; L1 weight penalty; early stopping '
                '(patience 10, max 200 epochs, most recent 20% of training window); average of 5 seeds; standardised inputs and target',
                'L1 penalty {1e-5,1e-4,1e-3} x learning rate {0.001,0.01}')
rows = [dict(model=m, family=FIXED[m][0], fixed_configuration=FIXED[m][1], validation_grid=FIXED[m][2],
             validation_selected=s.loc[m, 'params'], mean_validation_r2_pct=round(100 * s.loc[m, 'mean_val_r2'], 3),
             n_candidate_configurations=int((g.model == m).sum())) for m in ML]
df = pd.DataFrame(rows)
df['common_settings'] = ('Predictors: own realized-RC lags 1,2,3,6,12; rolling mean and SD over 3,6,12 months; sine/cosine of target calendar month. '
                         'Separate model per signal; expanding window re-estimated every month; criterion = mean validation R2 vs expanding historical mean '
                         'over six signals, target months 2007/01-2011/12; one configuration per model fixed for 2012/01-2024/12.')
df.to_csv(os.path.join(OUT, 'deliverables', 'appendix_A1_model_configurations.csv'), index=False); print(df[['model', 'validation_selected', 'mean_validation_r2_pct']].to_string())
