# %%

import os
if os.getcwd().endswith('scripts'):
    path_root = '../'
else:
    path_root = './' 
import sys
sys.path.insert(0, path_root + 'src/')

import numpy as np
import pandas as pd

from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns

import utils

%reload_ext autoreload
%autoreload 2

# %%

path_vmoutput = path_root + 'vm_output_xgboost/'
path_output = path_root + 'output/regression/'
path_figures = path_output + 'figures/'

# %%

# XGBoost: Fish data, both concentrations, 2023-08-14

# completely new data pre-processing from ECOTOX 2022-09-15
# for the four fingerprints
# including chemical properties (mw, mp, ws, clogp)
# groupsplit: totallyrandom, occurrence (no scaffolds)

#param_grid = [
    #{
     ## features
     #'chem_fp': ['MACCS', 'pcp', 'Morgan', 'mol2vec', 'ToxPrint'], 
     #'chem_prop': ['chemprop'],                 #['none', 'chemprop'],
     #'tax_pdm': ['none'],                       #['none', 'pdm', 'pdm-squared'],
     #'tax_prop': ['taxprop-migrate2'],          #['none', 'taxprop-migrate2', 'taxprop-migrate5'],
     #'exp': ['exp-dropfirst'],                  #['none', 'exp-dropnone', 'exp-dropfirst'],     # use dropfirst
     ## splits
     #'groupsplit': ['occurrence', 'totallyrandom'], 
     ## concentration
     #'conctype': ['molar', 'mass'] 
    #}
#]
#hyperparam_grid = [
    #{
    ## model hyperparameters     
    #'n_estimators': [50, 100],
    #'eta': [0.1, 0.2, 0.3],
    #'gamma': [0, 1, 10], 
    #'max_depth': [3, 6, 9, 12], 
    #'min_child_weight': [1, 3, 5],
    #'subsample': [0.5, 1.],
    #}
#]

path_output_dir = path_vmoutput + '2023-08-14_bothconcentrations/'
df_errors = utils.read_result_files(path_output_dir, file_type='error')

# %%

# categorical variables for fingerprints
col = 'chem_fp'
list_categories = ['MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)

# categorical variable for folds
col = 'fold'
list_categories = ['mean', '0', '1', '2', '3', '4']
df_errors[col] = df_errors[col].astype('str')
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)

# categorical variable for groupsplit
col = 'groupsplit'
list_categories = ['totallyrandom', 'occurrence']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)

# categorical variable for conctype
col = 'conctype'
list_categories = ['molar', 'mass']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)

# sort
df_errors = df_errors.sort_values(['chem_fp', 'groupsplit'])

# check loading
if df_errors.isna().sum().sum() > 0:
    print("warning: check whether loading was correct")
else:
    print("loading seems correct")

# %%

# only look at results with best hyperparameters
df_oi = df_errors[df_errors['best_hp'] == True].copy()

# get data frame with best hyperparmeters only
df_e_oi = df_errors[(df_errors['best_hp'] == True) &
                    (df_errors['set'] == 'valid') &
                    (df_errors['fold'] == 'mean')]
df_e_oi

# %%

# mean errors (train and valid of 5-fold CV)
df_e_v = df_oi[(df_oi['fold'] == 'mean')
#               & (df_oi['set'] == 'valid')
               ].copy()

list_cols = ['chem_fp', 'groupsplit', 'conctype', 'set', 'fold']
list_cols += ['chem_prop', 'tax_pdm', 'tax_prop', 'exp']
list_cols += ['best_hp', 'idx_hp', 'n_estimators', 'eta', 'gamma', 'max_depth', 'min_child_weight', 'subsample']
list_cols += ['r2', 'rmse', 'mae', 'pearson']
df_e_v[df_e_v['best_hp'] == True][list_cols].round(2)

# %%

# store 
df_e_v[df_e_v['best_hp'] == True][list_cols].round(5).to_csv(path_output + 'xgboost_CV-errors.csv', index=False)

# %%

# compare mass and molar concentration
metric = 'rmse'
(ggplot(data=df_e_v, mapping=aes(x='set', y=metric, fill='conctype'))
    + geom_col(position='dodge')
    + scale_fill_manual(values=['#7fc97f', '#beaed4'])
    + facet_grid("chem_fp ~ groupsplit") 
    + theme_minimal()
    + labs(y='RMSE', title='XGBoost')
 )

# %%

# plot metric vs a hyperparameter

    #'n_estimators': [50, 100],
    #'eta': [0.1, 0.2, 0.3],
    #'gamma': [0, 1, 10], 
    #'max_depth': [3, 6, 9, 12], 
    #'min_child_weight': [1, 3, 5],
    #'subsample': [0.5, 1.],

# choose metric
# TODO run for both metrics
metric = 'r2'
#metric = 'rmse'

# for each hyperparameter
list_cols = ['n_estimators', 'eta', 'gamma', 'max_depth', 'min_child_weight', 'subsample']
for col_x in list_cols:
    print(col_x)
    # only mean errors
    df_e_tv = df_errors[(df_errors['fold'] == 'mean')]

    # initialize
    dict_cols = {}
    list_tmp2 = []

    # for each fingerprint and groupsplit
    for chem_fp in ['MACCS', 'pcp', 'Morgan', 'mol2vec']:
        for groupsplit in ['totallyrandom', 'occurrence']:

            # get best hyperparams
            df_tmp = df_e_tv[(df_e_tv['chem_fp'] == chem_fp) & 
                             (df_e_tv['groupsplit'] == groupsplit) &
                             (df_e_tv['best_hp'] == True)]
            list_cols_tmp = [col for col in list_cols if col != col_x]
            for col_tmp in list_cols_tmp:
                dict_cols[col_tmp] = df_tmp[col_tmp].iloc[0]

            # get data frame with selected values
            df_tmp2 = df_e_tv[(df_e_tv['chem_fp'] == chem_fp) & 
                              (df_e_tv['groupsplit'] == groupsplit)].copy()

            for col, value in dict_cols.items():
                df_tmp2 = df_tmp2[df_tmp2[col] == value].copy()

            # apend
            list_tmp2.append(df_tmp2)

    # concatenate
    df_e_tv = pd.concat(list_tmp2)

    # prepare plotting
    df_plot = df_e_tv.copy()
    df_plot_oi = df_e_oi.copy()
    df_plot['conctype_groupsplit'] = df_plot['conctype'].astype('str') + ' ' + df_plot['groupsplit'].astype('str')
    df_plot_oi['conctype_groupsplit'] = df_plot_oi['conctype'].astype('str') + ' ' + df_plot_oi['groupsplit'].astype('str')

    ymin = 0 if metric == 'rmse' else -1
    ymax = df_plot[metric].max() + 0.1 if metric == 'rmse' else 1

    g = (ggplot(data=df_plot, mapping=aes(x=col_x, 
                                          y=metric, 
                                          group='set', 
                                          fill='set', 
                                          color='set',
                                          ))
        + geom_vline(data=df_plot_oi, mapping=aes(xintercept=col_x), color='grey', size=1.5, alpha=0.4)
        + geom_line()
        + geom_point(alpha=0.5)
        + scale_fill_manual(values=['#d8b365', '#67a9cf'])
        + scale_color_manual(values=['#d8b365', '#67a9cf'])
        + scale_y_continuous(limits=(ymin, ymax))
        + facet_grid("conctype_groupsplit ~ chem_fp")
        + theme_minimal()
        + theme(strip_text_y=element_text(angle=0))
    )
    if metric == 'r2':
        g = g + labs(y="R$^2$")
    elif metric == 'rmse':
        g = g + labs(y="RMSE")
#    g.save(path_figures + '25_XGB_' + metric + '-vs-' + col_x + '.png', facecolor='white')
    print(g)

# %%

# look at errors per fold
df_oi = df_errors[(df_errors['best_hp'] == True)].copy()

df_plot = df_oi.copy()
df_plot['conctype_groupsplit'] = df_plot['conctype'].astype('str') + ' ' + df_plot['groupsplit'].astype('str')

g = (ggplot(data=df_plot, mapping=aes(x='fold', 
                                      y=metric, 
                                      fill='set', 
                                      color='set',
                                      group='set'))
    + geom_line()
    + geom_point()
#    + geom_col(position='position_dodge')
    + facet_grid("conctype_groupsplit ~ chem_fp")
    + scale_fill_manual(values=['#d8b365', '#67a9cf'])
    + scale_color_manual(values=['#d8b365', '#67a9cf'])
    + scale_y_continuous(limits=(ymin, ymax))
    + theme_minimal()
    + theme(axis_text_x=element_text(angle=90))
    + theme(strip_text_y=element_text(angle=0))
)
if metric == 'r2':
    g = g + labs(y="R$^2$")
elif metric == 'rmse':
    g = g + labs(y="RMSE")
#g.save(path_figures + '25_XGB_' + metric + '-vs-fold.png', facecolor='white')
g

# %%
