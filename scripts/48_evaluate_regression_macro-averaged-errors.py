
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import utils
import modeling as mod

%reload_ext autoreload
%autoreload 2

# %%

path_output = path_root + 'output/regression/'
path_figures = path_output + 'figures/'

# %%

def compile_predictions(modeltype):

    # load file
    df = pd.read_csv(path_output + modeltype + '_predictions.csv')

    # add modeltype
    if modeltype == 'lasso':
        df['model'] = 'LASSO'
    elif modeltype == 'rf':
        df['model'] = 'RF'
    elif modeltype == 'xgboost':
        df['model'] = 'XGBoost'
    elif modeltype == 'gp':
        df['model'] = 'GP'

    return df
    

# %%

# load LASSO: 
# 6 fps x 2 groupsplits x 4 sets x 2 concentrations = 96 entries
modeltype = 'lasso'
df_pr_lasso = compile_predictions(modeltype)

# %%

# load RF
# 6 fps x 2 groupsplits x 4 sets x 2 concentrations = 96 entries
modeltype = 'rf'
df_pr_rf = compile_predictions(modeltype)

# %%

# load XGBoost
# 6 fps x 2 groupsplits x 4 sets x 2 concentrations = 96 entries
modeltype = 'xgboost'
df_pr_xgboost = compile_predictions(modeltype)

# %%

# load GP
# 6 fps x 2 groupsplits x 4 sets x 2 tax_pdm x 2 concentrations = 192 entries
modeltype = 'gp'
df_pr_gp = compile_predictions(modeltype)

# %%

# GP: without tax_pdm
df_pr_gp = df_pr_gp[df_pr_gp['tax_pdm'] == 'none'].copy()

# %%

# concatenate all predictions
df_preds = pd.concat([df_pr_lasso, df_pr_rf, df_pr_xgboost, df_pr_gp], axis=0)

# make column with all true concentrations
df_preds['conc_true'] = np.nan
df_preds.loc[df_preds['conctype'] == 'mass', 'conc_true'] = df_preds.loc[df_preds['conctype'] == 'mass', 'result_conc1_mean_log']
df_preds.loc[df_preds['conctype'] == 'molar', 'conc_true'] = df_preds.loc[df_preds['conctype'] == 'molar', 'result_conc1_mean_mol_log']
#df_preds[['result_conc1_mean_log', 'result_conc1_mean_mol_log', 'conc_true', 'conc_pred']]

# %%

# set columns
col_true = 'conc_true'
col_pred = 'conc_pred'

# initialize
list_dfs = []

# calculate microRMSE
list_cols = ['model', 'chem_fp', 'groupsplit', 'conctype', 'fold']
df_e = df_preds.groupby(list_cols).apply(mod._calculate_mse,
                                         col_true=col_true,
                                         col_pred=col_pred)
df_e = df_e.to_frame().rename(columns={0:'MSE'})
df_e['RMSE'] = np.sqrt(df_e['MSE'])
df_e = df_e.reset_index()
df_e['averagetype'] = 'micro'
list_dfs.append(df_e)
df_e

# %%

# calculate taxonRMSE
df_e = df_preds.groupby(list_cols + ['tax_name']).apply(mod._calculate_mse,
                                                        col_true=col_true,
                                                        col_pred=col_pred)
df_e = df_e.to_frame().rename(columns={0:'MSE'})
df_e['RMSE'] = np.sqrt(df_e['MSE'])
df_e = df_e.reset_index()
df_e_t = df_e.copy()
df_e = df_e.groupby(list_cols)['MSE'].mean()
df_e = df_e.to_frame().rename(columns={0:'MSE'})
df_e['RMSE'] = np.sqrt(df_e['MSE'])
df_e = df_e.reset_index()
df_e['averagetype'] = 'taxon'
list_dfs.append(df_e)
df_e

# %%

# calculate chemRMSE
df_e = df_preds.groupby(list_cols + ['chem_name']).apply(mod._calculate_mse,
                                                         col_true=col_true,
                                                         col_pred=col_pred)
df_e = df_e.to_frame().rename(columns={0:'MSE'})
df_e['RMSE'] = np.sqrt(df_e['MSE'])
df_e = df_e.reset_index()
df_e_c = df_e.copy()
df_e = df_e.groupby(list_cols)['MSE'].mean()
df_e = df_e.to_frame().rename(columns={0:'MSE'})
df_e['RMSE'] = np.sqrt(df_e['MSE'])
df_e = df_e.reset_index()
df_e['averagetype'] = 'chemical'
list_dfs.append(df_e)
df_e

# %%

# calculate macroRMSE
df_e = df_preds.groupby(list_cols + ['tax_name', 'chem_name']).apply(mod._calculate_mse,
                                                                     col_true=col_true,
                                                                     col_pred=col_pred)
df_e = df_e.to_frame().rename(columns={0:'MSE'})
df_e['RMSE'] = np.sqrt(df_e['MSE'])
df_e = df_e.reset_index()
df_e_ma = df_e.copy()
df_e = df_e.groupby(list_cols)['MSE'].mean()
df_e = df_e.to_frame().rename(columns={0:'MSE'})
df_e['RMSE'] = np.sqrt(df_e['MSE'])
df_e = df_e.reset_index()
df_e['averagetype'] = 'macro'
list_dfs.append(df_e)
df_e

# %%

df_es = pd.concat(list_dfs)
df_es

# %%

# prepare plotting

# color specifications

# for chem_fp: colors from CH2018 report
# and purple from https://www.pinterest.ch/pin/1130403575204052700/
# and yellow from https://www.pinterest.ch/pin/57632070223416732/
list_colors = ['#75aab9', '#dfc85e', '#998478', '#c194ac', '#80a58b', '#fbba76']

# for averagetype
list_colors_at = ['#998478', '#80a58b', '#75aab9', '#dfc85e']

# %%

df_plot = df_es[(df_es['fold'] == 'test')
                & (df_es['groupsplit'] == 'occurrence')
                & (df_es['conctype'] == 'molar')].copy()
list_categories = ['MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred']
df_plot = utils._transform_to_categorical(df_plot, 'chem_fp', list_categories)
df_plot = utils._transform_to_categorical(df_plot, 'model', ['LASSO', 'RF', 'XGBoost', 'GP'])
df_plot = utils._transform_to_categorical(df_plot, 'averagetype', ['micro', 'macro', 'taxon', 'chemical'])

(ggplot(data=df_plot, mapping=aes(x='chem_fp', y='RMSE', fill='averagetype'))
 + geom_col(position='dodge')
 + scale_fill_manual(values=list_colors_at)
 + facet_wrap('~ model')
 + theme_minimal()
 + theme(axis_text_x=element_text(angle=90))
 + labs(x='', fill='average')
 )

# %%

(ggplot(data=df_plot, mapping=aes(x='averagetype', y='RMSE', fill='chem_fp'))
 + geom_col(position='dodge')
 + scale_fill_manual(values=list_colors)
 + facet_wrap('~ model')
 + theme_minimal()
 + theme(axis_text_x=element_text(angle=90))
 + labs(x='', fill='fingerprint')
 )

# %%

(ggplot(data=df_plot, mapping=aes(x='model', y='RMSE', fill='chem_fp'))
 + geom_col(position='dodge')
 + scale_fill_manual(values=list_colors)
 + facet_wrap('~ averagetype')
 + theme_minimal()
 + theme(axis_text_x=element_text(angle=90))
 + labs(x='', fill='fingerprint')
 )

# %%

title = 'taxon'
#title = 'chemical'
#title = 'macro'

if title == 'taxon':
    df_plot = df_e_t.copy()
elif title == 'chemical':
    df_plot = df_e_c.copy()
elif title == 'macro':
    df_plot = df_e_ma.copy()
else:
    print('warning!')
df_plot = df_plot[(df_plot['fold'] == 'test')
                 & (df_plot['groupsplit'] == 'occurrence')
                 & (df_plot['conctype'] == 'molar')].copy()
list_categories = ['MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred']
df_plot = utils._transform_to_categorical(df_plot, 'chem_fp', list_categories)
df_plot = utils._transform_to_categorical(df_plot, 'model', ['LASSO', 'RF', 'XGBoost', 'GP'])

(ggplot(data=df_plot, mapping=aes(x='RMSE'))
 + geom_density()
 + scale_color_manual(values=list_colors)
 + facet_grid('chem_fp ~ model')
 + theme_minimal()
 + theme(figure_size=(5,6))
 + labs(title=title)
 )


# %%

# taxon
df_plot = df_e_t.copy()
df_plot = df_plot.sort_values(['model', 'chem_fp', 'RMSE'])
df_plot.groupby(['model', 'chem_fp']).head(5)['tax_name'].value_counts()
df_plot.groupby(['model', 'chem_fp']).tail(5)['tax_name'].value_counts()
#df_plot[df_plot['RMSE'] < 0.1]['tax_name'].value_counts()
#df_plot[df_plot['RMSE'] > 2]['tax_name'].value_counts()

# %%

# chemical
df_plot = df_e_c.copy()
df_plot = df_plot.sort_values(['model', 'chem_fp', 'RMSE'])
df_plot.groupby(['model', 'chem_fp']).head(5)['chem_name'].value_counts()
df_plot.groupby(['model', 'chem_fp']).tail(5)['chem_name'].value_counts()
#df_plot[df_plot['RMSE'] < 0.1]['chem_name'].value_counts()
#df_plot[df_plot['RMSE'] > 3]['chem_name'].value_counts()

# %%

# macro
df_plot = df_e_ma.copy()
df_plot['names'] = df_plot['tax_name'] + ' - ' + df_plot['chem_name']
df_plot = df_plot.sort_values(['model', 'chem_fp', 'RMSE'])
df_plot.groupby(['model', 'chem_fp']).head(5)['names'].value_counts()
df_plot.groupby(['model', 'chem_fp']).tail(5)['names'].value_counts()
#df_plot[df_plot['RMSE'] < 0.1]['names'].value_counts()
#df_plot[df_plot['RMSE'] > 3]['names'].value_counts()

# %%
