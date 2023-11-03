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
import evaluation as eval

%reload_ext autoreload
%autoreload 2

# %%

path_vmoutput_gp = path_root + 'vm_output_gp/'
path_vmoutput_lasso = path_root + 'vm_output_lasso/'
path_vmoutput_rf = path_root + 'vm_output_rf/'
path_vmoutput_xgboost = path_root + 'vm_output_xgboost/'


# %%

# load predictions (from cross-validation)

# sparse GP
path_output_dir = path_vmoutput_gp + '2023-09-15_from-updated-adore/'
df_p_gp_all = utils.read_result_files(path_output_dir, file_type='preds')

# LASSO
path_output_dir = path_vmoutput_lasso + '2023-09-15_from-updated-adore/'
df_p_lasso_all = utils.read_result_files(path_output_dir, file_type='preds')

# RF
path_output_dir = path_vmoutput_rf + '2023-09-15_from-updated-adore/'
df_p_rf_all = utils.read_result_files(path_output_dir, file_type='preds')

# XGBoost
path_output_dir = path_vmoutput_xgboost + '2023-09-15_from-updated-adore/'
df_p_xgboost_all = utils.read_result_files(path_output_dir, file_type='preds')

# %%

# select run 

#groupsplit = 'totallyrandom'
groupsplit = 'occurrence'

conctype = 'molar'
#conctype = 'mass'

tax_pdm = 'none'

df_p = eval.filter_and_merge_predictions(df_p_gp_all, 
                                         df_p_lasso_all,
                                         df_p_rf_all,
                                         df_p_xgboost_all,
                                         groupsplit,
                                         conctype,
                                         tax_pdm)

# %%

# calculate residuals
df_p['gp_residual'] = df_p['gp_pred'] - df_p['true'] 
df_p['lasso_residual'] = df_p['lasso_pred'] - df_p['true'] 
df_p['rf_residual'] = df_p['rf_pred'] - df_p['true'] 
df_p['xgboost_residual'] = df_p['xgboost_pred'] - df_p['true'] 

# count chemicals and species
df_p['n_chemicals'] = df_p.groupby(['chem_fp', 'test_cas'])['result_id'].transform('count')
df_p['n_species'] = df_p.groupby(['chem_fp', 'tax_name'])['result_id'].transform('count')

df_p

# %%

# wide to long (predictions))
id_vars = ['chem_fp', 'result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs', 'n_species', 'n_chemicals', 'true']
value_vars = ['gp_pred', 'lasso_pred', 'rf_pred', 'xgboost_pred']
df_p_long = df_p.melt(id_vars=id_vars, 
                      value_vars=value_vars,
                      value_name='pred',
                      var_name='type')

df_p_long['type'] = df_p_long['type'].str.replace('gp_pred', 'GP')
df_p_long['type'] = df_p_long['type'].str.replace('lasso_pred', 'LASSO')
df_p_long['type'] = df_p_long['type'].str.replace('rf_pred', 'RF')
df_p_long['type'] = df_p_long['type'].str.replace('xgboost_pred', 'XGBoost')

# wide to long (residuals)
id_vars = ['chem_fp', 'result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs', 'n_species', 'n_chemicals', 'true']
value_vars = ['gp_residual', 'lasso_residual', 'rf_residual', 'xgboost_residual']
df_p_long2 = df_p.melt(id_vars=id_vars, 
                      value_vars=value_vars,
                      value_name='residual',
                      var_name='type')

# add residual to long data frame
df_p_long['residual'] = df_p_long2['residual']

# categorical variable
list_cols_fps = ['MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred']
df_p_long = utils._transform_to_categorical(df_p_long, 'chem_fp', list_cols_fps)
df_p_long

# %%

# Tukey Anscombe plot: residual vs fitted

# not colored
df_plot = df_p_long.copy()
ymax = df_plot['residual'].abs().max()
(ggplot(data=df_plot, mapping=aes(x='pred', y='residual'))
    + geom_point(shape='.', color='grey', alpha=0.1) 
    + scale_y_continuous(limits=[-ymax, ymax])
    + facet_grid('chem_fp ~ type')
    + theme_minimal()
    + theme(figure_size=(15, 9))
)

# %%

# colored by true concentration
df_plot = df_p_long.copy()
ymax = df_plot['residual'].abs().max()
(ggplot(data=df_plot, mapping=aes(x='pred', y='residual', color='true'))
    + geom_point(shape='.', alpha=0.1) 
    + scale_y_continuous(limits=[-ymax, ymax])
    + scale_color_cmap('cividis')
    + facet_grid('chem_fp ~ type')
    + theme_minimal()
    + theme(figure_size=(15, 9))
)

# %%

# check LASSO top right cluster
df_oi = df_p_long[(df_p_long['type'] == 'LASSO')].copy()
df_oi['special'] = 'normal'
#df_oi.loc[(df_p_long['pred'] > -2.2) & (df_p_long['residual'] > 3), 'special'] = 'weird'
df_oi.loc[(df_p_long['chem_name'].isin(['Potassium cyanide', 'Sodium cyanide'])), 'special'] = 'weird'

(ggplot(data=df_oi, mapping=aes(x='pred', y='residual', color='special'))
    + geom_point(shape='.', alpha=0.1) 
    + facet_wrap('~ chem_fp')
    + theme_minimal()
)

# %%

# nothing suspicious for the species, but this cluster mainly contains entries for two chemicals:
# - sodium cyanide, potassium cyanide
# - also there are progargyl alcohol and Potassium dimethyldithiocarbamate
df_oi[df_oi['special'] == 'weird']['tax_gs'].value_counts()
df_oi[df_oi['special'] == 'weird']['chem_name'].value_counts()
#df_oi[df_oi['special'] == 'weird']['n_chemicals'].value_counts()

# %%

# TA colored by cyanides
df_plot = df_p_long.copy()
df_plot['compounds'] = 'others'
df_plot.loc[(df_plot['chem_name'].str.contains('cyanide')), 'compounds'] = 'cyanide'
df_plot = df_plot.sort_values('compounds', ascending=False)

ymax = df_plot['residual'].abs().max()
(ggplot(data=df_plot, mapping=aes(x='pred', y='residual', color='compounds'))
    + geom_point(shape='.', alpha=0.5) 
    + scale_y_continuous(limits=[-ymax, ymax])
    + facet_grid('chem_fp ~ type')
    + theme_minimal()
    + theme(figure_size=(15, 8))
)

# %%

# TA colored by cyanides (4 colors)
df_plot = df_p_long.copy()
df_plot['compounds'] = 'others'
df_plot.loc[(df_plot['chem_name'].isin(['Potassium cyanide'])), 'compounds'] = 'Potassium cyanide'
df_plot.loc[(df_plot['chem_name'].isin(['Sodium cyanide'])), 'compounds'] = 'Sodium cyanide'
df_plot.loc[(df_plot['chem_name'].isin(['Octyl cyanide'])), 'compounds'] = 'Octyl cyanide'
df_plot.loc[(df_plot['chem_name'].isin(['Allyl cyanide'])), 'compounds'] = 'Allyl cyanide'
df_plot = df_plot.sort_values('compounds', ascending=False)

list_colors = ['#EFC201', '#E47900', '#B5134B', '#46093E']   # '#83920E', 

ymax = df_plot['residual'].abs().max()
(ggplot(data=df_plot, mapping=aes(x='pred', y='residual', color='compounds'))
    + geom_point(shape='.', alpha=1) 
    + scale_y_continuous(limits=[-ymax, ymax])
    + scale_x_continuous(limits=[-5, 0])
    + scale_color_manual(values=list_colors+['grey'])
    + facet_grid('chem_fp ~ type')
    + theme_minimal()
    + theme(figure_size=(15, 8))
)
# %%

# check RF bottom left cluster
df_oi = df_p_long[(df_p_long['type'] == 'RF')].copy()
df_oi['special'] = 'normal'
#df_oi.loc[(df_p_long['pred'] < -8) & (df_p_long['residual'] < 0), 'special'] = 'weird'
df_oi.loc[(df_p_long['chem_name'] == 'Dieldrin'), 'special'] = 'weird'
#df_oi.loc[(df_p_long['chem_name'] == 'Cypermethrin'), 'special'] = 'weird'

(ggplot(data=df_oi, mapping=aes(x='pred', y='residual', color='special', shape='special'))
    + geom_point(shape='.', alpha=0.1) 
   # + geom_point(alpha=0.1) 
    + facet_wrap('~ chem_fp')
    + theme_minimal()
)

# %%

df_oi[df_oi['special'] == 'weird']['tax_gs'].value_counts()
df_oi[df_oi['special'] == 'weird']['chem_name'].value_counts()

# %%

# TA colored by Dieldrin
df_plot = df_p_long.copy()
df_plot['compounds'] = 'others'
df_plot.loc[(df_plot['chem_name'].isin(['Dieldrin'])), 'compounds'] = 'Dieldrin'

ymax = df_plot['residual'].abs().max()
(ggplot(data=df_plot, mapping=aes(x='pred', y='residual', color='compounds'))
    + geom_point(shape='.', alpha=0.1) 
    + scale_y_continuous(limits=[-ymax, ymax])
#    + scale_color_cmap('cividis')
    + facet_grid('chem_fp ~ type')
    + theme_minimal()
    + theme(figure_size=(15, 8))
)

# %%

# Fenitrothion

# TA colored by Fenitrothion    
df_plot = df_p_long.copy()
df_plot['compounds'] = 'others'
df_plot.loc[(df_plot['chem_name'].isin(['Fenitrothion'])), 'compounds'] = 'Fenitrothion'
#df_plot.loc[(df_plot['chem_name'].isin(['Fenitrothion'])) & (df_plot['residual'] > 1.5), 'compounds'] = 'Fenitrothion-resid>1.5'
df_plot.loc[(df_plot['chem_name'].isin(['Fenitrothion'])) & (df_plot['tax_name'] == 'Western Mosquitofish'), 'compounds'] = 'Fenitrothion-Western'
df_plot = df_plot.sort_values('compounds', ascending=False)

ymax = df_plot['residual'].abs().max()
(ggplot(data=df_plot, mapping=aes(x='pred', y='residual', color='compounds'))
    + geom_point(shape='.', alpha=0.1) 
    + scale_y_continuous(limits=[-ymax, ymax])
    + facet_grid('chem_fp ~ type')
    + theme_minimal()
    + theme(figure_size=(15, 8))
)

# %%

(ggplot(data=df_plot, mapping=aes(x='pred', y='true', color='compounds'))
    + geom_point(shape='.', alpha=0.1) 
#    + scale_y_continuous(limits=[-ymax, ymax])
    + facet_grid('chem_fp ~ type')
    + theme_minimal()
    + theme(figure_size=(15, 8))
)

# %%

# all on Western Mosquitofish
df_plot[(df_plot['compounds'] != 'others') & (df_plot['residual'] > 1.5)]['tax_name'].unique()
df_plot[(df_plot['compounds'] != 'others') & (df_plot['residual'] > 1.5)]['result_id'].unique()

# %%
# prepare plotting

# colors
# https://www.pinterest.ch/pin/70439181665757203/
# TODO check colorblind safeness
list_colors = ['#83920E', '#EFC201', '#E47900', '#B5134B', '#46093E']

# %%

# histogram for all residuals
(ggplot(data=df_p_long, mapping=aes(x='residual', color='type'))
# + geom_histogram(position='identity', alpha=1.0, fill='none', binwidth=0.25)
 + geom_density()
 + scale_color_manual(values=list_colors)
 + facet_wrap('~ chem_fp')
 + theme_minimal()
 + labs(color='model')
)

# %%

# histograms for most common species
df_plot = df_p_long.copy()
#df_plot = df_plot[df_plot['n_species'] > 500]
#df_plot = df_plot[(df_plot['n_species'] > 230) & (df_plot['n_species'] <= 500)]
df_plot = df_plot[(df_plot['n_species'] > 100) & (df_plot['n_species'] <= 230)]
list_categories = df_plot['tax_name'].value_counts().index
df_plot['tax_name'] = pd.Categorical(df_plot['tax_name'],
                                     categories=list_categories,
                                     ordered=True)

xmax = df_plot['residual'].abs().max()

(ggplot(data=df_plot, 
        mapping=aes(x='residual', color='type'))
    + geom_histogram(position='identity', alpha=1.0, fill='none', binwidth=0.25)
    + geom_vline(xintercept = -2, color='grey')
    + geom_vline(xintercept = 0, color='grey', linetype='dashed')
    + geom_vline(xintercept = 2, color='grey')
    + facet_grid('chem_fp ~ tax_name')
    + scale_x_continuous(limits=[-xmax, xmax])
    + scale_color_manual(values=list_colors)
    + theme_minimal()
    + labs(color='model')
    + theme(figure_size=(18, 10))
)

# %%

# histograms for most common chemicals
df_plot = df_p_long.copy()
#df_plot = df_plot[df_plot['n_chemicals'] > 350]
#df_plot = df_plot[(df_plot['n_chemicals'] > 250) & (df_plot['n_chemicals'] <= 350)]
#df_plot = df_plot[(df_plot['n_chemicals'] > 150) & (df_plot['n_chemicals'] <= 250)]
#df_plot = df_plot[(df_plot['n_chemicals'] > 125) & (df_plot['n_chemicals'] <= 150)]
#df_plot = df_plot[(df_plot['n_chemicals'] > 100) & (df_plot['n_chemicals'] <= 125)]
df_plot = df_plot[(df_plot['n_chemicals'] > 80) & (df_plot['n_chemicals'] <= 100)]
list_categories = df_plot['chem_name'].value_counts().index
df_plot['chem_name'] = pd.Categorical(df_plot['chem_name'],
                                      categories=list_categories,
                                      ordered=True)
xmax = df_plot['residual'].abs().max()

(ggplot(data=df_plot, 
        mapping=aes(x='residual', color='type'))
    + geom_histogram(position='identity', alpha=1.0, fill='none', binwidth=0.25)
    + geom_vline(xintercept = -2, color='grey')
    + geom_vline(xintercept = 0, color='grey', linetype='dashed')
    + geom_vline(xintercept = 2, color='grey')
    + facet_grid('chem_fp ~ chem_name')
    + scale_x_continuous(limits=[-xmax, xmax])
    + scale_color_manual(values=list_colors)
    + theme_minimal()
    + labs(color='model')
    + theme(figure_size=(18, 10))
)

# %%

# histograms for most common chemicals and top fish
df_plot = df_p_long.copy()

list_fish = ['Rainbow Trout', 'Fathead Minnow', 'Bluegill']

df_plot = df_plot[df_plot['tax_name'].isin(list_fish)].copy()

chem_name = "p,p'-DDT"
chem_name = 'Carbaryl'
chem_name = 'Fenitrothion'
df_plot = df_plot[df_plot['chem_name'] == chem_name]

xmax = df_plot['residual'].abs().max()

(ggplot(data=df_plot, 
        mapping=aes(x='residual', fill='tax_name'))
    + geom_histogram(position='stack', alpha=0.5, color='darkgrey', binwidth=0.25)
    + geom_vline(xintercept = -2, color='grey')
    + geom_vline(xintercept = 0, color='grey', linetype='dashed')
    + geom_vline(xintercept = 2, color='grey')
    + facet_grid('chem_fp ~ type')
    + scale_x_continuous(limits=[-xmax, xmax])
    + scale_fill_manual(values=list_colors)
    + theme_minimal()
    + labs(fill='model', title=chem_name)
    + theme(figure_size=(18, 10))
)







# %%

# order of magnitude
oom = 1

# entries
print('entries')
print(df_p[(df_p['gp_residual']).abs() > oom].shape[0])
print(df_p[(df_p['lasso_residual']).abs() > oom].shape[0])
print()

# chemicals
print('chemicals')
print(df_p[(df_p['gp_residual']).abs() > oom]['test_cas'].nunique())
print(df_p[(df_p['lasso_residual']).abs() > oom]['test_cas'].nunique())
print()

# species
print('species')
print(df_p[(df_p['gp_residual']).abs() > oom]['tax_gs'].nunique())
print(df_p[(df_p['lasso_residual']).abs() > oom]['tax_gs'].nunique())
print()

# %%
# %%

# LASSO: experimental features not relevant --> same prediction for same fish and chemical
df_p.groupby(['chem_fp'])['lasso_pred'].value_counts().head(10)
# %%

# top prediction
conc_top = df_p['lasso_pred'].value_counts().index[0]
print(df_p[df_p['lasso_pred'] == conc_top]['chem_name'].unique())
print(df_p[df_p['lasso_pred'] == conc_top]['tax_name'].unique())

# %%

# top 2 prediction
conc_top2 = df_p['lasso_pred'].value_counts().index[6]
print(df_p[df_p['lasso_pred'] == conc_top2]['chem_name'].unique())
print(df_p[df_p['lasso_pred'] == conc_top2]['tax_name'].unique())

# %%

# GP:
df_p.groupby(['chem_fp'])['gp_pred'].value_counts().head(10)

# %%

# top prediction
conc_top = df_p['gp_pred'].value_counts().index[0]
print(df_p[df_p['gp_pred'] == conc_top]['chem_name'].unique())
print(df_p[df_p['gp_pred'] == conc_top]['tax_name'].unique())

# %%

# top 2 prediction
conc_top2 = df_p['gp_pred'].value_counts().index[6]
print(df_p[df_p['gp_pred'] == conc_top2]['chem_name'].unique())
print(df_p[df_p['gp_pred'] == conc_top2]['tax_name'].unique())
# %%

df_p[(df_p['chem_name'] == 'Trichlorfon')
     & (df_p['tax_name'] == 'Rainbow Trout')
     & (df_p['chem_fp'] == 'MACCS')]#['true'].mean()

# %%

df_p[(df_p['chem_name'] == 'Methoxychlor')
     & (df_p['tax_name'] == 'Brook Trout')
     & (df_p['chem_fp'] == 'MACCS')]#['true'].mean()

# %%

# experimental setting not stored
list_cols_exp = [c for c in df_p.columns if ('result' in c) or ('test' in c)]
list_cols_exp


# %%

# Random Forest
df_p.groupby(['chem_fp'])['rf_pred'].value_counts().head(10)

# %%

# XGBoost
df_p.groupby(['chem_fp'])['xgboost_pred'].value_counts().head(10)

# %%
# top prediction
conc_top = df_p['xgboost_pred'].value_counts().index[0]
print(df_p[df_p['xgboost_pred'] == conc_top]['chem_name'].unique())
print(df_p[df_p['xgboost_pred'] == conc_top]['tax_name'].unique())
# %%

# top 2 prediction
conc_top2 = df_p['gp_pred'].value_counts().index[1]
print(df_p[df_p['gp_pred'] == conc_top2]['chem_name'].unique())
print(df_p[df_p['gp_pred'] == conc_top2]['tax_name'].unique())
# %%
