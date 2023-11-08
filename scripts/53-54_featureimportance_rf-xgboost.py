# Feature importance of RF and XGBoost

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

import pickle
import shap
from copy import copy

import matplotlib.pyplot as plt
from plotnine import *

import utils
import evaluation as eval

#%reload_ext autoreload
#%autoreload 2

# %%

# set paths
path_data = path_root + 'data/'
path_output = path_root + 'output/regression/'
path_figures = path_output + 'figures/'
path_features = path_output + 'features/'
path_shap = path_output + 'shap/'
path_pi = path_output + 'permimp/'
path_chem = path_root + 'data/chemicals/'
path_vmoutput_gp = path_root + 'vm_output_gp/'
path_vmoutput_lasso = path_root + 'vm_output_lasso/'
path_vmoutput_rf = path_root + 'vm_output_rf/'
path_vmoutput_xgboost = path_root + 'vm_output_xgboost/'

# %%

# load bits information
df_maccs = pd.read_csv(path_chem + 'maccs_bits.tsv', sep='\t')
df_pcp = pd.read_csv(path_chem + 'pubchem_bits.csv', sep='\t')
df_toxprint = pd.read_csv(path_chem + 'toxprint_bits.csv')
dict_bits = {}
dict_bits['MACCS'] = df_maccs
dict_bits['pcp'] = df_pcp
dict_bits['ToxPrint'] = df_toxprint

# load fish data
df_eco = pd.read_csv(path_data + 'processed/t-F2F_mortality.csv', low_memory=False)

# %%

# load all predictions (from cross-validation)

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

# set
modeltype = 'xgboost'
chem_fp = 'MACCS'
groupsplit = 'occurrence'
conctype = 'molar'

title = ' '.join((modeltype, chem_fp))
max_display = 10

# %%

# get selected predictions
df_p = eval.filter_and_merge_predictions(df_p_gp_all, 
                                         df_p_lasso_all,
                                         df_p_rf_all,
                                         df_p_xgboost_all,
                                         groupsplit,
                                         conctype,
                                         tax_pdm='none')
df_p = df_p[df_p['chem_fp'] == chem_fp]

# %%

# load features
filename_ending = '_'.join((modeltype, 'data', chem_fp, groupsplit, conctype)) + '.csv'
filename_features = path_features + filename_ending
df_data = pd.read_csv(filename_features)
list_cols = ['test_id', 'result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs']
df_features = df_data[[c for c in df_data.columns if c not in list_cols]]

# load permutation importance results
filename_ending = '_'.join((modeltype, 'permimp-trainvalid', chem_fp, groupsplit, conctype)) + '.p'
filename_pi_tv = path_pi + filename_ending
pi_result_tv = pickle.load(open(filename_pi_tv, 'rb'))
#print(pi_result_tv)
filename_ending = '_'.join((modeltype, 'permimp-test', chem_fp, groupsplit, conctype)) + '.p'
filename_pi_test = path_pi + filename_ending
pi_result_test = pickle.load(open(filename_pi_test, 'rb'))
#print(pi_result_test)

# %%

# add split column to df_data
col_split = '_'.join(('split', groupsplit))
df_data = pd.merge(df_data,
                   df_eco[['result_id', col_split]],
                   left_on=['result_id'],
                   right_on=['result_id'],
                   how='left')

# apply train test split
trainvalid_idx = df_data[df_data[col_split] != 'test'].index
test_idx = df_data[df_data[col_split] == 'test'].index
df_data_trainvalid = df_data.iloc[trainvalid_idx, :].reset_index(drop=True)
df_data_test = df_data.iloc[test_idx, :].reset_index(drop=True)

#%%

# sort by test importances
idx_sorted_importances = pi_result_test['importances_mean'].argsort()[::-1][:max_display][::-1]

# get long dataframe for trainvalid
df_pi_tv = pd.DataFrame(
    pi_result_tv['importances'][idx_sorted_importances].transpose(),
    columns=df_features.columns[idx_sorted_importances])
df_pi_tv_long = pd.melt(df_pi_tv, 
                        id_vars=[], 
                        value_vars=df_pi_tv.columns,
                        var_name='feature',
                        value_name='importance')
df_pi_tv_long['set'] = 'trainvalid'

# get long dataframe for test
df_pi_test = pd.DataFrame(
    pi_result_test['importances'][idx_sorted_importances].transpose(),
    columns=df_features.columns[idx_sorted_importances])
df_pi_test_long = pd.melt(df_pi_test, 
                        id_vars=[], 
                        value_vars=df_pi_test.columns,
                        var_name='feature',
                        value_name='importance')
df_pi_test_long['set'] = 'test'

# concatenate trainvalid and test
df_pi_long = pd.concat((df_pi_tv_long, df_pi_test_long))

# add bits explanation
if chem_fp in ['MACCS', 'pcp', 'ToxPrint']:
    df_bits = dict_bits[chem_fp]
    df_pi_long['bit'] = df_pi_long['feature'].apply(lambda x: int(x[-3:]) if chem_fp in x else -1)
    df_pi_long = pd.merge(df_pi_long,
                          df_bits,
                          left_on=['bit'],
                          right_on=['bit'],
                          how='left')
    df_pi_long['feature2'] = df_pi_long['feature'].astype('str') + ': ' + df_pi_long['description'].astype('str')
    df_pi_long['feature2'] = df_pi_long['feature2'].str.replace(': nan', '')
else:
    df_pi_long['feature2'] = df_pi_long['feature']

#df_pi_long

# %%

# permutation feature importances plot
df_pi_long['feature2'] = pd.Categorical(df_pi_long['feature2'],
                                        df_pi_long['feature2'].unique(),
                                        ordered=True)
df_plot = df_pi_long[df_pi_long['set'] == 'test'].copy()
(ggplot(data=df_plot, mapping=aes(y='importance', x='feature2'))
    + geom_boxplot(outlier_alpha=0)
    + geom_point(color='darkgrey', shape='.')
    + geom_vline(xintercept=0, linetype='--')
    + labs(y='Decrease in accuracy score', x='', title=title)
    + coord_flip()
    + theme_minimal()
    + theme(figure_size=(8, 6))
 )
# %%

# %%

# load explainer and SHAP values
filename_ending = '_'.join((modeltype, 'explainer', chem_fp, groupsplit, conctype)) + '.sav'
filename_expl = path_shap + filename_ending
#explainer = pickle.load(open(filename_expl, 'rb'))
#print(explainer)

filename_ending = '_'.join((modeltype, 'shapvalues', chem_fp, groupsplit, conctype)) + '.sav'
filename_sv = path_shap + filename_ending
shap_values = pickle.load(open(filename_sv, 'rb'))
#print(shap_values)

# rename feature names
if chem_fp in ['MACCS', 'pcp', 'ToxPrint']:
    list_featnames = shap_values.feature_names
    list_featnames2 = [': '.join((fn, df_bits.loc[df_bits['bit'] == int(fn[-3:]), 'description'].iloc[0])) if chem_fp in fn else fn for fn in list_featnames]
    shap_values.feature_names = list_featnames2

# %%

# Calculate weighted SHAP values

# get original shap values as dataframe and add identifier columns
list_features = shap_values.feature_names
df_shap = pd.DataFrame(shap_values.values,
                       columns=list_features)
list_cols = ['test_cas', 'chem_name', 'tax_name', 'tax_gs', 'split_occurrence']
df_shap2 = pd.concat((df_data_trainvalid[list_cols], df_shap), axis=1)

# calculate SHAPs weighted by chemical 
list_cols_gb = ['test_cas', 'chem_name']
df_shap_gb_chem = df_shap2.groupby(list_cols_gb)[list_features].mean()
shap_values_chem = copy(shap_values)
shap_values_chem.values = df_shap_gb_chem.to_numpy()

# calculate SHAPs weighted by taxon
list_cols_gb = ['tax_gs', 'tax_name']
df_shap_gb_tax = df_shap2.groupby(list_cols_gb)[list_features].mean()
shap_values_tax = copy(shap_values)
shap_values_tax.values = df_shap_gb_tax.to_numpy()

# calculate SHAPs weighted by chemical and taxon
list_cols_gb = ['test_cas', 'chem_name', 'tax_gs', 'tax_name']
df_shap_gb_chemtax = df_shap2.groupby(list_cols_gb)[list_features].mean()
shap_values_chemtax = copy(shap_values)
shap_values_chemtax.values = df_shap_gb_chemtax.to_numpy()

# %%

# Plotting remarks

# The default colormap should not be used as it is not colorblind safe.

# %%

# Plots for entire test set

# bar plot (averaged (=global)): micro-average
shap.plots.bar(shap_values,
               max_display=max_display+1,
               show=False)
plt.title('micro-average')
plt.tight_layout()
plt.show()

# bar plot (averaged (=global)): macro-average
shap.plots.bar(shap_values_chemtax,
               max_display=max_display+1,
               show=False)
plt.title('macro-average')
plt.tight_layout()
plt.show()

# bar plot (averaged (=global)): chemical macro-average
shap.plots.bar(shap_values_chem,
               max_display=max_display+1,
               show=False)
plt.title('chemical macro-average')
plt.tight_layout()
plt.show()

# bar plot (averaged (=global)): taxon macro-average
shap.plots.bar(shap_values_tax,
               max_display=max_display+1,
               show=False)
plt.title('taxon macro-average')
plt.tight_layout()
plt.show()


# %%

# beeswarm option 1
shap.summary_plot(shap_values, 
                  max_display=max_display, 
                  cmap='cividis', 
                  alpha=0.4,
                  )

# %%

# beeswarm option 2
# (color map cannot be passed to function)
if 0:
    shap.plots.beeswarm(shap_values,
                        max_display=max_display+1, 
                        show=False,
                        )
    plt.tight_layout()
    plt.show()

# %%

# violin
# (color map cannot be passed to function)
if 0:
    shap.summary_plot(shap_values, 
                      max_display=max_display, 
                      alpha=0.4,
                      plot_type='violin'
                      )

# %%

# layered violin
# (color map cannot be passed to function)
if 0: 
    shap.summary_plot(shap_values, 
                      max_display=max_display, 
                      alpha=0.4,
                      plot_type='layered_violin'
                      )

# %%

# prepare plotting

# colors for models (from 47)

# https://www.pinterest.ch/pin/70439181665757203/
list_colors = ['#83920E', '#EFC201', '#E47900', '#B5134B', '#46093E']

# %%

# boxplot for most repeated experiments

# merge df_eco with predictions
df_eco_p = pd.merge(df_eco,
                    df_p[['result_id', 'true', 'lasso_pred', 'rf_pred', 'xgboost_pred', 'gp_pred']],
                    left_on=['result_id'],
                    right_on=['result_id'],
                    how='left')

# groupby chemical, species and all experimental conditions
list_cols_gb = ['test_cas', 'chem_name', 'tax_gs']
list_cols_gb += ['result_obs_duration_mean', 'result_conc1_type', 'test_exposure_type', 'test_media_type']
df_eco_p['count'] = df_eco_p.groupby(list_cols_gb)['result_id'].transform('count')
df_eco_p['conc_true_median'] = df_eco_p.groupby(list_cols_gb)['true'].transform('median')
df_eco_p['conc_lasso_pred_median'] = df_eco_p.groupby(list_cols_gb)['lasso_pred'].transform('median')
df_eco_p['conc_rf_pred_median'] = df_eco_p.groupby(list_cols_gb)['rf_pred'].transform('median')
df_eco_p['conc_xgboost_pred_median'] = df_eco_p.groupby(list_cols_gb)['xgboost_pred'].transform('median')
df_eco_p['conc_gp_pred_median'] = df_eco_p.groupby(list_cols_gb)['gp_pred'].transform('median')
df_eco_p['label'] = df_eco_p['chem_name'] + ' (' + df_eco_p['test_cas'] + ')' + ', ' + df_eco_p['tax_gs'] + '\n' + df_eco_p['result_obs_duration_mean'].astype('str') + ' hours, ' + df_eco_p['test_media_type'] + ', ' + df_eco_p['test_exposure_type'] + ', ' + df_eco_p['result_conc1_type'] + ', n=' + df_eco_p['count'].astype('str')
df_eco_p = df_eco_p.sort_values('conc_true_median')

df_plot = df_eco_p[df_eco_p['count'] >= 25].copy()
df_plot['label'] = pd.Categorical(df_plot['label'],
                                  categories=df_plot['label'].unique()[::-1],
                                  ordered=True)

# wide to long (predictions))
id_vars = ['result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs', 'label', 'true']
value_vars = ['gp_pred', 'lasso_pred', 'rf_pred', 'xgboost_pred']
df_plot_long = df_plot.melt(id_vars=id_vars, 
                         value_vars=value_vars,
                         value_name='pred',
                         var_name='type')

df_plot_long['type'] = df_plot_long['type'].str.replace('gp_pred', 'GP')
df_plot_long['type'] = df_plot_long['type'].str.replace('lasso_pred', 'LASSO')
df_plot_long['type'] = df_plot_long['type'].str.replace('rf_pred', 'RF')
df_plot_long['type'] = df_plot_long['type'].str.replace('xgboost_pred', 'XGBoost')

# %%

(ggplot(data=df_plot, mapping=aes(x='label', y='true'))
    + geom_boxplot(outlier_alpha=0)
    + geom_jitter(#mapping=aes(fill='test_media_type'), 
                  fill='grey',
                  alpha=0.8,
                  color='none', 
                  size=0.8, 
                  height=0, 
                  width=0.2)
    + geom_jitter(data=df_plot_long,
                 mapping=aes(y='pred', fill='type'), 
                 color='none',
                 size=0.8,
                 height=0, 
                 width=0.1)
    + scale_fill_manual(values=list_colors)
    + coord_flip()
    + theme_minimal()
    + labs(x='', y='log10(molar concentration)', fill='model')
    + theme(figure_size=(10, 8))
 )

# %%
# %%

# box plots for most tested chemicals

# merge df_eco with predictions
df_eco_p = pd.merge(df_eco,
                    df_p[['result_id', 'true', 'lasso_pred', 'rf_pred', 'xgboost_pred', 'gp_pred']],
                    left_on=['result_id'],
                    right_on=['result_id'],
                    how='left')

# groupby by chemical only
list_cols_gb = ['test_cas', 'chem_name']
df_eco_p['count'] = df_eco_p.groupby(list_cols_gb)['result_id'].transform('count')
df_eco_p['conc_true_median'] = df_eco_p.groupby(list_cols_gb)['true'].transform('median')
df_eco_p['conc_lasso_pred_median'] = df_eco_p.groupby(list_cols_gb)['lasso_pred'].transform('median')
df_eco_p['conc_rf_pred_median'] = df_eco_p.groupby(list_cols_gb)['rf_pred'].transform('median')
df_eco_p['conc_xgboost_pred_median'] = df_eco_p.groupby(list_cols_gb)['xgboost_pred'].transform('median')
df_eco_p['conc_gp_pred_median'] = df_eco_p.groupby(list_cols_gb)['gp_pred'].transform('median')
df_eco_p['label'] = df_eco_p['chem_name'] + ' (' + df_eco_p['test_cas'] + ') n=' + df_eco_p['count'].astype('str')
df_eco_p = df_eco_p.sort_values('conc_true_median')

df_plot = df_eco_p[df_eco_p['count'] >= 200].copy()
df_plot['label'] = pd.Categorical(df_plot['label'],
                                  categories=df_plot['label'].unique()[::-1],
                                  ordered=True)

# wide to long (predictions))
id_vars = ['result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs', 'label', 'true']
value_vars = ['gp_pred', 'lasso_pred', 'rf_pred', 'xgboost_pred']
df_plot_long = df_plot.melt(id_vars=id_vars, 
                         value_vars=value_vars,
                         value_name='pred',
                         var_name='type')

df_plot_long['type'] = df_plot_long['type'].str.replace('gp_pred', 'GP')
df_plot_long['type'] = df_plot_long['type'].str.replace('lasso_pred', 'LASSO')
df_plot_long['type'] = df_plot_long['type'].str.replace('rf_pred', 'RF')
df_plot_long['type'] = df_plot_long['type'].str.replace('xgboost_pred', 'XGBoost')

# %%

(ggplot(data=df_plot.dropna(subset=['conc_true_median']), mapping=aes(x='label', y='true'))
    + geom_boxplot(outlier_alpha=0)
    + geom_jitter(#mapping=aes(fill='test_media_type'), 
                  fill='grey',
                  alpha=0.8,
                  color='none', 
                  size=0.8, 
                  height=0, 
                  width=0.15)
    + geom_jitter(data=df_plot_long.dropna(subset=['pred']),
                  mapping=aes(y='pred', fill='type'), 
                  color='none',
                  size=0.8,
                  height=0,
                  width=0.05)
    + scale_fill_manual(values=list_colors)
    + coord_flip()
    + theme_minimal()
    + labs(x='', y='log10(molar concentration)', fill='model')
    + theme(figure_size=(10, 10))
 )

# %%
# %%

# function to summarize feature importance by variable type (tax, exp, chem)
def get_abs_mean_shap(shap_values, df_data_trainvalid, list_idx, chem_name):

    df_shap = pd.DataFrame(shap_values[list_idx].values,
                           columns=shap_values.feature_names,
                           index=df_data_trainvalid.loc[list_idx, 'result_id'])
    df_shap_mean = df_shap.abs().mean().reset_index()
    df_shap_mean = df_shap_mean.rename(columns={'index': 'feature', 0: 'shap_value'})
    df_shap_mean['chem_name'] = chem_name

    return df_shap_mean

def add_feature_category(df):

    df['feature_category'] = ''
    df.loc[df['feature'].str.startswith('chem'), 'feature_category'] = 'chemical'
    df.loc[df['feature'].str.contains(chem_fp), 'feature_category'] = 'molecular repr'
    df.loc[df['feature'].str.startswith('tax'), 'feature_category'] = 'taxonomic'
    df.loc[df['feature'].str.startswith('result'), 'feature_category'] = 'experimental'
    df.loc[df['feature'].str.startswith('test'), 'feature_category'] = 'experimental'

    return df

# colors
# https://www.pinterest.ch/pin/2251868550407753/
# TODO check colorblind safeness
list_colors = ['#e26449', '#f19a6e', '#628291', '#f5c100']  #, '#b8ce2d']

# %%

# get chemicals from last plot
list_chem_names = ['all'] + list(df_plot_long.dropna(subset=['pred'])['chem_name'].unique())

# initialize
list_dfs = []

# get mean absolute shap values
for chem_name in list_chem_names:
    
    # on trainvalidation data (update to test data if looking at test shap values)
    if chem_name == 'all':
        list_idx = list(df_data_trainvalid.index)
    else:
        list_idx = list(df_data_trainvalid[df_data_trainvalid['chem_name'] == chem_name].index)
    df_shap = get_abs_mean_shap(shap_values, df_data_trainvalid, list_idx, chem_name)
    list_dfs.append(df_shap)

# add feature category
df_shaps = pd.concat(list_dfs, axis=0)
df_shaps = add_feature_category(df_shaps)
#df_shaps

# %%

# plot feature importances across settings, e.g., all chemicals vs selected chemicals
df_plot = df_shaps.copy()
df_plot['feature_category'] = pd.Categorical(df_plot['feature_category'],
                                             categories=['chemical', 'molecular repr', 'experimental', 'taxonomic'][::-1],
                                             ordered=True)
df_plot['chem_name'] = pd.Categorical(df_plot['chem_name'],
                                      categories=list_chem_names,
                                      ordered=True)

(ggplot(data=df_plot, mapping=aes(x='chem_name', y='shap_value', fill='feature_category'))
    + geom_bar(stat='identity')
    + scale_fill_manual(list_colors[::-1])
    + theme_minimal()
    + labs(x='', y='mean absolute SHAP value', fill='category')
    + theme(axis_text_x=element_text(angle=90))
 )

# %%

# SHAP plots for selected chemicals
for chem_name in list_chem_names[:]:
    if chem_name == 'all':
        continue

    # on trainvalidation data (update to test data if looking at test shap values)
    list_idx = list(df_data_trainvalid[df_data_trainvalid['chem_name'] == chem_name].index)

    # bar chart
    shap.plots.bar(shap_values[list_idx],
                   max_display=max_display+1,
                   show=False)
    plt.title(chem_name)
    plt.tight_layout()
    plt.show()

    # beeswarm
    shap.summary_plot(shap_values[list_idx], 
                      max_display=max_display,
                      cmap='cividis', 
                      alpha=0.4,
                      show=False,
                      )
    plt.title(chem_name)
    plt.tight_layout()
    plt.show()

# %%


# single data points (local)
shap.plots.bar(shap_values[list_idx[0]], show=False)
plt.tight_layout()
plt.show()
shap.plots.waterfall(shap_values[list_idx[0]], show=False)
plt.tight_layout()
plt.show()

# a subset (not possible for waterfall plot)
#shap.plots.waterfall(shap_values[list_idx], show=False)
#plt.tight_layout()
#plt.show()

# do not run
#shap.plots.embedding(0, shap_values)
#shap.plots.force(shap_values)
# %%
