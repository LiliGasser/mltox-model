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

import matplotlib
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

# set challenge
challenge = 't-F2F'
#challenge = 't-C2C'
#challenge = 't-A2A'

# load data
df_eco = pd.read_csv(path_data + 'processed/' + challenge + '_mortality.csv', low_memory=False)

# %%

# load all predictions (from cross-validation)

# sparse GP
path_output_dir = path_vmoutput_gp + '2023-09-15_from-updated-adore/'
df_p_gp_all = utils.read_result_files(path_output_dir, file_type='preds')
df_p_gp_all['challenge'] = df_p_gp_all['challenge'].fillna('t-F2F')

# LASSO
path_output_dir = path_vmoutput_lasso + '2023-09-15_from-updated-adore/'
df_p_lasso_all = utils.read_result_files(path_output_dir, file_type='preds')
df_p_lasso_all['challenge'] = df_p_lasso_all['challenge'].fillna('t-F2F')

# RF
path_output_dir = path_vmoutput_rf + '2023-09-15_from-updated-adore/'
df_p_rf_all = utils.read_result_files(path_output_dir, file_type='preds')
df_p_rf_all['challenge'] = df_p_rf_all['challenge'].fillna('t-F2F')

# XGBoost
path_output_dir = path_vmoutput_xgboost + '2023-09-15_from-updated-adore/'
df_p_xgboost_all = utils.read_result_files(path_output_dir, file_type='preds')
df_p_xgboost_all['challenge'] = df_p_xgboost_all['challenge'].fillna('t-F2F')

# %%

# set
#modeltype = 'rf'
modeltype = 'xgboost'
chem_fp = 'MACCS'
groupsplit = 'occurrence'
conctype = 'molar'

title = ' '.join((challenge, modeltype, chem_fp))
title_medium = '_'.join((groupsplit, conctype, challenge, chem_fp))
title_long = '_'.join((groupsplit, conctype, challenge, modeltype, chem_fp))
max_display = 10

# %%

# get selected predictions
df_p = eval.filter_and_merge_predictions(df_p_gp_all, 
                                         df_p_lasso_all,
                                         df_p_rf_all,
                                         df_p_xgboost_all,
                                         challenge,
                                         groupsplit,
                                         conctype,
                                         tax_pdm='none')
df_p = df_p[df_p['chem_fp'] == chem_fp]

# %%

# TODO fix for challenges!! --> when rf t-F2F is rerun

# load features
filename_ending = '_'.join((modeltype, 'data', challenge, chem_fp, groupsplit, conctype)) + '.csv'
filename_features = path_features + filename_ending
df_data = pd.read_csv(filename_features)
list_cols = ['test_id', 'result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs']
df_features = df_data[[c for c in df_data.columns if c not in list_cols]]
# %%

# load permutation importance results
filename_ending = '_'.join((modeltype, 'permimp-trainvalid', challenge, chem_fp, groupsplit, conctype)) + '.p'
filename_pi_tv = path_pi + filename_ending
pi_result_tv = pickle.load(open(filename_pi_tv, 'rb'))
#print(pi_result_tv)
filename_ending = '_'.join((modeltype, 'permimp-test', challenge, chem_fp, groupsplit, conctype)) + '.p'
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
g = (ggplot(data=df_plot, mapping=aes(y='importance', x='feature2'))
    + geom_boxplot(outlier_alpha=0)
    + geom_jitter(color='#444', shape='.', width=0.1, height=0)
    + geom_vline(xintercept=0, linetype='--')
    + labs(y='decrease in accuracy score', x='', title='')
    + coord_flip()
    + theme_classic()
    + theme(panel_grid_major_y=element_line(color='#ccc', linetype='dotted'))
    + theme(axis_text=element_text(size=12, color='black'))
    + theme(axis_title=element_text(size=13, color='black'))
    + theme(figure_size=(8, 6))
 )
g.save(path_figures + '53-54_Permimp_' + title_long + '.pdf')
g

# %%

# %%

# load explainer and SHAP values
filename_ending = '_'.join((modeltype, 'explainer', challenge, chem_fp, groupsplit, conctype)) + '.sav'
filename_expl = path_shap + filename_ending
#explainer = pickle.load(open(filename_expl, 'rb'))
#print(explainer)

filename_ending = '_'.join((modeltype, 'shapvalues', challenge, chem_fp, groupsplit, conctype)) + '.sav'
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

# Change default SHAP colors based on this article 
# https://towardsdatascience.com/how-to-easily-customize-shap-plots-in-python-fdff9c0483f2
# Default SHAP colors
default_pos_color = "#ff0051"
default_neg_color = "#008bfb"

# Custom colors
positive_color = "#666"  #"#ca0020"
negative_color = "#92c5de"

# colors for feature categories
# https://www.pinterest.ch/pin/2251868550407753/
list_colors = ['#e26449', '#f19a6e', '#628291', '#b8ce2d'][::-1]  #, '#f5c100']
list_fcs = ['chemical', 'mol repr', 'taxonomic', 'experimental'][::-1]
dict_colors = dict(zip(list_fcs, list_colors))

def get_feature_category(col, chem_fp):
    ''' helper function to get feature category
    
    '''

    if col.startswith('chem'):
        fc = 'chemical'
    if chem_fp in col:
        fc = 'mol repr'
        #fc = 'molecular representation'
    if col.startswith('tax'):
        fc = 'taxonomic'
    if col.startswith('result'):
        fc = 'experimental'
    if col.startswith('test'):
        fc = 'experimental'

    return fc

# get ordered features and their cateogories
sv = np.array(shap_values.values)
sv_mean=np.abs(sv).mean(0)
order = np.argsort(sv_mean)[::-1]
list_ordered_cols = list(df_features.columns[order])
list_ordered_featcats =[get_feature_category(col, chem_fp) for col in list_ordered_cols] 


# %%

# Plots for entire test set

# TODO legend for feature group

# bar plot (averaged (=global)): micro-average
shap.plots.bar(shap_values,
               max_display=max_display+1,
               show=False)

# initialize
i = 0

# change the bar and text colors
for fc in plt.gcf().get_children():
    # Ignore last Rectangle
    for fcc in fc.get_children()[:-1]:
        if (isinstance(fcc, matplotlib.patches.Rectangle)):
            if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                #fcc.set_color(positive_color)
                if i < max_display:
                    fcc.set_color(dict_colors[list_ordered_featcats[i]])
                else:
                    fcc.set_color(positive_color)

            elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                fcc.set_color(negative_color)
            i = i + 1
        elif (isinstance(fcc, plt.Text)):
            if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color):
                fcc.set_color(positive_color)
            elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                fcc.set_color(negative_color)

#plt.title('micro-average')
plt.xlabel('mean absolute SHAP value')
plt.gcf().set_size_inches(8,6)
plt.tight_layout()
plt.savefig(path_figures + '53-54_SHAPglobal_' + title_long + '.pdf',
            bbox_inches='tight')
plt.show()

# %%

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
                  show=False,
                  )
plt.gcf().set_size_inches(8,6)
plt.tight_layout()
plt.savefig(path_figures + '53-54_SHAPlocal_' + title_long + '.pdf',
            bbox_inches='tight')
plt.show()

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
df_plot_long = utils._transform_to_categorical(df_plot_long, 'type', ['LASSO', 'RF', 'XGBoost', 'GP'])

# plot
(ggplot(data=df_plot, mapping=aes(x='label', y='true'))
    + geom_boxplot(outlier_alpha=0)
    + geom_jitter(#mapping=aes(color='test_media_type'), 
                  fill='grey',
                  #fill='none',
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
#    + scale_color_cmap('cividis_r')
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
df_plot_long = utils._transform_to_categorical(df_plot_long, 'type', ['LASSO', 'RF', 'XGBoost', 'GP'])

# plot
(ggplot(data=df_plot, mapping=aes(x='label', y='true'))
    + geom_boxplot(outlier_alpha=0)
    + geom_jitter(#mapping=aes(color='result_obs_duration_mean'), 
                  #fill='none',
                  fill='grey',
                  alpha=0.8,
                  #color='none', 
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
    + scale_color_cmap('cividis')  # TODO why is cividis_r not available anymore?
    + coord_flip()
    + theme_minimal()
    + labs(x='', y='log10(molar concentration)', fill='model')
    + theme(figure_size=(10, 8))
 )

# %%
# %%

# box plots for most tested species

# merge df_eco with predictions
df_eco_p = pd.merge(df_eco,
                    df_p[['result_id', 'true', 'lasso_pred', 'rf_pred', 'xgboost_pred', 'gp_pred']],
                    left_on=['result_id'],
                    right_on=['result_id'],
                    how='left')

# groupby by species only
list_cols_gb = ['tax_gs', 'tax_name']
df_eco_p['count'] = df_eco_p.groupby(list_cols_gb)['result_id'].transform('count')
df_eco_p['conc_true_median'] = df_eco_p.groupby(list_cols_gb)['true'].transform('median')
df_eco_p['conc_lasso_pred_median'] = df_eco_p.groupby(list_cols_gb)['lasso_pred'].transform('median')
df_eco_p['conc_rf_pred_median'] = df_eco_p.groupby(list_cols_gb)['rf_pred'].transform('median')
df_eco_p['conc_xgboost_pred_median'] = df_eco_p.groupby(list_cols_gb)['xgboost_pred'].transform('median')
df_eco_p['conc_gp_pred_median'] = df_eco_p.groupby(list_cols_gb)['gp_pred'].transform('median')
df_eco_p['label'] = df_eco_p['tax_name'] + ' (' + df_eco_p['tax_gs'] + ') n=' + df_eco_p['count'].astype('str')
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
df_plot_long = utils._transform_to_categorical(df_plot_long, 'type', ['LASSO', 'RF', 'XGBoost', 'GP'])

# plot
(ggplot(data=df_plot, mapping=aes(x='label', y='true'))
    + geom_boxplot(outlier_alpha=0)
    + geom_jitter(#mapping=aes(color='chem_rdkit_clogp'), 
                  fill='grey',
                  #fill='none',
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
#    + scale_color_cmap('cividis')
    + coord_flip()
    + theme_minimal()
    + labs(x='', y='log10(molar concentration)', fill='model')
    + theme(figure_size=(10, 8))
 )

# %%

# Species sensitivity distribution (SSD)

# merge df_eco with predictions
df_ssd = pd.merge(df_eco,
                  df_p[['result_id', 'true', 'lasso_pred', 'rf_pred', 'xgboost_pred', 'gp_pred']],
                  left_on=['result_id'],
                  right_on=['result_id'],
                  how='inner')   # with inner: only trainvalid predictions

# only chemicals which are tested on at least 15 species
df_ssd['n_species'] = df_ssd.groupby(['chem_name', 'test_cas'])['tax_gs'].transform('nunique')
df_ssd = df_ssd[df_ssd['n_species'] >= 15]
print(df_ssd.shape)

# set column for concentration
if conctype == 'molar':
    col_conc = 'result_conc1_mean_mol'
elif conctype == 'mass':
    col_conc = 'result_conc1_mean'

# get minimum and maximum concentration
conc_min = df_ssd[col_conc].min()
conc_max = df_ssd[col_conc].max()
conc_min_log10 = np.floor(np.log10(conc_min))
conc_max_log10 = np.ceil(np.log10(conc_max))

# only tests with lasted 96 hours and with an active ingredient
#df_ssd = df_ssd[(df_ssd['result_obs_duration_mean'] == 96) 
                #& (df_ssd['result_conc1_type'] == 'A')
                #].copy()

# from wide to long
id_vars = ['result_id', 'chem_name', 'test_cas', 'tax_gs', 'tax_name', 'n_species', col_conc]
value_vars = ['true', 'lasso_pred', 'rf_pred', 'xgboost_pred', 'gp_pred']
df_ssd_long = pd.melt(df_ssd, 
                      id_vars=id_vars, 
                      value_vars=value_vars,
                      var_name='model',
                      value_name='conc_log10')

# replace model names
df_ssd_long['model'] = df_ssd_long['model'].str.replace('lasso_pred', 'LASSO')
df_ssd_long['model'] = df_ssd_long['model'].str.replace('rf_pred', 'RF')
df_ssd_long['model'] = df_ssd_long['model'].str.replace('xgboost_pred', 'XGBoost')
df_ssd_long['model'] = df_ssd_long['model'].str.replace('gp_pred', 'GP')

# calculate backtransformation of predicted concentrations
df_ssd_long['conc'] = 10**df_ssd_long['conc_log10']

# calculate median and standard deviation
list_cols_gb = ['test_cas', 'chem_name', 'tax_gs', 'tax_name', 'model']
#list_cols_gb += ['result_obs_duration_mean', 'result_conc1_type', 'test_exposure_type', 'test_media_type']
df_ssd_gb = df_ssd_long.groupby(list_cols_gb).agg(n_tests=('result_id', 'count'),
                                                  conc_median=('conc', 'mean'),
                                                  conc_std=('conc', 'std'),
                                                  ).reset_index()
# fill NAs in standard deviation with 0
df_ssd_gb = df_ssd_gb.fillna(0)

# calculate log10 transformations
# TODO how to handle std that is larger than median?
df_ssd_gb['conc_median_log10'] = np.log10(df_ssd_gb['conc_median'])
df_ssd_gb['conc_median-std_log10'] = np.log10(df_ssd_gb['conc_median'] - df_ssd_gb['conc_std'])
df_ssd_gb['conc_median+std_log10'] = np.log10(df_ssd_gb['conc_median'] + df_ssd_gb['conc_std'])

# %%

# SSD with plotnine

list_cols_models = ['true', 'LASSO', 'RF', 'XGBoost', 'GP']
list_colors_models = ['black'] + list_colors[:4]

df_ssd_gb['chemical'] = df_ssd_gb['chem_name'] + ' (' + df_ssd_gb['test_cas'] + ')'
list_chemicals = list(df_ssd_gb['chemical'].unique())

for chemical in list_chemicals:
    #chemical = 'Potassium cyanide (151-50-8)'
    df_plot = df_ssd_gb[df_ssd_gb['chemical'] == chemical].copy()
    chem_name = chemical.split(' (')[0]

    # sort by true median concentration and calculate index fractions
    df_pt = df_plot[df_plot['model'] == 'true'].sort_values('conc_median_log10')
    df_pt = df_pt.reset_index(drop=True).reset_index()
    df_pt['index_frac'] = df_pt['index'] / df_pt['index'].max()

    # merge back with df_plot
    df_plot = pd.merge(df_plot,
                       df_pt[['test_cas', 'chem_name', 'tax_gs', 'index', 'index_frac']],
                       left_on=['test_cas', 'chem_name', 'tax_gs'],
                       right_on=['test_cas', 'chem_name', 'tax_gs'],
                       how='left')

    df_plot['model'] = pd.Categorical(df_plot['model'],
                                      categories=list_cols_models,
                                      ordered=True)

    g = (ggplot(data=df_plot, mapping=aes(x='index_frac', color='model'))
        + geom_segment(aes(xend='index_frac', y='conc_median-std_log10', yend='conc_median+std_log10'), 
                       size=0.5,
                       show_legend=False)
        + geom_point(aes(y='conc_median_log10')) 
        + scale_color_manual(values=list_colors_models)
        + scale_y_continuous(limits=(conc_min_log10, conc_max_log10), breaks=(-10, -8, -6, -4, -2, 0, 2))
        + coord_flip()
        + theme_classic()
        + theme(legend_position=(0.9, 0.3), legend_direction='vertical')
        + theme(plot_title=element_text(size=13, color='black'))
        + theme(axis_text=element_text(size=12, color='black'))
        + theme(axis_title=element_text(size=13, color='black'))
        + theme(legend_title=element_text(size=13, color='black'))
        + theme(legend_text=element_text(size=12, color='black'))
        + theme(figure_size=(8, 6))
        + labs(title=chemical,
                x='potentially affected fraction', 
                y='log10(LC50 in $mol/L$)')
     )
    if chem_name == 'Potassium cyanide':
        g = g + theme(legend_position=(0.2, 0.3), legend_direction='vertical')
    #g.save(path_figures + '53-54_SSD_' + title_medium + '_' + chem_name + '.pdf')
    print(g) 

# %%


# How to do the SSD?
# - 1) summarize for chemical and species
# - 2) for each chemical and species, provide summary for each experimental setting?
# - 3) for each experimental setting? but then there are fewer tests

# summary statistics are calculated from raw LC50, not log10-transformed(LC50)



# %%
# # %%

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
list_colors = ['#e26449', '#f19a6e', '#628291', '#f5c100']  #, '#b8ce2d']

# %%

# get chemicals from last plot (-> run lines 422ff. again)
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

# %%
