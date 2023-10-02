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

import matplotlib.pyplot as plt
from plotnine import *

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

# set
modeltype = 'xgboost'
chem_fp = 'MACCS'
groupsplit = 'occurrence'
conctype = 'molar'

title = ' '.join((modeltype, chem_fp))
max_display = 10

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

# Plotting remarks

# The default colormap should not be used as it is not colorblind safe.

# %%

# Plots for entire test set

# bar plot (averaged (=global))
shap.plots.bar(shap_values,
               max_display=max_display+1,
               show=False)
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

# Plots for selected chemicals

# TODO select 10 chemicals across toxicity range
chem_name = 'Carbaryl'
#chem_name = 'Malathion'
#chem_name = 'Endosulfan'
#chem_name = 'Dieldrin'

# on trainvalidation data (update to test data if looking at test shap values)
list_idx = list(df_data_trainvalid[df_data_trainvalid['chem_name'] == chem_name].index)
shap_values[list_idx]

# bar chart
shap.plots.bar(shap_values[list_idx],
               max_display=max_display+1,
               show=False)
plt.tight_layout()
plt.show()

# beeswarm
shap.summary_plot(shap_values[list_idx], 
                  max_display=max_display,
                  cmap='cividis', 
                  alpha=0.4,
                  )

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
list_colors = ['#e26449', '#f19a6e', '#f5c100', '#628291']  #, '#b8ce2d']

# %%

# select chemicals
# TODO select 10 chemicals across toxicity range (or repeated experiments??)
list_chem_names = ['all', 'Carbaryl', 'Dieldrin', 'Trichlorfon']

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
df_shaps

# %%

# plot feature importances across settings, e.g., all chemicals vs selected chemicals
df_plot = df_shaps.copy()
df_plot['feature_category'] = pd.Categorical(df_plot['feature_category'],
                                             categories=['chemical', 'molecular repr', 'experimental', 'taxonomic'][::-1],
                                             ordered=True)

(ggplot(data=df_plot, mapping=aes(x='chem_name', y='shap_value', fill='feature_category'))
    + geom_bar(stat='identity')
    + scale_fill_manual(list_colors[::-1])
    + theme_minimal()
    + labs(x='', y='mean absolute SHAP value', fill='category')
 )



# %%

# boxplot for most repeated experiments

#fish %>%
  #add_count(test_cas, chem_name, tax_gs, result_obs_duration_mean, result_conc1_type, test_exposure_type, test_media_type, tax_group, result_effect) %>%
  #filter(n >= 25) %>%
  #group_by(test_cas, chem_name, tax_gs, result_obs_duration_mean, result_conc1_type, test_exposure_type, test_media_type, tax_group, result_effect) %>%
  #mutate(median = median(result_conc1_mean)) %>%
  #ungroup() %>%
  #mutate(label = paste0(chem_name," (",test_cas,")", ", ", gsub("_", " ", tax_gs), " (",result_effect,", ",result_obs_duration_mean," h, N=",n,")")) %>%
  #ggplot(aes(x=fct_reorder(label, desc(median)), y=result_conc1_mean, fill = tax_group ))+
  #geom_boxplot(col = "#000000", outlier.alpha = 0)+
  #geom_jitter(alpha=0.2, size = 0.8)+
  #scale_y_log10()+
  #scale_fill_manual(values = tax_group_colors)+
  #coord_flip()+
  #guides(fill = "none")+
  #theme(legend.position = "bottom")+
  #theme_bw(base_size = 12)+
  #labs(y="EC50 [mg/L]", x = "Chemical name (CAS), Species, (effect type, duration, N)", fill = "Taxonomic group")


# groupby chemical, species and all experimental conditions
list_cols_gb = ['test_cas', 'chem_name', 'tax_gs']
list_cols_gb += ['result_obs_duration_mean', 'result_conc1_type', 'test_exposure_type', 'test_media_type']
df_eco['count'] = df_eco.groupby(list_cols_gb)['result_id'].transform('count')
df_eco['conc_median'] = df_eco.groupby(list_cols_gb)['result_conc1_mean_mol'].transform('median')
df_eco['label'] = df_eco['chem_name'] + ' (' + df_eco['test_cas'] + ')' + ', ' + df_eco['tax_gs'] + ', ' + df_eco['result_obs_duration_mean'].astype('str') + ' hours, n=' + df_eco['count'].astype('str')
df_eco = df_eco.sort_values('conc_median')

df_plot = df_eco[df_eco['count'] >= 25].copy()
df_plot['label'] = pd.Categorical(df_plot['label'],
                                  categories=df_plot['label'].unique()[::-1],
                                  ordered=True)


(ggplot(data=df_plot, mapping=aes(x='label', y='result_conc1_mean_mol'))
    + geom_boxplot(outlier_alpha=0)
    + geom_jitter(#mapping=aes(fill='test_media_type'), 
                  fill='grey',
                  alpha=0.8,
                  color='none', 
                  size=0.8, 
                  height=0, 
                  width=0.15)
    + scale_y_log10()
    + coord_flip()
    + theme_minimal()
    + theme(figure_size=(10, 8))
 )

# TODO load predictions
# TODO aggregate the same way and plot on top!!


# %%
# %%

# groupby by chemical only
list_cols_gb = ['test_cas', 'chem_name']
df_eco['count'] = df_eco.groupby(list_cols_gb)['result_id'].transform('count')
df_eco['conc_median'] = df_eco.groupby(list_cols_gb)['result_conc1_mean_mol'].transform('median')
df_eco['label'] = df_eco['chem_name'] + ' (' + df_eco['test_cas'] + ') n=' + df_eco['count'].astype('str')
df_eco = df_eco.sort_values('conc_median')

df_plot = df_eco[df_eco['count'] >= 100].copy()
df_plot['label'] = pd.Categorical(df_plot['label'],
                                  categories=df_plot['label'].unique()[::-1],
                                  ordered=True)


(ggplot(data=df_plot, mapping=aes(x='label', y='result_conc1_mean_mol'))
    + geom_boxplot(outlier_alpha=0)
    + geom_jitter(#mapping=aes(fill='test_media_type'), 
                  fill='grey',
                  alpha=0.8,
                  color='none', 
                  size=0.8, 
                  height=0, 
                  width=0.15)
    + scale_y_log10()
    + coord_flip()
    + theme_minimal()
    + theme(figure_size=(10, 12))
 )

# %%


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

# do not run
#shap.plots.embedding(0, shap_values)
#shap.plots.force(shap_values)
# %%
