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

from plotnine import *

import utils

#%reload_ext autoreload
#%autoreload 2

# %%

# TODO evaluate permimp
#      - test vs tv
#      - then focus on test: best x features
# TODO evaluate SHAP
#      - using tv: best x features


# TODO slides --> evaluate fingerprints
#      - RMSE: molar occurrence --> focus on RF, XGBoost
#      - for (RF, XGBoost)
#           for all fingerprints:
#               permimp figure with 15 best features
#               SHAP overview
#               SHAP beeswarm


# TODO evaluate SHAP
#      - summarize for some chemicals
# TODO evaluate residuals and single entries 
#      --> 1 file for all models based on 21a

# set paths
path_output = path_root + 'output/regression/'
path_figures = path_output + 'figures/'
path_shap = path_output + 'shap/'
path_pi = path_output + 'permimp/'

# %%

# set
modeltype = 'rf'
chem_fp = 'MACCS'
groupsplit = 'occurrence'
conctype = 'molar'

title = ' '.join((modeltype, chem_fp))
max_display = 15

# %%

# load features
filename_ending = '_'.join((modeltype, 'data', chem_fp, groupsplit, conctype)) + '.csv'
filename_features = path_output + filename_ending
df_features = pd.read_csv(filename_features)

# load permutation importance results
filename_ending = '_'.join((modeltype, 'pi-trainvalid', chem_fp, groupsplit, conctype)) + '.p'
filename_pi_tv = path_pi + filename_ending
pi_result_tv = pickle.load(open(filename_pi_tv, 'rb'))
#print(pi_result_tv)
filename_ending = '_'.join((modeltype, 'pi-test', chem_fp, groupsplit, conctype)) + '.p'
filename_pi_test = path_pi + filename_ending
pi_result_test = pickle.load(open(filename_pi_test, 'rb'))
#print(pi_result_test)

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

# %%

df_pi_long['feature'] = pd.Categorical(df_pi_long['feature'],
                                       df_pi_long['feature'].unique(),
                                       ordered=True)
df_plot = df_pi_long[df_pi_long['set'] == 'test'].copy()
(ggplot(data=df_plot, mapping=aes(y='importance', x='feature'))
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


# %%

# Plotting remarks

# The default colormap should not be used as it is not colorblind safe.

# %%

# Plots for entire test set

# bar plot (averaged (=global))
shap.plots.bar(shap_values,
               max_display=max_display+1)

# %%

# beeswarm option 1
# ! passing color map not possible
shap.plots.beeswarm(shap_values,
                    max_display=max_display+1, 
                    )

# %%

# beeswarm 2
shap.summary_plot(shap_values, 
                  max_display=max_display, 
                  cmap='cividis', 
                  alpha=0.4,
                  )

# %%

# violin
# ! passing color map not possible
shap.summary_plot(shap_values, 
                  max_display=max_display, 
                  alpha=0.4,
                  plot_type='violin'
                  )

# %%

# layered violin
# ! passing color map not possible
shap.summary_plot(shap_values, 
                  max_display=max_display, 
                  alpha=0.4,
                  plot_type='layered_violin'
                  )

# %%

# %%

# single data points (local)
shap.plots.bar(shap_values[100])
shap.plots.waterfall(shap_values[100])

# a subset
shap.plots.waterfall(shap_values[:25])

# %%

# do not run
#shap.plots.embedding(0, shap_values)
#shap.plots.force(shap_values)
# %%

#chem_name = 'Malathion'
#chem_name = 'Endosulfan'
chem_name = 'Chlorpyrifos'
list_idx = list(df_features[df_features['chem_name'] == chem_name].index)
shap_values[list_idx]

# bar chart
shap.plots.bar(shap_values[list_idx],
               max_display=11)

# beeswarm
shap.summary_plot(shap_values[list_idx], 
                  max_display=10,
                  cmap='cividis', 
                  alpha=0.4,
                  )
# %%
