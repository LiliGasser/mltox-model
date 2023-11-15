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

path_vmoutput = path_root + 'vm_output_gp/'
path_output = path_root + 'output/regression/'
path_figures = path_output + 'figures/'

# %%

# GP: Fish data, updated ADORE, 2023-09-15, top 3 features

# data pre-processing from ECOTOX 2022-09-15
# groupsplit: totallyrandom, occurrence

#param_grid = [
    #{
     ## features
     #'chem_fp': ['none'],
     #'chem_prop': ['chemprop'],
     #'tax_pdm': ['none'],  
     #'tax_prop': ['none'],    
     #'exp': ['none'],     
     ## splits
     #'groupsplit': ['totallyrandom', 'occurrence'], 
     ## concentration
     #'conctype': ['molar', 'mass'] 
    #}
#]

#hyperparam_grid = [
    #{
     ## model hyperparameters     
     #'n_inducing': [100, 250, 500, 1000],
    #}
#]

path_output_dir = path_vmoutput + '2023-11-13_from-updated-adore_top3features/'
df_errors = utils.read_result_files(path_output_dir, file_type='error')
df_params = utils.read_result_files(path_output_dir, file_type='param')

# %%

# categorical variable for folds
col = 'fold'
list_categories = ['mean', '0', '1', '2', '3', '4']
df_errors[col] = df_errors[col].astype('str')
df_params[col] = df_params[col].astype('str')
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)
df_params = utils._transform_to_categorical(df_params, col, list_categories[1:])

# categorical variable for groupsplit
col = 'groupsplit'
list_categories = ['totallyrandom', 'occurrence']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)
df_params = utils._transform_to_categorical(df_params, col, list_categories)

# categorical variable for conctype
col = 'conctype'
list_categories = ['molar', 'mass']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)
df_params = utils._transform_to_categorical(df_params, col, list_categories)

# sort
df_errors = df_errors.sort_values(['chem_fp', 'groupsplit'])
df_params = df_params.sort_values(['chem_fp', 'groupsplit'])

# check loading
if df_errors.isna().sum().sum() > 0:
    print("warning: check whether loading was correct")
else:
    print("loading seems correct")

# %%

# only look at results with best hyperparameters
df_oi = df_errors[df_errors['best_hp'] == True].copy()

# mean errors (train and valid of 5-fold CV)
df_e_v = df_oi[(df_oi['fold'] == 'mean')].copy()

list_cols = ['chem_fp', 'groupsplit', 'conctype', 'set', 'fold']
list_cols += ['chem_prop', 'tax_pdm', 'tax_prop', 'exp']
list_cols += ['best_hp', 'idx_hp', 'n_inducing', 'tax_pdm']
list_cols += ['r2', 'rmse', 'mae', 'pearson']
df_e_v[list_cols].round(2)

# %%

# store 
df_e_v[list_cols].round(5).to_csv(path_output + 'gp_CV-errors_top3features.csv', index=False)

# %%

# compare mass and molar concentration for none
metric = 'rmse'
tax_pdm = 'none'
df_plot = df_e_v[df_e_v['tax_pdm'] == tax_pdm].copy()
(ggplot(data=df_plot, mapping=aes(x='set', 
                                  y=metric, 
                                  fill='conctype'))
    + geom_col(position='dodge')
    + scale_fill_manual(values=['#7fc97f', '#beaed4'])
    + facet_grid("chem_fp ~ groupsplit") 
    + theme_minimal()
    + labs(y='RMSE', title='GP')
 )

# %%

# check hyperparemeter settings

# get data frame with best hyperparmeters only and with no pdm!
df_e_oi = df_errors[(df_errors['best_hp'] == True) &
                    (df_errors['set'] == 'valid') &
                    (df_errors['fold'] == 'mean') &
                    (df_errors['tax_pdm'] == 'none')]

# only trainvalidation results and with no pdm!
df_e_tv = df_errors[(df_errors['fold'] == 'mean')
                    & (df_errors['tax_pdm'] == 'none')
                    ]

# plot metric vs hyperparameter
col_x = 'n_inducing'
df_plot = df_e_tv.copy()
df_plot_oi = df_e_oi.copy()

df_plot['group'] = df_plot['set'] + '_' + df_plot['tax_pdm']
df_plot['conctype_groupsplit'] = df_plot['conctype'].astype('str') + ' ' + df_plot['groupsplit'].astype('str')
df_plot_oi['conctype_groupsplit'] = df_plot_oi['conctype'].astype('str') + ' ' + df_plot['groupsplit'].astype('str')
# %%
g = (ggplot(data=df_plot, mapping=aes(x=col_x, 
                                      y='rmse', 
                                      group='group', 
                                      fill='set', 
                                      color='set',
                                      shape='tax_pdm'))
    + geom_vline(data=df_plot_oi, mapping=aes(xintercept=col_x), color='grey', size=1.5, alpha=0.4)
    + geom_line()
    + geom_point(alpha=0.5, size=1)
#    + scale_x_log10()
#    + scale_y_continuous(limits=(-1, 1))
    + scale_fill_manual(values=['#d8b365', '#67a9cf'])
    + scale_color_manual(values=['#d8b365', '#67a9cf'])
    + facet_grid('conctype_groupsplit ~ chem_fp')
    + labs(y='RMSE')
    + theme_minimal()
    + theme(axis_text_x=element_text(angle=90))
    + theme(strip_text_y=element_text(angle=0))
)
#g.save(path_output_dir + '_GP_RMSE-vs-' + col_x + '.png', facecolor='white')
g

# %%
