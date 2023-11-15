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

path_vmoutput = path_root + 'vm_output_lasso/'
path_output_add = path_root + 'output/additional/'
path_output = path_root + 'output/regression/'
path_figures = path_output + 'figures/'

# %%

# load feature counts
df_fc_orig = pd.read_csv(path_output_add + 'featurecounts.csv')

# %%

# LASSO: Fish data, updated ADORE, 2023-09-15, top3 features

# data pre-processing from ECOTOX 2022-09-15
# groupsplit: totallyrandom, occurrence
# alpha from 1 to 1e-5

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
     #'alpha': [np.round(i, 5) for i in np.logspace(-5, 0, num=26)],
    #}
#]

path_output_dir = path_vmoutput + '2023-11-13_from-updated-adore_top3features/'
df_errors = utils.read_result_files(path_output_dir, file_type='error')
df_params = utils.read_result_files(path_output_dir, file_type='param')
df_preds = utils.read_result_files(path_output_dir, file_type='preds')

# %%

# categorical variable for folds
col = 'fold'
list_categories = ['mean', '0', '1', '2', '3', '4']
df_errors[col] = df_errors[col].astype('str')
df_params[col] = df_params[col].astype('str')
df_preds[col] = df_preds[col].astype('str')
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)
df_params = utils._transform_to_categorical(df_params, col, list_categories[1:])
df_preds = utils._transform_to_categorical(df_preds, col, list_categories[1:])

# categorical variable for groupsplit
col = 'groupsplit'
list_categories = ['totallyrandom', 'occurrence']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)
df_params = utils._transform_to_categorical(df_params, col, list_categories)
df_preds = utils._transform_to_categorical(df_preds, col, list_categories)

# categorical variable for conctype
col = 'conctype'
list_categories = ['molar', 'mass']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)
df_params = utils._transform_to_categorical(df_params, col, list_categories)
df_preds = utils._transform_to_categorical(df_preds, col, list_categories)

# sort
df_errors = df_errors.sort_values(['chem_fp', 'groupsplit'])
df_params = df_params.sort_values(['chem_fp', 'groupsplit'])
df_preds = df_preds.sort_values(['chem_fp', 'groupsplit'])

# check loading
if df_errors.isna().sum().sum() > 0:
    print("warning: check whether loading was correct")
else:
    print("loading seems correct")

# %%

# only look at results with best hyperparameters
df_oi = df_errors[df_errors['best_hp'] == True].copy()

# mean errors (train and valid of 5-fold CV)
df_e_v = df_oi[(df_oi['fold'] == 'mean')]

list_cols = ['chem_fp', 'groupsplit', 'conctype', 'set', 'fold']
list_cols += ['chem_prop', 'tax_pdm', 'tax_prop', 'exp']
list_cols += ['best_hp', 'idx_hp', 'alpha']
list_cols += ['r2', 'rmse', 'mae', 'pearson']
df_e_v[list_cols].round(2)

# %%

# store 
df_e_v[list_cols].round(5).to_csv(path_output + 'lasso_CV-errors_top3features.csv', index=False)

# %%

# compare mass and molar concentration
metric = 'rmse'
(ggplot(data=df_e_v, mapping=aes(x='set', y=metric, fill='conctype'))
    + geom_col(position='dodge')
    + scale_fill_manual(values=['#7fc97f', '#beaed4'])
    + facet_grid("chem_fp ~ groupsplit") 
    + theme_minimal()
    + labs(y='RMSE', title='LASSO')
 )

# %%
