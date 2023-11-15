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

# XGBoost: Fish data, updated ADORE, 2023-09-15, top 3 features

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
    #'n_estimators': [50, 100],
    #'eta': [0.1, 0.2, 0.3],
    #'gamma': [0, 1, 10], 
    #'max_depth': [3, 6, 9, 12], 
    #'min_child_weight': [1, 3, 5],
    #'subsample': [0.5, 1.],
    #}
#]

path_output_dir = path_vmoutput + '2023-11-13_from-updated-adore_top3features/'
df_errors = utils.read_result_files(path_output_dir, file_type='error')

# %%

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

# mean errors (train and valid of 5-fold CV)
df_e_v = df_oi[(df_oi['fold'] == 'mean')].copy()

list_cols = ['chem_fp', 'groupsplit', 'conctype', 'set', 'fold']
list_cols += ['chem_prop', 'tax_pdm', 'tax_prop', 'exp']
list_cols += ['best_hp', 'idx_hp', 'n_estimators', 'eta', 'gamma', 'max_depth', 'min_child_weight', 'subsample']
list_cols += ['r2', 'rmse', 'mae', 'pearson']
df_e_v[df_e_v['best_hp'] == True][list_cols].round(2)

# %%

# store 
df_e_v[df_e_v['best_hp'] == True][list_cols].round(5).to_csv(path_output + 'xgboost_CV-errors_top3features.csv', index=False)

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
