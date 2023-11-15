# Regression analysis with Gaussian Processes

# %%
# check tensorflow installation
import tensorflow as tf
tf.config.list_physical_devices('GPU')

# %%

import os
if os.getcwd().endswith('scripts'):
    path_root = '../'
else:
    path_root = './' 
import sys
sys.path.insert(0, path_root + 'src/')

import time

import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid

import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary

from plotnine import *
import matplotlib.pyplot as plt
import seaborn as sns

import utils
import modeling as mod

#%reload_ext autoreload
#%autoreload 2

# %%

# load data
path_data = path_root + 'data/'
path_output = path_root + 'output/regression/'
path_figures = path_output + 'figures/'

# %%

# load CV errors (includes best hyperparameters)
modeltype = 'gp'
df_cv = pd.read_csv(path_output + modeltype + '_CV-errors_top3features.csv')

# only one entry per chem_fp x groupsplit combination
df_cv = df_cv[df_cv['set'] == 'valid'].copy()

# %% 

# load test output
#df_e_test = pd.read_csv(path_output + modeltype + '_test-errors.csv')
#df_pa_test = pd.read_csv(path_output + modeltype + '_trainvalid-coefficients.csv')
#df_pr_test = pd.read_csv(path_output + modeltype + '_predictions.csv')

df_e_test = pd.DataFrame()
df_pa_test = pd.DataFrame()
df_pr_test = pd.DataFrame()

# %%

# parameter grids

param_grid = [
    {
     # features
     'chem_fp': ['none'], 
     # splits
     'groupsplit': ['totallyrandom', 'occurrence'], 
     # tax_pdm (only for GP!)
     'tax_pdm': ['none'],   # 'pdm'
     # concentration
     'conctype': ['molar', 'mass'] 
    }
]

show_heatmaps = False
do_monitor = False
maxiter = 10000    # TODO set to 10000

# fix some parameters
GP_type = 'sparse'                    # 'full'
ind_type = 'kmeans'                   # 'random'
which_kernel_fp = 'RBF'               # 'Linear', 'Tanimoto'
which_kernel_other = 'RBF'            # 'Linear'
do_ARD_fp = True
do_ARD_other = True
lengthscales = 3.
variance = 1.
noise_variance = 1.
lengthscales_tax_pdm = 200

# set taxonomic pdm column
col_tax_pdm = 'tax_pdm_enc'

# which cols in predicted df to save
list_cols_preds = ['test_id', 'result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs']

# %%

# initialize
list_df_errors = []
list_df_params = []
list_df_preds = []

for i, param in enumerate(ParameterGrid(param_grid)):

    print("run " + str(i + 1) + " of " + str(len(ParameterGrid(param_grid))))
    param_sorted = {k:param[k] for k in param_grid[0].keys()}
    print(param_sorted)
    print("-------------------------------")

    # get settings
    chem_fp = param['chem_fp']
    groupsplit = param['groupsplit']
    tax_pdm = param['tax_pdm']
    conctype = param['conctype']

    # check whether this test run is already done
    if len(df_e_test) > 0:
        df_tmp = df_e_test[(df_e_test['chem_fp'] == chem_fp)
                           & (df_e_test['conctype'] == conctype)
                           & (df_e_test['groupsplit'] == groupsplit)
                           & (df_e_test['tax_pdm'] == tax_pdm)]
        if len(df_tmp) > 0:
            continue

    # get other parameters
    df_e_sel = df_cv[(df_cv['chem_fp'] == chem_fp)
                     & (df_cv['groupsplit'] == groupsplit)
                     & (df_cv['tax_pdm'] == tax_pdm)
                     & (df_cv['conctype'] == conctype)]
    if df_e_sel.shape[0] == 0:
        continue
    chem_prop = df_e_sel['chem_prop'].iloc[0]
    tax_prop = df_e_sel['tax_prop'].iloc[0]
    exp = df_e_sel['exp'].iloc[0]

    # get hyperparameter
    n_inducing = df_e_sel['n_inducing'].iloc[0]
    hyperparam = {}
    hyperparam['n_inducing'] = n_inducing

    # concentration type
    if conctype == 'mass':
        col_conc = 'result_conc1_mean_log'
    elif conctype == 'molar':
        col_conc = 'result_conc1_mean_mol_log'

    # load fish data
    df_eco = pd.read_csv(path_data + 'processed/t-F2F_mortality.csv', low_memory=False)

    # load phylogenetic distance matrix
    tax_group = 'FCA'
    path_pdm = path_data + 'taxonomy/' + tax_group + '_pdm_species.csv'
    df_pdm = utils.load_phylogenetic_tree(path_pdm)

    # load chemical properties 
    # ! only use 3 most important features mw, ws, logp
    list_cols_chem_prop = ['chem_mw', 
                           #'chem_mp', 
                           'chem_ws', 
                           'chem_rdkit_clogp',
                           #'chem_pcp_heavy_atom_count',
                           #'chem_pcp_bonds_count', 'chem_pcp_doublebonds_count', 'chem_pcp_triplebonds_count',
                           #'chem_rings_count', 'chem_OH_count',
                           ]
    df_chem_prop_all = df_eco[list_cols_chem_prop].reset_index(drop=True)

    # encode experimental variables
    df_exp_all = mod.get_encoding_for_experimental_features(df_eco, exp)

    # encode taxonomic pairwise distances
    df_eco, df_pdm, df_enc = mod.get_encoding_for_taxonomic_pdm(df_eco, df_pdm, col_tax='tax_gs')

    # encode taxonomic Add my Pet features 
    df_tax_prop_all = mod.get_encoding_for_taxonomic_addmypet(df_eco)

    # print summary
    print("# entries:", df_eco.shape[0])
    print("# species:", df_eco['tax_all'].nunique())
    print("# chemicals:", df_eco['test_cas'].nunique())

    # get target
    df_label = df_eco[col_conc]
    
    # train-test-split
    col_split = '_'.join(('split', groupsplit))
    df_eco['split'] = df_eco[col_split]
    trainvalid_idx = df_eco[df_eco['split'] != 'test'].index
    test_idx = df_eco[df_eco['split'] == 'test'].index
    
    # initialize
    lol_cols_ARD = []

    # get experimental features
    df_exp, len_exp = mod.get_df_exp(df_exp_all)
    lol_cols_ARD = mod._update_lol_cols_ARD(lol_cols_ARD, exp, do_ARD_other, df_exp)

    # get chemical fingerprints
    df_chem_fp, len_chem_fp, lengthscales_fp = mod.get_df_chem_fp(chem_fp, 
                                                                  df_eco, 
                                                                  lengthscales, 
                                                                  trainvalid_idx, 
                                                                  test_idx)
    lol_cols_ARD = mod._update_lol_cols_ARD(lol_cols_ARD, chem_fp, do_ARD_fp, df_chem_fp)

    # get chemical properties
    df_chem_prop, len_chem_prop, lengthscales_prop = mod.get_df_chem_prop(chem_prop, 
                                                                          df_chem_prop_all, 
                                                                          lengthscales, 
                                                                          trainvalid_idx, 
                                                                          test_idx)
    lol_cols_ARD = mod._update_lol_cols_ARD(lol_cols_ARD, chem_prop, do_ARD_other, df_chem_prop)

    # get taxonomic pairwise distances
    df_tax_pdm, len_tax_pdm, squared = mod.get_df_tax_pdm(tax_pdm, df_eco, col_tax_pdm)
    if tax_pdm != 'none':
        lol_cols_ARD.append([])

    # get taxonomic properties
    df_tax_prop, len_tax_prop = mod.get_df_tax_prop(tax_prop, 
                                                    df_tax_prop_all,
                                                    trainvalid_idx, 
                                                    test_idx)
    lol_cols_ARD = mod._update_lol_cols_ARD(lol_cols_ARD, tax_prop, do_ARD_other, df_tax_prop)

    # concatenate features
    df_features = pd.concat((df_exp, df_chem_fp, df_chem_prop, df_tax_pdm, df_tax_prop), axis=1)
    if len(df_features) == 0:
        print('no features selected')
        continue

    # apply train test split
    df_trainvalid = df_features.iloc[trainvalid_idx, :].reset_index(drop=True)
    df_test = df_features.iloc[test_idx, :].reset_index(drop=True)
    df_eco_trainvalid = df_eco.iloc[trainvalid_idx, :].reset_index(drop=True)
    df_eco_test = df_eco.iloc[test_idx, :].reset_index(drop=True)
    X_trainvalid = df_trainvalid.to_numpy()
    X_test = df_test.to_numpy()
    y_trainvalid = np.array(df_label[trainvalid_idx]).reshape(-1, 1)
    y_test = np.array(df_label[test_idx]).reshape(-1, 1)

    # GP regression
    opt_logs_message = 'ABNORMAL_TERMINATION_IN_LNSRCH'
    count = 0
    try: 
        
        while opt_logs_message == 'ABNORMAL_TERMINATION_IN_LNSRCH' and count < 5:

            mean_function = gpflow.mean_functions.Constant(0)
            #mean_function = None
            
            kernel, len_tot =  mod.get_complete_kernel(len_exp, len_chem_fp, len_chem_prop, len_tax_pdm, len_tax_prop,
                                                       which_kernel_fp, which_kernel_other, 
                                                       variance, 
                                                       lengthscales, lengthscales_fp, lengthscales_prop, lengthscales_tax_pdm,
                                                       do_ARD_fp, do_ARD_other,
                                                       df_pdm, squared)

            if show_heatmaps:
                # heatmap for kernel before training
                df_tmp = pd.DataFrame(kernel(X_trainvalid).numpy()).round(4)
                plt.figure(figsize = (12,8))
                plt.title(' '.join(('before training')))
                sns.heatmap(df_tmp)
                plt.show()

            time_start = time.time()

            # run sparse GP
            opt_logs, model = mod.run_GP(X_trainvalid, 
                                         y_trainvalid, 
                                         kernel,
                                         mean_function, 
                                         noise_variance,
                                         maxiter,
                                         GP_type, 
                                         ind_type, 
                                         n_inducing)

            time_end = time.time()
            print("execution time:", (time_end-time_start)/60)
            #df_opt = mod.get_df_opt(opt_logs)
            #df_opt = mod._add_params_fold_to_df(df_opt, hyperparam, fold)
            #list_df_opt_grid.append(df_opt)

            opt_logs_message = opt_logs['message']
            count += 1
            if opt_logs['message'] == 'ABNORMAL_TERMINATION_IN_LNSRCH':
                print()
                print()
                continue
            
            if show_heatmaps:
                # heatmap for kernel after training
                df_tmp = pd.DataFrame(kernel(X_trainvalid).numpy()).round(4)
                plt.figure(figsize = (12,8))
                plt.title(' '.join(('after training')))
                sns.heatmap(df_tmp)
                plt.show()
        
            print_summary(model)
        
    except tf.errors.ResourceExhaustedError:

        print("OOM error")
        print(param_sorted)
        continue

    # predict
    y_tv_pred, var_tv_pred = model.predict_f(X_trainvalid)
    y_tv_pred = y_tv_pred.numpy()
    var_tv_pred = var_tv_pred.numpy()
    y_test_pred, var_test_pred = model.predict_f(X_test)
    y_test_pred = y_test_pred.numpy()
    var_test_pred = var_test_pred.numpy()

    # generate output
    df_pred_tv = df_eco_trainvalid.copy()
    df_pred_tv['conc_pred'] = y_tv_pred
    df_pred_tv['conc_pred_var'] = var_tv_pred
    df_pred_tv = mod._add_params_fold_to_df(df_pred_tv, 
                                            hyperparam, 
                                            'trainvalid')
    df_pred_test = df_eco_test.copy()
    df_pred_test['conc_pred'] = y_test_pred
    df_pred_test['conc_pred_var'] = var_test_pred
    df_pred_test = mod._add_params_fold_to_df(df_pred_test, 
                                              hyperparam, 
                                              'test')

    # get parameters values from model
    list_rows_ind = None
    list_cols_ind = None
    if GP_type == 'sparse':
        list_rows_ind = ['inducing_point_' + str(i) for i in range(n_inducing)]
        list_cols_ind = list(df_features.columns)
    df_param, df_ind = mod.get_paramvalues_from_module(model, 
                                                       lol_cols_ARD=lol_cols_ARD,
                                                       list_rows_ind=list_rows_ind,
                                                       list_cols_ind=list_cols_ind)
    df_param = df_param.reset_index().rename(columns={'index': 'feature'})
    df_param['set'] = 'trainvalid'
    df_param['chem_prop'] = chem_prop
    df_param['tax_prop'] = tax_prop
    df_param['exp'] = exp 
    df_param = mod._add_params_fold_to_df(df_param, param)
    df_param = mod._add_params_fold_to_df(df_param, 
                                          hyperparam, 
                                          'trainvalid')
    list_df_params.append(df_param)

    # evaluate
    col_true = col_conc
    col_pred = 'conc_pred'
    df_error = mod.calculate_evaluation_metrics(df_pred_tv, 
                                                  df_pred_test,
                                                  col_true, 
                                                  col_pred, 
                                                  -1)
    df_error['set'] = df_error['fold']
    df_error['chem_prop'] = chem_prop
    df_error['tax_prop'] = tax_prop
    df_error['exp'] = exp 
    df_error = mod._add_params_fold_to_df(df_error, param)
    df_error = mod._add_params_fold_to_df(df_error, hyperparam)
    list_df_errors.append(df_error)

    # store predictions
    df_pred = pd.concat([df_pred_tv, df_pred_test])
    list_cols_conc = ['fold', col_conc, 'conc_pred', 'conc_pred_var']
    df_pred = df_pred[list_cols_preds + list_cols_conc].copy()
    df_pred = mod._add_params_fold_to_df(df_pred, param_sorted)
    df_pred = mod._add_params_fold_to_df(df_pred, hyperparam)
    list_df_preds.append(df_pred)

# concatenate and store
df_errors = pd.concat(list_df_errors)
df_errors = pd.concat((df_e_test, df_errors))
df_errors.round(5).to_csv(path_output + modeltype + '_test-errors_top3features.csv', index=False)

df_params = pd.concat(list_df_params)
df_params = pd.concat((df_pa_test, df_params))
df_params.round(5).to_csv(path_output + modeltype + '_trainvalid-coefficients_top3features.csv', index=False)

df_preds = pd.concat(list_df_preds)
df_preds = pd.concat((df_pr_test, df_preds))
df_preds.round(5).to_csv(path_output + modeltype + '_predictions_top3features.csv', index=False)

print('done')
# %%
