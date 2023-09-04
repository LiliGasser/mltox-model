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

#%reload_ext autoreload
#%autoreload 2

# %%

# set paths
path_data = path_root + 'data/'
path_vmoutput = path_root + 'vm_output_gp/tmp/'

# %%

# parameter grids

# TODO select all
param_grid = [
    {
     # features
#     'chem_fp': ['MACCS'],
     'chem_fp': ['MACCS', 'pcp', 'Morgan', 'mol2vec'], 
     'chem_prop': ['chemprop'],                  #['none', 'chemprop'],
     'tax_pdm': ['pdm'],                         #['none', 'pdm', 'pdm-squared'],
     'tax_prop': ['taxprop-migrate2'],           #['none', 'taxprop-migrate2', 'taxprop-migrate5'],
     'exp': ['exp-dropfirst'],                   #['none', 'exp-dropnone', 'exp-dropfirst'],     # use dropfirst
     # splits
     'groupsplit': ['occurrence', 'totallyrandom'],  
     # concentration
     'conctype': ['molar', 'mass'] 
    }
]

# TODO select all
hyperparam_grid = [
    {
     # model hyperparameters     
     #'n_inducing': [100, 250],
     'n_inducing': [100, 250, 500, 1000],
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

# set taxonomic pdm columns
col_tax_pdm = 'tax_pdm_enc'

# which cols in predicted df to save
list_cols_preds = ['test_id', 'result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs']

# %%

for i, param in enumerate(ParameterGrid(param_grid)):

    print("run " + str(i + 1) + " of " + str(len(ParameterGrid(param_grid))))
    param_sorted = {k:param[k] for k in param_grid[0].keys()}
    print(param_sorted)
    print("-------------------------------")

    # get settings
    chem_fp = param['chem_fp']
    chem_prop = param['chem_prop']
    tax_pdm = param['tax_pdm']
    tax_prop = param['tax_prop']
    exp = param['exp']
    groupsplit = param['groupsplit']
    conctype = param['conctype']

    # skip non-sensible runs
    if do_ARD_fp and which_kernel_fp == 'Tanimoto':
        print("Warning: skipped for Tanimoto with ARD")
        print()
        continue
    
    if chem_fp  == 'mol2vec' and which_kernel_fp == 'Tanimoto':
        print("Warning: skipped for Tanimoto on mol2vec")
        print()
        continue

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
    list_cols_chem_prop = ['chem_mw', 'chem_mp', 'chem_ws', 
                           'chem_rdkit_clogp',
                           #'chem_pcp_heavy_atom_count',
                           #'chem_pcp_bonds_count', 'chem_pcp_doublebonds_count', 'chem_pcp_triplebonds_count',
                           #'chem_rings_count', 'chem_OH_count',
                           ]
    df_chem_prop_all = df_eco[list_cols_chem_prop].reset_index(drop=True)

    # encode experimental variables
    df_exp_all = utils.get_encoding_for_experimental_features(df_eco, exp)

    # encode taxonomic pairwise distances
    df_eco, df_pdm, df_enc = utils.get_encoding_for_taxonomic_pdm(df_eco, df_pdm, col_tax='tax_gs')

    # encode taxonomic Add my Pet features 
    df_tax_prop_all = utils.get_encoding_for_taxonomic_addmypet(df_eco)

    # print summary
    print("# entries:", df_eco.shape[0])
    print("# species:", df_eco['tax_all'].nunique())
    print("# chemicals:", df_eco['test_cas'].nunique())

    # get targets and groups
    df_label = df_eco[col_conc]

    # train-test-split
    col_split = '_'.join(('split', groupsplit))
    df_eco['split'] = df_eco[col_split]
    trainvalid_idx = df_eco[df_eco['split'] != 'test'].index
    test_idx = df_eco[df_eco['split'] == 'test'].index

    # initialize
    lol_cols_ARD = []

    # get experimental features
    df_exp, len_exp = utils.get_df_exp(df_exp_all)
    lol_cols_ARD = utils._update_lol_cols_ARD(lol_cols_ARD, exp, do_ARD_other, df_exp)
    
    # get chemical fingerprints
    df_chem_fp, len_chem_fp, lengthscales_fp = utils.get_df_chem_fp(chem_fp, 
                                                                    df_eco, 
                                                                    lengthscales, 
                                                                    trainvalid_idx, 
                                                                    test_idx)
    lol_cols_ARD = utils._update_lol_cols_ARD(lol_cols_ARD, chem_fp, do_ARD_fp, df_chem_fp)
        
    # get chemical properties
    df_chem_prop, len_chem_prop, lengthscales_prop = utils.get_df_chem_prop(chem_prop, 
                                                                            df_chem_prop_all, 
                                                                            lengthscales, 
                                                                            trainvalid_idx, 
                                                                            test_idx)
    lol_cols_ARD = utils._update_lol_cols_ARD(lol_cols_ARD, chem_prop, do_ARD_other, df_chem_prop)
    
    # get taxonomic pairwise distances
    df_tax_pdm, len_tax_pdm, squared = utils.get_df_tax_pdm(tax_pdm, df_eco, col_tax_pdm)
    if tax_pdm != 'none':
        lol_cols_ARD.append([])

    # get taxonomic properties
    df_tax_prop, len_tax_prop = utils.get_df_tax_prop(tax_prop, 
                                                      df_tax_prop_all,
                                                      trainvalid_idx, 
                                                      test_idx)
    lol_cols_ARD = utils._update_lol_cols_ARD(lol_cols_ARD, tax_prop, do_ARD_other, df_tax_prop)
   
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

    # initialize
    list_df_param_grid = []
    list_df_error_grid = []
    list_df_opt_grid = []
    list_df_preds_v_grid = []
    
    ## crossvalidation
    # get splits
    if not 'loo' in groupsplit:
        dict_splits = utils.get_precalculated_cv_splits(df_eco_trainvalid)
        n_splits_cv = df_eco_trainvalid['split'].astype('int').max() + 1
    else:
        dict_splits = {}
        dict_splits['loo'] = (trainvalid_idx, test_idx)
        n_splits_cv = 1

    # grid search over hyperparameter grid
    for idx_hp, hyperparam in enumerate(ParameterGrid(hyperparam_grid)):
        print(idx_hp, hyperparam)
        n_inducing = hyperparam['n_inducing']

        # initialize
        list_df_pred_v_grid = []
        list_df_pred_t_grid = []

        # run crossvalidation
        for fold, (train_idx, valid_idx) in dict_splits.items():
        
            print('fold:', fold)

            # apply train validation split
            if not 'loo' in groupsplit:
                X_train = df_trainvalid.loc[train_idx].to_numpy()
                X_valid = df_trainvalid.loc[valid_idx].to_numpy()
                df_eco_train = df_eco_trainvalid.loc[train_idx]
                df_eco_valid = df_eco_trainvalid.loc[valid_idx]
                y_train = y_trainvalid[train_idx]
                y_valid = y_trainvalid[valid_idx]
            else:
                X_train = df_trainvalid.to_numpy()
                X_valid = df_test.to_numpy()
                df_eco_train = df_eco_trainvalid.copy()
                df_eco_valid = df_eco_test.copy()
                y_train = y_trainvalid
                y_valid = y_test
            print("train and validation", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
            
            # GP regression
            try: 
        
                mean_function = gpflow.mean_functions.Constant(0)
                #mean_function = None
            
                kernel, len_tot =  utils.get_complete_kernel(len_exp, len_chem_fp, len_chem_prop, len_tax_pdm, len_tax_prop,
                                                             which_kernel_fp, which_kernel_other, 
                                                             variance, 
                                                             lengthscales, lengthscales_fp, lengthscales_prop, lengthscales_tax_pdm,
                                                             do_ARD_fp, do_ARD_other,
                                                             df_pdm, squared)

                if show_heatmaps:
                    # heatmap for kernel before training
                    df_tmp = pd.DataFrame(kernel(X_train).numpy()).round(4)
                    plt.figure(figsize = (12,8))
                    plt.title(' '.join(('before training')))
                    sns.heatmap(df_tmp)
                    plt.show()

                time_start = time.time()

                # run sparse GP
                opt_logs, model = utils.run_GP(X_train, y_train, 
                                               kernel, mean_function, noise_variance,
                                               maxiter,
                                               GP_type, ind_type, n_inducing,
                                               do_monitor)

                time_end = time.time()
                print("execution time:", (time_end-time_start)/60)
                df_opt = utils.get_df_opt(opt_logs)
                df_opt = utils._add_params_fold_to_df(df_opt, hyperparam, fold)
                list_df_opt_grid.append(df_opt)

                if opt_logs['message'] == 'ABNORMAL_TERMINATION_IN_LNSRCH':
                    print()
                    print()
                    continue
            
                if show_heatmaps:
                    # heatmap for kernel after training
                    df_tmp = pd.DataFrame(kernel(X_train).numpy()).round(4)
                    plt.figure(figsize = (12,8))
                    plt.title(' '.join(('after training')))
                    sns.heatmap(df_tmp)
                    plt.show()
        
                print_summary(model)
        
            except tf.errors.ResourceExhaustedError:

                print("OOM error")
                print(param_sorted)
                continue

            # predict for validation data
            y_train_pred, var_train_pred = model.predict_f(X_train)
            y_train_pred = y_train_pred.numpy()
            var_train_pred = var_train_pred.numpy()
            y_valid_pred, var_valid_pred = model.predict_f(X_valid)
            y_valid_pred = y_valid_pred.numpy()
            var_valid_pred = var_valid_pred.numpy()
        
            # generate output
            df_pred_t = df_eco_train.copy()
            df_pred_t['conc_pred'] = y_train_pred
            df_pred_t['conc_pred_var'] = var_train_pred
            df_pred_t = utils._add_params_fold_to_df(df_pred_t, 
                                                     hyperparam, 
                                                     fold)
            list_df_pred_t_grid.append(df_pred_t)
            df_pred_v = df_eco_valid.copy()
            df_pred_v['conc_pred'] = y_valid_pred
            df_pred_v['conc_pred_var'] = var_valid_pred
            df_pred_v = utils._add_params_fold_to_df(df_pred_v, 
                                                     hyperparam, 
                                                     fold)
            list_df_pred_v_grid.append(df_pred_v)
        
            # get parameters values from model
            list_rows_ind = None
            list_cols_ind = None
            if GP_type == 'sparse':
                list_rows_ind = ['inducing_point_' + str(i) for i in range(n_inducing)]
                list_cols_ind = list(df_features.columns)
            df_param_grid, df_ind = utils.get_paramvalues_from_module(model, 
                                                                     lol_cols_ARD=lol_cols_ARD,
                                                                     list_rows_ind=list_rows_ind,
                                                                     list_cols_ind=list_cols_ind)
            df_param_grid = utils._add_params_fold_to_df(df_param_grid, hyperparam, fold)
            list_df_param_grid.append(df_param_grid)

            print('length of validation grid', len(list_df_pred_v_grid))

        if len(list_df_pred_v_grid) > 0:

            # evaluate for crossvalidation
            col_true = col_conc
            col_pred = 'conc_pred'
            df_preds_t_grid = pd.concat(list_df_pred_t_grid)
            df_preds_v_grid = pd.concat(list_df_pred_v_grid)
            df_preds_v_grid['idx_hp'] = idx_hp
            list_df_preds_v_grid.append(df_preds_v_grid)
            df_error_grid = utils.calculate_evaluation_metrics(df_preds_t_grid, 
                                                               df_preds_v_grid,
                                                               col_true, 
                                                               col_pred, 
                                                               n_splits_cv)
            df_error_grid = utils._add_params_fold_to_df(df_error_grid, hyperparam)
            df_error_grid['idx_hp'] = idx_hp
            list_df_error_grid.append(df_error_grid)

    print('length of error grid', len(list_df_error_grid))

    if len(list_df_error_grid) > 0:

        # concatenate errors for hyperparameter grid
        df_errors_grid = pd.concat(list_df_error_grid).reset_index(drop=True)
        df_preds_v = pd.concat(list_df_preds_v_grid).reset_index(drop=True)

        # find best hyperparameters based on validation error
        df_e_v = df_errors_grid[(df_errors_grid['fold'] == 'mean') &
                                (df_errors_grid['set'] == 'valid')]
        if len(df_e_v) > 0:
            metric = 'rmse'
            df_e_v_best = df_e_v.loc[df_e_v[metric].idxmin()]
            idx_hp_best = df_e_v_best['idx_hp']
            df_errors_grid['best_hp'] = False
            df_errors_grid.loc[df_errors_grid['idx_hp'] == idx_hp_best, 'best_hp'] = True
        else:
            df_errors_grid['best_hp'] = np.nan
    
        # append / store
        df_errors_grid = utils._add_params_fold_to_df(df_errors_grid, param_sorted)
        str_file = '_'.join([str(i) for i in param_sorted.values()])
        df_errors_grid.round(5).to_csv(path_vmoutput + 'errors_' + str_file + '.csv', index=False)
    
        # store predictions for best hyperparameter
        df_preds_v_best = df_preds_v[df_preds_v['idx_hp'] == idx_hp_best].copy()
        list_cols_conc = ['fold', col_conc, 'conc_pred', 'conc_pred_var']
        df_store = df_preds_v_best[list_cols_preds + list_cols_conc].copy()
        df_store = utils._add_params_fold_to_df(df_store, param_sorted)
        df_store = utils._add_params_fold_to_df(df_store, hyperparam)
        df_store.round(5).to_csv(path_vmoutput + 'preds_' + str_file + '.csv', index=False)
        
        # concatenate and store parameters
        df_params_grid = pd.concat(list_df_param_grid)
        df_params_grid = df_params_grid.reset_index().rename(columns={'index': 'feature'})
        df_params_grid = utils._add_params_fold_to_df(df_params_grid, param_sorted)
        df_params_grid.round(5).to_csv(path_vmoutput + 'params_' + str_file + '.csv', index=False)

        # concatenate and store options
        df_opts_grid = pd.concat(list_df_opt_grid).reset_index(drop=True)
        df_opts_grid = utils._add_params_fold_to_df(df_opts_grid, param_sorted)
        df_opts_grid.round(5).to_csv(path_vmoutput + 'opts_' + str_file + '.csv', index=False)

    print()
    print()

print('done')

 # %%
