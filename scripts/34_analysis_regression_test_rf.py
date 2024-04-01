# Regression analysis with random forests

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

from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import pickle
from joblib import dump, load
import shap

from plotnine import *

import utils
import modeling as mod

#%reload_ext autoreload
#%autoreload 2

# %%

# load data
path_data = path_root + 'data/'
path_output = path_root + 'output/regression/'
path_figures = path_output + 'figures/'
path_shap = path_output + 'shap/'
path_pi = path_output + 'permimp/'
path_features = path_output + 'features/'

# %%

# load CV errors (includes best hyperparameters)
modeltype = 'rf'
df_cv = pd.read_csv(path_output + modeltype + '_CV-errors.csv')

# only one entry per challenge x chem_fp x groupsplit x conctype combination
df_cv = df_cv[df_cv['set'] == 'valid'].copy()

# %%

# load test output
df_e_test = pd.read_csv(path_output + modeltype + '_test-errors.csv')
df_pr_test = pd.read_csv(path_output + modeltype + '_predictions.csv')

# TODO remove after run
df_e_test['challenge'] = 't-F2F'
df_pr_test['challenge'] = 't-F2F'

# %%

# parameter grids

param_grid = [
    {
     # data
     'challenge': ['t-F2F', 't-C2C', 't-A2A'],
     # features
     'chem_fp': ['none', 'MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred'], 
     # splits
     'groupsplit': ['totallyrandom', 'occurrence'],
     # concentration
     'conctype': ['molar', 'mass'],
    }
]

# set for what to calculate feature importances
list_groupsplit_fi = ['occurrence']
list_conctype_fi =['molar']

lengthscales = -1  # initialize for some functions but it is not actually needed

# set taxonomic pdm column
col_tax_pdm = 'tax_pdm_enc'

# which cols in predicted df to save
list_cols_preds = ['test_id', 'result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs']

# %%

# initialize
list_df_errors = []
list_df_preds = []

for i, param in enumerate(ParameterGrid(param_grid)):

    print("run " + str(i + 1) + " of " + str(len(ParameterGrid(param_grid))))
    param_sorted = {k:param[k] for k in param_grid[0].keys()}
    print(param_sorted)
    print("-------------------------------")

    # get settings
    challenge = param['challenge']
    chem_fp = param['chem_fp']
    groupsplit = param['groupsplit']
    conctype = param['conctype']

    # check whether this test run is already done
    df_tmp = df_e_test[(df_e_test['challenge'] == challenge)
                       & (df_e_test['chem_fp'] == chem_fp)
                       & (df_e_test['conctype'] == conctype)
                       & (df_e_test['groupsplit'] == groupsplit)]
    if len(df_tmp) > 0:
        continue

    # get other parameters
    df_e_sel = df_cv[(df_cv['challenge'] == challenge)
                     & (df_cv['chem_fp'] == chem_fp)
                     & (df_cv['groupsplit'] == groupsplit)
                     & (df_cv['conctype'] == conctype)]
    chem_prop = df_e_sel['chem_prop'].iloc[0]
    tax_pdm = df_e_sel['tax_pdm'].iloc[0]
    tax_prop = df_e_sel['tax_prop'].iloc[0]
    exp = df_e_sel['exp'].iloc[0]

    # get hyperparameters as a dictionary
    hyperparam = {}
    list_cols_hp = ['max_depth', 'max_features', 'max_samples', 'min_samples_leaf', 'min_samples_split', 'n_estimators']
    for col in list_cols_hp:
        hyperparam[col] = df_e_sel[col].iloc[0]

    # concentration type
    if conctype == 'mass':
        col_conc = 'result_conc1_mean_log'
    elif conctype == 'molar':
        col_conc = 'result_conc1_mean_mol_log'

    # load data
    df_eco = pd.read_csv(path_data + 'processed/' + challenge + '_mortality.csv', low_memory=False)

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
    df_exp_all = mod.get_encoding_for_experimental_features(df_eco, exp)

    # encode taxonomic pairwise distances
    df_eco, df_pdm, df_enc = mod.get_encoding_for_taxonomic_pdm(df_eco, df_pdm, col_tax='tax_gs')

    # encode taxonomic Add my Pet features 
    if challenge not in ['s-A2A', 't-A2A']:
        df_tax_prop_all = mod.get_encoding_for_taxonomic_addmypet(df_eco)
    else:
        df_tax_prop_all = pd.DataFrame()
        tax_prop = 'none'

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
    
    # get experimental features
    df_exp, len_exp = mod.get_df_exp(df_exp_all)

    # get chemical fingerprints
    df_chem_fp, len_chem_fp, lengthscales_fp = mod.get_df_chem_fp(chem_fp, 
                                                                  df_eco, 
                                                                  lengthscales, 
                                                                  trainvalid_idx, 
                                                                  test_idx)

    # get chemical properties
    df_chem_prop, len_chem_prop, lengthscales_prop = mod.get_df_chem_prop(chem_prop, 
                                                                          df_chem_prop_all, 
                                                                          lengthscales, 
                                                                          trainvalid_idx, 
                                                                          test_idx)

    # get taxonomic pairwise distances
    df_tax_pdm, len_tax_pdm, squared = mod.get_df_tax_pdm(tax_pdm, df_eco, col_tax_pdm)

    # get taxonomic properties
    df_tax_prop, len_tax_prop = mod.get_df_tax_prop(tax_prop, 
                                                    df_tax_prop_all,
                                                    trainvalid_idx, 
                                                    test_idx)

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

    # random forest
    model = RandomForestRegressor(**hyperparam)
    model.fit(X_trainvalid, y_trainvalid.ravel())

    # predict
    y_tv_pred = model.predict(X_trainvalid)
    y_test_pred = model.predict(X_test)
        
    # generate output
    df_pred_tv = df_eco_trainvalid.copy()
    df_pred_tv['conc_pred'] = y_tv_pred
    df_pred_tv = mod._add_params_fold_to_df(df_pred_tv, 
                                            hyperparam, 
                                            'trainvalid')
    df_pred_test = df_eco_test.copy()
    df_pred_test['conc_pred'] = y_test_pred
    df_pred_test = mod._add_params_fold_to_df(df_pred_test, 
                                              hyperparam, 
                                              'test')

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
    df_error['tax_pdm'] = tax_pdm 
    df_error['tax_prop'] = tax_prop
    df_error['exp'] = exp 
    df_error = mod._add_params_fold_to_df(df_error, param)
    df_error = mod._add_params_fold_to_df(df_error, hyperparam)
    list_df_errors.append(df_error)

    # store predictions
    df_pred = pd.concat([df_pred_tv, df_pred_test])
    list_cols_conc = ['fold', col_conc, 'conc_pred']
    df_pred = df_pred[list_cols_preds + list_cols_conc].copy()
    df_pred = mod._add_params_fold_to_df(df_pred, param_sorted)
    df_pred = mod._add_params_fold_to_df(df_pred, hyperparam)
    list_df_preds.append(df_pred)

    if (conctype in list_conctype_fi) & (groupsplit in list_groupsplit_fi):
        print(conctype, groupsplit)

        # impurity based feature importances
        filename_ending = '_'.join((modeltype, 'featimp-impurity', chem_fp, groupsplit, conctype)) + '.csv'
        filename_fi_impurity = path_pi + filename_ending
        df_fi = pd.DataFrame([a.feature_importances_ for a in model.estimators_],
                                columns=df_features.columns)
        df_fi.to_csv(filename_fi_impurity, index=False)
            
        # permutation importance for trainvalidation data
        pi_result_tv = permutation_importance(
            model, 
            X_trainvalid, 
            y_trainvalid, 
            n_repeats=10, 
            random_state=123, 
            n_jobs=4
        )
        # save permutation importance results
        filename_ending = '_'.join((modeltype, 'permimp-trainvalid', chem_fp, groupsplit, conctype)) + '.p'
        filename_pi_tv = path_pi + filename_ending
        pickle.dump(pi_result_tv, open(filename_pi_tv, 'wb'))
        #pi_result_tv_loaded = pickle.load(open(filename_pi_tv, 'rb'))
        #print(pi_result_tv_loaded)

        # permutation importance for test data
        pi_result_test = permutation_importance(
            model, 
            X_test, 
            y_test, 
            n_repeats=10, 
            random_state=123, 
            n_jobs=4
        )
        filename_ending = '_'.join((modeltype, 'permimp-test', chem_fp, groupsplit, conctype)) + '.p'
        filename_pi_test = path_pi + filename_ending
        pickle.dump(pi_result_test, open(filename_pi_test, 'wb'))
        #pi_result_test_loaded = pickle.load(open(filename_pi_test, 'rb'))
        #print(pi_result_test_loaded)

        # calculate SHAP values and store explainer and values
        # Fits the explainer
        explainer = shap.Explainer(model.predict, df_trainvalid)
        # Calculates the SHAP values - It takes some time
        shap_values = explainer(df_trainvalid, max_evals=1500)
        # Save Explainer 
        filename_ending = '_'.join((modeltype, 'explainer', chem_fp, groupsplit, conctype)) + '.sav'
        filename_expl = path_shap + filename_ending
        pickle.dump(explainer, open(filename_expl, 'wb'))
        #load_explainer = pickle.load(open(filename_expl, 'rb'))
        #print(load_explainer)
        # Save shap values
        filename_ending = '_'.join((modeltype, 'shapvalues', chem_fp, groupsplit, conctype)) + '.sav'
        filename_sv = path_shap + filename_ending
        pickle.dump(shap_values, open(filename_sv, 'wb'))
        #load_shap_values = pickle.load(open(filename_sv, 'rb'))
        #print(load_shap_values)

        # store features data frame
        filename_ending = '_'.join((modeltype, 'data', chem_fp, groupsplit, conctype)) + '.csv'
        filename_data = path_features + filename_ending
        df_features2 = pd.concat((df_eco[list_cols_preds], df_features), axis=1)
        df_features2.to_csv(filename_data, index=False)

        # store model
        filename_ending = '_'.join((modeltype, 'model', chem_fp, groupsplit, conctype)) + '.joblib'
        filename_model = path_features + filename_ending
        dump(model, filename_model) 
        # model = load(filename_model) 

# concatenate and store
df_errors = pd.concat(list_df_errors)
df_errors = pd.concat((df_e_test, df_errors))
df_errors.round(5).to_csv(path_output + modeltype + '_test-errors.csv', index=False)

df_preds = pd.concat(list_df_preds)
df_preds = pd.concat((df_pr_test, df_preds))
df_preds.round(5).to_csv(path_output + modeltype + '_predictions.csv', index=False)

print('done')
# %%
