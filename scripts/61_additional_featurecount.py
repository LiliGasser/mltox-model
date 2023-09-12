# Feature count per fingerprint and groupsplit combination

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

from plotnine import *

import utils
import modeling as mod

%reload_ext autoreload
%autoreload 2

# %%

# set paths
path_data = path_root + 'data/'
path_output = path_root + 'output/additional/'

# %%

### parameter grid

param_grid = [

    {
     # features
     'chem_fp': ['MACCS', 'pcp', 'Morgan', 'mol2vec'], 
     # splits
     'groupsplit': ['totallyrandom', 'occurrence'],
    }
]

lengthscales = -1  # initialize for some functions but it is not actually needed

# set taxonomic pdm columns
col_tax_pdm = 'tax_pdm_enc'

# set features
chem_prop = 'chemprop'
tax_pdm = 'none'
tax_prop = 'taxprop-migrate2'
exp = 'exp-dropfirst'

# concentration type (doesn't matter)
conctype = 'molar'

# %%

# initialize
list_outer = []


for i, param in enumerate(ParameterGrid(param_grid)):

    print("run " + str(i + 1) + " of " + str(len(ParameterGrid(param_grid))))
    param_sorted = {k:param[k] for k in param_grid[0].keys()}
    print(param_sorted)
    print("-------------------------------")

    # get settings
    chem_fp = param['chem_fp']
    groupsplit = param['groupsplit']

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
    df_exp_all = mod.get_encoding_for_experimental_features(df_eco, exp)

    # encode taxonomic pairwise distances
    df_eco, df_pdm, df_enc = mod.get_encoding_for_taxonomic_pdm(df_eco, df_pdm, col_tax='tax_gs')

    # encode taxonomic Add my Pet features 
    df_tax_prop_all = mod.get_encoding_for_taxonomic_addmypet(df_eco)

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

    # count features
    n_all = df_features.shape[1]
    list_features_chemfp = [i for i in df_features.columns if chem_fp in i]
    list_features_chemfp = [i.split('_')[1] if '_' in i else i for i in list_features_chemfp]
    list_features_chemfp = [i.replace(chem_fp, '') for i in list_features_chemfp]
    n_chemfp = len(list_features_chemfp)
    n_tax = len([i for i in df_features.columns if 'tax' in i])
    n_exp = len([i for i in df_features.columns if ('result' in i) or ('test' in i)])
    n_chemprop = len([i for i in df_features.columns if ('chem' in i) and (chem_fp not in i)])

    # feature names
    features_all = ','.join(df_features.columns)
    features_chemfp = ','.join(list_features_chemfp)

    # add to list
    list_inner = [chem_fp, groupsplit, n_all, n_exp, n_tax, n_chemprop, n_chemfp, features_all, features_chemfp]
    list_outer.append(list_inner)


df_all = pd.DataFrame(list_outer, 
                      columns=['chem_fp', 'groupsplit', 'n_all', 'n_exp', 'n_tax', 'n_chemprop', 'n_chemfp', 'features_all', 'features_chemfp'])
df_all.to_csv(path_output + 'featurecounts.csv', index=False)
df_all

# %%

list_cols = ['chem_fp', 'groupsplit', 'n_all', 'n_chemfp']
print(df_all[list_cols].to_latex(index=False))
# %%
