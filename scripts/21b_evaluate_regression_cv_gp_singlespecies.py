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

# GP: Single species data, updated ADORE, 2023-11-24

# data pre-processing from ECOTOX 2022-09-15
# groupsplit: totallyrandom, occurrence
# tax_pdm: none and pdm

#param_grid = [
    #{
     ## data
     #'challenge': ['s-F2F-1', 's-F2F-2', 's-F2F-3', 's-C2C', 's-A2A'],
     ## features
     #'chem_fp': ['MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred', 'none'], 
     #'chem_prop': ['chemprop'],                 #['none', 'chemprop'],
     #'tax_pdm': ['none'],                       #['none', 'pdm', 'pdm-squared'],
     #'tax_prop': ['none'],                      #['none', 'taxprop-migrate2', 'taxprop-migrate5'],
     #'exp': ['exp-dropfirst'],                  #['none', 'exp-dropnone', 'exp-dropfirst'],     # use dropfirst
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

path_output_dir = path_vmoutput + '2023-11-24_from-updated-adore_singlespecies/'
df_errors = utils.read_result_files(path_output_dir, file_type='error')
df_params = utils.read_result_files(path_output_dir, file_type='param')

# %%

# categorical variables for fingerprints
col = 'challenge'
list_categories = ['s-F2F-1', 's-F2F-2', 's-F2F-3', 's-C2C'] #, 's-A2A']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)
df_params = utils._transform_to_categorical(df_params, col, list_categories)

# categorical variables for fingerprints
col = 'chem_fp'
list_categories = ['none', 'MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)
df_params = utils._transform_to_categorical(df_params, col, list_categories)
                                          
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
df_errors = df_errors.sort_values(['challenge', 'chem_fp', 'groupsplit'])
df_params = df_params.sort_values(['challenge', 'chem_fp', 'groupsplit'])

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

list_cols = ['challenge', 'chem_fp', 'groupsplit', 'conctype', 'set', 'fold']
list_cols += ['chem_prop', 'tax_pdm', 'tax_prop', 'exp']
list_cols += ['best_hp', 'idx_hp', 'n_inducing', 'tax_pdm']
list_cols += ['r2', 'rmse', 'mae', 'pearson']
df_e_v[list_cols].round(2)

# %%

# store 
df_e_v[list_cols].round(5).to_csv(path_output + 'gp_CV-errors_singlespecies.csv', index=False)

# %%

# TODO update for singlespecies
# compare mass and molar concentration for tax_pdm none
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

# compare tax_pdm: none vs pdm
metric = 'rmse'
df_plot = df_e_v[df_e_v['conctype'] == 'molar'].copy()
df_plot['group'] = df_plot['chem_fp'].astype('str') + '_' + df_plot['set']
(ggplot(data=df_plot, mapping=aes(x='tax_pdm', 
                                  y=metric, 
                                  group='group', 
                                  color='set'))
    + geom_point() 
    + geom_line()
    + scale_y_continuous(limits=(0, 1.2))
    + scale_color_manual(values=['#d8b365', '#67a9cf'])
    + facet_grid("chem_fp ~ groupsplit")
    + theme_minimal()
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

# TODO check what is still needed of the code below
# !! only look at tax_pdm != none?

# %%

# have a look at characteristic lengthscales

# get parameters for a run

# specify run
chem_fp = 'MACCS'
groupsplit = 'occurrence'
conctype = 'molar'
tax_pdm = 'none'

# get error and and corresponding number of inducing points
df_e_oi = df_e_v[(df_e_v['groupsplit'] == groupsplit)
                 & (df_e_v['conctype'] == conctype)
                 & (df_e_v['chem_fp'] == chem_fp)
                 & (df_e_v['tax_pdm'] == tax_pdm)]
n_inducing = df_e_oi['n_inducing'].iloc[0]

# get corresponding parameters
df_p_oi = df_params[(df_params['groupsplit'] == groupsplit)
                    & (df_params['conctype'] == conctype) 
                    & (df_params['chem_fp'] == chem_fp) 
                    & (df_params['tax_pdm'] == tax_pdm)
                    & (df_params['n_inducing'] == n_inducing)]
df_p_oi

# %%

# look at variances

# get list of kernels
exp = df_p_oi['exp'].unique()[0]
chem_prop = df_p_oi['chem_prop'].unique()[0]
tax_prop = df_p_oi['tax_prop'].unique()[0]
list_kernels = [exp, chem_fp, chem_prop, tax_pdm, tax_prop]
list_kernels = [k + '.variance' for k in list_kernels if k != 'none']

# get variance parameters only and replace their names with meaningful kernel names
df_p_oi_var = df_p_oi[df_p_oi['feature'].str.contains('variance')]
list_kernels_raw = ['.kernel.kernels[' + str(i) + '].variance' for i in range(len(list_kernels))]
dict_replace = dict(zip(list_kernels_raw, list_kernels))
df_p_oi_var = df_p_oi_var.replace({'feature': dict_replace})

# from long to wide
df_plot = df_p_oi_var.copy()
col_value = 'value'
df_plot_wide = df_plot.pivot(index=['feature'], columns=['fold'], values=[col_value])
df_plot_wide.columns = [item[1] for item in df_plot_wide.columns]
list_variances = ['.likelihood.variance'] + list_kernels
df_plot_wide.index = pd.CategoricalIndex(df_plot_wide.index, 
                                         categories=list_variances, 
                                         ordered=True)
df_plot_wide.sort_index(inplace=True)

# relative to noise level
df_plot_wide_rel = df_plot_wide.div(df_plot_wide.loc['.likelihood.variance'])
# %%

# heatmap of variance parameters
sns.heatmap(df_plot_wide_rel, 
            linewidth=0.5, 
            cmap='PiYG', 
            center=0,
            square=True,
            annot=True,
            fmt='.1f')
plt.title(chem_fp)
plt.xlabel('fold')
plt.ylabel('')
plt.show()

# %%

# look at lengthscales

# keep lengthscales only
df_p_oi_l = df_p_oi[(df_p_oi['feature'].str.contains('lengthscales'))].copy()

# take inverse
df_p_oi_l['1/value'] = 1/df_p_oi_l['value']

# generate short feature name (only last part that contains actual feature name)
col_feature = 'feature'
df_p_oi_l['feature_short'] = df_p_oi_l[col_feature].apply(lambda x: x.split('.')[-1] if len(x.split('.')) > 4 else x)

# sort by 'trainvalid'
# TODO sort by mean of 5 folds
df_oi = df_p_oi_l[df_p_oi_l['fold'] == 'trainvalid']
df_oi = df_oi.sort_values(['1/value'], ascending=False)
list_features_sorted = list(df_oi['feature_short'])
df_p_oi_l['feature_short'] = pd.Categorical(df_p_oi_l['feature_short'],
                                            categories=list_features_sorted,
                                            ordered=True)
df_p_oi_l = df_p_oi_l.sort_values(['fold', 'feature_short'])

# only values above a threshold
# TODO continue here: some threshold!
#df_p_oi_l = df_p_oi_l[df_p_oi_l['value'] >= 15].copy()

# generate column for facet wrap
list_range = list(range(df_p_oi_l['feature_short'].nunique()))
list_wrap = [int(i/50) for i in list_range]
df_p_oi_l['range'] = [str(i).zfill(3) for i in list_range] * df_p_oi_l['fold'].nunique()
df_p_oi_l['wrap'] = list_wrap * df_p_oi_l['fold'].nunique()
df_p_oi_l = df_p_oi_l.reset_index(drop=True)
df_p_oi_l

# %%

# TODO for all fingerprints and groupsplits

# lengthscales plot
n_wraps = df_p_oi_l['wrap'].max() + 1
fig, axes = plt.subplots(nrows=1, ncols=n_wraps, figsize=(n_wraps*3.5, 10))
fig.tight_layout(w_pad=6)

col_value = '1/value'
vmin = 0
#vmin = df_p_oi_l[col_value].min()
vmax = df_p_oi_l[col_value].max()

df_range = df_p_oi_l[['range', 'feature_short']].drop_duplicates()

for i in df_p_oi_l['wrap'].unique():
    # select wrap
    df_plot = df_p_oi_l[df_p_oi_l['wrap'] == i]

    # from long to wide
    df_plot_wide = df_plot.pivot(index=['range'], columns=['fold'], values=[col_value])
    df_plot_wide.columns = [item[1] for item in df_plot_wide.columns]

    # rename index
    df_tmp = pd.merge(df_plot_wide,
                      df_range,
                      left_on=['range'],
                      right_on=['range'],
                      how='left')
    df_tmp = df_tmp.drop('range', axis=1).set_index('feature_short')

    # only show color bar for last wrap
    if i == n_wraps - 1:
        cbar = True
    else:
        cbar = False

    # heatmap
    sns.heatmap(df_tmp, 
                linewidth=0.5, 
                cmap='PiYG', 
                vmin=vmin,
                vmax=vmax,
                center=0,
                square=True,
                ax=axes[i], 
                cbar=cbar)
    axes[i].set_ylabel('')
    if i == 0:
        axes[i].set_title(chem_fp)
    else:
        axes[i].set_title('')

filepath = path_figures + '21_GP_featureimportance_' + chem_fp + '_' + groupsplit + '_' + conctype + '.png'
#fig.savefig(filepath, facecolor='white')
plt.show()

# %%
