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
# TODO how for other t-X2X?
df_fc_orig = pd.read_csv(path_output_add + 'featurecounts.csv')

# %%

# LASSO: Fish data, updated ADORE, 2023-09-15
# in March 2024: Crustaceans and algae

# data pre-processing from ECOTOX 2022-09-15
# groupsplit: totallyrandom, occurrence
# alpha from 1 to 1e-5

#param_grid = [

    #{
     ## data
     #'challenge': ['t-F2F', 't-C2C', 't-A2A'],
     ## features
     #'chem_fp': ['MACCS', 'pcp', 'Morgan', 'mol2vec', 'ToxPrint'], 
     #'chem_prop': ['chemprop'],                 #['none', 'chemprop'],
     #'tax_pdm': ['none'],                       #['none', 'pdm', 'pdm-squared'],
     #'tax_prop': ['taxprop-migrate2'],          #['none', 'taxprop-migrate2', 'taxprop-migrate5'],
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
     #'alpha': [np.round(i, 5) for i in np.logspace(-5, 0, num=26)],
    #}
#]

path_output_dir = path_vmoutput + '2023-09-15_from-updated-adore/'
df_errors = utils.read_result_files(path_output_dir, file_type='error')
df_params = utils.read_result_files(path_output_dir, file_type='param')
df_preds = utils.read_result_files(path_output_dir, file_type='preds')

# %%

# update challenge entry for t-F2F
df_errors['challenge'] = df_errors['challenge'].fillna('t-F2F')
df_params['challenge'] = df_params['challenge'].fillna('t-F2F')
df_preds['challenge'] = df_preds['challenge'].fillna('t-F2F')

# %%

# categorical variables for challenge
col = 'challenge'
list_categories = ['t-F2F', 't-C2C', 't-A2A']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)
df_params = utils._transform_to_categorical(df_params, col, list_categories)
df_preds = utils._transform_to_categorical(df_preds, col, list_categories)

# categorical variables for fingerprints
col = 'chem_fp'
list_categories = ['none', 'MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred']
df_errors = utils._transform_to_categorical(df_errors, col, list_categories)
df_params = utils._transform_to_categorical(df_params, col, list_categories)
df_preds = utils._transform_to_categorical(df_preds, col, list_categories)

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

list_cols = ['challenge', 'chem_fp', 'groupsplit', 'conctype', 'set', 'fold']
list_cols += ['chem_prop', 'tax_pdm', 'tax_prop', 'exp']
list_cols += ['best_hp', 'idx_hp', 'alpha']
list_cols += ['r2', 'rmse', 'mae', 'pearson']
df_e_v[list_cols].round(2)

# %%

# store 
df_e_v[list_cols].round(5).to_csv(path_output + 'lasso_CV-errors.csv', index=False)

# %%

# compare mass and molar concentration
challenge = 't-F2F'
df_plot = df_e_v[df_e_v['challenge'] == challenge].copy()
metric = 'rmse'
(ggplot(data=df_plot, mapping=aes(x='set', y=metric, fill='conctype'))
    + geom_col(position='dodge')
    + scale_fill_manual(values=['#7fc97f', '#beaed4'])
    + facet_grid("chem_fp ~ groupsplit") 
    + theme_minimal()
    + labs(y='RMSE', title='LASSO')
 )

# %%

# calculate the number of selected features
 
# for training (cross-validation)
list_cols = ['challenge', 'chem_fp', 'groupsplit', 'conctype', 'fold', 'alpha']
df_fc = df_params[df_params['feature'] != 'intercept'].groupby(list_cols)['feature'].count().reset_index()
df_fc = df_fc.rename(columns={'feature': 'count'})
df_fc = df_fc[df_fc['count'] > 0].copy()
df_fc

# %%

# training: the feature counts behave similarly for all combinations of fps and splits
# TODO how for each challenge?
df_plot = df_fc.copy()
(ggplot(data=df_plot, mapping=aes(x='alpha',
                                 y='count',
                                 group='fold',
                                 color='fold',
                                 shape='conctype'))
    + geom_point()
    + scale_x_log10()
    + facet_grid("groupsplit ~ chem_fp")
    + theme_minimal()
)

# %%

# TODO how for the different challenges?
# average the feature counts of the 5 cv folds
list_cols = ['chem_fp', 'groupsplit', 'conctype', 'alpha']
df_fc_mean = df_fc[df_fc['fold'] != 'trainvalid'].groupby(list_cols)['count'].mean().reset_index()

# feature counts of trainvalidation run
# TODO move to 42 script
df_fc_tv = df_fc[df_fc['fold'] == 'trainvalid'].copy()
df_fc_tv = df_fc_tv.rename(columns={'count': 'count_tv'})

# merge the average feature counts with the errors
df_e_tv = df_errors[(df_errors['fold'] == 'mean')]
df_e_mean = pd.merge(df_e_tv,
                     df_fc_mean,
                     left_on=list_cols,
                     right_on=list_cols,
                     how='left')

# get data frame with best hyperparmeters only
df_e_oi = df_e_mean[(df_e_mean['best_hp'] == True) &
                    (df_e_mean['set'] == 'valid')]

# merge with trainvalidation feature count
df_e_oi = pd.merge(df_e_oi,
                   df_fc_tv.drop('alpha', axis=1),
                   left_on=['chem_fp', 'groupsplit', 'conctype'],
                   right_on=['chem_fp', 'groupsplit', 'conctype'],
                   how='left')

# merge with feature counts
list_cols = ['chem_fp', 'groupsplit']
list_cols_fc = list_cols + ['n_all']
df_e_oi = pd.merge(df_e_oi,
                   df_fc_orig[list_cols_fc],
                   left_on=list_cols,
                   right_on=list_cols,
                   how='left')

# calculate percentage
df_e_oi['perc'] = df_e_oi['count'] / df_e_oi['n_all']
df_e_oi

# %%

# number of features for best alpha
df_plot = df_e_oi.copy()
df_plot['chem_fp'] = pd.Categorical(df_plot['chem_fp'],
                                    categories=['none', 'MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred'],
                                    ordered=True)
df_plot['groupsplit'] = pd.Categorical(df_plot['groupsplit'],
                                       categories=['totallyrandom', 'occurrence'],
                                       ordered=True)

g = (ggplot(data=df_plot, mapping=aes(x='conctype', y='count'))
    + geom_col()
    + facet_grid('groupsplit ~ chem_fp')
    + theme_minimal()
    + theme(axis_text_x=element_text(angle=90))
    + labs(y='number of features')
)
#g.save(path_figures + '22_LASSO_featurecounts.png', facecolor='white')
g

# %%

# percentage of features for best alpha

g = (ggplot(data=df_plot, mapping=aes(x='conctype', y='perc'))
    + geom_col()
    + facet_grid('groupsplit ~ chem_fp')
    + theme_minimal()
    + theme(axis_text_x=element_text(angle=90))
    + labs(y='percentage of features')
)
#g.save(path_figures + '22_LASSO_featurepercentages.png', facecolor='white')
g

# %%

# check hyperparemeter settings

# plot R2 vs alpha

# select column
col_x = 'alpha'
#col_x = 'count'

# select metric
metric = 'rmse'
#metric = 'r2'

# dataframes
df_plot = df_e_mean.copy()
df_plot_oi = df_e_oi.copy()
df_plot['conctype_groupsplit'] = df_plot['conctype'].astype('str') + ' ' + df_plot['groupsplit'].astype('str')
df_plot_oi['conctype_groupsplit'] = df_plot_oi['conctype'].astype('str') + ' ' + df_plot_oi['groupsplit'].astype('str')

# y axis range
ymin = 0 if metric == 'rmse' else -1
ymax = df_plot[metric].max() + 0.1 if metric == 'rmse' else 1

g = (ggplot(data=df_plot, mapping=aes(x=col_x, 
                                      y=metric, 
                                      group='set', 
                                      fill='set', 
                                      color='set'))
    + geom_vline(data=df_plot_oi, 
                 mapping=aes(xintercept=col_x), 
                 color='grey', 
                 size=1.5, 
                 alpha=0.4)
    + geom_line()
    + geom_point(alpha=0.5, size=0.8)
    + scale_x_log10()
    + scale_fill_manual(values=['#d8b365', '#67a9cf'])
    + scale_color_manual(values=['#d8b365', '#67a9cf'])
    + scale_y_continuous(limits=(ymin, ymax))
    + facet_grid('conctype_groupsplit ~ chem_fp')
    + theme_minimal()
    + theme(strip_text_y=element_text(angle=0))
)
if metric == 'r2':
    g = g + labs(y="R$^2$")
elif metric == 'rmse':
    g = g + labs(y="RMSE")
#g.save(path_figures + '22_LASSO_' + metric + '-vs-' + col_x + '.png', facecolor='white')
g


# %%

# look at errors per fold
metric = 'rmse'

df_oi = df_errors[(df_errors['best_hp'] == True)].copy()

df_plot = df_oi.copy()
df_plot['conctype_groupsplit'] = df_plot['conctype'].astype('str') + ' ' + df_plot['groupsplit'].astype('str')

g = (ggplot(data=df_plot, mapping=aes(x='fold', 
                                  y=metric, 
                                  fill='set', 
                                  color='set',
                                  group='set'))
    + geom_line()
    + geom_point()
#    + geom_col(position='position_dodge')
    + facet_grid("conctype_groupsplit ~ chem_fp")
    + scale_fill_manual(values=['#d8b365', '#67a9cf'])
    + scale_color_manual(values=['#d8b365', '#67a9cf'])
    + scale_y_continuous(limits=(0, 2.5))
    + theme_minimal()
    + theme(axis_text_x=element_text(angle=90))
    + theme(strip_text_y=element_text(angle=0))
)
if metric == 'r2':
    g = g + labs(y="R$^2$")
elif metric == 'rmse':
    g = g + labs(y="RMSE")
#g.save(path_figures + '22_LASSO_' + metric + '-vs-fold.png', facecolor='white')
g

# %%

# feature importance plots

# only cv folds

# only look at validation folds
df_e_oi = df_errors[(df_errors['best_hp'] == True)
                  & (df_errors['set'] == 'valid')
                  & (df_errors['fold'] != 'mean')].copy()

# merge with params
list_cols = ['chem_fp', 'groupsplit', 'conctype', 'fold', 'alpha']
df_p_oi = pd.merge(df_e_oi,
                   df_params[list_cols + ['feature', 'value']],
                   left_on=list_cols,
                   right_on=list_cols,
                   how='left')

# %%

conctype = 'molar'
for chem_fp in ['none', 'MACCS', 'pcp', 'Morgan', 'ToxPrint', 'mol2vec']:
    for groupsplit in ['totallyrandom', 'occurrence']:
        print(chem_fp, groupsplit)

        # get parameters (for best alpha only)
        df_p = df_p_oi[(df_p_oi['chem_fp'] == chem_fp) &
                       (df_p_oi['groupsplit'] == groupsplit) &
                       (df_p_oi['conctype'] == conctype)].copy()
        #df_p_tv = df_params_tv[(df_params_tv['chem_fp'] == chem_fp) &
                               #(df_params_tv['groupsplit'] == groupsplit) & 
                               #(df_params_tv['conctype'] == conctype)].copy()

        # calculate number of feature and mean across cv folds
        df_p['feature_count'] = df_p.groupby(['feature'])['feature'].transform('count')
        df_p['feature_mean'] = df_p.groupby(['feature'])['value'].transform('mean')
        df_p['feature_mean_abs'] = df_p['feature_mean'].abs()

        # only features which were found in 3 and more folds
        # TODO apply this filter?
        #df_p = df_p[df_p['feature_count'] >= 3]

        # sort by feature count and mean
        df_p = df_p.sort_values(['feature_count', 'feature_mean_abs'], ascending=False)
        list_features = list(df_p['feature'].unique())#[::-1]

        # concatenate
        #df_p = pd.concat((df_p, df_p_tv))

        # from long to wide
        df_p_wide = df_p.pivot(index=['feature'], columns=['fold'], values=['value'])
        df_p_wide.columns = [item[1] for item in df_p_wide.columns]

        # sort wide data frame
        df_p_wide = df_p_wide.loc[list_features, :].copy()

        # ranges and wraps
        list_range = list(range(df_p_wide.shape[0]))
        n_per_row = 50
        list_wrap = [int(i/n_per_row) for i in list_range]
        df_p_wide['range'] = list_range
        df_p_wide['wrap'] = list_wrap

        # reset index
        df_p_wide = df_p_wide.reset_index()

        # remove intercept
        df_p_wide = df_p_wide[df_p_wide['feature'] != 'intercept']

        # add empty rows
        n_wraps = df_p_wide['wrap'].max() + 1
        n_missing = n_per_row - df_p_wide['wrap'].value_counts().tail(1).iloc[0]
        arr_tmp0 = np.full(n_missing, '').reshape(-1,1)
        arr_tmp1 = np.full((n_missing, 6), np.nan)    # set to 7 if test set included
        arr_tmp2 = np.full(n_missing, n_wraps - 1).reshape(-1,1)
        arr_tmp = np.concatenate((arr_tmp0, arr_tmp1, arr_tmp2), axis=1)
        df_tmp = pd.DataFrame(arr_tmp, columns=df_p_wide.columns)
        df_tmp['0'] = df_tmp['0'].astype('float')
        df_tmp['1'] = df_tmp['1'].astype('float')
        df_tmp['2'] = df_tmp['2'].astype('float')
        df_tmp['3'] = df_tmp['3'].astype('float')
        df_tmp['4'] = df_tmp['4'].astype('float')
        #df_tmp['trainvalid'] = df_tmp['trainvalid'].astype('float')
        df_tmp['range'] = df_tmp['range'].astype('float')
        df_tmp['wrap'] = df_tmp['wrap'].astype('int')

        df_p_wide = pd.concat((df_p_wide, df_tmp))

        # matplotlib figure
        n_wraps = df_p_wide['wrap'].max() + 1
        fig, axes = plt.subplots(nrows=1, ncols=n_wraps, figsize=(n_wraps*3.5, 10))
        fig.tight_layout(w_pad=7)

        list_folds = ['0', '1', '2', '3', '4']  #, 'trainvalid']
        vmin = df_p_wide[list_folds].min().min()
        vmax = df_p_wide[list_folds].max().max()

        for i in df_p_wide['wrap'].unique():
            # select wrap
            df_plot = df_p_wide[df_p_wide['wrap'] == i][['feature'] + list_folds].set_index('feature')

            # only show color bar for last wrap
            if i == n_wraps - 1:
                cbar = True
            else:
                cbar = False

            # heatmap
            sns.heatmap(df_plot, 
                        linewidth=0.5, 
                        cmap='PiYG', #'cividis',
                        vmin=vmin,
                        vmax=vmax,
                        center=0,
                        square=True,
                        ax=axes[i] if n_wraps > 1 else axes, 
                        cbar=cbar)
            if n_wraps > 1:
                axes[i].set_ylabel('')
                axes[i].set_title('')
            else:
                axes.set_ylabel('')
                axes.set_title('')
        fig.suptitle(chem_fp + '  ' + groupsplit, fontsize=18)
        fig.tight_layout()
        filepath = path_figures + '22_LASSO_featureimportance_' + chem_fp + '_' + groupsplit + '_' + conctype + '.png'
        #fig.savefig(filepath, facecolor='white')
        plt.show()


# %%

# TODO look at what? (pragmatism needed!)
#      - only for occurrence molar?
#      - only fold trainvalid aka test data? 
#      - only features from trainvalid?
#      - only above a certain threshold?

# %%
