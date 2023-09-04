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

path_vmoutput_gp = path_root + 'vm_output_gp/'
path_vmoutput_rf = path_root + 'vm_output_rf/'
path_vmoutput_lasso = path_root + 'vm_output_lasso/'


# %%

# sparse GP: Fish data, molar concentration, 2023-06-26

# completely new data pre-processing from ECOTOX 2022-09-15
# for the four fingerprints
# including chemical properties (mw, mp, ws, clogp)
# groupsplit: totallyrandom, occurrence (no scaffolds)

#param_grid = [
    #{
     ## features
     #'chem_fp': ['MACCS', 'mol2vec', 'pcp', 'Morgan'],
     #'chem_prop': ['chemprop'],                  #['none', 'chemprop'],
     #'tax_pdm': ['none'],                        #['none', 'pdm', 'pdm-squared'],
     #'tax_prop': ['taxprop-migrate2'],           #['none', 'taxprop-migrate2', 'taxprop-migrate5'],
     #'exp': ['exp-dropfirst'],                   #['none', 'exp-dropnone', 'exp-dropfirst'],     # use dropfirst
     ## splits
     #'groupsplit': ['totallyrandom', 'occurrence'],  
    #}
#]

#hyperparam_grid = [
    #{
     ## model hyperparameters     
     #'n_inducing': [100, 250, 500, 1000],
    #}
#]

path_output_dir = path_vmoutput_gp + '2023-06-26_molarconcentration/'
df_e_gp_molar = utils.read_result_files(path_output_dir, file_type='error')
df_p_gp_molar = utils.read_result_files(path_output_dir, file_type='preds')

# %%

# categorical variabls for fingerprints and fold
df_e_gp_molar['chem_fp'] = pd.Categorical(df_e_gp_molar['chem_fp'],
                                      categories=['MACCS', 'pcp', 'Morgan', 'mol2vec'],
                                      ordered=True)
df_p_gp_molar['chem_fp'] = pd.Categorical(df_p_gp_molar['chem_fp'],
                                     categories=['MACCS', 'pcp', 'Morgan', 'mol2vec'],
                                     ordered=True)
                                          
df_e_gp_molar['fold'] = df_e_gp_molar['fold'].astype('str')
df_p_gp_molar['fold'] = df_p_gp_molar['fold'].astype('str')
df_e_gp_molar['fold'] = pd.Categorical(df_e_gp_molar['fold'],
                                   categories=['mean', '0', '1', '2', '3', '4'],
                                   ordered=True)
df_p_gp_molar['fold'] = pd.Categorical(df_p_gp_molar['fold'],
                                  categories=['0', '1', '2', '3', '4'],
                                  ordered=True)

df_e_gp_molar['groupsplit'] = pd.Categorical(df_e_gp_molar['groupsplit'],
                                         categories=['totallyrandom', 'occurrence'],
                                         ordered=True)
df_p_gp_molar['groupsplit'] = pd.Categorical(df_p_gp_molar['groupsplit'],
                                         categories=['totallyrandom', 'occurrence'],
                                        ordered=True)

df_e_gp_molar = df_e_gp_molar.sort_values(['chem_fp', 'groupsplit'])
df_p_gp_molar = df_p_gp_molar.sort_values(['chem_fp', 'groupsplit'])

# check loading
if df_e_gp_molar.isna().sum().sum() > 0:
    print("warning: check whether loading was correct")
else:
    print("loading seems correct")

# %%

# Random forest: Fish data, molar concentration, 2023-06-21

# completely new data pre-processing from ECOTOX 2022-09-15
# for the four fingerprints
# including chemical properties (mw, mp, ws, clogp)
# groupsplit: totallyrandom, occurrence (no scaffolds)

#param_grid = [
    #{
     ## features
     #'chem_fp': ['MACCS', 'pcp', 'Morgan', 'mol2vec'], 
     #'chem_prop': ['chemprop'],                 #['none', 'chemprop'],
     #'tax_pdm': ['none'],                       #['none', 'pdm', 'pdm-squared'],
     #'tax_prop': ['taxprop-migrate2'],          #['none', 'taxprop-migrate2', 'taxprop-migrate5'],
     #'exp': ['exp-dropfirst'],                  #['none', 'exp-dropnone', 'exp-dropfirst'],     # use dropfirst
     ## splits
     #'groupsplit': ['totallyrandom', 'occurrence'], 
    #}
#]

#hyperparam_grid = [
    #{
    ## model hyperparameters     
    #'n_estimators': [50, 100, 150, 300],
    #'max_depth': [50, 100, 200], 
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1],
    #'max_samples': [0.25, 0.5, 1.0],  
    #'max_features': ['sqrt'],
    #}
#]

path_output_dir = path_vmoutput_rf + '2023-06-21_molarconcentration/'
df_e_rf_molar = utils.read_result_files(path_output_dir, file_type='error')
df_p_rf_molar = utils.read_result_files(path_output_dir, file_type='preds')

# %%

# categorical variabls for fingerprints and fold
df_e_rf_molar['chem_fp'] = pd.Categorical(df_e_rf_molar['chem_fp'],
                                      categories=['MACCS', 'pcp', 'Morgan', 'mol2vec'],
                                      ordered=True)
df_p_rf_molar['chem_fp'] = pd.Categorical(df_p_rf_molar['chem_fp'],
                                     categories=['MACCS', 'pcp', 'Morgan', 'mol2vec'],
                                     ordered=True)
                                          
df_e_rf_molar['fold'] = df_e_rf_molar['fold'].astype('str')
df_p_rf_molar['fold'] = df_p_rf_molar['fold'].astype('str')
df_e_rf_molar['fold'] = pd.Categorical(df_e_rf_molar['fold'],
                                   categories=['mean', '0', '1', '2', '3', '4'],
                                   ordered=True)
df_p_rf_molar['fold'] = pd.Categorical(df_p_rf_molar['fold'],
                                  categories=['0', '1', '2', '3', '4'],
                                  ordered=True)

df_e_rf_molar['groupsplit'] = pd.Categorical(df_e_rf_molar['groupsplit'],
                                         categories=['totallyrandom', 'occurrence'],
                                         ordered=True)
df_p_rf_molar['groupsplit'] = pd.Categorical(df_p_rf_molar['groupsplit'],
                                         categories=['totallyrandom', 'occurrence'],
                                        ordered=True)

df_e_rf_molar = df_e_rf_molar.sort_values(['chem_fp', 'groupsplit'])
df_p_rf_molar = df_p_rf_molar.sort_values(['chem_fp', 'groupsplit'])

# check loading
if df_e_rf_molar.isna().sum().sum() > 0:
    print("warning: check whether loading was correct")
else:
    print("loading seems correct")


# %%

chem_fp = 'MACCS'
groupsplit = 'occurrence'
tax_pdm = 'none'

df_p_gp = df_p_gp_molar[(df_p_gp_molar['chem_fp'] == chem_fp)
                        & (df_p_gp_molar['groupsplit'] == groupsplit)
                        & (df_p_gp_molar['tax_pdm'] == tax_pdm)].copy()
df_p_rf = df_p_rf_molar[(df_p_rf_molar['chem_fp'] == chem_fp)
                        & (df_p_rf_molar['groupsplit'] == groupsplit)].copy()
df_p_rf


# %%

# rename
df_p_gp = df_p_gp.rename(columns={'result_conc1_mean_mol_log': 'true',
                                  'conc_pred': 'gp_pred',
                                  #'conc_pred_var': 'gp_var',
                                  })

# calculate standard deviation
#df_p_gp['gp_sd'] = np.sqrt(df_p_gp['gp_var'])

# add random forest prediction
df_p = pd.merge(df_p_gp, 
                df_p_rf[['result_id', 'conc_pred']],
                left_on=['result_id'],
                right_on=['result_id'],
                how='left')

# rename
df_p = df_p.rename(columns={'conc_pred': 'rf_pred'})

# calculate residuals
df_p['gp_residual'] = df_p['gp_pred'] - df_p['true'] 
df_p['rf_residual'] = df_p['rf_pred'] - df_p['true'] 

# count chemicals and species
df_p['n_chemicals'] = df_p.groupby(['test_cas'])['result_id'].transform('count')
df_p['n_species'] = df_p.groupby(['tax_name'])['result_id'].transform('count')

df_p

# %%

# wide to long
id_vars=['result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs', 'n_species', 'n_chemicals']
value_vars=['gp_residual', 'rf_residual']
df_p_long = df_p.melt(id_vars=id_vars, 
                      value_vars=value_vars,
                      value_name='residual',
                      var_name='type')

df_p_long['type'] = df_p_long['type'].str.replace('gp_residual', 'GP')
df_p_long['type'] = df_p_long['type'].str.replace('rf_residual', 'RF')

# %%

# histogram for all residuals
(ggplot(data=df_p_long, mapping=aes(x='residual', fill='type'))
 + geom_histogram(position='identity', alpha=0.8)
 + scale_fill_manual(values=['#018571', '#dfc27d'])
 + theme_minimal()
 + labs(fill='')

)

# %%

# histograms for most common species
df_plot = df_p_long.copy()
df_plot = df_plot[df_plot['n_species'] > 100]
list_categories = df_plot['tax_name'].value_counts().index
df_plot['tax_name'] = pd.Categorical(df_plot['tax_name'],
                                     categories=list_categories,
                                     ordered=True)

xmax = df_plot['residual'].abs().max()

(ggplot(data=df_plot, 
        mapping=aes(x='residual', fill='type'))
 + geom_histogram(position='identity', alpha=0.8)
 + geom_vline(xintercept = -2, color='grey')
 + geom_vline(xintercept = 0, color='grey', linetype='dashed')
 + geom_vline(xintercept = 2, color='grey')
 + facet_wrap('~ tax_name')
 + scale_x_continuous(limits=[-xmax, xmax])
 + scale_fill_manual(values=['#018571', '#dfc27d'])
 + theme_minimal()
 + labs(fill='')
 + theme(figure_size=(18, 10))
)


# %%

# histograms for most common chemicals
df_plot = df_p_long.copy()
df_plot = df_plot[df_plot['n_chemicals'] > 100]
list_categories = df_plot['chem_name'].value_counts().index
df_plot['chem_name'] = pd.Categorical(df_plot['chem_name'],
                                      categories=list_categories,
                                      ordered=True)
xmax = df_plot['residual'].abs().max()

(ggplot(data=df_plot, 
        mapping=aes(x='residual', fill='type'))
 + geom_histogram(position='identity', alpha=0.8)
 + geom_vline(xintercept = -2, color='grey')
 + geom_vline(xintercept = 0, color='grey', linetype='dashed')
 + geom_vline(xintercept = 2, color='grey')
 + facet_wrap('~ chem_name')
 + scale_x_continuous(limits=[-xmax, xmax])
 + scale_fill_manual(values=['#018571', '#dfc27d'])
 + theme_minimal()
 + labs(fill='')
 + theme(figure_size=(18, 10))
)


# %%

oom = 1

# entries
print('entries')
print(df_p[(df_p['gp_residual']).abs() > oom].shape[0])
print(df_p[(df_p['rf_residual']).abs() > oom].shape[0])
print()

# chemicals
print('chemicals')
print(df_p[(df_p['gp_residual']).abs() > oom]['test_cas'].nunique())
print(df_p[(df_p['rf_residual']).abs() > oom]['test_cas'].nunique())
print()

# species
print('species')
print(df_p[(df_p['gp_residual']).abs() > oom]['tax_gs'].nunique())
print(df_p[(df_p['rf_residual']).abs() > oom]['tax_gs'].nunique())
print()

# %%

# TODO check residual plots correctly!!!
(ggplot(data=df_p, mapping=aes(x='true', y='rf_residual'))
    + geom_point(alpha=0.1) 
    + theme_minimal()
)
# %%
(ggplot(data=df_p, mapping=aes(x='true', y='gp_residual'))
    + geom_point(alpha=0.1) 
    + theme_minimal()
)



# %%

# sparse GP: Fish data, mass concentration, 2023-02-03

# !!! 'none' not available for most combinations!

# completely new data pre-processing from ECOTOX 2022-09-15
# for the four fingerprints
# including chemical properties (mw, mp, ws, clogp)
# groupsplit: totallyrandom, occurrence, murcko and general scaffold, (not loo!)
# with train error!

#param_grid = [
    #{
     ## features
     #'str_overlap': [str_overlap],
     #'chem_fp': ['MACCS', 'pcp', 'Morgan', 'mol2vec'], 
     #'chem_prop': ['chemprop'],  
     #'tax_pdm': ['none', 'pdm'],  
     #'tax_prop': ['taxprop-migrate2'],
     #'tax_prop_lh': ['narrow'],
     #'exp': ['exp-dropfirst'],  
     ## splits
    ##'groupsplit': ['totallyrandom', 'occurrence', 'scaffold-murcko', 'scaffold-generic'],  
    #}
#]

#hyperparam_grid = [
    #{
     ## model hyperparameters     
     #'n_inducing': [100, 250, 500, 1000],
    #}
#]

path_output_dir = path_vmoutput_gp + '2023-02-03_new-data_with-train-error_with-params/'
df_errors = utils.read_result_files(path_output_dir, file_type='errors')
df_params = utils.read_result_files(path_output_dir, file_type='param')
df_preds = utils.read_result_files(path_output_dir, file_type='pred')

# %%

# load corresponding RF output

# !!! so far only Morgan occurrence and by hyperparams!

# Morgan, occurrence, best hyperparams

#'n_estimators': [75],
#'max_depth': [200], 
#'min_samples_split': [2],
#'min_samples_leaf': [1],
#'max_samples': [1.0],  
#'max_features': ['sqrt'],


path_output_dir = path_vmoutput_rf + '2023-02-09_with-predictions/'
df_preds_rf = utils.read_result_files(path_output_dir, file_type='pred')


# %%

# categorical variabls for fingerprints and fold
df_errors['chem_fp'] = pd.Categorical(df_errors['chem_fp'],
                                      categories=['MACCS', 'pcp', 'Morgan', 'mol2vec'],
                                      ordered=True)
df_params['chem_fp'] = pd.Categorical(df_params['chem_fp'],
                                      categories=['MACCS', 'pcp', 'Morgan', 'mol2vec'],
                                      ordered=True)
df_preds['chem_fp'] = pd.Categorical(df_preds['chem_fp'],
                                     categories=['MACCS', 'pcp', 'Morgan', 'mol2vec'],
                                     ordered=True)
                                          
df_errors['fold'] = df_errors['fold'].astype('str')
df_params['fold'] = df_params['fold'].astype('str')
#df_preds['fold'] = df_preds['fold'].astype('str')
df_errors['fold'] = pd.Categorical(df_errors['fold'],
                                   categories=['mean', '0', '1', '2', '3', '4'],
                                   ordered=True)
df_params['fold'] = pd.Categorical(df_params['fold'],
                                   categories=['0', '1', '2', '3', '4'],
                                   ordered=True)
#df_preds['fold'] = pd.Categorical(df_preds['fold'],
                                  #categories=['0', '1', '2', '3', '4'],
                                  #ordered=True)

df_errors['groupsplit'] = pd.Categorical(df_errors['groupsplit'],
                                         categories=['totallyrandom', 'occurrence', 'scaffold-murcko', 'scaffold-generic'],
                                         ordered=True)
df_params['groupsplit'] = pd.Categorical(df_params['groupsplit'],
                                         categories=['totallyrandom', 'occurrence', 'scaffold-murcko', 'scaffold-generic'],
                                         ordered=True)
df_preds['groupsplit'] = pd.Categorical(df_preds['groupsplit'],
                                        categories=['totallyrandom', 'occurrence', 'scaffold-murcko', 'scaffold-generic'],
                                        ordered=True)

df_errors = df_errors.sort_values(['chem_fp', 'groupsplit'])
df_params = df_params.sort_values(['chem_fp', 'groupsplit'])
df_preds = df_preds.sort_values(['chem_fp', 'groupsplit'])

# check loading
if df_errors.isna().sum().sum() > 0:
    print("warning: check whether loading was correct")
else:
    print("loading seems correct")
# %%

chem_fp = 'Morgan'
groupsplit = 'occurrence'
tax_pdm = 'pdm'    # !!! 'none' not available for most combinations

df_errors[(df_errors['chem_fp'] == chem_fp) &
          (df_errors['groupsplit'] == groupsplit) &
          (df_errors['tax_pdm'] == tax_pdm) &
          (df_errors['fold'] == 'mean') &
          (df_errors['set'] == 'valid')
]

# %%
n_inducing = 100

df_p = df_preds[(df_preds['chem_fp'] == chem_fp) &
                (df_preds['groupsplit'] == groupsplit) &
                (df_preds['tax_pdm'] == tax_pdm) &
                (df_preds['n_inducing'] == n_inducing)
                ].copy()

df_p

# %%

# function
def calculate_pdf(mean, std_dev):

    import scipy.stats

    x_min=-5
    x_max=5

    x = np.linspace(x_min, x_max, 10000)
    y = scipy.stats.norm.pdf(x, mean, std_dev)

    df = pd.DataFrame(zip(x, y), columns=['x', 'y'])

    return df

# %%

df_plot = df_p.head(6)

# rename
df_plot = df_plot.rename(columns={'result_conc1_mean_log': 'true',
                                  'conc_pred': 'gp_pred',
                                  'conc_pred_var': 'gp_var'})
# calculate standard deviation
df_plot['gp_sd'] = np.sqrt(df_plot['gp_var'])

# add random forest prediction
# ! this is for Morgan, occurrence
df_plot = pd.merge(df_plot, 
                   df_preds_rf[['result_id', 'conc_pred']],
                   left_on=['result_id'],
                   right_on=['result_id'],
                   how='left')

# rename
df_plot = df_plot.rename(columns={'conc_pred': 'rf_pred'})

# wide to long
id_vars=['result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs']
value_vars=['true', 'gp_pred', 'rf_pred']
df_plot_long = df_plot.melt(id_vars=id_vars, 
                            value_vars=value_vars,
                            value_name='conc',
                            var_name='type')

# calculate pdf
list_dfs = []
for result_id in df_plot['result_id'].unique():

    df_oi = df_plot[df_plot['result_id'] == result_id]
    mean = df_oi['gp_pred'].iloc[0]
    std_dev = df_oi['gp_sd'].iloc[0]

    df = calculate_pdf(mean, std_dev)
    df['result_id'] = result_id
    list_dfs.append(df)

df_pdfs = pd.concat(list_dfs)


# %%

g = (ggplot()
    + geom_line(data=df_pdfs, mapping=aes(x='x', y='y'), color='#018571', linetype='dotted')
    + geom_point(data=df_plot_long, 
                 mapping=aes(x='conc', y=0, color='type', shape='type'), 
                 alpha=0.7,
                 size=3)
    + scale_color_manual(values=['black', '#018571', '#dfc27d'],
                         limits=['true', 'gp_pred', 'rf_pred'],
                         labels=['true', 'GP', 'RF'])
    + scale_shape_manual(values=[6, 7, 7],
                         limits=['true', 'gp_pred', 'rf_pred'],
                         labels=['true', 'GP', 'RF'])
    + facet_wrap('~ result_id')
    + theme_minimal()
    + labs(x='log$_{10}$(EC50)', 
           y='',
           color='',
           shape='')
)
#g.save(path_output_dir + '_GP-RF_prediction-uncertainty.png', facecolor='white')
g

# %%

#df_plot.to_csv(path_output_dir + '_GP-RF_prediction-uncertainty_data.csv')


# %%

# rename
df_p = df_p.rename(columns={'result_conc1_mean_log': 'true',
                            'conc_pred': 'gp_pred',
                            'conc_pred_var': 'gp_var'})

# calculate standard deviation
df_p['gp_sd'] = np.sqrt(df_p['gp_var'])

# add random forest prediction
# ! this is for Morgan, occurrence
df_p = pd.merge(df_p, 
                df_preds_rf[['result_id', 'conc_pred']],
                left_on=['result_id'],
                right_on=['result_id'],
                how='left')

# rename
df_p = df_p.rename(columns={'conc_pred': 'rf_pred'})

# calculate residuals
df_p['gp_residual'] = df_p['gp_pred'] - df_p['true'] 
df_p['rf_residual'] = df_p['rf_pred'] - df_p['true'] 

# count chemicals and species
df_p['n_chemicals'] = df_p.groupby(['test_cas'])['result_id'].transform('count')
df_p['n_species'] = df_p.groupby(['tax_name'])['result_id'].transform('count')

df_p

# %%

# wide to long
id_vars=['result_id', 'test_cas', 'chem_name', 'tax_name', 'tax_gs', 'n_species', 'n_chemicals']
value_vars=['gp_residual', 'rf_residual']
df_p_long = df_p.melt(id_vars=id_vars, 
                      value_vars=value_vars,
                      value_name='residual',
                      var_name='type')

df_p_long['type'] = df_p_long['type'].str.replace('gp_residual', 'GP')
df_p_long['type'] = df_p_long['type'].str.replace('rf_residual', 'RF')

# %%

# histogram for all residuals
(ggplot(data=df_p_long, mapping=aes(x='residual', fill='type'))
 + geom_histogram(position='identity', alpha=0.8)
 + scale_fill_manual(values=['#018571', '#dfc27d'])
 + theme_minimal()
 + labs(fill='')

)

# %%

# histograms for most common species
df_plot = df_p_long.copy()
df_plot = df_plot[df_plot['n_species'] > 100]
list_categories = df_plot['tax_name'].value_counts().index
df_plot['tax_name'] = pd.Categorical(df_plot['tax_name'],
                                     categories=list_categories,
                                     ordered=True)

xmax = df_plot['residual'].abs().max()

(ggplot(data=df_plot, 
        mapping=aes(x='residual', fill='type'))
 + geom_histogram(position='identity', alpha=0.8)
 + geom_vline(xintercept = -2, color='grey')
 + geom_vline(xintercept = 0, color='grey', linetype='dashed')
 + geom_vline(xintercept = 2, color='grey')
 + facet_wrap('~ tax_name')
 + scale_x_continuous(limits=[-xmax, xmax])
 + scale_fill_manual(values=['#018571', '#dfc27d'])
 + theme_minimal()
 + labs(fill='')
 + theme(figure_size=(18, 10))
)


# %%

# histograms for most common chemicals
df_plot = df_p_long.copy()
df_plot = df_plot[df_plot['n_chemicals'] > 100]
list_categories = df_plot['chem_name'].value_counts().index
df_plot['chem_name'] = pd.Categorical(df_plot['chem_name'],
                                      categories=list_categories,
                                      ordered=True)
xmax = df_plot['residual'].abs().max()

(ggplot(data=df_plot, 
        mapping=aes(x='residual', fill='type'))
 + geom_histogram(position='identity', alpha=0.8)
 + geom_vline(xintercept = -2, color='grey')
 + geom_vline(xintercept = 0, color='grey', linetype='dashed')
 + geom_vline(xintercept = 2, color='grey')
 + facet_wrap('~ chem_name')
 + scale_x_continuous(limits=[-xmax, xmax])
 + scale_fill_manual(values=['#018571', '#dfc27d'])
 + theme_minimal()
 + labs(fill='')
 + theme(figure_size=(18, 10))
)


# %%

oom = 1

# entries
print('entries')
print(df_p[(df_p['gp_residual']).abs() > oom].shape[0])
print(df_p[(df_p['rf_residual']).abs() > oom].shape[0])
print()

# chemicals
print('chemicals')
print(df_p[(df_p['gp_residual']).abs() > oom]['test_cas'].nunique())
print(df_p[(df_p['rf_residual']).abs() > oom]['test_cas'].nunique())
print()

# species
print('species')
print(df_p[(df_p['gp_residual']).abs() > oom]['tax_gs'].nunique())
print(df_p[(df_p['rf_residual']).abs() > oom]['tax_gs'].nunique())
print()

# %%

# TODO check residual plots correctly!!!
(ggplot(data=df_p, mapping=aes(x='true', y='rf_residual'))
    + geom_point(alpha=0.1) 
)
# %%
(ggplot(data=df_p, mapping=aes(x='true', y='gp_residual'))
    + geom_point(alpha=0.01) 
)
# %%
(ggplot(data=df_p, mapping=aes(x='true', y='gp_pred'))
    #+ geom_point(alpha=0.01) 
    + geom_bin2d(bins=80)
    + geom_abline()
    + scale_fill_cmap('cividis')
    + scale_x_continuous(limits=(-9,2))
    + scale_y_continuous(limits=(-9,2))
)

# %%
