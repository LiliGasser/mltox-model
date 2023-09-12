
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import utils

%reload_ext autoreload
%autoreload 2

# %%

path_output = path_root + 'output/regression/'
path_figures = path_output + 'figures/'

# %%

def compile_errors(modeltype):

    # load error files
    df_cv = pd.read_csv(path_output + modeltype + '_CV-errors.csv')
    df_test = pd.read_csv(path_output + modeltype + '_test-errors.csv')
    df = pd.concat((df_cv, df_test)).reset_index(drop=True)

    # add modeltype
    if modeltype == 'lasso':
        df['model'] = 'LASSO'
    elif modeltype == 'rf':
        df['model'] = 'RF'
    elif modeltype == 'xgboost':
        df['model'] = 'XGBoost'
    elif modeltype == 'gp':
        df['model'] = 'GP'

    return df

# %%

# load LASSO: 
# 4 fps x 2 groupsplits x 4 sets x 2 concentrations = 64 entries
df_lasso = compile_errors(modeltype='lasso')
df_lasso

# %%

# load RF
# 4 fps x 2 groupsplits x 4 sets x 2 concentrations = 64 entries
df_rf = compile_errors(modeltype='rf')
df_rf

# %%

# load XGBoost
# 4 fps x 2 groupsplits x 4 sets x 2 concentrations = 64 entries
df_xgboost = compile_errors(modeltype='xgboost')
df_xgboost

# %%

# load GP
# 4 fps x 2 groupsplits x 4 sets x 2 tax_pdm x 2 concentrations - 8 entries = 120 entries
# missing for molar: Morgan, totallyrandom, none, test
#df_gp[(df_gp['conctype'] == 'molar') & (df_gp['chem_fp'] == 'Morgan')]
# missing for mass: pcp, totallyrandom, none; Morgan, occurrence, none, training and test
#df_gp[(df_gp['conctype'] == 'mass') & (df_gp['chem_fp'] == 'Morgan')]
#df_gp[(df_gp['conctype'] == 'mass') & (df_gp['chem_fp'] == 'pcp')]
df_gp = compile_errors(modeltype='gp')
df_gp

# %%

# GP: without tax_pdm
df_gp = df_gp[df_gp['tax_pdm'] == 'none'].copy()

# %%

# load GP forward
# !! so far only on mass concentration
#df_gpforward_cv = pd.read_csv(path_output + 'gpforward_CV-errors.csv')
#df_gpforward_test = pd.read_csv(path_output + 'gpforward_test-errors.csv')
#df_gpforward = pd.concat((df_gpforward_cv, df_gpforward_test)).reset_index(drop=True)
#df_gpforward['model'] = 'GPforward'
#df_gpforward

# %%

### Intermezzo: GP forward for different runs and iterations

#df_plot = df_gpforward.copy()
##df_plot = df_gpforward[df_gpforward['groupsplit'] == 'totallyrandom']
#df_plot['set_run'] = df_plot['set'] + df_plot['run'].astype('str')

#(ggplot(data=df_plot, 
        #mapping=aes(x='iteration', 
                    #y='rmse',
                    #group='set_run',
                    #fill='set',
                    #color='set'))
    #+ geom_point() 
    #+ geom_line()
    #+ facet_grid("chem_fp ~ groupsplit")
#)

# %%

# GP forward: get best test error for each fp and groupsplit
#list_df_new = []

#metric = 'rmse'
#for chem_fp in ['MACCS', 'pcp', 'Morgan', 'mol2vec']:
    #for groupsplit in ['totallyrandom', 'occurrence', 'scaffold-murcko', 'scaffold-generic']:
        #print(chem_fp, groupsplit)
        #df_t = df_gpforward[(df_gpforward['chem_fp'] == chem_fp)
                            #& (df_gpforward['groupsplit'] == groupsplit)
                            #& (df_gpforward['set'] == 'test')]
        #df_t_min = df_t.loc[df_t[metric].idxmin(), ]
        #run = df_t_min['run']
        #iteration = df_t_min['iteration']

        #df_new = df_gpforward[(df_gpforward['chem_fp'] == chem_fp)
                              #& (df_gpforward['groupsplit'] == groupsplit)
                              #& (df_gpforward['run'] == run)
                              #& (df_gpforward['iteration'] == iteration)]
        #list_df_new.append(df_new)

        #print(run, iteration)
        #print(df_new.shape)

#df_gpf = pd.concat(list_df_new)
#df_gpf


# %%

# concatenate all error files
# for GP forward, use only those with best test errors
#df_errors = pd.concat([df_lasso, df_rf, df_xgboost, df_gp, df_gpf], axis=0)
df_errors = pd.concat([df_lasso, df_rf, df_xgboost, df_gp], axis=0)
df_errors

# %%

# only two group splits
list_cols = ['totallyrandom', 'occurrence']
df_errors = df_errors[df_errors['groupsplit'].isin(list_cols)].copy()

# categorical variables
df_errors = utils._transform_to_categorical(df_errors, 'groupsplit', ['totallyrandom', 'occurrence'])
df_errors = utils._transform_to_categorical(df_errors, 'chem_fp', ['MACCS', 'pcp', 'Morgan', 'mol2vec'])
df_errors = utils._transform_to_categorical(df_errors, 'model', ['LASSO', 'RF', 'XGBoost', 'GP'])
df_errors = utils._transform_to_categorical(df_errors, 'set', ['train', 'valid', 'trainvalid', 'test'])
df_errors = utils._transform_to_categorical(df_errors, 'conctype', ['molar', 'mass'])

# %%

# best hyperparameters for LASSO
df_lasso = utils._transform_to_categorical(df_lasso, 'conctype', ['molar', 'mass'])
df_lasso = utils._transform_to_categorical(df_lasso, 'groupsplit', ['totallyrandom', 'occurrence'])
df_lasso = utils._transform_to_categorical(df_lasso, 'chem_fp', ['MACCS', 'pcp', 'Morgan', 'mol2vec'])

list_cols_hp = ['alpha']
list_cols = ['conctype', 'groupsplit', 'chem_fp']  #, 'rmse', 'mae', 'r2']
list_cols += list_cols_hp
list_cols_sort = ['conctype', 'groupsplit', 'chem_fp']
df_l = df_lasso[df_lasso['set'] == 'train'][list_cols].sort_values(list_cols_sort).copy()
print(df_l.to_latex(index=False))

# %%

# best hyperparameters for RF
df_rf = utils._transform_to_categorical(df_rf, 'conctype', ['molar', 'mass'])
df_rf = utils._transform_to_categorical(df_rf, 'groupsplit', ['totallyrandom', 'occurrence'])
df_rf = utils._transform_to_categorical(df_rf, 'chem_fp', ['MACCS', 'pcp', 'Morgan', 'mol2vec'])

# 'max_features', 'min_samples_leaf', 
list_cols_hp = ['n_estimators', 'max_depth', 'max_samples', 'min_samples_split', 'max_features']
list_cols = ['conctype', 'groupsplit', 'chem_fp']  #, 'rmse', 'mae', 'r2']
list_cols += list_cols_hp
list_cols_sort = ['conctype', 'groupsplit', 'chem_fp']
df_l = df_rf[df_rf['set'] == 'train'][list_cols].sort_values(list_cols_sort).copy()
print(df_l.to_latex(index=False))

# %%

# best hyperparameters for XGBoost
df_xgboost = utils._transform_to_categorical(df_xgboost, 'conctype', ['molar', 'mass'])
df_xgboost = utils._transform_to_categorical(df_xgboost, 'groupsplit', ['totallyrandom', 'occurrence'])
df_xgboost = utils._transform_to_categorical(df_xgboost, 'chem_fp', ['MACCS', 'pcp', 'Morgan', 'mol2vec'])

list_cols_hp = ['n_estimators', 'eta', 'gamma', 'max_depth', 'min_child_weight', 'subsample']
list_cols = ['conctype', 'groupsplit', 'chem_fp']  #, 'rmse', 'mae', 'r2']
list_cols += list_cols_hp
list_cols_sort = ['conctype', 'groupsplit', 'chem_fp']
df_l = df_xgboost[df_xgboost['set'] == 'train'][list_cols].sort_values(list_cols_sort).copy()
print(df_l.to_latex(index=False))

# %%

# best hyperparameters for GP
df_gp = utils._transform_to_categorical(df_gp, 'conctype', ['molar', 'mass'])
df_gp = utils._transform_to_categorical(df_gp, 'groupsplit', ['totallyrandom', 'occurrence'])
df_gp = utils._transform_to_categorical(df_gp, 'chem_fp', ['MACCS', 'pcp', 'Morgan', 'mol2vec'])

list_cols_hp = ['n_inducing']
list_cols = ['conctype', 'groupsplit', 'chem_fp']  #, 'rmse', 'mae', 'r2']
list_cols += list_cols_hp
list_cols_sort = ['conctype', 'groupsplit', 'chem_fp']
df_l = df_gp[df_gp['set'] == 'train'][list_cols].sort_values(list_cols_sort).copy()
print(df_l.to_latex(index=False))


# %%

# prepare plotting

# color specifications

# for chem_fp: colors from CH2018 report
list_colors = ['#75aab9', '#998478', '#80a58b', '#fbba76']

# for errors
list_colors_points = ['#ccc', '#444', '#999', 'black']

# assign colors
list_chem_fps = ['MACCS', 'pcp', 'Morgan', 'mol2vec']
dict_colors_fps = dict(zip(list_chem_fps, list_colors))
list_sets = ['train', 'valid', 'trainvalid', 'test']
dict_colors_points = dict(zip(list_sets, list_colors_points))

# %%

# overview of test error  (outdated using plotnine)
# !! outdated using plotnine

df_plot = df_errors.copy()
df_plot = df_plot.sort_values(['model', 'chem_fp', 'set'], ascending=[True, True, False])
df_plot['model_chem_fp'] = df_plot['model'].astype('str') + ' ' + df_plot['chem_fp'].astype('str')
df_plot['groupsplit_conctype'] = df_plot['groupsplit'].astype('str') + ' ' + df_plot['conctype'].astype('str')
df_plot_test = df_plot[df_plot['set'] == 'test'].copy()

df_plot['model_chem_fp'] = pd.Categorical(df_plot['model_chem_fp'], 
                                          categories=df_plot['model_chem_fp'].unique(),
                                          ordered=True)
df_plot_test['model_chem_fp'] = pd.Categorical(df_plot_test['model_chem_fp'], 
                                               categories=df_plot['model_chem_fp'].unique(),
                                               ordered=True)

df_plot['groupsplit_conctype'] = pd.Categorical(df_plot['groupsplit_conctype'], 
                                          categories=df_plot['groupsplit_conctype'].unique(),
                                          ordered=True)
df_plot_test['groupsplit_conctype'] = pd.Categorical(df_plot_test['groupsplit_conctype'], 
                                               categories=df_plot['groupsplit_conctype'].unique(),
                                               ordered=True)

# TODO run for RMSE and R2
metric = 'rmse'
#metric = 'mae'
#metric = 'r2'
g = (ggplot(data=df_plot, mapping=aes(x='model_chem_fp', 
                                      y=metric, 
                                      color='set', 
                                      fill='chem_fp',
                                      ))
    + geom_col(data=df_plot_test, color='none')
    + geom_point(alpha=0.9, shape='o', fill='none')
    + facet_grid("groupsplit ~ conctype")     # TODO labels on the left --> plot with matploblib or plotly?
    + scale_color_manual(values=list_colors_points)
    + scale_fill_manual(values=list_colors)
    + geom_hline(yintercept=0.25, color='white', size=0.25)
    + geom_hline(yintercept=0.5, color='white', size=0.25)
    + geom_hline(yintercept=0.75, color='white', size=0.25)
    + geom_hline(yintercept=1., color='white', size=0.25)
    + geom_hline(yintercept=1.25, color='white', size=0.25)
    + geom_hline(yintercept=1.5, color='white', size=0.25)
    + theme_tufte()
    + labs(x='', fill='fingerprint', color='error type')
    + theme(axis_text_x=element_blank())
)
if metric == 'r2':
    g = g + labs(y="R$^2$")
elif metric == 'rmse':
    g = g + labs(y="RMSE")
elif metric == 'mae':
    g = g + labs(y="MAE")
#g.save(path_figures + '29_all_' + metric + '-vs-models.png', facecolor='white')
g
# %%

def _calculate_metric_stuff(metric):

    if metric == 'r2':
        metric_max = 1.01
        metric_step = 0.2
        str_metric = r'$R^2$'
    elif metric == 'rmse':
        metric_max = df_errors[metric].max() + 0.1
        metric_step = 0.25
        str_metric = 'RMSE'
    elif metric == 'mae':
        metric_max = df_errors[metric].max() + 0.1
        metric_step = 0.25
        str_metric = 'MAE'

    return metric_max, metric_step, str_metric

# %%

# overview plot in plotly (test error)
# !!! test error

# TODO run for RMSE and R2
metric = 'rmse'
#metric = 'mae'
#metric = 'r2'

# calculate maximum
metric_max, metric_step, str_metric = _calculate_metric_stuff(metric)
list_tickvals = list(np.arange(0, metric_max, metric_step))

# Initialize figure with subplots
fig = make_subplots(
    rows=2, 
    cols=2, 
    subplot_titles=("totally random split<br>mass concentration", 
                    "totally random split<br>molar concentration", 
                    "split by occurrence<br>mass concentration", 
                    "split by occurrence<br>molar concentration"),
)

# Add traces
# totallyrandom, mass
df_plot = df_errors[(df_errors['groupsplit'] == 'totallyrandom')
                    & (df_errors['conctype'] == 'mass')].copy()
df_plot['model_chem_fp'] = df_plot['model'].astype('str') + ' ' + df_plot['chem_fp'].astype('str')
df_plot_test = df_plot[(df_plot['set'] == 'test')].copy()
row = 1
col = 1
for i, chem_fp in enumerate(list_chem_fps):     # add bars
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_test[df_plot_test['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         name=chem_fp,
                         offsetgroup=i+1,
                         showlegend=False),
                  row=row, 
                  col=col)
for y in list_tickvals:    # add horizontal lines
    fig.add_hline(y=y, 
                  line_width=1.,
                  line_color='#eee',
                  row=row,
                  col=col)
for i, chem_fp in enumerate(list_chem_fps):    # add points
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Scatter(x=df_p['model'],
                             y=df_p[metric],
                             mode='markers',
                             marker_color=df_p['set'].map(dict_colors_points),
                             marker_symbol='circle-open',
                             marker_size=7,
                             name=chem_fp,
                             offsetgroup=i+1,
                             showlegend=False),
                  row=row, 
                  col=col)

# totallyrandom, molar
df_plot = df_errors[(df_errors['groupsplit'] == 'totallyrandom')
                    & (df_errors['conctype'] == 'molar')].copy()
df_plot['model_chem_fp'] = df_plot['model'].astype('str') + ' ' + df_plot['chem_fp'].astype('str')
df_plot_test = df_plot[(df_plot['set'] == 'test')].copy()
row = 1
col = 2
for i, chem_fp in enumerate(list_chem_fps):     # add bars
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_test[df_plot_test['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         name=chem_fp,
                         offsetgroup=i+1,
                         showlegend=False),
                  row=row, 
                  col=col)
for y in list_tickvals:    # add horizontal lines
    fig.add_hline(y=y, 
                  line_width=1.,
                  line_color='#eee',
                  row=row,
                  col=col)
for i, chem_fp in enumerate(list_chem_fps):    # add points
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Scatter(x=df_p['model'],
                             y=df_p[metric],
                             mode='markers',
                             marker_color=df_p['set'].map(dict_colors_points),
                             marker_symbol='circle-open',
                             marker_size=7,
                             name=chem_fp,
                             offsetgroup=i+1,
                             showlegend=False),
                  row=row, 
                  col=col)

# occurrence, mass
df_plot = df_errors[(df_errors['groupsplit'] == 'occurrence')
                    & (df_errors['conctype'] == 'mass')].copy()
df_plot['model_chem_fp'] = df_plot['model'].astype('str') + ' ' + df_plot['chem_fp'].astype('str')
df_plot_test = df_plot[(df_plot['set'] == 'test')].copy()
row = 2
col = 1
for i, chem_fp in enumerate(list_chem_fps):     # add bars
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_test[df_plot_test['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         name=chem_fp,
                         offsetgroup=i+1,
                         showlegend=False),
                  row=row, 
                  col=col)
for y in list_tickvals:    # add horizontal lines
    fig.add_hline(y=y, 
                  line_width=1.,
                  line_color='#eee',
                  row=row,
                  col=col)
for i, chem_fp in enumerate(list_chem_fps):    # add points
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Scatter(x=df_p['model'],
                             y=df_p[metric],
                             mode='markers',
                             marker_color=df_p['set'].map(dict_colors_points),
                             marker_symbol='circle-open',
                             marker_size=7,
                             name=chem_fp,
                             offsetgroup=i+1,
                             showlegend=False),
                  row=row, 
                  col=col)

# occurrence, molar
df_plot = df_errors[(df_errors['groupsplit'] == 'occurrence')
                    & (df_errors['conctype'] == 'molar')].copy()
df_plot['model_chem_fp'] = df_plot['model'].astype('str') + ' ' + df_plot['chem_fp'].astype('str')
df_plot_test = df_plot[(df_plot['set'] == 'test')].copy()
row = 2
col = 2
for i, chem_fp in enumerate(list_chem_fps):     # add bars
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_test[df_plot_test['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         name=chem_fp,
                         offsetgroup=i+1,
                         showlegend=False),
                  row=row, 
                  col=col)
for y in list_tickvals:    # add horizontal lines
    fig.add_hline(y=y, 
                  line_width=1.,
                  line_color='#eee',
                  row=row,
                  col=col)
for i, chem_fp in enumerate(list_chem_fps):    # add points
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Scatter(x=df_p['model'],
                             y=df_p[metric],
                             mode='markers',
                             marker_color=df_p['set'].map(dict_colors_points),
                             marker_symbol='circle-open',
                             marker_size=7,
                             name=chem_fp,
                             offsetgroup=i+1,
                             showlegend=False),
                  row=row, 
                  col=col)

# Update xaxis properties
fig.update_xaxes(title_text='', row=1, col=1)
fig.update_xaxes(title_text='', row=1, col=2)
fig.update_xaxes(title_text='', row=2, col=1)
fig.update_xaxes(title_text='', row=2, col=2)

# Update yaxis properties
fig.update_yaxes(title_text=str_metric, row=1, col=1)
fig.update_yaxes(title_text='', row=1, col=2)
fig.update_yaxes(title_text=str_metric, row=2, col=1)
fig.update_yaxes(title_text='', row=2, col=2)

# Set y axis limits
fig.update_yaxes(range=[0, metric_max], tickvals=list_tickvals)

# Grouped bars and scatter
#fig.update_layout(barmode='group')
fig.update_layout(scattermode='group')

# Update title and height
fig.update_layout(height=600, width=900)

# add legend for fingerprints
for chem_fp in list_chem_fps:
    fig.add_trace(go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             name=chem_fp,
                             legendgroup='chem_fp',
                             legendgrouptitle_text='molecular<br>representation',
                             marker_color=dict_colors_fps[chem_fp],
                             marker_symbol='square',
                             marker_size=12))
for errortype in list_sets:
    fig.add_trace(go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             name=errortype,
                             legendgroup='set',
                             legendgrouptitle_text='error type',
                             marker_color=dict_colors_points[errortype],
                             marker_symbol='circle-open',
                             marker_size=7))
# TODO legend at bottom horizontally aligned
#fig.update_layout(legend_orientation='h', legend_xanchor='center', legend_x=0.5)

fig.update_layout(template='plotly_white')

fig.write_image(path_figures + '29_all_' + metric + '-vs-models.png')
fig.show()

# %%
# %%

# overview plot in plotly (validation error)
# !!! validation error

# TODO run for RMSE and R2
metric = 'rmse'
#metric = 'mae'
#metric = 'r2'

# calculate maximum
metric_max, metric_step, str_metric = _calculate_metric_stuff(metric)
list_tickvals = list(np.arange(0, metric_max, metric_step))

# Initialize figure with subplots
fig = make_subplots(
    rows=2, 
    cols=2, 
    subplot_titles=("totally random split<br>mass concentration", 
                    "totally random split<br>molar concentration", 
                    "split by occurrence<br>mass concentration", 
                    "split by occurrence<br>molar concentration"),
)

# Add traces
# totallyrandom, mass
df_plot = df_errors[(df_errors['groupsplit'] == 'totallyrandom')
                    & (df_errors['conctype'] == 'mass')
                    & (df_errors['set'].isin(['train', 'valid']))].copy()
#df_plot['model_chem_fp'] = df_plot['model'].astype('str') + ' ' + df_plot['chem_fp'].astype('str')
df_plot_bar = df_plot[(df_plot['set'] == 'valid')].copy()
row = 1
col = 1
for i, chem_fp in enumerate(list_chem_fps):     # add bars
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_bar[df_plot_bar['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         name=chem_fp,
                         offsetgroup=i+1,
                         showlegend=False),
                  row=row, 
                  col=col)
for y in list_tickvals:    # add horizontal lines
    fig.add_hline(y=y, 
                  line_width=0.5,
                  line_color='#eee',
                  row=row,
                  col=col)
#for i, chem_fp in enumerate(list_chem_fps):    # add points
    #df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    #fig.add_trace(go.Scatter(x=df_p['model'],
                             #y=df_p[metric],
                             #mode='markers',
                             #marker_color=df_p['set'].map(dict_colors_points),
                             #marker_symbol='circle-open',
                             #marker_size=7,
                             #name=chem_fp,
                             #offsetgroup=i+1,
                             #showlegend=False),
                  #row=row, 
                  #col=col)

# totallyrandom, molar
df_plot = df_errors[(df_errors['groupsplit'] == 'totallyrandom')
                    & (df_errors['conctype'] == 'molar')
                    & (df_errors['set'].isin(['train', 'valid']))].copy()
#df_plot['model_chem_fp'] = df_plot['model'].astype('str') + ' ' + df_plot['chem_fp'].astype('str')
df_plot_bar = df_plot[(df_plot['set'] == 'valid')].copy()
row = 1
col = 2
for i, chem_fp in enumerate(list_chem_fps):     # add bars
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_bar[df_plot_bar['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         name=chem_fp,
                         offsetgroup=i+1,
                         showlegend=False),
                  row=row, 
                  col=col)
for y in list_tickvals:    # add horizontal lines
    fig.add_hline(y=y, 
                  line_width=0.5,
                  line_color='#eee',
                  row=row,
                  col=col)
#for i, chem_fp in enumerate(list_chem_fps):    # add points
    #df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    #fig.add_trace(go.Scatter(x=df_p['model'],
                             #y=df_p[metric],
                             #mode='markers',
                             #marker_color=df_p['set'].map(dict_colors_points),
                             #marker_symbol='circle-open',
                             #marker_size=7,
                             #name=chem_fp,
                             #offsetgroup=i+1,
                             #showlegend=False),
                  #row=row, 
                  #col=col)

# occurrence, mass
df_plot = df_errors[(df_errors['groupsplit'] == 'occurrence')
                    & (df_errors['conctype'] == 'mass')
                    & (df_errors['set'].isin(['train', 'valid']))].copy()
#df_plot['model_chem_fp'] = df_plot['model'].astype('str') + ' ' + df_plot['chem_fp'].astype('str')
df_plot_bar = df_plot[(df_plot['set'] == 'valid')].copy()
row = 2
col = 1
for i, chem_fp in enumerate(list_chem_fps):     # add bars
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_bar[df_plot_bar['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         name=chem_fp,
                         offsetgroup=i+1,
                         showlegend=False),
                  row=row, 
                  col=col)
for y in list_tickvals:    # add horizontal lines
    fig.add_hline(y=y, 
                  line_width=0.5,
                  line_color='#eee',
                  row=row,
                  col=col)
#for i, chem_fp in enumerate(list_chem_fps):    # add points
    #df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    #fig.add_trace(go.Scatter(x=df_p['model'],
                             #y=df_p[metric],
                             #mode='markers',
                             #marker_color=df_p['set'].map(dict_colors_points),
                             #marker_symbol='circle-open',
                             #marker_size=7,
                             #name=chem_fp,
                             #offsetgroup=i+1,
                             #showlegend=False),
                  #row=row, 
                  #col=col)

# occurrence, molar
df_plot = df_errors[(df_errors['groupsplit'] == 'occurrence')
                    & (df_errors['conctype'] == 'molar')
                    & (df_errors['set'].isin(['train', 'valid']))].copy()
#df_plot['model_chem_fp'] = df_plot['model'].astype('str') + ' ' + df_plot['chem_fp'].astype('str')
df_plot_bar = df_plot[(df_plot['set'] == 'valid')].copy()
row = 2
col = 2
for i, chem_fp in enumerate(list_chem_fps):     # add bars
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_bar[df_plot_bar['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         name=chem_fp,
                         offsetgroup=i+1,
                         showlegend=False),
                  row=row, 
                  col=col)
for y in list_tickvals:    # add horizontal lines
    fig.add_hline(y=y, 
                  line_width=0.5,
                  line_color='#eee',
                  row=row,
                  col=col)
#for i, chem_fp in enumerate(list_chem_fps):    # add points
    #df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    #fig.add_trace(go.Scatter(x=df_p['model'],
                             #y=df_p[metric],
                             #mode='markers',
                             #marker_color=df_p['set'].map(dict_colors_points),
                             #marker_symbol='circle-open',
                             #marker_size=7,
                             #name=chem_fp,
                             #offsetgroup=i+1,
                             #showlegend=False),
                  #row=row, 
                  #col=col)

# Update xaxis properties
fig.update_xaxes(title_text='', row=1, col=1)
fig.update_xaxes(title_text='', row=1, col=2)
fig.update_xaxes(title_text='', row=2, col=1)
fig.update_xaxes(title_text='', row=2, col=2)

# Update yaxis properties
fig.update_yaxes(title_text=str_metric, row=1, col=1)
fig.update_yaxes(title_text='', row=1, col=2)
fig.update_yaxes(title_text=str_metric, row=2, col=1)
fig.update_yaxes(title_text='', row=2, col=2)

# Set y axis limits
fig.update_yaxes(range=[0, metric_max], tickvals=list_tickvals)

# Grouped bars and scatter
#fig.update_layout(barmode='group')
fig.update_layout(scattermode='group')

# Update title and height
fig.update_layout(height=600, width=900)

# add legend for fingerprints
for chem_fp in list_chem_fps:
    fig.add_trace(go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             name=chem_fp,
                             #legendgroup='chem_fp',
                             legendgrouptitle_text='molecular representation',
                             marker_color=dict_colors_fps[chem_fp],
                             marker_symbol='square',
                             marker_size=12))
#for errortype in ['train', 'valid']:
    #fig.add_trace(go.Scatter(x=[None],
                             #y=[None],
                             #mode='markers',
                             #name=errortype,
                             #legendgroup='set',
                             #legendgrouptitle_text='error type',
                             #marker_color=dict_colors_points[errortype],
                             #marker_symbol='circle-open',
                             #marker_size=7))
fig.update_layout(legend_orientation='h', legend_xanchor='center', legend_x=0.5)

fig.update_layout(template='plotly_white')

fig.write_image(path_figures + '29_all_' + metric + '-vs-models_validation.png')
fig.show()

# %% 
# %%

# overview plot in plotly (only molar occurrence)
# !!! test error only for occurrence and molar

# TODO also include MAE?

# Initialize figure with subplots
fig = make_subplots(
    rows=2, 
    cols=1, 
    subplot_titles=('split by occurrence<br>molar concentration', 
                    ''),
)

# Add traces
# occurrence, molar, RMSE
metric='rmse'
metric_max, metric_step, str_metric = _calculate_metric_stuff(metric)
list_tickvals = list(np.arange(0, metric_max, metric_step))
df_plot = df_errors[(df_errors['groupsplit'] == 'occurrence')
                    & (df_errors['conctype'] == 'molar')].copy()
df_plot_bar = df_plot[(df_plot['set'] == 'test')].copy()
df_plot_dot = df_plot[(df_plot['set'] == 'valid')].copy()
row = 1
col = 1
for i, chem_fp in enumerate(list_chem_fps):     # add bars
    df_pb = df_plot_bar[df_plot_bar['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pb['model'], 
                         y=df_pb[metric],
                         marker_color=df_pb['chem_fp'].map(dict_colors_fps),
                         name=chem_fp,
                         offsetgroup=i+1,
                         showlegend=False),
                  row=row, 
                  col=col)
for y in list_tickvals:    # add horizontal lines
    fig.add_hline(y=y, 
                  line_width=1.,
                  line_color='#eee',
                  row=row,
                  col=col)
for i, chem_fp in enumerate(list_chem_fps):    # add points
    df_pt = df_plot_dot[df_plot_dot['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Scatter(x=df_pt['model'],
                             y=df_pt[metric],
                             mode='markers',
                             marker_color=df_pt['set'].map(dict_colors_points),
                             marker_symbol='diamond-open',
                             marker_size=7,
                             name=chem_fp,
                             offsetgroup=i+1,
                             showlegend=False),
                  row=row, 
                  col=col)

# update axes
fig.update_xaxes(title_text='', row=row, col=col)
fig.update_yaxes(title_text=str_metric, row=row, col=col)
fig.update_yaxes(range=[0, metric_max], tickvals=list_tickvals, row=row, col=col)

# occurrence, molar, RMSE
metric='r2'
metric_max, metric_step, str_metric = _calculate_metric_stuff(metric)
list_tickvals = list(np.arange(0, metric_max, metric_step))
df_plot = df_errors[(df_errors['groupsplit'] == 'occurrence')
                    & (df_errors['conctype'] == 'molar')].copy()
df_plot_bar = df_plot[(df_plot['set'] == 'test')].copy()
df_plot_dot = df_plot[(df_plot['set'] == 'valid')].copy()
row = 2
col = 1
for i, chem_fp in enumerate(list_chem_fps):     # add bars
    df_pb = df_plot_bar[df_plot_bar['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pb['model'], 
                         y=df_pb[metric],
                         marker_color=df_pb['chem_fp'].map(dict_colors_fps),
                         name=chem_fp,
                         offsetgroup=i+1,
                         showlegend=False),
                  row=row, 
                  col=col)
for y in list_tickvals:    # add horizontal lines
    fig.add_hline(y=y, 
                  line_width=1.,
                  line_color='#eee',
                  row=row,
                  col=col)
for i, chem_fp in enumerate(list_chem_fps):    # add points
    df_pt = df_plot_dot[df_plot_dot['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Scatter(x=df_pt['model'],
                             y=df_pt[metric],
                             mode='markers',
                             marker_color=df_pt['set'].map(dict_colors_points),
                             marker_symbol='diamond-open',
                             marker_size=7,
                             name=chem_fp,
                             offsetgroup=i+1,
                             showlegend=False),
                  row=row, 
                  col=col)

# update axes
fig.update_xaxes(title_text='', row=row, col=col)
fig.update_yaxes(title_text=str_metric, row=row, col=col)
fig.update_yaxes(range=[0, metric_max], tickvals=list_tickvals, row=row, col=col)

# Grouped bars and scatter
#fig.update_layout(barmode='group')
fig.update_layout(scattermode='group')

# Update title and height
fig.update_layout(height=600, width=500)

# add legend for fingerprints
for chem_fp in list_chem_fps:
    fig.add_trace(go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             name=chem_fp,
                             legendgroup='chem_fp',
                             legendgrouptitle_text='molecular<br>representation',
                             marker_color=dict_colors_fps[chem_fp],
                             marker_symbol='square',
                             marker_size=12))
for errortype in ['valid']:
    fig.add_trace(go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             name=errortype,
                             legendgroup='set',
                             legendgrouptitle_text='error type',
                             marker_color=dict_colors_points[errortype],
                             marker_symbol='diamond-open',
                             marker_size=7))
# TODO legend at bottom horizontally aligned
#fig.update_layout(legend_orientation='h', legend_xanchor='center', legend_x=0.5)

fig.update_layout(template='plotly_white')

fig.write_image(path_figures + '29_all_rmse-r2-vs-models_test.png')
fig.show()

# %%
# %%

# one plot per groupsplit, only test errors, for molar concentration
df_plot = df_errors[df_errors['conctype'] == 'molar'].copy()
df_plot = df_plot.sort_values(['model', 'chem_fp', 'set'], ascending=[True, True, False])
df_plot_test = df_plot[df_plot['set'] == 'test'].copy()

# TODO run the next three cells for RMSE and R2
metric = 'rmse'
#metric = 'mae'
#metric = 'r2'

# calculate maximum
metric_max, metric_step, str_metric = _calculate_metric_stuff(metric)

for groupsplit in ['totallyrandom', 'occurrence']:
    df_p = df_plot_test[df_plot_test['groupsplit'] == groupsplit]

    g = (ggplot(data=df_p, mapping=aes(x='chem_fp', y=metric, fill='chem_fp'))
        + geom_col() 
        + ylim((0, metric_max))
        + facet_wrap("~ model", ncol=5)
        + scale_fill_manual(values=list_colors)
        + labs(x='', fill='fingerprint', title=groupsplit)
        + geom_hline(yintercept=0.25, color='white', size=0.25)
        + geom_hline(yintercept=0.5, color='white', size=0.25)
        + geom_hline(yintercept=0.75, color='white', size=0.25)
        + geom_hline(yintercept=1., color='white', size=0.25)
        + geom_hline(yintercept=1.25, color='white', size=0.25)
        + geom_hline(yintercept=1.5, color='white', size=0.25)
        + theme_tufte()
        + theme(axis_text_x=element_blank())
        + theme(axis_ticks_major_x=element_blank())
        + labs(y=str_metric)
    )
    g.save(path_figures + '29_' + groupsplit + '_' + metric + '-vs-models.png', facecolor='white')
    print(g)

# %%

# one plot for all splits and 1 fingerprint

for chem_fp in ['MACCS', 'pcp', 'Morgan', 'mol2vec']:

    # corresponding color
    list_cats = list(df_plot_test['chem_fp'].cat.categories)
    fill = list_colors[list_cats.index(chem_fp)]

    # only subset
    df_p = df_plot_test[(df_plot_test['chem_fp'] == chem_fp)]

    g = (ggplot(data=df_p, mapping=aes(x='model', y=metric, alpha='model'))
        + geom_col(fill=fill) 
        + ylim((0, metric_max))
        + facet_wrap("~ groupsplit", ncol=2)
        + scale_alpha_manual(values=[0.4, 0.55, 0.7, 0.85, 1])
        + labs(x='', fill='fingerprint', title=chem_fp)
        + geom_hline(yintercept=0.25, color='white', size=0.25)
        + geom_hline(yintercept=0.5, color='white', size=0.25)
        + geom_hline(yintercept=0.75, color='white', size=0.25)
        + geom_hline(yintercept=1., color='white', size=0.25)
        + geom_hline(yintercept=1.25, color='white', size=0.25)
        + geom_hline(yintercept=1.5, color='white', size=0.25)
        + theme_tufte()
        + theme(axis_text_x=element_blank())
        + theme(axis_ticks_major_x=element_blank())
        + labs(y=str_metric)
    )
    g.save(path_figures + '29_' + chem_fp + '_' + metric + '-vs-models.png', facecolor='white')
    print(g)

# %%

df_p = df_plot_test.copy()
# inverse order for fingerprint
df_p['chem_fp'] = pd.Categorical(df_p['chem_fp'],
                                 categories=['MACCS', 'pcp', 'Morgan', 'mol2vec'][::-1],
                                 ordered=True)

# heatmap for test errors
g = (ggplot(data=df_p, mapping=aes(x='model', y='chem_fp', fill=metric, label=metric))
    + geom_tile()
    + geom_text(format_string='{:.2f}')
    + facet_wrap('~ groupsplit', ncol=1)
    + scale_fill_cmap('cividis_r')
    + labs(x='model', y='fingerprint', fill=metric)
    + theme_tufte()
    + theme(axis_ticks_major=element_blank())
    + theme(figure_size=(7, 12))
)
g.save(path_figures + '29_all_heatmap_' + metric + '.png', facecolor='white')
print(g)

# %%
