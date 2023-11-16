
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

def compile_errors(modeltype, load_top3=False):

    # load error files
    df_cv = pd.read_csv(path_output + modeltype + '_CV-errors.csv')
    df_test = pd.read_csv(path_output + modeltype + '_test-errors.csv')

    # load top3 features error files
    if load_top3:
        df_cv_top3 = pd.read_csv(path_output + modeltype + '_CV-errors_top3features.csv')
        df_cv_top3['chem_fp'] = 'top 3'
        df_test_top3 = pd.read_csv(path_output + modeltype + '_test-errors_top3features.csv')
        df_test_top3['chem_fp'] = 'top 3'
    else:
        df_cv_top3 = pd.DataFrame()
        df_test_top3 = pd.DataFrame()

    # concatenate
    df = pd.concat((df_cv, df_test, df_cv_top3, df_test_top3)).reset_index(drop=True)
    df['chem_fp'] = df['chem_fp'].str.replace('pcp', 'PubChem')

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
# 7 fps x 2 groupsplits x 4 sets x 2 concentrations = 112 entries
# + 1 fps x 2 groupsplits x 4 sets x 2 concentrations = 16 entries (top3 features)
# total: 128 entries
df_lasso = compile_errors(modeltype='lasso', load_top3=True)
df_lasso

# %%

# load RF
# 7 fps x 2 groupsplits x 4 sets x 2 concentrations = 112 entries
# + 1 fps x 2 groupsplits x 4 sets x 2 concentrations = 16 entries (top3 features)
# total: 128 entries
df_rf = compile_errors(modeltype='rf', load_top3=True)
df_rf

# %%

# load XGBoost
# 7 fps x 2 groupsplits x 4 sets x 2 concentrations = 112 entries
# + 1 fps x 2 groupsplits x 4 sets x 2 concentrations = 16 entries (top3 features)
# total: 128 entries
df_xgboost = compile_errors(modeltype='xgboost', load_top3=True)
df_xgboost

# %%

# load GP
# 7 fps x 2 groupsplits x 4 sets x 2 tax_pdm x 2 concentrations = 224 entries
# + 1 fps x 2 groupsplits x 4 sets x 2 concentrations = 16 entries (top3 features)
# total: 240 entries
df_gp = compile_errors(modeltype='gp', load_top3=True)
df_gp

# %%

# GP: without tax_pdm
df_gp_all = df_gp.copy()
df_gp = df_gp[df_gp['tax_pdm'] == 'none'].copy()

# %%

# concatenate all error files
df_errors = pd.concat([df_lasso, df_rf, df_xgboost, df_gp], axis=0)
df_errors

# %%

# only two group splits
list_cols = ['totallyrandom', 'occurrence']
df_errors = df_errors[df_errors['groupsplit'].isin(list_cols)].copy()
df_gp_all = df_gp_all[df_gp_all['groupsplit'].isin(list_cols)].copy()

# categorical variables
# the fingerprint 'none' corresponds to the top 3 features models
list_cols_fps = ['MACCS', 'PubChem', 'Morgan', 'ToxPrint', 'mol2vec', 'Mordred', 'none']
list_cols_fps_none = list_cols_fps + ['top 3']
df_errors = utils._transform_to_categorical(df_errors, 'groupsplit', ['totallyrandom', 'occurrence'])
df_errors = utils._transform_to_categorical(df_errors, 'chem_fp', list_cols_fps_none)
df_errors = utils._transform_to_categorical(df_errors, 'model', ['LASSO', 'RF', 'XGBoost', 'GP'])
df_errors = utils._transform_to_categorical(df_errors, 'set', ['train', 'valid', 'trainvalid', 'test'])
df_errors = utils._transform_to_categorical(df_errors, 'conctype', ['molar', 'mass'])
df_gp_all = utils._transform_to_categorical(df_gp_all, 'groupsplit', ['totallyrandom', 'occurrence'])
df_gp_all = utils._transform_to_categorical(df_gp_all, 'chem_fp', list_cols_fps)
df_gp_all = utils._transform_to_categorical(df_gp_all, 'tax_pdm', ['none', 'pdm'])
df_gp_all = utils._transform_to_categorical(df_gp_all, 'set', ['train', 'valid', 'trainvalid', 'test'])
df_gp_all = utils._transform_to_categorical(df_gp_all, 'conctype', ['molar', 'mass'])

# %%

# best hyperparameters for LASSO
df_lasso = utils._transform_to_categorical(df_lasso, 'conctype', ['molar', 'mass'])
df_lasso = utils._transform_to_categorical(df_lasso, 'groupsplit', ['totallyrandom', 'occurrence'])
df_lasso = utils._transform_to_categorical(df_lasso, 'chem_fp', list_cols_fps_none)

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
df_rf = utils._transform_to_categorical(df_rf, 'chem_fp', list_cols_fps_none)

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
df_xgboost = utils._transform_to_categorical(df_xgboost, 'chem_fp', list_cols_fps_none)

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
df_gp = utils._transform_to_categorical(df_gp, 'chem_fp', list_cols_fps_none)

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
# and purple from https://www.pinterest.ch/pin/1130403575204052700/
# and yellow from https://www.pinterest.ch/pin/57632070223416732/
list_colors = ['#75aab9', '#dfc85e', '#998478', '#c194ac', '#80a58b', '#fbba76', '#999', '#ddd']

# for errors
list_colors_points = ['#ccc', '#444', '#999', 'black']

# assign colors
dict_colors_fps = dict(zip(list_cols_fps, list_colors))
dict_colors_fps_none = dict(zip(list_cols_fps_none, list_colors))
list_sets = ['train', 'valid', 'trainvalid', 'test']
dict_colors_points = dict(zip(list_sets, list_colors_points))

# function to set maximum values, etc. in plot
def _calculate_metric_stuff(df_errors, metric):

    if metric == 'r2':
        metric_max = 1.01
        metric_step = 0.2
        str_metric = r'R$^2$'
    elif metric == 'rmse':
        metric_max = df_errors[metric].max() + 0.1
        metric_step = 0.25
        str_metric = 'RMSE'
    elif metric == 'mae':
        metric_max = df_errors[metric].max() + 0.1
        metric_step = 0.25
        str_metric = 'MAE'

    return metric_max, metric_step, str_metric

# store images flag
do_store_images = False

# %%

# overview plot in plotly (test error)
# !!! test error

# TODO run for RMSE, MAE and R2
metric = 'rmse'
#metric = 'mae'
#metric = 'r2'

# calculate maximum
metric_max, metric_step, str_metric = _calculate_metric_stuff(df_errors, metric)
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
df_plot_test = df_plot[(df_plot['set'] == 'test')].copy()
row = 1
col = 1
for i, chem_fp in enumerate(list_cols_fps_none):     # add bars
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_test[df_plot_test['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps_none),
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
for i, chem_fp in enumerate(list_cols_fps_none):    # add points
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
df_plot_test = df_plot[(df_plot['set'] == 'test')].copy()
row = 1
col = 2
for i, chem_fp in enumerate(list_cols_fps_none):    # add points
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_test[df_plot_test['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps_none),
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
for i, chem_fp in enumerate(list_cols_fps_none):    # add points
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
df_plot_test = df_plot[(df_plot['set'] == 'test')].copy()
row = 2
col = 1
for i, chem_fp in enumerate(list_cols_fps_none):    # add points
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_test[df_plot_test['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps_none),
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
for i, chem_fp in enumerate(list_cols_fps_none):    # add points
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
df_plot_test = df_plot[(df_plot['set'] == 'test')].copy()
row = 2
col = 2
for i, chem_fp in enumerate(list_cols_fps_none):     # add bars
    df_p = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    df_pt = df_plot_test[df_plot_test['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps_none),
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
for i, chem_fp in enumerate(list_cols_fps_none):    # add points
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
fig.update_layout(height=700, width=1000)

# add legend for fingerprints
for chem_fp in list_cols_fps_none:
    fig.add_trace(go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             name=chem_fp,
                             legendgroup='chem_fp',
                             legendgrouptitle_text='molecular<br>representation',
                             marker_color=dict_colors_fps_none[chem_fp],
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

if do_store_images:
    fig.write_image(path_figures + '46_all_' + metric + '-vs-models.pdf')
fig.show()

# %%
# %%

# overview plot in plotly (validation error)
# !!! validation error

# TODO run for RMSE, MAE and R2
metric = 'rmse'
#metric = 'mae'
#metric = 'r2'

# calculate maximum
metric_max, metric_step, str_metric = _calculate_metric_stuff(df_errors, metric)
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
                    & (df_errors['set'] == 'valid')].copy()
row = 1
col = 1
for i, chem_fp in enumerate(list_cols_fps_none):     # add bars
    df_pt = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps_none),
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

# totallyrandom, molar
df_plot = df_errors[(df_errors['groupsplit'] == 'totallyrandom')
                    & (df_errors['conctype'] == 'molar')
                    & (df_errors['set'] == 'valid')].copy()
row = 1
col = 2
for i, chem_fp in enumerate(list_cols_fps_none):     # add bars
    df_pt = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps_none),
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

# occurrence, mass
df_plot = df_errors[(df_errors['groupsplit'] == 'occurrence')
                    & (df_errors['conctype'] == 'mass')
                    & (df_errors['set'] == 'valid')].copy()
row = 2
col = 1
for i, chem_fp in enumerate(list_cols_fps_none):     # add bars
    df_pt = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps_none),
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

# occurrence, molar
df_plot = df_errors[(df_errors['groupsplit'] == 'occurrence')
                    & (df_errors['conctype'] == 'molar')
                    & (df_errors['set'] == 'valid')].copy()
row = 2
col = 2
for i, chem_fp in enumerate(list_cols_fps_none):     # add bars
    df_pt = df_plot[df_plot['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pt['model'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps_none),
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
fig.update_layout(height=700, width=1000)

# add legend for fingerprints
for chem_fp in list_cols_fps_none:
    fig.add_trace(go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             name=chem_fp,
                             #legendgroup='chem_fp',
                             legendgrouptitle_text='molecular representation',
                             marker_color=dict_colors_fps_none[chem_fp],
                             marker_symbol='square',
                             marker_size=12))
fig.update_layout(legend_orientation='h', legend_xanchor='center', legend_x=0.5)

fig.update_layout(template='plotly_white')

if do_store_images:
    fig.write_image(path_figures + '46_all_' + metric + '-vs-models_validation.pdf')
fig.show()

# %% 
# %%

# overview plot in plotly (only molar occurrence)
# !!! test error only for occurrence and molar

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
metric_max, metric_step, str_metric = _calculate_metric_stuff(df_errors, metric)
list_tickvals = list(np.arange(0, metric_max, metric_step))
df_plot = df_errors[(df_errors['groupsplit'] == 'occurrence')
                    & (df_errors['conctype'] == 'molar')].copy()
df_plot_bar = df_plot[(df_plot['set'] == 'test')].copy()
df_plot_dot = df_plot[(df_plot['set'] == 'valid')].copy()
row = 1
col = 1
for i, chem_fp in enumerate(list_cols_fps_none):     # add bars
    df_pb = df_plot_bar[df_plot_bar['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pb['model'], 
                         y=df_pb[metric],
                         marker_color=df_pb['chem_fp'].map(dict_colors_fps_none),
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
for i, chem_fp in enumerate(list_cols_fps_none):    # add points
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
metric_max, metric_step, str_metric = _calculate_metric_stuff(df_errors, metric)
list_tickvals = list(np.arange(0, metric_max, metric_step))
df_plot = df_errors[(df_errors['groupsplit'] == 'occurrence')
                    & (df_errors['conctype'] == 'molar')].copy()
df_plot_bar = df_plot[(df_plot['set'] == 'test')].copy()
df_plot_dot = df_plot[(df_plot['set'] == 'valid')].copy()
row = 2
col = 1
for i, chem_fp in enumerate(list_cols_fps_none):     # add bars
    df_pb = df_plot_bar[df_plot_bar['chem_fp'] == chem_fp].copy()
    fig.add_trace(go.Bar(x=df_pb['model'], 
                         y=df_pb[metric],
                         marker_color=df_pb['chem_fp'].map(dict_colors_fps_none),
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
for i, chem_fp in enumerate(list_cols_fps_none):    # add points
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
for chem_fp in list_cols_fps_none:
    fig.add_trace(go.Scatter(x=[None],
                             y=[None],
                             mode='markers',
                             name=chem_fp,
                             legendgroup='chem_fp',
                             legendgrouptitle_text='molecular<br>representation',
                             marker_color=dict_colors_fps_none[chem_fp],
                             marker_symbol='square',
                             marker_size=12))
#for errortype in ['valid']:
    #fig.add_trace(go.Scatter(x=[None],
                             #y=[None],
                             #mode='markers',
                             #name=errortype,
                             #legendgroup='set',
                             #legendgrouptitle_text='error type',
                             #marker_color=dict_colors_points[errortype],
                             #marker_symbol='diamond-open',
                             #marker_size=7))
# TODO legend at bottom horizontally aligned
#fig.update_layout(legend_orientation='h', legend_xanchor='center', legend_x=0.5)

fig.update_layout(template='plotly_white')

if do_store_images:
    fig.write_image(path_figures + '46_all_rmse-r2-vs-models_test.pdf')
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
metric_max, metric_step, str_metric = _calculate_metric_stuff(df_errors, metric)

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

    if do_store_images:
        g.save(path_figures + '46_' + groupsplit + '_' + metric + '-vs-models.pdf', facecolor='white')
    print(g)

# %%

# one plot for all splits and 1 fingerprint

for chem_fp in list_cols_fps:

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
    
    if do_store_images:
        g.save(path_figures + '46_' + chem_fp + '_' + metric + '-vs-models.pdf', facecolor='white')
    print(g)

# %%

# heatmap for test errors

df_p = df_plot_test.copy()

# inverse order for fingerprint
df_p['chem_fp'] = pd.Categorical(df_p['chem_fp'],
                                 categories=list_cols_fps_none[::-1],
                                 ordered=True)

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

if do_store_images:
    g.save(path_figures + '46_all_heatmap_' + metric + '.pdf', facecolor='white')
print(g)

# %%

# compare GP runs in plotly (validation error)
# !!! validation error

# TODO run for RMSE, MAE and R2
metric = 'rmse'
#metric = 'mae'
#metric = 'r2'

# calculate maximum
metric_max, metric_step, str_metric = _calculate_metric_stuff(df_gp_all, metric)
list_tickvals = list(np.arange(0, metric_max, metric_step))

# pattern for pdm
# ['', '/', '\\', 'x', '-', '|', '+', '.']
dict_patterns_pdm = {}
dict_patterns_pdm['none'] = ''
dict_patterns_pdm['pdm'] = '/'

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
df_plot = df_gp_all[(df_gp_all['groupsplit'] == 'totallyrandom')
                    & (df_gp_all['conctype'] == 'mass')
                    & (df_gp_all['set'] == 'valid')].copy()
row = 1
col = 1
for i, tax_pdm in enumerate(['none', 'pdm']):     # add bars
    df_pt = df_plot[df_plot['tax_pdm'] == tax_pdm].copy()
    fig.add_trace(go.Bar(x=df_pt['chem_fp'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         marker_pattern_shape=dict_patterns_pdm[tax_pdm],
                         name=tax_pdm,
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

# totallyrandom, molar
df_plot = df_gp_all[(df_gp_all['groupsplit'] == 'totallyrandom')
                    & (df_gp_all['conctype'] == 'molar')
                    & (df_gp_all['set'] == 'valid')].copy()
row = 1
col = 2
for i, tax_pdm in enumerate(['none', 'pdm']):     # add bars
    df_pt = df_plot[df_plot['tax_pdm'] == tax_pdm].copy()
    fig.add_trace(go.Bar(x=df_pt['chem_fp'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         marker_pattern_shape=dict_patterns_pdm[tax_pdm],
                         name=tax_pdm,
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

# occurrence, mass
df_plot = df_gp_all[(df_gp_all['groupsplit'] == 'occurrence')
                    & (df_gp_all['conctype'] == 'mass')
                    & (df_gp_all['set'] == 'valid')].copy()
row = 2
col = 1
for i, tax_pdm in enumerate(['none', 'pdm']):     # add bars
    df_pt = df_plot[df_plot['tax_pdm'] == tax_pdm].copy()
    fig.add_trace(go.Bar(x=df_pt['chem_fp'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         marker_pattern_shape=dict_patterns_pdm[tax_pdm],
                         name=tax_pdm,
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

# occurrence, molar
df_plot = df_gp_all[(df_gp_all['groupsplit'] == 'occurrence')
                    & (df_gp_all['conctype'] == 'molar')
                    & (df_gp_all['set'] == 'valid')].copy()
row = 2
col = 2
for i, tax_pdm in enumerate(['none', 'pdm']):     # add bars
    df_pt = df_plot[df_plot['tax_pdm'] == tax_pdm].copy()
    fig.add_trace(go.Bar(x=df_pt['chem_fp'], 
                         y=df_pt[metric],
                         marker_color=df_pt['chem_fp'].map(dict_colors_fps),
                         marker_pattern_shape=dict_patterns_pdm[tax_pdm],
                         name=tax_pdm,
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
fig.update_layout(height=700, width=1000)

# add legend for tax_pdm
for tax_pdm in ['none', 'pdm']:
    if tax_pdm == 'none':
        str_tax_pdm = 'no'
    else:
        str_tax_pdm = 'yes'
    fig.add_trace(go.Bar(x=[None],
                         y=[None],
                         name=str_tax_pdm,
                         legendgrouptitle_text='pairwise distance matrix included',
                         marker_color='lightgrey',
                         marker_pattern_shape=dict_patterns_pdm[tax_pdm])
                         )
fig.update_layout(legend_orientation='h', legend_xanchor='center', legend_x=0.5)

fig.update_layout(template='plotly_white')

if do_store_images:
    fig.write_image(path_figures + '46_GP_' + metric + '-vs-taxpdm_validation.pdf')
fig.show()

# %%

# compare GP runs in plotly (only molar occurrence)
# !!! test error only for occurrence and molar

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
metric_max, metric_step, str_metric = _calculate_metric_stuff(df_gp_all, metric)
list_tickvals = list(np.arange(0, metric_max, metric_step))
df_plot = df_gp_all[(df_gp_all['groupsplit'] == 'occurrence')
                    & (df_gp_all['conctype'] == 'molar')].copy()
df_plot_bar = df_plot[(df_plot['set'] == 'test')].copy()
df_plot_dot = df_plot[(df_plot['set'] == 'valid')].copy()
row = 1
col = 1
for i, tax_pdm in enumerate(['none', 'pdm']):     # add bars
    df_pb = df_plot_bar[df_plot_bar['tax_pdm'] == tax_pdm].copy()
    fig.add_trace(go.Bar(x=df_pb['chem_fp'], 
                         y=df_pb[metric],
                         marker_color=df_pb['chem_fp'].map(dict_colors_fps),
                         marker_pattern_shape=dict_patterns_pdm[tax_pdm],
                         name=tax_pdm,
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
#for i, chem_fp in enumerate(list_cols_fps_none):    # add points
    #df_pt = df_plot_dot[df_plot_dot['chem_fp'] == chem_fp].copy()
    #fig.add_trace(go.Scatter(x=df_pt['model'],
                             #y=df_pt[metric],
                             #mode='markers',
                             #marker_color=df_pt['set'].map(dict_colors_points),
                             #marker_symbol='diamond-open',
                             #marker_size=7,
                             #name=chem_fp,
                             #offsetgroup=i+1,
                             #showlegend=False),
                  #row=row, 
                  #col=col)

# update axes
fig.update_xaxes(title_text='', row=row, col=col)
fig.update_yaxes(title_text=str_metric, row=row, col=col)
fig.update_yaxes(range=[0, metric_max], tickvals=list_tickvals, row=row, col=col)

# occurrence, molar, R2
metric='r2'
metric_max, metric_step, str_metric = _calculate_metric_stuff(df_errors, metric)
list_tickvals = list(np.arange(0, metric_max, metric_step))
df_plot = df_gp_all[(df_gp_all['groupsplit'] == 'occurrence')
                    & (df_gp_all['conctype'] == 'molar')].copy()
df_plot_bar = df_plot[(df_plot['set'] == 'test')].copy()
df_plot_dot = df_plot[(df_plot['set'] == 'valid')].copy()
row = 2
col = 1
for i, tax_pdm in enumerate(['none', 'pdm']):     # add bars
    df_pb = df_plot_bar[df_plot_bar['tax_pdm'] == tax_pdm].copy()
    fig.add_trace(go.Bar(x=df_pb['chem_fp'], 
                         y=df_pb[metric],
                         marker_color=df_pb['chem_fp'].map(dict_colors_fps),
                         marker_pattern_shape=dict_patterns_pdm[tax_pdm],
                         name=tax_pdm,
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
#for i, chem_fp in enumerate(list_cols_fps_none):    # add points
    #df_pt = df_plot_dot[df_plot_dot['chem_fp'] == chem_fp].copy()
    #fig.add_trace(go.Scatter(x=df_pt['model'],
                             #y=df_pt[metric],
                             #mode='markers',
                             #marker_color=df_pt['set'].map(dict_colors_points),
                             #marker_symbol='diamond-open',
                             #marker_size=7,
                             #name=chem_fp,
                             #offsetgroup=i+1,
                             #showlegend=False),
                  #row=row, 
                  #col=col)

# update axes
fig.update_xaxes(title_text='', row=row, col=col)
fig.update_yaxes(title_text=str_metric, row=row, col=col)
fig.update_yaxes(range=[0, metric_max], tickvals=list_tickvals, row=row, col=col)

# Grouped bars and scatter
#fig.update_layout(barmode='group')
fig.update_layout(scattermode='group')

# Update title and height
fig.update_layout(height=600, width=500)

#for errortype in ['valid']:
    #fig.add_trace(go.Scatter(x=[None],
                             #y=[None],
                             #mode='markers',
                             #name=errortype,
                             #legendgroup='set',
                             #legendgrouptitle_text='error type',
                             #marker_color=dict_colors_points[errortype],
                             #marker_symbol='diamond-open',
                             #marker_size=7))
# add legend for tax_pdm
for tax_pdm in ['none', 'pdm']:
    if tax_pdm == 'none':
        str_tax_pdm = 'no'
    else:
        str_tax_pdm = 'yes'
    fig.add_trace(go.Bar(x=[None],
                         y=[None],
                         name=str_tax_pdm,
                         legendgrouptitle_text='pairwise distance matrix included',
                         marker_color='lightgrey',
                         marker_pattern_shape=dict_patterns_pdm[tax_pdm])
                         )
fig.update_layout(legend_orientation='h', legend_xanchor='center', legend_x=0.5)

fig.update_layout(template='plotly_white')

if do_store_images:
    fig.write_image(path_figures + '46_GP_rmse-r2-vs-taxpdm_test.pdf')
fig.show()

# %%