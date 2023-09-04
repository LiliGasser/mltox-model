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

import pickle
import shap

from plotnine import *

import utils

#%reload_ext autoreload
#%autoreload 2

# %%

# load data
path_data = path_root + 'data/'
path_output = path_root + 'output/regression/'


# %%

# specify fingerprint and groupsplit
chem_fp = 'Morgan'
groupsplit = 'occurrence'

# load explainer and shap values
filename_ending = '_'.join(('rf', 'explainer', chem_fp, groupsplit)) + '.sav'
filename_expl = path_output + filename_ending
filename_ending = '_'.join(('rf', 'shapvalues', chem_fp, groupsplit)) + '.sav'
filename_sv = path_output + filename_ending

explainer = pickle.load(open(filename_expl, 'rb'))
shap_values = pickle.load(open(filename_sv, 'rb'))

# load corresponding data (here test data)
filename_ending = '_'.join(('rf', 'data', chem_fp, groupsplit)) + '.csv'
filename_data = path_output + filename_ending
df_eco_test = pd.read_csv(filename_data)

# %%


# Plotting remarks

# The default colormap should not be used as it is not colorblind safe.

# %%

# Plots for entire test set

# bar plot
# ! passing feature names not possible
shap.plots.bar(shap_values)

# %%

# beeswarm
shap.summary_plot(shap_values, 
                  cmap='cividis', 
                  alpha=0.4,
                  )

# %%

# beeswarm option 2
# ! passing feature names not possible
shap.plots.beeswarm(shap_values,
                    max_display=10, 
                    )

# %%

# violin
# ! passing color map not possible
shap.summary_plot(shap_values, 
                  alpha=0.4,
                  plot_type='violin'
                  )

# %%
# layered violin
# ! passing color map not possible
shap.summary_plot(shap_values, 
                  alpha=0.4,
                  plot_type='layered_violin'
                  )

# %%

# single data points

shap.plots.bar(shap_values[100])
shap.plots.waterfall(shap_values[100])
# %%

shap.plots.bar(shap_values[:1000])
# %%

df_eco_test['chem_name'].value_counts()

# %%

#chem_name = 'Malathion'
#chem_name = 'Endosulfan'
chem_name = 'Chlorpyrifos'
list_idx = list(df_eco_test[df_eco_test['chem_name'] == chem_name].index)
shap_values[list_idx]

# bar chart
shap.plots.bar(shap_values[list_idx],
               max_display=11)

# beeswarm
shap.summary_plot(shap_values[list_idx], 
                  max_display=10,
                  cmap='cividis', 
                  alpha=0.4,
                  )
# %%
