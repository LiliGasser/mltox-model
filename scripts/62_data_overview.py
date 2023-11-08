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

#%reload_ext autoreload
#%autoreload 2

# %%

# set paths
path_data = path_root + 'data/'
path_output = path_root + 'output/additional/'

# %%

# load data
df_eco = pd.read_csv(path_data + 'processed/t-F2F_mortality.csv', low_memory=False)

# %%

# concentration in mg/L
g = (ggplot(data=df_eco, mapping=aes(x="result_conc1_mean_log"))
    + geom_histogram(bins=500)
    + theme_minimal()
)
g

# %%

# molar concentration in mol/L
(ggplot(data = df_eco, mapping=aes(x="result_conc1_mean_mol_log"))
    + geom_histogram(bins=500)
    + theme_minimal()
)

# %%

# mass and molar concentration
df_long = pd.melt(df_eco, 
                  id_vars='result_id', 
                  value_vars=['result_conc1_mean_log', 'result_conc1_mean_mol_log'],
                  var_name='concentration_type',
                  value_name='concentration')
df_long['concentration_type'] = df_long['concentration_type'].replace({'result_conc1_mean_log': 'mass concentration', 'result_conc1_mean_mol_log': 'molar concentration'})

g = (ggplot(data=df_long, mapping=aes(x='concentration'))
    + geom_histogram(color='black', fill='black', bins=100)
    + facet_wrap('~ concentration_type', ncol=1)
    + theme_minimal()
    + labs(x='log10-transformed concentration')
 )
g.save(path_output + '62_concentration_histograms.pdf')
g

# %%