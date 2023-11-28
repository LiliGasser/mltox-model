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
df_long['concentration_type'] = df_long['concentration_type'].replace({'result_conc1_mean_log': 'LC50 in $mg/L$', 'result_conc1_mean_mol_log': 'LC50 in $mol/L$'})

g = (ggplot(data=df_long, mapping=aes(x='concentration'))
    + geom_histogram(color='#000', fill='#000', bins=100)
    + facet_wrap('~ concentration_type', ncol=1)
    + theme_classic()
    + theme(figure_size=(6,6))
    + scale_x_continuous(breaks=[-10, -7.5, -5, -2.5, 0, 2.5, 5])
    + theme(axis_text=element_text(size=12, color='black'))
    + theme(axis_title=element_text(size=13, color='black'))
    + labs(x='log10(LC50)')
    + theme(strip_background=element_rect(fill='none', linetype='dashed'))
    + theme(strip_text=element_text(size=12))
    + theme(strip_align='left')
 )
g.save(path_output + '62_concentration_histograms.pdf')
g

# %%