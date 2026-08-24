#!/usr/bin/env python
# coding: utf-8

#%% Import all modules. Set path for importing dishpill_models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = 150

sns.set()
pd.set_option('display.max_rows', 250)
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings('ignore')
from adaptive_behaviour import paths

#: Directory this script reads its inputs from and writes its figures to.
DATA = paths.RESULTS_SI

df99 = pd.DataFrame()

new_data = pd.read_csv(DATA / 'c_data.csv')
df99 = pd.concat([df99, new_data])

# --- normalisation against the analytic maximum (revision, R2 C4) ---
# Previously each entropy trace was divided by the largest value it happened to attain, which set
# every agent to 1.0 at its own least-informed point and made the panels incomparable across agents.
# The maximum entropy of each parameter is known in closed form -- it is attained when every
# distribution the parameter contains is uniform -- so we divide by that constant instead. A value
# of 1.0 now means "carries no information", and the same number means the same thing everywhere.
import numpy as _np
_MAXIMA = {'entropy_1': _np.log(38),            # one distribution over each modality
           'entropy_2': _np.log(8),
           'entropy_3': _np.log(8)}
for column in ['entropy_1', 'entropy_2', 'entropy_3']:
    df99[column] = df99[column] / _MAXIMA[column]

e1 = df99[['entropy_1', 'session_num',
       'elapse_minute_rounded', 'half']]
e1.rename(columns = {'entropy_1':'entropy'}, inplace = True)
e1['group'] = 0
e1['group_name'] = 'ball-x'

e2 = df99[['entropy_2', 'session_num',
       'elapse_minute_rounded', 'half']]
e2.rename(columns = {'entropy_2':'entropy'}, inplace = True)
e2['group'] = 1
e2['group_name'] = 'ball-y'

e3 = df99[['entropy_3', 'session_num',
       'elapse_minute_rounded', 'half']]
e3.rename(columns = {'entropy_3':'entropy'}, inplace = True)
e3['group'] = 2
e3['group_name'] = 'paddle-y'

df99 = pd.concat([e1,e2,e3])

#%%Gamma box-plot
plt.clf()

df_test2 = df99

labels = df_test2.group_name.unique()
x_pos = np.arange(len(labels))

x = df_test2['group_name']
y = df_test2['entropy']

hue = df_test2['half']

sns.set(style="darkgrid")
sns.set(font_scale=1.4)

ax = sns.boxplot(data=df99, x=x, y=y, hue=hue, palette="Set2", 
                 showfliers=False,
                 showmeans = True,
                 meanprops= {"markerfacecolor":"black",
                       "markeredgecolor":"black",
                      "markersize":"5"})

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=16)
ax.set_ylabel('NTE of prior preference $\mathbb{C}$ (AIF-1)',
              fontsize = 14)
ax.set_xlabel('Observation Modality',fontsize = 14)
ax.grid(False)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles[:2], ["0-5", "6-20"], title="Minutes",
          loc="best", fontsize=14, title_fontsize=14)


# Absolute scale, with the uniform baseline marked: 1.0 means the parameter carries no information.
ax.axhline(1.0, ls='--', lw=1, color='#A64B2A', zorder=0)
ax.annotate('uniform (uninformed)', xy=(0.02, 1.0), xycoords=('axes fraction', 'data'),
            va='bottom', ha='left', fontsize=10, color='#A64B2A')
ax.set_ylim(0, 1.06)

plt.savefig(DATA / 'entropy_c_si_boxplot.png',dpi = 500, bbox_inches='tight')
plt.show()

#%% Gamma Reg plot

plt.clf()

df2 = df99.groupby(['group',
                   'session_num',
                   'group_name',
                   'elapse_minute_rounded']).mean(numeric_only = True)

lines = df2.reset_index()

sns.set(font_scale=1.4)
f, ax = plt.subplots(figsize=(7,7))

for i in lines.group.unique():
    
    control = lines[lines['group'] == i]
    label = control.group_name.unique()[0]
    
    px = control[control['elapse_minute_rounded']<21]['elapse_minute_rounded']
    py = control[control['elapse_minute_rounded']<21]['entropy']
    
    sns.regplot(x=ax.xaxis.convert_units(px), y=py, x_estimator=np.mean, 
                ci=95, seed=0, scatter = False, order = 1, label = label, fit_reg=True)

sns.set(style="darkgrid")

ax.set_ylabel('NTE of prior preference $\mathbb{C}$ (AIF-1)',
              fontsize =20)
ax.set_xlabel('Elapsed Minute',fontsize =20)
plt.legend(loc='upper left',fontsize =14)


# Absolute scale, with the uniform baseline marked: 1.0 means the parameter carries no information.
ax.axhline(1.0, ls='--', lw=1, color='#A64B2A', zorder=0)
ax.annotate('uniform (uninformed)', xy=(0.02, 1.0), xycoords=('axes fraction', 'data'),
            va='bottom', ha='left', fontsize=10, color='#A64B2A')
ax.set_ylim(0, 1.06)

plt.savefig(DATA / 'entropy_c_si_regression.png',dpi = 500, bbox_inches='tight')
plt.show()
