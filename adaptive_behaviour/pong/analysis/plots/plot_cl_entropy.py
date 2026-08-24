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
DATA = paths.RESULTS_CL

df99 = pd.DataFrame()

new_data = pd.read_csv(DATA / 'c_data_M1.csv')
new_data['group_name'] = 'CFL-1'
new_data['group'] = 0
df99 = pd.concat([df99, new_data])

new_data = pd.read_csv(DATA / 'c_data_M2.csv')
new_data['group_name'] = 'CFL-2'
new_data['group'] = 1
df99 = pd.concat([df99, new_data])

new_data = pd.read_csv(DATA / 'c_data_M3.csv')
new_data['group_name'] = 'CFL-3'
new_data['group'] = 2
df99 = pd.concat([df99, new_data])

new_data = pd.read_csv(DATA / 'c_data_M4.csv')
new_data['group_name'] = 'CFL-4'
new_data['group'] = 3
df99 = pd.concat([df99, new_data])


# --- normalisation against the analytic maximum (revision, R2 C4) ---
# Previously each entropy trace was divided by the largest value it happened to attain, which set
# every agent to 1.0 at its own least-informed point and made the panels incomparable across agents.
# The maximum entropy of each parameter is known in closed form -- it is attained when every
# distribution the parameter contains is uniform -- so we divide by that constant instead. A value
# of 1.0 now means "carries no information", and the same number means the same thing everywhere.
import numpy as _np
_CL_MAX = 2432 * _np.log(3)          # 2432 states, one distribution over 3 actions each
df99['entropy_1'] = df99['entropy_1'] / _CL_MAX

#%%Gamma box-plot
plt.clf()

df_test2 = df99

labels = df_test2.group_name.unique()
x_pos = np.arange(len(labels))

x = df_test2['group_name']
y = df_test2['entropy_1']

hue = df_test2['half']

sns.set(style="darkgrid")
sns.set(font_scale=1.4)

ax = sns.boxplot(data=df99, x=x, y=y, hue=hue, palette="Set2", showfliers=False,
                 showmeans = True,
                 meanprops= {"markerfacecolor":"black",
                       "markeredgecolor":"black",
                      "markersize":"5"}, linewidth=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=16)
ax.set_ylabel('NTE of $\mathbb{CL}$ mapping',fontsize = 14)
ax.set_xlabel('Group',fontsize = 12)
ax.grid(False)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles[:2], ["0-5", "6-20"], title="Minutes",
          loc="best", fontsize=14, title_fontsize=14)


# The CL mapping stays close to uniform throughout, so the variation of interest occupies a narrow
# band. Zoom the axis to make the ordering by memory horizon legible, and draw the uniform baseline
# so the zoom cannot overstate how far the mapping actually moves.
ax.axhline(1.0, ls='--', lw=1, color='#A64B2A', zorder=0)
ax.annotate('uniform (uninformed)', xy=(0.02, 1.0), xycoords=('axes fraction', 'data'),
            va='bottom', ha='left', fontsize=10, color='#A64B2A')
ax.set_ylim(min(float(df99['entropy_1'].min()) - 0.004, 0.985), 1.004)

plt.savefig(DATA / 'entropy_cl_boxplot.png', dpi = 500,bbox_inches='tight')
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
    py = control[control['elapse_minute_rounded']<21]['entropy_1']
    
    sns.regplot(x=ax.xaxis.convert_units(px), y=py, x_estimator=np.mean, ci=95, seed=0, 
            scatter = False, order = 1, label = label, fit_reg=True)

sns.set(style="darkgrid")

ax.set_ylabel('NTE of $\mathbb{CL}$ mapping',fontsize = 20)
ax.set_xlabel('Elapsed Minute',fontsize =20)
plt.legend(loc='upper left',fontsize =14)


# The CL mapping stays close to uniform throughout, so the variation of interest occupies a narrow
# band. Zoom the axis to make the ordering by memory horizon legible, and draw the uniform baseline
# so the zoom cannot overstate how far the mapping actually moves.
ax.axhline(1.0, ls='--', lw=1, color='#A64B2A', zorder=0)
ax.annotate('uniform (uninformed)', xy=(0.02, 1.0), xycoords=('axes fraction', 'data'),
            va='bottom', ha='left', fontsize=10, color='#A64B2A')
ax.set_ylim(min(float(df99['entropy_1'].min()) - 0.004, 0.985), 1.004)

plt.savefig(DATA / 'entropy_cl_regression.png',dpi = 500, bbox_inches='tight')
plt.show()
