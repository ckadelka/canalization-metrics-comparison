#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import utils_function
import utils_network
import scipy.stats as stats


CURRENT_DIR = os.getcwd()
FIGURE_FOLDER = '../figures/'
TABLE_FOLDER = '../tables/'
OTHER_OUTPUT_FOLDER = '../other/'


# Load data:
n = 4
df = utils_function.get_function_level_results(n)
df = df[df['mean degenerate'] == 0]
network_metrics, _ = utils_network.get_network_level_results(n)

# Settings:
store_as_png = False
store_as_pdf = True

# Merge data:
keys_network_metrics = [(float(a),float(b)) for (a,b) in list(network_metrics[list(network_metrics.keys())[0]].keys())]
keys_df = [(a,b) for a,b in zip(df['canalizing strength'],df['input redundancy'])]
dict_network_metrics_to_pos = dict(zip(keys_network_metrics,range(len(keys_network_metrics))))

pos = []
for key in keys_df:
    pos.append(dict_network_metrics_to_pos[key])
pos = np.array(pos)

for key in network_metrics.keys():
    mean_network_metric = [np.mean(array) 
                           for k, array in network_metrics[key].items()]
    mean_network_metric = np.array(mean_network_metric)
    df[key] = mean_network_metric[pos]
df['new'] = df['mean canalizing depth'] + df['mean excess canalization']

# Modify data:
use_absolute_spearman = True

if not use_absolute_spearman:
    df['1 - mean average sensitivity'] = 1 - df['mean average sensitivity']

metrics = ['canalizing strength', 'input redundancy', ('' if use_absolute_spearman else '1 - ')+'mean average sensitivity','mean canalizing depth','mean excess canalization']
metrics.append('new')
metrics.extend(list(network_metrics.keys()))

n_metrics = len(metrics)
metrics_labels = metrics[:]
for i in range(n_metrics):
    metrics_labels[i] = metrics_labels[i].replace('mean ','')
    if metrics_labels[i] == 'average attractor length':
        metrics_labels[i] = 'mean attractor length'
    if metrics_labels[i] == 'new':
        metrics_labels[i] = 'canalizing depth +\nexcess canalization'
    

res = np.ones((n_metrics,n_metrics))
positive_correlation = np.ones((n_metrics,n_metrics),dtype=bool)
for i in range(n_metrics):
    for j in range(i+1,n_metrics):
        res[i,j] = stats.spearmanr(df[metrics[i]],df[metrics[j]]).statistic
        positive_correlation[i,j] = res[i,j]>=0
        if use_absolute_spearman:
            res[i,j] = abs(res[i,j])
        res[j,i] = res[i,j]
        positive_correlation[j,i] = positive_correlation[i,j]

# Plot:
offset = 4.8
f,ax = plt.subplots()

if use_absolute_spearman:
    im = ax.imshow(res,vmin=0,vmax=1,cmap=cm.hot_r,origin='upper')
else:
    im = ax.imshow(res,vmin=-1,vmax=1,cmap=cm.RdBu,origin='upper')

ax.plot([-0.5-offset,-0.5],[5.5,5.5],'k-',lw=0.5,clip_on=False)
ax.plot([-0.5,n_metrics-0.5],[5.5,5.5],'w-',lw=0.5)
ax.plot([5.5,5.5],[n_metrics-0.5,n_metrics-0.5+offset],'k-',lw=0.5,clip_on=False)
ax.plot([5.5,5.5],[-0.5,n_metrics-0.5],'w-',lw=0.5)
ax.set_xlim([-0.5,n_metrics-0.5])
ax.set_ylim([n_metrics-0.5,-0.5])
ax.set_xticks(range(n_metrics))
ax.set_yticks(range(n_metrics))
ax.set_xticklabels(metrics_labels,rotation=90)
ax.set_yticklabels(metrics_labels)
ax.tick_params(bottom=False, left=False)
for i in  range(len(res)):
    for j in range(len(res)):
        if not positive_correlation[i,j]:
            label = r'$-$'
        else:
            label = ''
        label += ("{:.2f}".format(res[i,j]) if res[i,j]<1 else '1')
        color = 'k' if res[i,j]<0.75 else 'white'
        ax.text(i,j,label,
                va='center',ha='center',
                fontsize=7,color=color)

cbar = f.colorbar(im)
cbar.set_label(('absolute ' if use_absolute_spearman else '')+'Spearman correlation')

if store_as_png:
    plt.savefig(os.path.join(FIGURE_FOLDER, f"metrics_combined_correlation_n{n}.png"), dpi=300, bbox_inches='tight')
if store_as_pdf:
    plt.savefig(os.path.join(FIGURE_FOLDER, f"metrics_combined_correlation_n{n}.pdf"), bbox_inches='tight')

