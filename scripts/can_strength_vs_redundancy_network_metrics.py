import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utils_network as utils


CURRENT_DIR = os.getcwd()
FIGURE_FOLDER = '../figures/'
TABLE_FOLDER = '../tables/'
OTHER_OUTPUT_FOLDER = '../other/'


# Load/generate data:
for n in [4, 5]:
    redundancy_metric = 'input redundancy'
    random_networks = utils.simulate_random_network_dynamics()
    network_metrics, dict_x_groups = utils.get_network_level_results(n, redundancy_metric)

    store_as_png = True
    store_as_pdf = True


    # Figure 5a:
    base = os.path.join(FIGURE_FOLDER, "random_network_dynamics_N12_sims1000")

    for ext, should_store in [("png", store_as_png), ("pdf", store_as_pdf)]:
        if not should_store:
            continue
        filename = f"{base}.{ext}"
        if os.path.isfile(filename):
            print(f"Skipping {filename}, already exists.")
            continue

    ks=np.arange(1, 5)
    absolute_biases=np.linspace(0, 0.9, 10)
    metrics = [
        'Derrida value',
        'number of attractors',
        'average attractor length',
        'basin entropy',
        'coherence']

    # Plot:
    f, ax = plt.subplots(figsize=(4, 10), nrows=len(metrics), sharex='col', sharey='row')

    for i, metric in enumerate(metrics):
        metric_label = metric
        if metric == 'average attractor length':
            metric_label = 'mean attractor length'
        sns.heatmap(
            random_networks[metric].mean(2),
            ax=ax[i],
            xticklabels=np.round(absolute_biases, 2),
            yticklabels=ks,
            annot=False,
            cmap="magma_r",
            cbar_kws={"label": f"{metric_label}"})

        ax[i].invert_yaxis()
        ax[i].set_ylabel("in-degree")

        if i == len(metrics) - 1:
            ax[i].set_xlabel("absolute bias")
        else:
            ax[i].set_xlabel("")
        ax[i].tick_params(bottom=False, left=False)
    
    f.align_ylabels()
    if store_as_png:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"random_network_dynamics_N12_sims1000.png"), dpi=300, bbox_inches='tight')
    if store_as_pdf:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"random_network_dynamics_N12_sims1000.pdf"), bbox_inches='tight')


    # Figure 5b:
    metric_x_labels = ['canalizing strength', redundancy_metric]

    dynamical_metrics = [
            'number of attractors',
            'average attractor length',
            'basin entropy',
            'coherence']
    n_metrics = len(dynamical_metrics)

    cmap1 = plt.get_cmap("tab10")
    cmap2 = plt.get_cmap('Set2')

    f, ax = plt.subplots(figsize=(8,8.6), nrows=n_metrics, ncols=2, sharex='col', sharey='row')

    for ii, metric_y_label in enumerate(dynamical_metrics):
        count = 0
        for jj in range(2):
            sorted_dict = dict(sorted(dict_x_groups[1-jj].items()))

            for group_by_metric, equiv_classes in sorted_dict.items():
                x_values = []
                y_values = []

                for equiv_class in equiv_classes:
                    x_values.append(equiv_class[jj])
                    y_values.append(network_metrics[metric_y_label][equiv_class].mean())

                x_values = np.array(x_values)
                y_values = np.array(y_values)

                color = cmap1(count) if count < 10 else cmap2(count - 10)

                order = np.argsort(x_values)
                x_values = x_values[order]
                y_values = y_values[order]

                ax[ii, jj].scatter(x_values, y_values, alpha=0.3, color=color)

                if len(x_values) > 1:
                    m, b = np.polyfit(x_values, y_values, 1)
                    xfit = np.linspace(x_values.min(), x_values.max(), 2)

                    ax[ii, jj].plot(xfit, m*xfit + b, color=color, label=str(round(group_by_metric, 4)))

                count+=1

            ax[ii,jj].spines[['top','right']].set_visible(False)

            if ii == n_metrics-1:
                ax[ii,jj].set_xlabel(metric_x_labels[jj])

            if jj == 0:
                ax[ii,jj].set_ylabel(metric_y_label if metric_y_label != 'average attractor length' else 'mean attractor length')

            if ii == 0:
                handles, labels = ax[ii,jj].get_legend_handles_labels()

                ax[ii,jj].legend(loc='center', bbox_to_anchor=(0.5,1.35), ncol=3,
                                frameon=False, title=f"fixed {metric_x_labels[1-jj]}")
    f.align_ylabels()
    if store_as_png:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"network_dynamics_strength_{redundancy_metric.replace(' ', '_')}_N12_indegree{n}_sims1000_non_degenerate.png"), dpi=300, bbox_inches='tight')
    if store_as_pdf:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"network_dynamics_strength_{redundancy_metric.replace(' ', '_')}_N12_indegree{n}_sims1000_non_degenerate.png"), bbox_inches='tight')

