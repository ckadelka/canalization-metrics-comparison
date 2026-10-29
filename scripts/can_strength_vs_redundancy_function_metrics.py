#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import utils_function 
import utils_network


CURRENT_DIR = os.getcwd()
FIGURE_FOLDER = '../figures/'
TABLE_FOLDER = '../tables/'
OTHER_OUTPUT_FOLDER = '../other/'


for n in [4,5]:
    # Load data:
    df = utils_function.get_function_level_results(n)
    classifiers_auc_df = utils_function.run_canalization_classifiers(df, n)
    per_var_df = utils_function.get_metrics_per_variable(n, allow_degeneracy = False, keeping_unique_values = True)

    # Settings:
    show_dots_in_fig2 = True if n <= 5 else False
    allow_degeneracy_in_fig4 = False
    legend_outside = False
    store_as_png = True
    store_as_pdf = True
    
    # Figure 1:
    # Plot:
    fig, ax = plt.subplots(1, 2, sharey='col', figsize=(6.5, 2.5))
    ax0, ax1 = ax[0], ax[1]
    
    ax0.spines[['top', 'right']].set_visible(False)
    ax1.tick_params(labelleft=False)
    
    x = df['canalizing strength']
    y = df['input redundancy']
    
    sc = ax0.scatter(x, y, c=df['mean average sensitivity'], cmap='rainbow',
                     s=40, edgecolor='none' if n > 4 else 'k', linewidth=0.3)
    
    ax0.set_xlabel('canalizing strength')
    ax0.set_ylabel('input redundancy')
    
    cax = inset_axes(ax0, width="55%", height="5%", loc='center',
                     bbox_to_anchor=(0.15, -0.2, 1,1), bbox_transform=ax0.transAxes, borderpad=0)
    cbar = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cbar.set_label('average sensitivity', labelpad=5)
    
    degenerate = df['mean degenerate'].astype(int)
    
    ax1.scatter(x[degenerate == 0], y[degenerate == 0], color='tab:blue',
                label='non-degenerate', s=40, edgecolor='k', linewidth=0.3)
    ax1.scatter(x[degenerate == 1], y[degenerate == 1], color='tab:red',
                label='degenerate', s=40, edgecolor='k', linewidth=0.3)
    
    ax1.set_xlabel('canalizing strength')
    ax1.legend(loc='lower right', frameon=False)
    ax1.spines[['top', 'right']].set_visible(False)
    
    fig.subplots_adjust(wspace=0.05) 
    
    if store_as_png:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"avg_sensitivity_degenerate_n{n}.png"), dpi=300, bbox_inches='tight')
    if store_as_pdf:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"avg_sensitivity_degenerate_n{n}.pdf"), bbox_inches='tight')
    
    
    # Figure 2:
    cmap1 = plt.get_cmap("tab10")
    cmap2 = plt.get_cmap('Set2')
    
    panel_specs = [
        {
            'group_col': 'input redundancy',
            'vary_col': 'canalizing strength',
            'sort_col': 'canalizing strength',
            'xlabel': 'canalizing strength',
            'legend_title': 'fixed input redundancy',
            'min_size' : 5,
        },
        {
            'group_col': 'canalizing strength',
            'vary_col': 'input redundancy',
            'sort_col': 'input redundancy',
            'xlabel': 'input redundancy',
            'legend_title': 'fixed canalizing strength',
            'min_size' : 5,
        }
    ]
    
    if n>4:
        _, dict_x_groups = utils_network.get_network_level_results(n)

    # Plot:
    fig, ax = plt.subplots(1, 2, figsize=(6.5,2.5), sharey=True)
    
    for i, spec in enumerate(panel_specs):
        count = 0
        min_size = int(spec['min_size'])
        
        if n>4:
            if spec['group_col'] == 'canalizing strength':
                keys = list(dict_x_groups[0].keys())
            else:
                keys = list(dict_x_groups[1].keys())
            
            filtered = df[df[spec['group_col']].isin(keys)]
            grouped = filtered.groupby(spec['group_col'])
        elif n==4:
            grouped = df.groupby(spec['group_col'])
        
        for group_value, group in grouped:
            if len(group) < min_size:
                continue
    
            x_values = group[spec['sort_col']].values
            y_values = group['mean average sensitivity'].values
            
            color = cmap1(count)
            if count >= 10:
                color = cmap2(count-9)
            
            if show_dots_in_fig2:
                ax[i].scatter(x_values, y_values, s=30, alpha=0.3, color=color)
    
            m, b = np.polyfit(x_values, y_values, 1)
    
            xfit = np.linspace(x_values.min(), x_values.max(), 100)
            ax[i].plot(xfit, m * xfit + b, color=color, label=f"{group_value:.3f}")
            
            count += 1
            
        ax[i].set_xlabel(spec['xlabel'])
        handles, labels = ax[i].get_legend_handles_labels()
        ax[i].legend(handles, labels, title=spec['legend_title'], loc='lower center',
                       bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8)
        ax[i].spines[['top', 'right']].set_visible(False)
    
    ax[0].set_ylabel('average sensitivity')
    
    if store_as_png:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"avg_sensitivity_comparison_n{n}.png"), dpi=300, bbox_inches='tight')
    if store_as_pdf:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"avg_sensitivity_comparison_n{n}.pdf"), bbox_inches='tight')
    
    
    # Figure 3a:
    x1 = df['canalizing strength']
    x2 = df['input redundancy']
    x0 = df['mean average sensitivity']
    y = df['mean canalizing'].astype(int)
    
    df_all = pd.DataFrame({
        'canalizing strength': x1,
        'input redundancy': x2,
        'average sensitivity': x0,
    })
    
    y_target = y
    
    feature_pairs = [
        ('canalizing strength', 'average sensitivity'),
        ('input redundancy', 'average sensitivity')
    ]
    
    # 5-fold stratified cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    
    for idx, (feat_x, feat_y) in enumerate(feature_pairs):
        ax1 = axes[idx]
        # Original data (used for plotting)
        X = df_all[[feat_x, feat_y]]
        # Cross-validated AUC
        cv_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight='balanced')
        )
    
        auc_scores = cross_val_score(cv_model, X, y_target, cv=cv, scoring='roc_auc')
    
        # Fit model for plotting
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
        model = LogisticRegression(class_weight='balanced')
        model.fit(X_scaled, y_target)
    
        b0 = model.intercept_[0]
        b1 = model.coef_[0][0]
        b2 = model.coef_[0][1]
    
        # Plot limits (ORIGINAL units)
        x1_min, x1_max = X[feat_x].min() - 0.1, X[feat_x].max() + 0.1
        x2_min, x2_max = X[feat_y].min() - 0.1, X[feat_y].max() + 0.1
    
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, 0.02),
            np.arange(x2_min, x2_max, 0.02)
        )
    
        # Transform grid before prediction
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        grid_scaled = scaler.transform(grid)
    
        grid_probs = model.predict_proba(grid_scaled)[:, 1]
        grid_probs = grid_probs.reshape(xx1.shape)
    
        contour = ax1.contourf(xx1, xx2, grid_probs, levels=100, cmap='RdBu_r', alpha=0.6)
        contour.set_edgecolor("none")
        
        
        if idx == len(feature_pairs) - 1:
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="7%", pad=0.12)
    
            cbar = fig.colorbar(contour, cax=cax)
            cbar.set_ticks([0,0.25,0.5,0.75,1])
            cbar.set_label('P(f is canalizing)', rotation=270, labelpad=15)
    
        # Decision boundary in original units
        x_boundary = np.linspace(x1_min, x1_max, 300)
    
        x_boundary_scaled = (x_boundary - scaler.mean_[0]) / scaler.scale_[0]
        y_boundary_scaled = -(b0 + b1 * x_boundary_scaled) / b2
        y_boundary = (y_boundary_scaled * scaler.scale_[1] + scaler.mean_[1])
        boundary_line, = ax1.plot(x_boundary, y_boundary, color='black', linestyle='--', linewidth=2)
    
        hue_labels = y_target.map({0: 'false', 1: 'true'})
    
        sns.scatterplot(data=X, x=feat_x, y=feat_y, hue=hue_labels,
                        palette={'false': 'blue', 'true': 'red'},
                        edgecolor='k', s=60, alpha=0.9, legend=False, ax=ax1)
    
        ax1.set_xlim(x1_min, x1_max)
        ax1.set_ylim(x2_min, x2_max)
        ax1.set_xlabel(feat_x)
    
        if idx == 0:
            ax1.set_ylabel(feat_y)
    
        # ax1.set_title(f'AUC-ROC = {auc_scores.mean():.4f} ± {auc_scores.std():.4f}', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.75])
    
    false_dot = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor='blue',
                           markeredgecolor='k',
                           markersize=8)
    
    true_dot = plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='red',
                          markeredgecolor='k',
                          markersize=8)
    
    final_handles = [false_dot, true_dot, boundary_line]
    final_labels = ['non-canalizing', 'canalizing', 'decision boundary']
    
    fig.legend(handles=final_handles, labels=final_labels, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, 0.7), frameon=False)
    
    #plt.subplots_adjust(wspace=0.1)

    if store_as_png:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"avg_sensitivity_canalizing_logistic_regressionauc_n{n}.png"), dpi=300, bbox_inches='tight')
    if store_as_pdf:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"avg_sensitivity_canalizing_logistic_regressionauc_n{n}.pdf"), bbox_inches='tight')
    
    
    # Figure 3b:
    # Logistic regression, Decision tree, Support Vector Classification
    models = [("logistic regression", "Logistic regression"),
              ("decision tree", "Decision tree"),
              ("svc", "Support Vector Classification"),]
    
    rows = []
    
    for model_key, model_name in models:
        model_df = classifiers_auc_df[classifiers_auc_df["model"] == model_key]
        row = [model_name]
    
        for feature in ["canalizing strength", "input redundancy"]:
            result = model_df[model_df["feat_x"] == feature].iloc[0]
            row.append((result["auc_cv_mean"], result["auc_cv_std"]))
    
        rows.append(row)
    
    # KNN
    for k in [3, 5]:
        model_df = classifiers_auc_df[(classifiers_auc_df["model"] == "knn") &
                                      (classifiers_auc_df["param"] == k)]
    
        row = [f"KNN – {k} neighbors"]
    
        for feature in ["canalizing strength", "input redundancy"]:
            result = model_df[model_df["feat_x"] == feature].iloc[0]
            row.append((result["auc_cv_mean"], result["auc_cv_std"]))
    
        rows.append(row)
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    
    # Color scale
    norm = mcolors.Normalize(vmin=0.85, vmax=1.00)
    greens = plt.cm.Greens
    cmap = LinearSegmentedColormap.from_list("light_greens", greens(np.linspace(0, 1, 256)))
    
    # Table dimensions
    table_width = 1
    col_widths = [0.39, 0.305, 0.305]
    x_pos = [0]+list(np.cumsum(col_widths))
    
    # x positions of columns
    x0 = 0
    x1 = col_widths[0]
    x2 = x1 + col_widths[1]
    x3 = table_width
    
    # Header heights
    top_header_height = 0.10
    subheader_height = 0.10
    
    # Data row height
    data_row_height = 0.105
    
    # Bottom position of table
    y_bottom = 0.02
    
    # Number of classifiers
    n_rows = len(rows)
    
    # Total data height
    data_height = n_rows * data_row_height
    
    # Header positions
    subheader_y = y_bottom + data_height
    top_header_y = subheader_y + subheader_height
    table_top = top_header_y + top_header_height
    
    # Draw data cells
    for i, row in enumerate(rows):
        # y coordinate of this row
        y = y_bottom + (n_rows - 1 - i) * data_row_height
    
        # Classifier
        ax.add_patch(
            Rectangle((x_pos[0], y), col_widths[0], data_row_height, facecolor="white",
                      edgecolor="black", linewidth=0.8, transform=ax.transAxes)
                    )
        ax.text(x0 + 0.012, y + data_row_height / 2, row[0], 
                ha="left", va="center", transform=ax.transAxes)
    
        for j,label in enumerate(['canalizing strength','input redundancy']):
            data = row[j+1][0]
            ax.add_patch(
                Rectangle((x_pos[j+1], y), col_widths[j+1], data_row_height,
                          facecolor=cmap(norm(data)), edgecolor="black",
                          linewidth=0.8, transform=ax.transAxes)
                        )
            ax.text(x_pos[j+1] + col_widths[j+1] / 2, y + data_row_height / 2, f"{row[j+1][0]:.4f} ± {row[j+1][1]:.4f}",
                    ha="center", va="center", transform=ax.transAxes,
                    color='white' if float(row[j+1][0])>0.98 else 'black')
    
    # "Classifier" header — spans both header rows
    ax.add_patch(
        Rectangle((x_pos[0], subheader_y), col_widths[0], top_header_height + subheader_height,
                  facecolor="white", edgecolor="black", linewidth=0.8, transform=ax.transAxes)
                )
    ax.text(x0 + 0.012, subheader_y + (top_header_height + subheader_height) / 2,
            "Classifier", ha="left", va="center", fontweight="normal", transform=ax.transAxes)
    
    # "Average sensitivity" header — spans both metrics
    ax.add_patch(
        Rectangle((x1, top_header_y), col_widths[1] + col_widths[2], top_header_height,
                  facecolor="white", edgecolor="black", linewidth=0.8, transform=ax.transAxes)
                )
    ax.text(x1 + (col_widths[1] + col_widths[2]) / 2, top_header_y + top_header_height / 2,
            "Predict canalization by average sensitivity and", ha="center", va="center", fontweight="normal", transform=ax.transAxes)
    
    # "Canalizing strength" subheader
    ax.add_patch(
        Rectangle((x1, subheader_y), col_widths[1], subheader_height, facecolor="white",
                  edgecolor="black", linewidth=0.8, transform=ax.transAxes)
                )
    ax.text(x1 + col_widths[1] / 2, subheader_y + subheader_height / 2, "canalizing strength",
            ha="center", va="center", fontweight="normal", transform=ax.transAxes)
    
    # "Input redundancy" subheader
    ax.add_patch(
        Rectangle((x2, subheader_y), col_widths[2], subheader_height, facecolor="white",
                  edgecolor="black", linewidth=0.8, transform=ax.transAxes)
                )
    ax.text(x2 + col_widths[2] / 2, subheader_y + subheader_height / 2, "input redundancy",
            ha="center", va="center", fontweight="normal", transform=ax.transAxes)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    bottom_fig = fig.transFigure.inverted().transform(ax.transAxes.transform((0, y_bottom)))[1]
    top_fig = fig.transFigure.inverted().transform(ax.transAxes.transform((0, y_bottom + data_height)))[1]
    cbar_height = top_fig - bottom_fig
    cbar_ax = fig.add_axes([0.91, bottom_fig, 0.018, cbar_height])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0.85, 0.90, 0.95, 1.00])
    cbar.set_label("AUC-ROC", rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)
    
    # Figure limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    if store_as_png:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"avg_sensitivity_canalizing_classifiers_n{n}.png"), dpi=300, bbox_inches='tight')
    if store_as_pdf:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"avg_sensitivity_canalizing_classifiers_n{n}.pdf"), bbox_inches='tight')


    # Figure 4:
    fig, ax = plt.subplots(figsize=(4, 3))

    x = per_var_df['canalizing strength']
    y = per_var_df['input redundancy']

    unique_layer_positions = sorted(per_var_df['canalizing layer position'].unique())
    unique_layer_sizes = sorted(per_var_df['canalizing layer size'].unique())

    colors = ['gray'] + [cm.tab10(i) for i in range(len(unique_layer_positions) - 1)]
    colors[1] = cm.tab10(2)
    colors[2] = cm.tab10(1)
    colors[3] = cm.tab10(0)
    
    markers = ['x', 'X', 'o', 'P', 'd', 's', '^', 'v', '<', '>']
    if n==4:
        sizes = [30, 40, 70, 20, 30]
    elif n==5:
        sizes = [30, 30, 40, 70, 20, 30, 30, 30, 30, 30]
    else:
        assert len(unique_layer_sizes) <= len(markers), f"Not enough markers for {len(unique_layer_sizes)} layer sizes"

    color_map = dict(zip(unique_layer_positions, colors))
    marker_map = dict(zip(unique_layer_sizes, markers))
    sizes_map = dict(zip(unique_layer_sizes, sizes))

    for (layer_size, layer_pos), group in per_var_df.groupby(['canalizing layer size','canalizing layer position']):
        ax.scatter(
            group['canalizing strength'], group['input redundancy'],
            color=color_map[layer_pos],
            marker=marker_map[layer_size],
            s=sizes_map[layer_size], 
            linewidth=0.3
        )

    ax.set_xlabel('per-input canalizing strength')
    ax.set_ylabel('edge redundancy')
    ax.spines[['top', 'right']].set_visible(False)

    color_handles = [
        Line2D([0], [0], marker='s', linestyle='None',
            markerfacecolor=color_map[val], markeredgewidth=0, markersize=7,
            label='none' if val == 0 else f'layer {int(val)}')
        for val in unique_layer_positions
    ]

    marker_handles = [
        Line2D([0], [0], marker=marker_map[val], linestyle='None',
            color='k', markerfacecolor='k', markeredgecolor='k',
            markeredgewidth=0.3, markersize=7,
            label=str(max(0, int(val))))
        for val in unique_layer_sizes
    ]

    color_legend = ax.legend(
        handles=color_handles,
        title='canalizing layer',
        frameon=False,
        ncol=1,
        loc='center',
        bbox_to_anchor=(1.1, 0.5)
    )

    marker_legend = ax.legend(
        handles=marker_handles,
        title='number of canalizing variables in the layer',
        frameon=False,
        ncol=len(marker_handles),
        loc='center',
        bbox_to_anchor=(0.5, 1.15)
    )

    ax.add_artist(color_legend)
    ax.add_artist(marker_legend)

    if store_as_png:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"metric_per_var_n{n}_degenerate{bool(allow_degeneracy_in_fig4)}.png"),
                    dpi=300, bbox_inches='tight', bbox_extra_artists=[color_legend, marker_legend])
    if store_as_pdf:
        plt.savefig(os.path.join(FIGURE_FOLDER, f"metric_per_var_n{n}_degenerate{bool(allow_degeneracy_in_fig4)}.pdf"),
                    bbox_inches='tight', bbox_extra_artists=[color_legend, marker_legend])

