from collections import defaultdict
import pickle
import numpy as np
from contextlib import redirect_stdout
import plotly.express as px
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import yaml
import math
import copy
import pdb


AXIS_LABEL_FONTSIZE = 18
TICK_SIZE=16
TITLE_FONTSIZE=20
SMALL_AXIS_LABEL_FONTSIZE = 14
SMALL_TICK_SIZE=13
SMALL_TITLE_FONTSIZE = 16
BACKGROUND_COLOR = '#f7f7f7'  # Light gray hex color
# as defined in our data generation process, modality 1 is transformed
UNTRANSFORMED_MODALITY = 0
TRANSFORMED_MODALITY = 1
FIXED_DEPTH = 1
NUM_DEPTH = 10
NUM_SYN_DEPTH = 10
NCOLS = 5
NROWS = 1
MAX_U_VAL = 8
U_VALS = range(0, MAX_U_VAL + 1, 1)
SKIP_U_VALS = range(0, MAX_U_VAL + 1, 2)
SYN_DEPTH_VALS = range(0, NUM_SYN_DEPTH)
SKIP_SYN_DEPTH_VALS = range(0, NUM_SYN_DEPTH, 2)
METRICS=["unbiased_cka", "mutual_knn"]
MODES = ["best"]
MI_LEVELS = ["1.159"]

def parse_str(input_str, match_str, end_char="_"):
    start_idx = input_str.index(match_str)
    end_idx = input_str.index(end_char, start_idx+len(match_str))
    return input_str[start_idx+len(match_str):end_idx]

def get_nrows_ncols(num_plots):
    nrows = math.floor(np.sqrt(num_plots))
    ncols = math.ceil(num_plots / nrows)
    return nrows, ncols

def plot_scatter(curr_ax, xs, ys, color="tab:blue", cmap=None, alpha=0.8):
    return curr_ax.scatter(xs, ys, 
                    color=color,
                    cmap=cmap,
                    edgecolor="black",  # Add edge color to markers
                    s=40,  # Increase marker size
                    alpha=alpha  # Slightly transparent markers
                    )
    
parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", default="experiments/hyp/rus_prob", type=str)
parser.add_argument("--plot-dir", default="experiments/rus_prob_plots_multiseed", type=str)
parser.add_argument("--data-pref", default="experiments/data/uni_DATA_mi=", type=str)
parser.add_argument("--final-suffix", default="_final.yaml", type=str)
args = parser.parse_known_args()[0]

# TODO: Load in data
if not os.path.exists(args.plot_dir):
    os.makedirs(args.plot_dir)
results_paths = []
for file in os.listdir(args.save_dir):
    if file.endswith(args.final_suffix):
        with open(os.path.join(args.save_dir, file), "r") as stream:
            results_path = yaml.safe_load(stream)["results_path"]
            results_paths.append(results_path)

# Load metadata
mi_data = {}
data_dir, fname_pref = os.path.split(args.data_pref)
for file in os.listdir(data_dir):
    if file.startswith(fname_pref):
        mi_level = parse_str(file, "mi=")
        with open(os.path.join(data_dir, file), 'rb') as f:
            data = pickle.load(f)
            if 'metadata' in data:
                mi_data[mi_level] = data['metadata']

lst_mis = sorted(mi_data.keys())
path = results_paths[0]

feat_start_idx = path.index("feat_")
feat_end_idx = path.index("_", feat_start_idx+5)
num_feat = int(path[feat_start_idx + 5:feat_end_idx])

for data_split in ["test"]:
    full_plot_dir = os.path.join(args.plot_dir, data_split)
    if not os.path.exists(full_plot_dir):
        os.makedirs(full_plot_dir)
    # Initialize results arrays
    results=np.load(path, allow_pickle=True)
    test_perf_results = results['perf_results'].item()[data_split]
    # Map modal to u to perf arr
    perf_dct = {modal: {mi_level: defaultdict(dict) for mi_level in mi_data.keys()} for modal in test_perf_results.keys()}
    pairwise_align_perf_corr_het = {}
    pairwise_align_vs_perf_y = {mi_level: defaultdict(dict) for mi_level in mi_data.keys()}
    all_metrics = []
    for metric in METRICS:
        pairwise_align_perf_corr_het[metric] = {"best": {mi_level: defaultdict(list) for mi_level in mi_data.keys()}, 
                                    "mean": {mi_level: defaultdict(list) for mi_level in mi_data.keys()} }
        all_metrics.append(metric)
        
    
    # Separate plot for each level of heterogeneity of Alignment vs Unique 
    num_points = 0
    max_xs = {
        mi_level: [] for mi_level in mi_data.keys()
    }
    # Iterate over the uniqueness
    for idx, path in enumerate(results_paths):
        # Parse mi level
        curr_mi_level = parse_str(path, "_mi=")
        if curr_mi_level not in MI_LEVELS:
            continue
        curr_r = int(parse_str(path, "_r_", end_char="/"))
        curr_seed = int(parse_str(path, "_seed="))
        results=np.load(path, allow_pickle=True)
        curr_u = num_feat - curr_r
        max_xs[curr_mi_level].append(curr_u)
        # Get performances
        metric_perf_arrs = results['perf_results'].item()[data_split]
        for modal, modal_perf in metric_perf_arrs.items():
            perf_dct[modal][curr_mi_level][curr_seed][curr_u] = modal_perf
        pairwise_align_vs_perf_y[curr_mi_level][curr_seed][curr_u] = results['pairwise_align_results']
       
        for metric in METRICS:
            for mode in MODES:
                for syn_depth in range(NUM_SYN_DEPTH):
                    pairwise_align = pairwise_align_vs_perf_y[curr_mi_level][curr_seed][curr_u][syn_depth][data_split][UNTRANSFORMED_MODALITY][TRANSFORMED_MODALITY][metric][mode][FIXED_DEPTH]
                    het_modal_perf = metric_perf_arrs[TRANSFORMED_MODALITY][:, syn_depth]
                    pairwise_align_perf_corr_het[metric][mode][curr_mi_level][curr_seed].append(np.corrcoef(pairwise_align, het_modal_perf)[0, 1])
    

    print("Pairwise: Alignment vs Unique, Alignment vs D', Alignment vs Depth")
    for metric in METRICS:
        for mode in MODES:
            for fixed_depth in range(3):
                fig, axs = plt.subplots(NROWS, NCOLS, figsize=(3 * NCOLS, 3 * NROWS))
                axs = axs.flatten()
                syn_all_xs = []
                syn_all_ys = []
                syn_all_u = []
                syn_depths = range(0, NUM_SYN_DEPTH, 2)
                syn_combined_max_ys = np.zeros((MAX_U_VAL + 1, NUM_SYN_DEPTH))
                for syn_idx, syn_depth in enumerate(range(NUM_SYN_DEPTH)):
                    combined_max_ys = np.zeros(MAX_U_VAL + 1)
                    combined_max_xs = range(MAX_U_VAL + 1)
                    all_xs = []
                    all_ys = []
                    for idx, u_val in enumerate(range(MAX_U_VAL)):
                        for mi_level in MI_LEVELS:
                            for seed_dct in pairwise_align_vs_perf_y[mi_level].values():
                                # This is a matrix where x is (fixed_depth_idx, depth_idx)
                                syn_pairwise_mat = seed_dct[u_val][syn_depth][data_split][UNTRANSFORMED_MODALITY][TRANSFORMED_MODALITY][metric][mode]
                                fixed_align = syn_pairwise_mat[fixed_depth, :]
                                xs = [u_val for _ in range(len(fixed_align))]
                                all_xs.extend(xs)
                                all_ys.extend(fixed_align)
                                combined_max_ys[u_val] = max(np.max(fixed_align), combined_max_ys[u_val])
                                syn_all_u.extend(xs)
                                syn_all_xs.extend([syn_depth for _ in range(len(xs))])
                                syn_all_ys.extend(fixed_align)
                                syn_combined_max_ys[u_val][syn_depth] = max(np.max(fixed_align), syn_combined_max_ys[u_val][syn_depth])
                    # Make plot
                    if syn_idx % 2 == 0:
                        curr_ax = axs[syn_idx  // 2]
                        curr_ax.set_facecolor(BACKGROUND_COLOR)
                        # Add grid lines
                        curr_ax.grid(True, linestyle='--', alpha=0.6)
                        plot_scatter(curr_ax, all_xs, all_ys, alpha=0.6)
                        plot_scatter(curr_ax, combined_max_xs, combined_max_ys, color="red")
                        res = stats.linregress(combined_max_xs, combined_max_ys)
                        curr_ax.set_ylim(0, 1.0)
                        curr_ax.set_xlabel("Unique", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                        curr_ax.set_ylabel("Alignment", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                        curr_ax.tick_params(axis='both', which='major', labelsize=SMALL_TICK_SIZE)
                        curr_ax.set_title(f"$D_\phi$={syn_depth + 1},  r={res.rvalue:.3f}", fontsize=SMALL_TITLE_FONTSIZE)
                fig.tight_layout()
                plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_pairwise_fixed_{fixed_depth + 1}_syn-depth_syn_align_scatter.pdf"))
                plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_pairwise_fixed_{fixed_depth + 1}_syn-depth_syn_align_scatter.png"))
                plt.close()

                # For each uniqueness, plot Alignment vs Synthetic Depth 
                syn_all_xs = np.array(syn_all_xs)
                syn_all_ys = np.array(syn_all_ys)
                syn_all_u = np.array(syn_all_u)
                fig, axs = plt.subplots(NROWS, NCOLS, figsize=(3 * NCOLS, 3 * NROWS))
                axs = axs.flatten()
                for idx, u_val in enumerate(SKIP_U_VALS):
                    indices = np.where(syn_all_u == u_val)
                    xs = syn_all_xs[indices]
                    ys = syn_all_ys[indices]
                    axs[idx].set_facecolor(BACKGROUND_COLOR)
                    # Add grid lines
                    axs[idx].grid(True, linestyle='--', alpha=0.6)
                    plot_scatter(axs[idx], xs, ys)
                    axs[idx].tick_params(axis='both', which='major', labelsize=SMALL_AXIS_LABEL_FONTSIZE)
                    axs[idx].set_ylim(0, 1.0)
                    axs[idx].set_xlabel("$D_\phi$", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                    axs[idx].set_ylabel("Alignment", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                    max_xs = range(0, NUM_SYN_DEPTH)
                    axs[idx].scatter(max_xs, syn_combined_max_ys[u_val, :], color="red")
                    res = stats.linregress(max_xs, syn_combined_max_ys[u_val])
                    axs[idx].set_title(f"U={u_val}, r={res.rvalue:.3f}", fontsize=SMALL_TITLE_FONTSIZE)
                for j in range(u_val + 1, len(axs)):
                    axs[j].axis('off')
                fig.tight_layout()
                plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_fixed_{fixed_depth + 1}_pairwise_align_het_scatter.pdf"))
                plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_fixed_{fixed_depth + 1}_pairwise_align_het_scatter.png"))
                plt.close()

                # Plot the alignment heatmaps for each uniqueness
                fig, axs = plt.subplots(NROWS, NCOLS, figsize=(3 * NCOLS, 3 * NROWS))
                axs = axs.flatten()
                all_align_d_corr = []
                for idx, u_val in enumerate(U_VALS):
                    mi_level = MI_LEVELS[0]
                    sum_heatmap = np.zeros((NUM_DEPTH, NUM_SYN_DEPTH))
                    align_d_corr = []
                    for seed_dct in pairwise_align_vs_perf_y[mi_level].values():
                        for syn_depth in SYN_DEPTH_VALS:
                            syn_pairwise_mat = seed_dct[u_val][syn_depth][data_split][UNTRANSFORMED_MODALITY][TRANSFORMED_MODALITY][metric][mode]
                            sum_heatmap[:, syn_depth] += syn_pairwise_mat[fixed_depth, :]
                            align_d_corr.append(np.corrcoef(np.arange(len(syn_pairwise_mat)), syn_pairwise_mat[fixed_depth, :])[0, 1])
                    all_align_d_corr.append(align_d_corr)
                    if idx % 2 == 0:
                        curr_ax = axs[idx  // 2]
                        # Take the average alignment over multiple seeds
                        arr = sum_heatmap / len(pairwise_align_vs_perf_y[mi_level])
                        sns.heatmap(arr, cmap="coolwarm",  ax=curr_ax, vmin=0, vmax=np.max(arr), 
                        cbar_kws={"shrink": 0.75}, linewidth=0.5)
                        curr_ax.invert_yaxis() 
                        curr_ax.tick_params(axis='both', which='major', labelsize=SMALL_TICK_SIZE)
                        curr_ax.set_xticks([pos + 0.5 for pos in range(len(arr[0]))], labels=range(1, NUM_DEPTH + 1))
                        curr_ax.set_yticks([pos + 0.5 for pos in range(len(arr))] , labels=range(1, NUM_SYN_DEPTH + 1))
                        curr_ax.set_xlabel('$D_\phi$', fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                        curr_ax.set_ylabel("$D_\\text{Enc}$", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                        curr_ax.set_title(f"U={u_val}", fontsize=SMALL_TITLE_FONTSIZE)
                for j in range(u_val + 1, len(axs)):
                    axs[j].axis('off')
                fig.tight_layout()
                plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_fixed_{fixed_depth + 1}_pairwise_align_heatmap_unique.png"))
                plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_fixed_{fixed_depth + 1}_pairwise_align_heatmap_unique.pdf"))
                plt.close()

                plt.figure()
                # Create the box plot
                plt.boxplot(all_align_d_corr, tick_labels=U_VALS)

                # Set title and labels
                plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
                plt.xlabel("Uniqueness", fontsize=AXIS_LABEL_FONTSIZE)
                plt.ylabel("Alignment/Depth", fontsize=AXIS_LABEL_FONTSIZE)
                # Save the plot
                plt.tight_layout()
                plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_pairwise_align_depth_vs_unique.pdf"))
                plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_pairwise_align_depth_vs_unique.png"))
                plt.close()

                                
    print("Pairwise: Performance vs Alignment")
    plot_scaled = True if len(MI_LEVELS) > 0 else False
    for metric in METRICS:
        for mode in MODES:
            all_perf_align_corr = {key: [] for key in U_VALS}
            all_perf_d_corr = {key: [] for key in U_VALS}
            all_perf = {key: [] for key in U_VALS}
            all_align = {key: [] for key in U_VALS}
            all_depths = {key: [] for key in U_VALS}
            all_syn_depths = {key: [] for key in U_VALS}
            last_d_all_perf = {key: [] for key in U_VALS}
            last_d_all_align = {key: [] for key in U_VALS}
            last_d_all_syn_depths = {key: [] for key in U_VALS}
            for syn_depth in range(NUM_SYN_DEPTH):
                for fixed_depth in range(3):
                    nrows, ncols = get_nrows_ncols(len(U_VALS))
                    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
                    axs = axs.flatten() 
                    for idx, u_val in enumerate(U_VALS):
                        # Generate a range of colors based on the number of plots
                        cmap = plt.get_cmap('viridis')
                        # Generate 3 distinct colors from the colormap
                        colors = [cmap(i / NUM_DEPTH) for i in range(NUM_DEPTH)]
                        all_syn_perf = []
                        all_syn_pairwise_mat = []
                        curr_all_depths = []
                        for mi_idx, mi_level in enumerate(MI_LEVELS):
                            for curr_seed, seed_dct in pairwise_align_vs_perf_y[mi_level].items():
                                syn_pairwise_mat = seed_dct[u_val][syn_depth][data_split][UNTRANSFORMED_MODALITY][TRANSFORMED_MODALITY][metric][mode]
                                
                                # Scale performance to plot experiments across different MI on the same plot
                                perf = perf_dct[TRANSFORMED_MODALITY][mi_level][curr_seed]
                                syn_perf = perf[u_val][:, syn_depth]
                                
                                if len(MI_LEVELS) > 1:
                                    syn_perf = syn_perf / syn_perf.max()
                                all_syn_perf.extend(syn_perf)
                                syn_align = syn_pairwise_mat[fixed_depth, :]
                                all_syn_pairwise_mat.extend(syn_align)
                                curr_all_depths.extend(np.arange(syn_pairwise_mat.shape[1]))
                                if fixed_depth == 0:
                                    last_d_all_perf[u_val].append(syn_perf[-1])
                                    last_d_all_align[u_val].append(syn_align[-1])
                                    last_d_all_syn_depths[u_val].append(syn_depth + 1)
                                axs[idx].set_facecolor(BACKGROUND_COLOR)
                                # Add grid lines
                                axs[idx].grid(True, linestyle='--', alpha=0.6)
                                axs[idx].tick_params(axis='both', which='major', labelsize=SMALL_TICK_SIZE)
                                axs[idx].set_xlabel("Uniqueness", fontsize=AXIS_LABEL_FONTSIZE)
                                axs[idx].set_ylabel("Alignment/Depth", fontsize=AXIS_LABEL_FONTSIZE)
                                scatter = plot_scatter(axs[idx], syn_perf, syn_pairwise_mat[fixed_depth, :], color=colors)
                        # Adding color bar
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=NUM_DEPTH))
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=axs[idx])
                        cbar.ax.tick_params(labelsize=SMALL_TICK_SIZE)
                        cbar.set_label('Depth', fontsize=SMALL_AXIS_LABEL_FONTSIZE) 
                        # After the loop, calculate the overall correlation coefficient
                        all_syn_perf = np.array(all_syn_perf)
                        all_syn_pairwise_mat = np.array(all_syn_pairwise_mat)
                        # Compute the Pearson correlation coefficient over all the data points
                        corr_matrix = np.corrcoef(all_syn_perf, all_syn_pairwise_mat)
                        correlation_coefficient = corr_matrix[0, 1]  # Extract the correlation coefficient
                        if fixed_depth == 0:
                            all_perf_align_corr[u_val].append(correlation_coefficient)
                            all_perf_d_corr[u_val].append(np.corrcoef(all_syn_perf, curr_all_depths)[0, 1])
                            all_perf[u_val].extend(all_syn_perf)
                            all_align[u_val].extend(all_syn_pairwise_mat)
                            all_depths[u_val].extend(curr_all_depths)
                            all_syn_depths[u_val].extend([syn_depth + 1 for _ in range(len(curr_all_depths))])
                        axs[idx].set_facecolor(BACKGROUND_COLOR)
                        # Add grid lines
                        axs[idx].grid(True, linestyle='--', alpha=0.6)
                        axs[idx].set_title(f"$U={u_val}$, r={correlation_coefficient:.3f}", fontsize=SMALL_TITLE_FONTSIZE)
                        axs[idx].set_xlabel("Performance", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                        axs[idx].set_ylabel("Alignment", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                        axs[idx].tick_params(axis='both', which='major', labelsize=SMALL_TICK_SIZE)
                    
                    fig.suptitle(f"$D_\phi$={syn_depth+1}", fontsize=TITLE_FONTSIZE)
                    fig.tight_layout()
                    if plot_scaled:
                        plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_fixed_depth_{fixed_depth + 1}_syn_depth_{syn_depth + 1}_scaled_pairwise_perf_vs_align.pdf"))
                        plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_fixed_depth_{fixed_depth + 1}_syn_depth_{syn_depth + 1}_scaled_pairwise_perf_vs_align.png"))
                    else:
                        plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_mi={mi_level}_fixed_depth_{fixed_depth + 1}_syn_depth_{syn_depth + 1}_scaled_pairwise_perf_vs_align.pdf"))
                        plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_mi={mi_level}_fixed_depth_{fixed_depth + 1}_syn_depth_{syn_depth + 1}_pairwise_perf_vs_align.png"))
                    plt.close()
            # Make box plots of alignment/performance and depth/performance for every level of uniqueness
            plt.figure()
            boxplot_data = []

            for u_val in U_VALS:
                # Collect all_ys corresponding to the current u
                boxplot_data.append(all_perf_align_corr[u_val])

            # Create the box plot
            plt.boxplot(boxplot_data, tick_labels=U_VALS)

            # Set title and labels
            plt.xlabel("Uniqueness", fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel("Alignment/Performance", fontsize=AXIS_LABEL_FONTSIZE)
            plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_pairwise_align_perf_vs_unique.pdf"))
            plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_pairwise_align_perf_vs_unique.png"))
            plt.close()
            
            plt.figure()
            boxplot_data = []

            for u_val in U_VALS:
                # Collect all_ys corresponding to the current u
                boxplot_data.append(all_perf_d_corr[u_val])

            # Create the box plot
            plt.boxplot(boxplot_data, tick_labels=U_VALS)

            # Set title and labels
            plt.xlabel("Uniqueness", fontsize=AXIS_LABEL_FONTSIZE)
            plt.ylabel("Performance/Depth", fontsize=AXIS_LABEL_FONTSIZE)
            plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_pairwise_perf_depth_vs_unique.pdf"))
            plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_pairwise_perf_depth_vs_unique.png"))
            plt.close()

            fig, axs = plt.subplots(NROWS, NCOLS, figsize=(4 * NCOLS, 3 * NROWS))
            axs = axs.flatten() 
            for idx, u_val in enumerate(SKIP_U_VALS):
                axs[idx].set_facecolor(BACKGROUND_COLOR)
                # Add grid lines
                axs[idx].grid(True, linestyle='--', alpha=0.6)
                scatter = axs[idx].scatter(last_d_all_perf[u_val], last_d_all_align[u_val], 
                                           c=last_d_all_syn_depths[u_val], cmap='viridis', edgecolor='k',  # Add edge color to markers
                                           s=40,  # Increase marker size
                                           alpha=0.8 )
                cbar = plt.colorbar(scatter, ax=axs[idx])
                cbar.ax.tick_params(labelsize=SMALL_TICK_SIZE)
                cbar.set_label('$D_\phi$', fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                axs[idx].set_xlabel("Performance", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                axs[idx].set_ylabel("Alignment", fontsize=SMALL_AXIS_LABEL_FONTSIZE)
                axs[idx].tick_params(axis='both', which='major', labelsize=SMALL_TICK_SIZE)
                axs[idx].set_title(f"U={u_val}", fontsize=SMALL_TITLE_FONTSIZE)
            for j in range(u_val + 1, len(axs)):
                axs[j].axis('off')
            # Adjust the layout to avoid overlap
            plt.tight_layout()
            # Save the entire figure as one file
            plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_align_perf_unique_scatter_last_depth_syn_depth.pdf"), bbox_inches="tight")
            plt.savefig(os.path.join(full_plot_dir, f"{metric}_{mode}_align_perf_unique_scatter_last_depth_syn_depth.png"), bbox_inches="tight")
            plt.close()