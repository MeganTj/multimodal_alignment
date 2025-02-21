import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(arr, xaxis_label, xtick_labels, yaxis_label, ytick_labels, fig_title, plot_path, 
                 mask=None, plot_loss=False):
    plt.figure()
    cmap = "coolwarm" if not plot_loss else "coolwarm_r"
    ax = sns.heatmap(arr, cmap=cmap, mask=mask, linewidth=0.5)
    ax.invert_yaxis() 
    ax.set_xticks([pos + 0.5 for pos in range(len(arr[0]))], labels=xtick_labels)
    ax.set_yticks([pos + 0.5 for pos in range(len(arr))] , labels=ytick_labels)
    if xaxis_label == "width" or xaxis_label == "depth":
        ax.set(xlabel=f'Synthetic {xaxis_label.capitalize()}', ylabel=yaxis_label)
    else:
        ax.set(xlabel=f'{xaxis_label}', ylabel=yaxis_label)
    ax.set_title(fig_title)
    plt.savefig(plot_path)
    plt.close()

def plot_perf_heatmaps(train_results, val_results, test_results,  xaxis_label, xtick_labels, yaxis_label, ytick_labels, plot_results_pref, model_modalnums):
    for modal_num in model_modalnums:
        is_recon = isinstance(modal_num, tuple)
        plot_heatmap(train_results[modal_num], xaxis_label, xtick_labels, yaxis_label, ytick_labels, f"Modality {modal_num} Train Performance",  plot_results_pref + f"_modal_{modal_num}_train_perf.png", plot_loss=is_recon)
        plot_heatmap(val_results[modal_num], xaxis_label, xtick_labels, yaxis_label, ytick_labels, f"Modality {modal_num} Val Performance",  plot_results_pref + f"_modal_{modal_num}_val_perf.png", plot_loss=is_recon)
        plot_heatmap(test_results[modal_num], xaxis_label, xtick_labels, yaxis_label, ytick_labels, f"Modality {modal_num} Test Performance", plot_results_pref + f"_modal_{modal_num}_test_perf.png", plot_loss=is_recon)
