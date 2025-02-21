import numpy as np
from contextlib import redirect_stdout
from train_compute_align import run_align_exp, get_configs_and_results_path, get_exp_args
import argparse
import os
import yaml
import copy
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--tune-lrs", nargs='+', type=float, help="learning rates")
parser.add_argument("--tune-wds", nargs='+', type=float, help="weight decays")
parser.add_argument("--share-hyp", action="store_true", default=False)
parser.add_argument("--final-max-depth", default=10, type=int)
parser.add_argument("--final-suffix", default="final.yaml", type=str)
parser.add_argument("--tune-only", default="None", choices=["None", "all", "diag"], help="How to do hyperpameter tuning runs")
parser.add_argument("--save-dir", default="experiments/hyp", type=str)
hyp_args = parser.parse_known_args()[0]
exp_args = get_exp_args()
lst_perf = [[], []]
lst_hyp = []
lst_paths = []

exp_config_name = os.path.basename(exp_args.exp_config)[:-5]
save_dir = os.path.join(hyp_args.save_dir, exp_config_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = exp_config_name if exp_args.dataset_name is None else exp_args.dataset_name
# For probabilistic dataset
if exp_args.red_key is not None:
    save_name +=  f"_{exp_args.red_key}"
mod_str = '_'.join([str(modality) for modality in exp_args.modalities])
if exp_args.contrastive_expert_modality != -1:
    mod_str += f"_exp_{exp_args.contrastive_expert_modality}"
hyp_file = os.path.join(save_dir, f"{save_name}_mod_{mod_str}.yaml")
hyp_log_file = os.path.join(save_dir, f"{save_name}_mod_{mod_str}_log.txt")
# Tune the hyperparameters
if not os.path.exists(hyp_file):
    with open(hyp_log_file, "w") as f:
        print(hyp_args.tune_wds)
        for wd in hyp_args.tune_wds:
            for lr in hyp_args.tune_lrs:
                print(f'lr={lr}, wd={wd}', file=f)
                curr_args = copy.deepcopy(exp_args)
                # Modify with current lr and wd
                curr_args.lr = [lr]
                curr_args.wd = [wd]
                curr_args.disable_alignment = True
                _, _, _, _, results_path = get_configs_and_results_path(curr_args)
                run_align_exp(curr_args)

                with torch.no_grad():
                    torch.cuda.empty_cache()
                lst_hyp.append({"lr": lr, "wd": wd})
                # Keep track of directory for later use in analyzing results
                lst_paths.append(results_path)
                results=np.load(results_path, allow_pickle=True)
                modal_means = []
                for idx, (modalnum, val) in enumerate(results['perf_results'].item()["val"].items()):
                    # Filter for positive values
                    mean_val_results = val[val > 0].mean()
                    print(modalnum, val, mean_val_results, file=f)
                    lst_perf[idx].append(mean_val_results)
                    modal_means.append(mean_val_results)
                print(f'Mean over modalities: {np.array(modal_means).mean()}\n',  file=f)
        perfs = np.array(lst_perf)
        overall_perfs = perfs.mean(axis=0)

        if hyp_args.tune_only == "None":
            # Log the best hyperparameters, performances, and models
            best0_idx = np.argmax(perfs[0])
            best0_hyp = lst_hyp[best0_idx]
            best0_dir = lst_paths[best0_idx]
            print(f"Modality 0 Best Hyp: {best0_hyp}", file=f)
            print(f"Modality 0 Best Perf: {perfs[0][best0_idx]}", file=f)
            best1_dir = None
            best1_hyp = {"lr": None, "wd": None}
            if len(lst_perf) > 1:
                best1_idx = np.argmax(perfs[1])
                best1_hyp = lst_hyp[best1_idx]
                best1_dir = lst_paths[best1_idx]
                print(f"Modality 1 Best Hyp: {best1_hyp}", file=f)
                print(f"Modality 1 Best Perf: {perfs[1][best1_idx]}", file=f)
            bestoverall_idx = np.argmax(overall_perfs)
            bestoverall_hyp = lst_hyp[bestoverall_idx]
            bestoverall_dir = lst_paths[bestoverall_idx]
            print(f"Overall Best Hyp: {bestoverall_hyp}", file=f)
            print(f"Overall Best Perf: {overall_perfs[bestoverall_idx]}", file=f)
            print(f"Overall Best Modality 0 Perf: {perfs[0][bestoverall_idx]}", file=f)
            if len(lst_perf) > 1:
                print(f"Overall Best Modality 1 Perf: {perfs[1][bestoverall_idx]}", file=f)
            hyp_dct = {"lr": [best0_hyp["lr"], best1_hyp["lr"], bestoverall_hyp["lr"]], 
                    "wd": [best0_hyp["wd"], best1_hyp["wd"], bestoverall_hyp["wd"]],
                    "dirs": [best0_dir, best1_dir, bestoverall_dir]}
    if hyp_args.tune_only == "None":
        with open(hyp_file, "w") as file:
            yaml.dump(hyp_dct, file)

if hyp_args.tune_only == "None":
    with open(hyp_file, "r") as stream:
        hyp_dct = yaml.safe_load(stream)

    hyp_final_file = os.path.join(save_dir, f"{save_name}_mod_{mod_str}_sha_{str(hyp_args.share_hyp)[:3]}_seed_{exp_args.base_seed}_{hyp_args.final_suffix}")
    final_args = copy.deepcopy(exp_args)
    final_args.tune_only = "None"
    # Modify with current lr and wd
    if hyp_args.share_hyp:
        final_args.lr = [hyp_dct["lr"][-1], hyp_dct["lr"][-1]]
        final_args.wd = [hyp_dct["wd"][-1], hyp_dct["wd"][-1]]
    else:
        final_args.lr = hyp_dct["lr"][:-1]
        final_args.wd = hyp_dct["wd"][:-1]
    final_args.max_depth = hyp_args.final_max_depth
    final_args.max_syn_depth = hyp_args.final_max_depth
    _, _, _, _, results_path = get_configs_and_results_path(final_args)
    with open(hyp_final_file, "w") as file:
        yaml.dump({"results_path": results_path}, file)
    run_align_exp(final_args)