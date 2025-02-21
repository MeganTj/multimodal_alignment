import argparse
import copy
from functools import partial
import logging
import os
import pickle

import yaml
import optuna
import numpy as np
import torch
from optuna.samplers import TPESampler
from train_compute_align import run_align_exp, get_configs_and_results_path, get_exp_args

def setup_logging(log_file):
    # Configure logging
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.INFO
    )
    optuna.logging.get_logger("optuna").addHandler(logging.FileHandler(log_file))

def objective(trial, exp_args, all_lr, all_wd, modal_num, search_space="discrete"):
    # Define the search space for learning rate (lr)
    if search_space == "continuous":
        lr = trial.suggest_float(f"lr_modal_{modal_num}", all_lr[0], all_lr[-1], log=True)
        wd = trial.suggest_float(f"wd_modal_{modal_num}", all_wd[0], all_wd[-1], log=True)
    elif search_space == "discrete":
        lr = trial.suggest_categorical(f"lr_modal_{modal_num}", all_lr)
        wd = trial.suggest_categorical(f"wd_modal_{modal_num}", all_wd)
    else:
        raise ValueError("search_space must be either 'discrete' or 'continuous'")

    # Modify exp_args with current lr and wd
    exp_args.lr = [lr]
    exp_args.wd = [wd]
    exp_args.disable_alignment = True

    # Run the experiment
    _, _, _, _, results_path = get_configs_and_results_path(exp_args)
    run_align_exp(exp_args)

    # Clear CUDA memory
    with torch.no_grad():
        torch.cuda.empty_cache()

    # Load results and compute mean performance for the modality
    results = np.load(results_path, allow_pickle=True)
    val = results['perf_results'].item()["val"][modal_num]
    mean_val_results = val[val > 0].mean()  # Filter positive values and compute mean

    return mean_val_results

# Wrapper functions for each modality
def objective_helper(exp_args, all_lr, all_wd, modal_num, trial, search_space="discrete"):
    # Create and optimize study for a given modality
    exp_args_modal = copy.deepcopy(exp_args)
    exp_args_modal.epoch = [exp_args.epoch[modal_num]]
    exp_args_modal.modalities = [modal_num]
    return objective(trial, exp_args_modal, all_lr, all_wd, modal_num, search_space=search_space)

def run_final_experiment(exp_args, hyp_args, hyp_file, hyp_final_file):
    # Load optimal hyperparameters from results file
    with open(hyp_file, "r") as stream:
        hyp_dct = yaml.safe_load(stream)

    # Create a deep copy of exp_args for modification
    final_args = copy.deepcopy(exp_args)
    final_args.tune_only = "None"

    # Update final_args with optimal lr and wd for each modality
    final_args.lr = [
        hyp_dct["modal_0"]["params"]["lr_modal_0"],
        hyp_dct["modal_1"]["params"]["lr_modal_1"]
    ]
    final_args.wd = [
        hyp_dct["modal_0"]["params"]["wd_modal_0"],
        hyp_dct["modal_1"]["params"]["wd_modal_1"]
    ]

    final_args.max_depth = hyp_args.final_max_depth
    final_args.max_syn_depth = hyp_args.final_max_depth

    # Get results path and save it to a YAML file
    _, _, _, _, results_path = get_configs_and_results_path(final_args)
    with open(hyp_final_file, "w") as file:
        yaml.dump({"results_path": results_path}, file)
    final_args.use_saved_model = False
    # Run the final experiment
    run_align_exp(final_args)

# Initialize and run the Optuna studies
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-lrs", nargs='+', type=float, help="learning rates")
    parser.add_argument("--tune-wds", nargs='+', type=float, help="weight decays")
    parser.add_argument("--search-space", default="discrete", type=str)
    parser.add_argument("--n_trials", default=8, type=int)
    parser.add_argument("--sampler-seed", default=42, type=int)
    parser.add_argument("--final-max-depth", default=10, type=int)
    parser.add_argument("--final-suffix", default="final.yaml", type=str)
    parser.add_argument("--tune-only", default="None", choices=["None", "all", "diag"], help="How to do hyperpameter tuning runs")
    parser.add_argument("--save-dir", default="experiments/hyp", type=str)
    hyp_args = parser.parse_known_args()[0]
    exp_args = get_exp_args()

    exp_config_name = os.path.basename(exp_args.exp_config)[:-5]
    save_dir = os.path.join(hyp_args.save_dir, exp_config_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = exp_config_name if exp_args.dataset_name is None else exp_args.dataset_name
    # For probabilistic dataset
    if exp_args.red_key is not None:
        save_name +=  f"_{exp_args.red_key}"
    mod_str = '_'.join([str(modality) for modality in exp_args.modalities])
    hyp_file = os.path.join(save_dir, f"{save_name}_mod_{mod_str}_sea_{hyp_args.search_space[:3]}.yaml")
    # Tune the hyperparameters
    if not os.path.exists(hyp_file):
        hyp_log_file = os.path.join(save_dir, f"{save_name}_mod_{mod_str}_sea_{hyp_args.search_space[:3]}_log.txt")
        setup_logging(hyp_log_file)
        sampler = TPESampler(seed=hyp_args.sampler_seed)
        # Create and optimize study for modality 0
        objective_modal_0 = partial(objective_helper, exp_args, 
                                    hyp_args.tune_lrs, hyp_args.tune_wds, 0, search_space=hyp_args.search_space)
        study_modal_0 = optuna.create_study(direction="maximize", sampler=sampler)
        # study_modal_0.enqueue_trial({"learning_rate": 1e-2, "weight_decay": 1e-3}, skip_if_exists=True)
        study_modal_0.optimize(objective_modal_0, n_trials=hyp_args.n_trials)

        print("Best trial for modal 0:")
        trial0 = study_modal_0.best_trial
        print(f"  Value: {trial0.value}")
        print("  Params:")
        for key, value in trial0.params.items():
            print(f"    {key}: {value}")
        
        # Create and optimize study for modality2
        objective_modal_1 = partial(objective_helper, exp_args, 
                                    hyp_args.tune_lrs, hyp_args.tune_wds, 1, search_space=hyp_args.search_space)
        study_modal_1 = optuna.create_study(direction="maximize", sampler=sampler)
        # study_modal_1.enqueue_trial({"learning_rate": 1e-2, "weight_decay": 1e-3}, skip_if_exists=True)
        study_modal_1.optimize(objective_modal_1, n_trials=hyp_args.n_trials)

        print("Best trial for modal 1:")
        trial1 = study_modal_1.best_trial
        print(f"  Value: {trial1.value}")
        print("  Params:")
        for key, value in trial1.params.items():
            print(f"    {key}: {value}")

        # Save results of both studies to a single file
        results = {
            "modal_0": {
                "value": trial0.value,
                "params": trial0.params
            },
            "modal_1": {
                "value": trial1.value,
                "params": trial1.params
            }
        }

        with open(hyp_file, "w") as f:
            yaml.dump(results, f)

        print(f"Results saved to {hyp_file}")
    
    hyp_final_file = os.path.join(save_dir, f"{save_name}_{mod_str}_seed_{exp_args.base_seed}_{hyp_args.final_suffix}")
    run_final_experiment(exp_args, hyp_args, hyp_file, hyp_final_file)