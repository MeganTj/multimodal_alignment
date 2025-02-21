from collections import defaultdict
import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import sys
sys.path.append(os.getcwd())
import yaml
from datetime import datetime
from synthetic.get_data import get_dataloader, get_dataloader_prob # noqa
from datasets.get_dataset_dataloader import get_dataloader_fn
from training_structures.train_eval import train_eval_supervised
from get_alignment import get_alignment_all_layers
from utils.utils import seed_everything, save_args_to_file
from utils.plot import plot_perf_heatmaps
from utils.align_metrics import AlignmentMetrics
all_metrics = AlignmentMetrics.SUPPORTED_METRICS
from contextlib import redirect_stdout
import itertools
import copy
import json

DATA_KEYS = ["train", "val", "test"]

def get_alignment_all_metrics(depth, syn_depth, min_depth, max_depth, x1, encoder1, x2, encoder2, 
                              metrics, align_results, **kwargs):
    for metric in metrics:
        align_mat = get_alignment_all_layers(x1, encoder1, x2, encoder2, metric, **kwargs)
        # Pad to depth
        align_mat = np.pad(align_mat, [(0, max_depth - len(align_mat)), (0, max_depth - len(align_mat[0]))])
        align_results[metric][depth - min_depth][syn_depth - min_depth] = align_mat
    return align_results

def get_pairwise_align_synthetic(saved_model_paths, metrics, data_keys, all_aligndata, num_depth, **kwargs):
    fixed_key = 0
    depth_key = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pairwise_align = {data_key: {} for data_key in data_keys}
    for data_key, aligndata in zip(data_keys, all_aligndata):
        pairwise_align[data_key][fixed_key] = {}
        fixed_modal = saved_model_paths[fixed_key]
        pairwise_align[data_key][fixed_key][depth_key] = {}
        depth_modal = saved_model_paths[depth_key]
        for metric in metrics:
            pairwise_align[data_key][fixed_key][depth_key][metric] = {
                                                                    "all": np.zeros((num_depth, num_depth, num_depth, num_depth)),
                                                                    "best": np.zeros((num_depth, num_depth)), 
                                                                    }
            for fixed_idx, fixed_modal_model in enumerate(fixed_modal):
                fixed_encoder = torch.load(fixed_modal_model + "_final_encoder.pt").to(device)
                for depth_idx, depth_modal_model in enumerate(depth_modal):
                    depth_encoder = torch.load(depth_modal_model + "_final_encoder.pt").to(device)
                    # Measure alignment
                    batch = next(iter(aligndata))
                    align_mat = get_alignment_all_layers(batch[fixed_key], fixed_encoder, batch[depth_key], depth_encoder, metric, **kwargs)
                    assert len(align_mat.shape) == 2
                    align_mat = np.pad(align_mat, [(0, num_depth - len(align_mat)), (0, num_depth - len(align_mat[0]))])
                    pairwise_align[data_key][fixed_key][depth_key][metric]["all"][fixed_idx][depth_idx] = align_mat
                    pairwise_align[data_key][fixed_key][depth_key][metric]["best"][fixed_idx][depth_idx] = np.max(align_mat)
    return pairwise_align


def get_pairwise_align_real(saved_model_paths, metrics, data_keys, all_aligndata, num_depth, perf_results, **kwargs):
    # Cmpute the pairwise align only for the best models 
    assert len(saved_model_paths) == 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pairwise_align = {data_key: {} for data_key in data_keys}
    pairwise_align_corr = {}
    modal_perf_summary = defaultdict(dict)
    best_modal_model = {}
    for data_key, aligndata in zip(data_keys, all_aligndata):
        pairwise_align_corr[data_key] = {}
        for fixed_idx, fixed_key in enumerate(saved_model_paths.keys()):
            pairwise_align[data_key][fixed_key] = {}
            # Get the best model of modality with fixed depth
            modal_val_perf = perf_results["val"][fixed_key]
            val_perf_idx = np.argmax(modal_val_perf)
            for perf_data_key in data_keys:
                modal_perf_summary[fixed_key][perf_data_key] = perf_results[perf_data_key][fixed_key][val_perf_idx]
            best_modal_model[fixed_key] = saved_model_paths[fixed_key][val_perf_idx] + "_final_encoder.pt"
            for depth_key in saved_model_paths.keys():
                if depth_key != fixed_key:
                    pairwise_align_corr[data_key][depth_key] = {}
                    pairwise_align[data_key][fixed_key][depth_key] = {}
                    depth_modal = saved_model_paths[depth_key]
                    for metric in metrics:
                        pairwise_align_corr[data_key][depth_key][metric] = {}
                        pairwise_align[data_key][fixed_key][depth_key][metric] = {"best": np.zeros((num_depth)), }
                        for depth_idx, depth_modal_model in enumerate(depth_modal):
                            fixed_encoder = torch.load(best_modal_model[fixed_key]).to(device)
                            depth_encoder = torch.load(depth_modal_model + "_final_encoder.pt").to(device)
                            # Measure alignment
                            batch = next(iter(aligndata))
                            align_mat = get_alignment_all_layers(batch[fixed_key], fixed_encoder, batch[depth_key], depth_encoder, metric, **kwargs)
                            assert len(align_mat.shape) == 2
                            pairwise_align[data_key][fixed_key][depth_key][metric]["best"][depth_idx] = np.max(align_mat)
                            del fixed_encoder, depth_encoder, batch
                            with torch.no_grad():
                                torch.cuda.empty_cache()
                        # Compute correlation 
                        pairwise_align_metric = pairwise_align[data_key][fixed_key][depth_key][metric]
                        pairwise_align_corr[data_key][depth_key][metric]["best"] = np.corrcoef(pairwise_align_metric["best"], perf_results[data_key][depth_key].squeeze())[0, 1]

    return pairwise_align, pairwise_align_corr, modal_perf_summary, best_modal_model

def get_exp_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, help="path to config")
    parser.add_argument("--metrics",  nargs='+', default=[ "mutual_knn", "unbiased_cka"], type=str, help="list of metrics to compute alignment")
    parser.add_argument("--q", default=1.0, type=float)
    parser.add_argument("--topk", default=100, type=int)
    parser.add_argument("--dataset-name", default=None, help="Specifies dataset name")
    # These arguments are for probabilistic dataset
    parser.add_argument("--red-key", type=str, default=None, help="Specifies dataset name")

    parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
    parser.add_argument("--task", default="classification", type=str, choices=["classification", "regression", "reconstruction"])
    parser.add_argument("--eval-task", default="classification", type=str, choices=["classification", "posneg-classification", "reconstruction"])
    parser.add_argument("--keys", nargs='+', default=['0', '1','label'], type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
    parser.add_argument("--align-bs", default=4096, type=int, help="batch size for computing alignment")
    parser.add_argument("--num-workers", default=4, type=int)
    # MLP arguments -- these will be overidden if specified in config file
    parser.add_argument("--input-dim", default=30, type=int)
    parser.add_argument("--num-hidden", default=1, type=int, help="Number of hidden layers")
    parser.add_argument("--hidden-dim", default=512, type=int)
    parser.add_argument("--nonlin", default="leaky_relu", type=str, choices=["relu, leaky_relu, tanh"])
    # Training arguments
    parser.add_argument("--epoch",  nargs='+', default=[50, 600], type=int, help="specify the number of epochs to train each model for")
    parser.add_argument("--lr", nargs='+', default=[1e-4], type=float, help="Learning rate")
    parser.add_argument("--wd", nargs='+', default=[0.0], type=float, help="Weight decay")
    parser.add_argument("--mode", default="depth", type=str, choices=["width", "depth"])
    parser.add_argument("--xaxis_label", default="depth", type=str, choices=["width", "depth"])
    parser.add_argument("--min-depth", default=1, type=int, help="Min depth of synthetic dataset")
    parser.add_argument("--max-depth", default=10, type=int, help="Max depth of synthetic dataset")
    parser.add_argument("--max-syn-depth", default=None, type=int, help="Max depth of synthetic dataset")
    parser.add_argument("--base-seed", default=2, type=int)
    parser.add_argument("--same-seed", default=False, action="store_true", help="Whether or not to use the same seed for all omdels")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--use-saved-model", action="store_true", default=False, help="Whether or not to load saved models")
    parser.add_argument("--tune-only", default="None", choices=["None", "all", "diag"], help="How to do hyperpameter tuning runs")
    parser.add_argument("--disable-alignment", action="store_true", default=False, help="Whether or not to compute alignment")
    parser.add_argument("--results-dir", default="experiments/syn_align", type=str)
    parser.add_argument("--summary-dir", default="experiments/align_summary", type=str)
    args = parser.parse_known_args()[0]
    return args

def get_configs_and_results_path(args):
    with open(args.exp_config, 'r') as stream:
        config = yaml.safe_load(stream)
        dataset_config = config["dataset"]
        model_config = config["model"]
    config_name = os.path.basename(args.exp_config)[:-5]
    lr_str = '_'.join([str(lr) for lr in args.lr])
    wd_str = '_'.join([str(wd) for wd in args.wd])
    mod_str = '_'.join([str(modality) for modality in args.modalities])
    folder_name = f"cfg_{config_name}_mod_{mod_str}_task_{args.task[:3]}_seed={args.base_seed}_same_{args.same_seed}_mode_{args.mode}_hd_{args.hidden_dim}_nh_{args.num_hidden}_lr_{lr_str}_wd_{wd_str}"
    dataset_config["name"] = args.dataset_name if args.dataset_name is not None else dataset_config["name"]
    # If synthetic, run on multiple synthetic datasets
    if "synthetic" in dataset_config["type"]:
        # Get maximum depth of synthetic dataset
        max_syn_depth = args.max_depth if args.max_syn_depth is None else args.max_syn_depth
        num_syn_depth = max_syn_depth - args.min_depth + 1
        full_dataset_name = f"{dataset_config['setting']}_{dataset_config['name']}"
        if dataset_config["type"] == "synthetic_prob":
            full_dataset_name += f"_{args.red_key}"
    else:
        max_syn_depth = 1
        num_syn_depth = 1
        full_dataset_name = dataset_config['name'] 
    # Organize models by config and hyperparameters
    base_dir = os.path.join(args.results_dir, full_dataset_name, folder_name)
    model_dir = os.path.join(args.results_dir, full_dataset_name, f"{model_config['setting'][:3]}_models_seed_{args.base_seed}", f"cfg_{config_name}") 
    results_path = os.path.join(base_dir, f"{model_config['setting'][:3]}_min_{args.min_depth}_max_{args.max_depth}_syn-max_{max_syn_depth}_results.npz")
    return (dataset_config, full_dataset_name), model_config, (max_syn_depth, num_syn_depth), (base_dir, model_dir), results_path

def get_dataloader(args, dataset_config, data_setting, dataset_name, syn_depth):
    if dataset_config["type"] == "synthetic_prob":
        dataloader_fn = get_dataloader_prob
        data_path = dataset_config["format"].format(data_setting[:3], dataset_name)
        traindata, validdata, _, testdata  = dataloader_fn(data_path, args.red_key, syn_depth, **dataset_config["args"])
        train_aligndata, val_aligndata, _, test_aligndata = dataloader_fn(data_path, args.red_key, syn_depth, **dataset_config["args"])
    else:
        # Get the dataloader for the real dataset
        dataloader_fn = get_dataloader_fn(dataset_name)
        traindata, validdata, testdata  = dataloader_fn(dataset_config["path"], **dataset_config["args"])
        align_dataset_args = copy.deepcopy(dataset_config["args"])
        align_dataset_args["batch_size"] = args.align_bs
        train_aligndata, val_aligndata, test_aligndata = dataloader_fn(dataset_config["path"], **align_dataset_args)
    return traindata, validdata, testdata, [train_aligndata, val_aligndata, test_aligndata]

def run_align_exp(args):
    (dataset_config, _), model_config, (max_syn_depth, num_syn_depth), (base_dir, model_dir), results_path = get_configs_and_results_path(args)
    args.num_classes = dataset_config["num_classes"]
    dataset_name = dataset_config["name"]
    data_setting = dataset_config["setting"] if "setting" in dataset_config else "uni"
    model_setting = model_config["setting"]

    # Save alignment plots
    align_dir = os.path.join(base_dir, f"topk={args.topk}_q={args.q}")
    if not os.path.exists(align_dir):
        os.makedirs(align_dir)

    # If synthetic, run on multiple synthetic datasets
    if "synthetic" in dataset_config["type"]:
        # Get maximum depth of synthetic dataset
        max_syn_depth = args.max_depth if args.max_syn_depth is None else args.max_syn_depth
        num_syn_depth = max_syn_depth - args.min_depth + 1
    else:
        max_syn_depth = 1
        num_syn_depth = 1
    
    if not args.use_saved_model:
        with open(os.path.join(base_dir, 'args.json'), 'w') as fp:
            json.dump(vars(args), fp)
    
    num_models = args.max_depth - args.min_depth + 1
    model_modalnums = args.modalities
    model_epochs = args.epoch * len(model_modalnums) if len(args.epoch) == 1 else args.epoch
    model_lrs = args.lr * len(model_modalnums) if len(args.lr) == 1 else args.lr
    model_wds = args.wd * len(model_modalnums) if len(args.wd) == 1 else args.wd

    # Initialize performance matrices 
    perf_results = {}
    for key in DATA_KEYS:
        perf_results[key] = {}
        for modalnum in model_modalnums:
            perf_results[key][modalnum] = -np.ones((num_models, num_syn_depth))
            perf_results[key][modalnum] = -np.ones((num_models, num_syn_depth))
    # Initialize kwargs for mutual knn metric
    metric_kwargs = {"topk": args.topk, "q": args.q}
    pairwise_align_results = []
    pairwise_align_corr, modal_perf_summary = None, None
    # Store training logs
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%m-%d_%H:%M:%S")
    args_path = os.path.join(base_dir, f"args_{formatted_datetime}.json")
    save_args_to_file(args, args_path)
    log_path = os.path.join(base_dir, f"{model_setting[:3]}_min_{args.min_depth}_max_{args.max_depth}_syn-max_{max_syn_depth}_saved_{args.use_saved_model}_{formatted_datetime}_log.txt")
    log_mode = "a+" if args.debug else "w"
    with open(log_path, log_mode) as log_file:
        stdout = sys.stdout if args.debug else log_file
        with redirect_stdout(stdout):
            print(f"Saving results to: {results_path}")
            assert model_setting == "uni", "Fusion not implemented"
            model_save_name =  "dp_{}_d_{}"
            for syn_depth in range(args.min_depth, max_syn_depth + 1):
                # Load data
                traindata, validdata, testdata, all_aligndata = get_dataloader(args, dataset_config, 
                                                                               data_setting, dataset_name, syn_depth)
                saved_model_paths = {modalnum: [] for modalnum in model_modalnums}
                # Train models
                for depth in range(args.min_depth, args.max_depth + 1): 
                    if args.tune_only == "diag":
                        # Don't run experiments where model depth is far off from synthetic depth
                        if (depth - syn_depth) not in set([0, -1]):
                            continue
                    if args.mode == "width":
                        print(f"\nSynthetic depth: {syn_depth}, Width Multiplier: {depth}")
                    else:
                        print(f"\nSynthetic depth: {syn_depth}, Depth: {depth}")
                    seed_everything(args.base_seed + syn_depth + depth)
                    for model_idx, modalnum in enumerate(model_modalnums):
                        epochs = model_epochs[model_idx]
                        lr = model_lrs[model_idx]
                        wd = model_wds[model_idx]
                        model_hyp_dir = os.path.join(model_dir, f"modal_{modalnum}", f"lr_{lr}_wd_{wd}")
                        if not os.path.exists(model_hyp_dir):
                            os.makedirs(model_hyp_dir)
                        save_format = os.path.join(model_hyp_dir, model_save_name)
                        _, _, saved_model_path, model_perf_results = train_eval_supervised(args, traindata, validdata, testdata, save_format, model_config[modalnum], syn_depth, depth, epochs, lr, wd, modalnum, recon_modalnum=None)
                        for key, perf in zip(DATA_KEYS, model_perf_results):
                            perf_results[key][modalnum][depth - args.min_depth][syn_depth - args.min_depth] = perf
                        saved_model_paths[modalnum].append(saved_model_path)
                    np.savez(results_path, perf_results=perf_results, pairwise_align_results=pairwise_align_results)
                # Compute alignment between networks of different depths 
                if not args.disable_alignment:
                    if "synthetic" in dataset_config["type"]:
                        pairwise_align = get_pairwise_align_synthetic(saved_model_paths, args.metrics, DATA_KEYS, all_aligndata, num_models, **metric_kwargs)
                        pairwise_align_results.append(pairwise_align)
                        np.savez(results_path, perf_results=perf_results, 
                                pairwise_align_results=pairwise_align_results)
                    else:
                        pairwise_align, pairwise_align_corr, modal_perf_summary, best_modal_model = get_pairwise_align_real(saved_model_paths, args.metrics, 
                                                                                                                            DATA_KEYS, all_aligndata, num_models, perf_results, **metric_kwargs)
                        np.savez(results_path, perf_results=perf_results, pairwise_align_results=[pairwise_align], pairwise_align_corr=pairwise_align_corr, 
                                modal_perf_summary=modal_perf_summary, best_modal_model=best_modal_model)
            print("Val Performance:", perf_results["val"])
            print("Test Performance:", perf_results["test"])

    # Make performance plot
    if args.xaxis_label == "width":
        xtick_labels = [args.hidden_dim * (i ** 2) for i in range(1, num_syn_depth + 1)]
    else:
        xtick_labels = range(1, num_syn_depth + 1)
    if args.mode == "width":
        yaxis_label = "Width"
        ytick_labels = [args.hidden_dim * (i ** 2) for i in range(1, num_models + 1)]
    else:
        yaxis_label = "Depth"
        ytick_labels = range(1, num_models + 1)
    file_pref = f"{model_setting[:3]}_{dataset_name}_seed={args.base_seed}_same_{args.same_seed}_min_{args.min_depth}_max_{args.max_depth}_syn-max_{max_syn_depth}"
    perf_plot_pref = os.path.join(base_dir, file_pref)
    plot_perf_heatmaps(perf_results["train"], perf_results["val"], perf_results["test"], 
                       args.xaxis_label, xtick_labels, yaxis_label, ytick_labels, perf_plot_pref, model_modalnums)

if __name__ == "__main__":
    args = get_exp_args()
    run_align_exp(args)

