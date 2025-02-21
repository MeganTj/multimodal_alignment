from collections import defaultdict
import torch
import os
import numpy as np
import argparse
import sys

sys.path.append(os.getcwd())
import yaml
from datasets.get_dataset_dataloader import get_dataloader_fn
from get_alignment import get_alignment_all_layers
from utils.utils import seed_everything
import copy


def log_pairwise_metric_align_summary(pairwise_align_corr, logfile):
    with open(logfile, "w") as f:
        for data_key, align_corr in pairwise_align_corr.items():
            print(data_key, file=f)
            for depth_key, align_corr_arr in sorted(align_corr.items()):
                print(f"\tModality {depth_key}, Alignment/Performance", file=f)
                for metric, align_summary in align_corr_arr.items():
                    print(f"\t{metric}", file=f)
                    print(f"\t\tBest Over Layers: {align_summary['best']:.3f}", file=f)

def get_pairwise_align_real(best_model_paths, all_model_paths, metrics, data_keys, all_aligndata, all_perf_results, **kwargs):
    # Cmpute the pairwise align only for the best models 
    assert len(best_model_paths) == 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pairwise_align = {data_key: {} for data_key in data_keys}
    pairwise_align_corr = {}
    # Iterate over all splits of the data
    for data_key, aligndata in zip(data_keys, all_aligndata):
        if "val" in data_key:
            continue
        pairwise_align_corr[data_key] = {}
        for fixed_key, best_model in best_model_paths.items():
            pairwise_align[data_key][fixed_key] = {}
            for depth_key in all_model_paths.keys():
                if depth_key != fixed_key:
                    pairwise_align_corr[data_key][depth_key] = {}
                    pairwise_align[data_key][fixed_key][depth_key] = {}
                    depth_modal = all_model_paths[depth_key]
                    for metric in metrics:
                        pairwise_align_corr[data_key][depth_key][metric] = {}
                        pairwise_align[data_key][fixed_key][depth_key][metric] = {"best": np.zeros((len(depth_modal))), 
                                                                                "mean": np.zeros((len(depth_modal)))}
                        for depth_idx, depth_modal_model in enumerate(depth_modal):
                            fixed_encoder = torch.load(best_model + "_final_encoder.pt").to(device)
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
                        perf_results = np.concatenate(all_perf_results[data_key][depth_key]).squeeze()
                        pairwise_align_metric = pairwise_align[data_key][fixed_key][depth_key][metric]
                        pairwise_align_corr[data_key][depth_key][metric]["best"] = np.corrcoef(pairwise_align_metric["best"], perf_results)[0, 1]

    return pairwise_align, pairwise_align_corr

def get_exp_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp-dir", default="experiments/hyp", type=str)

    parser.add_argument("--exp-config", type=str, help="path to config")
    parser.add_argument("--results-dir", default="experiments/syn_align", type=str)
    parser.add_argument("--metrics",  nargs='+', default=[ "unbiased_cka"], type=str, help="list of metrics to compute alignment")

    parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
    parser.add_argument("--seeds", nargs='+', default=[2, 22, 42], type=int, help="specify the index of modalities in keys")
    parser.add_argument("--align-bs", default=4096, type=int)
    parser.add_argument("--min-depth", default=1, type=int, help="Min depth of synthetic dataset")
    parser.add_argument("--max-depth", default=10, type=int, help="Max depth of synthetic dataset")
    args = parser.parse_known_args()[0]
    return args

def get_model_dir(args, config, seed):
    dataset_config = config["dataset"]
    model_config = config["model"]
    assert "synthetic" not in dataset_config["type"]
    full_dataset_name = dataset_config['name'] 
    config_name = os.path.basename(args.exp_config)[:-5]
    model_dir = os.path.join(args.results_dir, full_dataset_name, f"{model_config['setting'][:3]}_models_seed_{seed}", f"cfg_{config_name}") 
    return model_dir

def get_summary_path(args, config):
    experiments_dir = os.path.dirname(args.results_dir)
    summary_dir = os.path.join(experiments_dir, "pairwise_results", config["dataset"]["name"])
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    mod_str = '_'.join([str(modality) for modality in args.modalities])
    return summary_dir, os.path.join(summary_dir, f"mod_{mod_str}_pairwise_summary.txt")

def get_all_aligndata(config):
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"]
    dataloader_fn = get_dataloader_fn(dataset_name)
    align_dataset_args = copy.deepcopy(dataset_config["args"])
    align_dataset_args["batch_size"] = args.align_bs
    train_aligndata, val_aligndata, test_aligndata = dataloader_fn(dataset_config["path"], **align_dataset_args)
    return [train_aligndata, val_aligndata, test_aligndata]

def run_eval(args):
    with open(args.exp_config, 'r') as stream:
        config = yaml.safe_load(stream)
    best_val_score = {modal: 0 for modal in args.modalities}
    best_test_score = {modal: 0 for modal in args.modalities}
    best_model_paths = {modal: None for modal in args.modalities}
    all_model_paths = {modal: [] for modal in args.modalities}
    data_keys = ["train", "val", "test"]
    all_perf_results = {key: {modal: [] for modal in args.modalities} for key in data_keys}
    mod_str = '_'.join([str(modality) for modality in args.modalities])
    for seed in args.seeds:
        model_dir = get_model_dir(args, config, seed)
        config_name = os.path.basename(args.exp_config)[:-5]
        full_hyp_dir = os.path.join(args.hyp_dir, config_name)
        result_file = f"{config_name}_mod_{mod_str}_sha_Fal_seed_{seed}_final.yaml"
        hyp_file = f"{config_name}_mod_{mod_str}.yaml"
        # This will remain -1 if the best model isn't from the current seed
        best_modal_depth = {modal: -1 for modal in args.modalities}
        lr = None
        wd = None
        with open(os.path.join(full_hyp_dir, hyp_file), "r") as stream:
            dct = yaml.safe_load(stream)
            # Get the lr and wd for the modalities
            lr = dct["lr"][:-1]
            wd = dct["wd"][:-1]
        with open(os.path.join(full_hyp_dir, result_file), "r") as stream:
            results_path = yaml.safe_load(stream)["results_path"]
            results=np.load(results_path, allow_pickle=True)
            perf_results = val_results=results['perf_results'].item()
            for key in data_keys:
                for modalnum, modal_results in perf_results[key].items():
                    assert not np.any(modal_results < 0) 
                    all_perf_results[key][modalnum].append(modal_results)
            val_results=perf_results['val']
            for modalnum, modal_val_results in val_results.items():
                curr_best_indices = np.argmax(modal_val_results)
                curr_best_val_score = modal_val_results[curr_best_indices].item()
                if curr_best_val_score > best_val_score[modalnum]:
                    best_val_score[modalnum] = curr_best_val_score
                    best_test_score[modalnum] = perf_results['test'][modalnum][curr_best_indices].item()
                    best_modal_depth[modalnum] = curr_best_indices + 1 # The minimum depth is 1ß
        for modal_idx, modalnum in enumerate(args.modalities):
            for depth in range(args.min_depth, args.max_depth + 1): 
                modal_lr = lr[modal_idx]
                modal_wd = wd[modal_idx]
                model_hyp_dir = os.path.join(model_dir, f"modal_{modalnum}", f"lr_{modal_lr}_wd_{modal_wd}")
                model_path = os.path.join(model_hyp_dir, f"dp_1_d_{depth}")
                all_model_paths[modalnum].append(model_path)
                if depth == best_modal_depth[modalnum]:
                    best_model_paths[modalnum] = model_path
    # Compute pairwise alignment
    seed_everything(0)
    aligndata = get_all_aligndata(config)
    summary_dir, summary_path = get_summary_path(args, config)
    # Save the best model paths
    for modalnum in args.modalities:
        with open(os.path.join(summary_dir, f"best_models_mod_{modalnum}.yaml"), "w") as f:
            save_dct = {"model_path": best_model_paths[modalnum], 
                        "val_score": best_val_score[modalnum], 
                        "test_score": best_test_score[modalnum]}
            yaml.dump(save_dct, f)
    _, pairwise_align_corr = get_pairwise_align_real(best_model_paths, all_model_paths, args.metrics, data_keys, aligndata, all_perf_results)
    # Save the logs
    log_pairwise_metric_align_summary(pairwise_align_corr, summary_path)


if __name__ == "__main__":
    args = get_exp_args()
    run_eval(args)
