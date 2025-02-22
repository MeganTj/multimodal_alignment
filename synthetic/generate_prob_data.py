import numpy as np
import pickle
import os
import math
import torch
import argparse
from torch import nn
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from utils.utils import seed_everything
from synthetic.generate_prob_utils import plot_mi, generate_all_xs, get_data_distr_estimate_mi, get_feature_subsets
from sklearn.model_selection import train_test_split
import yaml
import copy


def save_data(data, filename):
    with open(os.path.join(args.out_path, filename), 'wb') as f:
        pickle.dump(data, f)

def sample_data(p_xc, p_xu1, p_xu2, p_y_given_x1_x2, all_xc, all_xu1, all_xu2, num_data):
    # First sample xc, xu1, xu2 from their distributions
    xc = np.random.choice(len(p_xc), size=num_data, p=p_xc)  # Sampling xc from p_xc
    xu1 = np.random.choice(len(p_xu1), size=num_data, p=p_xu1)  # Sampling xu1 from p_xu1
    xu2 = np.random.choice(len(p_xu2), size=num_data, p=p_xu2)  # Sampling xu2 from p_xu2

    _, _, num_classes = p_y_given_x1_x2.shape
    x1_vals = []
    x2_vals = []
    labels = []
    for curr_xc, curr_xu1, curr_xu2 in zip(xc, xu1, xu2):
        curr_x1_val = np.concatenate((all_xc[curr_xc], all_xu1[curr_xu1]))
        curr_x2_val = np.concatenate((all_xc[curr_xc], all_xu2[curr_xu2]))
        # Compute x1, x2 deterministically
        curr_x1 = curr_xc * len(p_xu1) + curr_xu1
        curr_x2 = curr_xc * len(p_xu2) + curr_xu2
        curr_label = np.random.choice(num_classes, p=p_y_given_x1_x2[curr_x1, curr_x2]) 
        x1_vals.append(curr_x1_val)
        x2_vals.append(curr_x2_val)
        labels.append(curr_label)       
    # Sample y from q_y_given_x1_x2 for each (x1, x2)
    x1_vals = np.array(x1_vals)
    x2_vals = np.array(x2_vals)
    labels = np.array(labels)
    return [x1_vals, x2_vals], labels

def get_transformed_data(xs, labels, transforms):
    transformed_data = []
    for idx in range(len(xs)):
        modality_data = transforms[idx](torch.Tensor(np.array(xs[idx]))).detach().numpy()
        transformed_data.append(modality_data)
    # result is (# examples, # modalities, # features)
    transformed_data = np.array(transformed_data).transpose((1,0,2))
    data = dict()
    data['train'] = dict()
    data['valid'] = dict()
    data['test'] = dict()
    X_train, X_test, y_train, y_test = train_test_split(transformed_data, labels, test_size=0.3, stratify=labels)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)
    for i in range(len(xs)):
        data['train'][str(i)] = np.array(X_train)[:,i,:]
        data['valid'][str(i)] = np.array(X_valid)[:,i,:]
        data['test'][str(i)] = np.array(X_test)[:,i,:]
    data['train']['label'] = np.array(y_train)
    data['valid']['label'] = np.array(y_valid)
    data['test']['label'] = np.array(y_test)
    return data

def get_transforms(total_feature_dim, depth=None, mode="depth", nonlin="tanh"):
    transforms = []
    # Get transformations on the two modalities
    for num_transform in depth:
        if num_transform == -1:
            transform = nn.Identity()
        else:
            if mode == "width":
                hidden_width = max(total_feature_dim * (num_transform ** 2), total_feature_dim)
                if nonlin == "tanh":
                    nonlin_type = nn.Tanh
                elif nonlin == "sigmoid":
                    nonlin_type = nn.Sigmoid
                else:
                    raise NotImplementedError
                transform = nn.Sequential(
                    nn.Linear(total_feature_dim, hidden_width),
                    nonlin_type(),
                    nn.Linear(hidden_width, total_feature_dim)
                )
            else:
                seq = [nn.Linear(total_feature_dim, total_feature_dim)]
                for _ in range(0, num_transform):
                    if nonlin == "tanh":
                        seq.append(nn.Tanh())
                    elif nonlin == "sigmoid":
                        seq.append(nn.Sigmoid())
                    else:
                        raise NotImplementedError
                    seq.append(nn.Linear(total_feature_dim, total_feature_dim))
                transform = nn.Sequential(*seq)
        transforms.append(transform)
    return transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-path", default=None, type=str, help="Path to template config. Used for R, U, S generation")
    parser.add_argument('--num-mi', default=1, type=int)
    parser.add_argument('--config-dir', default='synthetic/configs', type=str)
    parser.add_argument('--copy-data-dir', default=None, type=str)
    parser.add_argument('--out-path', default='synthetic/data', type=str)
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    with open(args.template_path, 'r') as stream:
        template_config = yaml.safe_load(stream)
    shared_feature_dim = template_config["xs"]["shared_feature_dim"]
    unique_feature_dim = template_config["xs"]["unique_feature_dim"]
    total_feature_dim = template_config["xs"]["shared_feature_dim"] + template_config["xs"]["unique_feature_dim"]
    num_classes = template_config["num_classes"]
    # Use the same xs and transforms for all depths
    all_transforms = []
    all_depths = []
    seed_everything(template_config["transforms_seed"])
    for depth in range(template_config["min_depth"], template_config["max_depth"]+ 1):
        transforms_config = copy.deepcopy(template_config["transforms"])
        if template_config["het_setting"] == "uni":
            transforms_config["depth"] = [-1, depth]
        elif template_config["het_setting"] == "multi":
            transforms_config["depth"]  = [depth, depth]
        all_depths.append(depth)
        all_transforms.append(get_transforms(total_feature_dim, **transforms_config))
    
    # Exponential scaling of temperature
    scaling_factors = np.linspace(0, 2.0, 10)[:args.num_mi]  # Scale from 0 to 2.0
    print(scaling_factors)
    temperatures = template_config["base_temp"] * np.exp(scaling_factors)  # Exponentially increase temperature
    feature_dim = template_config["max_red"]
    # For every MI level, generate datasets with varying levels of redundancy
    # Then for each level of redundancy, generate datasets with varying levels of heterogeneity
    all_mi_x1_x2_y = []
    for temp in temperatures: 
        mi_data_dct = {}
        all_mis = {}
        all_xc, all_xu1, all_xu2, all_x1, all_x2, all_combined_indices = generate_all_xs(shared_feature_dim, unique_feature_dim)
        mi_x1x2_y = None
        for red in range(template_config["min_red"], template_config["max_red"] + 1):
            total_u = feature_dim - red
            u1 = total_u // 2
            u2 = feature_dim - (u1 + red)
            red_rule, u1_rule, u2_rule = get_feature_subsets(red, u1, u2, shared_feature_dim, unique_feature_dim, features_seed=template_config["features_seed"])
            print(total_u, u1, u2)
            red_key = f"feat_{feature_dim}_r_{red}"
            mi_data_dct[red_key] = {}
            all_mis[red_key] = {}
            p_xc, p_xu1, p_xu2, p_y_given_x1_x2, mi_dct = get_data_distr_estimate_mi(all_xc, all_xu1, all_xu2, all_x1, all_x2, 
                                                                                    all_combined_indices, num_classes, feature_dim,
                                                                                    temp, red_rule=red_rule, u1_rule=u1_rule, u2_rule=u2_rule, 
                                                                                    weights_seed=template_config["weights_seed"])
            if mi_x1x2_y is None:
                mi_x1x2_y = mi_dct["mi_x1x2_y"]
            assert (mi_x1x2_y - mi_dct["mi_x1x2_y"]) < 1e-5
            all_mis[red_key] = mi_dct
            xs, labels = sample_data(p_xc, p_xu1, p_xu2, p_y_given_x1_x2, all_xc, all_xu1, all_xu2, template_config["xs"]["num_data"])
            for depth, transforms in zip(all_depths, all_transforms):
                data_dct = get_transformed_data(xs, labels, transforms)
                mi_data_dct[red_key][depth] = data_dct
        seeds_str = f"{template_config['transforms_seed']}_{template_config['features_seed']}_{template_config['weights_seed']}"
        save_path = f"{template_config['het_setting'][:3]}_DATA_mi={mi_x1x2_y:.3f}_seeds_{seeds_str}.pickle"
        if args.copy_data_dir is not None:
            copy_data_path = os.path.join(args.copy_data_dir, save_path)
            with open(copy_data_path, 'rb') as f:
                copy_data = pickle.load(f)
            copy_data_dct = copy_data["data"]
            
            for red_key, red_data_dct in copy_data_dct.items():
                for depth, _ in red_data_dct.items():
                    for data_split in ["train", "valid", "test"]:
                        for mod in ['0', '1']:
                            assert np.all(copy_data_dct[red_key][depth][data_split][mod] == mi_data_dct[red_key][depth][data_split][mod])
            
            print(f"Determinism check passed for: {save_path}")
        data_with_metadata = {
            "data": mi_data_dct,
            "metadata": all_mis
        }
        all_mi_x1_x2_y.append(mi_x1x2_y)
        save_data(data_with_metadata, save_path)
    plot_save_dir = "generated_mi"
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
    plot_mi(all_mi_x1_x2_y, "mi_x1_x2_y", template_config["weights_seed"], plot_save_dir)