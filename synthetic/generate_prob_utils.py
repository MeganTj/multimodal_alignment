import numpy as np
import itertools
import sys 
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from utils.utils import seed_everything
import matplotlib.pyplot as plt
from synthetic.compute_mi import compute_all_marginals, compute_mi_x1_x2_y_vec, compute_mi_x1x2_y_vec, compute_mi_pairwise_vec, compute_mi_x1_x2_y, compute_mi_x1x2_y, compute_mi_pairwise

def adjust_q_y_given_x1_x2(p_y_given_x1_x2, p_x1_given_xu1_xc, p_x2_given_xu2_xc, p_xc, 
                                       p_xu1, p_xu2, desired_p_y, tol=0.01, max_iter=100):
    """
    Rescale p(y | x1, x2) such that p(y) is uniform.
    
    Parameters:
    - p_y_given_x1_x2: np.ndarray of shape (n_x1, n_x2, num_classes)
    - p_x1_given_xu1_xc: np.ndarray of shape (n_xc, n_xu1, n_x1)
    - p_x2_given_xu2_xc: np.ndarray of shape (n_xc, n_xu2, n_x2)
    - p_xc: np.ndarray of shape (n_xc,)
    - p_xu1: np.ndarray of shape (n_xu1,)
    - p_xu2: np.ndarray of shape (n_xu2,)
    - n_xc, n_xu1, n_xu2, n_x1, n_x2, num_classes: dimensions of the problem
    
    Returns:
    - p_y_given_x1_x2: Rescaled p(y | x1, x2) such that p(y) is uniform
    """

    def normalize(p_y_given_x1_x2):
        sums = np.sum(p_y_given_x1_x2, axis=2, keepdims=True)
        return np.divide(
            p_y_given_x1_x2, 
            sums, 
            out=np.zeros_like(p_y_given_x1_x2),  # where sums=0, output is set to 0
            where=(sums != 0)                   # only divide where sums != 0
        )

    p_y_given_x1_x2 = normalize(p_y_given_x1_x2)
    start_p_y = None
    for _ in range(max_iter):
        # Compute current marginal p(y)
        current_p_y = compute_all_marginals(p_y_given_x1_x2, p_x1_given_xu1_xc, 
                             p_x2_given_xu2_xc, p_xc, p_xu1, p_xu2)[2]
        if start_p_y is None:
            start_p_y = current_p_y
        # Check for convergence
        if np.allclose(current_p_y, desired_p_y, atol=tol):
            break
        
        # Compute scaling factors lambda(y)
        lambda_y = desired_p_y / current_p_y
        
        # Adjust p(y | x1, x2, xc, xu1, xu2)
        p_y_given_x1_x2 *= lambda_y[np.newaxis, np.newaxis, :]
        
        # Re-normalize (if necessary)
        p_y_given_x1_x2 = normalize(p_y_given_x1_x2)
    print(current_p_y)
    return p_y_given_x1_x2, start_p_y

def label_fun_prob(xc, xu1, xu2, linear_weights, temperature=1.0):
    """
    Generate a probability vector for the classes based on a weighted linear combination.

    Parameters:
    - xc: np.ndarray, the xc values (e.g., shared context variables)
    - xu1: np.ndarray, the xu1 values (e.g., unique to X1)
    - xu2: np.ndarray, the xu2 values (e.g., unique to X2)
    - linear_weights: np.ndarray or None, shape (num_features, num_classes), where num_features is
                      the length of the concatenated input vector [xc, xu1, xu2].
                      If None, defaults to uniform weights mapping to num_classes logits.

    Returns:
    - probabilities: np.ndarray, shape (num_classes,), a valid probability vector summing to 1
    """
    # Concatenate [xc, xu1, xu2] into a single input vector
    input_vector = np.concatenate([xc, xu1, xu2])
    # Compute logits by applying the weights to the input vector
    logits = input_vector @ linear_weights  # Shape: (num_classes,)

    # Scale logits by the temperature
    scaled_logits = logits / temperature

    # Map scaled logits to probabilities using softmax
    exp_logits = np.exp(scaled_logits)  # For numerical stability
    probabilities = exp_logits / exp_logits.sum()  # Normalize to get probabilities
    
    return probabilities

def initialize_balanced_weights(total_features, num_classes, weights_seed, scale=1.0):
    """
    Initialize weights to ensure balanced classes and balanced contributions 
    from each feature across classes.

    Parameters:
    - xc_size: int, size of shared context (X_c).
    - xu1_size: int, size of unique features for X1 (X_u1).
    - xu2_size: int, size of unique features for X2 (X_u2).
    - num_classes: int, number of output classes.
    - scale: float, scaling factor for weight magnitudes.

    Returns:
    - weights: np.ndarray, shape (xc_size + xu1_size + xu2_size, num_classes).
    """
    seed_everything(weights_seed)
    # Step 1: Generate random weights
    weights = np.random.uniform(0, scale, size=(total_features, num_classes))

    # Step 2: Normalize rows to ensure balanced feature contributions across classes
    row_norms = np.linalg.norm(weights, axis=1, keepdims=True)
    assert np.all(row_norms > 0)
    weights = weights / row_norms

    return weights

def plot_mi(all_mi, title, seed, save_dir):
    # Create the plot
    plt.figure(figsize=(8, 6))
    if isinstance(all_mi, dict):
        for key, mi_values in all_mi.items():
            x_values = range(1, len(mi_values) + 1)  # Example: 1 to N
            plt.plot(x_values, mi_values, marker='o', linestyle='-', label=key)
    else:
        x_values = range(1, len(all_mi) + 1)  # Example: 1 to N
        plt.plot(x_values, all_mi, marker='o', linestyle='-')
    plt.title(f"{title}", fontsize=14)
    plt.ylabel("Mutual Information", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"seed={seed}_{title}.png"))
    plt.close()

def generate_all_xs(shared_feature_dim, unique_feature_dim):
    all_xc = list(itertools.product([0, 1], repeat=shared_feature_dim))
    all_xu1 = list(itertools.product([0, 1], repeat=unique_feature_dim))
    all_xu2 = list(itertools.product([0, 1], repeat=unique_feature_dim))
    # Gives indices (xc, xu1)
    all_x1 = list(itertools.product(*[range(len(v)) for v in [all_xc, all_xu1]]))
    # indices are (xc, xu2)
    all_x2= list(itertools.product(*[range(len(v)) for v in [all_xc, all_xu2]]))
    all_combined_indices = list(itertools.product(*[range(len(v)) for v in [all_xc, all_xu1, all_xu2]]))
    return all_xc, all_xu1, all_xu2, all_x1, all_x2, all_combined_indices

def get_data_distr_estimate_mi(all_xc, all_xu1, all_xu2, all_x1, all_x2, all_combined_indices, num_classes, 
              feature_dim, temp, red_rule=None, u1_rule=None, u2_rule=None, weights_seed=0):
    n_xc = len(all_xc)
    n_xu1 = len(all_xu1)
    n_xu2 = len(all_xu2)
    n_x1 = len(all_x1)
    n_x2 = len(all_x2)
    # Desired p(y) is uniform
    uniform_p_y = np.ones(num_classes) / num_classes

    p_xc = np.ones(n_xc) / n_xc  # (n_xc,)
    p_xu1 = np.ones(n_xu1) / n_xu1  # (n_xu1,)
    p_xu2 = np.ones(n_xu2) / n_xu2 # (n_xu2,)
    linear_weights = initialize_balanced_weights(feature_dim, num_classes, weights_seed)

    q_y_given_x1_x2 = np.zeros(
        (n_x1, n_x2, num_classes)
    ) 
    p_x1_given_xc_xu1 = np.zeros(
        (n_xc, n_xu1, n_x1)
    ) 
    p_x2_given_xc_xu2 = np.zeros(
        (n_xc, n_xu2, n_x2)
    ) 
    for xc, xu1, xu2 in all_combined_indices:
        x1 = xc * n_xu1 + xu1
        x2 = xc * n_xu2 + xu2
        xc_val = np.array(all_xc[xc])
        xu1_val = np.array(all_xu1[xu1])
        xu2_val = np.array(all_xu2[xu2])
        if red_rule is not None:
            xc_val = xc_val[red_rule]
        if u1_rule is not None:
            xu1_val = xu1_val[u1_rule]
        if u2_rule is not None:
            xu2_val = xu2_val[u2_rule]
        assert all_x1[x1] == (xc, xu1)
        assert all_x2[x2] == (xc, xu2)
        q_y_given_x1_x2[x1, x2] = label_fun_prob(xc_val, xu1_val, xu2_val, linear_weights,
        temperature=temp)
        p_x1_given_xc_xu1[xc, xu1, x1] = 1
        p_x2_given_xc_xu2[xc, xu2, x2] = 1

    # Reweight q_y_given_x1_x2
    p_y_given_x1_x2, _ = adjust_q_y_given_x1_x2(q_y_given_x1_x2, p_x1_given_xc_xu1, p_x2_given_xc_xu2, p_xc, p_xu1, p_xu2, uniform_p_y)
    p_x1, p_x2, p_y, p_x1_x2, p_x1_y, p_x2_y, p_x1_x2_y = compute_all_marginals(p_y_given_x1_x2, p_x1_given_xc_xu1,
                                                                                    p_x2_given_xc_xu2, p_xc, p_xu1, p_xu2)
    mi_x1_x2_y = compute_mi_x1_x2_y_vec(p_x1, p_x2, p_y, p_x1_y, p_x2_y, p_x1_x2, p_x1_x2_y)
    mi_x1x2_y= compute_mi_x1x2_y_vec(p_y, p_x1_x2, p_x1_x2_y)
    mi_pairwise = compute_mi_pairwise_vec(p_x1, p_x2, p_y, p_x1_y, p_x2_y, p_x1_x2)
    mi_dct = {
        "mi_x1_x2_y": mi_x1_x2_y,
        "mi_x1x2_y": mi_x1x2_y,
    } | mi_pairwise
    return p_xc, p_xu1, p_xu2, p_y_given_x1_x2, mi_dct

    
def vary_beta(all_xc, all_xu1, all_xu2, all_x1, all_x2, all_combined_indices, num_classes, 
              feature_dim, base_temp=0.1, red_rule=None, u1_rule=None, u2_rule=None, weights_seed=0, save_dir="mi"):
    n_xc = len(all_xc)
    n_xu1 = len(all_xu1)
    n_xu2 = len(all_xu2)
    n_x1 = len(all_x1)
    n_x2 = len(all_x2)
    # Desired p(y) is uniform
    uniform_p_y = np.ones(num_classes) / num_classes

    p_xc = np.ones(n_xc) / n_xc  # (n_xc,)
    p_xu1 = np.ones(n_xu1) / n_xu1  # (n_xu1,)
    p_xu2 = np.ones(n_xu2) / n_xu2 # (n_xu2,)
    linear_weights = initialize_balanced_weights(feature_dim, num_classes, weights_seed)
    all_mi_x1_x2_y = []
    all_mi_x1x2_y = []
    all_mi_pairwise = {
        "mi_x1_y": [], 
        "mi_x2_y": [], 
        "mi_x1_x2": []
    }
    # Exponential scaling of temperature
    scaling_factors = np.linspace(0, 2.0, 10)  # Scale from 0 to 2.0
    temperatures = base_temp * np.exp(scaling_factors)  # Exponentially increase temperature


    for temp in temperatures:
        q_y_given_x1_x2 = np.zeros(
            (n_x1, n_x2, num_classes)
        ) 
        p_x1_given_xc_xu1 = np.zeros(
            (n_xc, n_xu1, n_x1)
        ) 
        p_x2_given_xc_xu2 = np.zeros(
            (n_xc, n_xu2, n_x2)
        ) 
        for xc, xu1, xu2 in all_combined_indices:
            x1 = xc * n_xu1 + xu1
            x2 = xc * n_xu2 + xu2
            xc_val = np.array(all_xc[xc])
            xu1_val = np.array(all_xu1[xu1])
            xu2_val = np.array(all_xu2[xu2])
            if red_rule is not None:
                xc_val = xc_val[red_rule]
            if u1_rule is not None:
                xu1_val = xu1_val[u1_rule]
            if u2_rule is not None:
                xu2_val = xu2_val[u2_rule]
            assert all_x1[x1] == (xc, xu1)
            assert all_x2[x2] == (xc, xu2)
            # class_idx = label_fun(xc_val, xu1_val, xu2_val, num_classes)
            # q_y_given_x1_x2[x1, x2, class_idx] = 1
            q_y_given_x1_x2[x1, x2] = label_fun_prob(xc_val, xu1_val, xu2_val, linear_weights,
            temperature=temp)
            p_x1_given_xc_xu1[xc, xu1, x1] = 1
            p_x2_given_xc_xu2[xc, xu2, x2] = 1

        # Reweight q_y_given_x1_x2 to get uniform class distribution
        p_y_given_x1_x2, _ = adjust_q_y_given_x1_x2(q_y_given_x1_x2, p_x1_given_xc_xu1, p_x2_given_xc_xu2, p_xc, p_xu1, p_xu2, uniform_p_y)
        p_x1, p_x2, p_y, p_x1_x2, p_x1_y, p_x2_y, p_x1_x2_y = compute_all_marginals(p_y_given_x1_x2, p_x1_given_xc_xu1,
                                                                                     p_x2_given_xc_xu2, p_xc, p_xu1, p_xu2)
        # Compute mutual information
        mi_x1_x2_y = compute_mi_x1_x2_y_vec(p_x1, p_x2, p_y, p_x1_y, p_x2_y, p_x1_x2, p_x1_x2_y)
        mi_x1x2_y= compute_mi_x1x2_y_vec(p_y, p_x1_x2, p_x1_x2_y)
        mi_pairwise = compute_mi_pairwise_vec(p_x1, p_x2, p_y, p_x1_y, p_x2_y, p_x1_x2)
        all_mi_x1_x2_y.append(mi_x1_x2_y)
        all_mi_x1x2_y.append(mi_x1x2_y)
        for key, value in mi_pairwise.items():
            all_mi_pairwise[key].append(value)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plot_mi(all_mi_x1_x2_y, "mi_x1_x2_y", weights_seed, save_dir)
    plot_mi(all_mi_x1x2_y, "mi_x1x2_y", weights_seed, save_dir)
    plot_mi(all_mi_pairwise, "mi_pairwise", weights_seed, save_dir)

def get_feature_subsets(red_rule, u1_rule, u2_rule, shared_feature_dim, unique_feature_dim, features_seed=0):
    seed_everything(features_seed)
    if isinstance(red_rule, int):
        red_rule = np.random.choice(np.arange(shared_feature_dim), size=red_rule, replace=False)
    if isinstance(u1_rule, int):
        u1_rule = np.random.choice(np.arange(unique_feature_dim), size=u1_rule, replace=False)
    if isinstance(u2_rule, int):
        u2_rule = np.random.choice(np.arange(unique_feature_dim), size=u2_rule, replace=False)
    return red_rule, u1_rule, u2_rule

def test_small():
    shared_feature_dim = 2
    unique_feature_dim = 2
    num_classes = 3
    feature_dim = 6

    all_xc, all_xu1, all_xu2, all_x1, all_x2, all_combined_indices = generate_all_xs(shared_feature_dim, unique_feature_dim)
    for weights_seed in range(0, 10):
        vary_beta(all_xc, all_xu1, all_xu2, all_x1, all_x2, all_combined_indices, num_classes,
                feature_dim,  weights_seed=weights_seed, 
                save_dir=f"mi_test")

if __name__ == "__main__":
    # Test with small feature sizes
    test_small()

    # Test with redundancy/uniqueness. 
    shared_feature_dim = 8
    unique_feature_dim = 4
    num_classes = 4
    feature_dim = 8
    weight_seed_start = 1
    weight_seed_end = 3
    for red in range(0, feature_dim + 1):
        total_u = feature_dim - red
        u1 = total_u // 2
        u2 = feature_dim - (u1 + red)
        red_rule, u1_rule, u2_rule = get_feature_subsets(red, u1, u2, shared_feature_dim, unique_feature_dim)
        print(red_rule, u1_rule, u2_rule)
        all_xc, all_xu1, all_xu2, all_x1, all_x2, all_combined_indices = generate_all_xs(shared_feature_dim, unique_feature_dim)
        for weights_seed in range(weight_seed_start, weight_seed_end):
            vary_beta(all_xc, all_xu1, all_xu2, all_x1, all_x2, all_combined_indices, num_classes,
                    feature_dim,  base_temp=0.05, red_rule=red_rule, u1_rule=u1_rule, u2_rule=u2_rule, 
                    weights_seed=weights_seed, save_dir=f"total_{feature_dim}_r_{red}_u_{total_u}_mi")