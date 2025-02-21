import numpy as np

def compute_all_marginals(p_y_given_x1_x2, p_x1_given_xc_xu1, p_x2_given_xc_xu2, p_xc, p_xu1, p_xu2):
    """
    Compute all marginals: p(y), p(x1, x2), p(x1, y), p(x2, y), p(x1, x2, y), p(x1), p(x2)
    by looping over all variables (xc, xu1, xu2, x1, x2, y).
    """
    n_x1, n_x2, num_classes = p_y_given_x1_x2.shape
    n_xc, = p_xc.shape
    n_xu1, = p_xu1.shape
    n_xu2, = p_xu2.shape
    
    # Initialize all marginals
    p_x1 = np.zeros(n_x1)  # p(x1)
    p_x2 = np.zeros(n_x2)  # p(x2)
    p_y = np.zeros(num_classes)  # p(y)
    p_x1_x2 = np.zeros((n_x1, n_x2))  # p(x1, x2)
    p_x1_y = np.zeros((n_x1, num_classes))  # p(x1, y)
    p_x2_y = np.zeros((n_x2, num_classes))  # p(x2, y)
    p_x1_x2_y = np.zeros((n_x1, n_x2, num_classes))  # p(x1, x2, y)
    # Precompute the constant part p(xc) * p(xu1) * p(xu2) for each combination of (xc, xu1, xu2)
    # IMPORTANT: assumes p_xc, p_xu1, p_xu2 are uniforms
    p_xc_xu1_xu2 = p_xc[0] * p_xu1[0] * p_xu2[0]
    
    for xc in range(n_xc):
        for xu1 in range(n_xu1):
            x1 = xc * n_xu1 + xu1
            for xu2 in range(n_xu2):
                x2 = xc * n_xu2 + xu2
                # prod_mar = p_xc_xu1_xu2[xc, xu1, xu2]
                for y in range(num_classes):
                    assert p_x1_given_xc_xu1[xc, xu1, x1] == 1.0
                    assert p_x2_given_xc_xu2[xc, xu2, x2] == 1.0

                    joint_prob = p_y_given_x1_x2[x1, x2, y] * p_x1_given_xc_xu1[xc, xu1, x1] * p_x2_given_xc_xu2[xc, xu2, x2] * p_xc_xu1_xu2
                    
                    # Compute all marginals by accumulating the relevant terms
                    p_y[y] += joint_prob
                    p_x1_x2[x1, x2] += joint_prob
                    p_x1_y[x1, y] += joint_prob  # Update p(x1, y) correctly with joint_prob
                    p_x2_y[x2, y] += joint_prob  # Update p(x2, y) correctly with joint_prob
                    p_x1_x2_y[x1, x2, y] += joint_prob  # Update p(x1, x2, y) correctly with joint_prob
                    p_x1[x1] += joint_prob  # Update p(x1) correctly with joint_prob
                    p_x2[x2] += joint_prob  # Update p(x2) correctly with joint_prob


    return p_x1, p_x2, p_y, p_x1_x2, p_x1_y, p_x2_y, p_x1_x2_y 

def compute_mi_x1_x2_y_vec(p_x1, p_x2, p_y, p_x1_y, p_x2_y, p_x1_x2, p_x1_x2_y, eps=1e-5):
    """
    Optimized computation of mutual information I(X1, X2; Y) from a joint distribution,
    by removing explicit for loops and using vectorized operations.
    
    Parameters:
    ----------
    p_x1 (np.ndarray): Marginal distribution p(x1).
    p_x2 (np.ndarray): Marginal distribution p(x2).
    p_y (np.ndarray): Marginal distribution p(y).
    p_x1_y (np.ndarray): Joint distribution p(x1, y).
    p_x2_y (np.ndarray): Joint distribution p(x2, y).
    p_x1_x2 (np.ndarray): Joint distribution p(x1, x2).
    p_x1_x2_y (np.ndarray): Joint distribution p(x1, x2, y).
    eps (float): Small epsilon for numerical stability and normalization checks.
    
    Returns:
    -------
    float: The computed mutual information I(X1, X2; Y).
    """

    # Ensure probabilities are normalized
    assert abs(p_x1.sum() - 1.0) < eps, "p_x1 must sum to 1."
    assert abs(p_x2.sum() - 1.0) < eps, "p_x2 must sum to 1."
    assert abs(p_y.sum() - 1.0) < eps, "p_y must sum to 1."
    assert abs(p_x1_x2.sum() - 1.0) < eps, "p_x1_x2 must sum to 1."
    assert abs(p_x1_y.sum() - 1.0) < eps, "p_x1_y must sum to 1."
    assert abs(p_x2_y.sum() - 1.0) < eps, "p_x2_y must sum to 1."
    assert abs(p_x1_x2_y.sum() - 1.0) < eps, "p_x1_x2_y must sum to 1."

    # Broadcasting the values of p_x1, p_x2, p_y, p_x1_y, p_x2_y, p_x1_x2 to vectorize the computation
    # We can avoid explicit loops by performing the operations element-wise

    # Ensure the shape of each variable is compatible with broadcasting
    p_x1 = p_x1[:, np.newaxis, np.newaxis]  # Shape: (n_x1, 1, 1)
    p_x2 = p_x2[np.newaxis, :, np.newaxis]  # Shape: (1, n_x2, 1)
    p_y = p_y[np.newaxis, np.newaxis, :]    # Shape: (1, 1, num_classes)

    p_x1_y = p_x1_y[:, np.newaxis, :]       # Shape: (n_x1, 1, num_classes)
    p_x2_y = p_x2_y[np.newaxis, :, :]       # Shape: (1, n_x2, num_classes)
    p_x1_x2 = p_x1_x2[:, :, np.newaxis]     # Shape: (n_x1, n_x2, 1)

    # Calculate the numerator and denominator for mutual information
    numerator = p_x1_y * p_x1_x2 * p_x2_y  # Shape: (n_x1, n_x2, num_classes)
    denominator = p_x1 * p_x2 * p_y * p_x1_x2_y  # Shape: (n_x1, n_x2, num_classes)
    valid_mask = (numerator > 0) & (denominator > 0)
    mutual_information = np.sum(p_x1_x2_y[valid_mask] * np.log(numerator[valid_mask] / denominator[valid_mask] ))

    return mutual_information


import numpy as np

def compute_mi_x1x2_y_vec(p_y, p_x1_x2, p_x1_x2_y, eps=1e-5):
    """
    Computes the mutual information I(X1, X2; Y) from a joint distribution
    p(xc, xu1, xu2, x1, x2, y).

    Parameters
    ----------
    p_y : np.ndarray
        A 1D numpy array of shape (num_classes,) representing p(y).
    p_x1_x2 : np.ndarray
        A 2D numpy array of shape (n_x1, n_x2) representing p(x1, x2).
    p_x1_x2_y : np.ndarray
        A 3D numpy array of shape (n_x1, n_x2, num_classes) representing p(x1, x2, y).
    eps : float
        A small epsilon for numerical stability and assertions.

    Returns
    -------
    float
        The mutual information I(X1, X2; Y).
    """

    # Check normalization
    assert abs(p_x1_x2.sum() - 1.0) < eps, "p_x1_x2 must sum to 1."
    assert abs(p_y.sum() - 1.0) < eps, "p_y must sum to 1."
    assert abs(p_x1_x2_y.sum() - 1.0) < eps, "p_x1_x2_y must sum to 1."

    # Compute the product p(x1, x2) * p(y) using broadcasting
    p_prod = p_x1_x2[:, :, np.newaxis] * p_y[np.newaxis, np.newaxis, :]  # Shape: (n_x1, n_x2, num_classes)
    # Mask out invalid values where p_x1_x2_y or p_prod are zero
    mask = (p_x1_x2_y > 0) & (p_prod > 0)


    # Compute mutual information by summing over the valid entries
    mutual_information = np.sum(p_x1_x2_y[mask] * np.log(p_x1_x2_y[mask] / p_prod[mask]))

    return mutual_information

def compute_mi_pairwise_vec(p_x1, p_x2, p_y, p_x1_y, p_x2_y, p_x1_x2):
    """
    Compute pairwise mutual information I(X1; Y), I(X2; Y), I(X1; X2) from joint distribution.

    Parameters
    ----------
    p_x1 : np.ndarray, shape (n_x1,), marginal distribution p(x1)
    p_x2 : np.ndarray, shape (n_x2,), marginal distribution p(x2)
    p_y : np.ndarray, shape (num_classes,), marginal distribution p(y)
    p_x1_y : np.ndarray, shape (n_x1, num_classes), joint distribution p(x1, y)
    p_x2_y : np.ndarray, shape (n_x2, num_classes), joint distribution p(x2, y)
    p_x1_x2 : np.ndarray, shape (n_x1, n_x2), joint distribution p(x1, x2)

    Returns
    -------
    mi_x1_y : float, mutual information I(X1; Y)
    mi_x2_y : float, mutual information I(X2; Y)
    mi_x1_x2 : float, mutual information I(X1; X2)
    """

    # Compute the pairwise mutual information using broadcasting
    # Shape of p_x1, p_x2, p_y, p_x1_y, p_x2_y, p_x1_x2 is known

    # Broadcasting for p(x1) * p(y), p(x2) * p(y), and p(x1) * p(x2)
    p_y_broadcasted = p_y[np.newaxis, :]    # Shape: (1, 1, num_classes)

    # Compute the masks where values are greater than 0 to avoid log(0)
    mask_x1_y = (p_x1_y > 0) & (p_x1[:, np.newaxis] > 0) & (p_y_broadcasted > 0)
    mask_x2_y = (p_x2_y > 0) & (p_x2[:, np.newaxis]  > 0) & (p_y_broadcasted > 0)
    mask_x1_x2 = (p_x1_x2 > 0) & (p_x1[:, np.newaxis] > 0) & (p_x2[np.newaxis, :] > 0)

    # I(X1; Y) = sum p(x1, y) * log(p(x1, y) / (p(x1) * p(y)))
    mi_x1_y = np.sum(p_x1_y[mask_x1_y] * np.log(p_x1_y[mask_x1_y] / (p_x1[:, np.newaxis] * p_y_broadcasted)[mask_x1_y]))

    # I(X2; Y) = sum p(x2, y) * log(p(x2, y) / (p(x2) * p(y)))
    mi_x2_y = np.sum(p_x2_y[mask_x2_y] * np.log(p_x2_y[mask_x2_y] / (p_x2[:, np.newaxis] * p_y_broadcasted)[mask_x2_y]))

    # I(X1; X2) = sum p(x1, x2) * log(p(x1, x2) / (p(x1) * p(x2)))
    mi_x1_x2 = np.sum(p_x1_x2[mask_x1_x2] * np.log(p_x1_x2[mask_x1_x2] / (p_x1[:, np.newaxis] * p_x2[np.newaxis, :])[mask_x1_x2]))

    return {
        "mi_x1_y": mi_x1_y, 
        "mi_x2_y": mi_x2_y, 
        "mi_x1_x2": mi_x1_x2
    }


#####
# SLOW VERSIONS OF MI COMPUTATION
#####

def compute_mi_x1_x2_y(p_x1, p_x2, p_y, p_x1_y, p_x2_y, p_x1_x2, p_x1_x2_y, eps=1e-5):
    # Extract the shape 
    n_x1, n_x2, num_classes = p_x1_x2_y.shape

    # Ensure probabilities are normalized
    assert (p_x1.sum() - 1.0) < eps
    assert (p_x2.sum() - 1.0) < eps
    assert (p_y.sum() - 1.0) < eps
    assert (p_x1_x2.sum() - 1.0) < eps
    assert (p_x1_y.sum() - 1.0) < eps
    assert (p_x2_y.sum() - 1.0) < eps
    assert (p_x1_x2_y.sum() - 1.0) < eps

    # Now calculate the mutual information
    mutual_information = 0.0
    for x1 in range(n_x1):
        for x2 in range(n_x2):
            for y in range(num_classes):
                # Extract values with more descriptive names
                p_x1_x2_y_val = p_x1_x2_y[x1, x2, y]
                p_x1_x2_val = p_x1_x2[x1, x2]
                p_x1_y_val = p_x1_y[x1, y]
                p_x2_y_val = p_x2_y[x2, y]
                p_x1_val = p_x1[x1]
                p_x2_val = p_x2[x2]
                p_y_val = p_y[y]
                
                # Ensure probabilities are not zero before computing log
                if p_x1_x2_y_val > 0 and p_x1_x2_val > 0 and p_x1_y_val > 0 and p_x2_y_val > 0:
                    mutual_information += p_x1_x2_y_val * np.log(
                        (p_x1_y_val * p_x1_x2_val * p_x2_y_val) / 
                        (p_x1_x2_y_val * p_x1_val * p_x2_val * p_y_val)
                    )
    return mutual_information

def compute_mi_x1x2_y(p_y, p_x1_x2, p_x1_x2_y, eps=1e-5):
    """
    Computes the mutual information I(X1, X2; Y) from a joint distribution
    p(xc, xu1, xu2, x1, x2, y).

    Parameters
    ----------
    p_y : np.ndarray
        A 1D numpy array of shape (num_classes,) representing p(y).
    p_x1_x2 : np.ndarray
        A 2D numpy array of shape (n_x1, n_x2) representing p(x1, x2).
    p_x1_x2_y : np.ndarray
        A 3D numpy array of shape (n_x1, n_x2, num_classes) representing p(x1, x2, y).
    eps : float
        A small epsilon for numerical stability and assertions.

    Returns
    -------
    float
        The mutual information I(X1, X2; Y).
    """

    # Extract the dimensions 
    n_x1, n_x2, num_classes = p_x1_x2_y.shape

    # Check normalization
    assert abs(p_x1_x2.sum() - 1.0) < eps, "p_x1_x2 must sum to 1."
    assert abs(p_y.sum() - 1.0) < eps, "p_y must sum to 1."
    assert abs(p_x1_x2_y.sum() - 1.0) < eps, "p_x1_x2_y must sum to 1."

    # Compute I(X1, X2; Y)
    mutual_information = 0.0
    for x1 in range(n_x1):
        for x2 in range(n_x2):
            for y in range(num_classes):
                p_joint = p_x1_x2_y[x1, x2, y]
                if p_joint > 0.0:
                    p_prod = p_x1_x2[x1, x2] * p_y[y]
                    if p_prod > 0.0:
                        mutual_information += p_joint * np.log(p_joint / p_prod)
    
    return mutual_information


def compute_mi_pairwise(p_x1, p_x2, p_y, p_x1_y, p_x2_y, p_x1_x2, eps=1e-5):
    """
    Compute pairwise mutual information I(X1; Y), I(X2; Y), I(X1; X2) from joint distribution.

    Parameters:
    - joint_dist: np.ndarray, shape (n_xc, n_xu1, n_xu2, n_x1, n_x2, num_classes), joint probabilities p(xc, xu1, xu2, x1, x2, y)
    - eps: float, small constant to avoid log(0)

    Returns:
    - mi_x1_y: float, mutual information I(X1; Y)
    - mi_x2_y: float, mutual information I(X2; Y)
    - mi_x1_x2: float, mutual information I(X1; X2)
    """

    # Compute I(X1; Y)
    mi_x1_y = 0.0
    for x1 in range(p_x1.shape[0]):
        for y in range(p_y.shape[0]):
            if p_x1_y[x1, y] > eps and p_x1[x1] > eps and p_y[y] > eps:
                mi_x1_y += p_x1_y[x1, y] * np.log(p_x1_y[x1, y] / (p_x1[x1] * p_y[y]))

    # Compute I(X2; Y)
    mi_x2_y = 0.0
    for x2 in range(p_x2.shape[0]):
        for y in range(p_y.shape[0]):
            if p_x2_y[x2, y] > eps and p_x2[x2] > eps and p_y[y] > eps:
                mi_x2_y += p_x2_y[x2, y] * np.log(p_x2_y[x2, y] / (p_x2[x2] * p_y[y]))

    # Compute I(X1; X2)
    mi_x1_x2 = 0.0
    for x1 in range(p_x1.shape[0]):
        for x2 in range(p_x2.shape[0]):
            if p_x1_x2[x1, x2] > eps and p_x1[x1] > eps and p_x2[x2] > eps:
                mi_x1_x2 += p_x1_x2[x1, x2] * np.log(p_x1_x2[x1, x2] / (p_x1[x1] * p_x2[x2]))

    return {
        "mi_x1_y": mi_x1_y, 
        "mi_x2_y": mi_x2_y, 
        "mi_x1_x2": mi_x1_x2
    }