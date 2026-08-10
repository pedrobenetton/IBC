import numpy as np
import pickle

def merge_intensity_and_sigma(I, sigma):
    """
    Compute merged intensity and its standard error.

    Implements:
    - Equation (3): inverse-variance weighted mean
    - Equation (4): standard error of the weighted mean

    Parameters
    ----------
    I : array-like
        Intensities for a given reflection
    sigma : array-like
        Corresponding uncertainties

    Returns
    -------
    tuple
        (merged_intensity, merged_sigma)
    """
    I = np.asarray(I)
    sigma = np.asarray(sigma)

    weights = 1.0 / (sigma ** 2)

    merged_intensity = np.sum(weights * I) / np.sum(weights)
    merged_sigma = np.sqrt(1.0 / np.sum(weights))

    return merged_intensity, merged_sigma

def compute_merged_dataset_with_sigma(dataset):
    """
    Merge all reflections in a dataset.

    Parameters
    ----------
    dataset : dict
        Keys = reflection ids (e.g. hkl)
        Values = (I_array, sigma_array)

    Returns
    -------
    dict
        Keys = reflection ids
        Values = (merged_intensity, merged_sigma)
    """
    hkl_list = []
    I_list = []
    sigma_list = []

    for (h, k, l), (I_vals, sigma_vals) in dataset.items():
        hkl_encoded = ((int(h) + 512) << 40) | ((int(k) + 512) << 20) | (int(l) + 512)
        for I_val, sig_val in zip(I_vals, sigma_vals):
            hkl_list.append(hkl_encoded)
            I_list.append(I_val)
            sigma_list.append(sig_val)
    hkl = np.array(hkl_list, dtype=np.int64)
    I = np.array(I_list, dtype=np.float32)
    sigma = np.array(sigma_list, dtype=np.float32)
    sort_idx = np.argsort(hkl)
    hkl = hkl[sort_idx]
    I = I[sort_idx]
    sigma = sigma[sort_idx]
    unique_hkl, index, counts = np.unique(hkl, return_index=True, return_counts=True)

    if len(unique_hkl) == len(hkl):
        return {
            "hkl": unique_hkl,
            "I": I,
            "sigma": sigma
        }

    weights = 1.0 / (sigma ** 2)
    unq_inv = np.searchsorted(unique_hkl, hkl)

    sum_w = np.bincount(unq_inv, weights=weights)
    sum_wI = np.bincount(unq_inv, weights=weights * I)

    merged_I = sum_wI / sum_w
    merged_sigma = np.sqrt(1.0 / sum_w)

    return {
        "hkl": unique_hkl,
        "I": merged_I,
        "sigma": merged_sigma
    }

def weighted_cc(x, y, sigma_x, sigma_y):
    """
    Compute weighted Pearson correlation coefficient.

    Implements:
    - Equation (7): weights definition
    - Equation (6): weighted means
    - Equation (5): weighted covariance and variances

    Parameters
    ----------
    x, y : array-like
        Merged intensities for common reflections
    sigma_x, sigma_y : array-like
        Standard errors of merged intensities

    Returns
    -------
    float
        Weighted correlation coefficient
    """
    sigma2 = sigma_x**2 + sigma_y**2
    w = 1.0 / sigma2

    w_sum = np.sum(w)
    if w_sum == 0:
        return 0.0

    x_bar = np.sum(w * x) / w_sum
    y_bar = np.sum(w * y) / w_sum

    s_xy = np.sum(w * (x - x_bar) * (y - y_bar)) / w_sum
    s_xx = np.sum(w * (x - x_bar)**2) / w_sum
    s_yy = np.sum(w * (y - y_bar)**2) / w_sum

    denominator = np.sqrt(s_xx * s_yy)
    if denominator == 0:
        return 0.0

    return s_xy / denominator

def compute_pairwise_cc(dataset_x, dataset_y):
    """
    - Merge intensities and compute uncertainties
    - Select common reflections
    - Compute weighted Pearson CC

    Parameters
    ----------
    dataset_x, dataset_y : dict
        Keys = reflection ids (e.g. hkl)
        Values = (I_array, sigma_array)

    Returns
    -------
    float
        Weighted correlation coefficient
    """
    hkl_x, I_x, sig_x = dataset_x["hkl"], dataset_x["I"], dataset_x["sigma"]
    hkl_y, I_y, sig_y = dataset_y["hkl"], dataset_y["I"], dataset_y["sigma"]

    _, idx_x, idx_y = np.intersect1d(hkl_x, hkl_y, assume_unique=True, return_indices=True)

    n_common = len(idx_x)
    if n_common == 0:
        return 0.0, 0

    x = I_x[idx_x]
    y = I_y[idx_y]
    sigma_x = sig_x[idx_x]
    sigma_y = sig_y[idx_y]

    cc = weighted_cc(x, y, sigma_x, sigma_y)

    return cc, n_common