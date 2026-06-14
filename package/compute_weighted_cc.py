import numpy as np

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
    merged = {}
    for refl, (I, sigma) in dataset.items():
        merged[refl] = merge_intensity_and_sigma(I, sigma)
    return merged

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
    x = np.asarray(x)
    y = np.asarray(y)
    sigma_x = np.asarray(sigma_x)
    sigma_y = np.asarray(sigma_y)

    # Eq (7):
    sigma2 = sigma_x**2 + sigma_y**2
    w = 1.0 / sigma2

    # Eq (6):
    w_sum = np.sum(w)
    x_bar = np.sum(w * x) / w_sum
    y_bar = np.sum(w * y) / w_sum

    # Eq (5):
    s_xy = np.sum(w * (x - x_bar) * (y - y_bar)) / w_sum
    s_xx = np.sum(w * (x - x_bar)**2) / w_sum
    s_yy = np.sum(w * (y - y_bar)**2) / w_sum

    cc = s_xy / np.sqrt(s_xx * s_yy)

    return cc

def compute_weighted_cc_between_datasets(dataset_x, dataset_y):
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
    merged_x = compute_merged_dataset_with_sigma(dataset_x)
    merged_y = compute_merged_dataset_with_sigma(dataset_y)
    # merged_x and merged_y are dictionaries
    # whose values are tuples
    # TODO: compute_merged_dataset_with_sigma should be called
    # outside this function, so it doesn't re-run for each
    # dataset on every cc calculation

    common_keys = set(merged_x.keys()) & set(merged_y.keys())

    x_vals = []
    y_vals = []
    sigma_x_vals = []
    sigma_y_vals = []

    for k in common_keys:
        x, sx = merged_x[k]
        y, sy = merged_y[k]

        x_vals.append(x)
        y_vals.append(y)
        sigma_x_vals.append(sx)
        sigma_y_vals.append(sy)
    
    cc = weighted_cc(x_vals, y_vals, sigma_x_vals, sigma_y_vals)

    n_common = len(common_keys)

    return cc, n_common
