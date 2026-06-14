import numpy as np

def merge_intensities(I, sigma):
    """
    Equation (3): inverse-variance weighted mean.
    Calculates the merged intensity for a single
    reflection with one or more registered intensities.

    Parameters
    ----------
    I : array-like
        Intensities for a given reflection
    sigma : array-like
        Corresponding uncertainties for a given reflection

    Returns
    -------
    float
        Merged intensity
    """
    I = np.asarray(I)
    sigma = np.asarray(sigma)
    weights = 1.0 / (sigma ** 2)
    return np.sum(weights * I) / np.sum(weights)

def compute_merged_dataset(dataset):
    """
    Applies merging to all reflections in a single dataset.

    Parameters
    ----------
    dataset : dict
        Keys = reflection ids (e.g. hkl)
        Values = I_array

    Returns
    -------
    dict
        Keys = reflection ids
        Values = merged intensities
    """
    merged = {}
    for refl, (I, sigma) in dataset.items():
        merged[refl] = merge_intensities(I, sigma)
    return merged

def pearson_cc(x, y):
    """
    Equation (2): Pearson correlation coefficient
    Calculates the Pearson cc between two arrays
    Note that it must 


    Parameters
    ----------
    x, y : array-like
        Merged intensities for common reflections

    Returns
    -------
    float
        Correlation coefficient
    """
    x = np.asarray(x)
    y = np.asarray(y)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(
        np.sum((x - x_mean) ** 2) *
        np.sum((y - y_mean) ** 2)
    )

    return numerator / denominator

def compute_cc_between_datasets(dataset_x, dataset_y):
    """
    - Merge intensities
    - Select common reflections
    - Compute Pearson CC

    Parameters
    ----------
    dataset_x, dataset_y : dict
        Keys = reflection ids (e.g. hkl)
        Values = (I_array, sigma_array)

    Returns
    -------
    float
        Correlation coefficient
    """
    merged_x = compute_merged_dataset(dataset_x)
    merged_y = compute_merged_dataset(dataset_y)

    common_keys = set(merged_x.keys()) & set(merged_y.keys())

    x_vals = [merged_x[k] for k in common_keys]
    y_vals = [merged_y[k] for k in common_keys]

    cc = pearson_cc(x_vals, y_vals)

    n_common = len(common_keys)

    return cc, n_common
