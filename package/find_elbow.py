import numpy as np

def find_elbow(phi_values):
    """
    Simple elbow detection using second derivative.

    Parameters
    ----------
    phi_values : list

    Returns
    -------
    int
        Optimal dimension (1-based index)
    """
    phi_values = np.array(phi_values)

    second_diff = np.diff(phi_values, n=2)

    elbow_idx = np.argmax(-second_diff) + 2  # shift index

    return elbow_idx