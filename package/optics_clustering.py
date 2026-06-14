
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt

def compute_smin(n, d, b=0.5):
    """
    Compute s_min parameter (Eq. 9).

    Parameters
    ----------
    n : int
        Number of datasets
    d : int
        Embedding dimension
    b : float, optional
        Buffer parameter (default = 0.5)

    Returns
    -------
    int
        Minimum samples for OPTICS
    """
    return max(5, int(b * (n / d)))

def run_optics_clustering(X, b=0.5, xi=0.05):
    """
    Perform OPTICS clustering on embedded coordinates.

    Parameters
    ----------
    X : ndarray (n, d)
        Cosym embedding coordinates
    b : float, optional
        Buffer parameter for s_min (default = 0.5)
    xi : float, optional
        Steepness parameter for cluster detection (default = 0.05)

    Returns
    -------
    labels : ndarray (n,)
        Cluster labels (-1 = outlier)
    model : OPTICS object
        Fitted OPTICS model (contains reachability, ordering, etc.)
    """
    n, d = X.shape

    smin = compute_smin(n, d, b=b)

    model = OPTICS(
        min_samples=smin,
        xi=xi,
        metric='euclidean'
    )

    model.fit(X)

    return model.labels_, model

def plot_reachability(model):
    """
    Plot OPTICS reachability diagram.

    Parameters
    ----------
    model : OPTICS object
        Fitted model
    """
    reachability = model.reachability_[model.ordering_]
    labels = model.labels_[model.ordering_]

    plt.figure(figsize=(10, 4))
    plt.plot(reachability, marker='.', linestyle='none')

    plt.xlabel("Ordered points")
    plt.ylabel("Reachability distance")
    plt.title("OPTICS Reachability Plot")

    plt.show()