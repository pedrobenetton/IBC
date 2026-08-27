
from sklearn.cluster import OPTICS
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
matplotlib.use('Agg') 

def compute_smin(n, d, b):
    """
    Compute s_min parameter (Eq. 9).

    Parameters
    ----------
    n : int
        Number of datasets
    d : int
        Embedding dimension
    b : float, optional
        Buffer parameter (default = 0.5 from arg parsing)

    Returns
    -------
    int
        Minimum samples for OPTICS
    """
    return max(5, int(b * (n / d)))

def run_optics_clustering(X, buffer_parameter, xi=0.05):
    """
    Perform OPTICS clustering on embedded coordinates.

    Parameters
    ----------
    X : ndarray (n, d)
        Cosym embedding coordinates
    b : float, optional
        Buffer parameter for s_min (default = 0.5 from arg parsing)
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

    smin = compute_smin(n, d, b=buffer_parameter)

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

def plot_multidimensional_grid(X, labels, project_name):
    n_samples, d = X.shape
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename=f'{project_name}/multidim_grid_{d}_{timestamp}.png'
    fig, axes = plt.subplots(d, d, figsize=(3*d, 3*d))
    unique_labels=np.unique(labels)
    cmap = plt.get_cmap('tab10')
    cluster_only_labels = [l for l in unique_labels if l != -1]
    for row in range(d):
        for col in range(d):
            ax = axes[row, col]

            if row == col:
                ax.text(0.5, 0.5, f"Dim {row+1}", horizontalalignment='center', verticalalignment='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            for label_id in unique_labels:
                mask = (labels == label_id)
                points = X[mask]
                color = 'lightgray' if label_id == -1 else cmap(cluster_only_labels.index(label_id) % cmap.N)
                alpha = 0.4 if label_id == -1 else 0.8
                ax.scatter(points[:,col], points[:,row], color=color, alpha=alpha, s=20, edgecolors='none')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.tick_params(labelsize=8)
            
    plt.suptitle(f"Pairwise Matrix Dimension View (d={d} Configuration)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200)
    plt.close('all')
