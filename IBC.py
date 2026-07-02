import numpy as np
from datetime import datetime
import dask.distributed
from dask.diagnostics import Profiler, ResourceProfiler, visualize

from package.load_and_canonicalize import load_and_canonicalize_datasets
from package.build_cc_matrix import build_cc_matrix
from package.optimize import scan_dimensions
from package.find_elbow import find_elbow
from package.optics_clustering import run_optics_clustering, plot_reachability, plot_points
from utils.print_utils import print_cluster_summary, print_phi_table, print_separator

def pipeline_function():

    client = dask.distributed.Client(threads_per_worker=4)

    print_separator("Loading and merging reflections from all datasets")

    canonical_datasets = load_and_canonicalize_datasets(
        "datasets",
        sg_symbol="P21"
    )

    print(f"Datasets loaded: {len(canonical_datasets)}")

    print_separator("Building correlation matrix")

    names, R, W = build_cc_matrix(canonical_datasets)

    print("\nCorrelation matrix:")
    print(np.array_str(R, precision=3, suppress_small=True))

    print_separator("Scanning embedding dimensions")

    X_vals, phi_vals = scan_dimensions(R, d_max=5, W=W)

    print_phi_table(phi_vals)

    optimal_d = find_elbow(phi_vals)

    print(f"\nOptimal dimension: {optimal_d}")

    X = X_vals[optimal_d - 1]

    final_phi = phi_vals[optimal_d - 1]

    print_separator("Final embedding")

    print(f"Embedding shape: {X.shape}")
    print(f"Final Phi: {final_phi:.6f}")

    X = X / np.linalg.norm(
        X,
        axis=1,
        keepdims=True
    )

    print_separator("Running OPTICS clustering")

    labels, model = run_optics_clustering(X)

    print_cluster_summary(labels)

    print("\nCluster labels:")

    for name, label in zip(names, labels):
        print(f"{name:30s} -> {label}")

    plot_reachability(model)

def main():
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof:
        pipeline_function()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"dask_profile_{timestamp}.html"
    visualize([prof, rprof], filename=report_filename)

if __name__ == '__main__':
    main()
