import numpy as np
from datetime import datetime
import time
import dask.distributed
from dask.diagnostics import Profiler, ResourceProfiler, visualize
import os

from package.load_and_canonicalize import load_and_canonicalize_datasets
from package.build_cc_matrix import build_cc_matrix
from package.optimize import scan_dimensions
from package.find_elbow import find_elbow
from package.optics_clustering import run_optics_clustering, plot_reachability, plot_points
from utils.print_utils import print_cluster_summary, print_phi_table, print_separator

def pipeline_function():

    print_separator("Loading and merging reflections from all datasets")

    start_time_load = time.perf_counter()
    canonical_datasets = load_and_canonicalize_datasets(
        "datasets",
        sg_symbol="P21"
    )
    end_time_load = time.perf_counter()
    elapsed_time_load = end_time_load - start_time_load
    print(f"\nTotal time taken for loading datasets: {elapsed_time_load:.2f} seconds")

    print(f"Datasets loaded: {len(canonical_datasets)}")

    print_separator("Building correlation matrix")

    start_time_cc = time.perf_counter()
    names, R, W = build_cc_matrix(canonical_datasets)
    end_time_cc = time.perf_counter()
    elapsed_time_load = end_time_cc - start_time_cc
    print(f"\nTotal time taken for building cc matrix: {elapsed_time_load:.2f} seconds")

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
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cpus = os.cpu_count()
    start_time = time.perf_counter()
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof:
        pipeline_function()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
    print(f"CPUs utilized: {available_cpus}\n")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"dask_profile_{timestamp}.html"
    visualize([prof, rprof], filename=report_filename)

if __name__ == '__main__':
    main()
