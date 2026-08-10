from datetime import datetime
import os
import time

import dask.config
from dask.diagnostics import Profiler, ResourceProfiler, visualize
import numpy as np

from package.build_cc_matrix import build_cc_matrix
from package.find_elbow import find_elbow
from package.load_and_canonicalize import load_and_canonicalize_datasets
from package.optics_clustering import (
    plot_multidimensional_grid,
    run_optics_clustering,
)
from package.optimize import scan_dimensions
from utils.print_utils import (
    print_cluster_summary,
    print_phi_table,
    print_separator,
)
from utils.parse_args import parse_args

def pipeline_function(input_folder, project_name, sg_number, d_max, use_cache, force_recalculate, minimize_method, use_spectral_init):

    dask.config.set(scheduler="threads")

    print_separator("Loading and merging reflections from all datasets")

    cache_file = f"{project_name}/cc_matrix.npz"

    canonical_datasets = load_and_canonicalize_datasets(
        input_folder, sg_number
    )

    print(f"Datasets loaded: {len(canonical_datasets)}")

    print_separator("Building correlation matrix")

    if use_cache and not force_recalculate and os.path.exists(cache_file):
        print_separator(f"Loading cached correlation matrix from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        names = data["names"]
        R = data["R"]
        W = data["W"]
    else:
        start_time_cc = time.perf_counter()
        names, R, W = build_cc_matrix(canonical_datasets)

        if use_cache:
            np.savez_compressed(cache_file, names=names, R=R, W=W)
            print(f"Saved correlation matrix cache to {cache_file}")

        end_time_cc = time.perf_counter()
        elapsed_time_load = end_time_cc - start_time_cc
        print(
            f"\nTotal time taken for building cc matrix: {elapsed_time_load:.2f} seconds"
        )

    print("\nCorrelation matrix:")
    print(np.array_str(R, precision=3, suppress_small=True))

    print_separator("Scanning embedding dimensions")

    X_vals, phi_vals = scan_dimensions(R, minimize_method, use_spectral_init, d_max=d_max, W=W)

    print_phi_table(phi_vals)

    optimal_d = find_elbow(phi_vals)

    print(f"\nOptimal dimension: {optimal_d}")

    X = X_vals[optimal_d - 1]

    final_phi = phi_vals[optimal_d - 1]

    print_separator("Final embedding")

    print(f"Embedding shape: {X.shape}")
    print(f"Final Phi: {final_phi:.6f}")

    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    print_separator("Running OPTICS clustering")

    labels, model = run_optics_clustering(X)

    print_cluster_summary(labels)

    print("\nCluster labels:")

    for name, label in zip(names, labels):
        print(f"{name:30s} -> {label}")

    plot_multidimensional_grid(X, labels, project_name)

def main():
    args = parse_args()

    use_cache = not args.no_cache
    force_recalculate = args.force_recalculate
    project_name = args.project_name
    sg_number = args.sg_number
    d_max = args.d_max
    minimize_method = args.scan_method
    use_spectral_init = args.use_spectral_init
    input_folder = args.input_folder

    os.makedirs(project_name, exist_ok=True)

    if args.benchmark:
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count()

        start_time = time.perf_counter()
        with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof:
            pipeline_function(
                input_folder,
                project_name,
                sg_number,
                d_max,
                use_cache,
                force_recalculate,
                minimize_method,
                use_spectral_init
            )
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print(f"\nTotal time taken: {elapsed_time:.2f} seconds")
        print(f"CPUs utilized: {available_cpus}\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_filename = f"{project_name}/dask_profile_{timestamp}.html"
        visualize([prof, rprof], filename=report_filename)
    else:
        pipeline_function(
            input_folder,
            project_name,
            sg_number,
            d_max,
            use_cache,
            force_recalculate,
            minimize_method,
            use_spectral_init
        )

if __name__ == "__main__":
    main()