import os
import time

import numpy as np

import fast_mtz

from package.find_elbow import find_elbow
from package.optics_clustering import (
    plot_multidimensional_grid,
    run_optics_clustering,
    plot_multidimensional_grid_annotated,
)
from package.optimize import scan_dimensions
from utils.print_utils import (
    print_cluster_summary,
    print_phi_table,
    print_separator,
)
from utils.parse_args import parse_args
from utils.create_run_folder import create_run_folder

from collections import Counter, defaultdict

def pipeline_function(
    args,
    input_folder,
    project_name,
    sg_number,
    d_max,
    forced_dimension,
    use_cache,
    force_recalculate,
    minimize_method,
    use_spectral_init,
    buffer_parameter,
    num_threads=8,
):
    print_separator("Building correlation matrix via C++")

    project_folder = create_run_folder(project_name)

    with open(f"{project_folder}/arguments.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    cache_file = f"{project_folder}/cc_matrix.npz"

    if use_cache and not force_recalculate and os.path.exists(cache_file):
        print_separator(f"Loading cached correlation matrix from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        names = data["names"]
        R = data["R"]
        W = data["W"]
    else:
        start_time_cc = time.perf_counter()

        matrix_result = fast_mtz.compute_cc_matrix(
            dir_path=input_folder,
            sg_number=sg_number,
            use_counts_as_weights=False,
            num_threads=num_threads,
        )

        names = matrix_result.filenames
        R = matrix_result.R
        W = matrix_result.W

        print(f"Datasets loaded and processed: {len(names)}")

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

    X_vals, phi_vals = scan_dimensions(
        R, minimize_method, use_spectral_init, d_max=d_max, W=W
    )

    print_phi_table(phi_vals)

    if forced_dimension == 0:
        optimal_d = find_elbow(phi_vals)
    else:
        print(f"\nUsing dimension {forced_dimension} (forced by input).")
        optimal_d = forced_dimension

    print(f"\nOptimal dimension: {optimal_d}")

    X = X_vals[optimal_d - 1]
    final_phi = phi_vals[optimal_d - 1]

    print_separator("Final embedding")

    print(f"Embedding shape: {X.shape}")
    print(f"Final Phi: {final_phi:.6f}")

    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    print_separator("Running OPTICS clustering")

    labels, model = run_optics_clustering(X, buffer_parameter)

    print_cluster_summary(labels)

    print("\nCluster labels:")

    sorted_pairs = sorted(zip(names, labels), key=lambda x: x[0])

    sorted_pairs = sorted(zip(names, labels), key=lambda x: x[0])
    sorted_names = [pair[0] for pair in sorted_pairs]

    sample_map = {}
    metadata_file = "runs/hTTR_P21212/hTTR_P21212.txt"
    with open(metadata_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                sample_map[parts[0].strip().replace('hTTR_P21212-x', '')] = parts[1].strip()

    cluster_groups = defaultdict(list)
    type_to_clusters = defaultdict(Counter)

    print("\n" + "="*80)
    print(f"{'Filename':<25} | {'Ground Truth State':<20} | {'Assigned Cluster'}")
    print("="*80)

    for name, label in sorted_pairs:
        clean_name = name.replace('.mtz', '')
        sample_type = sample_map.get(clean_name, "UNKNOWN")

        cluster_groups[label].append((name, sample_type))
        type_to_clusters[sample_type][label] += 1

        cluster_str = f"Cluster {label}" if label != -1 else "Noise / Outlier (-1)"
        print(f"{name:<25} | {sample_type:<20} | {cluster_str}")

    print("\n" + "="*80)
    print("CLUSTER COMPOSITION MATRIX")
    print("="*80)
    all_clusters = sorted(list(cluster_groups.keys()))
    header = f"{'Sample Type':<20} | " + " | ".join([f"{f'Cluster {c}' if c != -1 else 'Outliers (-1)':<12}" for c in all_clusters])
    print(header)
    print("-" * len(header))

    for stype, counts in type_to_clusters.items():
        row = f"{stype:<20} | " + " | ".join([f"{counts[c]:<12}" for c in all_clusters])
        print(row)
    print("="*80)

    plot_multidimensional_grid_annotated(X, labels, sorted_names, metadata_file, project_folder)

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
    forced_dimension = args.force_dim
    buffer_parameter = args.buffer_parameter

    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cpus = os.cpu_count() or 4

    os.makedirs(project_name, exist_ok=True)

    start_time = time.perf_counter()

    pipeline_function(
        args,
        input_folder,
        project_name,
        sg_number,
        d_max,
        forced_dimension,
        use_cache,
        force_recalculate,
        minimize_method,
        use_spectral_init,
        buffer_parameter,
        num_threads=available_cpus,
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"\nTotal pipeline time taken: {elapsed_time:.2f} seconds")
    print(f"CPUs utilized: {available_cpus}\n")

if __name__ == "__main__":
    main()