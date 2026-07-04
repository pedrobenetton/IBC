import numpy as np
import dask
from package.compute_weighted_cc import compute_merged_dataset_with_sigma, compute_pairwise_cc

def build_cc_matrix(canonical_datasets, weighted=True, use_counts_as_weights=False):

    names = list(canonical_datasets.keys())
    n = len(names)

    print("Building pre-merging task graph...")
    lazy_merge_func = dask.delayed(compute_merged_dataset_with_sigma)

    pre_merged_datasets = {}
    for name in names:
        pre_merged_datasets[name] = lazy_merge_func(canonical_datasets[name])

    print("Building pairwise correlation task graph...")
    lazy_cc_func = dask.delayed(compute_pairwise_cc)
    lazy_pairs = []
    pair_indices = []

    for i in range(n):
        for j in range(i + 1, n):
            lazy_task = lazy_cc_func(
                pre_merged_datasets[names[i]], 
                pre_merged_datasets[names[j]]
            )
            lazy_pairs.append(lazy_task)
            pair_indices.append((i, j))

    print(f"Triggering computation for {len(lazy_pairs)} unique pairs...")
    computed_pairs = dask.compute(*lazy_pairs)

    R = np.eye(n)
    N = np.zeros((n, n))

    for (i, j), (cc, n_common) in zip(pair_indices, computed_pairs):
        if np.isnan(cc):
            cc = 0.0
        R[i, j] = cc
        R[j, i] = cc
        N[i, j] = n_common
        N[j, i] = n_common

    if use_counts_as_weights:
        W = N
    else:
        W = np.zeros_like(R)
        mask = N > 1
        W[mask] = (np.sqrt(N[mask] - 1) / (1 - R[mask]**2))
        scale = (np.count_nonzero(W) / np.sum(W)) if np.sum(W) > 0 else 1.0
        W *= scale

    print("\nMatrix generation complete.\n")
    return names, R, W