import numpy as np
from package.compute_cc import compute_cc_between_datasets
from package.compute_weighted_cc import compute_weighted_cc_between_datasets

def build_cc_matrix(datasets, weighted=True, use_counts_as_weights = False):

    names = list(datasets.keys())

    n = len(names)

    total = (n * (n - 1)) // 2

    counter = 0

    R = np.eye(n)

    N = np.zeros((n, n))

    for i in range(n):

        for j in range(i + 1, n):

            counter += 1

            print(f"{counter:03d}/{total}: {names[i]} vs {names[j]} ")

            if weighted:

                cc, n_common = compute_weighted_cc_between_datasets(
                    datasets[names[i]],
                    datasets[names[j]]
                )

            else:

                cc, n_common = compute_cc_between_datasets(
                    datasets[names[i]],
                    datasets[names[j]]
                )

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

        scale = (np.count_nonzero(W) / np.sum(W))

        W *= scale

    print("\nDone.\n")

    return names, R, W