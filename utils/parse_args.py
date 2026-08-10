import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="X-ray crystallography IBC pipeline."
    )

    parser.add_argument(
        "-b",
        "--benchmark",
        action="store_true",
        help="Enable Dask resource profiling and benchmarking.",
    )

    parser.add_argument(
        "-i",
        "--input-folder",
        type=str,
        default="datasets",
        help="Path to the input datasets folder (default: 'datasets').",
    )

    parser.add_argument(
        "-p",
        "--project-name",
        type=str,
        default="project",
        help="Project name where cached files are stored",
    )

    parser.add_argument(
        "-s",
        "--sg-number",
        type=int,
        default=1,
        help="Space Group Number of the datasets",
    )

    parser.add_argument(
        "-d",
        "--d-max",
        type=int,
        default=7,
        help="Max number of dimensions used for scanning",
    )

    parser.add_argument(
        "-m",
        "--scan-method",
        choices=["gradient_descent", "simulated_annealing", "lbfgs"],
        default="gradient_descent",
        help="Optimization method to use for scanning dimensions. "
        "Options: gradient_descent, simulated_annealing, lbfgs (default: gradient_descent)",
    )

    parser.add_argument(
        "--use-spectral-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable spectral initialization.",
    )

    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading or saving the correlation matrix cache entirely.",
    )
    cache_group.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Ignore existing .npz cache file and force recalculation, but save the result.",
    )

    return parser.parse_args()