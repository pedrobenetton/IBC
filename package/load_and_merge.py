import gemmi
import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

prefetch_pool = ThreadPoolExecutor(
    max_workers=8
)

def read_hkl_file(path):
    """
    - Reads HKL file into reflection dictionary

    Parameters
    ----------
    path : str
        Path to the HKL file

    Returns
    -------
    dict

        keys:
            (h, k, l)

        values:
            ([I values], [sigma values])
    """

    dataset = defaultdict(lambda: ([], []))

    with open(path) as f:

        for line in f:

            if not line.strip():
                continue

            if line.startswith("!"):
                continue

            tokens = line.split()

            if len(tokens) < 5:
                continue

            h, k, l = map(int, tokens[:3])

            I = float(tokens[3])
            sigma = float(tokens[4])

            dataset[(h, k, l)][0].append(I)
            dataset[(h, k, l)][1].append(sigma)

    return dataset

def canonicalize_hkl(hkl, spacegroup):
    """
    - Converts reflection to canonical symmetry representative

    Parameters
    ----------
    hkl : (h, k, l)
        Path to the HKL file
    spacegroup: gemmi.Spacegroup

    Returns
    -------
    str
        The canonical key for that specific hkl file
    """

    ops = spacegroup.operations()

    equivalents = []

    for op in ops:

        transformed = op.apply_to_hkl(hkl)

        equivalents.append(tuple(transformed))

    return min(equivalents)

def merge_symmetry_equivalents(dataset, sg_symbol="P212121"):
    """
    - Converts reflection to canonical symmetry representative

    Parameters
    ----------
    dataset : dict
        dataset dictionary with hkl as keys and
        I and sigma as values
    sg_symbol: str
        Spacegroup symbol

    Returns
    -------
    dict
        Dictionary of merged intensities
    """

    sg = gemmi.SpaceGroup(sg_symbol)

    merged = defaultdict(lambda: ([], []))

    for hkl, (I_vals, sigma_vals) in dataset.items():

        canonical = canonicalize_hkl(hkl, sg)

        merged[canonical][0].extend(I_vals)
        merged[canonical][1].extend(sigma_vals)

    return merged

def merge_single_dataset(path, sg_symbol):
    print(f"Loading and merging dataset: {path}")
    dataset = read_hkl_file(path)

    dataset = merge_symmetry_equivalents(
        dataset,
        sg_symbol=sg_symbol
    )

    return dataset, path



def load_and_merge_datasets(folder, sg_symbol="P212121"):
    """
    - Converts reflection to canonical symmetry representative

    Parameters
    ----------
    folder : str
        Path to the folder with the HKL files
    sg_symbol: str
        String representing the spacegroup

    Returns
    -------
    dict

        keys:
            path: str
                Path to the HKL file

        values:
            dataset: dict
                Dictionary of merged intensities
    """

    datasets = {}
    futures = []

    for path in glob.glob(f"{folder}/*.HKL")[0:6]:
        futures.append(
            prefetch_pool.submit(
                merge_single_dataset,
                path,
                sg_symbol,
            )
        )

    for future in futures:
        dataset, path = future.result()
        datasets[path] = dataset

    return datasets