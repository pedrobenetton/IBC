import gemmi
import glob
from collections import defaultdict
import dask

@dask.delayed
def read_hkl_file(path):
    dataset = defaultdict(lambda: ([], []))
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("!"):
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
    ops = spacegroup.operations()
    equivalents = [tuple(op.apply_to_hkl(hkl)) for op in ops]
    return min(equivalents)

@dask.delayed
def group_symmetry_equivalents(dataset, sg_number):
    sg = gemmi.find_spacegroup_by_number(sg_number)
    merged = defaultdict(lambda: ([], []))

    for hkl, (I_vals, sigma_vals) in dataset.items():
        canonical = canonicalize_hkl(hkl, sg)
        merged[canonical][0].extend(I_vals)
        merged[canonical][1].extend(sigma_vals)

    return merged

def load_and_canonicalize_single_dataset(path, sg_number):
    lazy_dataset = read_hkl_file(path)
    lazy_merged = group_symmetry_equivalents(lazy_dataset, sg_number)
    return lazy_merged

def load_and_canonicalize_datasets(folder, sg_number):
    lazy_datasets = {}

    datasets_list = sorted(glob.glob(f"{folder}/*.HKL"))

    for path in datasets_list:
        lazy_dataset = load_and_canonicalize_single_dataset(path, sg_number)
        lazy_datasets[path] = lazy_dataset

    print(f"Created Dask task graph recipes for {len(lazy_datasets)} datasets.")
    return lazy_datasets