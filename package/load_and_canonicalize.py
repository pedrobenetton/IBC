import gemmi
import glob
from collections import defaultdict
import dask

@dask.delayed
def read_hkl_file(path):
    print(f"Reading HKL file: {path}")
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
def group_symmetry_equivalents(dataset, sg_symbol="C121"):
    print(f"Grouping symmetry equivalents for dataset")
    sg = gemmi.SpaceGroup(sg_symbol)
    merged = defaultdict(lambda: ([], []))

    for hkl, (I_vals, sigma_vals) in dataset.items():
        canonical = canonicalize_hkl(hkl, sg)
        merged[canonical][0].extend(I_vals)
        merged[canonical][1].extend(sigma_vals)

    return merged

def load_and_canonicalize_single_dataset(path, sg_symbol):
    lazy_dataset = read_hkl_file(path)
    lazy_merged = group_symmetry_equivalents(lazy_dataset, sg_symbol=sg_symbol)
    return lazy_merged, path


def load_and_canonicalize_datasets(folder, sg_symbol="C121"):
    lazy_results = []
    path_list = []

    for path in glob.glob(f"{folder}/*.HKL"):
        lazy_dataset, file_path = load_and_canonicalize_single_dataset(path, sg_symbol)
        path_list.append(file_path)
        lazy_results.append((lazy_dataset))

    print(f"Triggering parallel Dask execution graph...")

    computed_datasets = dask.delayed(lazy_results).compute()

    datasets = {path_list[i]: computed_datasets[i] for i in range(len(path_list))}
    return datasets