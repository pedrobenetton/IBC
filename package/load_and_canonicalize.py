# package/load_and_canonicalize.py
import numpy as np
import fast_mtz

def load_and_canonicalize_datasets(input_folder, sg_number, num_threads=8):
    """
    Parses MTZ files in parallel via C++ fast_mtz bindings
    and maps them to arrays expected by compute_weighted_cc functions.
    """
    print(f"Reading MTZ files from '{input_folder}' using C++ backend ({num_threads} threads)...")

    raw_summaries = fast_mtz.read_batch(input_folder, num_threads=num_threads)
    canonical_datasets = {}

    for item in raw_summaries:
        if not item.success:
            print(f"Skipping corrupt file {item.filename}: {item.error_msg}")
            continue

        canonical_datasets[item.filename] = {
            "spacegroup": item.spacegroup,
            "hkl_encoded": np.array(item.hkl_encoded, dtype=np.int64),
            "I": np.array(item.i_mean, dtype=np.float32),
            "sigma": np.array(item.sig_i_mean, dtype=np.float32),
        }

    return canonical_datasets