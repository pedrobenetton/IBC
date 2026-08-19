import glob
import dask
import gemmi
import numpy as np

def canonicalize_and_group(h_arr, k_arr, l_arr, I_arr, sig_arr, sg_number):
    """
    Directly maps Miller indices to the asymmetric unit (ASU) 
    using Gemmi's fast native C++ ReciprocalAsu class.
    """
    sg = gemmi.find_spacegroup_by_number(sg_number)
    asu = gemmi.ReciprocalAsu(sg)
    ops = sg.operations()

    # Combine h, k, l into (N, 3) int32 matrix
    hkl_matrix = np.column_stack((h_arr, k_arr, l_arr)).astype(np.int32)

    # Use Gemmi's direct C++ mapping routine
    # asu.to_asu maps each (h,k,l) to its canonical ASU representative
    canonical_hkl = np.array([asu.to_asu(hkl, ops)[0] for hkl in hkl_matrix], dtype=np.int32)

    return {
        "h": canonical_hkl[:, 0],
        "k": canonical_hkl[:, 1],
        "l": canonical_hkl[:, 2],
        "I": I_arr.astype(np.float32),
        "sigma": sig_arr.astype(np.float32),
    }

def find_intensity_and_sigma_columns(col_labels):
    """
    Finds the best available Intensity and Sigma columns in an MTZ file.
    """
    pairs_to_check = [
        ("IMEAN", "SIGIMEAN"),
        ("I", "SIGI"),
        ("I-obs", "SIGI-obs"),
        ("IOBS", "SIGIOBS"),
    ]
    for i_label, sig_label in pairs_to_check:
        if i_label in col_labels and sig_label in col_labels:
            return col_labels.index(i_label), col_labels.index(sig_label), False

    f_pairs = [
        ("FP", "SIGFP"),
        ("FOBS", "SIGFOBS"),
        ("F", "SIGF"),
    ]
    for f_label, sigf_label in f_pairs:
        if f_label in col_labels and sigf_label in col_labels:
            return col_labels.index(f_label), col_labels.index(sigf_label), True

    i_idx = next((i for i, label in enumerate(col_labels) if label.startswith("I")), None)
    sig_idx = next((i for i, label in enumerate(col_labels) if "SIG" in label), None)

    if i_idx is not None and sig_idx is not None:
        return i_idx, sig_idx, False

    raise ValueError(f"Could not find valid Intensity/Sigma or Amplitude columns in labels: {col_labels}")

@dask.delayed
def process_mtz_file(path, sg_number):
    """
    Reads MTZ files zero-copy via Gemmi NumPy buffers to prevent CPU locks.
    """
    mtz = gemmi.read_mtz_file(path)

    all_data = np.array(mtz, copy=False)
    if all_data.size == 0:
        return {"hkl_encoded": np.array([]), "I": np.array([]), "sigma": np.array([])}

    h_arr = all_data[:, 0].astype(np.int32)
    k_arr = all_data[:, 1].astype(np.int32)
    l_arr = all_data[:, 2].astype(np.int32)

    col_labels = mtz.column_labels()
    val_idx, sig_idx, is_amplitude = find_intensity_and_sigma_columns(col_labels)

    raw_val = all_data[:, val_idx].astype(np.float32)
    raw_sig = all_data[:, sig_idx].astype(np.float32)

    if is_amplitude:
        I_arr = raw_val ** 2
        sig_arr = 2.0 * raw_val * raw_sig
    else:
        I_arr = raw_val
        sig_arr = raw_sig

    canonical_data = canonicalize_and_group(h_arr, k_arr, l_arr, I_arr, sig_arr, sg_number)

    hkl_encoded = (
        ((canonical_data["h"] + 512).astype(np.int64) << 40)
        | ((canonical_data["k"] + 512).astype(np.int64) << 20)
        | (canonical_data["l"] + 512).astype(np.int64)
    )

    return {
        "hkl_encoded": hkl_encoded,
        "I": canonical_data["I"],
        "sigma": canonical_data["sigma"],
    }

def load_and_canonicalize_single_dataset(path, sg_number):
    return process_mtz_file(path, sg_number)

def load_and_canonicalize_datasets(folder, sg_number):
    lazy_datasets = {}
    datasets_list = sorted(glob.glob(f"{folder}/*.mtz"))

    for path in datasets_list:
        lazy_datasets[path] = load_and_canonicalize_single_dataset(path, sg_number)

    print(f"Created Dask task graph recipes for {len(lazy_datasets)} datasets.")
    return lazy_datasets