#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <gemmi/mtz.hpp>
#include <gemmi/symmetry.hpp>

namespace py = pybind11;
namespace fs = std::filesystem;

inline int64_t encode_hkl(int h, int k, int l) {
    return ((int64_t)(h + 512) << 20) | ((int64_t)(k + 512) << 10) | (int64_t)(l + 512);
}

inline gemmi::Miller canonicalize_hkl(const gemmi::Miller& hkl, const gemmi::GroupOps& group_ops) {
    gemmi::Miller best = hkl;
    for (const auto& op : group_ops.sym_ops) {
        gemmi::Miller eq = op.apply_to_hkl(hkl);
        if (eq < best) {
            best = eq;
        }
    }
    return best;
}

struct DatasetPayload {
    std::string filename;
    std::string spacegroup;
    bool success = false;
    std::string error_msg;

    std::vector<int64_t> hkl_encoded;
    std::vector<float> i_mean;
    std::vector<float> sig_i_mean;
};

struct PYBIND11_EXPORT MatrixResult {
    std::vector<std::string> filenames;
    py::array_t<float> R;
    py::array_t<float> W;
    py::array_t<float> N;
};

int find_column_index(const gemmi::Mtz& mtz, const std::string& label) {
    for (size_t i = 0; i < mtz.columns.size(); ++i) {
        if (mtz.columns[i].label == label) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

struct ReflectionAcc {
    double sum_wI = 0.0;
    double sum_w = 0.0;

    void add(float I, float sig) {
        if (sig <= 0.0f || std::isnan(I) || std::isnan(sig)) return;
        double w = 1.0 / (double(sig) * double(sig));
        sum_wI += w * double(I);
        sum_w += w;
    }
};

DatasetPayload process_single_file(const std::string& filepath, int override_sg_number = 0) {
    DatasetPayload payload;
    payload.filename = fs::path(filepath).filename().string();

    try {
        gemmi::Mtz mtz = gemmi::read_mtz_file(filepath);
        payload.spacegroup = mtz.spacegroup_name;

        const gemmi::SpaceGroup* sg = nullptr;
        if (override_sg_number > 0) {
            sg = gemmi::find_spacegroup_by_number(override_sg_number);
        } else if (!mtz.spacegroup_name.empty()) {
            sg = gemmi::find_spacegroup_by_name(mtz.spacegroup_name);
        }

        if (!sg) {
            payload.success = false;
            payload.error_msg = "Could not identify valid spacegroup for symmetry reduction.";
            return payload;
        }

        gemmi::GroupOps group_ops = sg->operations();

        int idx_i = find_column_index(mtz, "IMEAN");
        if (idx_i < 0) idx_i = find_column_index(mtz, "I");

        int idx_s = find_column_index(mtz, "SIGIMEAN");
        if (idx_s < 0) idx_s = find_column_index(mtz, "SIGI");

        int idx_h = find_column_index(mtz, "H");
        int idx_k = find_column_index(mtz, "K");
        int idx_l = find_column_index(mtz, "L");

        if (idx_i < 0 || idx_s < 0 || idx_h < 0 || idx_k < 0 || idx_l < 0) {
            payload.success = false;
            payload.error_msg = "Required HKL or Intensity/Sigma columns not found.";
            return payload;
        }

        size_t nref = mtz.nreflections;
        size_t ncols = mtz.columns.size();

        std::unordered_map<int64_t, ReflectionAcc> merged_map;
        merged_map.reserve(nref);

        for (size_t i = 0; i < nref; ++i) {
            int h = static_cast<int>(mtz.data[i * ncols + idx_h]);
            int k = static_cast<int>(mtz.data[i * ncols + idx_k]);
            int l = static_cast<int>(mtz.data[i * ncols + idx_l]);
            float val = mtz.data[i * ncols + idx_i];
            float sig = mtz.data[i * ncols + idx_s];

            gemmi::Miller canonical = canonicalize_hkl({h, k, l}, group_ops);
            int64_t key = encode_hkl(canonical[0], canonical[1], canonical[2]);

            merged_map[key].add(val, sig);
        }

        std::vector<int64_t> keys;
        keys.reserve(merged_map.size());
        for (const auto& [key, acc] : merged_map) {
            if (acc.sum_w > 0.0) {
                keys.push_back(key);
            }
        }

        std::sort(keys.begin(), keys.end());

        payload.hkl_encoded.reserve(keys.size());
        payload.i_mean.reserve(keys.size());
        payload.sig_i_mean.reserve(keys.size());

        for (int64_t key : keys) {
            const auto& acc = merged_map[key];
            payload.hkl_encoded.push_back(key);
            payload.i_mean.push_back(static_cast<float>(acc.sum_wI / acc.sum_w));
            payload.sig_i_mean.push_back(static_cast<float>(std::sqrt(1.0 / acc.sum_w)));
        }

        payload.success = true;
    } catch (const std::exception& e) {
        payload.success = false;
        payload.error_msg = e.what();
    }
    return payload;
}

inline std::pair<float, int> compute_weighted_cc_pair(const DatasetPayload& d1, const DatasetPayload& d2) {
    size_t i = 0, j = 0;
    size_t n1 = d1.hkl_encoded.size();
    size_t n2 = d2.hkl_encoded.size();

    double sum_w = 0.0;
    double sum_wx = 0.0;
    double sum_wy = 0.0;

    std::vector<double> wx, wy, w_vec;
    wx.reserve(std::min(n1, n2));
    wy.reserve(std::min(n1, n2));
    w_vec.reserve(std::min(n1, n2));

    while (i < n1 && j < n2) {
        int64_t hkl1 = d1.hkl_encoded[i];
        int64_t hkl2 = d2.hkl_encoded[j];

        if (hkl1 == hkl2) {
            double sig1 = d1.sig_i_mean[i];
            double sig2 = d2.sig_i_mean[j];

            double w = 1.0 / (sig1 * sig1 + sig2 * sig2);

            double x = d1.i_mean[i];
            double y = d2.i_mean[j];

            w_vec.push_back(w);
            wx.push_back(x);
            wy.push_back(y);

            sum_w += w;
            sum_wx += w * x;
            sum_wy += w * y;

            ++i;
            ++j;
        } else if (hkl1 < hkl2) {
            ++i;
        } else {
            ++j;
        }
    }

    int n_common = static_cast<int>(w_vec.size());
    if (n_common == 0 || sum_w == 0.0) return {0.0f, 0};

    double x_bar = sum_wx / sum_w;
    double y_bar = sum_wy / sum_w;

    double s_xy_sum = 0.0, s_xx_sum = 0.0, s_yy_sum = 0.0;

    for (size_t k = 0; k < w_vec.size(); ++k) {
        double dx = wx[k] - x_bar;
        double dy = wy[k] - y_bar;
        double w = w_vec[k];

        s_xy_sum += w * dx * dy;
        s_xx_sum += w * dx * dx;
        s_yy_sum += w * dy * dy;
    }

    double s_xy = s_xy_sum / sum_w;
    double s_xx = s_xx_sum / sum_w;
    double s_yy = s_yy_sum / sum_w;

    double denom = std::sqrt(s_xx * s_yy);
    if (denom == 0.0 || std::isnan(denom)) return {0.0f, n_common};

    float cc = static_cast<float>(s_xy / denom);
    if (std::isnan(cc)) cc = 0.0f;

    return {cc, n_common};
}

std::vector<DatasetPayload> read_mtz_batch(const std::string& dir_path, int sg_number = 0, int num_threads = 8) {
    std::vector<std::string> file_paths;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".mtz") {
            file_paths.push_back(entry.path().string());
        }
    }

    std::vector<DatasetPayload> results(file_paths.size());

    py::gil_scoped_release release;

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (size_t i = 0; i < file_paths.size(); ++i) {
        results[i] = process_single_file(file_paths[i], sg_number);
    }

    py::gil_scoped_acquire acquire;

    return results;
}

MatrixResult compute_cc_matrix_batch(const std::string& dir_path, 
                                     int sg_number = 0, 
                                     bool use_counts_as_weights = false, 
                                     int num_threads = 8) {
    std::vector<std::string> file_paths;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".mtz") {
            file_paths.push_back(entry.path().string());
        }
    }

    std::sort(file_paths.begin(), file_paths.end());

    size_t n = file_paths.size();
    std::vector<DatasetPayload> datasets(n);
    std::vector<std::string> filenames(n);

    py::gil_scoped_release release;

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (size_t i = 0; i < n; ++i) {
        datasets[i] = process_single_file(file_paths[i], sg_number);
        filenames[i] = fs::path(file_paths[i]).filename().string();
    }

    std::vector<float> R_data(n * n, 0.0f);
    std::vector<float> W_data(n * n, 0.0f);
    std::vector<float> N_data(n * n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        R_data[i * n + i] = 1.0f;
    }

    double sum_w = 0.0;
    size_t non_zero_w_count = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:sum_w, non_zero_w_count) num_threads(num_threads)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            auto [cc, n_common] = compute_weighted_cc_pair(datasets[i], datasets[j]);

            if (n_common < 2 || std::isnan(cc)) {
                cc = 0.0f;
            }

            R_data[i * n + j] = cc;
            R_data[j * n + i] = cc;

            N_data[i * n + j] = static_cast<float>(n_common);
            N_data[j * n + i] = static_cast<float>(n_common);

            if (use_counts_as_weights) {
                float w = static_cast<float>(n_common);
                W_data[i * n + j] = w;
                W_data[j * n + i] = w;
                sum_w += 2.0 * w;
                if (w > 0.0f) non_zero_w_count += 2;
            } else {
                if (n_common > 1) {
                    double r_sq = static_cast<double>(cc) * static_cast<double>(cc);
                    double denom = 1.0 - r_sq;
                    if (denom < 1e-6) denom = 1e-6;

                    double w = std::sqrt(static_cast<double>(n_common - 1)) / denom;
                    float w_float = static_cast<float>(w);

                    W_data[i * n + j] = w_float;
                    W_data[j * n + i] = w_float;

                    sum_w += 2.0 * w;
                    non_zero_w_count += 2;
                }
            }
        }
    }

    if (sum_w > 0.0 && non_zero_w_count > 0) {
        double scale = static_cast<double>(non_zero_w_count) / sum_w;
        for (size_t k = 0; k < n * n; ++k) {
            W_data[k] = static_cast<float>(W_data[k] * scale);
        }
    }

    py::gil_scoped_acquire acquire;
    py::array_t<float> R_arr({n, n}, R_data.data());
    py::array_t<float> W_arr({n, n}, W_data.data());
    py::array_t<float> N_arr({n, n}, N_data.data());

    return {filenames, R_arr, W_arr, N_arr};
}

PYBIND11_MODULE(fast_mtz, m) {
    py::class_<DatasetPayload>(m, "DatasetPayload")
        .def_readonly("filename", &DatasetPayload::filename)
        .def_readonly("spacegroup", &DatasetPayload::spacegroup)
        .def_readonly("success", &DatasetPayload::success)
        .def_readonly("error_msg", &DatasetPayload::error_msg)
        .def_readonly("hkl_encoded", &DatasetPayload::hkl_encoded)
        .def_readonly("i_mean", &DatasetPayload::i_mean)
        .def_readonly("sig_i_mean", &DatasetPayload::sig_i_mean);

    py::class_<MatrixResult>(m, "MatrixResult")
        .def_readonly("filenames", &MatrixResult::filenames)
        .def_readonly("R", &MatrixResult::R)
        .def_readonly("W", &MatrixResult::W)
        .def_readonly("N", &MatrixResult::N);

    m.def("read_batch", &read_mtz_batch, py::arg("dir_path"), py::arg("sg_number") = 0, py::arg("num_threads") = 8);
    m.def("compute_cc_matrix", &compute_cc_matrix_batch, py::arg("dir_path"), py::arg("sg_number") = 0, py::arg("use_counts_as_weights") = false, py::arg("num_threads") = 8);
}