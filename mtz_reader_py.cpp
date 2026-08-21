#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdint>
#include <stdexcept>
#include <omp.h>
#include <gemmi/mtz.hpp>
#include <iostream>

namespace py = pybind11;
namespace fs = std::filesystem;

inline int64_t encode_hkl(int h, int k, int l) {
    return ((int64_t)(h + 512) << 20) | ((int64_t)(k + 512) << 10) | (int64_t)(l + 512);
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

int find_column_index(const gemmi::Mtz& mtz, const std::string& label) {
    for (size_t i = 0; i < mtz.columns.size(); ++i) {
        if (mtz.columns[i].label == label) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

DatasetPayload process_single_file(const std::string& filepath) {
    DatasetPayload payload;
    payload.filename = fs::path(filepath).filename().string();

    try {
        gemmi::Mtz mtz = gemmi::read_mtz_file(filepath);
        payload.spacegroup = mtz.spacegroup_name;

        int idx_i = find_column_index(mtz, "IMEAN");
        if (idx_i < 0) idx_i = find_column_index(mtz, "I");

        int idx_s = find_column_index(mtz, "SIGIMEAN");
        if (idx_s < 0) idx_s = find_column_index(mtz, "SIGI");

        int idx_h = find_column_index(mtz, "H");
        int idx_k = find_column_index(mtz, "K");
        int idx_l = find_column_index(mtz, "L");

        if (idx_i < 0 || idx_s < 0 || idx_h < 0 || idx_k < 0 || idx_l < 0) {
            // std::cout << "idx_i: " << idx_i << std::endl;
            // std::cout << "idx_s: " << idx_s << std::endl;
            // std::cout << "idx_h: " << idx_h << std::endl;
            // std::cout << "idx_k: " << idx_k << std::endl;
            // std::cout << "idx_l: " << idx_l << std::endl;
            payload.success = false;
            payload.error_msg = "Required HKL or Intensity/Sigma columns not found.";
            return payload;
        }

        size_t nref = mtz.nreflections;
        size_t ncols = mtz.columns.size();

        payload.hkl_encoded.reserve(nref);
        payload.i_mean.reserve(nref);
        payload.sig_i_mean.reserve(nref);

        for (size_t i = 0; i < nref; ++i) {
            int h = static_cast<int>(mtz.data[i * ncols + idx_h]);
            int k = static_cast<int>(mtz.data[i * ncols + idx_k]);
            int l = static_cast<int>(mtz.data[i * ncols + idx_l]);
            float val = mtz.data[i * ncols + idx_i];
            float sig = mtz.data[i * ncols + idx_s];

            payload.hkl_encoded.push_back(encode_hkl(h, k, l));
            payload.i_mean.push_back(val);
            payload.sig_i_mean.push_back(sig);
        }

        payload.success = true;
    } catch (const std::exception& e) {
        payload.success = false;
        payload.error_msg = e.what();
    }
    return payload;
}

std::vector<DatasetPayload> read_mtz_batch(const std::string& dir_path, int num_threads = 8) {
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
        results[i] = process_single_file(file_paths[i]);
    }

    py::gil_scoped_acquire acquire;

    return results;
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

    m.def("read_batch", &read_mtz_batch, py::arg("dir_path"), py::arg("num_threads") = 8);
}