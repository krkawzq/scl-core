// =============================================================================
/// @file mmap.cpp
/// @brief Memory-Mapped Sparse Matrix C API Implementation
///
/// Provides C-ABI functions for memory-mapped sparse matrix operations.
/// Uses handle-based object management for Python/FFI interoperability.
// =============================================================================

#include "scl/binding/c_api/error.hpp"
#include "scl/core/error.hpp"
#include "scl/mmap/core.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace scl::binding::mmap_detail {

// =============================================================================
// Handle Registry (Thread-Safe)
// =============================================================================

using Handle = int64_t;

/// @brief Type names for registered objects
constexpr const char* TYPE_MAPPED_CSR = "MappedCSR";
[[maybe_unused]] constexpr const char* TYPE_MAPPED_CSC = "MappedCSC";
constexpr const char* TYPE_VIRTUAL_CSR = "VirtualCSR";
[[maybe_unused]] constexpr const char* TYPE_VIRTUAL_CSC = "VirtualCSC";

class Registry {
    std::mutex mutex_;
    Handle next_handle_ = 1;
    std::unordered_map<Handle, std::shared_ptr<void>> objects_;
    std::unordered_map<Handle, std::string> types_;

public:
    static Registry& instance() {
        static Registry inst;
        return inst;
    }

    template <typename T>
    Handle register_object(std::shared_ptr<T> obj, const char* type_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        Handle h = next_handle_++;
        objects_[h] = std::static_pointer_cast<void>(obj);
        types_[h] = type_name;
        return h;
    }

    template <typename T>
    std::shared_ptr<T> get(Handle h) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = objects_.find(h);
        if (it == objects_.end()) return nullptr;
        return std::static_pointer_cast<T>(it->second);
    }

    bool release(Handle h) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = objects_.find(h);
        if (it == objects_.end()) return false;
        objects_.erase(it);
        types_.erase(h);
        return true;
    }

    const char* type_of(Handle h) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = types_.find(h);
        if (it == types_.end()) return nullptr;
        return it->second.c_str();
    }
};

} // namespace scl::binding::mmap_detail

// =============================================================================
// C API Implementation
// =============================================================================

extern "C" {

using namespace scl::mmap;
using namespace scl::binding::mmap_detail;

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------

int scl_mmap_create_csr_from_ptr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Index max_pages,
    int64_t* out_handle
) {
    SCL_C_API_WRAPPER(
        MmapConfig config;
        config.max_resident_pages = static_cast<std::size_t>(max_pages);

        std::size_t data_bytes = static_cast<std::size_t>(nnz) * sizeof(scl::Real);
        std::size_t indices_bytes = static_cast<std::size_t>(nnz) * sizeof(scl::Index);
        std::size_t indptr_bytes = static_cast<std::size_t>(rows + 1) * sizeof(scl::Index);

        auto mat = std::make_shared<MappedCSR<scl::Real>>(
            rows, cols, nnz,
            make_ptr_loader(data, data_bytes),
            make_ptr_loader(indices, indices_bytes),
            make_ptr_loader(indptr, indptr_bytes),
            config
        );

        *out_handle = Registry::instance().register_object(mat, TYPE_MAPPED_CSR);
    )
}

int scl_mmap_open_csr_file(
    const char* filepath,
    scl::Index max_pages,
    int64_t* out_handle
) {
    SCL_C_API_WRAPPER(
        MmapConfig config;
        config.max_resident_pages = static_cast<std::size_t>(max_pages);

        auto mat = open_sparse_file<scl::Real, true>(filepath, config);
        auto shared = std::shared_ptr<MappedCSR<scl::Real>>(mat.release());

        *out_handle = Registry::instance().register_object(shared, TYPE_MAPPED_CSR);
    )
}

int scl_mmap_release(int64_t handle) {
    SCL_C_API_WRAPPER(
        if (!Registry::instance().release(handle)) {
            throw scl::ValueError("Invalid handle");
        }
    )
}

const char* scl_mmap_type(int64_t handle) {
    return Registry::instance().type_of(handle);
}

// -----------------------------------------------------------------------------
// Properties
// -----------------------------------------------------------------------------

int scl_mmap_csr_shape(int64_t handle, scl::Index* rows, scl::Index* cols, scl::Index* nnz) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        *rows = mat->rows();
        *cols = mat->cols();
        *nnz = mat->nnz();
    )
}

// -----------------------------------------------------------------------------
// Load Operations
// -----------------------------------------------------------------------------

int scl_mmap_csr_load_full(
    int64_t handle,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        load_full(*mat, data_out, indices_out, indptr_out);
    )
}

int scl_mmap_csr_load_masked(
    int64_t handle,
    const uint8_t* row_mask,
    const uint8_t* col_mask,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out,
    scl::Index* out_rows,
    scl::Index* out_cols,
    scl::Index* out_nnz
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        load_masked(*mat, row_mask, col_mask,
                    data_out, indices_out, indptr_out,
                    out_rows, out_cols, out_nnz);
    )
}

int scl_mmap_csr_compute_masked_nnz(
    int64_t handle,
    const uint8_t* row_mask,
    const uint8_t* col_mask,
    scl::Index* out_nnz
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        *out_nnz = compute_masked_nnz(*mat, row_mask, col_mask);
    )
}

int scl_mmap_csr_load_indexed(
    int64_t handle,
    const scl::Index* row_indices,
    scl::Index num_rows,
    const scl::Index* col_indices,
    scl::Index num_cols,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out,
    scl::Index* out_nnz
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        load_indexed(*mat, row_indices, num_rows, col_indices, num_cols,
                     data_out, indices_out, indptr_out, out_nnz);
    )
}

// -----------------------------------------------------------------------------
// View Operations
// -----------------------------------------------------------------------------

int scl_mmap_csr_create_view(
    int64_t handle,
    const uint8_t* row_mask,
    const uint8_t* col_mask,
    int64_t* out_handle
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        auto view = std::make_shared<MappedVirtualCSR<scl::Real>>(mat, row_mask, col_mask);
        *out_handle = Registry::instance().register_object(view, TYPE_VIRTUAL_CSR);
    )
}

int scl_mmap_view_shape(int64_t handle, scl::Index* rows, scl::Index* cols, scl::Index* nnz) {
    SCL_C_API_WRAPPER(
        auto view = Registry::instance().get<MappedVirtualCSR<scl::Real>>(handle);
        if (!view) throw scl::ValueError("Invalid handle");

        *rows = view->rows();
        *cols = view->cols();
        *nnz = view->nnz();
    )
}

// -----------------------------------------------------------------------------
// Reorder Operations
// -----------------------------------------------------------------------------

int scl_mmap_csr_reorder_rows(
    int64_t handle,
    const scl::Index* order,
    scl::Index count,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        reorder_rows(*mat, order, count, data_out, indices_out, indptr_out);
    )
}

int scl_mmap_csr_reorder_cols(
    int64_t handle,
    const scl::Index* col_order,
    scl::Index num_cols,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        reorder_cols(*mat, col_order, num_cols, data_out, indices_out, indptr_out);
    )
}

// -----------------------------------------------------------------------------
// Format Conversion
// -----------------------------------------------------------------------------

int scl_mmap_csr_to_csc(
    int64_t handle,
    scl::Real* csc_data,
    scl::Index* csc_indices,
    scl::Index* csc_indptr
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        csr_to_csc(*mat, csc_data, csc_indices, csc_indptr);
    )
}

int scl_mmap_csr_to_dense(int64_t handle, scl::Real* dense_out) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        to_dense(*mat, dense_out);
    )
}

// -----------------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------------

int scl_mmap_csr_row_sum(int64_t handle, scl::Real* out) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        row_sum(*mat, out);
    )
}

int scl_mmap_csr_row_mean(int64_t handle, scl::Real* out, int count_zeros) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        row_mean(*mat, out, count_zeros != 0);
    )
}

int scl_mmap_csr_row_var(int64_t handle, scl::Real* out, const scl::Real* means, int count_zeros) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        row_var(*mat, out, means, count_zeros != 0);
    )
}

int scl_mmap_csr_col_sum(int64_t handle, scl::Real* out) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        col_sum_csr(*mat, out);
    )
}

int scl_mmap_csr_global_sum(int64_t handle, scl::Real* out) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        *out = global_sum(*mat);
    )
}

// -----------------------------------------------------------------------------
// Normalization
// -----------------------------------------------------------------------------

int scl_mmap_csr_normalize_l1(
    int64_t handle,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        normalize_l1_rows(*mat, data_out, indices_out, indptr_out);
    )
}

int scl_mmap_csr_normalize_l2(
    int64_t handle,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        normalize_l2_rows(*mat, data_out, indices_out, indptr_out);
    )
}

// -----------------------------------------------------------------------------
// Transforms
// -----------------------------------------------------------------------------

int scl_mmap_csr_log1p(
    int64_t handle,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        log1p_transform(*mat, data_out, indices_out, indptr_out);
    )
}

int scl_mmap_csr_scale_rows(
    int64_t handle,
    const scl::Real* row_factors,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        scale_rows(*mat, row_factors, data_out, indices_out, indptr_out);
    )
}

int scl_mmap_csr_scale_cols(
    int64_t handle,
    const scl::Real* col_factors,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        scale_cols_csr(*mat, col_factors, data_out, indices_out, indptr_out);
    )
}

// -----------------------------------------------------------------------------
// SpMV
// -----------------------------------------------------------------------------

int scl_mmap_csr_spmv(int64_t handle, const scl::Real* x, scl::Real* y) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        spmv(*mat, x, y);
    )
}

// -----------------------------------------------------------------------------
// Filtering
// -----------------------------------------------------------------------------

int scl_mmap_csr_filter_threshold(
    int64_t handle,
    scl::Real threshold,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out,
    scl::Index* out_nnz
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        *out_nnz = filter_threshold(*mat, threshold, data_out, indices_out, indptr_out);
    )
}

int scl_mmap_csr_top_k(
    int64_t handle,
    scl::Index k,
    scl::Real* data_out,
    scl::Index* indices_out,
    scl::Index* indptr_out
) {
    SCL_C_API_WRAPPER(
        auto mat = Registry::instance().get<MappedCSR<scl::Real>>(handle);
        if (!mat) throw scl::ValueError("Invalid handle");

        top_k_per_row(*mat, k, data_out, indices_out, indptr_out);
    )
}

// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

int scl_mmap_get_config(scl::Index* page_size, scl::Index* default_pool_size) {
    SCL_C_API_WRAPPER(
        *page_size = SCL_MMAP_PAGE_SIZE;
        *default_pool_size = SCL_MMAP_DEFAULT_POOL_SIZE;
    )
}

int scl_mmap_estimate_memory(scl::Index rows, scl::Index nnz, scl::Index* out_bytes) {
    SCL_C_API_WRAPPER(
        *out_bytes = static_cast<scl::Index>(
            estimate_sparse_memory<scl::Real>(rows, nnz));
    )
}

int scl_mmap_suggest_backend(scl::Index data_bytes, scl::Index available_mb, int* out_backend) {
    SCL_C_API_WRAPPER(
        BackendHint hint = suggest_backend(
            static_cast<std::size_t>(data_bytes),
            static_cast<std::size_t>(available_mb));
        *out_backend = static_cast<int>(hint);
    )
}

} // extern "C"
