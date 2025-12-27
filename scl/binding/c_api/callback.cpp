// =============================================================================
/// @file callback.cpp
/// @brief C API for Callback-Based Sparse Matrices
///
/// Exposes CallbackSparse functionality to Python and other languages via C ABI.
///
/// Usage Pattern:
///
/// 1. Create vtable with callback functions
/// 2. Call scl_create_callback_csr/csc to create handle
/// 3. Use handle with operator functions
/// 4. Call scl_destroy_callback to release
///
/// Thread Safety:
///
/// - Each handle should only be used from one thread at a time
/// - Callbacks are invoked synchronously
/// - Python callbacks hold the GIL
// =============================================================================

#include "scl/binding/c_api/scl_c_api.h"
#include "scl/binding/c_api/error.hpp"
#include "scl/core/callback_sparse.hpp"
#include "scl/kernel/sparse.hpp"

#include <unordered_map>
#include <mutex>
#include <atomic>

using namespace scl;

// =============================================================================
// Handle Management
// =============================================================================

namespace {

// Thread-safe handle registry
std::mutex g_callback_mutex;
std::unordered_map<scl_callback_handle_t, void*> g_callback_handles;
std::atomic<scl_callback_handle_t> g_next_handle{1};

// Store handle and return ID
scl_callback_handle_t register_handle(void* raw_ptr) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    scl_callback_handle_t handle = g_next_handle++;
    g_callback_handles[handle] = raw_ptr;
    return handle;
}

// Get pointer from handle
void* get_handle_ptr(scl_callback_handle_t handle) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    auto it = g_callback_handles.find(handle);
    if (it == g_callback_handles.end()) {
        return nullptr;
    }
    return it->second;
}

// Remove handle and return pointer (for deletion)
void* unregister_handle(scl_callback_handle_t handle) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    auto it = g_callback_handles.find(handle);
    if (it == g_callback_handles.end()) {
        return nullptr;
    }
    void* raw_ptr = it->second;
    g_callback_handles.erase(it);
    return raw_ptr;
}

} // anonymous namespace

// =============================================================================
// Lifecycle Functions
// =============================================================================

extern "C" {

int scl_create_callback_csr(
    void* context,
    const scl_callback_vtable_t* vtable,
    scl_callback_handle_t* out_handle
) {
    SCL_C_API_WRAPPER(
        if (!vtable || !out_handle) {
            throw ValueError("Null vtable or out_handle");
        }

        // Convert C vtable to C++ vtable
        SparseCallbackVTable cpp_vtable;
        cpp_vtable.get_rows = vtable->get_rows;
        cpp_vtable.get_cols = vtable->get_cols;
        cpp_vtable.get_nnz = vtable->get_nnz;
        cpp_vtable.get_primary_values = vtable->get_primary_values;
        cpp_vtable.get_primary_indices = vtable->get_primary_indices;
        cpp_vtable.get_primary_length = vtable->get_primary_length;
        cpp_vtable.prefetch_range = vtable->prefetch_range;
        cpp_vtable.release_primary = vtable->release_primary;

        if (!cpp_vtable.is_valid()) {
            throw ValueError("Invalid vtable: missing required callbacks");
        }

        // Allocate vtable copy (persistent)
        auto* vtable_copy = new SparseCallbackVTable(cpp_vtable);

        // Create callback CSR
        auto* csr = new CallbackCSR<Real>(context, vtable_copy);

        *out_handle = register_handle(csr);
    );
}

int scl_create_callback_csc(
    void* context,
    const scl_callback_vtable_t* vtable,
    scl_callback_handle_t* out_handle
) {
    SCL_C_API_WRAPPER(
        if (!vtable || !out_handle) {
            throw ValueError("Null vtable or out_handle");
        }

        // Convert C vtable to C++ vtable
        SparseCallbackVTable cpp_vtable;
        cpp_vtable.get_rows = vtable->get_rows;
        cpp_vtable.get_cols = vtable->get_cols;
        cpp_vtable.get_nnz = vtable->get_nnz;
        cpp_vtable.get_primary_values = vtable->get_primary_values;
        cpp_vtable.get_primary_indices = vtable->get_primary_indices;
        cpp_vtable.get_primary_length = vtable->get_primary_length;
        cpp_vtable.prefetch_range = vtable->prefetch_range;
        cpp_vtable.release_primary = vtable->release_primary;

        if (!cpp_vtable.is_valid()) {
            throw ValueError("Invalid vtable: missing required callbacks");
        }

        // Allocate vtable copy (persistent)
        auto* vtable_copy = new SparseCallbackVTable(cpp_vtable);

        // Create callback CSC
        auto* csc = new CallbackCSC<Real>(context, vtable_copy);

        *out_handle = register_handle(csc);
    );
}

int scl_destroy_callback_csr(scl_callback_handle_t handle) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = unregister_handle(handle);
        if (!raw_ptr) {
            throw ValueError("Invalid callback handle");
        }

        auto* csr = static_cast<CallbackCSR<Real>*>(raw_ptr);
        // Delete vtable copy
        delete csr->vtable();
        delete csr;
    );
}

int scl_destroy_callback_csc(scl_callback_handle_t handle) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = unregister_handle(handle);
        if (!raw_ptr) {
            throw ValueError("Invalid callback handle");
        }

        auto* csc = static_cast<CallbackCSC<Real>*>(raw_ptr);
        // Delete vtable copy
        delete csc->vtable();
        delete csc;
    );
}

// =============================================================================
// Property Functions
// =============================================================================

int scl_callback_csr_shape(
    scl_callback_handle_t handle,
    scl_index_t* out_rows,
    scl_index_t* out_cols,
    scl_index_t* out_nnz
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr) {
            throw ValueError("Invalid callback handle");
        }

        auto* csr = static_cast<CallbackCSR<Real>*>(raw_ptr);
        if (out_rows) *out_rows = csr->rows();
        if (out_cols) *out_cols = csr->cols();
        if (out_nnz) *out_nnz = csr->nnz();
    );
}

int scl_callback_csc_shape(
    scl_callback_handle_t handle,
    scl_index_t* out_rows,
    scl_index_t* out_cols,
    scl_index_t* out_nnz
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr) {
            throw ValueError("Invalid callback handle");
        }

        auto* csc = static_cast<CallbackCSC<Real>*>(raw_ptr);
        if (out_rows) *out_rows = csc->rows();
        if (out_cols) *out_cols = csc->cols();
        if (out_nnz) *out_nnz = csc->nnz();
    );
}

// =============================================================================
// Statistical Operations (CSR)
// =============================================================================

int scl_callback_csr_row_sums(
    scl_callback_handle_t handle,
    scl_real_t* output
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr || !output) {
            throw ValueError("Invalid handle or null output");
        }

        auto* csr = static_cast<CallbackCSR<Real>*>(raw_ptr);
        Array<Real> out_array(output, static_cast<Size>(csr->rows()));

        // Use the generic kernel that works with any CSRLike type
        kernel::sparse::primary_sums(*csr, out_array);
    );
}

int scl_callback_csr_row_means(
    scl_callback_handle_t handle,
    scl_real_t* output
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr || !output) {
            throw ValueError("Invalid handle or null output");
        }

        auto* csr = static_cast<CallbackCSR<Real>*>(raw_ptr);
        Array<Real> out_array(output, static_cast<Size>(csr->rows()));

        kernel::sparse::primary_means(*csr, out_array);
    );
}

int scl_callback_csr_row_variances(
    scl_callback_handle_t handle,
    scl_real_t* output,
    int ddof
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr || !output) {
            throw ValueError("Invalid handle or null output");
        }

        auto* csr = static_cast<CallbackCSR<Real>*>(raw_ptr);
        Array<Real> out_array(output, static_cast<Size>(csr->rows()));

        kernel::sparse::primary_variances(*csr, out_array, ddof);
    );
}

int scl_callback_csr_row_nnz(
    scl_callback_handle_t handle,
    scl_index_t* output
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr || !output) {
            throw ValueError("Invalid handle or null output");
        }

        auto* csr = static_cast<CallbackCSR<Real>*>(raw_ptr);
        Index n_rows = csr->rows();

        // Manual implementation for nnz counts
        for (Index i = 0; i < n_rows; ++i) {
            output[i] = csr->primary_length(i);
        }
    );
}

// =============================================================================
// Statistical Operations (CSC)
// =============================================================================

int scl_callback_csc_col_sums(
    scl_callback_handle_t handle,
    scl_real_t* output
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr || !output) {
            throw ValueError("Invalid handle or null output");
        }

        auto* csc = static_cast<CallbackCSC<Real>*>(raw_ptr);
        Array<Real> out_array(output, static_cast<Size>(csc->cols()));

        kernel::sparse::primary_sums(*csc, out_array);
    );
}

int scl_callback_csc_col_means(
    scl_callback_handle_t handle,
    scl_real_t* output
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr || !output) {
            throw ValueError("Invalid handle or null output");
        }

        auto* csc = static_cast<CallbackCSC<Real>*>(raw_ptr);
        Array<Real> out_array(output, static_cast<Size>(csc->cols()));

        kernel::sparse::primary_means(*csc, out_array);
    );
}

int scl_callback_csc_col_variances(
    scl_callback_handle_t handle,
    scl_real_t* output,
    int ddof
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr || !output) {
            throw ValueError("Invalid handle or null output");
        }

        auto* csc = static_cast<CallbackCSC<Real>*>(raw_ptr);
        Array<Real> out_array(output, static_cast<Size>(csc->cols()));

        kernel::sparse::primary_variances(*csc, out_array, ddof);
    );
}

int scl_callback_csc_col_nnz(
    scl_callback_handle_t handle,
    scl_index_t* output
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr || !output) {
            throw ValueError("Invalid handle or null output");
        }

        auto* csc = static_cast<CallbackCSC<Real>*>(raw_ptr);
        Index n_cols = csc->cols();

        for (Index j = 0; j < n_cols; ++j) {
            output[j] = csc->primary_length(j);
        }
    );
}

// =============================================================================
// Utility Functions
// =============================================================================

int scl_callback_csr_prefetch(
    scl_callback_handle_t handle,
    scl_index_t start,
    scl_index_t end
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr) {
            throw ValueError("Invalid callback handle");
        }

        auto* csr = static_cast<CallbackCSR<Real>*>(raw_ptr);
        csr->prefetch(start, end);
    );
}

int scl_callback_csc_prefetch(
    scl_callback_handle_t handle,
    scl_index_t start,
    scl_index_t end
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr) {
            throw ValueError("Invalid callback handle");
        }

        auto* csc = static_cast<CallbackCSC<Real>*>(raw_ptr);
        csc->prefetch(start, end);
    );
}

int scl_callback_csr_invalidate_cache(scl_callback_handle_t handle) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr) {
            throw ValueError("Invalid callback handle");
        }

        auto* csr = static_cast<CallbackCSR<Real>*>(raw_ptr);
        csr->invalidate_cache();
    );
}

int scl_callback_csc_invalidate_cache(scl_callback_handle_t handle) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr) {
            throw ValueError("Invalid callback handle");
        }

        auto* csc = static_cast<CallbackCSC<Real>*>(raw_ptr);
        csc->invalidate_cache();
    );
}

// =============================================================================
// Direct Row/Column Access (for debugging/testing)
// =============================================================================

int scl_callback_csr_get_row(
    scl_callback_handle_t handle,
    scl_index_t row_idx,
    scl_real_t** out_values,
    scl_index_t** out_indices,
    scl_index_t* out_len
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr) {
            throw ValueError("Invalid callback handle");
        }

        auto* csr = static_cast<CallbackCSR<Real>*>(raw_ptr);

        auto values = csr->primary_values(row_idx);
        auto indices = csr->primary_indices(row_idx);

        if (out_values) *out_values = values.ptr;
        if (out_indices) *out_indices = indices.ptr;
        if (out_len) *out_len = static_cast<Index>(values.len);
    );
}

int scl_callback_csc_get_col(
    scl_callback_handle_t handle,
    scl_index_t col_idx,
    scl_real_t** out_values,
    scl_index_t** out_indices,
    scl_index_t* out_len
) {
    SCL_C_API_WRAPPER(
        void* raw_ptr = get_handle_ptr(handle);
        if (!raw_ptr) {
            throw ValueError("Invalid callback handle");
        }

        auto* csc = static_cast<CallbackCSC<Real>*>(raw_ptr);

        auto values = csc->primary_values(col_idx);
        auto indices = csc->primary_indices(col_idx);

        if (out_values) *out_values = values.ptr;
        if (out_indices) *out_indices = indices.ptr;
        if (out_len) *out_len = static_cast<Index>(values.len);
    );
}

} // extern "C"
