// =============================================================================
// FILE: scl/binding/c_api/unsafe.cpp
// BRIEF: Unsafe C API implementation for raw struct access
// =============================================================================

#include "scl/binding/c_api/core/unsafe.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/registry.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Sparse Matrix Unsafe Operations
// =============================================================================

scl_error_t scl_sparse_unsafe_get_raw(
    scl_sparse_t matrix,
    scl_sparse_raw_t* out)
{
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        out->is_csr = matrix->is_csr ? 1 : 0;
        
        matrix->visit([&](auto& m) {
            out->data_ptrs = reinterpret_cast<void**>(m.data_ptrs);
            out->indices_ptrs = reinterpret_cast<void**>(m.indices_ptrs);
            out->lengths = reinterpret_cast<scl_index_t*>(m.lengths);
            out->rows = static_cast<scl_index_t>(m.rows_);
            out->cols = static_cast<scl_index_t>(m.cols_);
            out->nnz = static_cast<scl_index_t>(m.nnz_);
            out->owns_data = m.owns_data_ ? 1 : 0;
            out->is_view = m.is_view_ ? 1 : 0;
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_unsafe_from_raw(
    const scl_sparse_raw_t* raw,
    scl_sparse_t* out)
{
    if (!raw || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<SparseWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->is_csr = (raw->is_csr != 0);
        
        // Create sparse matrix from raw pointers
        // WARNING: This is dangerous - assumes raw struct was properly allocated
        if (wrapper->is_csr) {
            CSR matrix;
            matrix.data_ptrs = reinterpret_cast<Pointer*>(raw->data_ptrs);
            matrix.indices_ptrs = reinterpret_cast<Pointer*>(raw->indices_ptrs);
            matrix.lengths = reinterpret_cast<Index*>(raw->lengths);
            matrix.rows_ = raw->rows;
            matrix.cols_ = raw->cols;
            matrix.nnz_ = raw->nnz;
            matrix.owns_data_ = (raw->owns_data != 0);
            matrix.is_view_ = (raw->is_view != 0);
            wrapper->matrix = std::move(matrix);
        } else {
            CSC matrix;
            matrix.data_ptrs = reinterpret_cast<Pointer*>(raw->data_ptrs);
            matrix.indices_ptrs = reinterpret_cast<Pointer*>(raw->indices_ptrs);
            matrix.lengths = reinterpret_cast<Index*>(raw->lengths);
            matrix.rows_ = raw->rows;
            matrix.cols_ = raw->cols;
            matrix.nnz_ = raw->nnz;
            matrix.owns_data_ = (raw->owns_data != 0);
            matrix.is_view_ = (raw->is_view != 0);
            wrapper->matrix = std::move(matrix);
        }
        
        *out = wrapper;
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Dense Matrix Unsafe Operations
// =============================================================================

scl_error_t scl_dense_unsafe_get_raw(
    scl_dense_t matrix,
    scl_dense_raw_t* out)
{
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    out->data = reinterpret_cast<scl_real_t*>(matrix->data);
    out->rows = static_cast<scl_index_t>(matrix->rows);
    out->cols = static_cast<scl_index_t>(matrix->cols);
    out->stride = static_cast<scl_index_t>(matrix->stride);
    out->owns_data = matrix->owns_data ? 1 : 0;
    
    clear_last_error();
    return SCL_OK;
}

scl_error_t scl_dense_unsafe_from_raw(
    const scl_dense_raw_t* raw,
    scl_dense_t* out)
{
    if (!raw || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto& reg = get_registry();
        
        // Forward declare the internal wrapper
        struct DenseMatrixWrapper {
            Real* data;
            Index rows;
            Index cols;
            Index stride;
            bool owns_data;
        };
        
        auto* wrapper = reg.new_object<DenseMatrixWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->data = reinterpret_cast<Real*>(raw->data);
        wrapper->rows = raw->rows;
        wrapper->cols = raw->cols;
        wrapper->stride = raw->stride;
        wrapper->owns_data = (raw->owns_data != 0);
        
        *out = reinterpret_cast<scl_dense_t>(wrapper);
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

