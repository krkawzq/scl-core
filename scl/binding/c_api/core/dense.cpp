// =============================================================================
// FILE: scl/binding/c_api/dense_matrix.cpp
// BRIEF: Dense matrix C API implementation (placeholder)
// =============================================================================

#include "scl/binding/c_api/core/dense.h"
#include "scl/binding/c_api/core/sparse.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/core/type.hpp"
#include "scl/core/registry.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Lifecycle Management
// =============================================================================

scl_error_t scl_dense_create(
    scl_dense_t* out,
    scl_index_t rows,
    scl_index_t cols,
    const scl_real_t* data)
{
    if (!out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Output pointer is null");
        return SCL_ERROR_NULL_POINTER;
    }
    if (!data) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Input data pointer is null");
        return SCL_ERROR_NULL_POINTER;
    }
    if (rows <= 0 || cols <= 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid dimensions");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<DenseMatrixWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        Size total_size = static_cast<Size>(rows) * static_cast<Size>(cols);
        wrapper->data = reg.new_array<Real>(total_size);
        if (!wrapper->data) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate data");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        std::memcpy(wrapper->data, data, total_size * sizeof(Real));
        wrapper->rows = rows;
        wrapper->cols = cols;
        wrapper->stride = cols;
        wrapper->owns_data = true;
        
        *out = wrapper;
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_dense_wrap(
    scl_dense_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* data,
    scl_index_t stride)
{
    if (!out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Output pointer is null");
        return SCL_ERROR_NULL_POINTER;
    }
    if (!data) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Input data pointer is null");
        return SCL_ERROR_NULL_POINTER;
    }
    if (rows <= 0 || cols <= 0 || stride < cols) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid dimensions or stride");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<DenseMatrixWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        wrapper->data = reinterpret_cast<Real*>(data);
        wrapper->rows = rows;
        wrapper->cols = cols;
        wrapper->stride = stride;
        wrapper->owns_data = false;
        
        *out = wrapper;
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_dense_clone(scl_dense_t src, scl_dense_t* out) {
    if (!src || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto& reg = get_registry();
        auto* wrapper = reg.new_object<DenseMatrixWrapper>();
        if (!wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        Size total_size = static_cast<Size>(src->rows) * static_cast<Size>(src->cols);
        wrapper->data = reg.new_array<Real>(total_size);
        if (!wrapper->data) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate data");
            return SCL_ERROR_OUT_OF_MEMORY;
        }
        
        // Copy data row by row if stride differs
        for (Index i = 0; i < src->rows; ++i) {
            std::memcpy(
                wrapper->data + i * src->cols,
                src->data + i * src->stride,
                src->cols * sizeof(Real)
            );
        }
        
        wrapper->rows = src->rows;
        wrapper->cols = src->cols;
        wrapper->stride = src->cols;
        wrapper->owns_data = true;
        
        *out = wrapper;
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_dense_destroy(scl_dense_t* matrix) {
    if (!matrix || !*matrix) {
        return SCL_OK;
    }
    
    try {
        auto& reg = get_registry();
        reg.unregister_ptr(*matrix);
        *matrix = nullptr;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Property Queries
// =============================================================================

scl_error_t scl_dense_rows(scl_dense_t matrix, scl_index_t* out) {
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    *out = static_cast<scl_index_t>(matrix->rows);
    clear_last_error();
    return SCL_OK;
}

scl_error_t scl_dense_cols(scl_dense_t matrix, scl_index_t* out) {
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    *out = static_cast<scl_index_t>(matrix->cols);
    clear_last_error();
    return SCL_OK;
}

scl_error_t scl_dense_stride(scl_dense_t matrix, scl_index_t* out) {
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    *out = static_cast<scl_index_t>(matrix->stride);
    clear_last_error();
    return SCL_OK;
}

scl_error_t scl_dense_is_valid(scl_dense_t matrix, int* out) {
    if (!matrix || !out) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    *out = matrix->valid() ? 1 : 0;
    clear_last_error();
    return SCL_OK;
}

// =============================================================================
// Data Access
// =============================================================================

scl_error_t scl_dense_get_data(
    scl_dense_t matrix,
    const scl_real_t** out,
    scl_size_t* size)
{
    if (!matrix || !out || !size) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    *out = reinterpret_cast<const scl_real_t*>(matrix->data);
    *size = static_cast<scl_size_t>(matrix->rows) * static_cast<scl_size_t>(matrix->cols);
    clear_last_error();
    return SCL_OK;
}

scl_error_t scl_dense_export(
    scl_dense_t matrix,
    scl_real_t* data)
{
    if (!matrix || !data) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        for (Index i = 0; i < matrix->rows; ++i) {
            std::memcpy(
                data + i * matrix->cols,
                matrix->data + i * matrix->stride,
                matrix->cols * sizeof(Real)
            );
        }
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Conversion
// =============================================================================

scl_error_t scl_dense_to_sparse(
    scl_dense_t src,
    scl_sparse_t* out,
    int is_csr,
    scl_real_t epsilon)
{
    if (!src || !out) {
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
        
        wrapper->is_csr = (is_csr != 0);
        
        // Create dense vector view
        std::vector<Real> dense_data;
        dense_data.reserve(static_cast<Size>(src->rows) * static_cast<Size>(src->cols));
        
        for (Index i = 0; i < src->rows; ++i) {
            for (Index j = 0; j < src->cols; ++j) {
                dense_data.push_back(src->data[i * src->stride + j]);
            }
        }
        
        // Use from_dense with epsilon threshold
        if (is_csr) {
            wrapper->matrix = CSR::from_dense(
                src->rows, src->cols,
                dense_data,
                [epsilon](Real val) { return std::abs(val) > epsilon; }
            );
        } else {
            wrapper->matrix = CSC::from_dense(
                src->rows, src->cols,
                dense_data,
                [epsilon](Real val) { return std::abs(val) > epsilon; }
            );
        }
        
        if (!wrapper->valid()) {
            reg.unregister_ptr(wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to convert to sparse");
            return SCL_ERROR_INTERNAL;
        }
        
        *out = wrapper;
        clear_last_error();
        return SCL_OK;
        
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

