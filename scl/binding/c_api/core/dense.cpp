// =============================================================================
// FILE: scl/binding/c_api/core/dense.cpp
// BRIEF: Dense matrix view C API implementation
// =============================================================================
//
// This file implements view-only operations for dense matrices.
// The library does NOT allocate dense matrices - all dense data is
// managed by the caller (e.g., NumPy, PyTorch).
// =============================================================================

#include "scl/binding/c_api/core/dense.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/core/type.hpp"
#include "scl/core/registry.hpp"
#include "scl/core/error.hpp"

#include <algorithm>
#include <cstring>

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Lifecycle Management
// =============================================================================

SCL_EXPORT scl_error_t scl_dense_wrap(
    scl_dense_t* out,
    const scl_index_t rows,
    const scl_index_t cols,
    scl_real_t* data,
    const scl_index_t stride) {
    
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK_NULL(data, "Input data pointer is null");
    SCL_C_API_CHECK(rows > 0 && cols > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Matrix dimensions must be positive");
    SCL_C_API_CHECK(stride >= cols, SCL_ERROR_INVALID_ARGUMENT,
                   "Stride must be >= cols");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        // Allocate view handle only (data is external)
        auto* handle = reg.new_object<scl_dense_matrix>(
            reinterpret_cast<Real*>(data),
            rows, cols, stride
        );
        SCL_CHECK_NULL(handle, "Failed to allocate dense view handle");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_dense_destroy(scl_dense_t* matrix) {
    if (matrix == nullptr || *matrix == nullptr) {
        SCL_C_API_RETURN_OK;  // Already null
    }
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        // Only unregister the handle, NOT the data (it's a view)
        reg.unregister_ptr(*matrix);
        *matrix = nullptr;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Property Queries
// =============================================================================

SCL_EXPORT scl_error_t scl_dense_rows(
    scl_dense_t matrix,
    scl_index_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = static_cast<scl_index_t>(matrix->rows);
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_dense_cols(
    scl_dense_t matrix,
    scl_index_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = static_cast<scl_index_t>(matrix->cols);
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_dense_stride(
    scl_dense_t matrix,
    scl_index_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = static_cast<scl_index_t>(matrix->stride);
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_dense_size(
    scl_dense_t matrix,
    scl_size_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = static_cast<scl_size_t>(matrix->rows) * 
           static_cast<scl_size_t>(matrix->cols);
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_dense_is_valid(
    scl_dense_t matrix,
    scl_bool_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = matrix->valid() ? SCL_TRUE : SCL_FALSE;
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_dense_is_contiguous(
    scl_dense_t matrix,
    scl_bool_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = (matrix->stride == matrix->cols) ? SCL_TRUE : SCL_FALSE;
    SCL_C_API_RETURN_OK;
}

// =============================================================================
// Data Access
// =============================================================================

SCL_EXPORT scl_error_t scl_dense_get_data(
    scl_dense_t matrix,
    const scl_real_t** out,
    scl_size_t* size) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output data pointer is null");
    SCL_C_API_CHECK_NULL(size, "Output size pointer is null");
    
    *out = reinterpret_cast<const scl_real_t*>(matrix->data);
    *size = static_cast<scl_size_t>(matrix->rows) * 
            static_cast<scl_size_t>(matrix->stride);
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_dense_get(
    scl_dense_t matrix,
    const scl_index_t row,
    const scl_index_t col,
    scl_real_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK(row >= 0 && row < matrix->rows, 
                   SCL_ERROR_INDEX_OUT_OF_BOUNDS,
                   "Row index out of bounds");
    SCL_C_API_CHECK(col >= 0 && col < matrix->cols, 
                   SCL_ERROR_INDEX_OUT_OF_BOUNDS,
                   "Column index out of bounds");
    
    // PERFORMANCE: Direct pointer arithmetic for element access
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    *out = static_cast<scl_real_t>(matrix->data[row * matrix->stride + col]);
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_dense_set(
    scl_dense_t matrix,
    const scl_index_t row,
    const scl_index_t col,
    const scl_real_t value) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK(row >= 0 && row < matrix->rows, 
                   SCL_ERROR_INDEX_OUT_OF_BOUNDS,
                   "Row index out of bounds");
    SCL_C_API_CHECK(col >= 0 && col < matrix->cols, 
                   SCL_ERROR_INDEX_OUT_OF_BOUNDS,
                   "Column index out of bounds");
    
    // PERFORMANCE: Direct pointer arithmetic for element access
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    matrix->data[row * matrix->stride + col] = static_cast<Real>(value);
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_dense_export(
    scl_dense_t matrix,
    scl_real_t* data) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(data, "Output data pointer is null");
    
    SCL_C_API_TRY
        // Copy data row by row (handles non-contiguous source)
        for (Index i = 0; i < matrix->rows; ++i) {
            // PERFORMANCE: Direct pointer arithmetic for row access
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            const Real* src_row = matrix->data + i * matrix->stride;
            // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            scl_real_t* dst_row = data + i * matrix->cols;
            
            std::memcpy(dst_row, src_row, matrix->cols * sizeof(Real));
        }
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// In-Place Operations
// =============================================================================

SCL_EXPORT scl_error_t scl_dense_fill(
    scl_dense_t matrix,
    const scl_real_t value) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    
    SCL_C_API_TRY
        const auto fill_val = static_cast<Real>(value);
        
        if (matrix->stride == matrix->cols) [[likely]] {
            // Contiguous: fill entire buffer at once
            const auto total_size = static_cast<Size>(matrix->rows) * 
                                   static_cast<Size>(matrix->cols);
            std::fill_n(matrix->data, total_size, fill_val);
        } else [[unlikely]] {
            // Non-contiguous: fill row by row
            for (Index i = 0; i < matrix->rows; ++i) {
                // PERFORMANCE: Direct pointer arithmetic for row access
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                Real* row = matrix->data + i * matrix->stride;
                std::fill_n(row, matrix->cols, fill_val);
            }
        }
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
