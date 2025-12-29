// =============================================================================
// FILE: scl/binding/c_api/core/unsafe.cpp
// BRIEF: Unsafe C API implementation (expert-only, ABI-unstable)
// =============================================================================

#include "scl/binding/c_api/core/unsafe.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/registry.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Sparse Matrix Unsafe Operations
// =============================================================================

SCL_EXPORT scl_error_t scl_sparse_unsafe_get_raw(
    scl_sparse_t matrix,
    scl_sparse_raw_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    SCL_C_API_TRY
        out->is_csr = matrix->is_csr ? SCL_TRUE : SCL_FALSE;
        
        // Use visitor pattern to extract internal pointers
        matrix->visit([&out](auto& m) {
            // WARNING: Direct pointer exposure - bypasses safety checks
            // These pointers are valid only while the matrix is alive
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            out->data_ptrs = reinterpret_cast<void**>(m.data_ptrs);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            out->indices_ptrs = reinterpret_cast<void**>(m.indices_ptrs);
            out->lengths = m.lengths;
            out->rows = m.rows_;
            out->cols = m.cols_;
            out->nnz = m.nnz_;
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_unsafe_from_raw(
    const scl_sparse_raw_t* raw,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(raw, "Raw structure is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        // Allocate handle (correct type ensures no unsafe downcasting needed)
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = (raw->is_csr != SCL_FALSE);
        
        // WARNING: DANGEROUS - Creates sparse matrix from raw pointers
        // Caller MUST ensure pointers are valid and properly registered
        // No validation is performed - incorrect usage = undefined behavior
        if (handle->is_csr) {
            CSR matrix;
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            matrix.data_ptrs = reinterpret_cast<Pointer*>(raw->data_ptrs);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            matrix.indices_ptrs = reinterpret_cast<Pointer*>(raw->indices_ptrs);
            matrix.lengths = raw->lengths;
            matrix.rows_ = raw->rows;
            matrix.cols_ = raw->cols;
            matrix.nnz_ = raw->nnz;
            handle->matrix = std::move(matrix);
        } else {
            CSC matrix;
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            matrix.data_ptrs = reinterpret_cast<Pointer*>(raw->data_ptrs);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
            matrix.indices_ptrs = reinterpret_cast<Pointer*>(raw->indices_ptrs);
            matrix.lengths = raw->lengths;
            matrix.rows_ = raw->rows;
            matrix.cols_ = raw->cols;
            matrix.nnz_ = raw->nnz;
            handle->matrix = std::move(matrix);
        }
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Dense Matrix Unsafe Operations
// =============================================================================

SCL_EXPORT scl_error_t scl_dense_unsafe_get_raw(
    scl_dense_t matrix,
    scl_dense_raw_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    // Direct struct copy (safe - just exposes internal state)
    // DenseView is always a view - never owns data
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    out->data = reinterpret_cast<scl_real_t*>(matrix->data);
    out->rows = static_cast<scl_index_t>(matrix->rows);
    out->cols = static_cast<scl_index_t>(matrix->cols);
    out->stride = static_cast<scl_index_t>(matrix->stride);
    
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_dense_unsafe_from_raw(
    const scl_dense_raw_t* raw,
    scl_dense_t* out) {
    
    SCL_C_API_CHECK_NULL(raw, "Raw structure is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        // Allocate handle (correct type ensures no unsafe downcasting needed)
        auto* handle = reg.new_object<scl_dense_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate dense matrix handle");
        
        // DenseView is always a view - caller manages data lifetime
        // WARNING: Direct pointer assignment - no validation
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        handle->data = reinterpret_cast<Real*>(raw->data);
        handle->rows = raw->rows;
        handle->cols = raw->cols;
        handle->stride = raw->stride;
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Direct Memory Access (Zero-Overhead, Maximum Risk)
// =============================================================================

SCL_EXPORT scl_error_t scl_sparse_unsafe_get_row(
    scl_sparse_t matrix,
    const scl_index_t row,
    scl_real_t** data,
    scl_index_t** indices,
    scl_index_t* length) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(data, "Data pointer is null");
    SCL_C_API_CHECK_NULL(indices, "Indices pointer is null");
    SCL_C_API_CHECK_NULL(length, "Length pointer is null");
    SCL_C_API_CHECK(matrix->is_csr, SCL_ERROR_TYPE_MISMATCH,
                   "Matrix must be CSR format");
    
    // Minimal bounds checking (only in debug builds via SCL_ASSERT)
    SCL_ASSERT(row >= 0 && row < matrix->rows(), "Row index out of bounds");
    
    SCL_C_API_TRY
        const auto& csr = matrix->as_csr();
        
        // PERFORMANCE: Zero-overhead direct pointer access
        // NO bounds checking in release builds for maximum speed
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        *data = reinterpret_cast<scl_real_t*>(
            static_cast<Real*>(csr.data_ptrs[row])
        );
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        *indices = reinterpret_cast<scl_index_t*>(
            static_cast<Index*>(csr.indices_ptrs[row])
        );
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        *length = csr.lengths[row];
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_unsafe_get_col(
    scl_sparse_t matrix,
    const scl_index_t col,
    scl_real_t** data,
    scl_index_t** indices,
    scl_index_t* length) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(data, "Data pointer is null");
    SCL_C_API_CHECK_NULL(indices, "Indices pointer is null");
    SCL_C_API_CHECK_NULL(length, "Length pointer is null");
    SCL_C_API_CHECK(!matrix->is_csr, SCL_ERROR_TYPE_MISMATCH,
                   "Matrix must be CSC format");
    
    SCL_ASSERT(col >= 0 && col < matrix->cols(), "Column index out of bounds");
    
    SCL_C_API_TRY
        const auto& csc = matrix->as_csc();
        
        // Zero-overhead direct access (CSC columns = CSR rows)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        *data = reinterpret_cast<scl_real_t*>(
            static_cast<Real*>(csc.data_ptrs[col])
        );
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        *indices = reinterpret_cast<scl_index_t*>(
            static_cast<Index*>(csc.indices_ptrs[col])
        );
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        *length = csc.lengths[col];
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_dense_unsafe_get_row(
    scl_dense_t matrix,
    const scl_index_t row,
    scl_real_t** data) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(data, "Data pointer is null");
    
    SCL_ASSERT(row >= 0 && row < matrix->rows, "Row index out of bounds");
    
    // Zero-overhead: just pointer arithmetic
    // Use stride for element access: data[j] for element at (row, j)
    // PERFORMANCE: Direct pointer arithmetic
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    *data = reinterpret_cast<scl_real_t*>(matrix->data + row * matrix->stride);
    
    SCL_C_API_RETURN_OK;
}

// =============================================================================
// Registry Integration
// =============================================================================

SCL_EXPORT scl_size_t scl_unsafe_register_buffer(
    void* ptr,
    const scl_size_t size) {
    
    if (ptr == nullptr || size == 0) {
        return 0;
    }
    
    try {
        auto& reg = get_registry();
        
        // Register as ArrayNew (assumes new[] allocation)
        const BufferID buf_id = reg.create_buffer(
            ptr,
            size,
            AllocType::ArrayNew
        );
        
        return static_cast<scl_size_t>(buf_id);
    } catch (...) {
        return 0;
    }
}

SCL_EXPORT scl_error_t scl_unsafe_create_alias(
    void* ptr,
    const scl_size_t buffer_id) {
    
    SCL_C_API_CHECK_NULL(ptr, "Pointer is null");
    SCL_C_API_CHECK(buffer_id != 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Invalid buffer ID");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        const auto buf_id = static_cast<BufferID>(buffer_id);
        // create_alias: ptr 地址本身包含偏移信息，无需额外 offset 参数
        // 第三参数是 initial_ref（默认为1）
        reg.create_alias(ptr, buf_id);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_unsafe_unregister(void* ptr) {
    SCL_C_API_CHECK_NULL(ptr, "Pointer is null");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        reg.unregister_ptr(ptr);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
