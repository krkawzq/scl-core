// =============================================================================
// FILE: scl/binding/c_api/core/sparse.cpp
// BRIEF: Sparse matrix C API implementation
// =============================================================================

#include "scl/binding/c_api/core/sparse.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/registry.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/sparse.hpp"

#include <span>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace scl;
using namespace scl::binding;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

/// @brief Convert C block strategy enum to C++ BlockStrategy
[[nodiscard]] constexpr auto convert_block_strategy(
    scl_block_strategy_t strategy) noexcept -> BlockStrategy {
    switch (strategy) {
        case SCL_BLOCK_STRATEGY_CONTIGUOUS:
            return BlockStrategy::contiguous();
        case SCL_BLOCK_STRATEGY_SMALL:
            return BlockStrategy::small_blocks();
        case SCL_BLOCK_STRATEGY_LARGE:
            return BlockStrategy::large_blocks();
        case SCL_BLOCK_STRATEGY_ADAPTIVE:
        default:
            return BlockStrategy::adaptive();
    }
}

} // anonymous namespace

extern "C" {

// =============================================================================
// Lifecycle Management
// =============================================================================

SCL_EXPORT scl_error_t scl_sparse_create(
    scl_sparse_t* out,
    const scl_index_t rows,
    const scl_index_t cols,
    const scl_index_t nnz,
    const scl_index_t* indptr,
    const scl_index_t* indices,
    const scl_real_t* data,
    const scl_bool_t is_csr) {
    
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK_NULL(indptr, "Offset array pointer is null");
    SCL_C_API_CHECK_NULL(indices, "Index array pointer is null");
    SCL_C_API_CHECK_NULL(data, "Data array pointer is null");
    SCL_C_API_CHECK(rows > 0 && cols > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Matrix dimensions must be positive");
    SCL_C_API_CHECK(nnz >= 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of non-zeros must be non-negative");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        // Allocate handle (correct type ensures no unsafe downcasting needed)
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = (is_csr != SCL_FALSE);
        
        // Create spans for input arrays
        const Index pdim = (is_csr != SCL_FALSE) ? rows : cols;
        std::span<const Index> offsets_span(
            reinterpret_cast<const Index*>(indptr),
            static_cast<std::size_t>(pdim + 1)
        );
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(indices),
            static_cast<std::size_t>(nnz)
        );
        std::span<const Real> data_span(
            reinterpret_cast<const Real*>(data),
            static_cast<std::size_t>(nnz)
        );
        
        // Create sparse matrix from traditional format
        if (is_csr != SCL_FALSE) {
            handle->matrix = CSR::from_traditional(
                rows, cols, data_span, indices_span, offsets_span
            );
        } else {
            handle->matrix = CSC::from_traditional(
                rows, cols, data_span, indices_span, offsets_span
            );
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to create sparse matrix");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_create_with_strategy(
    scl_sparse_t* out,
    const scl_index_t rows,
    const scl_index_t cols,
    const scl_index_t nnz,
    const scl_index_t* indptr,
    const scl_index_t* indices,
    const scl_real_t* data,
    const scl_bool_t is_csr,
    const scl_block_strategy_t strategy) {
    
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK_NULL(indptr, "Offset array pointer is null");
    SCL_C_API_CHECK_NULL(indices, "Index array pointer is null");
    SCL_C_API_CHECK_NULL(data, "Data array pointer is null");
    SCL_C_API_CHECK(rows > 0 && cols > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Matrix dimensions must be positive");
    SCL_C_API_CHECK(nnz >= 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of non-zeros must be non-negative");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        // Allocate handle (correct type ensures no unsafe downcasting needed)
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = (is_csr != SCL_FALSE);
        
        // Create spans
        const Index pdim = (is_csr != SCL_FALSE) ? rows : cols;
        std::span<const Index> offsets_span(
            reinterpret_cast<const Index*>(indptr),
            static_cast<std::size_t>(pdim + 1)
        );
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(indices),
            static_cast<std::size_t>(nnz)
        );
        std::span<const Real> data_span(
            reinterpret_cast<const Real*>(data),
            static_cast<std::size_t>(nnz)
        );
        
        const BlockStrategy block_strat = convert_block_strategy(strategy);
        
        // Create sparse matrix with strategy
        if (is_csr != SCL_FALSE) {
            handle->matrix = CSR::from_traditional(
                rows, cols, data_span, indices_span, offsets_span, block_strat
            );
        } else {
            handle->matrix = CSC::from_traditional(
                rows, cols, data_span, indices_span, offsets_span, block_strat
            );
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to create sparse matrix");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_wrap(
    scl_sparse_t* out,
    const scl_index_t rows,
    const scl_index_t cols,
    const scl_index_t nnz,
    scl_index_t* indptr,
    scl_index_t* indices,
    scl_real_t* data,
    const scl_bool_t is_csr) {
    
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK_NULL(indptr, "Offset array pointer is null");
    SCL_C_API_CHECK_NULL(indices, "Index array pointer is null");
    SCL_C_API_CHECK_NULL(data, "Data array pointer is null");
    SCL_C_API_CHECK(rows > 0 && cols > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Matrix dimensions must be positive");
    SCL_C_API_CHECK(nnz >= 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of non-zeros must be non-negative");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        // Allocate handle (correct type ensures no unsafe downcasting needed)
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = (is_csr != SCL_FALSE);
        
        // Create span for offsets
        const Index pdim = (is_csr != SCL_FALSE) ? rows : cols;
        std::span<const Index> offsets_span(
            reinterpret_cast<Index*>(indptr),
            static_cast<std::size_t>(pdim + 1)
        );
        
        // Wrap external data (zero-copy, caller manages lifetime)
        if (is_csr != SCL_FALSE) {
            handle->matrix = CSR::wrap_traditional(
                rows, cols,
                reinterpret_cast<Real*>(data),
                reinterpret_cast<Index*>(indices),
                offsets_span
            );
        } else {
            handle->matrix = CSC::wrap_traditional(
                rows, cols,
                reinterpret_cast<Real*>(data),
                reinterpret_cast<Index*>(indices),
                offsets_span
            );
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to wrap sparse matrix");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_wrap_and_own(
    scl_sparse_t* out,
    const scl_index_t rows,
    const scl_index_t cols,
    const scl_index_t nnz,
    scl_index_t* indptr,
    scl_index_t* indices,
    scl_real_t* data,
    const scl_bool_t is_csr) {
    
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK_NULL(indptr, "Offset array pointer is null");
    SCL_C_API_CHECK_NULL(indices, "Index array pointer is null");
    SCL_C_API_CHECK_NULL(data, "Data array pointer is null");
    SCL_C_API_CHECK(rows > 0 && cols > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Matrix dimensions must be positive");
    SCL_C_API_CHECK(nnz >= 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of non-zeros must be non-negative");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        const Index pdim = (is_csr != SCL_FALSE) ? rows : cols;
        
        // Register buffers with registry (data and indices only)
        // indptr is used only to calculate offsets, not owned by the matrix
        const BufferID data_buf = reg.create_buffer(
            data,
            static_cast<std::size_t>(nnz) * sizeof(Real),
            AllocType::ArrayNew
        );
        const BufferID indices_buf = reg.create_buffer(
            indices,
            static_cast<std::size_t>(nnz) * sizeof(Index),
            AllocType::ArrayNew
        );
        
        // Note: indptr is NOT registered - wrap_traditional() only reads it
        // to set up internal pointers. The caller retains ownership of indptr.
        // If caller wants scl to manage indptr, they should free it after this call.
        
        SCL_CHECK_ARG(data_buf && indices_buf,
                     "Failed to register buffers with registry");
        // Cast pointers once outside the loop for performance
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto* data_real = reinterpret_cast<Real*>(data);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto* indices_idx = reinterpret_cast<Index*>(indices);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto* indptr_idx = reinterpret_cast<Index*>(indptr);
        
        // Create aliases for row/column data slices
        for (Index i = 0; i < pdim; ++i) {
            const Index start = indptr_idx[i];
            const Index len = indptr_idx[i + 1] - start;
            
            if (len > 0) [[likely]] {
                // PERFORMANCE: Direct pointer arithmetic for slice pointers
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                Real* data_ptr = data_real + start;
                // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
                Index* idx_ptr = indices_idx + start;
                
                // create_alias 第三参数是 initial_ref（默认为1），不是 offset
                reg.create_alias(data_ptr, data_buf);
                reg.create_alias(idx_ptr, indices_buf);
            }
        }
        
        // Allocate handle (correct type ensures no unsafe downcasting needed)
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        handle->is_csr = (is_csr != SCL_FALSE);
        
        std::span<const Index> offsets_span(
            indptr_idx,
            static_cast<std::size_t>(pdim + 1)
        );
        
        // Wrap with ownership transferred
        if (is_csr != SCL_FALSE) {
            handle->matrix = CSR::wrap_traditional(
                rows, cols,
                reinterpret_cast<Real*>(data),
                reinterpret_cast<Index*>(indices),
                offsets_span
            );
        } else {
            handle->matrix = CSC::wrap_traditional(
                rows, cols,
                reinterpret_cast<Real*>(data),
                reinterpret_cast<Index*>(indices),
                offsets_span
            );
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to create sparse matrix");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_clone(
    scl_sparse_t src,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(src, "Source matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        // Allocate handle (correct type ensures no unsafe downcasting needed)
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = src->is_csr;
        
        // Clone sparse matrix
        if (src->is_csr) {
            handle->matrix = src->as_csr().clone();
        } else {
            handle->matrix = src->as_csc().clone();
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to clone sparse matrix");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_destroy(scl_sparse_t* matrix) {
    if (matrix == nullptr || *matrix == nullptr) {
        SCL_C_API_RETURN_OK;  // Already null
    }
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        reg.unregister_ptr(*matrix);
        *matrix = nullptr;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Property Queries
// =============================================================================

SCL_EXPORT scl_error_t scl_sparse_rows(
    scl_sparse_t matrix,
    scl_index_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = static_cast<scl_index_t>(matrix->rows());
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_sparse_cols(
    scl_sparse_t matrix,
    scl_index_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = static_cast<scl_index_t>(matrix->cols());
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_sparse_nnz(
    scl_sparse_t matrix,
    scl_index_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = static_cast<scl_index_t>(matrix->nnz());
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_sparse_is_csr(
    scl_sparse_t matrix,
    scl_bool_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = matrix->is_csr ? SCL_TRUE : SCL_FALSE;
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_sparse_is_csc(
    scl_sparse_t matrix,
    scl_bool_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = matrix->is_csr ? SCL_FALSE : SCL_TRUE;
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_sparse_is_valid(
    scl_sparse_t matrix,
    scl_bool_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    *out = matrix->valid() ? SCL_TRUE : SCL_FALSE;
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_sparse_is_contiguous(
    scl_sparse_t matrix,
    scl_bool_t* out) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    SCL_C_API_TRY
        *out = matrix->visit([](const auto& m) {
            return m.is_contiguous() ? SCL_TRUE : SCL_FALSE;
        });
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_layout_info(
    scl_sparse_t matrix,
    scl_sparse_layout_info_t* info) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(info, "Info pointer is null");
    
    SCL_C_API_TRY
        matrix->visit([&info](const auto& m) {
            const auto layout = m.layout_info();
            
            info->data_block_count = layout.data_block_count;
            info->index_block_count = layout.index_block_count;
            info->data_bytes = layout.data_bytes;
            info->index_bytes = layout.index_bytes;
            info->metadata_bytes = layout.metadata_bytes;
            info->is_contiguous = layout.is_contiguous ? SCL_TRUE : SCL_FALSE;
            info->is_traditional_format = 
                layout.is_traditional_format ? SCL_TRUE : SCL_FALSE;
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Data Export
// =============================================================================

SCL_EXPORT scl_error_t scl_sparse_export(
    scl_sparse_t matrix,
    scl_index_t* indptr,
    scl_index_t* indices,
    scl_real_t* data) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(indptr, "Offset array pointer is null");
    SCL_C_API_CHECK_NULL(indices, "Index array pointer is null");
    SCL_C_API_CHECK_NULL(data, "Data array pointer is null");
    
    SCL_C_API_TRY
        const Index pdim = matrix->is_csr ? matrix->rows() : matrix->cols();
        const Index nnz_val = matrix->nnz();
        
        matrix->visit([&](auto& m) {
            const auto trad = m.to_traditional();
            
            // Copy offset array
            for (Index i = 0; i <= pdim; ++i) {
                indptr[i] = static_cast<scl_index_t>(trad.offsets[i]);
            }
            
            // Copy indices and values
            for (Index i = 0; i < nnz_val; ++i) {
                indices[i] = static_cast<scl_index_t>(trad.indices[i]);
                data[i] = static_cast<scl_real_t>(trad.values[i]);
            }
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_get_lengths(
    scl_sparse_t matrix,
    const scl_index_t** lengths,
    scl_size_t* lengths_size) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(lengths, "Output pointer is null");
    SCL_C_API_CHECK_NULL(lengths_size, "Size pointer is null");
    
    // Direct access to internal lengths array (no copy)
    matrix->visit([&](const auto& m) {
        *lengths = reinterpret_cast<const scl_index_t*>(m.lengths);
        *lengths_size = static_cast<scl_size_t>(m.primary_dim());
    });
    
    SCL_C_API_RETURN_OK;
}

SCL_EXPORT scl_error_t scl_sparse_get_indices(
    scl_sparse_t matrix,
    const scl_index_t** indices,
    scl_size_t* indices_size) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(indices, "Output pointer is null");
    SCL_C_API_CHECK_NULL(indices_size, "Size pointer is null");
    
    SCL_C_API_TRY
        const bool is_cont = matrix->visit([](const auto& m) {
            return m.is_contiguous();
        });
        
        SCL_C_API_CHECK(is_cont, SCL_ERROR_INVALID_ARGUMENT,
                       "Matrix must be contiguous to get internal pointers");
        
        // Use contiguous_indices() for direct internal pointer access
        matrix->visit([&](const auto& m) {
            *indices = reinterpret_cast<const scl_index_t*>(m.contiguous_indices());
            *indices_size = static_cast<scl_size_t>(m.nnz());
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_get_data(
    scl_sparse_t matrix,
    const scl_real_t** data,
    scl_size_t* data_size) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(data, "Output pointer is null");
    SCL_C_API_CHECK_NULL(data_size, "Size pointer is null");
    
    SCL_C_API_TRY
        const bool is_cont = matrix->visit([](const auto& m) {
            return m.is_contiguous();
        });
        
        SCL_C_API_CHECK(is_cont, SCL_ERROR_INVALID_ARGUMENT,
                       "Matrix must be contiguous to get internal pointers");
        
        // Use contiguous_data() for direct internal pointer access
        matrix->visit([&](const auto& m) {
            *data = reinterpret_cast<const scl_real_t*>(m.contiguous_data());
            *data_size = static_cast<scl_size_t>(m.nnz());
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Format Conversion
// =============================================================================

SCL_EXPORT scl_error_t scl_sparse_transpose(
    scl_sparse_t src,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(src, "Source matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = !src->is_csr;  // Swap format
        
        if (src->is_csr) {
            handle->matrix = src->as_csr().transpose();
        } else {
            handle->matrix = src->as_csc().transpose();
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to transpose sparse matrix");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_to_contiguous(
    scl_sparse_t src,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(src, "Source matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = src->is_csr;
        
        if (src->is_csr) {
            handle->matrix = src->as_csr().to_contiguous();
        } else {
            handle->matrix = src->as_csc().to_contiguous();
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to convert to contiguous");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// COO Format
// =============================================================================

SCL_EXPORT scl_error_t scl_sparse_from_coo(
    scl_sparse_t* out,
    const scl_index_t rows,
    const scl_index_t cols,
    const scl_index_t nnz,
    const scl_index_t* row_indices,
    const scl_index_t* col_indices,
    const scl_real_t* values,
    const scl_bool_t is_csr,
    const scl_block_strategy_t strategy) {
    
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK_NULL(row_indices, "Row indices pointer is null");
    SCL_C_API_CHECK_NULL(col_indices, "Column indices pointer is null");
    SCL_C_API_CHECK_NULL(values, "Values pointer is null");
    SCL_C_API_CHECK(rows > 0 && cols > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Matrix dimensions must be positive");
    SCL_C_API_CHECK(nnz >= 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of non-zeros must be non-negative");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = (is_csr != SCL_FALSE);
        
        std::span<const Index> row_span(
            reinterpret_cast<const Index*>(row_indices),
            static_cast<std::size_t>(nnz)
        );
        std::span<const Index> col_span(
            reinterpret_cast<const Index*>(col_indices),
            static_cast<std::size_t>(nnz)
        );
        std::span<const Real> val_span(
            reinterpret_cast<const Real*>(values),
            static_cast<std::size_t>(nnz)
        );
        
        const BlockStrategy block_strat = convert_block_strategy(strategy);
        
        if (is_csr != SCL_FALSE) {
            handle->matrix = CSR::from_coo(
                rows, cols, row_span, col_span, val_span, block_strat
            );
        } else {
            handle->matrix = CSC::from_coo(
                rows, cols, row_span, col_span, val_span, block_strat
            );
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to create from COO");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_to_coo(
    scl_sparse_t matrix,
    scl_index_t** row_indices,
    scl_index_t** col_indices,
    scl_real_t** values,
    scl_index_t* nnz) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(row_indices, "Row indices pointer is null");
    SCL_C_API_CHECK_NULL(col_indices, "Column indices pointer is null");
    SCL_C_API_CHECK_NULL(values, "Values pointer is null");
    SCL_C_API_CHECK_NULL(nnz, "NNZ pointer is null");
    
    SCL_C_API_TRY
        matrix->visit([&](const auto& m) {
            const auto coo = scl::kernel::sparse::to_coo_arrays(m);
            
            *row_indices = reinterpret_cast<scl_index_t*>(coo.row_indices);
            *col_indices = reinterpret_cast<scl_index_t*>(coo.col_indices);
            *values = reinterpret_cast<scl_real_t*>(coo.values);
            *nnz = coo.nnz;
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Slicing Operations
// =============================================================================

SCL_EXPORT scl_error_t scl_sparse_row_range_view(
    scl_sparse_t src,
    const scl_index_t start,
    const scl_index_t end,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(src, "Source matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK(src->is_csr, SCL_ERROR_TYPE_MISMATCH,
                   "Row range view only supported for CSR matrices");
    SCL_C_API_CHECK(start >= 0 && start < end && end <= src->rows(),
                   SCL_ERROR_INDEX_OUT_OF_BOUNDS,
                   "Invalid row range");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = true;
        handle->matrix = src->as_csr().row_range_view(start, end);
        
        SCL_CHECK_ARG(handle->valid(), "Failed to create row range view");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_row_range_copy(
    scl_sparse_t src,
    const scl_index_t start,
    const scl_index_t end,
    const scl_block_strategy_t strategy,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(src, "Source matrix is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK(src->is_csr, SCL_ERROR_TYPE_MISMATCH,
                   "Row range copy only supported for CSR matrices");
    SCL_C_API_CHECK(start >= 0 && start < end && end <= src->rows(),
                   SCL_ERROR_INDEX_OUT_OF_BOUNDS,
                   "Invalid row range");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        const BlockStrategy block_strat = convert_block_strategy(strategy);
        
        handle->is_csr = true;
        handle->matrix = src->as_csr().row_range_copy(start, end, block_strat);
        
        SCL_CHECK_ARG(handle->valid(), "Failed to create row range copy");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_slice_rows(
    scl_sparse_t src,
    const scl_index_t* row_indices,
    const scl_size_t n_rows,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(src, "Source matrix is null");
    SCL_C_API_CHECK_NULL(row_indices, "Row indices pointer is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK(src->is_csr, SCL_ERROR_TYPE_MISMATCH,
                   "Row slicing only supported for CSR matrices");
    SCL_C_API_CHECK(n_rows > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of rows must be positive");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(row_indices),
            n_rows
        );
        
        handle->is_csr = true;
        handle->matrix = src->as_csr().row_slice_view(indices_span);
        
        SCL_CHECK_ARG(handle->valid(), "Failed to create row slice");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_row_slice_copy(
    scl_sparse_t src,
    const scl_index_t* row_indices,
    const scl_size_t n_rows,
    const scl_block_strategy_t strategy,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(src, "Source matrix is null");
    SCL_C_API_CHECK_NULL(row_indices, "Row indices pointer is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK(src->is_csr, SCL_ERROR_TYPE_MISMATCH,
                   "Row slicing only supported for CSR matrices");
    SCL_C_API_CHECK(n_rows > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of rows must be positive");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        const BlockStrategy block_strat = convert_block_strategy(strategy);
        
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(row_indices),
            n_rows
        );
        
        handle->is_csr = true;
        handle->matrix = src->as_csr().row_slice_copy(indices_span, block_strat);
        
        SCL_CHECK_ARG(handle->valid(), "Failed to create row slice copy");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_col_slice(
    scl_sparse_t src,
    const scl_index_t* col_indices,
    const scl_size_t n_cols,
    const scl_block_strategy_t strategy,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(src, "Source matrix is null");
    SCL_C_API_CHECK_NULL(col_indices, "Column indices pointer is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK(n_cols > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of columns must be positive");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        const BlockStrategy block_strat = convert_block_strategy(strategy);
        
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(col_indices),
            n_cols
        );
        
        handle->is_csr = src->is_csr;
        
        if (src->is_csr) {
            handle->matrix = src->as_csr().col_slice(indices_span, block_strat);
        } else {
            handle->matrix = src->as_csc().col_slice_copy(indices_span, block_strat);
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to create column slice");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_slice_cols(
    scl_sparse_t src,
    const scl_index_t* col_indices,
    const scl_size_t n_cols,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(src, "Source matrix is null");
    SCL_C_API_CHECK_NULL(col_indices, "Column indices pointer is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK(!src->is_csr, SCL_ERROR_TYPE_MISMATCH,
                   "Column slicing (view) only supported for CSC matrices");
    SCL_C_API_CHECK(n_cols > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of columns must be positive");
    
    SCL_C_API_TRY
        auto& reg = get_registry();
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        std::span<const Index> indices_span(
            reinterpret_cast<const Index*>(col_indices),
            n_cols
        );
        
        handle->is_csr = false;
        handle->matrix = src->as_csc().col_slice_view(indices_span);
        
        SCL_CHECK_ARG(handle->valid(), "Failed to create column slice");
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Batch Operations
// =============================================================================

SCL_EXPORT scl_error_t scl_sparse_vstack(
    const scl_sparse_t* matrices,
    const scl_size_t n_matrices,
    const scl_block_strategy_t strategy,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(matrices, "Matrices array is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK(n_matrices > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of matrices must be positive");
    
    SCL_C_API_TRY
        // Check all matrices are CSR
        for (std::size_t i = 0; i < n_matrices; ++i) {
            SCL_CHECK_NULL(matrices[i], "Matrix in array is null");
            SCL_CHECK_ARG(matrices[i]->is_csr,
                         "All matrices must be CSR for vstack");
        }
        
        auto& reg = get_registry();
        
        // Collect CSR matrix pointers (CSR is non-copyable)
        std::vector<const CSR*> csr_ptrs;
        csr_ptrs.reserve(n_matrices);
        for (std::size_t i = 0; i < n_matrices; ++i) {
            csr_ptrs.push_back(&matrices[i]->as_csr());
        }
        
        const BlockStrategy block_strat = convert_block_strategy(strategy);
        
        // First pass: compute dimensions
        Index total_rows = 0;
        Index cols = csr_ptrs[0]->cols();
        
        for (const auto* mat : csr_ptrs) {
            total_rows += mat->rows();
        }
        
        // Gather nnz counts per row
        std::vector<Index> nnzs;
        nnzs.reserve(static_cast<std::size_t>(total_rows));
        for (const auto* mat : csr_ptrs) {
            for (Index i = 0; i < mat->rows(); ++i) {
                nnzs.push_back(mat->lengths[i]);
            }
        }
        
        // Create result
        CSR result = CSR::create(total_rows, cols, nnzs, block_strat);
        
        // Copy data
        Index dst_row = 0;
        for (const auto* mat : csr_ptrs) {
            for (Index i = 0; i < mat->rows(); ++i) {
                if (mat->lengths[i] > 0) {
                    std::memcpy(result.data_ptrs[dst_row], mat->data_ptrs[i],
                               static_cast<std::size_t>(mat->lengths[i]) * sizeof(Real));
                    std::memcpy(result.indices_ptrs[dst_row], mat->indices_ptrs[i],
                               static_cast<std::size_t>(mat->lengths[i]) * sizeof(Index));
                }
                ++dst_row;
            }
        }
        
        SCL_CHECK_ARG(result.valid(), "Failed to vstack matrices");
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = true;
        handle->matrix = std::move(result);
        
        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_hstack(
    const scl_sparse_t* matrices,
    const scl_size_t n_matrices,
    const scl_block_strategy_t strategy,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(matrices, "Matrices array is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK(n_matrices > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of matrices must be positive");
    
    SCL_C_API_TRY
        // Check all matrices are CSC
        for (std::size_t i = 0; i < n_matrices; ++i) {
            SCL_CHECK_NULL(matrices[i], "Matrix in array is null");
            SCL_CHECK_ARG(!matrices[i]->is_csr,
                         "All matrices must be CSC for hstack");
        }
        
        auto& reg = get_registry();
        
        // Collect CSC matrix pointers (CSC is non-copyable)
        std::vector<const CSC*> csc_ptrs;
        csc_ptrs.reserve(n_matrices);
        for (std::size_t i = 0; i < n_matrices; ++i) {
            csc_ptrs.push_back(&matrices[i]->as_csc());
        }
        
        const BlockStrategy block_strat = convert_block_strategy(strategy);
        
        // Build result by manually concatenating
        // First pass: compute dimensions
        Index total_cols = 0;
        Index rows = csc_ptrs[0]->rows();
        
        for (const auto* mat : csc_ptrs) {
            total_cols += mat->cols();
        }
        
        // Gather nnz counts per column
        std::vector<Index> nnzs;
        nnzs.reserve(static_cast<std::size_t>(total_cols));
        for (const auto* mat : csc_ptrs) {
            for (Index j = 0; j < mat->cols(); ++j) {
                nnzs.push_back(mat->lengths[j]);
            }
        }
        
        // Create result
        CSC result = CSC::create(rows, total_cols, nnzs, block_strat);
        
        // Copy data
        Index dst_col = 0;
        for (const auto* mat : csc_ptrs) {
            for (Index j = 0; j < mat->cols(); ++j) {
                if (mat->lengths[j] > 0) {
                    std::memcpy(result.data_ptrs[dst_col], mat->data_ptrs[j],
                               static_cast<std::size_t>(mat->lengths[j]) * sizeof(Real));
                    std::memcpy(result.indices_ptrs[dst_col], mat->indices_ptrs[j],
                               static_cast<std::size_t>(mat->lengths[j]) * sizeof(Index));
                }
                ++dst_col;
            }
        }
        
        SCL_CHECK_ARG(result.valid(), "Failed to hstack matrices");
        
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate sparse matrix handle");
        
        handle->is_csr = false;
        handle->matrix = std::move(result);

        *out = handle;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
