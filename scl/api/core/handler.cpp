/// @file scl/api/core/handler.cpp
/// @brief Unified C-API handle implementation for SCL
///
/// This file provides:
///   - Internal structure definitions for opaque handles
///   - Dynamic dispatch system for type-erased operations
///   - All handle lifecycle and operation implementations
///
/// ## Architecture
///
/// Handles wrap C++ template instances using std::variant for type erasure.
/// Runtime dispatch is performed based on stored type tags (real_type, index_type, layout).
///
/// ## Memory Management
///
/// - All handles are allocated with `new` and freed with `delete`
/// - Internal data uses SharedSpan for reference-counted sharing
/// - Slicing operations share data via SharedSpan's reference counting

#include "dispatch.h"
#include "sparse.h"
#include "type.h"

#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/sparse/slice.hpp"

#include <cstring>
#include <variant>

// =============================================================================
// C-ABI Export Macros
// =============================================================================

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef SCL_BUILDING_LIBRARY
        #define SCL_API extern "C" __declspec(dllexport)
    #else
        #define SCL_API extern "C" __declspec(dllimport)
    #endif
#else
    #define SCL_API extern "C" __attribute__((visibility("default")))
#endif

// =============================================================================
// Exception Handling Helpers for Pointer-Returning Functions
// =============================================================================

/// @brief Try-catch wrapper for functions returning pointers
#define SCL_TRY_PTR \
    scl::clear_thread_error(); \
    try

#define SCL_CATCH_PTR(null_value) \
    catch (const scl::Error& e) { \
        scl::set_thread_error(e.code(), e.message().c_str()); \
    } catch (const std::bad_alloc&) { \
        scl::set_thread_error(scl::ErrorCode::OutOfMemory); \
    } catch (const std::exception& e) { \
        scl::set_thread_error(scl::ErrorCode::Unknown, e.what()); \
    } catch (...) { \
        scl::set_thread_error(scl::ErrorCode::Unknown); \
    } \
    return null_value

// =============================================================================
// SECTION 1: Internal Structure Definitions
// =============================================================================

namespace {

/// @brief Variant type holding all possible Sparse matrix types
using SparseVariant = std::variant<
    // CSR variants (indices 0-3)
    scl::Sparse<scl::Real32, scl::Index32, true>,   // 0
    scl::Sparse<scl::Real64, scl::Index32, true>,   // 1
    scl::Sparse<scl::Real32, scl::Index64, true>,   // 2
    scl::Sparse<scl::Real64, scl::Index64, true>,   // 3
    // CSC variants (indices 4-7)
    scl::Sparse<scl::Real32, scl::Index32, false>,  // 4
    scl::Sparse<scl::Real64, scl::Index32, false>,  // 5
    scl::Sparse<scl::Real32, scl::Index64, false>,  // 6
    scl::Sparse<scl::Real64, scl::Index64, false>   // 7
>;

/// @brief Get variant index from type information
[[nodiscard]]
constexpr
auto sparse_variant_index(
    scl_real_type_t real_type,
    scl_index_type_t index_type,
    scl_layout_t layout
) noexcept -> std::size_t {
    return static_cast<std::size_t>(real_type) |
           (static_cast<std::size_t>(index_type) << 1) |
           (static_cast<std::size_t>(layout) << 2);
}

}  // namespace

/// @brief Internal sparse handle structure
struct scl_sparse_s {
    scl_real_type_t  real_type;   ///< Value precision
    scl_index_type_t index_type;  ///< Index precision
    scl_layout_t     layout;      ///< CSR or CSC
    SparseVariant    data;        ///< Type-erased matrix storage
    
    /// @brief Construct with type information
    scl_sparse_s(
        scl_real_type_t rt,
        scl_index_type_t it,
        scl_layout_t ly
    ) : real_type(rt), index_type(it), layout(ly) {}
};

// =============================================================================
// SECTION 2: Dispatch Helpers
// =============================================================================

namespace {

/// @brief Apply a visitor to the sparse matrix with correct types
template<typename Visitor>
auto visit_sparse(scl_sparse_t handle, Visitor&& visitor) {
    return std::visit(std::forward<Visitor>(visitor), handle->data);
}

/// @brief Create a new handle with the same type configuration
[[nodiscard]]
auto create_handle_like(scl_sparse_t source) -> scl_sparse_t {
    return new scl_sparse_s(source->real_type, source->index_type, source->layout);
}

/// @brief Create a new handle with transposed layout
[[nodiscard]]
auto create_transposed_handle(scl_sparse_t source) -> scl_sparse_t {
    auto new_layout = (source->layout == SCL_LAYOUT_CSR) ? SCL_LAYOUT_CSC : SCL_LAYOUT_CSR;
    return new scl_sparse_s(source->real_type, source->index_type, new_layout);
}

}  // namespace

// =============================================================================
// SECTION 3: Creation Functions
// =============================================================================

SCL_API
auto scl_sparse_zeros(
    std::int64_t rows,
    std::int64_t cols,
    scl_real_type_t real_type,
    scl_index_type_t index_type,
    scl_layout_t layout
) -> scl_sparse_t {
    scl::clear_thread_error();
    try {
        SCL_CHECK_ARG(rows >= 0, "rows must be non-negative");
        SCL_CHECK_ARG(cols >= 0, "cols must be non-negative");
        
        auto* handle = new scl_sparse_s(real_type, index_type, layout);
        
        SCL_DISPATCH_SPARSE(real_type, index_type, layout, {
            using SparseT = scl::Sparse<SCL_REAL_TYPE, SCL_INDEX_TYPE, SCL_IS_CSR>;
            handle->data = SparseT::zeros(
                static_cast<SCL_INDEX_TYPE>(rows),
                static_cast<SCL_INDEX_TYPE>(cols)
            );
        });
        
        return handle;
    } catch (const scl::Error& e) {
        scl::set_thread_error(e.code(), e.message().c_str());
    } catch (const std::bad_alloc&) {
        scl::set_thread_error(scl::ErrorCode::OutOfMemory);
    } catch (const std::exception& e) {
        scl::set_thread_error(scl::ErrorCode::Unknown, e.what());
    } catch (...) {
        scl::set_thread_error(scl::ErrorCode::Unknown);
    }
    return SCL_NULL_SPARSE;
}

SCL_API
auto scl_sparse_identity(
    std::int64_t n,
    scl_real_type_t real_type,
    scl_index_type_t index_type
) -> scl_sparse_t {
    SCL_C_API_BEGIN
    SCL_CHECK_ARG(n >= 0, "dimension must be non-negative");
    
    auto* handle = new scl_sparse_s(real_type, index_type, SCL_LAYOUT_CSR);
    
    SCL_DISPATCH_REAL_INDEX(real_type, index_type, {
        using SparseT = scl::Sparse<SCL_REAL_TYPE, SCL_INDEX_TYPE, true>;
        handle->data = SparseT::identity(static_cast<SCL_INDEX_TYPE>(n));
    });
    
    return handle;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

SCL_API
auto scl_sparse_from_coo(
    std::int64_t rows,
    std::int64_t cols,
    const void* row_indices,
    const void* col_indices,
    const void* values,
    std::int64_t nnz,
    scl_real_type_t real_type,
    scl_index_type_t index_type,
    scl_layout_t layout
) -> scl_sparse_t {
    SCL_C_API_BEGIN
    SCL_CHECK_ARG(rows >= 0, "rows must be non-negative");
    SCL_CHECK_ARG(cols >= 0, "cols must be non-negative");
    SCL_CHECK_ARG(nnz >= 0, "nnz must be non-negative");
    SCL_CHECK_NOT_NULL(row_indices, "row_indices");
    SCL_CHECK_NOT_NULL(col_indices, "col_indices");
    SCL_CHECK_NOT_NULL(values, "values");
    
    auto* handle = new scl_sparse_s(real_type, index_type, layout);
    
    SCL_DISPATCH_SPARSE(real_type, index_type, layout, {
        using RealT = SCL_REAL_TYPE;
        using IndexT = SCL_INDEX_TYPE;
        using SparseT = scl::Sparse<RealT, IndexT, SCL_IS_CSR>;
        
        auto row_span = std::span<const IndexT>(
            static_cast<const IndexT*>(row_indices),
            static_cast<std::size_t>(nnz)
        );
        auto col_span = std::span<const IndexT>(
            static_cast<const IndexT*>(col_indices),
            static_cast<std::size_t>(nnz)
        );
        auto val_span = std::span<const RealT>(
            static_cast<const RealT*>(values),
            static_cast<std::size_t>(nnz)
        );
        
        handle->data = SparseT::from_coo(
            static_cast<IndexT>(rows),
            static_cast<IndexT>(cols),
            row_span, col_span, val_span
        );
    });
    
    return handle;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

SCL_API
auto scl_sparse_from_csr(
    std::int64_t rows,
    std::int64_t cols,
    const std::int64_t* row_ptrs,
    const void* col_indices,
    const void* values,
    scl_real_type_t real_type,
    scl_index_type_t index_type
) -> scl_sparse_t {
    SCL_C_API_BEGIN
    SCL_CHECK_ARG(rows >= 0, "rows must be non-negative");
    SCL_CHECK_ARG(cols >= 0, "cols must be non-negative");
    SCL_CHECK_NOT_NULL(row_ptrs, "row_ptrs");
    
    auto* handle = new scl_sparse_s(real_type, index_type, SCL_LAYOUT_CSR);
    
    // Build from row pointers
    SCL_DISPATCH_REAL_INDEX(real_type, index_type, {
        using RealT = SCL_REAL_TYPE;
        using IndexT = SCL_INDEX_TYPE;
        using SparseT = scl::Sparse<RealT, IndexT, true>;
        
        // Compute nnz counts per row
        std::vector<IndexT> nnz_counts(static_cast<std::size_t>(rows));
        for (std::int64_t i = 0; i < rows; ++i) {
            nnz_counts[static_cast<std::size_t>(i)] = 
                static_cast<IndexT>(row_ptrs[i + 1] - row_ptrs[i]);
        }
        
        auto matrix = SparseT::create(
            static_cast<IndexT>(rows),
            static_cast<IndexT>(cols),
            std::span<const IndexT>(nnz_counts)
        );
        
        // Copy data
        const auto* src_indices = static_cast<const IndexT*>(col_indices);
        const auto* src_values = static_cast<const RealT*>(values);
        
        for (std::int64_t r = 0; r < rows; ++r) {
            auto row_len = row_ptrs[r + 1] - row_ptrs[r];
            auto offset = row_ptrs[r];
            
            auto dst_indices = matrix.row_indices(static_cast<IndexT>(r));
            auto dst_values = matrix.row_values(static_cast<IndexT>(r));
            
            std::copy(src_indices + offset, src_indices + offset + row_len, dst_indices.data());
            std::copy(src_values + offset, src_values + offset + row_len, dst_values.data());
        }
        
        handle->data = std::move(matrix);
    });
    
    return handle;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

SCL_API
auto scl_sparse_from_csc(
    std::int64_t rows,
    std::int64_t cols,
    const std::int64_t* col_ptrs,
    const void* row_indices,
    const void* values,
    scl_real_type_t real_type,
    scl_index_type_t index_type
) -> scl_sparse_t {
    SCL_C_API_BEGIN
    SCL_CHECK_ARG(rows >= 0, "rows must be non-negative");
    SCL_CHECK_ARG(cols >= 0, "cols must be non-negative");
    SCL_CHECK_NOT_NULL(col_ptrs, "col_ptrs");
    
    auto* handle = new scl_sparse_s(real_type, index_type, SCL_LAYOUT_CSC);
    
    // Build from column pointers
    SCL_DISPATCH_REAL_INDEX(real_type, index_type, {
        using RealT = SCL_REAL_TYPE;
        using IndexT = SCL_INDEX_TYPE;
        using SparseT = scl::Sparse<RealT, IndexT, false>;
        
        // Compute nnz counts per column
        std::vector<IndexT> nnz_counts(static_cast<std::size_t>(cols));
        for (std::int64_t c = 0; c < cols; ++c) {
            nnz_counts[static_cast<std::size_t>(c)] = 
                static_cast<IndexT>(col_ptrs[c + 1] - col_ptrs[c]);
        }
        
        auto matrix = SparseT::create(
            static_cast<IndexT>(rows),
            static_cast<IndexT>(cols),
            std::span<const IndexT>(nnz_counts)
        );
        
        // Copy data
        const auto* src_indices = static_cast<const IndexT*>(row_indices);
        const auto* src_values = static_cast<const RealT*>(values);
        
        for (std::int64_t c = 0; c < cols; ++c) {
            auto col_len = col_ptrs[c + 1] - col_ptrs[c];
            auto offset = col_ptrs[c];
            
            auto dst_indices = matrix.col_indices(static_cast<IndexT>(c));
            auto dst_values = matrix.col_values(static_cast<IndexT>(c));
            
            std::copy(src_indices + offset, src_indices + offset + col_len, dst_indices.data());
            std::copy(src_values + offset, src_values + offset + col_len, dst_values.data());
        }
        
        handle->data = std::move(matrix);
    });
    
    return handle;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

SCL_API
auto scl_sparse_from_dense(
    std::int64_t rows,
    std::int64_t cols,
    const void* data,
    scl_real_type_t real_type,
    scl_index_type_t index_type,
    scl_layout_t layout,
    double tolerance
) -> scl_sparse_t {
    SCL_C_API_BEGIN
    SCL_CHECK_ARG(rows >= 0, "rows must be non-negative");
    SCL_CHECK_ARG(cols >= 0, "cols must be non-negative");
    SCL_CHECK_NOT_NULL(data, "data");
    SCL_CHECK_ARG(tolerance >= 0, "tolerance must be non-negative");
    
    auto* handle = new scl_sparse_s(real_type, index_type, layout);
    
    SCL_DISPATCH_SPARSE(real_type, index_type, layout, {
        using RealT = SCL_REAL_TYPE;
        using IndexT = SCL_INDEX_TYPE;
        using SparseT = scl::Sparse<RealT, IndexT, SCL_IS_CSR>;
        
        auto data_span = std::span<const RealT>(
            static_cast<const RealT*>(data),
            static_cast<std::size_t>(rows * cols)
        );
        
        if (tolerance > 0) {
            auto tol = static_cast<RealT>(tolerance);
            handle->data = SparseT::from_dense(
                static_cast<IndexT>(rows),
                static_cast<IndexT>(cols),
                data_span,
                [tol](RealT x) { return std::abs(x) > tol; }
            );
        } else {
            handle->data = SparseT::from_dense(
                static_cast<IndexT>(rows),
                static_cast<IndexT>(cols),
                data_span
            );
        }
    });
    
    return handle;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

// =============================================================================
// SECTION 4: Destruction
// =============================================================================

SCL_API
auto scl_sparse_destroy(scl_sparse_t handle) -> void {
    if (handle != SCL_NULL_SPARSE) {
        delete handle;
    }
}

// =============================================================================
// SECTION 5: Property Queries
// =============================================================================

SCL_API
auto scl_sparse_rows(scl_sparse_t handle) -> std::int64_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return 0;
    
    return visit_sparse(handle, [](const auto& mat) -> std::int64_t {
        return static_cast<std::int64_t>(mat.rows());
    });
}

SCL_API
auto scl_sparse_cols(scl_sparse_t handle) -> std::int64_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return 0;
    
    return visit_sparse(handle, [](const auto& mat) -> std::int64_t {
        return static_cast<std::int64_t>(mat.cols());
    });
}

SCL_API
auto scl_sparse_nnz(scl_sparse_t handle) -> std::int64_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return 0;
    
    return visit_sparse(handle, [](const auto& mat) -> std::int64_t {
        return static_cast<std::int64_t>(mat.nnz());
    });
}

SCL_API
auto scl_sparse_density(scl_sparse_t handle) -> double {
    if (!SCL_IS_VALID_SPARSE(handle)) return 0.0;
    
    return visit_sparse(handle, [](const auto& mat) -> double {
        return mat.density();
    });
}

SCL_API
auto scl_sparse_sparsity(scl_sparse_t handle) -> double {
    if (!SCL_IS_VALID_SPARSE(handle)) return 1.0;
    
    return visit_sparse(handle, [](const auto& mat) -> double {
        return mat.sparsity();
    });
}

SCL_API
auto scl_sparse_is_empty(scl_sparse_t handle) -> std::int32_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return 1;
    
    return visit_sparse(handle, [](const auto& mat) -> std::int32_t {
        return mat.empty() ? 1 : 0;
    });
}

SCL_API
auto scl_sparse_real_type(scl_sparse_t handle) -> scl_real_type_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return SCL_REAL64;
    return handle->real_type;
}

SCL_API
auto scl_sparse_index_type(scl_sparse_t handle) -> scl_index_type_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return SCL_INDEX32;
    return handle->index_type;
}

SCL_API
auto scl_sparse_layout(scl_sparse_t handle) -> scl_layout_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return SCL_LAYOUT_CSR;
    return handle->layout;
}

// =============================================================================
// SECTION 6: Row/Column Data Access
// =============================================================================

SCL_API
auto scl_sparse_primary_length(scl_sparse_t handle, std::int64_t idx) -> std::int64_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return -1;
    
    SCL_C_API_BEGIN
    return visit_sparse(handle, [idx](const auto& mat) -> std::int64_t {
        using MatT = std::decay_t<decltype(mat)>;
        using IndexT = typename MatT::index_type;
        return static_cast<std::int64_t>(mat.primary_length(static_cast<IndexT>(idx)));
    });
    SCL_C_API_END
    return -1;
}

SCL_API
auto scl_sparse_row_data(
    scl_sparse_t handle,
    std::int64_t row,
    const void** values,
    const void** indices,
    std::int64_t* length
) -> std::int32_t {
    SCL_C_API_BEGIN
    SCL_CHECK_NOT_NULL(handle, "handle");
    SCL_CHECK_ARG(handle->layout == SCL_LAYOUT_CSR, "row_data requires CSR layout");
    SCL_CHECK_NOT_NULL(values, "values");
    SCL_CHECK_NOT_NULL(indices, "indices");
    SCL_CHECK_NOT_NULL(length, "length");
    
    visit_sparse(handle, [row, values, indices, length](const auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using IndexT = typename MatT::index_type;
        
        if constexpr (MatT::is_csr) {
            auto row_idx = static_cast<IndexT>(row);
            auto val_span = mat.row_values(row_idx);
            auto idx_span = mat.row_indices(row_idx);
            
            *values = val_span.data();
            *indices = idx_span.data();
            *length = static_cast<std::int64_t>(val_span.size());
        }
    });
    
    return 0;
    SCL_C_API_END
    return static_cast<std::int32_t>(scl::ErrorCode::Unknown);
}

SCL_API
auto scl_sparse_col_data(
    scl_sparse_t handle,
    std::int64_t col,
    const void** values,
    const void** indices,
    std::int64_t* length
) -> std::int32_t {
    SCL_C_API_BEGIN
    SCL_CHECK_NOT_NULL(handle, "handle");
    SCL_CHECK_ARG(handle->layout == SCL_LAYOUT_CSC, "col_data requires CSC layout");
    SCL_CHECK_NOT_NULL(values, "values");
    SCL_CHECK_NOT_NULL(indices, "indices");
    SCL_CHECK_NOT_NULL(length, "length");
    
    visit_sparse(handle, [col, values, indices, length](const auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using IndexT = typename MatT::index_type;
        
        if constexpr (!MatT::is_csr) {
            auto col_idx = static_cast<IndexT>(col);
            auto val_span = mat.col_values(col_idx);
            auto idx_span = mat.col_indices(col_idx);
            
            *values = val_span.data();
            *indices = idx_span.data();
            *length = static_cast<std::int64_t>(val_span.size());
        }
    });
    
    return 0;
    SCL_C_API_END
    return static_cast<std::int32_t>(scl::ErrorCode::Unknown);
}

// =============================================================================
// SECTION 7: Element Access
// =============================================================================

SCL_API
auto scl_sparse_at(
    scl_sparse_t handle,
    std::int64_t row,
    std::int64_t col,
    void* value
) -> std::int32_t {
    SCL_C_API_BEGIN
    SCL_CHECK_NOT_NULL(handle, "handle");
    SCL_CHECK_NOT_NULL(value, "value");
    
    visit_sparse(handle, [row, col, value](const auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using RealT = typename MatT::value_type;
        using IndexT = typename MatT::index_type;
        
        auto result = mat.at(static_cast<IndexT>(row), static_cast<IndexT>(col));
        *static_cast<RealT*>(value) = result;
    });
    
    return 0;
    SCL_C_API_END
    return static_cast<std::int32_t>(scl::ErrorCode::Unknown);
}

SCL_API
auto scl_sparse_get(scl_sparse_t handle, std::int64_t row, std::int64_t col) -> double {
    if (!SCL_IS_VALID_SPARSE(handle)) return 0.0;
    
    SCL_C_API_BEGIN_VOID
    return visit_sparse(handle, [row, col](const auto& mat) -> double {
        using MatT = std::decay_t<decltype(mat)>;
        using IndexT = typename MatT::index_type;
        
        return static_cast<double>(mat.at(static_cast<IndexT>(row), static_cast<IndexT>(col)));
    });
    SCL_C_API_END_VOID
    return 0.0;
}

SCL_API
auto scl_sparse_exists(scl_sparse_t handle, std::int64_t row, std::int64_t col) -> std::int32_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return 0;
    
    SCL_C_API_BEGIN_VOID
    return visit_sparse(handle, [row, col](const auto& mat) -> std::int32_t {
        using MatT = std::decay_t<decltype(mat)>;
        using IndexT = typename MatT::index_type;
        
        return mat.exists(static_cast<IndexT>(row), static_cast<IndexT>(col)) ? 1 : 0;
    });
    SCL_C_API_END_VOID
    return 0;
}

// =============================================================================
// SECTION 8: Clone and Transform
// =============================================================================

SCL_API
auto scl_sparse_clone(scl_sparse_t handle) -> scl_sparse_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return SCL_NULL_SPARSE;
    
    SCL_C_API_BEGIN
    auto* result = create_handle_like(handle);
    
    visit_sparse(handle, [result](const auto& mat) {
        result->data = mat.clone();
    });
    
    return result;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

SCL_API
auto scl_sparse_clone_with_strategy(
    scl_sparse_t handle,
    scl_buffer_config_t config
) -> scl_sparse_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return SCL_NULL_SPARSE;
    
    SCL_C_API_BEGIN
    auto* result = create_handle_like(handle);
    
    // Convert C config to C++ strategy
    scl::SparseBufferStrategy strategy;
    switch (config.strategy) {
        case SCL_BUFFER_FRAGMENTED:
            strategy = scl::SparseBufferStrategy::fragmented();
            break;
        case SCL_BUFFER_SINGLE:
            strategy = scl::SparseBufferStrategy::single_buffer();
            break;
        case SCL_BUFFER_MIN_SIZE:
            strategy = scl::SparseBufferStrategy::min_size(config.param);
            break;
        case SCL_BUFFER_COUNT:
            strategy = scl::SparseBufferStrategy::buffer_count(config.param);
            break;
        case SCL_BUFFER_AUTO:
        default:
            strategy = scl::SparseBufferStrategy::auto_strategy();
            break;
    }
    
    visit_sparse(handle, [result, &strategy](const auto& mat) {
        result->data = mat.clone(strategy);
    });
    
    return result;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

SCL_API
auto scl_sparse_transpose(scl_sparse_t handle) -> scl_sparse_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return SCL_NULL_SPARSE;
    
    SCL_C_API_BEGIN
    auto* result = create_transposed_handle(handle);
    
    visit_sparse(handle, [result](const auto& mat) {
        result->data = mat.transpose();
    });
    
    return result;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

// =============================================================================
// SECTION 9: In-Place Operations
// =============================================================================

SCL_API
auto scl_sparse_scale(scl_sparse_t handle, double scalar) -> std::int32_t {
    SCL_C_API_BEGIN
    SCL_CHECK_NOT_NULL(handle, "handle");
    
    visit_sparse(handle, [scalar](auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using RealT = typename MatT::value_type;
        mat.scale(static_cast<RealT>(scalar));
    });
    
    return 0;
    SCL_C_API_END
    return static_cast<std::int32_t>(scl::ErrorCode::Unknown);
}

SCL_API
auto scl_sparse_sort_indices(scl_sparse_t handle) -> std::int32_t {
    SCL_C_API_BEGIN
    SCL_CHECK_NOT_NULL(handle, "handle");
    
    visit_sparse(handle, [](auto& mat) {
        mat.sort_indices();
    });
    
    return 0;
    SCL_C_API_END
    return static_cast<std::int32_t>(scl::ErrorCode::Unknown);
}

SCL_API
auto scl_sparse_is_sorted(scl_sparse_t handle) -> std::int32_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return 0;
    
    SCL_C_API_BEGIN_VOID
    return visit_sparse(handle, [](const auto& mat) -> std::int32_t {
        return mat.is_sorted() ? 1 : 0;
    });
    SCL_C_API_END_VOID
    return 0;
}

// =============================================================================
// SECTION 10: Slicing Operations
// =============================================================================

SCL_API
auto scl_sparse_row_slice(
    scl_sparse_t handle,
    std::int64_t start,
    std::int64_t end
) -> scl_sparse_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return SCL_NULL_SPARSE;
    
    SCL_C_API_BEGIN
    auto* result = create_handle_like(handle);
    
    visit_sparse(handle, [result, start, end](const auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using IndexT = typename MatT::index_type;
        
        result->data = scl::sparse::slice_rows(
            mat,
            static_cast<IndexT>(start),
            static_cast<IndexT>(end)
        );
    });
    
    return result;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

SCL_API
auto scl_sparse_col_slice(
    scl_sparse_t handle,
    std::int64_t start,
    std::int64_t end
) -> scl_sparse_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return SCL_NULL_SPARSE;
    
    SCL_C_API_BEGIN
    auto* result = create_handle_like(handle);
    
    visit_sparse(handle, [result, start, end](const auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using IndexT = typename MatT::index_type;
        
        result->data = scl::sparse::slice_cols(
            mat,
            static_cast<IndexT>(start),
            static_cast<IndexT>(end)
        );
    });
    
    return result;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

SCL_API
auto scl_sparse_row_select(
    scl_sparse_t handle,
    const std::int64_t* row_indices,
    std::int64_t count
) -> scl_sparse_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return SCL_NULL_SPARSE;
    if (row_indices == nullptr || count <= 0) return SCL_NULL_SPARSE;
    
    SCL_C_API_BEGIN
    auto* result = create_handle_like(handle);
    
    visit_sparse(handle, [result, row_indices, count](const auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using IndexT = typename MatT::index_type;
        
        // Convert int64_t indices to IndexT
        std::vector<IndexT> indices(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i) {
            indices[static_cast<std::size_t>(i)] = static_cast<IndexT>(row_indices[i]);
        }
        
        if constexpr (MatT::is_csr) {
            result->data = mat.row_select(std::span<const IndexT>(indices));
        } else {
            // For CSC, row_select is not a primary operation
            // Need to implement as secondary slice
            SCL_NOT_IMPLEMENTED("row_select for CSC matrix");
        }
    });
    
    return result;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

SCL_API
auto scl_sparse_col_select(
    scl_sparse_t handle,
    const std::int64_t* col_indices,
    std::int64_t count
) -> scl_sparse_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return SCL_NULL_SPARSE;
    if (col_indices == nullptr || count <= 0) return SCL_NULL_SPARSE;
    
    SCL_C_API_BEGIN
    auto* result = create_handle_like(handle);
    
    visit_sparse(handle, [result, col_indices, count](const auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using IndexT = typename MatT::index_type;
        
        // Convert int64_t indices to IndexT
        std::vector<IndexT> indices(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i) {
            indices[static_cast<std::size_t>(i)] = static_cast<IndexT>(col_indices[i]);
        }
        
        if constexpr (!MatT::is_csr) {
            result->data = mat.col_select(std::span<const IndexT>(indices));
        } else {
            // For CSR, col_select is not a primary operation
            SCL_NOT_IMPLEMENTED("col_select for CSR matrix");
        }
    });
    
    return result;
    SCL_C_API_END
    return SCL_NULL_SPARSE;
}

// =============================================================================
// SECTION 11: Export Functions
// =============================================================================

SCL_API
auto scl_sparse_to_dense(scl_sparse_t handle, void* data) -> std::int32_t {
    SCL_C_API_BEGIN
    SCL_CHECK_NOT_NULL(handle, "handle");
    SCL_CHECK_NOT_NULL(data, "data");
    
    visit_sparse(handle, [data](const auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using RealT = typename MatT::value_type;
        
        auto dense = mat.to_dense();
        auto* out = static_cast<RealT*>(data);
        std::copy(dense.begin(), dense.end(), out);
    });
    
    return 0;
    SCL_C_API_END
    return static_cast<std::int32_t>(scl::ErrorCode::Unknown);
}

SCL_API
auto scl_sparse_dense_buffer_size(scl_sparse_t handle) -> std::size_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return 0;
    
    auto rows = scl_sparse_rows(handle);
    auto cols = scl_sparse_cols(handle);
    auto elem_size = scl_real_type_size(handle->real_type);
    
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * elem_size;
}

SCL_API
auto scl_sparse_to_coo(
    scl_sparse_t handle,
    void* row_indices,
    void* col_indices,
    void* values
) -> std::int32_t {
    SCL_C_API_BEGIN
    SCL_CHECK_NOT_NULL(handle, "handle");
    SCL_CHECK_NOT_NULL(row_indices, "row_indices");
    SCL_CHECK_NOT_NULL(col_indices, "col_indices");
    SCL_CHECK_NOT_NULL(values, "values");
    
    visit_sparse(handle, [row_indices, col_indices, values](const auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using RealT = typename MatT::value_type;
        using IndexT = typename MatT::index_type;
        
        auto* out_rows = static_cast<IndexT*>(row_indices);
        auto* out_cols = static_cast<IndexT*>(col_indices);
        auto* out_vals = static_cast<RealT*>(values);
        
        std::size_t idx = 0;
        
        if constexpr (MatT::is_csr) {
            for (IndexT r = 0; r < mat.rows(); ++r) {
                auto row_vals = mat.row_values(r);
                auto row_idxs = mat.row_indices(r);
                
                for (std::size_t j = 0; j < row_vals.size(); ++j) {
                    out_rows[idx] = r;
                    out_cols[idx] = row_idxs[j];
                    out_vals[idx] = row_vals[j];
                    ++idx;
                }
            }
        } else {
            for (IndexT c = 0; c < mat.cols(); ++c) {
                auto col_vals = mat.col_values(c);
                auto col_idxs = mat.col_indices(c);
                
                for (std::size_t j = 0; j < col_vals.size(); ++j) {
                    out_rows[idx] = col_idxs[j];
                    out_cols[idx] = c;
                    out_vals[idx] = col_vals[j];
                    ++idx;
                }
            }
        }
    });
    
    return 0;
    SCL_C_API_END
    return static_cast<std::int32_t>(scl::ErrorCode::Unknown);
}

// =============================================================================
// SECTION 12: Unsafe Access Functions
// =============================================================================

// Include unsafe type definitions (always needed for implementation)
#define SCL_UNSAFE_ACCESS
#include "unsafe.h"
#undef SCL_UNSAFE_ACCESS

namespace {

/// @brief Owned sentinel address (matches SharedSpan implementation)
inline
auto owned_sentinel_ptr() -> void* {
    static const char sentinel = 0;
    return const_cast<char*>(&sentinel);
}

}  // namespace

SCL_API
auto scl_unsafe_span_mode(const scl_span_t* span) -> scl_span_mode_t {
    if (!span) return SCL_SPAN_VIEW;
    if (span->buffer == nullptr) return SCL_SPAN_VIEW;
    if (span->buffer == owned_sentinel_ptr()) return SCL_SPAN_OWNED;
    return SCL_SPAN_SHARED;
}

SCL_API
auto scl_unsafe_span_use_count(const scl_span_t* span) -> std::int32_t {
    if (!span) return 0;
    auto mode = scl_unsafe_span_mode(span);
    if (mode == SCL_SPAN_VIEW) return 0;
    if (mode == SCL_SPAN_OWNED) return 1;
    // Shared mode - get use count from SharedBuffer
    auto* buffer = static_cast<scl::SharedBuffer*>(span->buffer);
    return static_cast<std::int32_t>(buffer->use_count());
}

SCL_API
auto scl_unsafe_span_offset_bytes(const scl_span_t* span) -> std::int64_t {
    if (!span || scl_unsafe_span_mode(span) != SCL_SPAN_SHARED) return 0;
    auto* buffer = static_cast<scl::SharedBuffer*>(span->buffer);
    return static_cast<std::int64_t>(
        static_cast<const char*>(span->data) - 
        static_cast<const char*>(buffer->data())
    );
}

SCL_API
auto scl_unsafe_span_incref(scl_span_t* span) -> void {
    if (!span || scl_unsafe_span_mode(span) != SCL_SPAN_SHARED) return;
    auto* buffer = static_cast<scl::SharedBuffer*>(span->buffer);
    buffer->incref();
}

SCL_API
auto scl_unsafe_span_decref(scl_span_t* span) -> void {
    if (!span || scl_unsafe_span_mode(span) != SCL_SPAN_SHARED) return;
    auto* buffer = static_cast<scl::SharedBuffer*>(span->buffer);
    buffer->decref();
}

SCL_API
auto scl_unsafe_sparse_primary_dim(scl_sparse_t handle) -> std::int64_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return 0;
    return visit_sparse(handle, [](const auto& mat) -> std::int64_t {
        return static_cast<std::int64_t>(mat.primary_dim());
    });
}

SCL_API
auto scl_unsafe_sparse_secondary_dim(scl_sparse_t handle) -> std::int64_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return 0;
    return visit_sparse(handle, [](const auto& mat) -> std::int64_t {
        return static_cast<std::int64_t>(mat.secondary_dim());
    });
}

SCL_API
auto scl_unsafe_sparse_rows_ptr(scl_sparse_t handle, std::int64_t* count) -> scl_sparse_row_t* {
    if (!SCL_IS_VALID_SPARSE(handle)) return nullptr;
    
    scl_sparse_row_t* result = nullptr;
    
    visit_sparse(handle, [&result, count](auto& mat) {
        // values and indices are in separate vectors, so we can't return a 
        // contiguous scl_sparse_row_t*
        if (count) *count = static_cast<std::int64_t>(mat.primary_dim());
        result = nullptr;
    });
    
    return result;
}

SCL_API
auto scl_unsafe_sparse_row_ptr(scl_sparse_t handle, std::int64_t idx) -> scl_sparse_row_t* {
    // values and indices are in separate vectors - use batch access instead
    (void)handle;
    (void)idx;
    return nullptr;
}

SCL_API
auto scl_unsafe_sparse_handle_size() -> std::size_t {
    return sizeof(scl_sparse_s);
}

SCL_API
auto scl_unsafe_sparse_handle_align() -> std::size_t {
    return alignof(scl_sparse_s);
}

SCL_API
auto scl_unsafe_sparse_variant_offset() -> std::size_t {
    return offsetof(scl_sparse_s, data);
}

SCL_API
auto scl_unsafe_sparse_get_all_rows(
    scl_sparse_t handle,
    void** values,
    void** indices,
    std::int64_t* lengths
) -> std::int32_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return -1;
    if (!values || !indices || !lengths) return -1;
    
    visit_sparse(handle, [values, indices, lengths](const auto& mat) {
        using MatT = std::decay_t<decltype(mat)>;
        using IndexT = typename MatT::index_type;
        
        auto primary = mat.primary_dim();
        for (IndexT i = 0; i < primary; ++i) {
            auto v = mat.primary_values(i);
            auto idx = mat.primary_indices(i);
            
            values[i] = const_cast<void*>(static_cast<const void*>(v.data()));
            indices[i] = const_cast<void*>(static_cast<const void*>(idx.data()));
            lengths[i] = static_cast<std::int64_t>(v.size());
        }
    });
    
    return 0;
}

SCL_API
auto scl_unsafe_sparse_set_row_view(
    scl_sparse_t handle,
    std::int64_t idx,
    void* values,
    void* indices,
    std::int64_t length
) -> std::int32_t {
    if (!SCL_IS_VALID_SPARSE(handle)) return -1;
    
    SCL_TRY_PTR {
        visit_sparse(handle, [idx, values, indices, length](auto& mat) {
            using MatT = std::decay_t<decltype(mat)>;
            using RealT = typename MatT::value_type;
            using IndexT = typename MatT::index_type;
            
            auto row_idx = static_cast<IndexT>(idx);
            auto len = static_cast<scl::Size>(length);
            
            // Create view spans from raw pointers
            auto values_view = scl::SharedSpan<RealT>::view(
                static_cast<RealT*>(values), len
            );
            auto indices_view = scl::SharedSpan<IndexT>::view(
                static_cast<IndexT*>(indices), len
            );
            
            // Replace the spans in the matrix
            mat.values()[static_cast<std::size_t>(row_idx)] = std::move(values_view);
            mat.indices()[static_cast<std::size_t>(row_idx)] = std::move(indices_view);
        });
        
        return 0;
    }
    SCL_CATCH_PTR(-1);
}

