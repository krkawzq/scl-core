#pragma once

/// @file scl/core/sparse.hpp
/// @brief Sparse Matrix with SharedSpan-based Storage
///
/// This header provides:
///   - Sparse<ValueT, IndexT, IsCSR>: CSR/CSC sparse matrix with flexible storage
///   - CSRMatrix<T>, CSCMatrix<T>: Type aliases for convenience
///
/// ## Design Philosophy
///
/// Each row (CSR) or column (CSC) is stored as an independent pair of:
///   - SharedSpan<ValueT>: Non-zero values
///   - SharedSpan<IndexT>: Column (CSR) or row (CSC) indices
///
/// This design enables:
///   - Zero-copy slicing via SharedSpan's reference counting
///   - Flexible memory management (owned/shared/view modes)
///   - Natural row/column independence
///
/// ## Memory Layout
///
/// ```
/// Sparse<T, IndexT, IsCSR>:
///   - values_:  std::vector<SharedSpan<T>>      (primary_dim elements)
///   - indices_: std::vector<SharedSpan<IndexT>> (primary_dim elements)
///   - rows_, cols_: IndexT
/// ```
///
/// Each SharedSpan manages its own lifecycle (24 bytes per span).
/// Total overhead: ~48 bytes per row/column + vector overhead.
///
/// ## Invariants
///
///   - Indices within each row (CSR) or column (CSC) are strictly ascending
///   - values_[i].size() == indices_[i].size() for all i
///   - All indices are within bounds [0, secondary_dim)
///   - values_.size() == indices_.size() == primary_dim
///
/// ## Thread Safety
///
///   - Read operations are thread-safe when SharedSpans are in shared mode
///   - Write operations require external synchronization
///   - Copy/move operations are not thread-safe
///
/// ## Zero-Copy Slicing
///
/// Slicing operations (row_slice, col_slice) create new Sparse objects that
/// share the underlying data via SharedSpan's reference counting. No data
/// is copied until explicitly requested (clone).

#include "scl/core/span.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/threading.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

namespace scl {

// =============================================================================
// SECTION 1: Forward Declarations
// =============================================================================

template<typename ValueT, typename IndexT, bool IsCSR>
class Sparse;

template<typename T, typename IndexT = Index>
using CSRMatrix = Sparse<T, IndexT, true>;

template<typename T, typename IndexT = Index>
using CSCMatrix = Sparse<T, IndexT, false>;

// Default types
using CSR = CSRMatrix<Real>;
using CSC = CSCMatrix<Real>;
using CSRf = CSRMatrix<float>;
using CSCf = CSCMatrix<float>;
using CSRd = CSRMatrix<double>;
using CSCd = CSCMatrix<double>;

// =============================================================================
// SECTION 2: Sparse Layout Info
// =============================================================================

/// @brief Information about sparse matrix memory layout
struct SparseLayoutInfo {
    std::size_t values_bytes = 0;    ///< Total bytes for all values
    std::size_t indices_bytes = 0;   ///< Total bytes for all indices
    std::size_t overhead_bytes = 0;  ///< Vector and SharedSpan overhead
    Size num_spans = 0;              ///< Number of SharedSpan objects
    Size num_shared = 0;             ///< Number of SharedSpans in shared mode
    Size num_owned = 0;              ///< Number of SharedSpans in owned mode
    Size num_view = 0;               ///< Number of SharedSpans in view mode
    
    [[nodiscard]]
    auto total_bytes() const noexcept -> std::size_t {
        return values_bytes + indices_bytes + overhead_bytes;
    }
};

// =============================================================================
// SECTION 3: Sparse Matrix Implementation
// =============================================================================

/// @brief CSR/CSC Sparse Matrix with SharedSpan-based storage
///
/// @tparam ValueT Value type (typically float or double)
/// @tparam IndexT Index type (typically Index = int32_t)
/// @tparam IsCSR true for CSR format, false for CSC format
template<typename ValueT = Real, typename IndexT = Index, bool IsCSR = true>
class Sparse {
public:
    using ValueType = ValueT;
    using IndexType = IndexT;
    using SelfType = Sparse<ValueT, IndexT, IsCSR>;
    using TransposeType = Sparse<ValueT, IndexT, !IsCSR>;

    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /// @brief Default: empty matrix
    constexpr
    Sparse() noexcept = default;

    /// @brief Construct from dimensions (empty rows/columns)
    /// @param[in] rows Number of rows
    /// @param[in] cols Number of columns
    Sparse(IndexT rows, IndexT cols) noexcept
        : rows_(rows)
        , cols_(cols) {
        const IndexT pdim = primary_dim();
        if (pdim > 0) {
            values_.resize(static_cast<std::size_t>(pdim));
            indices_.resize(static_cast<std::size_t>(pdim));
        }
    }

    // Default copy/move (SharedSpan handles reference counting)
    Sparse(const Sparse&) = default;
    Sparse(Sparse&&) noexcept = default;
    auto operator=(const Sparse&) -> Sparse& = default;
    auto operator=(Sparse&&) noexcept -> Sparse& = default;
    ~Sparse() = default;

    // -------------------------------------------------------------------------
    // Basic Queries
    // -------------------------------------------------------------------------

    /// @brief Check if matrix is valid (has proper dimensions)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto valid() const noexcept -> bool {
        const IndexT pdim = primary_dim();
        return static_cast<IndexT>(values_.size()) == pdim && 
               static_cast<IndexT>(indices_.size()) == pdim;
    }

    /// @brief Check if matrix is valid (bool conversion)
    [[nodiscard]]
    SCL_FORCE_INLINE
    explicit operator bool() const noexcept { return valid(); }

    [[nodiscard]] SCL_FORCE_INLINE auto rows() const noexcept -> IndexT { return rows_; }
    [[nodiscard]] SCL_FORCE_INLINE auto cols() const noexcept -> IndexT { return cols_; }
    
    /// @brief Get total number of non-zeros
    [[nodiscard]]
    auto nnz() const noexcept -> IndexT {
        IndexT total = 0;
        for (const auto& v : values_) {
            total += static_cast<IndexT>(v.size());
        }
        return total;
    }

    /// @brief Get primary dimension (rows for CSR, cols for CSC)
    [[nodiscard]]
    SCL_FORCE_INLINE
    constexpr
    auto primary_dim() const noexcept -> IndexT {
        return IsCSR ? rows_ : cols_;
    }

    /// @brief Get secondary dimension (cols for CSR, rows for CSC)
    [[nodiscard]]
    SCL_FORCE_INLINE
    constexpr
    auto secondary_dim() const noexcept -> IndexT {
        return IsCSR ? cols_ : rows_;
    }

    /// @brief Check if matrix is empty
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto empty() const noexcept -> bool {
        return rows_ == 0 || cols_ == 0;
    }

    /// @brief Calculate sparsity (fraction of zeros)
    [[nodiscard]]
    auto sparsity() const noexcept -> double {
        if (rows_ == 0 || cols_ == 0) return 1.0;
        const auto total = static_cast<double>(rows_) * static_cast<double>(cols_);
        return 1.0 - static_cast<double>(nnz()) / total;
    }

    /// @brief Calculate density (fraction of non-zeros)
    [[nodiscard]]
    auto density() const noexcept -> double {
        return 1.0 - sparsity();
    }

    /// @brief Get memory layout information
    [[nodiscard]]
    auto layout_info() const noexcept -> SparseLayoutInfo {
        SparseLayoutInfo info;
        info.num_spans = static_cast<Size>(values_.size());
        
        for (const auto& v : values_) {
            info.values_bytes += v.size_bytes();
            if (v.is_shared()) ++info.num_shared;
            else if (v.is_owned()) ++info.num_owned;
            else if (v.is_view()) ++info.num_view;
        }
        
        for (const auto& idx : indices_) {
            info.indices_bytes += idx.size_bytes();
        }
        
        // Overhead: vector capacity + SharedSpan objects
        info.overhead_bytes = 
            values_.capacity() * sizeof(SharedSpan<ValueT>) +
            indices_.capacity() * sizeof(SharedSpan<IndexT>) +
            2 * sizeof(std::vector<SharedSpan<ValueT>>);
        
        return info;
    }

    // -------------------------------------------------------------------------
    // Data Access
    // -------------------------------------------------------------------------

    /// @brief Get values vector
    [[nodiscard]] SCL_FORCE_INLINE auto values() noexcept -> std::vector<SharedSpan<ValueT>>& { 
        return values_; 
    }
    [[nodiscard]] SCL_FORCE_INLINE auto values() const noexcept -> const std::vector<SharedSpan<ValueT>>& { 
        return values_; 
    }

    /// @brief Get indices vector
    [[nodiscard]] SCL_FORCE_INLINE auto indices() noexcept -> std::vector<SharedSpan<IndexT>>& { 
        return indices_; 
    }
    [[nodiscard]] SCL_FORCE_INLINE auto indices() const noexcept -> const std::vector<SharedSpan<IndexT>>& { 
        return indices_; 
    }

    // -------------------------------------------------------------------------
    // Row/Column Access (CSR/CSC specific)
    // -------------------------------------------------------------------------

    /// @brief Get row length (CSR only)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto row_length(IndexT i) const noexcept -> IndexT
    requires (IsCSR) {
        return static_cast<IndexT>(values_[static_cast<std::size_t>(i)].size());
    }

    /// @brief Get row values as SharedSpan (CSR only)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto row_values(IndexT i) const noexcept -> const SharedSpan<ValueT>&
    requires (IsCSR) {
        return values_[static_cast<std::size_t>(i)];
    }

    /// @brief Get mutable row values (CSR only)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto row_values(IndexT i) noexcept -> SharedSpan<ValueT>&
    requires (IsCSR) {
        return values_[static_cast<std::size_t>(i)];
    }

    /// @brief Get row indices as SharedSpan (CSR only)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto row_indices(IndexT i) const noexcept -> const SharedSpan<IndexT>&
    requires (IsCSR) {
        return indices_[static_cast<std::size_t>(i)];
    }

    /// @brief Get column length (CSC only)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto col_length(IndexT j) const noexcept -> IndexT
    requires (!IsCSR) {
        return static_cast<IndexT>(values_[static_cast<std::size_t>(j)].size());
    }

    /// @brief Get column values as SharedSpan (CSC only)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto col_values(IndexT j) const noexcept -> const SharedSpan<ValueT>&
    requires (!IsCSR) {
        return values_[static_cast<std::size_t>(j)];
    }

    /// @brief Get mutable column values (CSC only)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto col_values(IndexT j) noexcept -> SharedSpan<ValueT>&
    requires (!IsCSR) {
        return values_[static_cast<std::size_t>(j)];
    }

    /// @brief Get column indices as SharedSpan (CSC only)
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto col_indices(IndexT j) const noexcept -> const SharedSpan<IndexT>&
    requires (!IsCSR) {
        return indices_[static_cast<std::size_t>(j)];
    }

    /// @brief Get primary dimension length
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto primary_length(IndexT i) const noexcept -> IndexT {
        return static_cast<IndexT>(values_[static_cast<std::size_t>(i)].size());
    }

    /// @brief Get primary dimension values
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto primary_values(IndexT i) const noexcept -> const SharedSpan<ValueT>& {
        return values_[static_cast<std::size_t>(i)];
    }

    /// @brief Get mutable primary dimension values
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto primary_values(IndexT i) noexcept -> SharedSpan<ValueT>& {
        return values_[static_cast<std::size_t>(i)];
    }

    /// @brief Get primary dimension indices
    [[nodiscard]]
    SCL_FORCE_INLINE
    auto primary_indices(IndexT i) const noexcept -> const SharedSpan<IndexT>& {
        return indices_[static_cast<std::size_t>(i)];
    }

    // -------------------------------------------------------------------------
    // Element Access
    // -------------------------------------------------------------------------

    /// @brief Get value at (row, col), returns 0 if not found
    /// @note O(log n) - uses binary search
    [[nodiscard]]
    auto at(IndexT row, IndexT col) const noexcept -> ValueT {
        if (!valid() || row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            return ValueT{0};
        }

        const IndexT primary_idx = IsCSR ? row : col;
        const IndexT secondary_idx = IsCSR ? col : row;

        const auto& idx_span = indices_[static_cast<std::size_t>(primary_idx)];
        if (idx_span.empty()) return ValueT{0};

        // Binary search
        auto it = std::lower_bound(idx_span.begin(), idx_span.end(), secondary_idx);
        if (it != idx_span.end() && *it == secondary_idx) {
            const auto pos = static_cast<Size>(it - idx_span.begin());
            return values_[static_cast<std::size_t>(primary_idx)][pos];
        }
        return ValueT{0};
    }

    /// @brief Check if element exists at (row, col)
    [[nodiscard]]
    auto exists(IndexT row, IndexT col) const noexcept -> bool {
        if (!valid() || row < 0 || row >= rows_ || col < 0 || col >= cols_) {
            return false;
        }

        const IndexT primary_idx = IsCSR ? row : col;
        const IndexT secondary_idx = IsCSR ? col : row;

        const auto& idx_span = indices_[static_cast<std::size_t>(primary_idx)];
        if (idx_span.empty()) return false;

        auto it = std::lower_bound(idx_span.begin(), idx_span.end(), secondary_idx);
        return it != idx_span.end() && *it == secondary_idx;
    }

    // -------------------------------------------------------------------------
    // Factory: Create Empty/Zero Matrix
    // -------------------------------------------------------------------------

    /// @brief Create zero matrix (empty rows/columns)
    [[nodiscard]]
    static
    auto zeros(IndexT rows, IndexT cols) -> Sparse {
        if (rows < 0 || cols < 0) return {};
        return Sparse(rows, cols);
    }

    // -------------------------------------------------------------------------
    // Factory: Create from NNZ Counts
    // -------------------------------------------------------------------------

    /// @brief Create matrix with specified NNZ per row/column
    /// @param[in] rows Number of rows
    /// @param[in] cols Number of columns
    /// @param[in] primary_nnzs NNZ count for each primary dimension
    /// @return Sparse matrix with allocated but uninitialized data
    [[nodiscard]]
    static
    auto create(IndexT rows, IndexT cols, std::span<const IndexT> primary_nnzs) -> Sparse {
        SCL_CHECK_ARG(rows >= 0 && cols >= 0, "dimensions must be non-negative");

        const IndexT pdim = IsCSR ? rows : cols;
        SCL_CHECK_ARG(static_cast<IndexT>(primary_nnzs.size()) == pdim,
                      "nnz array size mismatch");

        if (rows == 0 || cols == 0) return {};

        Sparse result(rows, cols);

        // Allocate SharedSpans for each row/column
        for (IndexT i = 0; i < pdim; ++i) {
            const IndexT count = primary_nnzs[static_cast<std::size_t>(i)];
            SCL_CHECK_ARG(count >= 0, "nnz counts must be non-negative");
            
            if (count > 0) {
                result.values_[static_cast<std::size_t>(i)] = 
                    SharedSpan<ValueT>::create_aligned(static_cast<Size>(count));
                result.indices_[static_cast<std::size_t>(i)] = 
                    SharedSpan<IndexT>::create_aligned(static_cast<Size>(count));
                
                if (!result.values_[static_cast<std::size_t>(i)] || 
                    !result.indices_[static_cast<std::size_t>(i)]) {
                    return {};  // Allocation failed
                }
            }
        }

        return result;
    }

    // -------------------------------------------------------------------------
    // Factory: Create from Traditional Format
    // -------------------------------------------------------------------------

    /// @brief Create from traditional CSR/CSC arrays (copies data)
    /// @param[in] rows Number of rows
    /// @param[in] cols Number of columns
    /// @param[in] values Non-zero values
    /// @param[in] indices Column (CSR) or row (CSC) indices
    /// @param[in] indptr Row (CSR) or column (CSC) pointers
    [[nodiscard]]
    static
    auto from_arrays(
        IndexT rows,
        IndexT cols,
        std::span<const ValueT> values,
        std::span<const IndexT> indices,
        std::span<const IndexT> indptr
    ) -> Sparse {
        const IndexT pdim = IsCSR ? rows : cols;
        SCL_CHECK_ARG(static_cast<IndexT>(indptr.size()) == pdim + 1,
                      "indptr size must be primary_dim + 1");
        SCL_CHECK_ARG(pdim == 0 || indptr[0] == 0, "indptr[0] must be 0");

        const IndexT nnz_count = pdim > 0 ? indptr[static_cast<std::size_t>(pdim)] : 0;
        SCL_CHECK_ARG(static_cast<IndexT>(values.size()) >= nnz_count,
                      "values size mismatch");
        SCL_CHECK_ARG(static_cast<IndexT>(indices.size()) >= nnz_count,
                      "indices size mismatch");

        if (rows == 0 || cols == 0 || nnz_count == 0) {
            return zeros(rows, cols);
        }

        // Extract row/column lengths
        std::vector<IndexT> primary_nnzs(static_cast<std::size_t>(pdim));
        for (IndexT i = 0; i < pdim; ++i) {
            primary_nnzs[static_cast<std::size_t>(i)] = 
                indptr[static_cast<std::size_t>(i + 1)] - indptr[static_cast<std::size_t>(i)];
        }

        auto result = create(rows, cols, primary_nnzs);
        if (!result) return {};

        // Copy data
        for (IndexT i = 0; i < pdim; ++i) {
            const IndexT start = indptr[static_cast<std::size_t>(i)];
            const IndexT len = primary_nnzs[static_cast<std::size_t>(i)];
            
            if (len > 0) {
                auto& val_span = result.values_[static_cast<std::size_t>(i)];
                auto& idx_span = result.indices_[static_cast<std::size_t>(i)];
                
                std::copy(
                    values.begin() + start,
                    values.begin() + start + len,
                    val_span.begin()
                );
                std::copy(
                    indices.begin() + start,
                    indices.begin() + start + len,
                    idx_span.begin()
                );
            }
        }

        return result;
    }

    /// @brief Wrap existing arrays (zero-copy view)
    /// @warning Caller must ensure arrays outlive the Sparse object
    [[nodiscard]]
    static
    auto view_arrays(
        IndexT rows,
        IndexT cols,
        ValueT* values,
        IndexT* indices,
        IndexT* indptr
    ) -> Sparse {
        const IndexT pdim = IsCSR ? rows : cols;

        Sparse result(rows, cols);

        // Create views for each row/column
        for (IndexT i = 0; i < pdim; ++i) {
            const IndexT start = indptr[i];
            const IndexT len = indptr[i + 1] - start;
            
            if (len > 0) {
                result.values_[static_cast<std::size_t>(i)] = 
                    SharedSpan<ValueT>::view(values + start, static_cast<Size>(len));
                result.indices_[static_cast<std::size_t>(i)] = 
                    SharedSpan<IndexT>::view(indices + start, static_cast<Size>(len));
            }
        }

        return result;
    }

    // -------------------------------------------------------------------------
    // Factory: Create from COO Format
    // -------------------------------------------------------------------------

    /// @brief Create from COO (Coordinate) format
    /// @note Indices are sorted; duplicates use last value
    [[nodiscard]]
    static
    auto from_coo(
        IndexT rows,
        IndexT cols,
        std::span<const IndexT> row_indices,
        std::span<const IndexT> col_indices,
        std::span<const ValueT> values
    ) -> Sparse {
        const auto nnz_count = static_cast<IndexT>(values.size());
        SCL_CHECK_ARG(static_cast<IndexT>(row_indices.size()) == nnz_count,
                      "row_indices size mismatch");
        SCL_CHECK_ARG(static_cast<IndexT>(col_indices.size()) == nnz_count,
                      "col_indices size mismatch");

        if (nnz_count == 0) return zeros(rows, cols);

        const IndexT pdim = IsCSR ? rows : cols;
        const IndexT sdim = IsCSR ? cols : rows;

        // Count NNZ per primary dimension
        std::vector<IndexT> nnz_per_primary(static_cast<std::size_t>(pdim), 0);
        for (IndexT i = 0; i < nnz_count; ++i) {
            const IndexT pidx = IsCSR ? row_indices[static_cast<std::size_t>(i)] : 
                                        col_indices[static_cast<std::size_t>(i)];
            const IndexT sidx = IsCSR ? col_indices[static_cast<std::size_t>(i)] : 
                                        row_indices[static_cast<std::size_t>(i)];
            SCL_CHECK_ARG(pidx >= 0 && pidx < pdim, "primary index out of bounds");
            SCL_CHECK_ARG(sidx >= 0 && sidx < sdim, "secondary index out of bounds");
            ++nnz_per_primary[static_cast<std::size_t>(pidx)];
        }

        // Create matrix
        auto result = create(rows, cols, nnz_per_primary);
        if (!result) return {};

        // Track insertion positions
        std::vector<IndexT> insert_pos(static_cast<std::size_t>(pdim), 0);

        // Fill data (unsorted)
        for (IndexT i = 0; i < nnz_count; ++i) {
            const IndexT pidx = IsCSR ? row_indices[static_cast<std::size_t>(i)] : 
                                        col_indices[static_cast<std::size_t>(i)];
            const IndexT sidx = IsCSR ? col_indices[static_cast<std::size_t>(i)] : 
                                        row_indices[static_cast<std::size_t>(i)];
            const IndexT pos = insert_pos[static_cast<std::size_t>(pidx)]++;

            result.values_[static_cast<std::size_t>(pidx)][static_cast<Size>(pos)] = 
                values[static_cast<std::size_t>(i)];
            result.indices_[static_cast<std::size_t>(pidx)][static_cast<Size>(pos)] = sidx;
        }

        // Sort indices within each primary dimension
        result.sort_indices();

        return result;
    }

    // -------------------------------------------------------------------------
    // Factory: Create Identity Matrix
    // -------------------------------------------------------------------------

    /// @brief Create identity matrix
    [[nodiscard]]
    static
    auto identity(IndexT n) -> Sparse {
        if (n <= 0) return {};

        std::vector<IndexT> nnzs(static_cast<std::size_t>(n), 1);
        auto result = create(n, n, nnzs);
        if (!result) return {};

        for (IndexT i = 0; i < n; ++i) {
            result.values_[static_cast<std::size_t>(i)][0] = ValueT{1};
            result.indices_[static_cast<std::size_t>(i)][0] = i;
        }

        return result;
    }

    // -------------------------------------------------------------------------
    // Factory: Create from Dense
    // -------------------------------------------------------------------------

    /// @brief Create from dense matrix (row-major)
    template<typename Pred = std::nullptr_t>
    [[nodiscard]]
    static
    auto from_dense(
        IndexT rows,
        IndexT cols,
        std::span<const ValueT> data,
        Pred&& is_nonzero = nullptr
    ) -> Sparse {
        SCL_CHECK_ARG(static_cast<IndexT>(data.size()) >= rows * cols,
                      "dense data size mismatch");

        auto check_nonzero = [&](ValueT val) -> bool {
            if constexpr (std::is_same_v<std::decay_t<Pred>, std::nullptr_t>) {
                return val != ValueT{0};
            } else {
                return is_nonzero(val);
            }
        };

        const IndexT pdim = IsCSR ? rows : cols;

        // Count NNZ per primary dimension
        std::vector<IndexT> nnz_per_primary(static_cast<std::size_t>(pdim), 0);

        if constexpr (IsCSR) {
            for (IndexT i = 0; i < rows; ++i) {
                for (IndexT j = 0; j < cols; ++j) {
                    if (check_nonzero(data[static_cast<std::size_t>(i * cols + j)])) {
                        ++nnz_per_primary[static_cast<std::size_t>(i)];
                    }
                }
            }
        } else {
            for (IndexT j = 0; j < cols; ++j) {
                for (IndexT i = 0; i < rows; ++i) {
                    if (check_nonzero(data[static_cast<std::size_t>(i * cols + j)])) {
                        ++nnz_per_primary[static_cast<std::size_t>(j)];
                    }
                }
            }
        }

        auto result = create(rows, cols, nnz_per_primary);
        if (!result) return {};

        // Fill data
        std::vector<IndexT> insert_pos(static_cast<std::size_t>(pdim), 0);

        if constexpr (IsCSR) {
            for (IndexT i = 0; i < rows; ++i) {
                for (IndexT j = 0; j < cols; ++j) {
                    ValueT val = data[static_cast<std::size_t>(i * cols + j)];
                    if (check_nonzero(val)) {
                        const IndexT pos = insert_pos[static_cast<std::size_t>(i)]++;
                        result.values_[static_cast<std::size_t>(i)][static_cast<Size>(pos)] = val;
                        result.indices_[static_cast<std::size_t>(i)][static_cast<Size>(pos)] = j;
                    }
                }
            }
        } else {
            for (IndexT j = 0; j < cols; ++j) {
                for (IndexT i = 0; i < rows; ++i) {
                    ValueT val = data[static_cast<std::size_t>(i * cols + j)];
                    if (check_nonzero(val)) {
                        const IndexT pos = insert_pos[static_cast<std::size_t>(j)]++;
                        result.values_[static_cast<std::size_t>(j)][static_cast<Size>(pos)] = val;
                        result.indices_[static_cast<std::size_t>(j)][static_cast<Size>(pos)] = i;
                    }
                }
            }
        }

        return result;
    }

    // -------------------------------------------------------------------------
    // Clone and Conversion
    // -------------------------------------------------------------------------

    /// @brief Deep copy with parallel execution
    /// @note Uses SIMD-optimized memory operations and parallel cloning
    [[nodiscard]]
    auto clone() const -> Sparse {
        if (!valid()) return {};

        Sparse result(rows_, cols_);
        const Size pdim_size = values_.size();

        // Parallel cloning for large matrices
        if (pdim_size > threading::MIN_PARALLEL_SIZE / 100) {
            threading::parallel_for(static_cast<threading::Index>(0),
                                   static_cast<threading::Index>(pdim_size),
                                   [this, &result](threading::Index i) {
                result.values_[static_cast<std::size_t>(i)] = 
                    values_[static_cast<std::size_t>(i)].clone();
                result.indices_[static_cast<std::size_t>(i)] = 
                    indices_[static_cast<std::size_t>(i)].clone();
            }, threading::DEFAULT_GRAIN_SIZE);
        } else {
            // Serial cloning for small matrices
            for (std::size_t i = 0; i < pdim_size; ++i) {
                result.values_[i] = values_[i].clone();
                result.indices_[i] = indices_[i].clone();
            }
        }

        return result;
    }

    /// @brief Convert to transposed format (CSR <-> CSC)
    [[nodiscard]]
    auto transpose() const -> TransposeType {
        if (!valid()) return {};

        const IndexT new_rows = cols_;
        const IndexT new_cols = rows_;
        const IndexT new_pdim = !IsCSR ? new_rows : new_cols;
        const IndexT old_pdim = primary_dim();

        // Count NNZ per new primary dimension
        std::vector<IndexT> new_nnzs(static_cast<std::size_t>(new_pdim), 0);

        for (IndexT i = 0; i < old_pdim; ++i) {
            const auto& idx_span = indices_[static_cast<std::size_t>(i)];
            for (const auto idx : idx_span) {
                ++new_nnzs[static_cast<std::size_t>(idx)];
            }
        }

        auto result = TransposeType::create(new_rows, new_cols, new_nnzs);
        if (!result) return {};

        // Fill transposed data
        std::vector<IndexT> insert_pos(static_cast<std::size_t>(new_pdim), 0);

        for (IndexT i = 0; i < old_pdim; ++i) {
            const auto& val_span = values_[static_cast<std::size_t>(i)];
            const auto& idx_span = indices_[static_cast<std::size_t>(i)];

            for (Size k = 0; k < val_span.size(); ++k) {
                const IndexT j = idx_span[k];
                const IndexT pos = insert_pos[static_cast<std::size_t>(j)]++;

                result.values_[static_cast<std::size_t>(j)][static_cast<Size>(pos)] = val_span[k];
                result.indices_[static_cast<std::size_t>(j)][static_cast<Size>(pos)] = i;
            }
        }

        result.sort_indices();
        return result;
    }

    // -------------------------------------------------------------------------
    // Row/Column Slicing (Zero-Copy)
    // -------------------------------------------------------------------------

    /// @brief Row range slice (zero-copy, CSR only)
    /// @note SharedSpans are copied (reference counted), no data copy
    [[nodiscard]]
    auto row_slice(IndexT start, IndexT end) const -> Sparse
    requires (IsCSR) {
        SCL_CHECK_ARG(start >= 0 && end <= rows_ && start <= end,
                      "invalid row range");

        if (start == end) return zeros(0, cols_);

        const IndexT new_rows = end - start;
        Sparse result(new_rows, cols_);

        // Copy SharedSpans (reference counted, zero-copy)
        for (IndexT i = 0; i < new_rows; ++i) {
            result.values_[static_cast<std::size_t>(i)] = 
                values_[static_cast<std::size_t>(start + i)];
            result.indices_[static_cast<std::size_t>(i)] = 
                indices_[static_cast<std::size_t>(start + i)];
        }

        return result;
    }

    /// @brief Row selection slice (zero-copy, CSR only)
    [[nodiscard]]
    auto row_select(std::span<const IndexT> row_indices) const -> Sparse
    requires (IsCSR) {
        if (row_indices.empty()) return zeros(0, cols_);

        const auto new_rows = static_cast<IndexT>(row_indices.size());
        Sparse result(new_rows, cols_);

        for (IndexT i = 0; i < new_rows; ++i) {
            const IndexT src = row_indices[static_cast<std::size_t>(i)];
            SCL_CHECK_ARG(src >= 0 && src < rows_, "row index out of bounds");
            
            result.values_[static_cast<std::size_t>(i)] = 
                values_[static_cast<std::size_t>(src)];
            result.indices_[static_cast<std::size_t>(i)] = 
                indices_[static_cast<std::size_t>(src)];
        }

        return result;
    }

    /// @brief Column range slice (zero-copy, CSC only)
    [[nodiscard]]
    auto col_slice(IndexT start, IndexT end) const -> Sparse
    requires (!IsCSR) {
        SCL_CHECK_ARG(start >= 0 && end <= cols_ && start <= end,
                      "invalid column range");

        if (start == end) return zeros(rows_, 0);

        const IndexT new_cols = end - start;
        Sparse result(rows_, new_cols);

        for (IndexT j = 0; j < new_cols; ++j) {
            result.values_[static_cast<std::size_t>(j)] = 
                values_[static_cast<std::size_t>(start + j)];
            result.indices_[static_cast<std::size_t>(j)] = 
                indices_[static_cast<std::size_t>(start + j)];
        }

        return result;
    }

    /// @brief Column selection slice (zero-copy, CSC only)
    [[nodiscard]]
    auto col_select(std::span<const IndexT> col_indices) const -> Sparse
    requires (!IsCSR) {
        if (col_indices.empty()) return zeros(rows_, 0);

        const auto new_cols = static_cast<IndexT>(col_indices.size());
        Sparse result(rows_, new_cols);

        for (IndexT j = 0; j < new_cols; ++j) {
            const IndexT src = col_indices[static_cast<std::size_t>(j)];
            SCL_CHECK_ARG(src >= 0 && src < cols_, "column index out of bounds");
            
            result.values_[static_cast<std::size_t>(j)] = 
                values_[static_cast<std::size_t>(src)];
            result.indices_[static_cast<std::size_t>(j)] = 
                indices_[static_cast<std::size_t>(src)];
        }

        return result;
    }

    // -------------------------------------------------------------------------
    // In-place Operations
    // -------------------------------------------------------------------------

    /// @brief Sort indices within each row/column
    /// @note Uses SIMD-optimized sorting (VQSort) and parallel execution
    auto sort_indices() -> void {
        if (!valid()) return;

        const IndexT pdim = primary_dim();
        
        // Parallel sorting with SIMD optimization
        threading::parallel_for(static_cast<threading::Index>(0), 
                               static_cast<threading::Index>(pdim),
                               [this](threading::Index i) {
            auto& vals = values_[static_cast<std::size_t>(i)];
            auto& idxs = indices_[static_cast<std::size_t>(i)];
            
            const Size len = vals.size();
            if (len <= 1) return;

            // Use SIMD-optimized sort_pairs (sorts indices while reordering values)
            simd::sort_pairs(
                idxs.data(), 
                vals.data(), 
                len
            );
        }, threading::DEFAULT_GRAIN_SIZE);
    }

    /// @brief Verify indices are sorted
    [[nodiscard]]
    auto is_sorted() const noexcept -> bool {
        if (!valid()) return true;

        const IndexT pdim = primary_dim();
        for (IndexT i = 0; i < pdim; ++i) {
            const auto& idx_span = indices_[static_cast<std::size_t>(i)];
            for (Size k = 1; k < idx_span.size(); ++k) {
                if (idx_span[k] <= idx_span[k - 1]) {
                    return false;
                }
            }
        }
        return true;
    }

    /// @brief Scale all values
    auto scale(ValueT factor) -> void {
        if (!valid()) return;
        
        for (auto& v : values_) {
            for (auto& val : v) {
                val *= factor;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Export
    // -------------------------------------------------------------------------

    /// @brief Export to dense matrix (row-major)
    [[nodiscard]]
    auto to_dense() const -> std::vector<ValueT> {
        const auto size = static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);
        std::vector<ValueT> result(size, ValueT{0});

        if (!valid()) return result;

        if constexpr (IsCSR) {
            for (IndexT i = 0; i < rows_; ++i) {
                const auto& vals = values_[static_cast<std::size_t>(i)];
                const auto& idxs = indices_[static_cast<std::size_t>(i)];
                for (Size k = 0; k < vals.size(); ++k) {
                    result[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_) + 
                           static_cast<std::size_t>(idxs[k])] = vals[k];
                }
            }
        } else {
            for (IndexT j = 0; j < cols_; ++j) {
                const auto& vals = values_[static_cast<std::size_t>(j)];
                const auto& idxs = indices_[static_cast<std::size_t>(j)];
                for (Size k = 0; k < vals.size(); ++k) {
                    result[static_cast<std::size_t>(idxs[k]) * static_cast<std::size_t>(cols_) + 
                           static_cast<std::size_t>(j)] = vals[k];
                }
            }
        }

        return result;
    }

private:
    std::vector<SharedSpan<ValueT>> values_;    ///< Values for each row/column
    std::vector<SharedSpan<IndexT>> indices_;   ///< Indices for each row/column
    IndexT rows_ = 0;
    IndexT cols_ = 0;
};

// =============================================================================
// SECTION 4: Static Assertions
// =============================================================================

static_assert(std::is_nothrow_move_constructible_v<CSR>,
              "CSR must be nothrow move constructible");
static_assert(std::is_nothrow_move_assignable_v<CSR>,
              "CSR must be nothrow move assignable");

}  // namespace scl
