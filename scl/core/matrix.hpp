#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"

// =============================================================================
/// @file matrix.hpp
/// @brief SCL Matrix Views (Dense & Sparse)
///
/// Defines lightweight, non-owning views for 2D data structures.
/// Designed to bridge the gap between Python/NumPy/SciPy memory layouts
/// and C++ computational kernels without copying data.
///
/// @section Layouts
/// - **DenseMatrix**: Row-Major contiguous (C-style). Compatible with `np.array`.
/// - **CSRMatrix**: Compressed Sparse Row. Compatible with `scipy.sparse.csr_matrix`.
/// - **CSCMatrix**: Compressed Sparse Column. Compatible with `scipy.sparse.csc_matrix`.
///
// =============================================================================

namespace scl {

// =============================================================================
// SECTION 1: Dense Matrix View (NumPy Compatible)
// =============================================================================

/// @brief Non-owning view of a dense, row-major matrix.
///
/// Used to pass `numpy.ndarray` (float32/64) to C++ kernels.
/// Assumes generic C-contiguous layout (stride == cols).
///
/// @tparam T Element type (Real or Index)
template <typename T>
struct DenseMatrix {
    T* ptr;         ///< Pointer to the first element (0,0)
    Index rows;     ///< Number of rows (M)
    Index cols;     ///< Number of columns (N)

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    
    constexpr DenseMatrix() noexcept : ptr(nullptr), rows(0), cols(0) {}

    constexpr DenseMatrix(T* p, Index r, Index c) noexcept 
        : ptr(p), rows(r), cols(c) {}

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /// @brief Get a specific element (Row-Major).
    SCL_FORCE_INLINE T& operator()(Index r, Index c) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "DenseMatrix: Row index out of bounds");
        SCL_ASSERT(c >= 0 && c < cols, "DenseMatrix: Col index out of bounds");
#endif
        return ptr[r * cols + c];
    }

    /// @brief Get a view of a specific row.
    /// Crucial for iterating over genes/cells in SIMD kernels.
    SCL_FORCE_INLINE Span<T> row(Index r) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "DenseMatrix: Row index out of bounds");
#endif
        return Span<T>(ptr + (r * cols), static_cast<Size>(cols));
    }

    /// @brief Get the total number of elements.
    constexpr Size size() const noexcept { return static_cast<Size>(rows * cols); }

    /// @brief Check if matrix is empty.
    constexpr bool empty() const noexcept { return rows == 0 || cols == 0; }
    
    /// @brief Get raw data pointer.
    constexpr T* data() const noexcept { return ptr; }
};

// =============================================================================
// SECTION 2: Sparse Matrix View (SciPy CSR Compatible)
// =============================================================================

/// @brief Non-owning view of a Compressed Sparse Row (CSR) matrix.
///
/// Used to pass `scipy.sparse.csr_matrix` or `AnnData.X` to C++ kernels.
///
/// Structure:
/// - `data`: The non-zero values (size = nnz).
/// - `indices`: Column indices for each value (size = nnz).
/// - `indptr`: Row start pointers (size = rows + 1).
///
/// @tparam T Value type (usually Real)
template <typename T>
struct CSRMatrix {
    T* data;            ///< data array (values)
    Index* indices;     ///< indices array (column indexes)
    Index* indptr;      ///< indptr array (row offsets)
    
    Index rows;         ///< Number of rows (shape[0])
    Index cols;         ///< Number of columns (shape[1])
    Index nnz;          ///< Number of non-zero elements

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    constexpr CSRMatrix() noexcept 
        : data(nullptr), indices(nullptr), indptr(nullptr), rows(0), cols(0), nnz(0) {}

    constexpr CSRMatrix(T* d, Index* idx, Index* ptr, Index r, Index c, Index n) noexcept
        : data(d), indices(idx), indptr(ptr), rows(r), cols(c), nnz(n) {}

    // -------------------------------------------------------------------------
    // Row Access
    // -------------------------------------------------------------------------

    /// @brief Get the span of values for a specific row.
    /// Note: These values are contiguous in the `data` array for CSR.
    SCL_FORCE_INLINE Span<T> row_values(Index r) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "CSRMatrix: Row index out of bounds");
#endif
        Index start = indptr[r];
        Index end = indptr[r + 1];
        return Span<T>(data + start, static_cast<Size>(end - start));
    }

    /// @brief Get the span of column indices for a specific row.
    SCL_FORCE_INLINE Span<Index> row_indices(Index r) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "CSRMatrix: Row index out of bounds");
#endif
        Index start = indptr[r];
        Index end = indptr[r + 1];
        return Span<Index>(indices + start, static_cast<Size>(end - start));
    }
};

// =============================================================================
// SECTION 3: Compressed Sparse Column (CSC) Matrix View
// =============================================================================

/// @brief Non-owning view of a Compressed Sparse Column (CSC) matrix.
///
/// Used to pass `scipy.sparse.csc_matrix` to C++ kernels.
/// CSC is the transpose of CSR, optimized for column-wise access patterns.
///
/// Structure:
/// - `data`: The non-zero values (size = nnz).
/// - `indices`: Row indices for each value (size = nnz).
/// - `indptr`: Column start pointers (size = cols + 1).
///
/// @tparam T Value type (usually Real)
template <typename T>
struct CSCMatrix {
    T* data;            ///< data array (values)
    Index* indices;     ///< indices array (row indexes)
    Index* indptr;      ///< indptr array (column offsets)
    
    Index rows;         ///< Number of rows (shape[0])
    Index cols;         ///< Number of columns (shape[1])
    Index nnz;          ///< Number of non-zero elements

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    constexpr CSCMatrix() noexcept 
        : data(nullptr), indices(nullptr), indptr(nullptr), rows(0), cols(0), nnz(0) {}

    constexpr CSCMatrix(T* d, Index* idx, Index* ptr, Index r, Index c, Index n) noexcept
        : data(d), indices(idx), indptr(ptr), rows(r), cols(c), nnz(n) {}

    // -------------------------------------------------------------------------
    // Column Access
    // -------------------------------------------------------------------------

    /// @brief Get the span of values for a specific column.
    /// Note: These values are contiguous in the `data` array for CSC.
    SCL_FORCE_INLINE Span<T> col_values(Index c) const {
#if !defined(NDEBUG)
        SCL_ASSERT(c >= 0 && c < cols, "CSCMatrix: Column index out of bounds");
#endif
        Index start = indptr[c];
        Index end = indptr[c + 1];
        return Span<T>(data + start, static_cast<Size>(end - start));
    }

    /// @brief Get the span of row indices for a specific column.
    SCL_FORCE_INLINE Span<Index> col_indices(Index c) const {
#if !defined(NDEBUG)
        SCL_ASSERT(c >= 0 && c < cols, "CSCMatrix: Column index out of bounds");
#endif
        Index start = indptr[c];
        Index end = indptr[c + 1];
        return Span<Index>(indices + start, static_cast<Size>(end - start));
    }
};

// =============================================================================
// SECTION 4: Common Aliases
// =============================================================================

/// @brief Standard Dense Matrix of Reals (e.g., float32 gene expression)
using RealMatrix = DenseMatrix<Real>;

/// @brief Standard Sparse Matrix of Reals
using SparseMatrix = CSRMatrix<Real>;

/// @brief Matrix of Indices (e.g., KNN graph adjacency)
using IndexMatrix = DenseMatrix<Index>;

} // namespace scl
