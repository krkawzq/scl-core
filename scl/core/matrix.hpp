#pragma once

#include "scl/config.hpp"
#include "scl/core/type.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/error.hpp"

#include <concepts>
#include <type_traits>

// =============================================================================
/// @file matrix.hpp
/// @brief SCL Matrix System: Concept-Based Zero-Overhead Views
///
/// Unified matrix interface using compile-time polymorphism.
///
/// Design Philosophy:
///
/// 1. Duck Typing + Tag Dispatch: Type safety without vtables
/// 2. Zero Overhead: All abstractions inline to pointer arithmetic
/// 3. Unified Interface: Standard and Virtual matrices share identical API
/// 4. Explicit Lengths: Support in-place filtered matrices
///
/// Key Innovation: Unified API
///
/// All CSR-like matrices provide:
/// - row_values(i): Get data span for row i
/// - row_indices(i): Get index span for row i
/// - row_length(i): Get valid length for row i
///
/// All CSC-like matrices provide:
/// - col_values(j): Get data span for column j
/// - col_indices(j): Get index span for column j  
/// - col_length(j): Get valid length for column j
///
/// This enables seamless switching between Standard and Virtual matrices.
// =============================================================================

namespace scl {

// =============================================================================
// Type Tags (Compile-Time Dispatch)
// =============================================================================

struct TagDense {};
struct TagCSR {};
struct TagCSC {};

// =============================================================================
// Matrix Concepts (C++20 Interface Contracts)
// =============================================================================

/// @brief Concept for CSR-like matrices.
template <typename M>
concept CSRLike = requires(const M& m, Index i) {
    typename M::ValueType;
    typename M::Tag;
    requires std::is_same_v<typename M::Tag, TagCSR>;
    { m.rows } -> std::convertible_to<const Index&>;
    { m.cols } -> std::convertible_to<const Index&>;
    { m.row_values(i) } -> std::convertible_to<Span<typename M::ValueType>>;
    { m.row_indices(i) } -> std::convertible_to<Span<Index>>;
    { m.row_length(i) } -> std::convertible_to<Index>;
};

/// @brief Concept for CSC-like matrices.
template <typename M>
concept CSCLike = requires(const M& m, Index j) {
    typename M::ValueType;
    typename M::Tag;
    requires std::is_same_v<typename M::Tag, TagCSC>;
    { m.rows } -> std::convertible_to<const Index&>;
    { m.cols } -> std::convertible_to<const Index&>;
    { m.col_values(j) } -> std::convertible_to<Span<typename M::ValueType>>;
    { m.col_indices(j) } -> std::convertible_to<Span<Index>>;
    { m.col_length(j) } -> std::convertible_to<Index>;
};

/// @brief Concept for Dense-like matrices.
template <typename M>
concept DenseLike = requires(const M& m, Index i, Index j) {
    typename M::ValueType;
    typename M::Tag;
    requires std::is_same_v<typename M::Tag, TagDense>;
    { m.rows } -> std::convertible_to<const Index&>;
    { m.cols } -> std::convertible_to<const Index&>;
    { m.ptr } -> std::convertible_to<typename M::ValueType* const&>;
    { m(i, j) } -> std::convertible_to<typename M::ValueType&>;
};

/// @brief Concept for any sparse matrix.
template <typename M>
concept SparseLike = CSRLike<M> || CSCLike<M>;

/// @brief Concept for matrices with contiguous data buffer.
template <typename M>
concept ContiguousData = requires(const M& m) {
    typename M::ValueType;
    { m.data } -> std::convertible_to<typename M::ValueType* const&>;
    { m.nnz } -> std::convertible_to<const Index&>;
};

// =============================================================================
// Dense Matrix (Row-Major)
// =============================================================================

/// @brief Dense row-major matrix view (NumPy compatible).
template <typename T>
struct DenseMatrix {
    using ValueType = T;
    using Tag = TagDense;

    T* ptr;
    Index rows;
    Index cols;

    constexpr DenseMatrix() noexcept : ptr(nullptr), rows(0), cols(0) {}
    constexpr DenseMatrix(T* p, Index r, Index c) noexcept : ptr(p), rows(r), cols(c) {}

    SCL_NODISCARD SCL_FORCE_INLINE T& operator()(Index r, Index c) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "DenseMatrix: Row out of bounds");
        SCL_ASSERT(c >= 0 && c < cols, "DenseMatrix: Col out of bounds");
#endif
        return ptr[r * cols + c];
    }

    SCL_NODISCARD SCL_FORCE_INLINE Span<T> row(Index r) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "DenseMatrix: Row out of bounds");
#endif
        return Span<T>(ptr + (r * cols), static_cast<Size>(cols));
    }

    SCL_NODISCARD constexpr T* data() const noexcept { return ptr; }
    SCL_NODISCARD constexpr Size size() const noexcept { return static_cast<Size>(rows * cols); }
};

// =============================================================================
// Standard CSR Matrix (Contiguous, SciPy Compatible)
// =============================================================================

/// @brief Compressed Sparse Row matrix (scipy.sparse.csr_matrix).
///
/// Unified Interface: Provides row_length() for consistency with Virtual matrices.
template <typename T>
struct CustomCSR {
    using ValueType = T;
    using Tag = TagCSR;

    T* data;
    Index* indices;
    Index* indptr;
    const Index* row_lengths;  ///< Optional: Explicit lengths (nullptr = use indptr)
    
    Index rows;
    Index cols;
    Index nnz;

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    constexpr CustomCSR() noexcept 
        : data(nullptr), indices(nullptr), indptr(nullptr), row_lengths(nullptr),
          rows(0), cols(0), nnz(0) {}

    constexpr CustomCSR(
        T* d, Index* idx, Index* ptr, Index r, Index c, Index n,
        const Index* lengths = nullptr
    ) noexcept
        : data(d), indices(idx), indptr(ptr), row_lengths(lengths),
          rows(r), cols(c), nnz(n) {}

    // -------------------------------------------------------------------------
    // Unified Interface (matching VirtualCSR)
    // -------------------------------------------------------------------------

    /// @brief Get valid length for row i.
    ///
    /// Returns explicit length if provided, otherwise computes from indptr.
    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "CSR: Row out of bounds");
#endif
        if (row_lengths) {
            return row_lengths[i];
        } else {
            return indptr[i + 1] - indptr[i];
        }
    }

    /// @brief Get values for row i.
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> row_values(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "CSR: Row out of bounds");
#endif
        Index start = indptr[i];
        Size len = static_cast<Size>(row_length(i));
        return Span<T>(data + start, len);
    }

    /// @brief Get column indices for row i.
    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> row_indices(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "CSR: Row out of bounds");
#endif
        Index start = indptr[i];
        Size len = static_cast<Size>(row_length(i));
        return Span<Index>(indices + start, len);
    }
};

// =============================================================================
// Standard CSC Matrix (Contiguous, SciPy Compatible)
// =============================================================================

/// @brief Compressed Sparse Column matrix (scipy.sparse.csc_matrix).
///
/// Unified Interface: Provides col_length() for consistency.
template <typename T>
struct CustomCSC {
    using ValueType = T;
    using Tag = TagCSC;

    T* data;
    Index* indices;
    Index* indptr;
    const Index* col_lengths;  ///< Optional: Explicit lengths
    
    Index rows;
    Index cols;
    Index nnz;

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    constexpr CustomCSC() noexcept 
        : data(nullptr), indices(nullptr), indptr(nullptr), col_lengths(nullptr),
          rows(0), cols(0), nnz(0) {}

    constexpr CustomCSC(
        T* d, Index* idx, Index* ptr, Index r, Index c, Index n,
        const Index* lengths = nullptr
    ) noexcept
        : data(d), indices(idx), indptr(ptr), col_lengths(lengths),
          rows(r), cols(c), nnz(n) {}

    // -------------------------------------------------------------------------
    // Unified Interface
    // -------------------------------------------------------------------------

    /// @brief Get valid length for column j.
    SCL_NODISCARD SCL_FORCE_INLINE Index col_length(Index j) const {
#if !defined(NDEBUG)
        SCL_ASSERT(j >= 0 && j < cols, "CSC: Col out of bounds");
#endif
        if (col_lengths) {
            return col_lengths[j];
        } else {
            return indptr[j + 1] - indptr[j];
        }
    }

    /// @brief Get values for column j.
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> col_values(Index j) const {
#if !defined(NDEBUG)
        SCL_ASSERT(j >= 0 && j < cols, "CSC: Col out of bounds");
#endif
        Index start = indptr[j];
        Size len = static_cast<Size>(col_length(j));
        return Span<T>(data + start, len);
    }

    /// @brief Get row indices for column j.
    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> col_indices(Index j) const {
#if !defined(NDEBUG)
        SCL_ASSERT(j >= 0 && j < cols, "CSC: Col out of bounds");
#endif
        Index start = indptr[j];
        Size len = static_cast<Size>(col_length(j));
        return Span<Index>(indices + start, len);
    }
};

// =============================================================================
// Virtual CSR Matrix (Zero-Copy Slicing + Explicit Lengths)
// =============================================================================

/// @brief Virtual CSR matrix with indirection and explicit length support.
///
/// Features:
/// 1. Indirection: Maps virtual_row → physical_row (zero-copy slicing)
/// 2. Explicit Lengths: Handles in-place filtered matrices
///
/// Performance: +1 L1 cache read per access (<1% overhead).
template <typename T>
struct VirtualCSR {
    using ValueType = T;
    using Tag = TagCSR;

    // Source matrix pointers (borrowed)
    T* src_data;
    Index* src_indices;
    Index* src_indptr;

    const Index* row_map;      ///< Indirection: virtual_row → physical_row
    const Index* src_row_lengths;  ///< Optional: Explicit lengths for physical rows
    
    Index rows;        ///< Virtual row count
    Index cols;        ///< Columns (same as source)
    Index src_rows;    ///< Source row count

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    constexpr VirtualCSR() noexcept
        : src_data(nullptr), src_indices(nullptr), src_indptr(nullptr),
          row_map(nullptr), src_row_lengths(nullptr), 
          rows(0), cols(0), src_rows(0) {}

    /// @brief Construct from source matrix.
    ///
    /// @param source Source CSR matrix
    /// @param map Row indirection array
    VirtualCSR(const CustomCSR<T>& source, Span<const Index> map) noexcept
        : src_data(source.data),
          src_indices(source.indices),
          src_indptr(source.indptr),
          row_map(map.ptr),
          src_row_lengths(source.row_lengths),  // Inherit from source
          rows(static_cast<Index>(map.size)),
          cols(source.cols),
          src_rows(source.rows) {}

    // -------------------------------------------------------------------------
    // Unified Interface (matching CustomCSR)
    // -------------------------------------------------------------------------

    /// @brief Get valid length for virtual row i.
    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "VirtualCSR: Row out of bounds");
#endif
        Index phys_row = row_map[i];
        
#if !defined(NDEBUG)
        SCL_ASSERT(phys_row >= 0 && phys_row < src_rows, 
                   "VirtualCSR: Physical row out of bounds");
#endif
        if (src_row_lengths) {
            return src_row_lengths[phys_row];
        } else {
            return src_indptr[phys_row + 1] - src_indptr[phys_row];
        }
    }

    /// @brief Get values for virtual row i.
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> row_values(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "VirtualCSR: Row out of bounds");
#endif
        Index phys_row = row_map[i];
        
#if !defined(NDEBUG)
        SCL_ASSERT(phys_row >= 0 && phys_row < src_rows, 
                   "VirtualCSR: Physical row out of bounds");
#endif
        Index start = src_indptr[phys_row];
        Size len = static_cast<Size>(row_length(i));  // Reuse row_length
        
        return Span<T>(src_data + start, len);
    }

    /// @brief Get column indices for virtual row i.
    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> row_indices(Index i) const {
#if !defined(NDEBUG)
        SCL_ASSERT(i >= 0 && i < rows, "VirtualCSR: Row out of bounds");
#endif
        Index phys_row = row_map[i];
        
#if !defined(NDEBUG)
        SCL_ASSERT(phys_row >= 0 && phys_row < src_rows,
                   "VirtualCSR: Physical row out of bounds");
#endif
        Index start = src_indptr[phys_row];
        Size len = static_cast<Size>(row_length(i));
        
        return Span<Index>(src_indices + start, len);
    }
};

// =============================================================================
// Virtual CSC Matrix (Zero-Copy Slicing + Explicit Lengths)
// =============================================================================

/// @brief Virtual CSC matrix with indirection and explicit length support.
template <typename T>
struct VirtualCSC {
    using ValueType = T;
    using Tag = TagCSC;

    T* src_data;
    Index* src_indices;
    Index* src_indptr;

    const Index* col_map;      ///< Indirection: virtual_col → physical_col
    const Index* src_col_lengths;  ///< Optional: Explicit lengths for physical cols
    
    Index rows;        ///< Rows (same as source)
    Index cols;        ///< Virtual column count
    Index src_cols;    ///< Source column count

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    constexpr VirtualCSC() noexcept
        : src_data(nullptr), src_indices(nullptr), src_indptr(nullptr),
          col_map(nullptr), src_col_lengths(nullptr), 
          rows(0), cols(0), src_cols(0) {}

    VirtualCSC(const CustomCSC<T>& source, Span<const Index> map) noexcept
        : src_data(source.data),
          src_indices(source.indices),
          src_indptr(source.indptr),
          col_map(map.ptr),
          src_col_lengths(source.col_lengths),  // Inherit from source
          rows(source.rows),
          cols(static_cast<Index>(map.size)),
          src_cols(source.cols) {}

    // -------------------------------------------------------------------------
    // Unified Interface (matching CustomCSC)
    // -------------------------------------------------------------------------

    /// @brief Get valid length for virtual column j.
    SCL_NODISCARD SCL_FORCE_INLINE Index col_length(Index j) const {
#if !defined(NDEBUG)
        SCL_ASSERT(j >= 0 && j < cols, "VirtualCSC: Col out of bounds");
#endif
        Index phys_col = col_map[j];
        
#if !defined(NDEBUG)
        SCL_ASSERT(phys_col >= 0 && phys_col < src_cols,
                   "VirtualCSC: Physical col out of bounds");
#endif
        if (src_col_lengths) {
            return src_col_lengths[phys_col];
        } else {
            return src_indptr[phys_col + 1] - src_indptr[phys_col];
        }
    }

    /// @brief Get values for virtual column j.
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> col_values(Index j) const {
#if !defined(NDEBUG)
        SCL_ASSERT(j >= 0 && j < cols, "VirtualCSC: Col out of bounds");
#endif
        Index phys_col = col_map[j];
        
#if !defined(NDEBUG)
        SCL_ASSERT(phys_col >= 0 && phys_col < src_cols,
                   "VirtualCSC: Physical col out of bounds");
#endif
        Index start = src_indptr[phys_col];
        Size len = static_cast<Size>(col_length(j));
        
        return Span<T>(src_data + start, len);
    }

    /// @brief Get row indices for virtual column j.
    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> col_indices(Index j) const {
#if !defined(NDEBUG)
        SCL_ASSERT(j >= 0 && j < cols, "VirtualCSC: Col out of bounds");
#endif
        Index phys_col = col_map[j];
        
#if !defined(NDEBUG)
        SCL_ASSERT(phys_col >= 0 && phys_col < src_cols,
                   "VirtualCSC: Physical col out of bounds");
#endif
        Index start = src_indptr[phys_col];
        Size len = static_cast<Size>(col_length(j));
        
        return Span<Index>(src_indices + start, len);
    }
};

// =============================================================================
// Backward Compatibility Aliases
// =============================================================================

/// @brief Standard CSR matrix (backward compatibility).
template <typename T>
using CSRMatrix = CustomCSR<T>;

/// @brief Standard CSC matrix (backward compatibility).
template <typename T>
using CSCMatrix = CustomCSC<T>;

/// @brief Standard dense matrix of reals.
using RealMatrix = DenseMatrix<Real>;

/// @brief Standard sparse matrix of reals (CSR).
using SparseMatrix = CSRMatrix<Real>;

/// @brief Matrix of indices (dense).
using IndexMatrix = DenseMatrix<Index>;

} // namespace scl
