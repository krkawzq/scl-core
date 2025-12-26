#pragma once

#include "scl/core/matrix.hpp"
#include <deque>

// =============================================================================
/// @file dense.hpp
/// @brief Dense Matrix Concrete Implementations
///
/// Contains dense matrix types with different storage strategies:
///
/// Contiguous Variants:
/// - DenseArray<T>: Standard row-major array (most common)
///
/// Discontiguous Variants:
/// - DenseDeque<T>: Deque-based storage (for incremental construction)
///
/// Design Principles:
///
/// 1. Pure Data Classes: All members public, direct access
/// 2. No Memory Management: User owns memory lifecycle
/// 3. Unified Accessors: Work with scl::rows(), scl::cols(), scl::ptr()
/// 4. Concept Compliance: All types satisfy DenseLike
///
/// All types are POD-like and FFI-friendly (except DenseDeque which manages deque).
// =============================================================================

namespace scl {

// =============================================================================
// DenseArray: Contiguous Row-Major Storage
// =============================================================================

/// @brief Dense row-major matrix with contiguous array storage.
///
/// Memory Layout: Single contiguous array (rows × cols elements)
/// Indexing: ptr[r * cols + c]
///
/// Ownership: Non-owning view (ptr must outlive this object)
///
/// Implements: DenseLike
///
/// Example:
///
/// std::vector<float> data(100);
/// DenseArray<float> mat(data.data(), 10, 10);
/// mat(0, 0) = 1.0f;  // Direct access
template <typename T>
struct DenseArray {
    using ValueType = T;
    using Tag = TagDense;
    
    // Pure data members (direct access)
    T* ptr;
    Index rows;
    Index cols;
    
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    
    constexpr DenseArray() noexcept : ptr(nullptr), rows(0), cols(0) {}
    
    constexpr DenseArray(T* p, Index r, Index c) noexcept 
        : ptr(p), rows(r), cols(c) {}
    
    // -------------------------------------------------------------------------
    // DenseLike Interface
    // -------------------------------------------------------------------------
    
    /// @brief Element access.
    SCL_NODISCARD SCL_FORCE_INLINE T& operator()(Index r, Index c) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "DenseArray: Row out of bounds");
        SCL_ASSERT(c >= 0 && c < cols, "DenseArray: Col out of bounds");
#endif
        return ptr[r * cols + c];
    }
    
    // -------------------------------------------------------------------------
    // Additional Utilities
    // -------------------------------------------------------------------------
    
    /// @brief Get entire row as span.
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> row(Index r) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "DenseArray: Row out of bounds");
#endif
        return Span<T>(ptr + (r * cols), static_cast<Size>(cols));
    }
    
    /// @brief Get column (non-contiguous, requires copy).
    void col(Index c, T* output) const {
#if !defined(NDEBUG)
        SCL_ASSERT(c >= 0 && c < cols, "DenseArray: Col out of bounds");
#endif
        for (Index r = 0; r < rows; ++r) {
            output[r] = ptr[r * cols + c];
        }
    }
    
    /// @brief Get raw data pointer.
    SCL_NODISCARD constexpr T* data() const noexcept { return ptr; }
    
    /// @brief Get total element count.
    SCL_NODISCARD constexpr Size size() const noexcept { 
        return static_cast<Size>(rows) * static_cast<Size>(cols); 
    }
    
    /// @brief Fill with constant value.
    void fill(T value) const {
        Size n = size();
        for (Size i = 0; i < n; ++i) {
            ptr[i] = value;
        }
    }
};

// =============================================================================
// DenseDeque: Discontiguous Row-Based Storage
// =============================================================================

/// @brief Dense matrix with row-pointer array storage (discontiguous).
///
/// Memory Layout: Array of row pointers (each row at arbitrary location)
/// Indexing: row_ptrs[r][c]
///
/// Design: Pure data class
/// - User provides array of row pointers
/// - User manages memory lifecycle
/// - No internal allocation/deallocation
///
/// Use Cases:
/// - Interface to external deque storage
/// - Wrapper for jagged arrays
/// - Memory-pool allocated rows
///
/// Implements: DenseLike
///
/// Example:
///
/// // User manages storage
/// std::vector<std::vector<float>> storage = {{1,2}, {3,4}};
/// 
/// // Create pointer array (user responsibility)
/// std::vector<float*> row_ptrs = {
///     storage[0].data(),
///     storage[1].data()
/// };
///
/// // Pure data view (no allocation)
/// DenseDeque<float> mat(row_ptrs.data(), 2, 2);
template <typename T>
struct DenseDeque {
    using ValueType = T;
    using Tag = TagDense;
    
    // Pure data members (NO private state)
    T** row_ptrs;  ///< Array of pointers to rows (user-provided)
    T* ptr;        ///< Always nullptr (signals discontiguous)
    Index rows;
    Index cols;
    
    // -------------------------------------------------------------------------
    // Constructors (NO memory allocation)
    // -------------------------------------------------------------------------
    
    constexpr DenseDeque() noexcept 
        : row_ptrs(nullptr), ptr(nullptr), rows(0), cols(0) {}
    
    constexpr DenseDeque(T** ptrs, Index r, Index c) noexcept
        : row_ptrs(ptrs), ptr(nullptr), rows(r), cols(c) {}
    
    // -------------------------------------------------------------------------
    // DenseLike Interface
    // -------------------------------------------------------------------------
    
    /// @brief Element access (via row pointers).
    SCL_NODISCARD SCL_FORCE_INLINE T& operator()(Index r, Index c) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "DenseDeque: Row out of bounds");
        SCL_ASSERT(c >= 0 && c < cols, "DenseDeque: Col out of bounds");
        SCL_ASSERT(row_ptrs != nullptr, "DenseDeque: row_ptrs is null");
        SCL_ASSERT(row_ptrs[r] != nullptr, "DenseDeque: row pointer is null");
#endif
        return row_ptrs[r][c];
    }
    
    /// @brief Data pointer (returns nullptr, signals discontiguous).
    SCL_NODISCARD constexpr T* data() const noexcept { return nullptr; }
    
    // -------------------------------------------------------------------------
    // Row Access
    // -------------------------------------------------------------------------
    
    /// @brief Get pointer to row.
    ///
    /// Returns raw pointer to row data (user-managed memory).
    SCL_NODISCARD SCL_FORCE_INLINE T* row_ptr(Index r) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "DenseDeque: Row out of bounds");
#endif
        return row_ptrs[r];
    }
    
    /// @brief Get row as span.
    ///
    /// Assumes each row is contiguous (user responsibility).
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> row(Index r) const {
#if !defined(NDEBUG)
        SCL_ASSERT(r >= 0 && r < rows, "DenseDeque: Row out of bounds");
#endif
        return Span<T>(row_ptrs[r], static_cast<Size>(cols));
    }
};

// =============================================================================
// Type Aliases (Backward Compatibility)
// =============================================================================

/// @brief Standard dense matrix (backward compatibility).
template <typename T>
using DenseMatrix = DenseArray<T>;

/// @brief Dense matrix of reals.
using RealMatrix = DenseArray<Real>;

/// @brief Matrix of indices.
using IndexMatrix = DenseArray<Index>;

} // namespace scl
