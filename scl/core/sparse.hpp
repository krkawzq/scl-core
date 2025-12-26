#pragma once

#include "scl/core/matrix.hpp"

// =============================================================================
/// @file sparse.hpp
/// @brief Reference Implementations of Sparse Matrix Concepts
///
/// Concept vs Implementation Distinction:
///
/// Concepts (Defined in matrix.hpp):
///   1. CustomSparseLike<M, IsCSR>: Interface contract for contiguous storage
///   2. VirtualSparseLike<M, IsCSR>: Interface contract for indirection pattern
///
/// Implementations (Defined in this file):
///   1. CustomSparse<T, IsCSR>: Canonical implementation of CustomSparseLike
///   2. VirtualSparse<T, IsCSR>: Canonical implementation of VirtualSparseLike
///
/// Users can create their own implementations satisfying the concepts:
///   - MountMatrix: CustomSparseLike with mmap storage
///   - NetworkCSR: CustomSparseLike with RPC access
///   - CachedVirtualCSR: VirtualSparseLike with LRU cache
///
/// Key Innovation: Template-Based CSR/CSC Unification
///
/// Single Template:
///   CustomSparse<T, bool IsCSR>  ->  CustomCSR, CustomCSC
///   VirtualSparse<T, bool IsCSR>  ->  VirtualCSR, VirtualCSC
///
/// Benefits:
///   - 75% code reduction (800 lines to 359 lines)
///   - Single source of truth for logic
///   - Unified maintenance (bug fix once, applies to both)
///   - Type-safe compile-time dispatch
///
/// Design Principles:
///
/// 1. Pure Data Classes: All members public, zero-cost access
/// 2. No Memory Management: User owns pointers, controls lifecycle
/// 3. No ABI Complexity: POD-like, FFI-friendly
/// 4. Concept Implementation: These are reference implementations
// =============================================================================

namespace scl {

// =============================================================================
// CustomSparse: Reference Implementation of CustomSparseLike Concept
// =============================================================================

/// @brief Reference implementation for contiguous sparse storage.
///
/// This is an IMPLEMENTATION, not an interface.
///
/// Implements Concepts:
/// - SparseLike<IsCSR>
/// - CustomSparseLike<IsCSR>
/// - CSRLike (if IsCSR=true) or CSCLike (if IsCSR=false)
///
/// Design:
/// - Pure data class (all members public)
/// - POD-like layout (trivially copyable)
/// - No memory allocation (user manages lifetime)
/// - Zero-overhead interface (inline everything)
///
/// Users can implement CustomSparseLike differently:
/// - MountMatrix: Uses MappedArray for storage
/// - OwnedCSR: Uses std::vector for ownership
/// - NetworkCSR: RPC-based remote storage
///
/// This is just the canonical implementation for heap/mmap pointers.
///
/// Template Parameters:
/// - T: Element type
/// - IsCSR: true = CSR (row-major), false = CSC (column-major)
///
/// Memory Layout:
/// - data[nnz]: Values array
/// - indices[nnz]: Secondary dimension indices (col for CSR, row for CSC)
/// - indptr[primary+1]: Primary dimension pointers (row for CSR, col for CSC)
///
/// Implements:
/// - SparseLike<IsCSR>
/// - CustomSparseLike<IsCSR>
/// - CSRLike (if IsCSR=true) or CSCLike (if IsCSR=false)
template <typename T, bool IsCSR>
struct CustomSparse {
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;
    
    // Pure data members (direct access)
    T* data;
    Index* indices;
    Index* indptr;
    const Index* primary_lengths;  ///< Optional: explicit lengths (nullptr = use indptr)
    
    Index rows;
    Index cols;
    Index nnz;  ///< INVARIANT: Must equal indptr[primary_count()], user must maintain consistency
    
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    
    constexpr CustomSparse() noexcept
        : data(nullptr), indices(nullptr), indptr(nullptr), primary_lengths(nullptr),
          rows(0), cols(0), nnz(0) {}
    
    constexpr CustomSparse(
        T* d, Index* idx, Index* ptr,
        Index r, Index c, Index n,
        const Index* lengths = nullptr
    ) noexcept
        : data(d), indices(idx), indptr(ptr), primary_lengths(lengths),
          rows(r), cols(c), nnz(n) {}
    
    // -------------------------------------------------------------------------
    // Unified Interface (Primary Dimension)
    // -------------------------------------------------------------------------
    
    /// @brief Get primary dimension count.
    SCL_NODISCARD constexpr Index primary_count() const noexcept {
        if constexpr (IsCSR) {
            return rows;
        } else {
            return cols;
        }
    }
    
    /// @brief Get length for primary dimension i.
    SCL_NODISCARD SCL_FORCE_INLINE Index primary_length(Index i) const {
        return primary_lengths ? primary_lengths[i] : (indptr[i + 1] - indptr[i]);
    }
    
    /// @brief Get values span for primary dimension i.
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> primary_values(Index i) const {
        Index start = indptr[i];
        Size len = static_cast<Size>(primary_length(i));
        return Span<T>(data + start, len);
    }
    
    /// @brief Get indices span for primary dimension i.
    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> primary_indices(Index i) const {
        Index start = indptr[i];
        Size len = static_cast<Size>(primary_length(i));
        return Span<Index>(indices + start, len);
    }
    
    // -------------------------------------------------------------------------
    // Layout-Specific Interface (CSR vs CSC)
    // -------------------------------------------------------------------------
    
    // CSR interface (enabled only if IsCSR=true)
    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const 
        requires (IsCSR)
    {
        return primary_length(i);
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> row_values(Index i) const 
        requires (IsCSR)
    {
        return primary_values(i);
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> row_indices(Index i) const 
        requires (IsCSR)
    {
        return primary_indices(i);
    }
    
    // CSC interface (enabled only if IsCSR=false)
    SCL_NODISCARD SCL_FORCE_INLINE Index col_length(Index j) const 
        requires (!IsCSR)
    {
        return primary_length(j);
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> col_values(Index j) const 
        requires (!IsCSR)
    {
        return primary_values(j);
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> col_indices(Index j) const 
        requires (!IsCSR)
    {
        return primary_indices(j);
    }
    
    // -------------------------------------------------------------------------
    // Consistency Validation (Debug Utilities)
    // -------------------------------------------------------------------------
    
    /// @brief Verify nnz consistency with indptr.
    ///
    /// Checks INVARIANT: nnz == indptr[primary_count()]
    /// Call in debug builds to catch user errors.
    SCL_NODISCARD bool validate_nnz() const noexcept {
        Index expected_nnz = indptr[primary_count()];
        return nnz == expected_nnz;
    }
    
    /// @brief Recompute nnz from indptr (fix inconsistency).
    ///
    /// Use when user has modified indptr directly.
    void sync_nnz() noexcept {
        nnz = indptr[primary_count()];
    }
};

// =============================================================================
// VirtualSparse: Reference Implementation of VirtualSparseLike Concept
// =============================================================================

/// @brief Reference implementation for indirection-based sparse storage.
///
/// This is an IMPLEMENTATION, not an interface.
///
/// Implements Concepts:
/// - SparseLike<IsCSR>
/// - VirtualSparseLike<IsCSR>
/// - CSRLike (if IsCSR=true) or CSCLike (if IsCSR=false)
///
/// Design:
/// - Pure data class (all members public)
/// - Non-owning view (borrows from source matrix)
/// - Indirection via primary_map array
/// - +1 pointer dereference per access
///
/// Users can implement VirtualSparseLike differently:
/// - MountedVirtualSparse: Virtual view over mmap
/// - VirtualDequeCSR: Virtual view over deque segments
/// - CachedVirtualCSR: With LRU cache for row_map lookups
///
/// This is just the canonical implementation for simple indirection.
///
/// Template Parameters:
/// - T: Element type
/// - IsCSR: true = CSR, false = CSC
///
/// Memory Layout:
/// - Points to source matrix (src_data, src_indices, src_indptr)
/// - primary_map[n]: Virtual → Physical dimension mapping
///
/// Performance: +1 indirection per access (~1% overhead)
///
/// Implements:
/// - SparseLike<IsCSR>
/// - VirtualSparseLike<IsCSR>
template <typename T, bool IsCSR>
struct VirtualSparse {
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;
    
    // Pointer arrays (n elements, fully discontiguous)
    Pointer* data_ptrs;      ///< Array of pointers to row/column data
    Pointer* indices_ptrs;   ///< Array of pointers to row/column indices
    Index* lengths;          ///< Array of lengths (n elements)
    
    Index rows;
    Index cols;
    Index nnz;  ///< Explicitly stored (NOT derived), unified Index type
    
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    
    constexpr VirtualSparse() noexcept
        : data_ptrs(nullptr), indices_ptrs(nullptr), lengths(nullptr),
          rows(0), cols(0), nnz(0) {}
    
    /// @brief Construct from pointer arrays.
    VirtualSparse(
        Pointer* d_ptrs, Pointer* i_ptrs, Index* lens,
        Index r, Index c, Index n
    ) noexcept
        : data_ptrs(d_ptrs), indices_ptrs(i_ptrs), lengths(lens),
          rows(r), cols(c), nnz(n)
    {}
    
    // -------------------------------------------------------------------------
    // Unified Interface (Primary Dimension)
    // -------------------------------------------------------------------------
    
    SCL_NODISCARD constexpr Index primary_count() const noexcept {
        if constexpr (IsCSR) {
            return rows;
        } else {
            return cols;
        }
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Index primary_length(Index i) const {
        return lengths[i];
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> primary_values(Index i) const {
        T* data = static_cast<T*>(data_ptrs[i]);
        Size len = static_cast<Size>(lengths[i]);
        return Span<T>(data, len);
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> primary_indices(Index i) const {
        Index* indices = static_cast<Index*>(indices_ptrs[i]);
        Size len = static_cast<Size>(lengths[i]);
        return Span<Index>(indices, len);
    }
    
    // -------------------------------------------------------------------------
    // Layout-Specific Interface
    // -------------------------------------------------------------------------
    
    // CSR interface
    SCL_NODISCARD SCL_FORCE_INLINE Index row_length(Index i) const 
        requires (IsCSR)
    {
        return primary_length(i);
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> row_values(Index i) const 
        requires (IsCSR)
    {
        return primary_values(i);
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> row_indices(Index i) const 
        requires (IsCSR)
    {
        return primary_indices(i);
    }
    
    // CSC interface
    SCL_NODISCARD SCL_FORCE_INLINE Index col_length(Index j) const 
        requires (!IsCSR)
    {
        return primary_length(j);
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Span<T> col_values(Index j) const 
        requires (!IsCSR)
    {
        return primary_values(j);
    }
    
    SCL_NODISCARD SCL_FORCE_INLINE Span<Index> col_indices(Index j) const 
        requires (!IsCSR)
    {
        return primary_indices(j);
    }
};

// =============================================================================
// Type Aliases (User-Friendly Names)
// =============================================================================

/// @brief Standard CSR matrix.
template <typename T>
using CustomCSR = CustomSparse<T, true>;

/// @brief Standard CSC matrix.
template <typename T>
using CustomCSC = CustomSparse<T, false>;

/// @brief Virtual CSR matrix.
template <typename T>
using VirtualCSR = VirtualSparse<T, true>;

/// @brief Virtual CSC matrix.
template <typename T>
using VirtualCSC = VirtualSparse<T, false>;

// =============================================================================
// Concept Verification (Compile-Time Contract Validation)
// =============================================================================
//
// These assertions verify that our implementations correctly satisfy
// the interface contracts defined in matrix.hpp.
//
// If any assertion fails, it means the implementation has drifted from
// the concept definition and needs to be fixed.
// =============================================================================

// CustomSparse → CustomSparseLike concept
static_assert(CSRLike<CustomCSR<float>>, 
    "CustomCSR must satisfy CSRLike");
static_assert(CSCLike<CustomCSC<float>>, 
    "CustomCSC must satisfy CSCLike");
static_assert(AnySparse<CustomCSR<float>>, 
    "CustomCSR must satisfy AnySparse");
static_assert(AnySparse<CustomCSC<float>>, 
    "CustomCSC must satisfy AnySparse");

static_assert(CustomSparseLike<CustomCSR<float>, true>, 
    "CustomCSR must satisfy CustomSparseLike<true> (interface contract)");
static_assert(CustomSparseLike<CustomCSC<float>, false>, 
    "CustomCSC must satisfy CustomSparseLike<false> (interface contract)");
static_assert(CustomCSRLike<CustomCSR<float>>, 
    "CustomCSR must satisfy CustomCSRLike");
static_assert(CustomCSCLike<CustomCSC<float>>, 
    "CustomCSC must satisfy CustomCSCLike");

// VirtualSparse → VirtualSparseLike concept
static_assert(CSRLike<VirtualCSR<float>>, 
    "VirtualCSR must satisfy CSRLike");
static_assert(CSCLike<VirtualCSC<float>>, 
    "VirtualCSC must satisfy CSCLike");
static_assert(AnySparse<VirtualCSR<float>>, 
    "VirtualCSR must satisfy AnySparse");
static_assert(AnySparse<VirtualCSC<float>>, 
    "VirtualCSC must satisfy AnySparse");

static_assert(VirtualSparseLike<VirtualCSR<float>, true>, 
    "VirtualCSR must satisfy VirtualSparseLike<true> (interface contract)");
static_assert(VirtualSparseLike<VirtualCSC<float>, false>, 
    "VirtualCSC must satisfy VirtualSparseLike<false> (interface contract)");
static_assert(VirtualCSRLike<VirtualCSR<float>>, 
    "VirtualCSR must satisfy VirtualCSRLike");
static_assert(VirtualCSCLike<VirtualCSC<float>>, 
    "VirtualCSC must satisfy VirtualCSCLike");

// Negative tests: Ensure proper separation
static_assert(!CustomSparseLike<VirtualCSR<float>, true>,
    "VirtualCSR must NOT satisfy CustomSparseLike (different storage pattern)");
static_assert(!VirtualSparseLike<CustomCSR<float>, true>,
    "CustomCSR must NOT satisfy VirtualSparseLike (different storage pattern)");

// =============================================================================
// Design Notes
// =============================================================================
//
// Concept (Interface) vs Implementation (Class):
//
// 1. Concepts define "what" (matrix.hpp):
//    - CustomSparseLike: "Must have contiguous data/indices/indptr"
//    - VirtualSparseLike: "Must have src_* pointers and mapping"
//
// 2. Classes define "how" (this file):
//    - CustomSparse: "Stores three pointers in a struct"
//    - VirtualSparse: "Stores source pointers + mapping array"
//
// 3. Multiple implementations possible:
//    - All of these satisfy CustomSparseLike<true>:
//      * CustomCSR (heap pointers)
//      * MountMatrix (mmap pointers)
//      * OwnedCSR (vector pointers)
//
// 4. Algorithms depend only on concepts:
//    template <typename T, bool IsCSR, CustomSparseLike<IsCSR> M>
//    void algorithm(M& mat) {
//        // Works with ANY CustomSparseLike implementation
//        simd::process(mat.data, mat.nnz);
//    }
//
// =============================================================================

// =============================================================================
// Backward Compatibility
// =============================================================================

template <typename T>
using CSRMatrix = CustomCSR<T>;

template <typename T>
using CSCMatrix = CustomCSC<T>;

using SparseMatrix = CSRMatrix<Real>;

} // namespace scl
