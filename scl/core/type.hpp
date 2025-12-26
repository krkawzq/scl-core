#pragma once

#include "scl/config.hpp"
#include "scl/core/macros.hpp"
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <concepts>
#include <cassert>

// =============================================================================
/// @file type.hpp
/// @brief SCL Unified Type System
///
/// This is the ONLY file that defines types and concepts for the entire library.
///
/// Contents:
/// - Basic types: Real, Index, Size, Byte, Pointer
/// - View type: Array<T> (lightweight 1D view)
/// - Concepts: ArrayLike, DenseLike, SparseLike, AnySparse
/// - Tags: TagDense, TagSparse<IsCSR>
/// - Unified accessors: rows(), cols(), nnz(), primary_*()
///
/// Design Philosophy:
/// 1. Zero Overhead: All abstractions compile away
/// 2. Const-Correctness: Array<T> vs Array<const T>
/// 3. No Memory Management: Views are non-owning
/// 4. Unified Abstraction: CSR/CSC share implementations
/// 5. Force Inline: All accessor methods are force-inlined
// =============================================================================

namespace scl {

// =============================================================================
// SECTION 1: Basic Types
// =============================================================================

// Floating-point precision
#if defined(SCL_USE_FLOAT32)
    using Real = float;
    constexpr int DTYPE_CODE = 0;
    constexpr const char* DTYPE_NAME = "float32";
#elif defined(SCL_USE_FLOAT64)
    using Real = double;
    constexpr int DTYPE_CODE = 1;
    constexpr const char* DTYPE_NAME = "float64";
#elif defined(SCL_USE_FLOAT16)
    #if defined(__FLT16_MANT_DIG__) || \
        (defined(__clang__) && __clang_major__ >= 15) || \
        (defined(__GNUC__) && __GNUC__ >= 12)
        using Real = _Float16;
    #else
        #error "SCL: _Float16 unsupported. Requires GCC ≥12 or Clang ≥15."
    #endif
    constexpr int DTYPE_CODE = 2;
    constexpr const char* DTYPE_NAME = "float16";
#else
    #error "SCL: No precision macro defined."
#endif

// Integer types
#if defined(SCL_USE_INT16)
    using Index = std::int16_t;
    constexpr int INDEX_DTYPE_CODE = 0;
    constexpr const char* INDEX_DTYPE_NAME = "int16";
#elif defined(SCL_USE_INT32)
    using Index = std::int32_t;
    constexpr int INDEX_DTYPE_CODE = 1;
    constexpr const char* INDEX_DTYPE_NAME = "int32";
#elif defined(SCL_USE_INT64)
    using Index = std::int64_t;
    constexpr int INDEX_DTYPE_CODE = 2;
    constexpr const char* INDEX_DTYPE_NAME = "int64";
#else
    #error "SCL: No index precision selected."
#endif

/// @brief Unsigned size type for memory operations
using Size = std::size_t;

/// @brief Byte type for raw memory
using Byte = std::uint8_t;

/// @brief Generic pointer type for discontiguous storage
using Pointer = void*;

// =============================================================================
// SECTION 2: Array - Lightweight 1D View
// =============================================================================

/// @brief Lightweight, non-owning view of a contiguous 1D array
///
/// Design:
/// - Non-owning: No destructor, no allocation
/// - Zero-overhead: Just pointer + size (16 bytes on 64-bit)
/// - Const-correct: Array<T> (mutable) vs Array<const T> (immutable)
/// - Public members: ptr and size accessible for fast path optimization
///
/// Usage:
///
/// // Generic algorithm using ArrayLike concept
/// template <ArrayLike A>
/// void generic_algo(const A& arr) {
///     for (Index i = 0; i < arr.size(); ++i) {
///         process(arr[i]);
///     }
/// }
///
/// // Optimized algorithm with direct member access
/// void fast_algo(Array<const Real> arr) {
///     const Real* data = arr.ptr;  // Direct access
///     Size n = arr.size;
///     // SIMD operations
/// }
template <typename T>
struct Array {
    using value_type = T;
    
    T* ptr;      ///< Pointer to first element (public for fast path)
    Size len;    ///< Number of elements (public for fast path)

    // Constructors
    constexpr Array() noexcept : ptr(nullptr), len(0) {}
    constexpr Array(T* p, Size s) noexcept : ptr(p), len(s) {}
    
    /// @brief Conversion: Array<T> -> Array<const T>
    template <typename U>
        requires (std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>)
    constexpr Array(const Array<U>& other) noexcept 
        : ptr(other.ptr), len(other.len) {}

    // ArrayLike Interface (force inline)
    SCL_FORCE_INLINE constexpr T& operator[](Index i) const noexcept {
#if !defined(NDEBUG)
        assert(i >= 0 && static_cast<Size>(i) < len && "Array index out of bounds");
#endif
        return ptr[i];
    }
    
    SCL_FORCE_INLINE constexpr T* data() const noexcept { return ptr; }
    SCL_FORCE_INLINE constexpr Size size() const noexcept { return len; }
    SCL_FORCE_INLINE constexpr bool empty() const noexcept { return len == 0; }
    SCL_FORCE_INLINE constexpr T* begin() const noexcept { return ptr; }
    SCL_FORCE_INLINE constexpr T* end() const noexcept { return ptr + len; }
};

// =============================================================================
// SECTION 3: ArrayLike Concept
// =============================================================================

/// @brief Concept for 1D array-like containers
///
/// Requirements:
/// - value_type: Element type alias
/// - data(): Returns pointer to data
/// - size(): Returns number of elements
/// - operator[]: Random access
/// - begin()/end(): Iterator support
///
/// Implementations:
/// - Array<T>
/// - std::vector<T>
/// - std::array<T, N>
/// - std::deque<T>
/// - MappedArray<T>
template <typename A>
concept ArrayLike = requires(const A& a, Index i) {
    typename A::value_type;
    { a.data() } -> std::convertible_to<const typename A::value_type*>;
    { a.size() } -> std::convertible_to<Size>;
    { a[i] } -> std::convertible_to<const typename A::value_type&>;
    { a.begin() };
    { a.end() };
};

// Verify Array satisfies ArrayLike
static_assert(ArrayLike<Array<Real>>);
static_assert(ArrayLike<Array<const Real>>);
static_assert(ArrayLike<Array<Index>>);

// =============================================================================
// SECTION 4: Matrix Tags
// =============================================================================

/// @brief Tag for dense matrices
struct TagDense {};

/// @brief Tag for sparse matrices (CSR or CSC)
template <bool IsCSR>
struct TagSparse {
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;
};

using TagCSR = TagSparse<true>;
using TagCSC = TagSparse<false>;

// Tag traits
template <typename Tag>
struct is_sparse_tag : std::false_type {};

template <bool IsCSR>
struct is_sparse_tag<TagSparse<IsCSR>> : std::true_type {};

template <typename Tag>
inline constexpr bool is_sparse_tag_v = is_sparse_tag<Tag>::value;

template <typename Tag>
struct tag_is_csr;

template <bool IsCSR>
struct tag_is_csr<TagSparse<IsCSR>> : std::bool_constant<IsCSR> {};

template <typename Tag>
inline constexpr bool tag_is_csr_v = tag_is_csr<Tag>::value;

// =============================================================================
// SECTION 5: Unified Accessors (POD + Virtual Support)
// =============================================================================

namespace detail {
    template <typename T> 
    concept HasRowMember = requires(const T& t) { 
        { t.rows } -> std::convertible_to<Index>; 
    };
    
    template <typename T> 
    concept HasColMember = requires(const T& t) { 
        { t.cols } -> std::convertible_to<Index>; 
    };
    
    template <typename T> 
    concept HasNnzMember = requires(const T& t) { 
        { t.nnz } -> std::convertible_to<Index>; 
    };
    
    template <typename T> 
    concept HasRowMethod = requires(const T& t) { 
        { t.rows() } -> std::convertible_to<Index>; 
    };
    
    template <typename T> 
    concept HasColMethod = requires(const T& t) { 
        { t.cols() } -> std::convertible_to<Index>; 
    };
    
    template <typename T> 
    concept HasNnzMethod = requires(const T& t) { 
        { t.nnz() } -> std::convertible_to<Index>; 
    };
}

/// @brief Unified accessor for row count
template <typename M>
SCL_NODISCARD SCL_FORCE_INLINE constexpr Index rows(const M& m) {
    if constexpr (detail::HasRowMember<M>) {
        return m.rows;
    } else {
        return m.rows();
    }
}

/// @brief Unified accessor for column count
template <typename M>
SCL_NODISCARD SCL_FORCE_INLINE constexpr Index cols(const M& m) {
    if constexpr (detail::HasColMember<M>) {
        return m.cols;
    } else {
        return m.cols();
    }
}

/// @brief Unified accessor for non-zero count
template <typename M>
SCL_NODISCARD SCL_FORCE_INLINE constexpr Index nnz(const M& m) {
    if constexpr (detail::HasNnzMember<M>) {
        return m.nnz;
    } else {
        return m.nnz();
    }
}

/// @brief Unified accessor for data pointer (dense matrices)
template <typename M>
SCL_NODISCARD SCL_FORCE_INLINE constexpr const typename M::ValueType* ptr(const M& m) {
    if constexpr (requires { { m.ptr } -> std::convertible_to<const typename M::ValueType*>; }) {
        return m.ptr;
    } else {
        return m.data();
    }
}

// =============================================================================
// SECTION 6: Matrix Concepts
// =============================================================================

// -----------------------------------------------------------------------------
// 6.1 DenseLike Concept
// -----------------------------------------------------------------------------

/// @brief Concept for 2D dense matrices
template <typename M>
concept DenseLike = requires(const M& m, Index i, Index j) {
    typename M::ValueType;
    typename M::Tag;
    requires std::is_same_v<typename M::Tag, TagDense>;
    
    { scl::rows(m) } -> std::convertible_to<Index>;
    { scl::cols(m) } -> std::convertible_to<Index>;
    { scl::ptr(m) } -> std::convertible_to<const typename M::ValueType*>;
    { m(i, j) } -> std::convertible_to<typename M::ValueType>;
};

// -----------------------------------------------------------------------------
// 6.2 SparseLike Concept
// -----------------------------------------------------------------------------

/// @brief Unified concept for sparse matrices (CSR or CSC)
template <bool IsCSR>
struct SparseLikeConcept {
    template <typename M>
    static constexpr bool value = requires(const M& m, Index i) {
        typename M::ValueType;
        typename M::Tag;
        requires is_sparse_tag_v<typename M::Tag>;
        requires tag_is_csr_v<typename M::Tag> == IsCSR;
        
        { scl::rows(m) } -> std::convertible_to<Index>;
        { scl::cols(m) } -> std::convertible_to<Index>;
        { scl::nnz(m) } -> std::convertible_to<Index>;
    } && (
        (IsCSR && requires(const M& m, Index i) {
            { m.row_values(i) } -> std::convertible_to<Array<typename M::ValueType>>;
            { m.row_indices(i) } -> std::convertible_to<Array<Index>>;
            { m.row_length(i) } -> std::convertible_to<Index>;
        })
        ||
        (!IsCSR && requires(const M& m, Index j) {
            { m.col_values(j) } -> std::convertible_to<Array<typename M::ValueType>>;
            { m.col_indices(j) } -> std::convertible_to<Array<Index>>;
            { m.col_length(j) } -> std::convertible_to<Index>;
        })
    );
};

template <typename M, bool IsCSR>
concept SparseLike = SparseLikeConcept<IsCSR>::template value<M>;

/// @brief Any sparse matrix (CSR or CSC)
template <typename M>
concept AnySparse = SparseLike<M, true> || SparseLike<M, false>;

/// @brief CSR-specific matrices
template <typename M>
concept CSRLike = SparseLike<M, true>;

/// @brief CSC-specific matrices
template <typename M>
concept CSCLike = SparseLike<M, false>;

// -----------------------------------------------------------------------------
// 6.3 Storage Pattern Concepts
// -----------------------------------------------------------------------------

/// @brief Custom pattern: contiguous arrays with indptr
template <typename M, bool IsCSR>
concept CustomSparseLike = SparseLike<M, IsCSR> && requires(const M& m) {
    { m.data } -> std::convertible_to<typename M::ValueType*>;
    { m.indices } -> std::convertible_to<Index*>;
    { m.indptr } -> std::convertible_to<Index*>;
};

/// @brief Virtual pattern: pointer arrays with explicit nnz
template <typename M, bool IsCSR>
concept VirtualSparseLike = SparseLike<M, IsCSR> && 
    requires(const M& m) {
        { m.data_ptrs } -> std::convertible_to<Pointer*>;
        { m.indices_ptrs } -> std::convertible_to<Pointer*>;
        { m.lengths } -> std::convertible_to<Index*>;
    } &&
    !requires(const M& m) {
        { m.indptr } -> std::convertible_to<Index*>;
    };

// Convenience aliases
template <typename M>
concept CustomCSRLike = CustomSparseLike<M, true>;

template <typename M>
concept VirtualCSRLike = VirtualSparseLike<M, true>;

template <typename M>
concept CustomCSCLike = CustomSparseLike<M, false>;

template <typename M>
concept VirtualCSCLike = VirtualSparseLike<M, false>;

// =============================================================================
// SECTION 7: Unified Accessors for Sparse Matrices
// =============================================================================

/// @brief Get primary dimension size (CSR: rows, CSC: cols)
template <typename M>
    requires (AnySparse<M>)
SCL_NODISCARD SCL_FORCE_INLINE Index primary_size(const M& mat) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return scl::rows(mat);
    } else {
        return scl::cols(mat);
    }
}

/// @brief Get secondary dimension size (CSR: cols, CSC: rows)
template <typename M>
    requires (AnySparse<M>)
SCL_NODISCARD SCL_FORCE_INLINE Index secondary_size(const M& mat) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return scl::cols(mat);
    } else {
        return scl::rows(mat);
    }
}

/// @brief Get values for primary dimension i
template <typename M>
    requires (AnySparse<M>)
SCL_NODISCARD SCL_FORCE_INLINE auto primary_values(const M& mat, Index i) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return mat.row_values(i);
    } else {
        return mat.col_values(i);
    }
}

/// @brief Get indices for primary dimension i
template <typename M>
    requires (AnySparse<M>)
SCL_NODISCARD SCL_FORCE_INLINE auto primary_indices(const M& mat, Index i) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return mat.row_indices(i);
    } else {
        return mat.col_indices(i);
    }
}

/// @brief Get length for primary dimension i
template <typename M>
    requires (AnySparse<M>)
SCL_NODISCARD SCL_FORCE_INLINE Index primary_length(const M& mat, Index i) {
    if constexpr (tag_is_csr_v<typename M::Tag>) {
        return mat.row_length(i);
    } else {
        return mat.col_length(i);
    }
}

// =============================================================================
// SECTION 8: Traits and Type Checks
// =============================================================================

/// @brief Sparse matrix traits
template <typename M>
    requires (AnySparse<M>)
struct sparse_traits {
    static constexpr bool is_csr = tag_is_csr_v<typename M::Tag>;
    static constexpr bool is_csc = !is_csr;
};

/// @brief Check if matrix has contiguous storage
template <typename M>
inline constexpr bool has_contiguous_storage_v = 
    (AnySparse<M> && CustomSparseLike<M, true>) ||
    (AnySparse<M> && CustomSparseLike<M, false>);

/// @brief Check if matrix uses indirection
template <typename M>
inline constexpr bool has_indirection_v = 
    (AnySparse<M> && VirtualSparseLike<M, true>) ||
    (AnySparse<M> && VirtualSparseLike<M, false>);

// Type check helpers
template <typename T>
inline constexpr bool is_csr_like_v = CSRLike<T>;

template <typename T>
inline constexpr bool is_csc_like_v = CSCLike<T>;

template <typename T>
inline constexpr bool is_any_sparse_v = AnySparse<T>;

template <typename T>
inline constexpr bool is_dense_like_v = DenseLike<T>;

template <typename T>
inline constexpr bool is_custom_csr_v = CustomCSRLike<T>;

template <typename T>
inline constexpr bool is_virtual_csr_v = VirtualCSRLike<T>;

} // namespace scl
