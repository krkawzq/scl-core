#pragma once

#include "scl/config.hpp"
#include "scl/core/macros.hpp"
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <concepts>
#include <cassert>

// =============================================================================
// FILE: scl/core/type.hpp
// BRIEF: Unified type system and zero-overhead views
// =============================================================================

namespace scl {

// =============================================================================
// SECTION 1: Basic Types
// =============================================================================

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

using Size = std::size_t;
using Byte = std::uint8_t;
using Pointer = void*;

// =============================================================================
// SECTION 2: Array View
// =============================================================================

template <typename T>
struct Array {
    using value_type = T;
    
    T* ptr;
    Size len;

    constexpr Array() noexcept : ptr(nullptr), len(0) {}
    constexpr Array(T* p, Size s) noexcept : ptr(p), len(s) {}
    
    template <typename U>
        requires (std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>)
    constexpr Array(const Array<U>& other) noexcept 
        : ptr(other.ptr), len(other.len) {}

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

template <typename A>
concept ArrayLike = requires(const A& a, Index i) {
    typename A::value_type;
    { a.size() } -> std::convertible_to<Size>;
    { a[i] } -> std::convertible_to<const typename A::value_type&>;
    { a.begin() };
    { a.end() };
};

static_assert(ArrayLike<Array<Real>>);
static_assert(ArrayLike<Array<const Real>>);
static_assert(ArrayLike<Array<Index>>);

// =============================================================================
// SECTION 4: Sparse Matrix Tags and Concepts
// =============================================================================

template <bool IsCSR>
struct TagSparse {
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;
};

template <typename M>
concept CSRLike = requires(const M& m, Index i) {
    typename M::ValueType;
    typename M::Tag;
    requires M::is_csr == true;
    { m.rows() } -> std::convertible_to<Index>;
    { m.cols() } -> std::convertible_to<Index>;
    { m.nnz() } -> std::convertible_to<Index>;
    { m.row_values(i) } -> std::convertible_to<Array<typename M::ValueType>>;
    { m.row_indices(i) } -> std::convertible_to<Array<Index>>;
    { m.row_length(i) } -> std::convertible_to<Index>;
};

template <typename M>
concept CSCLike = requires(const M& m, Index j) {
    typename M::ValueType;
    typename M::Tag;
    requires M::is_csc == true;
    { m.rows() } -> std::convertible_to<Index>;
    { m.cols() } -> std::convertible_to<Index>;
    { m.nnz() } -> std::convertible_to<Index>;
    { m.col_values(j) } -> std::convertible_to<Array<typename M::ValueType>>;
    { m.col_indices(j) } -> std::convertible_to<Array<Index>>;
    { m.col_length(j) } -> std::convertible_to<Index>;
};

template <typename M>
concept SparseLike = CSRLike<M> || CSCLike<M>;

} // namespace scl
