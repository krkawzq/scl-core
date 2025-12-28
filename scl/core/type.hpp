#pragma once

#include "scl/config.hpp"
#include "scl/core/macros.hpp"
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <concepts>
#include <cassert>
#include <iterator>
#include <span>

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
    // Type aliases for STL compatibility
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = Size;
    using difference_type = std::ptrdiff_t;
    using iterator = T*;
    using const_iterator = const T*;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    
    T* ptr;
    Size len;

    // Constructors
    constexpr Array() noexcept : ptr(nullptr), len(0) {}
    constexpr Array(T* p, Size s) noexcept : ptr(p), len(s) {}
    
    // Conversion from non-const to const
    template <typename U>
        requires (std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>)
    constexpr Array(const Array<U>& other) noexcept 
        : ptr(other.ptr), len(other.len) {}
    
    // Conversion from std::span (C++20)
    template <std::size_t Extent = std::dynamic_extent>
    constexpr Array(std::span<T, Extent> span) noexcept
        : ptr(span.data()), len(static_cast<Size>(span.size())) {}

    // Element access
    SCL_FORCE_INLINE constexpr auto operator[](Index i) const noexcept -> T& {
#if !defined(NDEBUG)
        assert(i >= 0 && static_cast<Size>(i) < len && "Array index out of bounds");
#endif
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        // Intentional: zero-overhead array indexing for performance
        return ptr[i];  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    }
    
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto front() const noexcept -> T& {
#if !defined(NDEBUG)
        assert(len > 0 && "Array is empty");
#endif
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        return ptr[0];
    }
    
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto back() const noexcept -> T& {
#if !defined(NDEBUG)
        assert(len > 0 && "Array is empty");
#endif
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        return ptr[len - 1];
    }
    
    // Capacity
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto data() const noexcept -> T* { return ptr; }
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto size() const noexcept -> Size { return len; }
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto empty() const noexcept -> bool { return len == 0; }
    
    // Iterators
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto begin() const noexcept -> T* { return ptr; }
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto end() const noexcept -> T* { 
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        return ptr + len; 
    }
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto cbegin() const noexcept -> const T* { return ptr; }
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto cend() const noexcept -> const T* { 
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        return ptr + len; 
    }
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto rbegin() const noexcept -> reverse_iterator {
        return reverse_iterator(end());
    }
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto rend() const noexcept -> reverse_iterator {
        return reverse_iterator(begin());
    }
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto crbegin() const noexcept -> const_reverse_iterator {
        return const_reverse_iterator(cend());
    }
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto crend() const noexcept -> const_reverse_iterator {
        return const_reverse_iterator(cbegin());
    }
    
    // Subviews
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto subspan(Index offset, Size count) const noexcept -> Array<T> {
#if !defined(NDEBUG)
        assert(offset >= 0 && static_cast<Size>(offset) <= len && "Offset out of bounds");
        assert(static_cast<Size>(offset) + count <= len && "Count exceeds array bounds");
#endif
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        return Array<T>(ptr + offset, count);
    }
    
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto first(Size count) const noexcept -> Array<T> {
#if !defined(NDEBUG)
        assert(count <= len && "Count exceeds array size");
#endif
        return Array<T>(ptr, count);
    }
    
    [[nodiscard]] SCL_FORCE_INLINE constexpr auto last(Size count) const noexcept -> Array<T> {
#if !defined(NDEBUG)
        assert(count <= len && "Count exceeds array size");
#endif
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        return Array<T>(ptr + (len - count), count);
    }
    
    // Conversion to std::span
    [[nodiscard]] constexpr auto as_span() const noexcept -> std::span<T> {
        return std::span<T>(ptr, len);
    }
};

// Static assertions for POD and trivial copyability
static_assert(std::is_trivially_copyable_v<Array<Real>>);
static_assert(std::is_trivially_copyable_v<Array<const Real>>);
static_assert(std::is_trivially_copyable_v<Array<Index>>);
static_assert(std::is_standard_layout_v<Array<Real>>);

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
    requires std::same_as<
        std::remove_const_t<std::iter_value_t<decltype(a.begin())>>,
        std::remove_const_t<typename A::value_type>
    >;
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
    requires M::Tag::is_csr == true;
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
    requires M::Tag::is_csc == true;
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
