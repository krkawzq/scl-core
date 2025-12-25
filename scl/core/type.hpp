#pragma once

#include "scl/config.hpp"
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cassert> // For debug assertions

// =============================================================================
/// @file type.hpp
/// @brief SCL Core Type System - Unified Data Types
///
/// This header defines the fundamental data types used throughout the SCL library.
/// It provides a zero-overhead abstraction layer for:
/// - Floating-point precision (`scl::Real`) - switchable at compile time
/// - Integer indexing (`scl::Index`) - compatible with NumPy int64
/// - Memory views (`scl::Span`) - non-owning array reference
///
/// @section Design Philosophy
///
/// The type system follows these principles:
/// 1. **Zero Overhead**: All types are zero-cost abstractions (no vtables, no RTTI).
/// 2. **Compile-Time Selection**: Precision chosen via CMake, not runtime checks.
/// 3. **C-ABI Compatible**: All types can be safely passed across language boundaries.
/// 4. **NumPy Aligned**: Type sizes and semantics match Python/NumPy conventions.
///
/// @section Precision Selection
///
/// The `Real` type is selected at compile time via `SCL_PRECISION`:
/// - `SCL_PRECISION=0` → `Real = float` (32-bit, default)
/// - `SCL_PRECISION=1` → `Real = double` (64-bit)
/// - `SCL_PRECISION=2` → `Real = _Float16` (16-bit, requires modern compiler)
///
// =============================================================================

namespace scl {

// =============================================================================
// SECTION 1: Floating-Point Precision (Real Type)
// =============================================================================

/// @defgroup Types Core Type Definitions
/// @{

// [Owner: AI]
// Precision-dependent type selection using preprocessor branches.
// The config.hpp ensures exactly one SCL_USE_FLOAT* macro is defined.

#if defined(SCL_USE_FLOAT32)
    /// @brief Unified floating-point type (f32)
    ///
    /// Selected when `SCL_PRECISION=0` (default).
    /// - Size: 4 bytes
    /// - Range: ±3.4e38
    /// - Precision: ~7 decimal digits
    /// - Use case: Standard simulations, memory-constrained environments
    using Real = float;
    
    /// @brief Runtime type identifier (compatible with NumPy typenum)
    ///
    /// Allows Python to verify type consistency at runtime.
    /// Maps to: np.float32 (typenum=11 in NumPy)
    constexpr int DTYPE_CODE = 0;
    
    /// @brief Human-readable type name
    constexpr const char* DTYPE_NAME = "float32";

#elif defined(SCL_USE_FLOAT64)
    /// @brief Unified floating-point type (f64)
    ///
    /// Selected when `SCL_PRECISION=1`.
    /// - Size: 8 bytes
    /// - Range: ±1.8e308
    /// - Precision: ~15 decimal digits
    /// - Use case: High-precision scientific computing
    using Real = double;
    
    /// @brief Runtime type identifier
    ///
    /// Maps to: np.float64 (typenum=12 in NumPy)
    constexpr int DTYPE_CODE = 1;
    
    /// @brief Human-readable type name
    constexpr const char* DTYPE_NAME = "float64";

#elif defined(SCL_USE_FLOAT16)
    /// @brief Unified floating-point type (f16)
    ///
    /// Selected when `SCL_PRECISION=2`.
    /// - Size: 2 bytes
    /// - Range: ±6.5e4
    /// - Precision: ~3 decimal digits
    /// - Use case: ML inference, GPU-accelerated pipelines
    ///
    /// @note Requires compiler support for `_Float16`:
    ///   - GCC ≥ 12
    ///   - Clang ≥ 15
    ///   - MSVC does not support f16 natively (use fallback library)
    
    // Platform-specific Float16 detection
    #if defined(__FLT16_MANT_DIG__) || \
        (defined(__clang__) && __clang_major__ >= 15) || \
        (defined(__GNUC__) && __GNUC__ >= 12)
        using Real = _Float16;
    #else
        #error "SCL Type Error: Native _Float16 unsupported by this compiler. " \
               "Requires GCC ≥12 or Clang ≥15. Consider using f32 instead."
    #endif
    
    /// @brief Runtime type identifier
    ///
    /// Maps to: np.float16 (typenum=23 in NumPy)
    constexpr int DTYPE_CODE = 2;
    
    /// @brief Human-readable type name
    constexpr const char* DTYPE_NAME = "float16";

#else
    #error "SCL Type Error: No precision macro defined in config.hpp! " \
           "Ensure SCL_PRECISION is set by CMake."
#endif

// =============================================================================
// SECTION 2: Integer Types
// =============================================================================

/// @brief Unified index type for array addressing and loop counters
///
/// **Design rationale**: We use `int64_t` (not `size_t`) because:
/// 1. **NumPy Compatibility**: NumPy defaults to int64 for indexing.
/// 2. **Large Dataset Support**: Biological datasets can exceed 2^31 cells.
/// 3. **Signed Arithmetic**: Allows negative indices (e.g., Python-style [-1]).
/// 4. **Cross-Platform Consistency**: `size_t` varies (32-bit on some platforms).
///
/// @note For pointer arithmetic, use `Size` instead.
using Index = std::int64_t;

/// @brief Unsigned size type (wraps `std::size_t`)
///
/// Use for:
/// - Memory allocation sizes
/// - Container capacity queries
/// - Pointer arithmetic
///
/// Do NOT use for array indexing (use `Index` instead for NumPy compatibility).
using Size = std::size_t;

/// @brief Byte type for raw memory manipulation
///
/// Use for:
/// - Low-level memory operations
/// - Buffer serialization/deserialization
/// - Byte-level data packing
using Byte = std::uint8_t;

// =============================================================================
// SECTION 3: Zero-Overhead View Types (Span)
// =============================================================================

/// @brief Lightweight, non-owning view of a contiguous memory array
///
/// **Purpose**: Replaces `std::vector<T>` in function signatures to:
/// 1. Avoid heap allocation overhead.
/// 2. Enable C-ABI compatibility (no STL mangling).
/// 3. Work seamlessly with NumPy arrays from Python.
///
/// **Memory Model**: 
/// - Does NOT own the data (no destructor, no allocation).
/// - Simply wraps a raw pointer + size.
/// - Size: 16 bytes (pointer + size_t) on 64-bit systems.
///
/// **Usage Example**:
/// @code{.cpp}
/// void process(Span<const Real> input, MutableSpan<Real> output) {
///     for (Index i = 0; i < input.size; ++i) {
///         output[i] = input[i] * 2.0;
///     }
/// }
/// @endcode
///
/// @tparam T Element type (typically `Real`, `const Real`, or `Index`)
///
/// [Owner: AI]
template <typename T>
struct Span {
    T* ptr;      ///< Pointer to the first element
    Size size;   ///< Number of elements (NOT bytes)

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    
    /// @brief Default constructor (creates empty span)
    constexpr Span() noexcept : ptr(nullptr), size(0) {}

    /// @brief Construct from raw pointer and element count
    ///
    /// @param p Pointer to data start
    /// @param s Number of elements (not bytes!)
    constexpr Span(T* p, Size s) noexcept : ptr(p), size(s) {}

    // -------------------------------------------------------------------------
    // Element Access
    // -------------------------------------------------------------------------
    
    /// @brief Array subscript operator (with debug bounds check)
    ///
    /// @param i Element index (0-based)
    /// @return Reference to element at position i
    ///
    /// @warning Does NOT perform bounds checking in Release mode for performance.
    constexpr T& operator[](Index i) const noexcept {
#if !defined(NDEBUG)
        assert(i >= 0 && static_cast<Size>(i) < size && "Span index out of bounds");
#endif
        return ptr[i];
    }

    // -------------------------------------------------------------------------
    // Iterator Interface (STL-compatible)
    // -------------------------------------------------------------------------
    
    /// @brief Get pointer to first element (begin iterator)
    constexpr T* begin() const noexcept { return ptr; }

    /// @brief Get pointer to one-past-last element (end iterator)
    constexpr T* end() const noexcept { return ptr + size; }
    
    // -------------------------------------------------------------------------
    // Utility Methods
    // -------------------------------------------------------------------------
    
    /// @brief Check if span is empty (size == 0)
    constexpr bool empty() const noexcept { return size == 0; }
    
    /// @brief Get raw pointer to data
    constexpr T* data() const noexcept { return ptr; }
    
    /// @brief Get byte size of the entire span
    ///
    /// @return Total size in bytes (size * sizeof(T))
    constexpr Size byte_size() const noexcept { return size * sizeof(T); }
};

// =============================================================================
// SECTION 4: Common Type Aliases
// =============================================================================

/// @brief Generic mutable view (alias for Span<T>)
/// Use this for template functions that output data.
template <typename T>
using MutableSpan = Span<T>;

/// @brief Generic read-only view (alias for Span<const T>)
/// Use this for template functions that only read data.
template <typename T>
using ConstSpan = Span<const T>;

/// @brief Read-only view of Real array (const Real*)
/// Example: `void analyze(RealSpan input)`
using RealSpan = ConstSpan<Real>;

/// @brief Mutable view of Real array (Real*)
/// Example: `void transform(MutableRealSpan output)`
using MutableRealSpan = MutableSpan<Real>;

/// @brief Read-only view of Index array
using IndexSpan = ConstSpan<Index>;

/// @brief Mutable view of Index array
using MutableIndexSpan = MutableSpan<Index>;

/// @}

} // namespace scl
