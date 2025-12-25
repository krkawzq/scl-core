#pragma once

#include "scl/config.hpp"
#include <cstdint>
#include <cstdlib>

// =============================================================================
/// @file macros.hpp
/// @brief SCL Core Compiler Abstractions & Optimization Hints
///
/// This header provides cross-platform macros for compiler-specific features
/// like branch prediction, alignment, inlining, and symbol visibility.
///
/// @section Naming Convention
/// All macros are prefixed with `SCL_` to avoid global namespace pollution.
///
// =============================================================================

// =============================================================================
// SECTION 1: Branch Prediction Hints
// =============================================================================

/// @defgroup BranchPrediction Branch Prediction Hints
/// @{

#if defined(__clang__) || defined(__GNUC__)
    /// @brief Hint that the condition is likely to be true.
    /// Use to optimize hot paths where the branch is taken >90% of the time.
    #define SCL_LIKELY(x)   (__builtin_expect(!!(x), 1))

    /// @brief Hint that the condition is unlikely to be true.
    /// Use for error checks or rare edge cases.
    #define SCL_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
    #define SCL_LIKELY(x)   (x)
    #define SCL_UNLIKELY(x) (x)
#endif

/// @}

// =============================================================================
// SECTION 2: Function Inlining & Visibility
// =============================================================================

/// @defgroup Inlining Inlining Control
/// @{

#if defined(_MSC_VER)
    /// @brief Force the compiler to inline a function (ignores cost model).
    #define SCL_FORCE_INLINE __forceinline
    
    /// @brief Hint that a pointer is not aliased (optimizes memory access).
    #define SCL_RESTRICT __restrict
    
    /// @brief Export symbol for DLL/Shared Library.
    #define SCL_EXPORT __declspec(dllexport)
#else
    /// @brief Force the compiler to inline a function.
    #define SCL_FORCE_INLINE inline __attribute__((always_inline))
    
    /// @brief Hint that a pointer is not aliased.
    #define SCL_RESTRICT __restrict__
    
    /// @brief Export symbol for Shared Library.
    #define SCL_EXPORT __attribute__((visibility("default")))
#endif

/// @}

// =============================================================================
// SECTION 3: Memory Alignment & Prefetching
// =============================================================================

/// @defgroup MemoryTools Memory Optimization Tools
/// @{

/// @brief Default alignment for SIMD operations (64 bytes for AVX-512 compatibility)
#define SCL_ALIGNMENT 64

// --- Alignment Specification ---
#if defined(_MSC_VER)
    #define SCL_ALIGN_AS(N) __declspec(align(N))
#else
    #define SCL_ALIGN_AS(N) __attribute__((aligned(N)))
#endif

// --- Pointer Alignment Hint ---
// Tells the compiler "I promise this pointer is aligned to N bytes"
#if defined(__clang__) || defined(__GNUC__)
    #define SCL_ASSUME_ALIGNED(ptr, N) reinterpret_cast<decltype(ptr)>(__builtin_assume_aligned((ptr), (N)))
#else
    #define SCL_ASSUME_ALIGNED(ptr, N) (ptr)
#endif

// --- Data Prefetching ---
// Locality: 0 (none), 1 (low), 2 (moderate), 3 (high - keep in L1)
#if defined(__clang__) || defined(__GNUC__)
    /// @brief Prefetch memory into cache for reading.
    #define SCL_PREFETCH_READ(ptr, locality) __builtin_prefetch((ptr), 0, (locality))
    
    /// @brief Prefetch memory into cache for writing.
    #define SCL_PREFETCH_WRITE(ptr, locality) __builtin_prefetch((ptr), 1, (locality))
#elif defined(_MSC_VER)
    #include <xmmintrin.h>
    // MSVC uses _mm_prefetch with hint constants
    // locality 0-3 maps to: _MM_HINT_NTA, _MM_HINT_T2, _MM_HINT_T1, _MM_HINT_T0
    #define SCL_PREFETCH_READ(ptr, locality) \
        _mm_prefetch(reinterpret_cast<const char*>(ptr), \
                     (locality) == 0 ? _MM_HINT_NTA : \
                     (locality) == 1 ? _MM_HINT_T2  : \
                     (locality) == 2 ? _MM_HINT_T1  : _MM_HINT_T0)
    #define SCL_PREFETCH_WRITE(ptr, locality) SCL_PREFETCH_READ(ptr, locality)
#else
    #define SCL_PREFETCH_READ(ptr, locality) ((void)0)
    #define SCL_PREFETCH_WRITE(ptr, locality) ((void)0)
#endif

/// @}

// =============================================================================
// SECTION 4: Aligned Memory Management (Low Level)
// =============================================================================

/// @defgroup MemoryAlloc Aligned Allocation Macros
/// @{
/// @warning Use these only when raw memory management is strictly required.
/// Consider using wrappers or containers in `scl::core` for safety.

#if defined(_MSC_VER)
    #include <malloc.h>
    
    /// @brief Allocate aligned memory (Raw).
    /// @param ptr  Pointer variable to store the result.
    /// @param size Size in bytes.
    #define SCL_MALLOC_ALIGNED(ptr, size) \
        do { (ptr) = static_cast<void*>(_aligned_malloc((size), SCL_ALIGNMENT)); } while(0)
    
    /// @brief Free aligned memory (Raw).
    #define SCL_FREE_ALIGNED(ptr) \
        do { _aligned_free((ptr)); } while(0)

#else
    /// @brief Allocate aligned memory (Raw).
    /// On POSIX, posix_memalign returns 0 on success.
    #define SCL_MALLOC_ALIGNED(ptr, size) \
        do { \
            void* _tmp_ptr = nullptr; \
            if (posix_memalign(&_tmp_ptr, SCL_ALIGNMENT, (size)) == 0) { \
                (ptr) = _tmp_ptr; \
            } else { \
                (ptr) = nullptr; \
            } \
        } while(0)

    /// @brief Free aligned memory (Raw).
    #define SCL_FREE_ALIGNED(ptr) \
        do { free((ptr)); } while(0)

#endif

/// @}
