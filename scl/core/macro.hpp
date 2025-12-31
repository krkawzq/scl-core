#pragma once

/// @file scl/core/macro.hpp
/// @brief Cross-platform compiler abstractions and optimization hints
///
/// This header provides MACRO-BASED utilities only:
/// - Compiler/platform/architecture detection
/// - SIMD feature detection
/// - Function attributes (only those without C++20 standard equivalents)
/// - Memory alignment and prefetching
/// - Loop optimization hints
/// - Compile-time string utilities
/// - Platform information
/// - Source location support (C++20 std::source_location with fallback)
///
/// @note This header does NOT provide:
///       - Error handling (see scl/core/error.hpp)
///       - Bit manipulation (see scl/core/bits.hpp)
///       - Debug utilities (see scl/core/debug.hpp)
///
/// @note C++20 attributes should be used directly:
///       - [[nodiscard]], [[maybe_unused]], [[deprecated]]
///       - [[likely]], [[unlikely]] (on branches, not expressions)
///       - [[no_unique_address]]

#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <cstdio>

// =============================================================================
// SECTION 0: C++ Standard Version Detection
// =============================================================================

#if defined(__cplusplus)
    #if __cplusplus >= 202302L
        #define SCL_CPP_VERSION 23
        #define SCL_CPP23_OR_LATER 1
        #define SCL_CPP20_OR_LATER 1
        #define SCL_CPP17_OR_LATER 1
    #elif __cplusplus >= 202002L
        #define SCL_CPP_VERSION 20
        #define SCL_CPP23_OR_LATER 0
        #define SCL_CPP20_OR_LATER 1
        #define SCL_CPP17_OR_LATER 1
    #elif __cplusplus >= 201703L
        #define SCL_CPP_VERSION 17
        #define SCL_CPP23_OR_LATER 0
        #define SCL_CPP20_OR_LATER 0
        #define SCL_CPP17_OR_LATER 1
    #else
        #error "scl-core requires C++17 or later"
    #endif
#else
    #error "scl-core requires a C++ compiler"
#endif

// Enforce minimum C++20 for this library
#if !SCL_CPP20_OR_LATER
    #error "scl-core requires C++20 or later. Please use -std=c++20 or higher."
#endif

// =============================================================================
// SECTION 1: Compiler Detection
// =============================================================================

// Compiler identification
#if defined(__clang__)
    #define SCL_COMPILER_CLANG 1
    #define SCL_COMPILER_GCC 0
    #define SCL_COMPILER_MSVC 0
    #define SCL_COMPILER_GCC_LIKE 1
    #define SCL_COMPILER_VERSION (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
    #define SCL_COMPILER_NAME "Clang"
#elif defined(__GNUC__)
    #define SCL_COMPILER_CLANG 0
    #define SCL_COMPILER_GCC 1
    #define SCL_COMPILER_MSVC 0
    #define SCL_COMPILER_GCC_LIKE 1
    #define SCL_COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
    #define SCL_COMPILER_NAME "GCC"
#elif defined(_MSC_VER)
    #define SCL_COMPILER_CLANG 0
    #define SCL_COMPILER_GCC 0
    #define SCL_COMPILER_MSVC 1
    #define SCL_COMPILER_GCC_LIKE 0
    #define SCL_COMPILER_VERSION _MSC_VER
    #define SCL_COMPILER_NAME "MSVC"
#else
    #define SCL_COMPILER_CLANG 0
    #define SCL_COMPILER_GCC 0
    #define SCL_COMPILER_MSVC 0
    #define SCL_COMPILER_GCC_LIKE 0
    #define SCL_COMPILER_VERSION 0
    #define SCL_COMPILER_NAME "Unknown"
#endif

// Minimum compiler version checks
#if SCL_COMPILER_GCC && SCL_COMPILER_VERSION < 110000
    #error "scl-core requires GCC 11 or later"
#endif
#if SCL_COMPILER_CLANG && SCL_COMPILER_VERSION < 140000
    #error "scl-core requires Clang 14 or later"
#endif
#if SCL_COMPILER_MSVC && SCL_COMPILER_VERSION < 1929
    #error "scl-core requires MSVC 19.29 or later"
#endif

// =============================================================================
// SECTION 2: Platform Detection
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
    #define SCL_PLATFORM_WINDOWS 1
    #define SCL_PLATFORM_POSIX 0
    #define SCL_PLATFORM_UNIX 0
    #define SCL_PLATFORM_MACOS 0
    #define SCL_PLATFORM_LINUX 0
    #define SCL_PLATFORM_NAME "Windows"
#elif defined(__APPLE__) && defined(__MACH__)
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 1
    #define SCL_PLATFORM_UNIX 1
    #define SCL_PLATFORM_MACOS 1
    #define SCL_PLATFORM_LINUX 0
    #define SCL_PLATFORM_NAME "macOS"
#elif defined(__linux__) || defined(__linux)
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 1
    #define SCL_PLATFORM_UNIX 1
    #define SCL_PLATFORM_MACOS 0
    #define SCL_PLATFORM_LINUX 1
    #define SCL_PLATFORM_NAME "Linux"
#elif defined(__unix__) || defined(__unix)
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 1
    #define SCL_PLATFORM_UNIX 1
    #define SCL_PLATFORM_MACOS 0
    #define SCL_PLATFORM_LINUX 0
    #define SCL_PLATFORM_NAME "Unix"
#else
    #define SCL_PLATFORM_WINDOWS 0
    #define SCL_PLATFORM_POSIX 0
    #define SCL_PLATFORM_UNIX 0
    #define SCL_PLATFORM_MACOS 0
    #define SCL_PLATFORM_LINUX 0
    #define SCL_PLATFORM_NAME "Unknown"
#endif

// Pointer size detection
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__) || defined(__ppc64__)
    #define SCL_ARCH_64BIT 1
    #define SCL_ARCH_32BIT 0
#else
    #define SCL_ARCH_64BIT 0
    #define SCL_ARCH_32BIT 1
#endif

// =============================================================================
// SECTION 3: Architecture Detection
// =============================================================================

// CPU Architecture
#if defined(__x86_64__) || defined(_M_X64)
    #define SCL_ARCH_X86_64 1
    #define SCL_ARCH_X86 1
    #define SCL_ARCH_ARM64 0
    #define SCL_ARCH_ARM 0
    #define SCL_ARCH_NAME "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
    #define SCL_ARCH_X86_64 0
    #define SCL_ARCH_X86 1
    #define SCL_ARCH_ARM64 0
    #define SCL_ARCH_ARM 0
    #define SCL_ARCH_NAME "x86"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define SCL_ARCH_X86_64 0
    #define SCL_ARCH_X86 0
    #define SCL_ARCH_ARM64 1
    #define SCL_ARCH_ARM 1
    #define SCL_ARCH_NAME "ARM64"
#elif defined(__arm__) || defined(_M_ARM)
    #define SCL_ARCH_X86_64 0
    #define SCL_ARCH_X86 0
    #define SCL_ARCH_ARM64 0
    #define SCL_ARCH_ARM 1
    #define SCL_ARCH_NAME "ARM"
#else
    #define SCL_ARCH_X86_64 0
    #define SCL_ARCH_X86 0
    #define SCL_ARCH_ARM64 0
    #define SCL_ARCH_ARM 0
    #define SCL_ARCH_NAME "Unknown"
#endif

// =============================================================================
// SECTION 4: SIMD Feature Detection
// =============================================================================

// x86 SIMD features
#if SCL_ARCH_X86

    #if defined(__AVX512F__)
        #define SCL_SIMD_AVX512 1
        #define SCL_SIMD_AVX512F 1
    #else
        #define SCL_SIMD_AVX512 0
        #define SCL_SIMD_AVX512F 0
    #endif
    
    #if defined(__AVX512BW__)
        #define SCL_SIMD_AVX512BW 1
    #else
        #define SCL_SIMD_AVX512BW 0
    #endif
    
    #if defined(__AVX512DQ__)
        #define SCL_SIMD_AVX512DQ 1
    #else
        #define SCL_SIMD_AVX512DQ 0
    #endif
    
    #if defined(__AVX512VL__)
        #define SCL_SIMD_AVX512VL 1
    #else
        #define SCL_SIMD_AVX512VL 0
    #endif
    
    #if defined(__AVX2__)
        #define SCL_SIMD_AVX2 1
    #else
        #define SCL_SIMD_AVX2 0
    #endif
    
    #if defined(__AVX__)
        #define SCL_SIMD_AVX 1
    #else
        #define SCL_SIMD_AVX 0
    #endif
    
    #if defined(__FMA__)
        #define SCL_SIMD_FMA 1
    #else
        #define SCL_SIMD_FMA 0
    #endif
    
    #if defined(__SSE4_2__)
        #define SCL_SIMD_SSE42 1
    #else
        #define SCL_SIMD_SSE42 0
    #endif
    
    #if defined(__SSE4_1__)
        #define SCL_SIMD_SSE41 1
    #else
        #define SCL_SIMD_SSE41 0
    #endif
    
    #if defined(__SSE2__) || SCL_ARCH_X86_64
        #define SCL_SIMD_SSE2 1
    #else
        #define SCL_SIMD_SSE2 0
    #endif
    
    // ARM SIMD not available on x86
    #define SCL_SIMD_NEON 0
    #define SCL_SIMD_SVE 0
    #define SCL_SIMD_SVE2 0

#elif SCL_ARCH_ARM

    // No x86 SIMD on ARM
    #define SCL_SIMD_AVX512 0
    #define SCL_SIMD_AVX512F 0
    #define SCL_SIMD_AVX512BW 0
    #define SCL_SIMD_AVX512DQ 0
    #define SCL_SIMD_AVX512VL 0
    #define SCL_SIMD_AVX2 0
    #define SCL_SIMD_AVX 0
    #define SCL_SIMD_FMA 0
    #define SCL_SIMD_SSE42 0
    #define SCL_SIMD_SSE41 0
    #define SCL_SIMD_SSE2 0
    
    // ARM NEON
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #define SCL_SIMD_NEON 1
    #else
        #define SCL_SIMD_NEON 0
    #endif
    
    // ARM SVE
    #if defined(__ARM_FEATURE_SVE)
        #define SCL_SIMD_SVE 1
    #else
        #define SCL_SIMD_SVE 0
    #endif
    
    #if defined(__ARM_FEATURE_SVE2)
        #define SCL_SIMD_SVE2 1
    #else
        #define SCL_SIMD_SVE2 0
    #endif

#else
    // Unknown architecture - no SIMD
    #define SCL_SIMD_AVX512 0
    #define SCL_SIMD_AVX512F 0
    #define SCL_SIMD_AVX512BW 0
    #define SCL_SIMD_AVX512DQ 0
    #define SCL_SIMD_AVX512VL 0
    #define SCL_SIMD_AVX2 0
    #define SCL_SIMD_AVX 0
    #define SCL_SIMD_FMA 0
    #define SCL_SIMD_SSE42 0
    #define SCL_SIMD_SSE41 0
    #define SCL_SIMD_SSE2 0
    #define SCL_SIMD_NEON 0
    #define SCL_SIMD_SVE 0
    #define SCL_SIMD_SVE2 0
#endif

// Best available SIMD level (for runtime dispatch)
#if SCL_SIMD_AVX512
    #define SCL_SIMD_LEVEL 512
    #define SCL_SIMD_LEVEL_NAME "AVX-512"
#elif SCL_SIMD_AVX2
    #define SCL_SIMD_LEVEL 256
    #define SCL_SIMD_LEVEL_NAME "AVX2"
#elif SCL_SIMD_AVX
    #define SCL_SIMD_LEVEL 256
    #define SCL_SIMD_LEVEL_NAME "AVX"
#elif SCL_SIMD_SSE42
    #define SCL_SIMD_LEVEL 128
    #define SCL_SIMD_LEVEL_NAME "SSE4.2"
#elif SCL_SIMD_SSE2
    #define SCL_SIMD_LEVEL 128
    #define SCL_SIMD_LEVEL_NAME "SSE2"
#elif SCL_SIMD_SVE2
    #define SCL_SIMD_LEVEL 2048  // SVE is variable width
    #define SCL_SIMD_LEVEL_NAME "SVE2"
#elif SCL_SIMD_SVE
    #define SCL_SIMD_LEVEL 2048
    #define SCL_SIMD_LEVEL_NAME "SVE"
#elif SCL_SIMD_NEON
    #define SCL_SIMD_LEVEL 128
    #define SCL_SIMD_LEVEL_NAME "NEON"
#else
    #define SCL_SIMD_LEVEL 64
    #define SCL_SIMD_LEVEL_NAME "Scalar"
#endif

// =============================================================================
// SECTION 5: SIMD Width Constants
// =============================================================================

namespace scl::platform {

/// @brief Maximum SIMD vector width in bytes
#if SCL_SIMD_AVX512
    inline constexpr std::size_t simd_width = 64;
    inline constexpr int simd_lanes_f64 = 8;
    inline constexpr int simd_lanes_f32 = 16;
    inline constexpr int simd_lanes_i32 = 16;
    inline constexpr int simd_lanes_i64 = 8;
#elif SCL_SIMD_AVX2 || SCL_SIMD_AVX
    inline constexpr std::size_t simd_width = 32;
    inline constexpr int simd_lanes_f64 = 4;
    inline constexpr int simd_lanes_f32 = 8;
    inline constexpr int simd_lanes_i32 = 8;
    inline constexpr int simd_lanes_i64 = 4;
#elif SCL_SIMD_SSE2 || SCL_SIMD_NEON
    inline constexpr std::size_t simd_width = 16;
    inline constexpr int simd_lanes_f64 = 2;
    inline constexpr int simd_lanes_f32 = 4;
    inline constexpr int simd_lanes_i32 = 4;
    inline constexpr int simd_lanes_i64 = 2;
#else
    inline constexpr std::size_t simd_width = 8;
    inline constexpr int simd_lanes_f64 = 1;
    inline constexpr int simd_lanes_f32 = 2;
    inline constexpr int simd_lanes_i32 = 2;
    inline constexpr int simd_lanes_i64 = 1;
#endif

/// @brief Cache line size (assumed 64 bytes for modern CPUs)
inline constexpr std::size_t cache_line_size = 64;

/// @brief Default memory alignment (matches AVX-512 requirements)
inline constexpr std::size_t default_alignment = 64;

/// @brief Number of SIMD lanes for a given type
template<typename T>
inline constexpr int simd_lanes = static_cast<int>(simd_width / sizeof(T));

}  // namespace scl::platform

// Macro versions for preprocessor use
#define SCL_SIMD_WIDTH      ::scl::platform::simd_width
#define SCL_CACHE_LINE_SIZE ::scl::platform::cache_line_size
#define SCL_ALIGNMENT       ::scl::platform::default_alignment

// =============================================================================
// SECTION 6: Function Attributes (Only Non-Standard)
// =============================================================================
//
// NOTE: Do NOT use macros for these C++20 standard attributes:
//   - [[nodiscard]]        - Use directly
//   - [[maybe_unused]]     - Use directly
//   - [[deprecated("x")]]  - Use directly
//   - [[likely]]           - Use on branch: if (...) [[likely]] { }
//   - [[unlikely]]         - Use on branch: if (...) [[unlikely]] { }
//   - [[no_unique_address]] - Use directly
//
// The macros below are ONLY for attributes without C++20 equivalents.
// =============================================================================

// Force inline (no standard equivalent)
#if SCL_COMPILER_MSVC
    #define SCL_FORCE_INLINE __forceinline
#elif SCL_COMPILER_GCC_LIKE
    #define SCL_FORCE_INLINE inline __attribute__((always_inline))
#else
    #define SCL_FORCE_INLINE inline
#endif

// Prevent inlining
#if SCL_COMPILER_MSVC
    #define SCL_NOINLINE __declspec(noinline)
#elif SCL_COMPILER_GCC_LIKE
    #define SCL_NOINLINE __attribute__((noinline))
#else
    #define SCL_NOINLINE
#endif

// Pointer aliasing hint (no standard equivalent)
#if SCL_COMPILER_MSVC
    #define SCL_RESTRICT __restrict
#elif SCL_COMPILER_GCC_LIKE
    #define SCL_RESTRICT __restrict__
#else
    #define SCL_RESTRICT
#endif

// DLL export/import
#if SCL_PLATFORM_WINDOWS
    #define SCL_EXPORT __declspec(dllexport)
    #define SCL_IMPORT __declspec(dllimport)
#elif SCL_COMPILER_GCC_LIKE
    #define SCL_EXPORT __attribute__((visibility("default")))
    #define SCL_IMPORT __attribute__((visibility("default")))
#else
    #define SCL_EXPORT
    #define SCL_IMPORT
#endif

// Hot/cold function hints
#if SCL_COMPILER_GCC_LIKE
    #define SCL_HOT __attribute__((hot))
    #define SCL_COLD __attribute__((cold))
#else
    #define SCL_HOT
    #define SCL_COLD
#endif

// Flatten: inline all calls within this function
#if SCL_COMPILER_GCC_LIKE
    #define SCL_FLATTEN __attribute__((flatten))
#else
    #define SCL_FLATTEN
#endif

// Pure function hints
#if SCL_COMPILER_GCC_LIKE
    /// @brief Function result depends only on arguments, no side effects
    #define SCL_PURE __attribute__((pure))
    /// @brief Function result depends only on arguments, no memory access
    #define SCL_CONST_FUNC __attribute__((const))
#else
    #define SCL_PURE
    #define SCL_CONST_FUNC
#endif

// No-throw for optimizer (use noexcept in code, this is for optimizer hints)
#if SCL_COMPILER_GCC_LIKE
    #define SCL_NOTHROW_ATTR __attribute__((nothrow))
#elif SCL_COMPILER_MSVC
    #define SCL_NOTHROW_ATTR __declspec(nothrow)
#else
    #define SCL_NOTHROW_ATTR
#endif

// =============================================================================
// SECTION 7: Memory Alignment Attributes
// =============================================================================

// Alignment specifier for declarations
#if SCL_COMPILER_MSVC
    #define SCL_ALIGN_AS(N) __declspec(align(N))
#elif SCL_COMPILER_GCC_LIKE
    #define SCL_ALIGN_AS(N) __attribute__((aligned(N)))
#else
    #define SCL_ALIGN_AS(N) alignas(N)
#endif

// Cache-line aligned for avoiding false sharing
#if SCL_COMPILER_MSVC
    #define SCL_CACHE_ALIGNED __declspec(align(64))
#elif SCL_COMPILER_GCC_LIKE
    #define SCL_CACHE_ALIGNED __attribute__((aligned(64)))
#else
    #define SCL_CACHE_ALIGNED alignas(64)
#endif

// Assume pointer is aligned (for optimizer)
#if SCL_COMPILER_GCC_LIKE
    #define SCL_ASSUME_ALIGNED(ptr, N) \
        static_cast<decltype(ptr)>(__builtin_assume_aligned((ptr), (N)))
#else
    #define SCL_ASSUME_ALIGNED(ptr, N) (ptr)
#endif

// =============================================================================
// SECTION 8: Prefetch and Cache Hints
// =============================================================================

/// @brief Prefetch locality levels
/// 0 = NTA (non-temporal, no cache)
/// 1 = T2 (low locality, L3)
/// 2 = T1 (medium locality, L2)
/// 3 = T0 (high locality, L1)

#if SCL_COMPILER_GCC_LIKE
    #define SCL_PREFETCH(addr, rw, locality) __builtin_prefetch((addr), (rw), (locality))
    #define SCL_PREFETCH_READ(addr, locality) __builtin_prefetch((addr), 0, (locality))
    #define SCL_PREFETCH_WRITE(addr, locality) __builtin_prefetch((addr), 1, (locality))
#elif SCL_COMPILER_MSVC
    #include <xmmintrin.h>
    #define SCL_PREFETCH(addr, rw, locality) \
        _mm_prefetch(reinterpret_cast<const char*>(addr), \
                     (locality) == 0 ? _MM_HINT_NTA : \
                     (locality) == 1 ? _MM_HINT_T2  : \
                     (locality) == 2 ? _MM_HINT_T1  : _MM_HINT_T0)
    #define SCL_PREFETCH_READ(addr, locality) SCL_PREFETCH(addr, 0, locality)
    #define SCL_PREFETCH_WRITE(addr, locality) SCL_PREFETCH(addr, 1, locality)
#else
    #define SCL_PREFETCH(addr, rw, locality) ((void)0)
    #define SCL_PREFETCH_READ(addr, locality) ((void)0)
    #define SCL_PREFETCH_WRITE(addr, locality) ((void)0)
#endif

// Prefetch for streaming (non-temporal) loads
#if SCL_COMPILER_GCC_LIKE
    #define SCL_PREFETCH_NTA(addr) __builtin_prefetch((addr), 0, 0)
#elif SCL_COMPILER_MSVC
    #define SCL_PREFETCH_NTA(addr) _mm_prefetch(reinterpret_cast<const char*>(addr), _MM_HINT_NTA)
#else
    #define SCL_PREFETCH_NTA(addr) ((void)0)
#endif

// =============================================================================
// SECTION 9: Optimizer Hints
// =============================================================================

// Assume condition is true (UB if false)
#if SCL_COMPILER_CLANG
    #define SCL_ASSUME(cond) __builtin_assume(cond)
#elif SCL_COMPILER_GCC && SCL_COMPILER_VERSION >= 130000
    #define SCL_ASSUME(cond) __attribute__((assume(cond)))
#elif SCL_COMPILER_MSVC
    #define SCL_ASSUME(cond) __assume(cond)
#else
    #define SCL_ASSUME(cond) ((void)0)
#endif

// Mark code as unreachable
#if SCL_COMPILER_GCC_LIKE
    #define SCL_UNREACHABLE() __builtin_unreachable()
#elif SCL_COMPILER_MSVC
    #define SCL_UNREACHABLE() __assume(0)
#else
    #define SCL_UNREACHABLE() ((void)0)
#endif

// =============================================================================
// SECTION 10: Loop Optimization Hints
// =============================================================================

// Helper for pragma stringification
#define SCL_PRAGMA_IMPL(x) _Pragma(#x)
#define SCL_PRAGMA(x) SCL_PRAGMA_IMPL(x)

#if SCL_COMPILER_CLANG
    #define SCL_UNROLL(n) SCL_PRAGMA(clang loop unroll_count(n))
    #define SCL_UNROLL_FULL SCL_PRAGMA(clang loop unroll(full))
    #define SCL_VECTORIZE SCL_PRAGMA(clang loop vectorize(enable))
    #define SCL_NO_VECTORIZE SCL_PRAGMA(clang loop vectorize(disable))
    #define SCL_INTERLEAVE(n) SCL_PRAGMA(clang loop interleave_count(n))
#elif SCL_COMPILER_GCC && SCL_COMPILER_VERSION >= 80000
    #define SCL_UNROLL(n) SCL_PRAGMA(GCC unroll n)
    #define SCL_UNROLL_FULL SCL_PRAGMA(GCC unroll 16)
    #define SCL_VECTORIZE SCL_PRAGMA(GCC ivdep)
    #define SCL_NO_VECTORIZE
    #define SCL_INTERLEAVE(n)
#elif SCL_COMPILER_MSVC
    #define SCL_UNROLL(n)
    #define SCL_UNROLL_FULL
    #define SCL_VECTORIZE SCL_PRAGMA(loop(ivdep))
    #define SCL_NO_VECTORIZE SCL_PRAGMA(loop(no_vector))
    #define SCL_INTERLEAVE(n)
#else
    #define SCL_UNROLL(n)
    #define SCL_UNROLL_FULL
    #define SCL_VECTORIZE
    #define SCL_NO_VECTORIZE
    #define SCL_INTERLEAVE(n)
#endif

// OpenMP SIMD hint
#if defined(_OPENMP) && _OPENMP >= 201307
    #define SCL_OMP_SIMD SCL_PRAGMA(omp simd)
    #define SCL_OMP_SIMD_ALIGNED(vars, align) SCL_PRAGMA(omp simd aligned(vars:align))
#else
    #define SCL_OMP_SIMD
    #define SCL_OMP_SIMD_ALIGNED(vars, align)
#endif

// =============================================================================
// SECTION 11: Stack Array Utilities
// =============================================================================

// Stack-allocated aligned array (portable VLA-like)
#if SCL_COMPILER_GCC_LIKE
    #define SCL_STACK_ARRAY(T, name, size) \
        T name[(size)] __attribute__((aligned(64)))
#elif SCL_COMPILER_MSVC
    #define SCL_STACK_ARRAY(T, name, size) \
        __declspec(align(64)) T name[(size)]
#else
    #define SCL_STACK_ARRAY(T, name, size) alignas(64) T name[(size)]
#endif

// Dynamic stack allocation (use with caution)
#if SCL_COMPILER_MSVC
    #include <malloc.h>
    #define SCL_ALLOCA(size) _alloca(size)
#elif SCL_COMPILER_GCC_LIKE
    #include <alloca.h>
    #define SCL_ALLOCA(size) alloca(size)
#else
    #define SCL_ALLOCA(size) __builtin_alloca(size)
#endif

/// @brief Aligned stack allocation
/// @param size Size in bytes to allocate
/// @param alignment Required alignment (must be power of 2)
/// @return Aligned pointer to stack memory
///
/// Implementation:
///   1. Allocate size + alignment - 1 bytes
///   2. Manually align the pointer to the nearest aligned address
///   3. Return aligned pointer
///
/// @note The returned pointer is valid until the calling function returns
/// @warning Do not free() the returned pointer - it's on the stack
#define SCL_ALLOCA_ALIGNED(size, alignment) \
    ({ \
        void* _base = SCL_ALLOCA((size) + (alignment) - 1); \
        void* _aligned = reinterpret_cast<void*>( \
            (reinterpret_cast<std::uintptr_t>(_base) + (alignment) - 1) & ~((alignment) - 1)); \
        _aligned; \
    })

// Padding to avoid false sharing
#define SCL_PAD_TO_CACHE_LINE(name) \
    char name[::scl::platform::cache_line_size]

// =============================================================================
// SECTION 12: Compile-Time String Utilities
// =============================================================================

// Stringify
#define SCL_STRINGIFY_IMPL(x) #x
#define SCL_STRINGIFY(x) SCL_STRINGIFY_IMPL(x)

// Concatenate tokens
#define SCL_CONCAT_IMPL(a, b) a##b
#define SCL_CONCAT(a, b) SCL_CONCAT_IMPL(a, b)

// Array size
#define SCL_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

// Unique identifier generation
#define SCL_UNIQUE_ID(prefix) SCL_CONCAT(prefix, __LINE__)

// Suppress unused warnings
#define SCL_UNUSED(x) ((void)(x))

// =============================================================================
// SECTION 13: Platform-Specific Includes for Memory Operations
// =============================================================================

#if SCL_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <malloc.h>
#elif SCL_PLATFORM_POSIX
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

// =============================================================================
// SECTION 14: Memory-Mapped File Types (Platform-Specific)
// =============================================================================

namespace scl::io {

#if SCL_PLATFORM_WINDOWS
    using FileHandle = HANDLE;
    using MapHandle = HANDLE;
    inline constexpr FileHandle INVALID_FILE_HANDLE = INVALID_HANDLE_VALUE;
    inline constexpr MapHandle INVALID_MAP_HANDLE = nullptr;
#elif SCL_PLATFORM_POSIX
    using FileHandle = int;
    using MapHandle = void*;
    inline constexpr FileHandle INVALID_FILE_HANDLE = -1;
    inline constexpr MapHandle INVALID_MAP_HANDLE = nullptr;
#endif

}  // namespace scl::io

// =============================================================================
// SECTION 15: Platform Information Summary
// =============================================================================

namespace scl::platform {

/// @brief Runtime platform information structure
struct PlatformInfo {
    static constexpr const char* compiler_name = SCL_COMPILER_NAME;
    static constexpr int compiler_version = SCL_COMPILER_VERSION;
    static constexpr const char* platform_name = SCL_PLATFORM_NAME;
    static constexpr const char* arch_name = SCL_ARCH_NAME;
    static constexpr const char* simd_level_name = SCL_SIMD_LEVEL_NAME;
    static constexpr int simd_level = SCL_SIMD_LEVEL;
    static constexpr int cpp_version = SCL_CPP_VERSION;
    static constexpr bool is_64bit = SCL_ARCH_64BIT;
};

/// @brief Print platform information (for debugging)
inline void print_platform_info() {
    std::printf("SCL Platform Info:\n");
    std::printf("  Compiler: %s (version %d)\n", 
                PlatformInfo::compiler_name, PlatformInfo::compiler_version);
    std::printf("  Platform: %s\n", PlatformInfo::platform_name);
    std::printf("  Architecture: %s (%d-bit)\n", 
                PlatformInfo::arch_name, PlatformInfo::is_64bit ? 64 : 32);
    std::printf("  SIMD Level: %s (%d-bit)\n", 
                PlatformInfo::simd_level_name, PlatformInfo::simd_level);
    std::printf("  C++ Standard: C++%d\n", PlatformInfo::cpp_version);
    std::printf("  Cache Line: %zu bytes\n", cache_line_size);
    std::printf("  SIMD Width: %zu bytes\n", simd_width);
}

}  // namespace scl::platform

// =============================================================================
// SECTION 16: Source Location Support (C++20 std::source_location with fallback)
// =============================================================================

// Feature test macros
#if __has_include(<version>)
    #include <version>
#endif

// C++20 source_location detection
#if __has_include(<source_location>) && defined(__cpp_lib_source_location) && __cpp_lib_source_location >= 201907L
    #include <source_location>
    #ifndef SCL_HAS_SOURCE_LOCATION
        #define SCL_HAS_SOURCE_LOCATION 1
    #endif
#else
    #ifndef SCL_HAS_SOURCE_LOCATION
        #define SCL_HAS_SOURCE_LOCATION 0
    #endif
#endif

namespace scl {

#if SCL_HAS_SOURCE_LOCATION
/// @brief Source location information (C++20 std::source_location)
using source_location = std::source_location;
#else
/// @brief Fallback source location for pre-C++20 compilers
/// @note Uses compiler builtins to provide similar functionality
struct source_location {
private:
    const char* file_;
    const char* function_;
    std::uint32_t line_;
    std::uint32_t column_;
    
public:
    /// @brief Default constructor
    constexpr source_location() noexcept
        : file_(""), function_(""), line_(0), column_(0) {}
    
    /// @brief Constructor with location information
    constexpr source_location(
        const char* file,
        const char* func,
        std::uint32_t line,
        std::uint32_t col = 0
    ) noexcept : file_(file), function_(func), line_(line), column_(col) {}
    
    /// @brief Get current source location
    /// @return Source location at call site
    [[nodiscard]] static constexpr auto current(
        const char* file = __builtin_FILE(),
        const char* func = __builtin_FUNCTION(),
        std::uint32_t line = __builtin_LINE()
    ) noexcept -> source_location {
        return source_location{file, func, line, 0};
    }
    
    /// @brief Get file name
    [[nodiscard]] constexpr auto file_name() const noexcept -> const char* { return file_; }
    
    /// @brief Get function name
    [[nodiscard]] constexpr auto function_name() const noexcept -> const char* { return function_; }
    
    /// @brief Get line number
    [[nodiscard]] constexpr auto line() const noexcept -> std::uint32_t { return line_; }
    
    /// @brief Get column number
    [[nodiscard]] constexpr auto column() const noexcept -> std::uint32_t { return column_; }
};
#endif

/// @brief Extract filename from full path
/// @param path Full file path
/// @return Pointer to filename portion
/// @note Constexpr utility for stripping directory path
[[nodiscard]] constexpr auto filename_only(const char* path) noexcept -> const char* {
    const char* result = path;
    while (*path) {
        if (*path == '/' || *path == '\\') {
            result = path + 1;
        }
        ++path;
    }
    return result;
}

}  // namespace scl
