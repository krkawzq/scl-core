#pragma once

/**
 * @file scl/config.hpp
 * @brief SCL Core Configuration Header
 *
 * This header provides ONLY configuration and detection logic:
 * - C++ standard version detection
 * - Compiler detection (GCC, Clang, MSVC) with version info
 * - Platform detection (Windows, macOS, Linux)
 * - Architecture detection (x86_64, ARM64)
 * - SIMD instruction set detection (AVX512, AVX2, AVX, SSE, NEON)
 * - Extended type detection (int128, float16, float128)
 * - Threading backend configuration
 *
 * @note This file does NOT depend on any other SCL headers
 */

#include <cstddef>
#include <cstdint>

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

#if defined(__clang__)
  #define SCL_CONFIG_COMPILER_CLANG 1
  #define SCL_CONFIG_COMPILER_GCC 0
  #define SCL_CONFIG_COMPILER_MSVC 0
  #define SCL_CONFIG_COMPILER_GCC_LIKE 1
  #define SCL_CONFIG_COMPILER_VERSION \
    (__clang_major__ * 10000 + __clang_minor__ * 100 + __clang_patchlevel__)
  #define SCL_CONFIG_COMPILER_NAME "Clang"
#elif defined(__GNUC__)
  #define SCL_CONFIG_COMPILER_CLANG 0
  #define SCL_CONFIG_COMPILER_GCC 1
  #define SCL_CONFIG_COMPILER_MSVC 0
  #define SCL_CONFIG_COMPILER_GCC_LIKE 1
  #define SCL_CONFIG_COMPILER_VERSION \
    (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
  #define SCL_CONFIG_COMPILER_NAME "GCC"
#elif defined(_MSC_VER)
  #define SCL_CONFIG_COMPILER_CLANG 0
  #define SCL_CONFIG_COMPILER_GCC 0
  #define SCL_CONFIG_COMPILER_MSVC 1
  #define SCL_CONFIG_COMPILER_GCC_LIKE 0
  #define SCL_CONFIG_COMPILER_VERSION _MSC_VER
  #define SCL_CONFIG_COMPILER_NAME "MSVC"
#else
  #define SCL_CONFIG_COMPILER_CLANG 0
  #define SCL_CONFIG_COMPILER_GCC 0
  #define SCL_CONFIG_COMPILER_MSVC 0
  #define SCL_CONFIG_COMPILER_GCC_LIKE 0
  #define SCL_CONFIG_COMPILER_VERSION 0
  #define SCL_CONFIG_COMPILER_NAME "Unknown"
#endif

// --- Minimum Compiler Version Checks ---

#if SCL_CONFIG_COMPILER_GCC && SCL_CONFIG_COMPILER_VERSION < 110000
  #error "scl-core requires GCC 11 or later"
#endif
#if SCL_CONFIG_COMPILER_CLANG && SCL_CONFIG_COMPILER_VERSION < 140000
  #error "scl-core requires Clang 14 or later"
#endif
#if SCL_CONFIG_COMPILER_MSVC && SCL_CONFIG_COMPILER_VERSION < 1929
  #error "scl-core requires MSVC 19.29 or later"
#endif

// =============================================================================
// SECTION 2: Platform Detection
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
  #define SCL_CONFIG_PLATFORM_WINDOWS 1
  #define SCL_CONFIG_PLATFORM_POSIX 0
  #define SCL_CONFIG_PLATFORM_MACOS 0
  #define SCL_CONFIG_PLATFORM_LINUX 0
  #define SCL_CONFIG_PLATFORM_NAME "Windows"
#elif defined(__APPLE__) && defined(__MACH__)
  #define SCL_CONFIG_PLATFORM_WINDOWS 0
  #define SCL_CONFIG_PLATFORM_POSIX 1
  #define SCL_CONFIG_PLATFORM_MACOS 1
  #define SCL_CONFIG_PLATFORM_LINUX 0
  #define SCL_CONFIG_PLATFORM_NAME "macOS"
#elif defined(__linux__) || defined(__linux)
  #define SCL_CONFIG_PLATFORM_WINDOWS 0
  #define SCL_CONFIG_PLATFORM_POSIX 1
  #define SCL_CONFIG_PLATFORM_MACOS 0
  #define SCL_CONFIG_PLATFORM_LINUX 1
  #define SCL_CONFIG_PLATFORM_NAME "Linux"
#elif defined(__unix__) || defined(__unix)
  #define SCL_CONFIG_PLATFORM_WINDOWS 0
  #define SCL_CONFIG_PLATFORM_POSIX 1
  #define SCL_CONFIG_PLATFORM_MACOS 0
  #define SCL_CONFIG_PLATFORM_LINUX 0
  #define SCL_CONFIG_PLATFORM_NAME "Unix"
#else
  #define SCL_CONFIG_PLATFORM_WINDOWS 0
  #define SCL_CONFIG_PLATFORM_POSIX 0
  #define SCL_CONFIG_PLATFORM_MACOS 0
  #define SCL_CONFIG_PLATFORM_LINUX 0
  #define SCL_CONFIG_PLATFORM_NAME "Unknown"
#endif

// =============================================================================
// SECTION 3: Architecture Detection
// =============================================================================

#if defined(__x86_64__) || defined(_M_X64)
  #define SCL_CONFIG_ARCH_X86_64 1
  #define SCL_CONFIG_ARCH_X86 1
  #define SCL_CONFIG_ARCH_ARM64 0
  #define SCL_CONFIG_ARCH_ARM 0
  #define SCL_CONFIG_ARCH_64BIT 1
  #define SCL_CONFIG_ARCH_NAME "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
  #define SCL_CONFIG_ARCH_X86_64 0
  #define SCL_CONFIG_ARCH_X86 1
  #define SCL_CONFIG_ARCH_ARM64 0
  #define SCL_CONFIG_ARCH_ARM 0
  #define SCL_CONFIG_ARCH_64BIT 0
  #define SCL_CONFIG_ARCH_NAME "x86"
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define SCL_CONFIG_ARCH_X86_64 0
  #define SCL_CONFIG_ARCH_X86 0
  #define SCL_CONFIG_ARCH_ARM64 1
  #define SCL_CONFIG_ARCH_ARM 1
  #define SCL_CONFIG_ARCH_64BIT 1
  #define SCL_CONFIG_ARCH_NAME "ARM64"
#elif defined(__arm__) || defined(_M_ARM)
  #define SCL_CONFIG_ARCH_X86_64 0
  #define SCL_CONFIG_ARCH_X86 0
  #define SCL_CONFIG_ARCH_ARM64 0
  #define SCL_CONFIG_ARCH_ARM 1
  #define SCL_CONFIG_ARCH_64BIT 0
  #define SCL_CONFIG_ARCH_NAME "ARM"
#else
  #define SCL_CONFIG_ARCH_X86_64 0
  #define SCL_CONFIG_ARCH_X86 0
  #define SCL_CONFIG_ARCH_ARM64 0
  #define SCL_CONFIG_ARCH_ARM 0
  #define SCL_CONFIG_ARCH_64BIT 0
  #define SCL_CONFIG_ARCH_NAME "Unknown"
#endif

// =============================================================================
// SECTION 4: SIMD Detection
// =============================================================================

#if SCL_CONFIG_ARCH_X86

  // --- x86 AVX-512 Family ---

  #if defined(__AVX512F__)
    #define SCL_CONFIG_SIMD_AVX512 1
    #define SCL_CONFIG_SIMD_AVX512F 1
  #else
    #define SCL_CONFIG_SIMD_AVX512 0
    #define SCL_CONFIG_SIMD_AVX512F 0
  #endif

  #if defined(__AVX512BW__)
    #define SCL_CONFIG_SIMD_AVX512BW 1
  #else
    #define SCL_CONFIG_SIMD_AVX512BW 0
  #endif

  #if defined(__AVX512DQ__)
    #define SCL_CONFIG_SIMD_AVX512DQ 1
  #else
    #define SCL_CONFIG_SIMD_AVX512DQ 0
  #endif

  #if defined(__AVX512VL__)
    #define SCL_CONFIG_SIMD_AVX512VL 1
  #else
    #define SCL_CONFIG_SIMD_AVX512VL 0
  #endif

  // --- x86 AVX Family ---

  #if defined(__AVX2__)
    #define SCL_CONFIG_SIMD_AVX2 1
  #else
    #define SCL_CONFIG_SIMD_AVX2 0
  #endif

  #if defined(__AVX__)
    #define SCL_CONFIG_SIMD_AVX 1
  #else
    #define SCL_CONFIG_SIMD_AVX 0
  #endif

  #if defined(__FMA__)
    #define SCL_CONFIG_SIMD_FMA 1
  #else
    #define SCL_CONFIG_SIMD_FMA 0
  #endif

  // --- x86 SSE Family ---

  #if defined(__SSE4_2__)
    #define SCL_CONFIG_SIMD_SSE4_2 1
  #else
    #define SCL_CONFIG_SIMD_SSE4_2 0
  #endif

  #if defined(__SSE4_1__)
    #define SCL_CONFIG_SIMD_SSE4_1 1
  #else
    #define SCL_CONFIG_SIMD_SSE4_1 0
  #endif

  #if defined(__SSE3__)
    #define SCL_CONFIG_SIMD_SSE3 1
  #else
    #define SCL_CONFIG_SIMD_SSE3 0
  #endif

  #if defined(__SSE2__) || SCL_CONFIG_ARCH_X86_64
    #define SCL_CONFIG_SIMD_SSE2 1
  #else
    #define SCL_CONFIG_SIMD_SSE2 0
  #endif

  // ARM SIMD not available on x86
  #define SCL_CONFIG_SIMD_NEON 0
  #define SCL_CONFIG_SIMD_SVE 0
  #define SCL_CONFIG_SIMD_SVE2 0

#elif SCL_CONFIG_ARCH_ARM

  // No x86 SIMD on ARM
  #define SCL_CONFIG_SIMD_AVX512 0
  #define SCL_CONFIG_SIMD_AVX512F 0
  #define SCL_CONFIG_SIMD_AVX512BW 0
  #define SCL_CONFIG_SIMD_AVX512DQ 0
  #define SCL_CONFIG_SIMD_AVX512VL 0
  #define SCL_CONFIG_SIMD_AVX2 0
  #define SCL_CONFIG_SIMD_AVX 0
  #define SCL_CONFIG_SIMD_FMA 0
  #define SCL_CONFIG_SIMD_SSE4_2 0
  #define SCL_CONFIG_SIMD_SSE4_1 0
  #define SCL_CONFIG_SIMD_SSE3 0
  #define SCL_CONFIG_SIMD_SSE2 0

  // --- ARM NEON ---

  #if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define SCL_CONFIG_SIMD_NEON 1
  #else
    #define SCL_CONFIG_SIMD_NEON 0
  #endif

  // --- ARM SVE ---

  #if defined(__ARM_FEATURE_SVE)
    #define SCL_CONFIG_SIMD_SVE 1
  #else
    #define SCL_CONFIG_SIMD_SVE 0
  #endif

  #if defined(__ARM_FEATURE_SVE2)
    #define SCL_CONFIG_SIMD_SVE2 1
  #else
    #define SCL_CONFIG_SIMD_SVE2 0
  #endif

#else

  // Unknown architecture - no SIMD
  #define SCL_CONFIG_SIMD_AVX512 0
  #define SCL_CONFIG_SIMD_AVX512F 0
  #define SCL_CONFIG_SIMD_AVX512BW 0
  #define SCL_CONFIG_SIMD_AVX512DQ 0
  #define SCL_CONFIG_SIMD_AVX512VL 0
  #define SCL_CONFIG_SIMD_AVX2 0
  #define SCL_CONFIG_SIMD_AVX 0
  #define SCL_CONFIG_SIMD_FMA 0
  #define SCL_CONFIG_SIMD_SSE4_2 0
  #define SCL_CONFIG_SIMD_SSE4_1 0
  #define SCL_CONFIG_SIMD_SSE3 0
  #define SCL_CONFIG_SIMD_SSE2 0
  #define SCL_CONFIG_SIMD_NEON 0
  #define SCL_CONFIG_SIMD_SVE 0
  #define SCL_CONFIG_SIMD_SVE2 0

#endif

// --- Best Available SIMD Level ---

#if SCL_CONFIG_SIMD_AVX512
  #define SCL_CONFIG_SIMD_LEVEL 512
  #define SCL_CONFIG_SIMD_LEVEL_NAME "AVX-512"
#elif SCL_CONFIG_SIMD_AVX2
  #define SCL_CONFIG_SIMD_LEVEL 256
  #define SCL_CONFIG_SIMD_LEVEL_NAME "AVX2"
#elif SCL_CONFIG_SIMD_AVX
  #define SCL_CONFIG_SIMD_LEVEL 256
  #define SCL_CONFIG_SIMD_LEVEL_NAME "AVX"
#elif SCL_CONFIG_SIMD_SSE4_2
  #define SCL_CONFIG_SIMD_LEVEL 128
  #define SCL_CONFIG_SIMD_LEVEL_NAME "SSE4.2"
#elif SCL_CONFIG_SIMD_SSE2
  #define SCL_CONFIG_SIMD_LEVEL 128
  #define SCL_CONFIG_SIMD_LEVEL_NAME "SSE2"
#elif SCL_CONFIG_SIMD_SVE2
  #define SCL_CONFIG_SIMD_LEVEL 2048
  #define SCL_CONFIG_SIMD_LEVEL_NAME "SVE2"
#elif SCL_CONFIG_SIMD_SVE
  #define SCL_CONFIG_SIMD_LEVEL 2048
  #define SCL_CONFIG_SIMD_LEVEL_NAME "SVE"
#elif SCL_CONFIG_SIMD_NEON
  #define SCL_CONFIG_SIMD_LEVEL 128
  #define SCL_CONFIG_SIMD_LEVEL_NAME "NEON"
#else
  #define SCL_CONFIG_SIMD_LEVEL 64
  #define SCL_CONFIG_SIMD_LEVEL_NAME "Scalar"
#endif

// =============================================================================
// SECTION 5: Extended Type Detection
// =============================================================================

// --- 128-bit Integer Detection ---

#if SCL_CONFIG_COMPILER_GCC_LIKE && SCL_CONFIG_ARCH_64BIT
  #define SCL_HAS_INT128 1
#else
  #define SCL_HAS_INT128 0
#endif

// --- Float16 Detection (IEEE 754 half-precision) ---

#if defined(__FLT16_MAX__)
  #define SCL_HAS_NATIVE_FLOAT16 1
#else
  #define SCL_HAS_NATIVE_FLOAT16 0
#endif

// --- Float128 Detection ---

#if SCL_CONFIG_COMPILER_GCC_LIKE && defined(__SIZEOF_FLOAT128__)
  #define SCL_HAS_FLOAT128 1
#else
  #define SCL_HAS_FLOAT128 0
#endif

// =============================================================================
// SECTION 6: Configure Extended Types
// =============================================================================

#if defined(SCL_ENABLE_EXTENDED_TYPES)
  #if SCL_HAS_INT128
    #define SCL_ENABLE_INT128 1
  #else
    #define SCL_ENABLE_INT128 0
  #endif
  #if SCL_HAS_NATIVE_FLOAT16
    #define SCL_ENABLE_FLOAT16 1
  #else
    #define SCL_ENABLE_FLOAT16 0
  #endif
  #if SCL_HAS_FLOAT128
    #define SCL_ENABLE_FLOAT128 1
  #else
    #define SCL_ENABLE_FLOAT128 0
  #endif
#else
  #define SCL_ENABLE_INT128 0
  #define SCL_ENABLE_FLOAT16 0
  #define SCL_ENABLE_FLOAT128 0
#endif

// =============================================================================
// SECTION 7: Configure Threading Backend
// =============================================================================

// --- Auto-select backend if none specified ---

#if !defined(SCL_BACKEND_SERIAL) && !defined(SCL_BACKEND_TBB) && \
  !defined(SCL_BACKEND_OPENMP) && !defined(SCL_BACKEND_BS)
  #if SCL_CONFIG_PLATFORM_MACOS
    // macOS: Prefer thread pool to avoid libomp issues
    #if defined(SCL_FORCE_OPENMP)
      #define SCL_BACKEND_OPENMP 1
    #else
      #define SCL_BACKEND_BS 1
    #endif
  #elif SCL_CONFIG_PLATFORM_WINDOWS || SCL_CONFIG_PLATFORM_LINUX
    #define SCL_BACKEND_OPENMP 1
  #else
    #define SCL_BACKEND_BS 1
  #endif
#endif

// --- Ensure exactly one backend is selected ---

#if defined(SCL_BACKEND_SERIAL) + defined(SCL_BACKEND_TBB) + \
  defined(SCL_BACKEND_OPENMP) + defined(SCL_BACKEND_BS) == 0
  #error "No threading backend selected"
#elif defined(SCL_BACKEND_SERIAL) + defined(SCL_BACKEND_TBB) + \
    defined(SCL_BACKEND_OPENMP) + defined(SCL_BACKEND_BS) > 1
  #error "Multiple threading backends selected"
#endif

// --- Feature flags ---

#if defined(SCL_BACKEND_OPENMP)
  #define SCL_USE_OPENMP 1
  #define SCL_THREADING_NAME "OpenMP"
#elif defined(SCL_BACKEND_TBB)
  #define SCL_USE_TBB 1
  #define SCL_THREADING_NAME "TBB"
#elif defined(SCL_BACKEND_BS)
  #define SCL_USE_BS 1
  #define SCL_THREADING_NAME "BS::thread_pool"
#elif defined(SCL_BACKEND_SERIAL)
  #define SCL_USE_SERIAL 1
  #define SCL_THREADING_NAME "Serial"
#endif

// --- macOS OpenMP warning ---

#if SCL_CONFIG_PLATFORM_MACOS && defined(SCL_USE_OPENMP)
  #pragma message("OpenMP on macOS requires libomp (brew install libomp)")
#endif
