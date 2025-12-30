#pragma once

/// @file scl/config.hpp
/// @brief SCL Core Configuration - Build Configuration and Feature Detection
///
/// This header provides ONLY configuration logic:
/// - Extended type detection (int128, float16, bfloat16, float128)
/// - Threading backend auto-selection
/// - Memory, parallelization, and algorithm configuration constants
/// - Build configuration summary
///
/// @note This file does NOT depend on any other SCL headers
/// @note This file does NOT contain actual type implementations or computations

#include <cstdint>
#include <cstddef>
#include <cstdio>

// =============================================================================
// SECTION 1: Minimal Compiler/Platform Detection (Self-Contained)
// =============================================================================

// Compiler detection (minimal - just what we need for type detection)
#if defined(__clang__)
    #define SCL_CONFIG_COMPILER_CLANG 1
    #define SCL_CONFIG_COMPILER_GCC 0
    #define SCL_CONFIG_COMPILER_MSVC 0
    #define SCL_CONFIG_COMPILER_GCC_LIKE 1
#elif defined(__GNUC__)
    #define SCL_CONFIG_COMPILER_CLANG 0
    #define SCL_CONFIG_COMPILER_GCC 1
    #define SCL_CONFIG_COMPILER_MSVC 0
    #define SCL_CONFIG_COMPILER_GCC_LIKE 1
#elif defined(_MSC_VER)
    #define SCL_CONFIG_COMPILER_CLANG 0
    #define SCL_CONFIG_COMPILER_GCC 0
    #define SCL_CONFIG_COMPILER_MSVC 1
    #define SCL_CONFIG_COMPILER_GCC_LIKE 0
#else
    #define SCL_CONFIG_COMPILER_CLANG 0
    #define SCL_CONFIG_COMPILER_GCC 0
    #define SCL_CONFIG_COMPILER_MSVC 0
    #define SCL_CONFIG_COMPILER_GCC_LIKE 0
#endif

// Platform detection (minimal)
#if defined(_WIN32) || defined(_WIN64)
    #define SCL_CONFIG_PLATFORM_WINDOWS 1
    #define SCL_CONFIG_PLATFORM_MACOS 0
    #define SCL_CONFIG_PLATFORM_LINUX 0
#elif defined(__APPLE__) && defined(__MACH__)
    #define SCL_CONFIG_PLATFORM_WINDOWS 0
    #define SCL_CONFIG_PLATFORM_MACOS 1
    #define SCL_CONFIG_PLATFORM_LINUX 0
#elif defined(__linux__) || defined(__linux)
    #define SCL_CONFIG_PLATFORM_WINDOWS 0
    #define SCL_CONFIG_PLATFORM_MACOS 0
    #define SCL_CONFIG_PLATFORM_LINUX 1
#else
    #define SCL_CONFIG_PLATFORM_WINDOWS 0
    #define SCL_CONFIG_PLATFORM_MACOS 0
    #define SCL_CONFIG_PLATFORM_LINUX 0
#endif

// Architecture detection (minimal)
#if defined(__x86_64__) || defined(_M_X64)
    #define SCL_CONFIG_ARCH_X86_64 1
    #define SCL_CONFIG_ARCH_ARM64 0
    #define SCL_CONFIG_ARCH_64BIT 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define SCL_CONFIG_ARCH_X86_64 0
    #define SCL_CONFIG_ARCH_ARM64 1
    #define SCL_CONFIG_ARCH_64BIT 1
#else
    #define SCL_CONFIG_ARCH_X86_64 0
    #define SCL_CONFIG_ARCH_ARM64 0
    #define SCL_CONFIG_ARCH_64BIT 0
#endif

// SIMD detection (minimal - only what's needed for type detection)
#if defined(__AVX512F__)
    #define SCL_CONFIG_SIMD_AVX512 1
#else
    #define SCL_CONFIG_SIMD_AVX512 0
#endif

// =============================================================================
// SECTION 2: Extended Type Detection (Hardware Capability)
// =============================================================================

// -----------------------------------------------------------------------------
// 128-bit Integer Detection
// -----------------------------------------------------------------------------

#if SCL_CONFIG_COMPILER_GCC_LIKE && SCL_CONFIG_ARCH_64BIT
    #if defined(__SIZEOF_INT128__)
        #define SCL_HAS_INT128 1
    #else
        #define SCL_HAS_INT128 0
    #endif
#else
    #define SCL_HAS_INT128 0
#endif

// -----------------------------------------------------------------------------
// Float16 Detection (IEEE 754 half-precision)
// -----------------------------------------------------------------------------

// Check for native _Float16 support (C23 / compiler extension)
#if defined(__FLT16_MAX__)
    #define SCL_HAS_NATIVE_FLOAT16 1
#elif SCL_CONFIG_COMPILER_GCC_LIKE && (SCL_CONFIG_ARCH_ARM64 || (SCL_CONFIG_ARCH_X86_64 && SCL_CONFIG_SIMD_AVX512))
    #if defined(__ARM_FP16_FORMAT_IEEE) || defined(__AVX512FP16__)
        #define SCL_HAS_NATIVE_FLOAT16 1
    #else
        #define SCL_HAS_NATIVE_FLOAT16 0
    #endif
#else
    #define SCL_HAS_NATIVE_FLOAT16 0
#endif

// Software float16 is always available via bit manipulation
#define SCL_HAS_FLOAT16 1

// -----------------------------------------------------------------------------
// BFloat16 Detection (Brain Floating Point)
// -----------------------------------------------------------------------------

#if defined(__AVX512BF16__) || defined(__AMX_BF16__)
    #define SCL_HAS_NATIVE_BFLOAT16 1
#elif SCL_CONFIG_COMPILER_GCC_LIKE && defined(__ARM_FEATURE_BF16)
    #define SCL_HAS_NATIVE_BFLOAT16 1
#else
    #define SCL_HAS_NATIVE_BFLOAT16 0
#endif

// Software bfloat16 is always available
#define SCL_HAS_BFLOAT16 1

// -----------------------------------------------------------------------------
// Float128 / Long Double Detection
// -----------------------------------------------------------------------------

#if SCL_CONFIG_COMPILER_GCC_LIKE && defined(__SIZEOF_FLOAT128__)
    #define SCL_HAS_FLOAT128 1
#else
    #define SCL_HAS_FLOAT128 0
#endif

// Check if long double is extended precision (80-bit or 128-bit)
#if LDBL_MANT_DIG > DBL_MANT_DIG
    #define SCL_HAS_LONG_DOUBLE_EXTENDED 1
#else
    #define SCL_HAS_LONG_DOUBLE_EXTENDED 0
#endif

// =============================================================================
// SECTION 3: Extended Types Configuration (User Control + Auto-Detection)
// =============================================================================
//
// Usage:
//   - Define SCL_EXTENDED_TYPES=1 to force enable all detected extended types
//   - Define SCL_EXTENDED_TYPES=0 to force disable all extended types
//   - Leave undefined for auto-detection (default: enable if supported)
//
// Individual overrides (higher priority than SCL_EXTENDED_TYPES):
//   - SCL_ENABLE_INT128=0/1
//   - SCL_ENABLE_FLOAT16=0/1
//   - SCL_ENABLE_BFLOAT16=0/1
//   - SCL_ENABLE_FLOAT128=0/1
//
// =============================================================================

// -----------------------------------------------------------------------------
// Master Switch: SCL_EXTENDED_TYPES
// -----------------------------------------------------------------------------

#if !defined(SCL_EXTENDED_TYPES)
    // Auto mode: enable extended types if hardware supports them
    #define SCL_EXTENDED_TYPES 1
    #define SCL_EXTENDED_TYPES_AUTO 1
#else
    #define SCL_EXTENDED_TYPES_AUTO 0
#endif

// -----------------------------------------------------------------------------
// Int128 Enable Logic
// -----------------------------------------------------------------------------

#if defined(SCL_ENABLE_INT128)
    // User explicit override
    #if SCL_ENABLE_INT128 && !SCL_HAS_INT128
        #error "SCL_ENABLE_INT128=1 but platform does not support __int128"
    #endif
#else
    // Derive from master switch and hardware detection
    #if SCL_EXTENDED_TYPES && SCL_HAS_INT128
        #define SCL_ENABLE_INT128 1
    #else
        #define SCL_ENABLE_INT128 0
    #endif
#endif

// -----------------------------------------------------------------------------
// Float16 Enable Logic
// -----------------------------------------------------------------------------

#if defined(SCL_ENABLE_FLOAT16)
    // User explicit override (software emulation always available)
#else
    // Derive from master switch
    #if SCL_EXTENDED_TYPES
        #define SCL_ENABLE_FLOAT16 1
    #else
        #define SCL_ENABLE_FLOAT16 0
    #endif
#endif

// Prefer native implementation when available
#if SCL_ENABLE_FLOAT16 && SCL_HAS_NATIVE_FLOAT16
    #define SCL_USE_NATIVE_FLOAT16 1
#else
    #define SCL_USE_NATIVE_FLOAT16 0
#endif

// -----------------------------------------------------------------------------
// BFloat16 Enable Logic
// -----------------------------------------------------------------------------

#if defined(SCL_ENABLE_BFLOAT16)
    // User explicit override (software emulation always available)
#else
    // Derive from master switch
    #if SCL_EXTENDED_TYPES
        #define SCL_ENABLE_BFLOAT16 1
    #else
        #define SCL_ENABLE_BFLOAT16 0
    #endif
#endif

// Prefer native implementation when available
#if SCL_ENABLE_BFLOAT16 && SCL_HAS_NATIVE_BFLOAT16
    #define SCL_USE_NATIVE_BFLOAT16 1
#else
    #define SCL_USE_NATIVE_BFLOAT16 0
#endif

// -----------------------------------------------------------------------------
// Float128 Enable Logic
// -----------------------------------------------------------------------------

#if defined(SCL_ENABLE_FLOAT128)
    // User explicit override
    #if SCL_ENABLE_FLOAT128 && !SCL_HAS_FLOAT128
        #error "SCL_ENABLE_FLOAT128=1 but platform does not support __float128"
    #endif
#else
    // Derive from master switch and hardware detection
    #if SCL_EXTENDED_TYPES && SCL_HAS_FLOAT128
        #define SCL_ENABLE_FLOAT128 1
    #else
        #define SCL_ENABLE_FLOAT128 0
    #endif
#endif

// -----------------------------------------------------------------------------
// Summary: Any Extended Types Enabled?
// -----------------------------------------------------------------------------

#if SCL_ENABLE_INT128 || SCL_ENABLE_FLOAT16 || SCL_ENABLE_BFLOAT16 || SCL_ENABLE_FLOAT128
    #define SCL_HAS_ANY_EXTENDED_TYPES 1
#else
    #define SCL_HAS_ANY_EXTENDED_TYPES 0
#endif

// =============================================================================
// SECTION 4: Threading Backend Configuration
// =============================================================================

// Auto-select backend if none specified
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

// Ensure exactly one backend is selected
#define SCL_BACKEND_COUNT \
    (defined(SCL_BACKEND_SERIAL) + defined(SCL_BACKEND_TBB) + \
     defined(SCL_BACKEND_OPENMP) + defined(SCL_BACKEND_BS))

#if SCL_BACKEND_COUNT == 0
    #error "No threading backend selected"
#elif SCL_BACKEND_COUNT > 1
    #error "Multiple threading backends selected"
#endif

#undef SCL_BACKEND_COUNT

// Feature flags
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

// macOS OpenMP warning
#if SCL_CONFIG_PLATFORM_MACOS && defined(SCL_USE_OPENMP)
    #pragma message("OpenMP on macOS requires libomp (brew install libomp)")
#endif

// =============================================================================
// SECTION 5: Extended Type Aliases (Conditional on ENABLE flags)
// =============================================================================

// -----------------------------------------------------------------------------
// 128-bit Integer Types
// -----------------------------------------------------------------------------

#if SCL_ENABLE_INT128
namespace scl::types {
    /// @brief 128-bit signed integer (compiler extension)
    using int128_t = __int128;
    /// @brief 128-bit unsigned integer (compiler extension)
    using uint128_t = unsigned __int128;
}
#endif

// -----------------------------------------------------------------------------
// Native Float16 Types
// -----------------------------------------------------------------------------

#if SCL_ENABLE_FLOAT16 && SCL_USE_NATIVE_FLOAT16
namespace scl::types {
    #if defined(__FLT16_MAX__)
        /// @brief Native float16 type (_Float16)
        using float16_native_t = _Float16;
    #elif SCL_CONFIG_COMPILER_GCC_LIKE && defined(__fp16)
        /// @brief Native float16 type (__fp16)
        using float16_native_t = __fp16;
    #endif
}
#endif

// -----------------------------------------------------------------------------
// Float128 Types
// -----------------------------------------------------------------------------

#if SCL_ENABLE_FLOAT128
namespace scl::types {
    /// @brief 128-bit quad-precision float (compiler extension)
    using float128_t = __float128;
}
#endif

// =============================================================================
// SECTION 6: Memory Configuration
// =============================================================================

namespace scl::config {

/// @brief Memory configuration constants
namespace memory {
    /// @brief Default alignment (AVX-512 compatible)
    inline constexpr std::size_t alignment = 64;
    
    /// @brief Cache line size for false sharing avoidance
    inline constexpr std::size_t cache_line = 64;
    
    /// @brief Threshold for non-temporal (streaming) stores
    /// @note Below this, regular stores are faster due to cache locality
    inline constexpr std::size_t stream_threshold = 256 * 1024;  // 256 KB
    
    /// @brief Default prefetch distance (cache lines ahead)
    inline constexpr std::size_t prefetch_distance = 8;
    
    /// @brief Maximum prefetches per loop iteration
    inline constexpr std::size_t max_prefetches = 16;
    
    /// @brief Page size for huge page allocation hints
    inline constexpr std::size_t huge_page_size = 2 * 1024 * 1024;  // 2 MB
    
    /// @brief Threshold for huge page usage
    inline constexpr std::size_t huge_page_threshold = 100 * 1024 * 1024;  // 100 MB
}

/// @brief Parallelization configuration
namespace parallel {
    /// @brief Minimum elements per thread
    inline constexpr std::size_t min_elements_per_thread = 1024;
    
    /// @brief Grain size for work stealing
    inline constexpr std::size_t grain_size = 4096;
    
    /// @brief Maximum threads (0 = auto-detect)
    inline constexpr std::size_t max_threads = 0;
}

/// @brief Algorithm-specific configuration
namespace algorithm {
    /// @brief Insertion sort threshold
    inline constexpr std::size_t insertion_sort_threshold = 16;
    
    /// @brief Stack buffer threshold for avoiding heap allocation
    inline constexpr std::size_t stack_buffer_threshold = 8 * 1024;  // 8 KB
    
    /// @brief Block size for cache-blocked algorithms
    inline constexpr std::size_t default_block_size = 64;
    
    /// @brief Tile size for 2D blocking (GEMM, etc.)
    inline constexpr std::size_t tile_size_m = 64;
    inline constexpr std::size_t tile_size_n = 64;
    inline constexpr std::size_t tile_size_k = 256;
}

/// @brief Registry and reference counting configuration
namespace registry {
    /// @brief Number of shards for concurrent data structures
    inline constexpr std::size_t num_shards = 16;
    
    /// @brief Initial capacity for hash tables
    inline constexpr std::size_t initial_capacity = 256;
    
    /// @brief Maximum load factor before rehash
    inline constexpr double max_load_factor = 0.7;
}

}  // namespace scl::config

// =============================================================================
// SECTION 7: Build Configuration Summary
// =============================================================================

namespace scl::config {

/// @brief Build configuration information
struct BuildInfo {
    // Hardware detection results
    static constexpr bool has_int128 = SCL_HAS_INT128;
    static constexpr bool has_float16_native = SCL_HAS_NATIVE_FLOAT16;
    static constexpr bool has_bfloat16_native = SCL_HAS_NATIVE_BFLOAT16;
    static constexpr bool has_float128 = SCL_HAS_FLOAT128;
    static constexpr bool has_long_double_extended = SCL_HAS_LONG_DOUBLE_EXTENDED;
    
    // Enabled features (after user config + auto-detection)
    static constexpr bool extended_types_enabled = SCL_EXTENDED_TYPES;
    static constexpr bool extended_types_auto = SCL_EXTENDED_TYPES_AUTO;
    static constexpr bool int128_enabled = SCL_ENABLE_INT128;
    static constexpr bool float16_enabled = SCL_ENABLE_FLOAT16;
    static constexpr bool float16_native = SCL_USE_NATIVE_FLOAT16;
    static constexpr bool bfloat16_enabled = SCL_ENABLE_BFLOAT16;
    static constexpr bool bfloat16_native = SCL_USE_NATIVE_BFLOAT16;
    static constexpr bool float128_enabled = SCL_ENABLE_FLOAT128;
    static constexpr bool any_extended_types = SCL_HAS_ANY_EXTENDED_TYPES;
    
    // Threading backend
    static constexpr const char* threading_backend = SCL_THREADING_NAME;
    
    // Platform info
    static constexpr bool is_windows = SCL_CONFIG_PLATFORM_WINDOWS;
    static constexpr bool is_macos = SCL_CONFIG_PLATFORM_MACOS;
    static constexpr bool is_linux = SCL_CONFIG_PLATFORM_LINUX;
    static constexpr bool is_64bit = SCL_CONFIG_ARCH_64BIT;
};

inline constexpr BuildInfo build_info{};

/// @brief Print build configuration information to stdout
inline void print_build_info() {
    std::printf("SCL Build Configuration:\n");
    std::printf("  Platform: %s (%d-bit)\n",
        build_info.is_windows ? "Windows" :
        build_info.is_macos ? "macOS" :
        build_info.is_linux ? "Linux" : "Unknown",
        build_info.is_64bit ? 64 : 32);
    std::printf("  Threading: %s\n", build_info.threading_backend);
    std::printf("  Extended Types (SCL_EXTENDED_TYPES=%d, auto=%s):\n",
        build_info.extended_types_enabled ? 1 : 0,
        build_info.extended_types_auto ? "yes" : "no"
    );
}

}  // namespace scl::config
