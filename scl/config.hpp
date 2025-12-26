#pragma once

// =============================================================================
/// @file config.hpp
/// @brief SCL Core Configuration Header
///
/// This header provides platform detection, threading backend selection, and
/// precision control for the SCL (Scientific Computing Library) core.
///
/// Architecture:
///
/// The configuration system follows a state machine pattern:
/// 1. Platform detection (OS identification)
/// 2. Threading backend selection (user-specified or platform-default)
/// 3. Validation and diagnostics
/// 4. Feature flag generation
/// 5. Precision control
///
/// Threading Backends:
///
/// Supported backends:
/// 1. SCL_BACKEND_SERIAL: Single-threaded execution
/// 2. SCL_BACKEND_OPENMP: OpenMP parallel execution
/// 3. SCL_BACKEND_TBB: Intel Threading Building Blocks
/// 4. SCL_BACKEND_BS: BS::thread_pool (header-only, zero dependency)
///
/// Precision Control:
///
/// Floating-point precision is controlled via SCL_PRECISION:
/// 1. 0: float32 (default)
/// 2. 1: float64
/// 3. 2: float16
///
/// Index precision is controlled via SCL_INDEX_PRECISION:
/// 1. 0: int16 (memory-constrained, max 32K)
/// 2. 1: int32 (standard, max 2B)
/// 3. 2: int64 (default, max 9E18, NumPy-compatible)
///
/// =============================================================================

// =============================================================================
// HWY Scalar-Only Control
// =============================================================================

/// @defgroup HWYScalarOnly HWY Scalar-Only Mode
/// @{
///
/// If SCL_ONLY_SCALAR is defined (e.g. via -DSCL_ONLY_SCALAR on the command line),
/// force-enable scalar-only mode for Highway by defining HWY_COMPILE_ONLY_SCALAR.
///

#ifdef SCL_ONLY_SCALAR
    #ifndef HWY_COMPILE_ONLY_SCALAR
        #define HWY_COMPILE_ONLY_SCALAR
    #endif
#endif

/// @}

// =============================================================================
// SECTION 1: Platform Detection
// =============================================================================

/// @defgroup PlatformDetection Platform Detection Macros
/// @{

#if defined(_WIN32) || defined(_WIN64)
    /// @brief Windows platform identifier
    #define SCL_OS_WINDOWS
#elif defined(__APPLE__) || defined(__MACH__)
    /// @brief macOS platform identifier
    #define SCL_OS_MAC
#elif defined(__linux__) || defined(__linux)
    /// @brief Linux platform identifier
    #define SCL_OS_LINUX
#else
    /// @brief Unknown platform identifier
    #define SCL_OS_UNKNOWN
#endif

/// @}

// =============================================================================
// SECTION 2: Threading Backend Selection
// =============================================================================

/// @defgroup ThreadingBackend Threading Backend Selection
/// @{
///
/// Backend selection follows this priority:
/// 1. User-explicit definition (highest priority)
/// 2. Platform-specific defaults (if no user definition)
/// 3. State machine validation (exactly one backend must be selected)
///

// --- Step 2.1: Check for user-specified backend ---
/// @brief Check if user has explicitly defined a backend
/// @note These macros are for runtime use only, not for #if directives

// --- Step 2.2: Auto-select backend based on platform if none specified ---
#if !defined(SCL_BACKEND_SERIAL) && !defined(SCL_BACKEND_TBB) && \
    !defined(SCL_BACKEND_OPENMP) && !defined(SCL_BACKEND_BS)
    #if defined(SCL_OS_MAC)
        // macOS Strategy: Prefer BS::thread_pool to avoid libomp dependency issues
        // Users can force OpenMP by defining SCL_MAC_USE_OPENMP
        #if defined(SCL_MAC_USE_OPENMP)
            #define SCL_BACKEND_OPENMP
        #else
            #define SCL_BACKEND_BS
        #endif
    #elif defined(SCL_OS_WINDOWS) || defined(SCL_OS_LINUX)
        // Windows/Linux Strategy: Default to OpenMP (industry standard for HPC)
        #define SCL_BACKEND_OPENMP
    #else
        // Unknown OS Strategy: Fallback to BS::thread_pool (zero dependency)
        #define SCL_BACKEND_BS
    #endif
#endif

// --- Step 2.3: State machine uniqueness check ---
/// @brief Validate that exactly one backend is selected

// Count how many backends are defined
#define SCL_COUNT_BACKENDS() ( \
    (defined(SCL_BACKEND_SERIAL) ? 1 : 0) + \
    (defined(SCL_BACKEND_TBB) ? 1 : 0) + \
    (defined(SCL_BACKEND_OPENMP) ? 1 : 0) + \
    (defined(SCL_BACKEND_BS) ? 1 : 0) \
)

// Check for no backend selected
#if !defined(SCL_BACKEND_SERIAL) && !defined(SCL_BACKEND_TBB) && \
    !defined(SCL_BACKEND_OPENMP) && !defined(SCL_BACKEND_BS)
    #error "SCL Configuration Error: No threading backend selected! " \
           "Please define exactly one backend: SCL_BACKEND_SERIAL, " \
           "SCL_BACKEND_TBB, SCL_BACKEND_OPENMP, or SCL_BACKEND_BS."
#endif

// Check for multiple backends selected (using explicit checks to avoid macro expansion)
#if (defined(SCL_BACKEND_SERIAL) && (defined(SCL_BACKEND_TBB) || defined(SCL_BACKEND_OPENMP) || defined(SCL_BACKEND_BS))) || \
    (defined(SCL_BACKEND_TBB) && (defined(SCL_BACKEND_OPENMP) || defined(SCL_BACKEND_BS))) || \
    (defined(SCL_BACKEND_OPENMP) && defined(SCL_BACKEND_BS))
    #error "SCL Configuration Error: Multiple threading backends defined! " \
           "Please define only one backend."
#endif

/// @}

// =============================================================================
// SECTION 3: Validation & Diagnostics
// =============================================================================

/// @defgroup Validation Validation and Diagnostics
/// @{

// --- 3.1: macOS OpenMP Warning ---
/// @brief Emit compiler warning when OpenMP is enabled on macOS
///
/// OpenMP on macOS requires libomp installation and proper linker flags.
/// This warning helps developers identify potential build issues early.
#if defined(SCL_OS_MAC) && defined(SCL_BACKEND_OPENMP)
    #pragma GCC warning "SCL_WARNING: OpenMP enabled on macOS. " \
                        "Ensure 'libomp' is installed (brew install libomp) " \
                        "and linker flags are correct."
#endif

/// @}

// =============================================================================
// SECTION 4: Feature Flags (Public API)
// =============================================================================

/// @defgroup FeatureFlags Feature Flags
/// @{
///
/// These macros provide a unified interface for querying the active backend
/// and precision settings in application code.
///

#if defined(SCL_BACKEND_OPENMP)
    /// @brief OpenMP backend is active
    #define SCL_USE_OPENMP 1
#elif defined(SCL_BACKEND_TBB)
    /// @brief TBB backend is active
    #define SCL_USE_TBB 1
#elif defined(SCL_BACKEND_BS)
    /// @brief BS::thread_pool backend is active
    #define SCL_USE_BS 1
#elif defined(SCL_BACKEND_SERIAL)
    /// @brief Serial (single-threaded) backend is active
    #define SCL_USE_SERIAL 1
#endif

/// @}

// =============================================================================
// SECTION 5: Precision Control
// =============================================================================

/// @defgroup PrecisionControl Precision Control
/// @{
///
/// Floating-point precision selection. The precision value must be defined
/// by the build system (typically CMake) before including this header.
///
/// @note Default precision is float32 if SCL_PRECISION is not defined.
///

#ifndef SCL_PRECISION
    /// @brief Default precision: float32
    #define SCL_PRECISION 0
#endif

#if SCL_PRECISION == 0
    /// @brief Use 32-bit floating-point precision
    #define SCL_USE_FLOAT32
#elif SCL_PRECISION == 1
    /// @brief Use 64-bit floating-point precision
    #define SCL_USE_FLOAT64
#elif SCL_PRECISION == 2
    /// @brief Use 16-bit floating-point precision
    #define SCL_USE_FLOAT16
#else
    #error "SCL Configuration Error: Invalid SCL_PRECISION value. " \
           "Must be 0 (f32), 1 (f64), or 2 (f16)."
#endif

/// @}

// =============================================================================
// SECTION 6: Index Precision Control
// =============================================================================

/// @defgroup IndexPrecision Index Precision Control
/// @{
///
/// Integer index type precision selection. Controls the Index type used
/// throughout SCL for array indexing, dimensions, and sparse matrix indices.
///
/// Trade-offs:
/// - int16: Max 32K elements, minimal memory, fast cache
/// - int32: Max 2B elements, standard, good balance
/// - int64: Max 9E18 elements, NumPy-compatible, future-proof
///
/// @note Default is int64 (NumPy-compatible) if not defined.
///

#ifndef SCL_INDEX_PRECISION
    /// @brief Default index precision: int64 (NumPy-compatible)
    #define SCL_INDEX_PRECISION 2
#endif

#if SCL_INDEX_PRECISION == 0
    /// @brief Use 16-bit signed integers for indexing
    ///
    /// Use case: Memory-constrained environments, small datasets (<32K elements)
    /// Max size: 32,767 (2^15 - 1)
    /// Memory: 2 bytes per index
    #define SCL_USE_INT16
#elif SCL_INDEX_PRECISION == 1
    /// @brief Use 32-bit signed integers for indexing
    ///
    /// Use case: Standard workloads, good memory/range balance
    /// Max size: 2,147,483,647 (2^31 - 1)
    /// Memory: 4 bytes per index
    #define SCL_USE_INT32
#elif SCL_INDEX_PRECISION == 2
    /// @brief Use 64-bit signed integers for indexing (default)
    ///
    /// Use case: Large-scale single-cell data, NumPy compatibility
    /// Max size: 9,223,372,036,854,775,807 (2^63 - 1)
    /// Memory: 8 bytes per index
    /// Compatibility: NumPy default, scipy.sparse compatible
    #define SCL_USE_INT64
#else
    #error "SCL Configuration Error: Invalid SCL_INDEX_PRECISION value. " \
           "Must be 0 (int16), 1 (int32), or 2 (int64)."
#endif

/// @}
