#pragma once

// =============================================================================
/// @file config.hpp
/// @brief SCL Core Configuration Header
///
/// This header provides platform detection, threading backend selection, and
/// precision control for the SCL (Scientific Computing Library) core.
///
/// @section Architecture
///
/// The configuration system follows a state machine pattern:
/// 1. Platform detection (OS identification)
/// 2. Threading backend selection (user-specified or platform-default)
/// 3. Validation and diagnostics
/// 4. Feature flag generation
/// 5. Precision control
///
/// @section Threading Backends
///
/// Supported backends:
/// - `SCL_BACKEND_SERIAL`: Single-threaded execution
/// - `SCL_BACKEND_OPENMP`: OpenMP parallel execution
/// - `SCL_BACKEND_TBB`: Intel Threading Building Blocks
/// - `SCL_BACKEND_BS`: BS::thread_pool (header-only, zero dependency)
///
/// @section Precision Control
///
/// Floating-point precision is controlled via `SCL_PRECISION`:
/// - `0`: float32 (default)
/// - `1`: float64
/// - `2`: float16
///
/// =============================================================================

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
#define SCL_BACKEND_SERIAL_STATE   (defined(SCL_BACKEND_SERIAL)   ? 1 : 0)
#define SCL_BACKEND_TBB_STATE      (defined(SCL_BACKEND_TBB)      ? 1 : 0)
#define SCL_BACKEND_OPENMP_STATE   (defined(SCL_BACKEND_OPENMP)   ? 1 : 0)
#define SCL_BACKEND_BS_STATE       (defined(SCL_BACKEND_BS)       ? 1 : 0)
#define SCL_BACKEND_SELECTED_COUNT (SCL_BACKEND_SERIAL_STATE +     \
                                    SCL_BACKEND_TBB_STATE +        \
                                    SCL_BACKEND_OPENMP_STATE +     \
                                    SCL_BACKEND_BS_STATE)

// --- Step 2.2: Auto-select backend based on platform if none specified ---
#if SCL_BACKEND_SELECTED_COUNT == 0
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

// --- Step 2.3: Final state validation ---
/// @brief Final backend state after auto-selection (if applicable)
#define SCL_FINAL_SERIAL_STATE   (defined(SCL_BACKEND_SERIAL)   ? 1 : 0)
#define SCL_FINAL_TBB_STATE      (defined(SCL_BACKEND_TBB)      ? 1 : 0)
#define SCL_FINAL_OPENMP_STATE   (defined(SCL_BACKEND_OPENMP)   ? 1 : 0)
#define SCL_FINAL_BS_STATE       (defined(SCL_BACKEND_BS)       ? 1 : 0)
#define SCL_FINAL_SELECTED_COUNT (SCL_FINAL_SERIAL_STATE +      \
                                  SCL_FINAL_TBB_STATE +          \
                                  SCL_FINAL_OPENMP_STATE +       \
                                  SCL_FINAL_BS_STATE)

// --- Step 2.4: State machine uniqueness check ---
#if SCL_FINAL_SELECTED_COUNT == 0
    #error "SCL Configuration Error: No threading backend selected! " \
           "Please define exactly one backend: SCL_BACKEND_SERIAL, " \
           "SCL_BACKEND_TBB, SCL_BACKEND_OPENMP, or SCL_BACKEND_BS."
#elif SCL_FINAL_SELECTED_COUNT > 1
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
