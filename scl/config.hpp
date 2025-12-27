#pragma once

// =============================================================================
// FILE: scl/config.hpp
// BRIEF: SCL Core Configuration Header
// =============================================================================

// =============================================================================
// HWY Scalar-Only Control
// =============================================================================

#ifdef SCL_ONLY_SCALAR
    #ifndef HWY_COMPILE_ONLY_SCALAR
        #define HWY_COMPILE_ONLY_SCALAR
    #endif
#endif

// =============================================================================
// Platform Detection
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
    #define SCL_OS_WINDOWS
#elif defined(__APPLE__) || defined(__MACH__)
    #define SCL_OS_MAC
#elif defined(__linux__) || defined(__linux)
    #define SCL_OS_LINUX
#else
    #define SCL_OS_UNKNOWN
#endif

// =============================================================================
// Threading Backend Selection
// =============================================================================

// Auto-select backend based on platform if none specified
#if !defined(SCL_BACKEND_SERIAL) && !defined(SCL_BACKEND_TBB) && \
    !defined(SCL_BACKEND_OPENMP) && !defined(SCL_BACKEND_BS)
    #if defined(SCL_OS_MAC)
        // macOS: Prefer BS::thread_pool to avoid libomp dependency issues
        // Users can force OpenMP by defining SCL_MAC_USE_OPENMP
        #if defined(SCL_MAC_USE_OPENMP)
            #define SCL_BACKEND_OPENMP
        #else
            #define SCL_BACKEND_BS
        #endif
    #elif defined(SCL_OS_WINDOWS) || defined(SCL_OS_LINUX)
        // Windows/Linux: Default to OpenMP (industry standard for HPC)
        #define SCL_BACKEND_OPENMP
    #else
        // Unknown OS: Fallback to BS::thread_pool (zero dependency)
        #define SCL_BACKEND_BS
    #endif
#endif

// State machine uniqueness check: exactly one backend must be selected
#if !defined(SCL_BACKEND_SERIAL) && !defined(SCL_BACKEND_TBB) && \
    !defined(SCL_BACKEND_OPENMP) && !defined(SCL_BACKEND_BS)
    #error "SCL Configuration Error: No threading backend selected! " \
           "Please define exactly one backend: SCL_BACKEND_SERIAL, " \
           "SCL_BACKEND_TBB, SCL_BACKEND_OPENMP, or SCL_BACKEND_BS."
#endif

// Check for multiple backends selected
#if (defined(SCL_BACKEND_SERIAL) && (defined(SCL_BACKEND_TBB) || defined(SCL_BACKEND_OPENMP) || defined(SCL_BACKEND_BS))) || \
    (defined(SCL_BACKEND_TBB) && (defined(SCL_BACKEND_OPENMP) || defined(SCL_BACKEND_BS))) || \
    (defined(SCL_BACKEND_OPENMP) && defined(SCL_BACKEND_BS))
    #error "SCL Configuration Error: Multiple threading backends defined! " \
           "Please define only one backend."
#endif

// =============================================================================
// Validation & Diagnostics
// =============================================================================

// macOS OpenMP Warning: Ensure libomp is installed
#if defined(SCL_OS_MAC) && defined(SCL_BACKEND_OPENMP)
    #pragma GCC warning "SCL_WARNING: OpenMP enabled on macOS. " \
                        "Ensure 'libomp' is installed (brew install libomp) " \
                        "and linker flags are correct."
#endif

// =============================================================================
// Feature Flags (Public API)
// =============================================================================

#if defined(SCL_BACKEND_OPENMP)
    #define SCL_USE_OPENMP 1
#elif defined(SCL_BACKEND_TBB)
    #define SCL_USE_TBB 1
#elif defined(SCL_BACKEND_BS)
    #define SCL_USE_BS 1
#elif defined(SCL_BACKEND_SERIAL)
    #define SCL_USE_SERIAL 1
#endif

// =============================================================================
// Precision Control
// =============================================================================

// Floating-point precision selection
// 0: float32 (default)
// 1: float64
// 2: float16
#ifndef SCL_PRECISION
    #define SCL_PRECISION 0
#endif

#if SCL_PRECISION == 0
    #define SCL_USE_FLOAT32
#elif SCL_PRECISION == 1
    #define SCL_USE_FLOAT64
#elif SCL_PRECISION == 2
    #define SCL_USE_FLOAT16
#else
    #error "SCL Configuration Error: Invalid SCL_PRECISION value. " \
           "Must be 0 (f32), 1 (f64), or 2 (f16)."
#endif

// =============================================================================
// Index Precision Control
// =============================================================================

// Integer index type precision selection
// 0: int16 - Max 32K elements, minimal memory
// 1: int32 - Max 2B elements, standard
// 2: int64 - Max 9E18 elements, NumPy-compatible (default)
#ifndef SCL_INDEX_PRECISION
    #define SCL_INDEX_PRECISION 2
#endif

#if SCL_INDEX_PRECISION == 0
    #define SCL_USE_INT16
#elif SCL_INDEX_PRECISION == 1
    #define SCL_USE_INT32
#elif SCL_INDEX_PRECISION == 2
    #define SCL_USE_INT64
#else
    #error "SCL Configuration Error: Invalid SCL_INDEX_PRECISION value. " \
           "Must be 0 (int16), 1 (int32), or 2 (int64)."
#endif
