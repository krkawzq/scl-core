/**
 * @file scl/api/core/type.h
 * @brief SCL Core Type Definitions for C-API
 *
 * This header provides fundamental type definitions for the SCL C-API:
 *   - Platform detection and compiler-specific macros
 *   - Opaque handle types (scl_sparse_t, scl_dense_t, etc.)
 *   - Precision type enumerations (real/index types)
 *   - Layout enumerations (CSR/CSC)
 *   - Version macros and compatibility checks
 *
 * ## Design Philosophy
 *
 * SCL uses opaque handles (struct pointers) for type safety at compile time.
 * The actual structure definitions are hidden in the implementation, allowing
 * internal changes without breaking ABI compatibility.
 *
 * ## Type Alignment with C++ Core
 *
 * The C-API precision enumerations map directly to C++ core types:
 *   - SCL_REAL32  -> scl::Real32 (float)
 *   - SCL_REAL64  -> scl::Real64 (double)
 *   - SCL_INDEX32 -> scl::Index32 (int32_t)
 *   - SCL_INDEX64 -> scl::Index64 (int64_t)
 *
 * ## Thread Safety
 *
 * Handle operations are thread-safe for reading. Writing to handles requires
 * external synchronization.
 *
 * ## Memory Management
 *
 * All handles created via `scl_*_create*` functions must be destroyed with
 * the corresponding `scl_*_destroy` function. Handles are not reference-counted
 * at the C-API level (internal reference counting exists for shared data).
 */

#ifndef SCL_API_CORE_TYPE_H_
#define SCL_API_CORE_TYPE_H_

#include <stddef.h>
#include <stdint.h>

/* ============================================================================
 * SECTION 0: Platform Detection
 * ============================================================================ */

/* Compiler detection */
#if defined(_MSC_VER)
    #define SCL_COMPILER_MSVC 1
    #define SCL_COMPILER_GCC 0
    #define SCL_COMPILER_CLANG 0
#elif defined(__clang__)
    #define SCL_COMPILER_MSVC 0
    #define SCL_COMPILER_GCC 0
    #define SCL_COMPILER_CLANG 1
#elif defined(__GNUC__)
    #define SCL_COMPILER_MSVC 0
    #define SCL_COMPILER_GCC 1
    #define SCL_COMPILER_CLANG 0
#else
    #define SCL_COMPILER_MSVC 0
    #define SCL_COMPILER_GCC 0
    #define SCL_COMPILER_CLANG 0
#endif

/* GCC-like compiler (GCC or Clang) */
#define SCL_COMPILER_GCC_LIKE (SCL_COMPILER_GCC || SCL_COMPILER_CLANG)

/* Architecture detection */
#if defined(__x86_64__) || defined(_M_X64)
    #define SCL_ARCH_X86_64 1
    #define SCL_ARCH_ARM64 0
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define SCL_ARCH_X86_64 0
    #define SCL_ARCH_ARM64 1
#else
    #define SCL_ARCH_X86_64 0
    #define SCL_ARCH_ARM64 0
#endif

/* OS detection */
#if defined(_WIN32) || defined(_WIN64)
    #define SCL_OS_WINDOWS 1
    #define SCL_OS_LINUX 0
    #define SCL_OS_MACOS 0
#elif defined(__linux__)
    #define SCL_OS_WINDOWS 0
    #define SCL_OS_LINUX 1
    #define SCL_OS_MACOS 0
#elif defined(__APPLE__)
    #define SCL_OS_WINDOWS 0
    #define SCL_OS_LINUX 0
    #define SCL_OS_MACOS 1
#else
    #define SCL_OS_WINDOWS 0
    #define SCL_OS_LINUX 0
    #define SCL_OS_MACOS 0
#endif

/* POSIX-like system */
#define SCL_OS_POSIX (SCL_OS_LINUX || SCL_OS_MACOS)

/* Extended type support detection */
#if defined(__SIZEOF_INT128__) && !defined(SCL_NO_INT128) && SCL_COMPILER_GCC_LIKE
    #define SCL_HAS_INT128 1
#else
    #define SCL_HAS_INT128 0
#endif

#if defined(__FLT16_MANT_DIG__) && !defined(SCL_NO_FLOAT16)
    #define SCL_HAS_FLOAT16 1
#elif defined(__ARM_FP16_FORMAT_IEEE) && !defined(SCL_NO_FLOAT16)
    #define SCL_HAS_FLOAT16 1
#else
    #define SCL_HAS_FLOAT16 0
#endif

#if defined(__SIZEOF_FLOAT128__) && !defined(SCL_NO_FLOAT128) && SCL_COMPILER_GCC_LIKE
    #define SCL_HAS_FLOAT128 1
#else
    #define SCL_HAS_FLOAT128 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * SECTION 1: API Version
 * ============================================================================ */

/** @brief Major version number */
#define SCL_API_VERSION_MAJOR 0

/** @brief Minor version number */
#define SCL_API_VERSION_MINOR 5

/** @brief Patch version number */
#define SCL_API_VERSION_PATCH 0

/** @brief Combined version as single integer: (major * 10000 + minor * 100 + patch) */
#define SCL_API_VERSION ((SCL_API_VERSION_MAJOR * 10000) + \
                         (SCL_API_VERSION_MINOR * 100) + \
                         SCL_API_VERSION_PATCH)

/** @brief Version string */
#define SCL_API_VERSION_STRING "0.5.0"

/* ============================================================================
 * SECTION 2: Opaque Handle Types
 * ============================================================================ */

/**
 * @brief Opaque handle for sparse matrix
 *
 * Represents a CSR or CSC sparse matrix with dynamic precision (Real32/64, Index32/64).
 * Create with `scl_sparse_create_*` functions, destroy with `scl_sparse_destroy`.
 */
typedef struct scl_sparse_s* scl_sparse_t;

/**
 * @brief Opaque handle for dense matrix
 *
 * Represents a row-major dense matrix with dynamic precision.
 * Create with `scl_dense_create_*` functions, destroy with `scl_dense_destroy`.
 */
typedef struct scl_dense_s* scl_dense_t;

/**
 * @brief Opaque handle for computation result
 *
 * Some operations return results that contain multiple values or require
 * staged access. Use the result accessors to retrieve data.
 */
typedef struct scl_result_s* scl_result_t;

/**
 * @brief Opaque handle for algorithm configuration
 *
 * Algorithm-specific configuration parameters.
 */
typedef struct scl_config_s* scl_config_t;

/* ============================================================================
 * SECTION 3: Null Handle Constants
 * ============================================================================ */

/** @brief Null sparse handle */
#define SCL_NULL_SPARSE  ((scl_sparse_t)0)

/** @brief Null dense handle */
#define SCL_NULL_DENSE   ((scl_dense_t)0)

/** @brief Null result handle */
#define SCL_NULL_RESULT  ((scl_result_t)0)

/** @brief Null config handle */
#define SCL_NULL_CONFIG  ((scl_config_t)0)

/* ============================================================================
 * SECTION 4: Precision Type Enumerations
 * ============================================================================ */

/**
 * @brief Real (floating-point) type enumeration
 *
 * Specifies the precision of floating-point values in matrices and operations.
 * Maps to C++ types in scl/core/type.hpp:
 *   - SCL_REAL32 -> scl::Real32 (float,  4 bytes)
 *   - SCL_REAL64 -> scl::Real64 (double, 8 bytes)
 */
typedef enum scl_real_type_e {
    SCL_REAL32 = 0,  /**< 32-bit float (float)  - sizeof = 4 */
    SCL_REAL64 = 1,  /**< 64-bit float (double) - sizeof = 8, default */
} scl_real_type_t;

/**
 * @brief Index type enumeration
 *
 * Specifies the precision of index values in sparse matrices.
 * Maps to C++ types in scl/core/type.hpp:
 *   - SCL_INDEX32 -> scl::Index32 (int32_t, 4 bytes)
 *   - SCL_INDEX64 -> scl::Index64 (int64_t, 8 bytes)
 */
typedef enum scl_index_type_e {
    SCL_INDEX32 = 0,  /**< 32-bit index (int32_t) - sizeof = 4, default */
    SCL_INDEX64 = 1,  /**< 64-bit index (int64_t) - sizeof = 8 */
} scl_index_type_t;

/**
 * @brief Matrix layout enumeration
 *
 * Specifies the storage layout of sparse matrices.
 * Maps to C++ template parameter Sparse<V, I, IsCSR>:
 *   - SCL_LAYOUT_CSR -> IsCSR = true
 *   - SCL_LAYOUT_CSC -> IsCSR = false
 */
typedef enum scl_layout_e {
    SCL_LAYOUT_CSR = 0,  /**< Compressed Sparse Row - default */
    SCL_LAYOUT_CSC = 1,  /**< Compressed Sparse Column */
} scl_layout_t;

/**
 * @brief Memory order enumeration for dense matrices
 */
typedef enum scl_order_e {
    SCL_ORDER_ROW_MAJOR = 0,  /**< Row-major (C-style) - default */
    SCL_ORDER_COL_MAJOR = 1,  /**< Column-major (Fortran-style) */
} scl_order_t;

/* ============================================================================
 * SECTION 4.1: Extended Precision Type Enumerations (Platform-Dependent)
 * ============================================================================ */

#if SCL_HAS_FLOAT16 || SCL_HAS_FLOAT128 || SCL_HAS_INT128

/**
 * @brief Extended real type enumeration (platform-dependent)
 *
 * These types may not be available on all platforms.
 * Use SCL_HAS_FLOAT16/SCL_HAS_FLOAT128 to check availability.
 */
typedef enum scl_real_type_ext_e {
    SCL_REAL_EXT_NONE = -1,  /**< No extended type */
#if SCL_HAS_FLOAT16
    SCL_REAL16 = 10,         /**< 16-bit float (_Float16) - sizeof = 2 */
#endif
#if SCL_HAS_FLOAT128
    SCL_REAL128 = 11,        /**< 128-bit float (__float128) - sizeof = 16 */
#endif
} scl_real_type_ext_t;

/**
 * @brief Extended index type enumeration (platform-dependent)
 *
 * These types may not be available on all platforms.
 * Use SCL_HAS_INT128 to check availability.
 */
typedef enum scl_index_type_ext_e {
    SCL_INDEX_EXT_NONE = -1,  /**< No extended type */
#if SCL_HAS_INT128
    SCL_INDEX128 = 10,        /**< 128-bit index (__int128) - sizeof = 16 */
#endif
} scl_index_type_ext_t;

#endif /* Extended types */

/* ============================================================================
 * SECTION 4.2: Type Compatibility Checks
 * ============================================================================ */

/**
 * @brief Check if a real type is valid (standard types only)
 * @param type Real type enumeration
 * @return 1 if valid, 0 otherwise
 */
static inline int scl_is_valid_real_type(scl_real_type_t type) {
    return (type == SCL_REAL32 || type == SCL_REAL64) ? 1 : 0;
}

/**
 * @brief Check if an index type is valid (standard types only)
 * @param type Index type enumeration
 * @return 1 if valid, 0 otherwise
 */
static inline int scl_is_valid_index_type(scl_index_type_t type) {
    return (type == SCL_INDEX32 || type == SCL_INDEX64) ? 1 : 0;
}

/**
 * @brief Check if a layout type is valid
 * @param layout Layout enumeration
 * @return 1 if valid, 0 otherwise
 */
static inline int scl_is_valid_layout(scl_layout_t layout) {
    return (layout == SCL_LAYOUT_CSR || layout == SCL_LAYOUT_CSC) ? 1 : 0;
}

/* ============================================================================
 * SECTION 5: Type Size Queries
 * ============================================================================ */

/**
 * @brief Get size in bytes for a real type
 * @param type Real type enumeration
 * @return Size in bytes (4 for REAL32, 8 for REAL64)
 */
static inline size_t scl_real_type_size(scl_real_type_t type) {
    return (type == SCL_REAL64) ? 8 : 4;
}

/**
 * @brief Get size in bytes for an index type
 * @param type Index type enumeration
 * @return Size in bytes (4 for INDEX32, 8 for INDEX64)
 */
static inline size_t scl_index_type_size(scl_index_type_t type) {
    return (type == SCL_INDEX64) ? 8 : 4;
}

/**
 * @brief Get name string for a real type
 * @param type Real type enumeration
 * @return Static string ("float32" or "float64")
 */
static inline const char* scl_real_type_name(scl_real_type_t type) {
    return (type == SCL_REAL64) ? "float64" : "float32";
}

/**
 * @brief Get name string for an index type
 * @param type Index type enumeration
 * @return Static string ("int32" or "int64")
 */
static inline const char* scl_index_type_name(scl_index_type_t type) {
    return (type == SCL_INDEX64) ? "int64" : "int32";
}

/**
 * @brief Get name string for a layout type
 * @param layout Layout enumeration
 * @return Static string ("CSR" or "CSC")
 */
static inline const char* scl_layout_name(scl_layout_t layout) {
    return (layout == SCL_LAYOUT_CSC) ? "CSC" : "CSR";
}

/* ============================================================================
 * SECTION 6: Handle Validation Macros
 * ============================================================================ */

/** @brief Check if sparse handle is valid (non-null) */
#define SCL_IS_VALID_SPARSE(h)  ((h) != SCL_NULL_SPARSE)

/** @brief Check if dense handle is valid (non-null) */
#define SCL_IS_VALID_DENSE(h)   ((h) != SCL_NULL_DENSE)

/** @brief Check if result handle is valid (non-null) */
#define SCL_IS_VALID_RESULT(h)  ((h) != SCL_NULL_RESULT)

/** @brief Check if config handle is valid (non-null) */
#define SCL_IS_VALID_CONFIG(h)  ((h) != SCL_NULL_CONFIG)

/* ============================================================================
 * SECTION 7: Precision Constraint Macros
 * ============================================================================ */

/**
 * @brief Mark a function as requiring REAL64 precision
 *
 * Use this to document functions that only support double precision.
 * Example: SCL_REQUIRES_REAL64 scl_sparse_eigenvalues(...);
 */
#define SCL_REQUIRES_REAL64  /* Documentation marker */

/**
 * @brief Mark a function as requiring INDEX64 precision
 *
 * Use this to document functions that only support 64-bit indices.
 * Example: SCL_REQUIRES_INDEX64 scl_sparse_large_scale_op(...);
 */
#define SCL_REQUIRES_INDEX64  /* Documentation marker */

/**
 * @brief Mark a function as supporting only CSR layout
 */
#define SCL_REQUIRES_CSR  /* Documentation marker */

/**
 * @brief Mark a function as supporting only CSC layout
 */
#define SCL_REQUIRES_CSC  /* Documentation marker */

/* ============================================================================
 * SECTION 8: Buffer Strategy
 * ============================================================================ */

/**
 * @brief Buffer allocation strategy for sparse matrix creation
 */
typedef enum scl_buffer_strategy_e {
    SCL_BUFFER_AUTO = 0,        /**< Automatic selection based on matrix size */
    SCL_BUFFER_FRAGMENTED = 1,  /**< Each row/column uses separate buffer */
    SCL_BUFFER_SINGLE = 2,      /**< All data in one contiguous buffer */
    SCL_BUFFER_MIN_SIZE = 3,    /**< Group by minimum buffer size */
    SCL_BUFFER_COUNT = 4,       /**< Divide into N buffers */
} scl_buffer_strategy_t;

/**
 * @brief Buffer strategy configuration
 */
typedef struct scl_buffer_config_s {
    scl_buffer_strategy_t strategy;  /**< Strategy type */
    size_t param;                    /**< Strategy parameter (buffer size or count) */
} scl_buffer_config_t;

/** @brief Default buffer configuration (auto strategy) */
#define SCL_BUFFER_CONFIG_DEFAULT ((scl_buffer_config_t){SCL_BUFFER_AUTO, 0})

#ifdef __cplusplus
}  /* extern "C" */
#endif

/* ============================================================================
 * SECTION 9: C++ Internal Dispatch Macros (Implementation Only)
 * ============================================================================ */

#ifdef __cplusplus

/**
 * @brief Compute type index for dispatch (internal use)
 *
 * Creates a combined index from real_type and index_type:
 *   - 0: Real32, Index32
 *   - 1: Real64, Index32
 *   - 2: Real32, Index64
 *   - 3: Real64, Index64
 */
#define SCL_TYPE_INDEX(real_type, index_type) \
    (static_cast<int>(real_type) | (static_cast<int>(index_type) << 1))

/**
 * @brief Dispatch macro for 2x2 real/index type combinations
 *
 * Usage:
 *   SCL_DISPATCH_REAL_INDEX(handle->real_type, handle->index_type, {
 *       using RealT = SCL_REAL_TYPE;
 *       using IndexT = SCL_INDEX_TYPE;
 *       // ... use RealT and IndexT
 *   });
 */
#define SCL_DISPATCH_REAL_INDEX(real_type, index_type, BLOCK) \
    do { \
        switch (SCL_TYPE_INDEX(real_type, index_type)) { \
            case 0: { \
                using SCL_REAL_TYPE = float; \
                using SCL_INDEX_TYPE = std::int32_t; \
                BLOCK \
            } break; \
            case 1: { \
                using SCL_REAL_TYPE = double; \
                using SCL_INDEX_TYPE = std::int32_t; \
                BLOCK \
            } break; \
            case 2: { \
                using SCL_REAL_TYPE = float; \
                using SCL_INDEX_TYPE = std::int64_t; \
                BLOCK \
            } break; \
            case 3: { \
                using SCL_REAL_TYPE = double; \
                using SCL_INDEX_TYPE = std::int64_t; \
                BLOCK \
            } break; \
            default: break; \
        } \
    } while (0)

/**
 * @brief Dispatch macro for layout (CSR/CSC)
 *
 * Usage:
 *   SCL_DISPATCH_LAYOUT(handle->layout, {
 *       constexpr bool IsCSR = SCL_IS_CSR;
 *       // ... use IsCSR
 *   });
 */
#define SCL_DISPATCH_LAYOUT(layout, BLOCK) \
    do { \
        if ((layout) == SCL_LAYOUT_CSR) { \
            constexpr bool SCL_IS_CSR = true; \
            BLOCK \
        } else { \
            constexpr bool SCL_IS_CSR = false; \
            BLOCK \
        } \
    } while (0)

/**
 * @brief Full dispatch macro for sparse matrices (2x2x2 = 8 combinations)
 *
 * Usage:
 *   SCL_DISPATCH_SPARSE(handle->real_type, handle->index_type, handle->layout, {
 *       using RealT = SCL_REAL_TYPE;
 *       using IndexT = SCL_INDEX_TYPE;
 *       constexpr bool IsCSR = SCL_IS_CSR;
 *       // ... use types
 *   });
 */
#define SCL_DISPATCH_SPARSE(real_type, index_type, layout, BLOCK) \
    SCL_DISPATCH_LAYOUT(layout, { \
        SCL_DISPATCH_REAL_INDEX(real_type, index_type, BLOCK); \
    })

/**
 * @brief Get variant index for sparse matrix storage
 *
 * Maps (real_type, index_type, layout) to variant index [0-7]:
 *   - 0: CSR<Real32, Index32>
 *   - 1: CSR<Real64, Index32>
 *   - 2: CSR<Real32, Index64>
 *   - 3: CSR<Real64, Index64>
 *   - 4: CSC<Real32, Index32>
 *   - 5: CSC<Real64, Index32>
 *   - 6: CSC<Real32, Index64>
 *   - 7: CSC<Real64, Index64>
 */
#define SCL_SPARSE_VARIANT_INDEX(real_type, index_type, layout) \
    (SCL_TYPE_INDEX(real_type, index_type) | (static_cast<int>(layout) << 2))

#endif  /* __cplusplus */

#endif  /* SCL_API_CORE_TYPE_H_ */

