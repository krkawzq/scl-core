/**
 * @file scl/include/core/type.h
 * @brief SCL Core Type Definitions for C-API
 *
 * @note This is a pure C header file (C99 compatible).
 *       Linter warnings about C++ style are expected and should be ignored.
 *
 * This header provides fundamental type definitions for the SCL C-API:
 *   - API version macros
 *   - Precision type enumerations (value types, index types)
 *   - Type query functions (runtime support detection, size, name)
 *
 * ## Design Philosophy
 *
 * This header is platform-independent and does NOT perform compile-time
 * platform detection. All precision types (including Real16/Real128) are
 * unconditionally defined in the enumeration.
 *
 * Runtime support detection is handled by the backend implementation:
 *   - Backend checks `scl::is_precision_supported<T>()`
 *   - Returns `ErrorCode::UnsupportedType` if type not available
 *   - C-API clients can query support via `scl_is_type_supported()`
 *
 * ## Type Alignment with C++ Core
 *
 * The C-API precision enumerations map directly to C++ core types:
 *   - SCL_REAL16  -> scl::Real16 (may be scl_null_t if unsupported)
 *   - SCL_REAL32  -> scl::Real32 (float)
 *   - SCL_REAL64  -> scl::Real64 (double)
 *   - SCL_REAL128 -> scl::Real128 (may be scl_null_t if unsupported)
 *   - SCL_INT8/16/32/64 -> scl::Int8/16/32/64
 *   - SCL_UINT8/16/32/64 -> scl::UInt8/16/32/64
 *   - SCL_INDEX32 -> scl::Index32 (int32_t)
 *   - SCL_INDEX64 -> scl::Index64 (int64_t)
 *
 * ## ABI Stability
 *
 * Enumeration values are fixed and will not change across versions.
 * This ensures ABI compatibility across platform and compiler boundaries.
 */

#ifndef SCL_INCLUDE_CORE_TYPE_H_
#define SCL_INCLUDE_CORE_TYPE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * SECTION 1: API Version
 * ============================================================================ */

/** @brief Major version number */
#define SCL_API_VERSION_MAJOR 1

/** @brief Minor version number */
#define SCL_API_VERSION_MINOR 0

/** @brief Patch version number */
#define SCL_API_VERSION_PATCH 0

/** @brief Combined version as single integer: (major * 10000 + minor * 100 + patch) */
#define SCL_API_VERSION \
  ((SCL_API_VERSION_MAJOR * 10000) + (SCL_API_VERSION_MINOR * 100) + \
   SCL_API_VERSION_PATCH)

/** @brief Version string */
#define SCL_API_VERSION_STRING "1.0.0"

/* ============================================================================
 * SECTION 2: Precision Type Enumerations
 * ============================================================================ */

/**
 * @brief Unified value type enumeration for numerical data
 *
 * Supports floating-point (Real), signed integer (Int), and unsigned integer (Uint).
 * All types are defined unconditionally for ABI stability.
 *
 * Encoding scheme (for efficient category/size queries):
 *   - Bits 0-3: Byte size (1, 2, 4, 8, 16)
 *   - Bits 4-7: Category (0=Real, 1=Int, 2=Uint)
 *
 * Maps to C++ types in scl/core/type.hpp:
 *   - Real types:  Real16/32/64/128
 *   - Int types:   Int8/16/32/64
 *   - Uint types:  UInt8/16/32/64
 *
 * @note Not all types are supported on all platforms. Use scl_is_type_supported()
 *       to check runtime availability.
 */
typedef enum scl_value_type_e {
  /* Floating-point types (category 0x00) */
  SCL_REAL16 = 0x02,  /**< 16-bit float (2 bytes, platform-dependent) */
  SCL_REAL32 = 0x04,  /**< 32-bit float (4 bytes) */
  SCL_REAL64 = 0x08,  /**< 64-bit double (8 bytes) - default for Real */
  SCL_REAL128 = 0x10, /**< 128-bit quad (16 bytes, platform-dependent) */

  /* Signed integer types (category 0x10) */
  SCL_INT8 = 0x11,   /**< 8-bit signed int (1 byte) */
  SCL_INT16 = 0x12,  /**< 16-bit signed int (2 bytes) */
  SCL_INT32 = 0x14,  /**< 32-bit signed int (4 bytes) - default for Int */
  SCL_INT64 = 0x18,  /**< 64-bit signed int (8 bytes) */

  /* Unsigned integer types (category 0x20) */
  SCL_UINT8 = 0x21,  /**< 8-bit unsigned int (1 byte) */
  SCL_UINT16 = 0x22, /**< 16-bit unsigned int (2 bytes) */
  SCL_UINT32 = 0x24, /**< 32-bit unsigned int (4 bytes) - default for Uint */
  SCL_UINT64 = 0x28, /**< 64-bit unsigned int (8 bytes) */
} scl_value_type_t;

/**
 * @brief Index type enumeration
 *
 * Specifies the precision of index values for array indexing operations.
 * Maps to C++ types in scl/core/type.hpp:
 *   - SCL_INDEX32 -> scl::Index32 (int32_t, 4 bytes)
 *   - SCL_INDEX64 -> scl::Index64 (int64_t, 8 bytes)
 */
typedef enum scl_index_type_e {
  SCL_INDEX32 = 0, /**< 32-bit index (int32_t) - sizeof = 4, default */
  SCL_INDEX64 = 1, /**< 64-bit index (int64_t) - sizeof = 8 */
} scl_index_type_t;

/* ============================================================================
 * SECTION 3: Default Type Configuration
 * ============================================================================ */

/**
 * @brief Default value types for each category
 *
 * These defaults are used when user doesn't specify explicit types.
 * Configuration philosophy:
 *   - Real:  64-bit (double) for numerical accuracy
 *   - Int:   32-bit (int32_t) for space efficiency
 *   - Uint:  32-bit (uint32_t) for space efficiency
 *   - Index: 32-bit (int32_t) for space efficiency
 */
#define SCL_VALUE_DEFAULT_REAL SCL_REAL64  /**< Default: double */
#define SCL_VALUE_DEFAULT_INT SCL_INT32    /**< Default: int32_t */
#define SCL_VALUE_DEFAULT_UINT SCL_UINT32  /**< Default: uint32_t */
#define SCL_INDEX_DEFAULT SCL_INDEX32      /**< Default: int32_t */

/* ============================================================================
 * SECTION 4: Type Encoding Bit Masks
 * ============================================================================ */

/** @brief Bit mask for extracting size from value type encoding */
#define SCL_TYPE_SIZE_MASK 0x0F

/** @brief Bit shift for extracting category from value type encoding */
#define SCL_TYPE_CATEGORY_SHIFT 4

/** @brief Bit mask for extracting category from value type encoding */
#define SCL_TYPE_CATEGORY_MASK 0x0F

/* ============================================================================
 * SECTION 5: Type Query Functions (Inline, Bit-Operation Optimized)
 * ============================================================================ */

/**
 * @brief Get value type category (bit operation, branch-free)
 * @param type Value type enumeration
 * @return 0=Real, 1=Int, 2=Uint
 *
 * @note Uses bit shift to extract category from encoding:
 *       category = (type >> 4) & 0x0F
 */
static inline int32_t scl_value_type_category(scl_value_type_t type) {
  return (int32_t)((type >> SCL_TYPE_CATEGORY_SHIFT) & SCL_TYPE_CATEGORY_MASK);
}

/**
 * @brief Get value type byte size (bit operation, branch-free)
 * @param type Value type enumeration
 * @return 1, 2, 4, 8, or 16 bytes
 *
 * @note Uses bit mask to extract size from encoding:
 *       size = type & 0x0F
 */
static inline int32_t scl_value_type_sizeof(scl_value_type_t type) {
  return (int32_t)(type & SCL_TYPE_SIZE_MASK);
}

/**
 * @brief Check if value type is floating-point (bit operation, branch-free)
 * @param type Value type enumeration
 * @return 1 if Real type, 0 otherwise
 *
 * @note Checks if category bits are 0: (type >> 4) == 0
 *       Returns 1 or 0 using arithmetic (no branching)
 */
static inline int32_t scl_value_type_is_real(scl_value_type_t type) {
  return ((type >> SCL_TYPE_CATEGORY_SHIFT) & SCL_TYPE_CATEGORY_MASK) == 0;
}

/**
 * @brief Check if value type is signed integer (bit operation, branch-free)
 * @param type Value type enumeration
 * @return 1 if Int type, 0 otherwise
 *
 * @note Checks if category bits are 1: (type >> 4) == 1
 */
static inline int32_t scl_value_type_is_int(scl_value_type_t type) {
  return ((type >> SCL_TYPE_CATEGORY_SHIFT) & SCL_TYPE_CATEGORY_MASK) == 1;
}

/**
 * @brief Check if value type is unsigned integer (bit operation, branch-free)
 * @param type Value type enumeration
 * @return 1 if Uint type, 0 otherwise
 *
 * @note Checks if category bits are 2: (type >> 4) == 2
 */
static inline int32_t scl_value_type_is_uint(scl_value_type_t type) {
  return ((type >> SCL_TYPE_CATEGORY_SHIFT) & SCL_TYPE_CATEGORY_MASK) == 2;
}

/**
 * @brief Check if an index type is valid (bit operation, branch-free)
 * @param type Index type enumeration
 * @return 1 if valid (0 or 1), 0 otherwise
 *
 * @note Uses bit mask: valid if (type & ~1) == 0
 *       Branchless: only INDEX32(0) and INDEX64(1) satisfy this
 */
static inline int32_t scl_is_valid_index_type(scl_index_type_t type) {
  return (type & ~1U) == 0;
}

/**
 * @brief Get index type byte size (bit operation, branch-free)
 * @param type Index type enumeration
 * @return 4 for INDEX32, 8 for INDEX64
 *
 * @note Uses bit shift: size = 4 << type (4 or 8)
 *       Branchless computation
 */
static inline size_t scl_index_type_sizeof(scl_index_type_t type) {
  return (size_t)4U << type;
}

/* ============================================================================
 * SECTION 6: Type Query Functions (Non-inline, Backend Implementation)
 * ============================================================================ */

/**
 * @brief Check if a value type is supported on this platform
 * @param type Value type enumeration
 * @return 1 if supported, 0 otherwise
 *
 * @note This function queries the backend implementation to determine
 *       runtime support for platform-dependent types (Real16, Real128).
 *       Implementation in scl/c_api/core/type.cpp
 */
int scl_is_type_supported(scl_value_type_t type);

/**
 * @brief Get value type name
 * @param type Value type enumeration
 * @return String like "Real64", "Int32", "Uint16", etc., or "Unknown"
 *
 * @note Implementation in scl/c_api/core/type.cpp
 */
const char* scl_value_type_name(scl_value_type_t type);

/**
 * @brief Get index type name
 * @param type Index type enumeration
 * @return String "Index32" or "Index64"
 *
 * @note Implementation in scl/c_api/core/type.cpp
 */
const char* scl_index_type_name(scl_index_type_t type);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SCL_INCLUDE_CORE_TYPE_H_ */
