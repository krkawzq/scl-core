/**
 * @file scl/api/core/dispatch.h
 * @brief Extended Dispatch Macros for SCL C-API
 *
 * This header extends the basic dispatch macros from type.h with:
 *   - Version-constrained dispatch with automatic type checking
 *   - Type compatibility checking macros
 *   - Binary operation dispatch helpers
 *   - Single-type dispatch variants
 *
 * ## Basic Dispatch Macros (from type.h)
 *
 * The following macros are defined in type.h and re-exported here:
 *   - SCL_TYPE_INDEX(real_type, index_type)
 *   - SCL_DISPATCH_REAL_INDEX(real_type, index_type, BLOCK)
 *   - SCL_DISPATCH_LAYOUT(layout, BLOCK)
 *   - SCL_DISPATCH_SPARSE(real_type, index_type, layout, BLOCK)
 *   - SCL_SPARSE_VARIANT_INDEX(real_type, index_type, layout)
 *
 * ## Extended Dispatch (this header)
 *
 *   - SCL_DISPATCH_REAL(real_type, BLOCK) - Real-only dispatch
 *   - SCL_DISPATCH_INDEX(index_type, BLOCK) - Index-only dispatch
 *   - SCL_DISPATCH_REAL64_ONLY(...) - Constrained to Real64
 *   - SCL_DISPATCH_INDEX64_ONLY(...) - Constrained to Index64
 *   - SCL_DISPATCH_CSR_ONLY(...) - Constrained to CSR layout
 *   - SCL_DISPATCH_CSC_ONLY(...) - Constrained to CSC layout
 *   - SCL_CHECK_SAME_*_TYPE(h1, h2) - Type compatibility checks
 *   - SCL_DISPATCH_BINARY_SPARSE(h1, h2, BLOCK) - Binary operation dispatch
 *
 * ## Performance Note
 *
 * Dispatch overhead is minimal (single switch on small enum values) and occurs
 * once per API call. The actual computation runs at full native speed.
 */

#ifndef SCL_API_CORE_DISPATCH_H_
#define SCL_API_CORE_DISPATCH_H_

#include "type.h"

/* Basic dispatch macros are now in type.h */
/* This header provides extended dispatch utilities */

#ifdef __cplusplus

#include <cstdint>
#include <type_traits>

/* ============================================================================
 * SECTION 1: Single-Type Dispatch Macros
 * ============================================================================ */

/**
 * @brief Dispatch based on value type (10 combinations)
 *
 * Defines within BLOCK:
 *   - SCL_VALUE_TYPE: float, double, int8_t, int16_t, int32_t, int64_t,
 *                     uint8_t, uint16_t, uint32_t, or uint64_t
 *
 * @note Uses variadic macro to handle code blocks containing commas
 */
#define SCL_DISPATCH_VALUE(value_type, ...) \
    do { \
        switch (value_type) { \
            case SCL_REAL32:  { using SCL_VALUE_TYPE = float;           __VA_ARGS__; } break; \
            case SCL_REAL64:  { using SCL_VALUE_TYPE = double;          __VA_ARGS__; } break; \
            case SCL_INT8:    { using SCL_VALUE_TYPE = std::int8_t;     __VA_ARGS__; } break; \
            case SCL_INT16:   { using SCL_VALUE_TYPE = std::int16_t;    __VA_ARGS__; } break; \
            case SCL_INT32:   { using SCL_VALUE_TYPE = std::int32_t;    __VA_ARGS__; } break; \
            case SCL_INT64:   { using SCL_VALUE_TYPE = std::int64_t;    __VA_ARGS__; } break; \
            case SCL_UINT8:   { using SCL_VALUE_TYPE = std::uint8_t;    __VA_ARGS__; } break; \
            case SCL_UINT16:  { using SCL_VALUE_TYPE = std::uint16_t;   __VA_ARGS__; } break; \
            case SCL_UINT32:  { using SCL_VALUE_TYPE = std::uint32_t;   __VA_ARGS__; } break; \
            case SCL_UINT64:  { using SCL_VALUE_TYPE = std::uint64_t;   __VA_ARGS__; } break; \
            default: \
                scl::set_thread_error(scl::ErrorCode::TypeMismatch, "Unsupported value type"); \
                break; \
        } \
    } while (0)

/**
 * @brief Dispatch based on real type only (2 combinations, legacy)
 *
 * Defines within BLOCK:
 *   - SCL_REAL_TYPE: float or double
 *
 * @note Uses variadic macro to handle code blocks containing commas
 * @deprecated Use SCL_DISPATCH_VALUE for new code
 */
#define SCL_DISPATCH_REAL(real_type, ...) \
    do { \
        if ((real_type) == SCL_REAL32) { \
            using SCL_REAL_TYPE = float; \
            __VA_ARGS__ \
        } else { \
            using SCL_REAL_TYPE = double; \
            __VA_ARGS__ \
        } \
    } while (0)

/**
 * @brief Dispatch based on index type only (2 combinations)
 *
 * Defines within BLOCK:
 *   - SCL_INDEX_TYPE: std::int32_t or std::int64_t
 *
 * @note Uses variadic macro to handle code blocks containing commas
 */
#define SCL_DISPATCH_INDEX(index_type, ...) \
    do { \
        if ((index_type) == SCL_INDEX32) { \
            using SCL_INDEX_TYPE = std::int32_t; \
            __VA_ARGS__ \
        } else { \
            using SCL_INDEX_TYPE = std::int64_t; \
            __VA_ARGS__ \
        } \
    } while (0)

/* ============================================================================
 * SECTION 4: Constrained Dispatch Macros (Version-Specific)
 * ============================================================================ */

/**
 * @brief Dispatch only for Real64 (double precision required)
 *
 * Throws ValueError if real_type is not SCL_REAL64.
 * Use for operations that require double precision (e.g., high-precision eigensolvers).
 *
 * Defines within BLOCK:
 *   - SCL_REAL_TYPE: always double
 *   - SCL_INDEX_TYPE: std::int32_t or std::int64_t
 */
#define SCL_DISPATCH_REAL64_ONLY(real_type, index_type, ...) \
    do { \
        if ((real_type) != SCL_REAL64) { \
            throw ::scl::TypeError( \
                ::scl::ErrorCode::TypeMismatch, \
                "Operation requires Real64 (double) precision", \
                ::scl::source_location::current()); \
        } \
        SCL_DISPATCH_INDEX(index_type, { \
            using SCL_REAL_TYPE = double; \
            __VA_ARGS__ \
        }); \
    } while (0)

/**
 * @brief Dispatch only for Index64 (large matrix support)
 *
 * Throws ValueError if index_type is not SCL_INDEX64.
 * Use for operations that require 64-bit indices (e.g., large-scale matrices).
 *
 * Defines within BLOCK:
 *   - SCL_REAL_TYPE: float or double
 *   - SCL_INDEX_TYPE: always std::int64_t
 */
#define SCL_DISPATCH_INDEX64_ONLY(real_type, index_type, ...) \
    do { \
        if ((index_type) != SCL_INDEX64) { \
            throw ::scl::TypeError( \
                ::scl::ErrorCode::TypeMismatch, \
                "Operation requires Index64 (int64_t) precision", \
                ::scl::source_location::current()); \
        } \
        SCL_DISPATCH_REAL(real_type, { \
            using SCL_INDEX_TYPE = std::int64_t; \
            __VA_ARGS__ \
        }); \
    } while (0)

/**
 * @brief Dispatch only for CSR layout
 *
 * Throws ValueError if layout is not SCL_LAYOUT_CSR.
 */
#define SCL_DISPATCH_CSR_ONLY(real_type, index_type, layout, ...) \
    do { \
        if ((layout) != SCL_LAYOUT_CSR) { \
            throw ::scl::ValueError( \
                ::scl::ErrorCode::InvalidArgument, \
                "Operation requires CSR layout", \
                ::scl::source_location::current()); \
        } \
        SCL_DISPATCH_REAL_INDEX(real_type, index_type, { \
            constexpr bool SCL_IS_CSR = true; \
            __VA_ARGS__ \
        }); \
    } while (0)

/**
 * @brief Dispatch only for CSC layout
 *
 * Throws ValueError if layout is not SCL_LAYOUT_CSC.
 */
#define SCL_DISPATCH_CSC_ONLY(real_type, index_type, layout, ...) \
    do { \
        if ((layout) != SCL_LAYOUT_CSC) { \
            throw ::scl::ValueError( \
                ::scl::ErrorCode::InvalidArgument, \
                "Operation requires CSC layout", \
                ::scl::source_location::current()); \
        } \
        SCL_DISPATCH_REAL_INDEX(real_type, index_type, { \
            constexpr bool SCL_IS_CSR = false; \
            __VA_ARGS__ \
        }); \
    } while (0)

/* ============================================================================
 * SECTION 5: Type Checking Utilities
 * ============================================================================ */

/**
 * @brief Check if two handles have matching real types
 */
#define SCL_CHECK_SAME_REAL_TYPE(h1, h2) \
    do { \
        if ((h1)->real_type != (h2)->real_type) { \
            throw ::scl::TypeError( \
                ::scl::ErrorCode::TypeMismatch, \
                "Real type mismatch between operands", \
                ::scl::source_location::current()); \
        } \
    } while (0)

/**
 * @brief Check if two handles have matching index types
 */
#define SCL_CHECK_SAME_INDEX_TYPE(h1, h2) \
    do { \
        if ((h1)->index_type != (h2)->index_type) { \
            throw ::scl::TypeError( \
                ::scl::ErrorCode::TypeMismatch, \
                "Index type mismatch between operands", \
                ::scl::source_location::current()); \
        } \
    } while (0)

/**
 * @brief Check if two handles have matching layouts
 */
#define SCL_CHECK_SAME_LAYOUT(h1, h2) \
    do { \
        if ((h1)->layout != (h2)->layout) { \
            throw ::scl::ValueError( \
                ::scl::ErrorCode::InvalidArgument, \
                "Layout mismatch between operands", \
                ::scl::source_location::current()); \
        } \
    } while (0)

/**
 * @brief Check if two sparse handles have fully matching types
 */
#define SCL_CHECK_SAME_SPARSE_TYPE(h1, h2) \
    do { \
        SCL_CHECK_SAME_REAL_TYPE(h1, h2); \
        SCL_CHECK_SAME_INDEX_TYPE(h1, h2); \
        SCL_CHECK_SAME_LAYOUT(h1, h2); \
    } while (0)

/* ============================================================================
 * SECTION 6: Dispatch with Return Value
 * ============================================================================ */

/**
 * @brief Dispatch and capture return value
 *
 * Usage:
 * ```cpp
 * int64_t result;
 * SCL_DISPATCH_REAL_INDEX_RET(real_type, index_type, result, {
 *     result = static_cast<int64_t>(some_operation<SCL_REAL_TYPE, SCL_INDEX_TYPE>());
 * });
 * return result;
 * ```
 */
#define SCL_DISPATCH_REAL_INDEX_RET(real_type, index_type, result_var, ...) \
    SCL_DISPATCH_REAL_INDEX(real_type, index_type, { \
        __VA_ARGS__ \
    })

/**
 * @brief Dispatch sparse and capture return value
 */
#define SCL_DISPATCH_SPARSE_RET(real_type, index_type, layout, result_var, ...) \
    SCL_DISPATCH_SPARSE(real_type, index_type, layout, { \
        __VA_ARGS__ \
    })

/* ============================================================================
 * SECTION 7: Binary Operation Dispatch
 * ============================================================================ */

/**
 * @brief Dispatch binary operation with two sparse handles
 *
 * Automatically checks type compatibility and dispatches.
 * Both handles must have matching types.
 *
 * @param h1    First sparse handle
 * @param h2    Second sparse handle
 * @param BLOCK Code block (receives SCL_REAL_TYPE, SCL_INDEX_TYPE, SCL_IS_CSR)
 */
#define SCL_DISPATCH_BINARY_SPARSE(h1, h2, ...) \
    do { \
        SCL_CHECK_SAME_SPARSE_TYPE(h1, h2); \
        SCL_DISPATCH_SPARSE((h1)->real_type, (h1)->index_type, (h1)->layout, __VA_ARGS__); \
    } while (0)

/* ============================================================================
 * SECTION 8: Future Extension Points
 * ============================================================================ */

/*
 * Reserved for future dispatch macros:
 *
 * - SCL_DISPATCH_COMPLEX: Complex number support (Complex64/Complex128)
 * - SCL_DISPATCH_QUANTIZED: Quantized integer types (Int8/Int16)
 * - SCL_DISPATCH_MIXED: Mixed precision operations
 * - SCL_DISPATCH_SIMD: SIMD-width-aware dispatch
 * - SCL_DISPATCH_GPU: Device-aware dispatch (CPU/CUDA/ROCm)
 *
 * Version-specific dispatch:
 * - SCL_DISPATCH_V1_COMPAT: Backward compatibility with API v1
 * - SCL_DISPATCH_EXPERIMENTAL: Experimental features
 */

#endif  /* __cplusplus */

#endif  /* SCL_API_CORE_DISPATCH_H_ */

