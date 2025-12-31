/// @file scl/api/kernel/log1p.cpp
/// @brief C-API implementation for log1p transformation kernels
///
/// This file implements the C-ABI interface for log1p, log2p1, and expm1
/// transformations on both sparse matrices and dense arrays.

#include "scl/api/kernel/log1p.h"
#include "scl/api/core/type.h"
#include "scl/api/core/error.h"
#include "scl/api/core/sparse.h"
#include "scl/api/core/dispatch.h"
#include "scl/kernel/log1p.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"

#include <span>
#include <cstdint>
#include <variant>

// =============================================================================
// C-ABI Export Macros
// =============================================================================

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef SCL_BUILDING_LIBRARY
        #define SCL_API extern "C" __declspec(dllexport)
    #else
        #define SCL_API extern "C" __declspec(dllimport)
    #endif
#else
    #define SCL_API extern "C" __attribute__((visibility("default")))
#endif

// =============================================================================
// SECTION 1: Internal Structure Definitions (from handler.cpp)
// =============================================================================

namespace {

/// @brief Variant type holding all possible Sparse matrix types
/// Total: 10 value types × 2 index types × 2 layouts = 40 combinations
using SparseVariant = std::variant<
    // Real32 (indices 0-3)
    scl::Sparse<scl::Real32, scl::Index32, true>,   scl::Sparse<scl::Real32, scl::Index32, false>,
    scl::Sparse<scl::Real32, scl::Index64, true>,   scl::Sparse<scl::Real32, scl::Index64, false>,

    // Real64 (indices 4-7)
    scl::Sparse<scl::Real64, scl::Index32, true>,   scl::Sparse<scl::Real64, scl::Index32, false>,
    scl::Sparse<scl::Real64, scl::Index64, true>,   scl::Sparse<scl::Real64, scl::Index64, false>,

    // Int8 (indices 8-11)
    scl::Sparse<scl::Int8, scl::Index32, true>,     scl::Sparse<scl::Int8, scl::Index32, false>,
    scl::Sparse<scl::Int8, scl::Index64, true>,     scl::Sparse<scl::Int8, scl::Index64, false>,

    // Int16 (indices 12-15)
    scl::Sparse<scl::Int16, scl::Index32, true>,    scl::Sparse<scl::Int16, scl::Index32, false>,
    scl::Sparse<scl::Int16, scl::Index64, true>,    scl::Sparse<scl::Int16, scl::Index64, false>,

    // Int32 (indices 16-19)
    scl::Sparse<scl::Int32, scl::Index32, true>,    scl::Sparse<scl::Int32, scl::Index32, false>,
    scl::Sparse<scl::Int32, scl::Index64, true>,    scl::Sparse<scl::Int32, scl::Index64, false>,

    // Int64 (indices 20-23)
    scl::Sparse<scl::Int64, scl::Index32, true>,    scl::Sparse<scl::Int64, scl::Index32, false>,
    scl::Sparse<scl::Int64, scl::Index64, true>,    scl::Sparse<scl::Int64, scl::Index64, false>,

    // UInt8 (indices 24-27)
    scl::Sparse<scl::UInt8, scl::Index32, true>,    scl::Sparse<scl::UInt8, scl::Index32, false>,
    scl::Sparse<scl::UInt8, scl::Index64, true>,    scl::Sparse<scl::UInt8, scl::Index64, false>,

    // UInt16 (indices 28-31)
    scl::Sparse<scl::UInt16, scl::Index32, true>,   scl::Sparse<scl::UInt16, scl::Index32, false>,
    scl::Sparse<scl::UInt16, scl::Index64, true>,   scl::Sparse<scl::UInt16, scl::Index64, false>,

    // UInt32 (indices 32-35)
    scl::Sparse<scl::UInt32, scl::Index32, true>,   scl::Sparse<scl::UInt32, scl::Index32, false>,
    scl::Sparse<scl::UInt32, scl::Index64, true>,   scl::Sparse<scl::UInt32, scl::Index64, false>,

    // UInt64 (indices 36-39)
    scl::Sparse<scl::UInt64, scl::Index32, true>,   scl::Sparse<scl::UInt64, scl::Index32, false>,
    scl::Sparse<scl::UInt64, scl::Index64, true>,   scl::Sparse<scl::UInt64, scl::Index64, false>
>;

} // anonymous namespace

/// @brief Internal sparse handle structure (must match handler.cpp)
struct scl_sparse_s {
    scl_value_type_t value_type;  ///< Value type (Real/Int/Uint + precision)
    scl_index_type_t index_type;  ///< Index precision
    scl_layout_t     layout;      ///< CSR or CSC
    SparseVariant    data;        ///< Type-erased matrix storage
};

namespace {

/// @brief Apply a visitor to the sparse matrix with correct types
template<typename Visitor>
auto visit_sparse(scl_sparse_t handle, Visitor&& visitor) {
    return std::visit(std::forward<Visitor>(visitor), handle->data);
}

} // anonymous namespace

// =============================================================================
// SECTION 2: Sparse Matrix Operations (In-Place)
// =============================================================================

SCL_API
auto scl_log1p_sparse(scl_sparse_t matrix) -> std::int32_t {
    SCL_C_API_BEGIN

    // Validation: NULL pointer check
    SCL_CHECK_NOT_NULL(matrix);

    // Validation: Type check - only Real32 and Real64 are supported
    SCL_CHECK_ARG(
        matrix->value_type == SCL_REAL32 || matrix->value_type == SCL_REAL64,
        "log1p_sparse: only Real32 and Real64 are supported"
    );

    // Apply transformation using visitor pattern
    visit_sparse(matrix, [](auto& mat) {
        using MatrixType = std::decay_t<decltype(mat)>;
        using ValueType = typename MatrixType::ValueType;

        // Only process Real32 and Real64 types
        if constexpr (std::is_same_v<ValueType, scl::Real32> ||
                      std::is_same_v<ValueType, scl::Real64>) {
            scl::kernel::log1p::log1p_inplace(mat);
        }
    });

    SCL_C_API_END
}

SCL_API
auto scl_log2p1_sparse(scl_sparse_t matrix) -> std::int32_t {
    SCL_C_API_BEGIN

    // Validation: NULL pointer check
    SCL_CHECK_NOT_NULL(matrix);

    // Validation: Type check - only Real32 and Real64 are supported
    SCL_CHECK_ARG(
        matrix->value_type == SCL_REAL32 || matrix->value_type == SCL_REAL64,
        "log2p1_sparse: only Real32 and Real64 are supported"
    );

    // Apply transformation using visitor pattern
    visit_sparse(matrix, [](auto& mat) {
        using MatrixType = std::decay_t<decltype(mat)>;
        using ValueType = typename MatrixType::ValueType;

        // Only process Real32 and Real64 types
        if constexpr (std::is_same_v<ValueType, scl::Real32> ||
                      std::is_same_v<ValueType, scl::Real64>) {
            scl::kernel::log1p::log2p1_inplace(mat);
        }
    });

    SCL_C_API_END
}

SCL_API
auto scl_expm1_sparse(scl_sparse_t matrix) -> std::int32_t {
    SCL_C_API_BEGIN

    // Validation: NULL pointer check
    SCL_CHECK_NOT_NULL(matrix);

    // Validation: Type check - only Real32 and Real64 are supported
    SCL_CHECK_ARG(
        matrix->value_type == SCL_REAL32 || matrix->value_type == SCL_REAL64,
        "expm1_sparse: only Real32 and Real64 are supported"
    );

    // Apply transformation using visitor pattern
    visit_sparse(matrix, [](auto& mat) {
        using MatrixType = std::decay_t<decltype(mat)>;
        using ValueType = typename MatrixType::ValueType;

        // Only process Real32 and Real64 types
        if constexpr (std::is_same_v<ValueType, scl::Real32> ||
                      std::is_same_v<ValueType, scl::Real64>) {
            scl::kernel::log1p::expm1_inplace(mat);
        }
    });

    SCL_C_API_END
}

// =============================================================================
// SECTION 3: Dense Array Operations
// =============================================================================

SCL_API
auto scl_log1p_array(
    const void* input,
    std::int64_t size,
    void* output,
    scl_value_type_t precision
) -> std::int32_t {
    SCL_C_API_BEGIN

    // Validation: NULL pointer checks
    SCL_CHECK_NOT_NULL(input);
    SCL_CHECK_NOT_NULL(output);

    // Validation: Size must be non-negative
    SCL_CHECK_ARG(size >= 0, "log1p_array: size must be non-negative");

    // Validation: Precision must be Real32 or Real64
    SCL_CHECK_ARG(
        precision == SCL_REAL32 || precision == SCL_REAL64,
        "log1p_array: only Real32 and Real64 are supported"
    );

    // Early return for empty array
    if (size == 0) [[unlikely]] {
        return static_cast<std::int32_t>(scl::ErrorCode::Success);
    }

    // Dispatch based on precision
    if (precision == SCL_REAL32) {
        // Real32 (float) processing
        const auto* input_typed = static_cast<const float*>(input);
        auto* output_typed = static_cast<float*>(output);

        // Create spans
        std::span<const float> input_span(input_typed, static_cast<std::size_t>(size));
        std::span<float> output_span(output_typed, static_cast<std::size_t>(size));

        // Apply transformation
        scl::kernel::log1p::log1p(input_span, output_span);

    } else {
        // Real64 (double) processing
        const auto* input_typed = static_cast<const double*>(input);
        auto* output_typed = static_cast<double*>(output);

        // Create spans
        std::span<const double> input_span(input_typed, static_cast<std::size_t>(size));
        std::span<double> output_span(output_typed, static_cast<std::size_t>(size));

        // Apply transformation
        scl::kernel::log1p::log1p(input_span, output_span);
    }

    SCL_C_API_END
}

SCL_API
auto scl_log2p1_array(
    const void* input,
    std::int64_t size,
    void* output,
    scl_value_type_t precision
) -> std::int32_t {
    SCL_C_API_BEGIN

    // Validation: NULL pointer checks
    SCL_CHECK_NOT_NULL(input);
    SCL_CHECK_NOT_NULL(output);

    // Validation: Size must be non-negative
    SCL_CHECK_ARG(size >= 0, "log2p1_array: size must be non-negative");

    // Validation: Precision must be Real32 or Real64
    SCL_CHECK_ARG(
        precision == SCL_REAL32 || precision == SCL_REAL64,
        "log2p1_array: only Real32 and Real64 are supported"
    );

    // Early return for empty array
    if (size == 0) [[unlikely]] {
        return static_cast<std::int32_t>(scl::ErrorCode::Success);
    }

    // Dispatch based on precision
    if (precision == SCL_REAL32) {
        // Real32 (float) processing
        const auto* input_typed = static_cast<const float*>(input);
        auto* output_typed = static_cast<float*>(output);

        // Create spans
        std::span<const float> input_span(input_typed, static_cast<std::size_t>(size));
        std::span<float> output_span(output_typed, static_cast<std::size_t>(size));

        // Apply transformation
        scl::kernel::log1p::log2p1(input_span, output_span);

    } else {
        // Real64 (double) processing
        const auto* input_typed = static_cast<const double*>(input);
        auto* output_typed = static_cast<double*>(output);

        // Create spans
        std::span<const double> input_span(input_typed, static_cast<std::size_t>(size));
        std::span<double> output_span(output_typed, static_cast<std::size_t>(size));

        // Apply transformation
        scl::kernel::log1p::log2p1(input_span, output_span);
    }

    SCL_C_API_END
}

SCL_API
auto scl_expm1_array(
    const void* input,
    std::int64_t size,
    void* output,
    scl_value_type_t precision
) -> std::int32_t {
    SCL_C_API_BEGIN

    // Validation: NULL pointer checks
    SCL_CHECK_NOT_NULL(input);
    SCL_CHECK_NOT_NULL(output);

    // Validation: Size must be non-negative
    SCL_CHECK_ARG(size >= 0, "expm1_array: size must be non-negative");

    // Validation: Precision must be Real32 or Real64
    SCL_CHECK_ARG(
        precision == SCL_REAL32 || precision == SCL_REAL64,
        "expm1_array: only Real32 and Real64 are supported"
    );

    // Early return for empty array
    if (size == 0) [[unlikely]] {
        return static_cast<std::int32_t>(scl::ErrorCode::Success);
    }

    // Dispatch based on precision
    if (precision == SCL_REAL32) {
        // Real32 (float) processing
        const auto* input_typed = static_cast<const float*>(input);
        auto* output_typed = static_cast<float*>(output);

        // Create spans
        std::span<const float> input_span(input_typed, static_cast<std::size_t>(size));
        std::span<float> output_span(output_typed, static_cast<std::size_t>(size));

        // Apply transformation
        scl::kernel::log1p::expm1(input_span, output_span);

    } else {
        // Real64 (double) processing
        const auto* input_typed = static_cast<const double*>(input);
        auto* output_typed = static_cast<double*>(output);

        // Create spans
        std::span<const double> input_span(input_typed, static_cast<std::size_t>(size));
        std::span<double> output_span(output_typed, static_cast<std::size_t>(size));

        // Apply transformation
        scl::kernel::log1p::expm1(input_span, output_span);
    }

    SCL_C_API_END
}
