#include "scl/binding/c_api/group.h"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/group.hpp"

#include <cstring>
#include <exception>

extern "C" {

// Type definitions (if not in header)
#ifndef SCL_C_API_TYPES_DEFINED
typedef scl::Real scl_real_t;
typedef scl::Index scl_index_t;
typedef scl::Size scl_size_t;
typedef void* scl_sparse_matrix_t;
typedef int32_t scl_error_t;
#define SCL_ERROR_OK 0
#define SCL_ERROR_UNKNOWN 1
#define SCL_ERROR_INTERNAL_ERROR 2
#define SCL_ERROR_INVALID_ARGUMENT 10
#define SCL_ERROR_DIMENSION_MISMATCH 11
#endif

// Convert C++ exception to error code
static scl_error_t exception_to_error(const std::exception& e) {
    if (auto* scl_err = dynamic_cast<const scl::Exception*>(&e)) {
        return static_cast<scl_error_t>(scl_err->code());
    }
    return SCL_ERROR_UNKNOWN;
}

scl_error_t scl_group_stats(
    scl_sparse_matrix_t matrix,
    const int32_t* group_ids,
    scl_size_t n_groups,
    const scl_size_t* group_sizes,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof,
    int include_zeros
) {
    try {
        if (!matrix || !group_ids || !group_sizes || !out_means || !out_vars) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(matrix);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::group::group_stats(
            *sparse,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(sparse->cols())),
            static_cast<scl::Size>(n_groups),
            scl::Array<const scl::Size>(group_sizes, static_cast<scl::Size>(n_groups)),
            scl::Array<scl::Real>(out_means, static_cast<scl::Size>(sparse->rows()) * n_groups),
            scl::Array<scl::Real>(out_vars, static_cast<scl::Size>(sparse->rows()) * n_groups),
            ddof,
            include_zeros != 0
        );

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

} // extern "C"

