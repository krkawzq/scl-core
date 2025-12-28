#include "scl/binding/c_api/hvg.h"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/hvg.hpp"

#include <cstring>
#include <exception>
#include <cstdint>

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

static scl_error_t exception_to_error(const std::exception& e) {
    if (auto* scl_err = dynamic_cast<const scl::Exception*>(&e)) {
        return static_cast<scl_error_t>(scl_err->code());
    }
    return SCL_ERROR_UNKNOWN;
}

scl_error_t scl_select_by_dispersion(
    scl_sparse_matrix_t matrix,
    scl_size_t n_top,
    scl_index_t* out_indices,
    uint8_t* out_mask,
    scl_real_t* out_dispersions
) {
    try {
        if (!matrix || !out_indices || !out_mask || !out_dispersions) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(matrix);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        const scl::Index n_genes = sparse->rows();
        if (static_cast<scl::Size>(n_top) > static_cast<scl::Size>(n_genes)) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::hvg::select_by_dispersion(
            *sparse,
            static_cast<scl::Size>(n_top),
            scl::Array<scl::Index>(out_indices, static_cast<scl::Size>(n_top)),
            scl::Array<uint8_t>(out_mask, static_cast<scl::Size>(n_genes)),
            scl::Array<scl::Real>(out_dispersions, static_cast<scl::Size>(n_genes))
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

scl_error_t scl_select_by_vst(
    scl_sparse_matrix_t matrix,
    const scl_real_t* clip_vals,
    scl_size_t n_top,
    scl_index_t* out_indices,
    uint8_t* out_mask,
    scl_real_t* out_variances
) {
    try {
        if (!matrix || !clip_vals || !out_indices || !out_mask || !out_variances) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(matrix);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        const scl::Index n_genes = sparse->rows();
        if (static_cast<scl::Size>(n_top) > static_cast<scl::Size>(n_genes)) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::hvg::select_by_vst(
            *sparse,
            scl::Array<const scl::Real>(clip_vals, static_cast<scl::Size>(n_genes)),
            static_cast<scl::Size>(n_top),
            scl::Array<scl::Index>(out_indices, static_cast<scl::Size>(n_top)),
            scl::Array<uint8_t>(out_mask, static_cast<scl::Size>(n_genes)),
            scl::Array<scl::Real>(out_variances, static_cast<scl::Size>(n_genes))
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

