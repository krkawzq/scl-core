// =============================================================================
// FILE: scl/binding/c_api/mwu.cpp
// BRIEF: C API implementation for Mann-Whitney U Test
// =============================================================================

#include "scl/binding/c_api/mwu.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/mwu.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

// Internal helper to convert C++ exception to error code
static scl_error_t handle_exception() {
    try {
        throw;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::TypeError&) {
        return SCL_ERROR_TYPE;
    } catch (const scl::IOError&) {
        return SCL_ERROR_IO;
    } catch (const scl::NotImplementedError&) {
        return SCL_ERROR_NOT_IMPLEMENTED;
    } catch (const scl::InternalError&) {
        return SCL_ERROR_INTERNAL;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_mwu_test(
    scl_sparse_matrix_t matrix,
    const int32_t* group_ids,
    scl_real_t* out_u_stats,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc,
    scl_real_t* out_auroc
) {
    if (!matrix || !group_ids || !out_u_stats || !out_p_values || !out_log2_fc) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* sparse = reinterpret_cast<scl::CSR*>(matrix);

        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        const scl::Index primary_dim = sparse->primary_dim();
        const scl::Size n_features = static_cast<scl::Size>(primary_dim);
        const scl::Size n_samples = static_cast<scl::Size>(sparse->secondary_dim());

        scl::Array<const int32_t> group_arr(group_ids, n_samples);
        scl::Array<scl::Real> u_arr(out_u_stats, n_features);
        scl::Array<scl::Real> p_arr(out_p_values, n_features);
        scl::Array<scl::Real> fc_arr(out_log2_fc, n_features);

        if (out_auroc) {
            scl::Array<scl::Real> auroc_arr(out_auroc, n_features);
            scl::kernel::mwu::mwu_test(*sparse, group_arr, u_arr, p_arr, fc_arr, auroc_arr);
        } else {
            scl::Array<scl::Real> empty_auroc(nullptr, 0);
            scl::kernel::mwu::mwu_test(*sparse, group_arr, u_arr, p_arr, fc_arr, empty_auroc);
        }

        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

// Explicit instantiation
template void scl::kernel::mwu::mwu_test<scl::Real, true>(
    const scl::Sparse<scl::Real, true>&,
    scl::Array<const int32_t>,
    scl::Array<scl::Real>,
    scl::Array<scl::Real>,
    scl::Array<scl::Real>,
    scl::Array<scl::Real>
);

