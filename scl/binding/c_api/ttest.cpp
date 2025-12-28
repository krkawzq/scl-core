// =============================================================================
// FILE: scl/binding/c_api/ttest.cpp
// BRIEF: C API implementation for T-test
// =============================================================================

#include "scl/binding/c_api/ttest.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/ttest.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

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

scl_error_t scl_ttest(
    scl_sparse_matrix_t matrix,
    const int32_t* group_ids,
    scl_real_t* out_t_stats,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc,
    int use_welch,
    scl_size_t n_samples
) {
    if (!matrix || !group_ids || !out_t_stats || !out_p_values || !out_log2_fc) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        const scl::Index primary_dim = sparse->primary_dim();
        const scl::Size n_features = static_cast<scl::Size>(primary_dim);
        
        scl::Array<const int32_t> group_arr(
            reinterpret_cast<const int32_t*>(group_ids),
            n_samples
        );
        scl::Array<scl::Real> t_arr(
            reinterpret_cast<scl::Real*>(out_t_stats),
            n_features
        );
        scl::Array<scl::Real> p_arr(
            reinterpret_cast<scl::Real*>(out_p_values),
            n_features
        );
        scl::Array<scl::Real> fc_arr(
            reinterpret_cast<scl::Real*>(out_log2_fc),
            n_features
        );
        
        scl::kernel::ttest::ttest(
            *sparse,
            group_arr,
            t_arr,
            p_arr,
            fc_arr,
            use_welch != 0
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::ttest::ttest<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const int32_t>,
    scl::Array<scl::Real>,
    scl::Array<scl::Real>,
    scl::Array<scl::Real>,
    bool
);

} // extern "C"

