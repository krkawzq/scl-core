// =============================================================================
// FILE: scl/binding/c_api/spatial.cpp
// BRIEF: C API implementation for spatial statistics
// =============================================================================

#include "scl/binding/c_api/spatial.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/spatial.hpp"
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

scl_real_t scl_spatial_weight_sum(
    scl_sparse_matrix_t graph
) {
    if (!graph) {
        return 0.0;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(graph);
        scl::Real sum = scl::Real(0);
        scl::kernel::spatial::weight_sum(*sparse, sum);
        return static_cast<scl_real_t>(sum);
    } catch (...) {
        return 0.0;
    }
}

scl_error_t scl_spatial_morans_i(
    scl_sparse_matrix_t graph,
    const scl_real_t* values,
    scl_real_t* out_i,
    scl_size_t n_nodes
) {
    if (!graph || !values || !out_i) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(graph);
        scl::Array<const scl::Real> values_arr(
            reinterpret_cast<const scl::Real*>(values),
            n_nodes
        );
        scl::Real morans_i = scl::Real(0);
        scl::kernel::spatial::morans_i(*sparse, values_arr, morans_i);
        *out_i = static_cast<scl_real_t>(morans_i);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_spatial_gearys_c(
    scl_sparse_matrix_t graph,
    const scl_real_t* values,
    scl_real_t* out_c,
    scl_size_t n_nodes
) {
    if (!graph || !values || !out_c) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(graph);
        scl::Array<const scl::Real> values_arr(
            reinterpret_cast<const scl::Real*>(values),
            n_nodes
        );
        scl::Real gearys_c = scl::Real(0);
        scl::kernel::spatial::gearys_c(*sparse, values_arr, gearys_c);
        *out_c = static_cast<scl_real_t>(gearys_c);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::spatial::weight_sum<scl::Real, true>(
    const scl::CSR&,
    scl::Real&
);

template void scl::kernel::spatial::morans_i<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Real&
);

template void scl::kernel::spatial::gearys_c<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Real&
);

} // extern "C"

