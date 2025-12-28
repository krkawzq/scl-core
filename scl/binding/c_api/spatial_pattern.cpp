// =============================================================================
// FILE: scl/binding/c_api/spatial_pattern.cpp
// BRIEF: C API implementation for spatial pattern detection
// =============================================================================

#include "scl/binding/c_api/spatial_pattern.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/spatial_pattern.hpp"
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

scl_error_t scl_spatial_pattern_detect_variable_genes(
    scl_sparse_matrix_t expression,
    const scl_real_t* coords,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_dims,
    scl_real_t* p_values,
    scl_real_t bandwidth
) {
    if (!expression || !coords || !p_values) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(expression);
        scl::kernel::spatial_pattern::detect_spatially_variable_genes(
            *sparse,
            reinterpret_cast<const scl::Real*>(coords),
            static_cast<scl::Size>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Size>(n_dims),
            reinterpret_cast<scl::Real*>(p_values),
            static_cast<scl::Real>(bandwidth)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::spatial_pattern::detect_spatially_variable_genes<scl::Real, true>(
    const scl::CSR&, const scl::Real*, scl::Size, scl::Index, scl::Size, scl::Real*, scl::Real);

} // extern "C"
