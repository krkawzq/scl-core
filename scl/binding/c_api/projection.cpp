// =============================================================================
// FILE: scl/binding/c_api/projection.cpp
// BRIEF: C API implementation for random projection
// =============================================================================

#include "scl/binding/c_api/projection.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/projection.hpp"
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

static scl::kernel::projection::ProjectionType convert_projection_type(scl_projection_type_t type) {
    switch (type) {
        case SCL_PROJECTION_GAUSSIAN: return scl::kernel::projection::ProjectionType::Gaussian;
        case SCL_PROJECTION_ACHLIOPTAS: return scl::kernel::projection::ProjectionType::Achlioptas;
        case SCL_PROJECTION_SPARSE: return scl::kernel::projection::ProjectionType::Sparse;
        case SCL_PROJECTION_COUNTSKETCH: return scl::kernel::projection::ProjectionType::CountSketch;
        case SCL_PROJECTION_FEATUREHASH: return scl::kernel::projection::ProjectionType::FeatureHash;
        default: return scl::kernel::projection::ProjectionType::Gaussian;
    }
}

scl_error_t scl_projection_project(
    scl_sparse_matrix_t input,
    scl_real_t* output,
    scl_index_t n_rows,
    scl_index_t n_input_dims,
    scl_index_t n_output_dims,
    scl_projection_type_t type,
    uint64_t seed
) {
    if (!input || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(input);
        scl::kernel::projection::project(
            *sparse,
            reinterpret_cast<scl::Real*>(output),
            static_cast<scl::Index>(n_rows),
            static_cast<scl::Index>(n_input_dims),
            static_cast<scl::Index>(n_output_dims),
            convert_projection_type(type),
            seed
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::projection::project<scl::Real, true>(
    const scl::CSR&, scl::Real*, scl::Index, scl::Index, scl::Index, scl::kernel::projection::ProjectionType, uint64_t);

} // extern "C"
