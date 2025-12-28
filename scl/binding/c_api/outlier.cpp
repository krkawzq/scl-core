// =============================================================================
// FILE: scl/binding/c_api/outlier.cpp
// BRIEF: C API implementation for outlier detection
// =============================================================================

#include "scl/binding/c_api/outlier.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/outlier.hpp"
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

scl_error_t scl_outlier_local_outlier_factor(
    scl_sparse_matrix_t knn_distances,
    scl_index_t n_cells,
    scl_index_t k,
    scl_real_t* lof_scores
) {
    if (!knn_distances || !lof_scores) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(knn_distances);
        scl::Array<scl::Real> lof_arr(
            reinterpret_cast<scl::Real*>(lof_scores),
            static_cast<scl::Size>(n_cells)
        );
        scl::kernel::outlier::local_outlier_factor(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(k),
            lof_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::outlier::local_outlier_factor<scl::Real, true>(
    const scl::CSR&, scl::Index, scl::Index, scl::Array<scl::Real>);

} // extern "C"
