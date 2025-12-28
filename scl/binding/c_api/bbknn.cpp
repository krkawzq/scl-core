// =============================================================================
// FILE: scl/binding/c_api/bbknn.cpp
// BRIEF: C API implementation for Batch Balanced KNN
// =============================================================================

#include "scl/binding/c_api/bbknn.h"
#include "scl/kernel/bbknn.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <cstring>

extern "C" {

scl_error_t scl_bbknn(
    scl_sparse_matrix_t matrix,
    const int32_t* batch_labels,
    scl_size_t n_batches,
    scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances
) {
    try {
        if (!matrix || !batch_labels || !out_indices || !out_distances) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        // Cast to C++ sparse matrix
        auto* sparse = static_cast<scl::CSR*>(matrix);
        
        const scl::Index primary_dim = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(primary_dim);
        const scl::Size neighbors_per_cell = n_batches * k;

        // Validate dimensions
        if (out_indices == nullptr || out_distances == nullptr) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        // Prepare arrays
        scl::Array<const int32_t> batch_arr(batch_labels, N);
        scl::Array<scl::Index> indices_arr(out_indices, N * neighbors_per_cell);
        scl::Array<scl::Real> distances_arr(out_distances, N * neighbors_per_cell);

        // Call C++ kernel
        scl::kernel::bbknn::bbknn(
            *sparse,
            batch_arr,
            n_batches,
            k,
            indices_arr,
            distances_arr
        );

        return SCL_ERROR_OK;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::Exception&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

} // extern "C"

