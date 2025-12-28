#pragma once

// =============================================================================
// FILE: scl/binding/c_api/outlier.h
// BRIEF: C API for outlier detection
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Local Outlier Factor
scl_error_t scl_outlier_local_outlier_factor(
    scl_sparse_matrix_t knn_distances,
    scl_index_t n_cells,
    scl_index_t k,
    scl_real_t* lof_scores
);

#ifdef __cplusplus
}
#endif
