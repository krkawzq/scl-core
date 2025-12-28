#pragma once

// =============================================================================
// FILE: scl/binding/c_api/propagation/propagation.h
// BRIEF: C API for label propagation
// =============================================================================

#include "scl/binding/c_api/core/core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Label propagation (hard labels)
scl_error_t scl_propagation_label_propagation(
    scl_sparse_t adjacency,
    scl_index_t* labels,              // [n] input/output, -1 for unlabeled
    scl_index_t n,
    scl_index_t max_iter,
    uint64_t seed
);

// Label spreading (soft labels with clamping)
scl_error_t scl_propagation_label_spreading(
    scl_sparse_t adjacency,
    scl_real_t* label_probs,          // [n * n_classes] input/output
    scl_index_t n,
    scl_index_t n_classes,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tolerance
);

// Harmonic function (semi-supervised learning)
scl_error_t scl_propagation_harmonic_function(
    scl_sparse_t adjacency,
    scl_real_t* label_probs,          // [n * n_classes] input/output
    scl_index_t n,
    scl_index_t n_classes,
    scl_real_t alpha
);

// Confidence propagation
scl_error_t scl_propagation_confidence(
    scl_sparse_t adjacency,
    scl_real_t* confidences,          // [n] output
    scl_index_t n,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tolerance
);

#ifdef __cplusplus
}
#endif
