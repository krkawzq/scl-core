#pragma once

// =============================================================================
// FILE: scl/binding/c_api/pseudotime/pseudotime.h
// BRIEF: C API for pseudotime inference
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Pseudotime Methods
// =============================================================================

typedef enum {
    SCL_PSEUDOTIME_DIFFUSION = 0,
    SCL_PSEUDOTIME_SHORTEST_PATH = 1,
    SCL_PSEUDOTIME_GRAPH_DISTANCE = 2,
    SCL_PSEUDOTIME_WATERSHED = 3
} scl_pseudotime_method_t;

// =============================================================================
// Pseudotime Computation
// =============================================================================

// Compute pseudotime from root cell
scl_error_t scl_pseudotime_compute(
    scl_sparse_t adjacency,
    scl_index_t root_cell,
    scl_real_t* pseudotime,           // [n] output
    scl_index_t n,
    scl_pseudotime_method_t method,
    scl_index_t n_dcs
);

// Diffusion-based pseudotime
scl_error_t scl_pseudotime_diffusion(
    scl_sparse_t adjacency,
    scl_index_t root_cell,
    scl_real_t* pseudotime,           // [n] output
    scl_index_t n,
    scl_index_t n_dcs
);

// Graph distance pseudotime (shortest path)
scl_error_t scl_pseudotime_graph(
    scl_sparse_t adjacency,
    scl_index_t root_cell,
    scl_real_t* pseudotime,           // [n] output
    scl_index_t n
);

// Multi-source shortest paths
scl_error_t scl_pseudotime_multi_source(
    scl_sparse_t adjacency,
    const scl_index_t* source_cells,
    scl_index_t n_sources,
    scl_real_t* distances,            // [n] output
    scl_index_t n
);

#ifdef __cplusplus
}
#endif
