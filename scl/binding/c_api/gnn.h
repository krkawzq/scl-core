#pragma once

// =============================================================================
// FILE: scl/binding/c_api/gnn/gnn.h
// BRIEF: C API for Graph Neural Network operations
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Aggregation Types
// =============================================================================

typedef enum {
    SCL_GNN_AGG_SUM = 0,
    SCL_GNN_AGG_MEAN = 1,
    SCL_GNN_AGG_MAX = 2,
    SCL_GNN_AGG_MIN = 3,
    SCL_GNN_AGG_WEIGHTED = 4,
    SCL_GNN_AGG_ATTENTION = 5
} scl_gnn_agg_type_t;

// =============================================================================
// Activation Types
// =============================================================================

typedef enum {
    SCL_GNN_ACT_NONE = 0,
    SCL_GNN_ACT_RELU = 1,
    SCL_GNN_ACT_LEAKY_RELU = 2,
    SCL_GNN_ACT_SIGMOID = 3,
    SCL_GNN_ACT_TANH = 4,
    SCL_GNN_ACT_ELU = 5,
    SCL_GNN_ACT_GELU = 6
} scl_gnn_act_type_t;

// =============================================================================
// Message Passing
// =============================================================================

scl_error_t scl_gnn_message_passing(
    scl_sparse_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_real_t* output,
    scl_size_t n_nodes,
    scl_gnn_agg_type_t agg_type
);

// =============================================================================
// Graph Attention
// =============================================================================

scl_error_t scl_gnn_graph_attention(
    scl_sparse_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    const scl_real_t* attention_vec,
    scl_real_t* output,
    scl_size_t n_nodes,
    int add_self_loops
);

// =============================================================================
// Graph Convolution
// =============================================================================

scl_error_t scl_gnn_graph_convolution(
    scl_sparse_t adjacency,
    const scl_real_t* node_features,
    scl_index_t in_dim,
    const scl_real_t* weight,
    scl_index_t out_dim,
    scl_real_t* output,
    scl_size_t n_nodes,
    int add_self_loops,
    scl_gnn_act_type_t activation
);

// =============================================================================
// Feature Smoothing
// =============================================================================

scl_error_t scl_gnn_feature_smoothing(
    scl_sparse_t adjacency,
    scl_real_t* features,
    scl_index_t n_nodes,
    scl_index_t feat_dim,
    scl_real_t alpha,
    scl_index_t n_iterations
);

#ifdef __cplusplus
}
#endif
