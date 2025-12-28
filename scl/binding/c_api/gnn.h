#pragma once

// =============================================================================
// FILE: scl/binding/c_api/gnn.h
// BRIEF: C API for Graph Neural Network Operations
// =============================================================================

#include "scl/binding/c_api/types.h"

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
    SCL_GNN_AGG_ATTENTION = 4,
    SCL_GNN_AGG_WEIGHTED = 5
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
} scl_gnn_activation_type_t;

// =============================================================================
// Message Passing
// =============================================================================

// Message passing with aggregation
// adjacency: Adjacency matrix handle (CSR format)
// node_features: Node features [n_nodes * feat_dim]
// feat_dim: Feature dimension
// output: Output aggregated features [n_nodes * feat_dim]
// agg_type: Aggregation type
// Returns: Error code
scl_error_t scl_gnn_message_passing(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_real_t* output,
    scl_gnn_agg_type_t agg_type
);

// =============================================================================
// Graph Attention
// =============================================================================

// Graph attention network (GAT)
// adjacency: Adjacency matrix handle (CSR format)
// node_features: Node features [n_nodes * feat_dim]
// feat_dim: Feature dimension
// attention_vec: Attention vector [2 * feat_dim]
// output: Output features [n_nodes * feat_dim]
// add_self_loops: Whether to add self-loops
// Returns: Error code
scl_error_t scl_gnn_graph_attention(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    const scl_real_t* attention_vec,
    scl_real_t* output,
    int add_self_loops
);

// Multi-head attention
// adjacency: Adjacency matrix handle (CSR format)
// node_features: Node features [n_nodes * feat_dim]
// feat_dim: Feature dimension
// n_heads: Number of attention heads
// attention_vecs: Attention vectors [n_heads * 2 * head_dim]
// output: Output features [n_nodes * n_heads * head_dim]
// add_self_loops: Whether to add self-loops
// Returns: Error code
scl_error_t scl_gnn_multi_head_attention(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_index_t n_heads,
    const scl_real_t* attention_vecs,
    scl_real_t* output,
    int add_self_loops
);

// =============================================================================
// Graph Convolution
// =============================================================================

// Graph convolution network (GCN)
// adjacency: Adjacency matrix handle (CSR format)
// node_features: Input node features [n_nodes * in_dim]
// in_dim: Input feature dimension
// weight: Weight matrix [out_dim * in_dim]
// out_dim: Output feature dimension
// output: Output features [n_nodes * out_dim]
// add_self_loops: Whether to add self-loops
// activation: Activation function type
// Returns: Error code
scl_error_t scl_gnn_graph_convolution(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t in_dim,
    const scl_real_t* weight,
    scl_index_t out_dim,
    scl_real_t* output,
    int add_self_loops,
    scl_gnn_activation_type_t activation
);

// =============================================================================
// GraphSAGE
// =============================================================================

// GraphSAGE aggregation
// adjacency: Adjacency matrix handle (CSR format)
// node_features: Node features [n_nodes * feat_dim]
// feat_dim: Feature dimension
// output: Output aggregated features [n_nodes * feat_dim]
// agg_type: Aggregation type
// max_neighbors: Maximum neighbors to sample (0 = use all)
// Returns: Error code
scl_error_t scl_gnn_sage_aggregate(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_real_t* output,
    scl_gnn_agg_type_t agg_type,
    scl_index_t max_neighbors
);

// =============================================================================
// Feature Smoothing
// =============================================================================

// Feature smoothing on graph
// adjacency: Adjacency matrix handle (CSR format)
// features: Node features [n_nodes * feat_dim] (modified in-place)
// n_nodes: Number of nodes
// feat_dim: Feature dimension
// alpha: Smoothing parameter (0-1)
// n_iterations: Number of smoothing iterations
// Returns: Error code
scl_error_t scl_gnn_feature_smoothing(
    scl_sparse_matrix_t adjacency,
    scl_real_t* features,
    scl_index_t n_nodes,
    scl_index_t feat_dim,
    scl_real_t alpha,
    scl_index_t n_iterations
);

// =============================================================================
// Pooling Operations
// =============================================================================

// Global pooling
// adjacency: Adjacency matrix handle (CSR format)
// node_features: Node features [n_nodes * feat_dim]
// feat_dim: Feature dimension
// graph_features: Output graph-level features [feat_dim]
// agg_type: Aggregation type
// Returns: Error code
scl_error_t scl_gnn_global_pool(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_real_t* graph_features,
    scl_gnn_agg_type_t agg_type
);

// Hierarchical pooling
// node_features: Node features [n_nodes * feat_dim]
// n_nodes: Number of nodes
// feat_dim: Feature dimension
// cluster_assignment: Cluster assignment for each node [n_nodes]
// n_clusters: Number of clusters
// pooled_features: Output pooled features [n_clusters * feat_dim]
// agg_type: Aggregation type
// Returns: Error code
scl_error_t scl_gnn_hierarchical_pool(
    const scl_real_t* node_features,
    scl_index_t n_nodes,
    scl_index_t feat_dim,
    const scl_index_t* cluster_assignment,
    scl_index_t n_clusters,
    scl_real_t* pooled_features,
    scl_gnn_agg_type_t agg_type
);

// =============================================================================
// Edge Features
// =============================================================================

// Compute edge features from node features
// adjacency: Adjacency matrix handle (CSR format)
// node_features: Node features [n_nodes * feat_dim]
// feat_dim: Feature dimension
// edge_features: Output edge features [n_edges * edge_feat_dim]
// concat: If true, concatenate (edge_feat_dim = 2*feat_dim), else subtract (edge_feat_dim = feat_dim)
// Returns: Error code
scl_error_t scl_gnn_compute_edge_features(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_real_t* edge_features,
    int concat
);

// =============================================================================
// Normalization and Activation
// =============================================================================

// Skip connection: output = input + alpha * residual
// input: Input features [n]
// residual: Residual features [n]
// output: Output features [n]
// alpha: Scaling factor
// n: Number of elements
// Returns: Error code
scl_error_t scl_gnn_skip_connection(
    const scl_real_t* input,
    const scl_real_t* residual,
    scl_real_t* output,
    scl_real_t alpha,
    scl_size_t n
);

// Layer normalization (in-place)
// features: Node features [n_nodes * feat_dim] (modified in-place)
// n_nodes: Number of nodes
// feat_dim: Feature dimension
// epsilon: Small constant for numerical stability
// Returns: Error code
scl_error_t scl_gnn_layer_norm(
    scl_real_t* features,
    scl_index_t n_nodes,
    scl_index_t feat_dim,
    scl_real_t epsilon
);

// Batch normalization
// features: Node features [n_nodes * feat_dim] (modified in-place)
// n_nodes: Number of nodes
// feat_dim: Feature dimension
// gamma: Scale parameters [feat_dim]
// beta: Shift parameters [feat_dim]
// epsilon: Small constant for numerical stability
// Returns: Error code
scl_error_t scl_gnn_batch_norm(
    scl_real_t* features,
    scl_index_t n_nodes,
    scl_index_t feat_dim,
    const scl_real_t* gamma,
    const scl_real_t* beta,
    scl_real_t epsilon
);

// Dropout (in-place, training mode)
// features: Features [n] (modified in-place)
// dropout_rate: Dropout rate (0-1)
// seed: Random seed
// training: If 0, no dropout applied
// n: Number of elements
// Returns: Error code
scl_error_t scl_gnn_dropout(
    scl_real_t* features,
    scl_real_t dropout_rate,
    uint64_t seed,
    int training,
    scl_size_t n
);

#ifdef __cplusplus
}
#endif
