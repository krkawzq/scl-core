// =============================================================================
// FILE: scl/binding/c_api/gnn/gnn.cpp
// BRIEF: C API implementation for Graph Neural Network operations
// =============================================================================

#include "scl/binding/c_api/gnn/gnn.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/gnn.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::gnn;

extern "C" {

// =============================================================================
// Helper: Convert aggregation type
// =============================================================================

static AggregationType convert_agg_type(scl_gnn_agg_type_t type) {
    switch (type) {
        case SCL_GNN_AGG_SUM: return AggregationType::Sum;
        case SCL_GNN_AGG_MEAN: return AggregationType::Mean;
        case SCL_GNN_AGG_MAX: return AggregationType::Max;
        case SCL_GNN_AGG_MIN: return AggregationType::Min;
        case SCL_GNN_AGG_WEIGHTED: return AggregationType::Weighted;
        case SCL_GNN_AGG_ATTENTION: return AggregationType::Attention;
        default: return AggregationType::Mean;
    }
}

// =============================================================================
// Helper: Convert activation type
// =============================================================================

static ActivationType convert_act_type(scl_gnn_act_type_t type) {
    switch (type) {
        case SCL_GNN_ACT_NONE: return ActivationType::None;
        case SCL_GNN_ACT_RELU: return ActivationType::ReLU;
        case SCL_GNN_ACT_LEAKY_RELU: return ActivationType::LeakyReLU;
        case SCL_GNN_ACT_SIGMOID: return ActivationType::Sigmoid;
        case SCL_GNN_ACT_TANH: return ActivationType::Tanh;
        case SCL_GNN_ACT_ELU: return ActivationType::ELU;
        case SCL_GNN_ACT_GELU: return ActivationType::GELU;
        default: return ActivationType::ReLU;
    }
}

// =============================================================================
// Message Passing
// =============================================================================

scl_error_t scl_gnn_message_passing(
    scl_sparse_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_real_t* output,
    scl_size_t n_nodes,
    scl_gnn_agg_type_t agg_type)
{
    if (!adjacency || !node_features || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(adjacency);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_nodes) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Node count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        scl_size_t expected_output_size = n_nodes * static_cast<scl_size_t>(feat_dim);
        
        Array<const Real> features_arr(
            reinterpret_cast<const Real*>(node_features),
            n_nodes * static_cast<scl_size_t>(feat_dim)
        );
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            expected_output_size
        );

        wrapper->visit([&](auto& adj) {
            message_passing(
                adj, features_arr, feat_dim, output_arr,
                convert_agg_type(agg_type)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

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
    int add_self_loops)
{
    if (!adjacency || !node_features || !attention_vec || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(adjacency);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_nodes) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Node count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        scl_size_t feat_size = n_nodes * static_cast<scl_size_t>(feat_dim);
        scl_size_t attn_size = 2 * static_cast<scl_size_t>(feat_dim);

        Array<const Real> features_arr(
            reinterpret_cast<const Real*>(node_features),
            feat_size
        );
        Array<const Real> attn_arr(
            reinterpret_cast<const Real*>(attention_vec),
            attn_size
        );
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            feat_size
        );

        wrapper->visit([&](auto& adj) {
            graph_attention(
                adj, features_arr, feat_dim, attn_arr, output_arr,
                add_self_loops != 0
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

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
    scl_gnn_act_type_t activation)
{
    if (!adjacency || !node_features || !weight || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(adjacency);
        
        Index n = wrapper->rows();
        if (static_cast<scl_size_t>(n) != n_nodes) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Node count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        scl_size_t feat_size = n_nodes * static_cast<scl_size_t>(in_dim);
        scl_size_t weight_size = static_cast<scl_size_t>(out_dim) * static_cast<scl_size_t>(in_dim);
        scl_size_t output_size = n_nodes * static_cast<scl_size_t>(out_dim);

        Array<const Real> features_arr(
            reinterpret_cast<const Real*>(node_features),
            feat_size
        );
        Array<const Real> weight_arr(
            reinterpret_cast<const Real*>(weight),
            weight_size
        );
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            output_size
        );

        wrapper->visit([&](auto& adj) {
            graph_convolution(
                adj, features_arr, in_dim, weight_arr, out_dim, output_arr,
                add_self_loops != 0, convert_act_type(activation)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Feature Smoothing
// =============================================================================

scl_error_t scl_gnn_feature_smoothing(
    scl_sparse_t adjacency,
    scl_real_t* features,
    scl_index_t n_nodes,
    scl_index_t feat_dim,
    scl_real_t alpha,
    scl_index_t n_iterations)
{
    if (!adjacency || !features) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(adjacency);
        
        Index n = wrapper->rows();
        if (n != n_nodes) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Node count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        scl_size_t feat_size = static_cast<scl_size_t>(n_nodes) * static_cast<scl_size_t>(feat_dim);

        Array<Real> features_arr(
            reinterpret_cast<Real*>(features),
            feat_size
        );

        wrapper->visit([&](auto& adj) {
            feature_smoothing(
                adj, features_arr, n_nodes, feat_dim, alpha, n_iterations
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

