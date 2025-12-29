// =============================================================================
// FILE: scl/binding/c_api/gnn/gnn.cpp
// BRIEF: C API implementation for Graph Neural Network operations
// =============================================================================

#include "scl/binding/c_api/gnn.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/gnn.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

[[nodiscard]] constexpr auto convert_agg_type(
    scl_gnn_agg_type_t type) noexcept -> scl::kernel::gnn::AggregationType {
    switch (type) {
        case SCL_GNN_AGG_SUM:
            return scl::kernel::gnn::AggregationType::Sum;
        case SCL_GNN_AGG_MEAN:
            return scl::kernel::gnn::AggregationType::Mean;
        case SCL_GNN_AGG_MAX:
            return scl::kernel::gnn::AggregationType::Max;
        case SCL_GNN_AGG_MIN:
            return scl::kernel::gnn::AggregationType::Min;
        case SCL_GNN_AGG_WEIGHTED:
            return scl::kernel::gnn::AggregationType::Weighted;
        case SCL_GNN_AGG_ATTENTION:
            return scl::kernel::gnn::AggregationType::Attention;
        default:
            return scl::kernel::gnn::AggregationType::Mean;
    }
}

[[nodiscard]] constexpr auto convert_act_type(
    scl_gnn_act_type_t type) noexcept -> scl::kernel::gnn::ActivationType {
    switch (type) {
        case SCL_GNN_ACT_NONE:
            return scl::kernel::gnn::ActivationType::None;
        case SCL_GNN_ACT_RELU:
            return scl::kernel::gnn::ActivationType::ReLU;
        case SCL_GNN_ACT_LEAKY_RELU:
            return scl::kernel::gnn::ActivationType::LeakyReLU;
        case SCL_GNN_ACT_SIGMOID:
            return scl::kernel::gnn::ActivationType::Sigmoid;
        case SCL_GNN_ACT_TANH:
            return scl::kernel::gnn::ActivationType::Tanh;
        case SCL_GNN_ACT_ELU:
            return scl::kernel::gnn::ActivationType::ELU;
        case SCL_GNN_ACT_GELU:
            return scl::kernel::gnn::ActivationType::GELU;
        default:
            return scl::kernel::gnn::ActivationType::ReLU;
    }
}

} // anonymous namespace

extern "C" {

// =============================================================================
// Message Passing
// =============================================================================

SCL_EXPORT scl_error_t scl_gnn_message_passing(
    scl_sparse_t adjacency,
    const scl_real_t* node_features,
    const scl_index_t feat_dim,
    scl_real_t* output,
    const scl_size_t n_nodes,
    const scl_gnn_agg_type_t agg_type) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(node_features, "Node features array is null");
    SCL_C_API_CHECK_NULL(output, "Output array is null");
    SCL_C_API_CHECK(n_nodes > 0 && feat_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Index n = adjacency->rows();
        SCL_C_API_CHECK(static_cast<scl_size_t>(n) == n_nodes,
                       SCL_ERROR_DIMENSION_MISMATCH, "Node count mismatch");
        
        const Size feature_size = n_nodes * static_cast<Size>(feat_dim);
        
        Array<const Real> features_arr(
            reinterpret_cast<const Real*>(node_features),
            feature_size
        );
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            feature_size
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::gnn::message_passing(
                adj, features_arr, feat_dim, output_arr,
                convert_agg_type(agg_type)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Graph Attention
// =============================================================================

SCL_EXPORT scl_error_t scl_gnn_graph_attention(
    scl_sparse_t adjacency,
    const scl_real_t* node_features,
    const scl_index_t feat_dim,
    const scl_real_t* attention_vec,
    scl_real_t* output,
    const scl_size_t n_nodes,
    const int add_self_loops) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(node_features, "Node features array is null");
    SCL_C_API_CHECK_NULL(attention_vec, "Attention vector is null");
    SCL_C_API_CHECK_NULL(output, "Output array is null");
    SCL_C_API_CHECK(n_nodes > 0 && feat_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size feature_size = n_nodes * static_cast<Size>(feat_dim);
        
        Array<const Real> features_arr(
            reinterpret_cast<const Real*>(node_features),
            feature_size
        );
        Array<const Real> attn_arr(
            reinterpret_cast<const Real*>(attention_vec),
            static_cast<Size>(feat_dim) * 2
        );
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            feature_size
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::gnn::graph_attention(
                adj, features_arr, feat_dim, attn_arr, output_arr,
                add_self_loops != 0
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Graph Convolution
// =============================================================================

SCL_EXPORT scl_error_t scl_gnn_graph_convolution(
    scl_sparse_t adjacency,
    const scl_real_t* node_features,
    const scl_index_t in_dim,
    const scl_real_t* weight,
    const scl_index_t out_dim,
    scl_real_t* output,
    const scl_size_t n_nodes,
    const int add_self_loops,
    const scl_gnn_act_type_t activation) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(node_features, "Node features array is null");
    SCL_C_API_CHECK_NULL(weight, "Weight matrix is null");
    SCL_C_API_CHECK_NULL(output, "Output array is null");
    SCL_C_API_CHECK(n_nodes > 0 && in_dim > 0 && out_dim > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size input_size = n_nodes * static_cast<Size>(in_dim);
        const Size output_size = n_nodes * static_cast<Size>(out_dim);
        const Size weight_size = static_cast<Size>(in_dim) * static_cast<Size>(out_dim);
        
        Array<const Real> features_arr(
            reinterpret_cast<const Real*>(node_features),
            input_size
        );
        Array<const Real> weight_arr(
            reinterpret_cast<const Real*>(weight),
            weight_size
        );
        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            output_size
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::gnn::graph_convolution(
                adj, features_arr, in_dim, weight_arr, out_dim, output_arr,
                add_self_loops != 0, convert_act_type(activation)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Feature Smoothing
// =============================================================================

SCL_EXPORT scl_error_t scl_gnn_feature_smoothing(
    scl_sparse_t adjacency,
    scl_real_t* features,
    const scl_index_t n_nodes,
    const scl_index_t feat_dim,
    const scl_real_t alpha,
    const scl_index_t n_iterations) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(features, "Features array is null");
    SCL_C_API_CHECK(n_nodes > 0 && feat_dim > 0 && n_iterations > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    SCL_C_API_CHECK(alpha >= 0 && alpha <= 1, SCL_ERROR_INVALID_ARGUMENT,
                   "Alpha must be in [0, 1]");
    
    SCL_C_API_TRY
        const Size feature_size = static_cast<Size>(n_nodes) * static_cast<Size>(feat_dim);
        
        Array<Real> features_arr(
            reinterpret_cast<Real*>(features),
            feature_size
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::gnn::feature_smoothing(
                adj, features_arr, feat_dim, static_cast<Real>(alpha), n_iterations
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
