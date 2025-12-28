// =============================================================================
// FILE: scl/binding/c_api/gnn.cpp
// BRIEF: C API implementation for Graph Neural Network Operations
// =============================================================================

#include "scl/binding/c_api/gnn.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/gnn.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>
#include <cstring>
#include <random>
#include <cmath>

extern "C" {

// Internal helper to convert C++ exception to error code
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

// Convert C aggregation type to C++
static scl::kernel::gnn::AggregationType convert_agg_type(scl_gnn_agg_type_t type) {
    switch (type) {
        case SCL_GNN_AGG_SUM:
            return scl::kernel::gnn::AggregationType::Sum;
        case SCL_GNN_AGG_MEAN:
            return scl::kernel::gnn::AggregationType::Mean;
        case SCL_GNN_AGG_MAX:
            return scl::kernel::gnn::AggregationType::Max;
        case SCL_GNN_AGG_MIN:
            return scl::kernel::gnn::AggregationType::Min;
        case SCL_GNN_AGG_ATTENTION:
            return scl::kernel::gnn::AggregationType::Attention;
        case SCL_GNN_AGG_WEIGHTED:
            return scl::kernel::gnn::AggregationType::Weighted;
        default:
            return scl::kernel::gnn::AggregationType::Mean;
    }
}

// Convert C activation type to C++
static scl::kernel::gnn::ActivationType convert_activation_type(scl_gnn_activation_type_t type) {
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
            return scl::kernel::gnn::ActivationType::None;
    }
}

scl_error_t scl_gnn_message_passing(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_real_t* output,
    scl_gnn_agg_type_t agg_type
) {
    if (!adjacency || !node_features || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);
        const scl::Size F = static_cast<scl::Size>(feat_dim);
        
        scl::Array<const scl::Real> features_arr(
            reinterpret_cast<const scl::Real*>(node_features),
            N * F
        );
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            N * F
        );
        
        scl::kernel::gnn::message_passing(
            *sparse,
            features_arr,
            static_cast<scl::Index>(feat_dim),
            output_arr,
            convert_agg_type(agg_type)
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_graph_attention(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    const scl_real_t* attention_vec,
    scl_real_t* output,
    int add_self_loops
) {
    if (!adjacency || !node_features || !attention_vec || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);
        const scl::Size F = static_cast<scl::Size>(feat_dim);
        
        scl::Array<const scl::Real> features_arr(
            reinterpret_cast<const scl::Real*>(node_features),
            N * F
        );
        scl::Array<const scl::Real> attn_arr(
            reinterpret_cast<const scl::Real*>(attention_vec),
            static_cast<scl::Size>(2 * feat_dim)
        );
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            N * F
        );
        
        scl::kernel::gnn::graph_attention(
            *sparse,
            features_arr,
            static_cast<scl::Index>(feat_dim),
            attn_arr,
            output_arr,
            add_self_loops != 0
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_multi_head_attention(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_index_t n_heads,
    const scl_real_t* attention_vecs,
    scl_real_t* output,
    int add_self_loops
) {
    if (!adjacency || !node_features || !attention_vecs || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);
        const scl::Size F = static_cast<scl::Size>(feat_dim);
        const scl::Index head_dim = feat_dim / n_heads;
        const scl::Size head_dim_size = static_cast<scl::Size>(head_dim);
        
        scl::Array<const scl::Real> features_arr(
            reinterpret_cast<const scl::Real*>(node_features),
            N * F
        );
        scl::Array<const scl::Real> attn_arr(
            reinterpret_cast<const scl::Real*>(attention_vecs),
            static_cast<scl::Size>(n_heads) * static_cast<scl::Size>(2 * head_dim)
        );
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            N * static_cast<scl::Size>(n_heads) * head_dim_size
        );
        
        scl::kernel::gnn::multi_head_attention(
            *sparse,
            features_arr,
            static_cast<scl::Index>(feat_dim),
            static_cast<scl::Index>(n_heads),
            attn_arr,
            output_arr,
            add_self_loops != 0
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_graph_convolution(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t in_dim,
    const scl_real_t* weight,
    scl_index_t out_dim,
    scl_real_t* output,
    int add_self_loops,
    scl_gnn_activation_type_t activation
) {
    if (!adjacency || !node_features || !weight || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);
        const scl::Size in_dim_size = static_cast<scl::Size>(in_dim);
        const scl::Size out_dim_size = static_cast<scl::Size>(out_dim);
        
        scl::Array<const scl::Real> features_arr(
            reinterpret_cast<const scl::Real*>(node_features),
            N * in_dim_size
        );
        scl::Array<const scl::Real> weight_arr(
            reinterpret_cast<const scl::Real*>(weight),
            out_dim_size * in_dim_size
        );
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            N * out_dim_size
        );
        
        scl::kernel::gnn::graph_convolution(
            *sparse,
            features_arr,
            static_cast<scl::Index>(in_dim),
            weight_arr,
            static_cast<scl::Index>(out_dim),
            output_arr,
            add_self_loops != 0,
            convert_activation_type(activation)
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_sage_aggregate(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_real_t* output,
    scl_gnn_agg_type_t agg_type,
    scl_index_t max_neighbors
) {
    if (!adjacency || !node_features || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);
        const scl::Size F = static_cast<scl::Size>(feat_dim);
        
        scl::Array<const scl::Real> features_arr(
            reinterpret_cast<const scl::Real*>(node_features),
            N * F
        );
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            N * F
        );
        
        scl::kernel::gnn::sage_aggregate(
            *sparse,
            features_arr,
            static_cast<scl::Index>(feat_dim),
            output_arr,
            convert_agg_type(agg_type),
            static_cast<scl::Index>(max_neighbors)
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_feature_smoothing(
    scl_sparse_matrix_t adjacency,
    scl_real_t* features,
    scl_index_t n_nodes,
    scl_index_t feat_dim,
    scl_real_t alpha,
    scl_index_t n_iterations
) {
    if (!adjacency || !features) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Size N = static_cast<scl::Size>(n_nodes);
        const scl::Size F = static_cast<scl::Size>(feat_dim);
        
        scl::Array<scl::Real> features_arr(
            reinterpret_cast<scl::Real*>(features),
            N * F
        );
        
        scl::kernel::gnn::feature_smoothing(
            *sparse,
            features_arr,
            static_cast<scl::Index>(n_nodes),
            static_cast<scl::Index>(feat_dim),
            static_cast<scl::Real>(alpha),
            static_cast<scl::Index>(n_iterations)
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_global_pool(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_real_t* graph_features,
    scl_gnn_agg_type_t agg_type
) {
    if (!adjacency || !node_features || !graph_features) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);
        const scl::Size F = static_cast<scl::Size>(feat_dim);
        
        scl::Array<const scl::Real> features_arr(
            reinterpret_cast<const scl::Real*>(node_features),
            N * F
        );
        scl::Array<scl::Real> graph_arr(
            reinterpret_cast<scl::Real*>(graph_features),
            F
        );
        
        scl::kernel::gnn::global_pool(
            *sparse,
            features_arr,
            static_cast<scl::Index>(feat_dim),
            graph_arr,
            convert_agg_type(agg_type)
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_hierarchical_pool(
    const scl_real_t* node_features,
    scl_index_t n_nodes,
    scl_index_t feat_dim,
    const scl_index_t* cluster_assignment,
    scl_index_t n_clusters,
    scl_real_t* pooled_features,
    scl_gnn_agg_type_t agg_type
) {
    if (!node_features || !cluster_assignment || !pooled_features) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const scl::Size N = static_cast<scl::Size>(n_nodes);
        const scl::Size F = static_cast<scl::Size>(feat_dim);
        const scl::Size C = static_cast<scl::Size>(n_clusters);
        
        scl::Array<const scl::Real> features_arr(
            reinterpret_cast<const scl::Real*>(node_features),
            N * F
        );
        scl::Array<const scl::Index> cluster_arr(
            reinterpret_cast<const scl::Index*>(cluster_assignment),
            N
        );
        scl::Array<scl::Real> pooled_arr(
            reinterpret_cast<scl::Real*>(pooled_features),
            C * F
        );
        
        scl::kernel::gnn::hierarchical_pool(
            features_arr,
            static_cast<scl::Index>(n_nodes),
            static_cast<scl::Index>(feat_dim),
            cluster_arr,
            static_cast<scl::Index>(n_clusters),
            pooled_arr,
            convert_agg_type(agg_type)
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_compute_edge_features(
    scl_sparse_matrix_t adjacency,
    const scl_real_t* node_features,
    scl_index_t feat_dim,
    scl_real_t* edge_features,
    int concat
) {
    if (!adjacency || !node_features || !edge_features) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(adjacency);
        const scl::Index n = sparse->primary_dim();
        const scl::Size N = static_cast<scl::Size>(n);
        const scl::Size F = static_cast<scl::Size>(feat_dim);
        const scl::Size nnz = static_cast<scl::Size>(sparse->nnz());
        const scl::Size edge_dim = concat ? (2 * F) : F;
        
        scl::Array<const scl::Real> features_arr(
            reinterpret_cast<const scl::Real*>(node_features),
            N * F
        );
        scl::Array<scl::Real> edge_arr(
            reinterpret_cast<scl::Real*>(edge_features),
            nnz * edge_dim
        );
        
        scl::kernel::gnn::compute_edge_features(
            *sparse,
            features_arr,
            static_cast<scl::Index>(feat_dim),
            edge_arr,
            concat != 0
        );
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_skip_connection(
    const scl_real_t* input,
    const scl_real_t* residual,
    scl_real_t* output,
    scl_real_t alpha,
    scl_size_t n
) {
    if (!input || !residual || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Real> input_arr(
            reinterpret_cast<const scl::Real*>(input),
            n
        );
        scl::Array<const scl::Real> residual_arr(
            reinterpret_cast<const scl::Real*>(residual),
            n
        );
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            n
        );
        
        for (scl::Size i = 0; i < n; ++i) {
            output_arr[i] = input_arr[i] + static_cast<scl::Real>(alpha) * residual_arr[i];
        }
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_layer_norm(
    scl_real_t* features,
    scl_index_t n_nodes,
    scl_index_t feat_dim,
    scl_real_t epsilon
) {
    if (!features) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const scl::Size N = static_cast<scl::Size>(n_nodes);
        const scl::Size F = static_cast<scl::Size>(feat_dim);
        
        scl::Array<scl::Real> features_arr(
            reinterpret_cast<scl::Real*>(features),
            N * F
        );
        
        for (scl::Size i = 0; i < N; ++i) {
            scl::Real* feat = features_arr.ptr + i * F;
            scl::Real mean = scl::Real(0);
            scl::Real var = scl::Real(0);
            
            for (scl::Index j = 0; j < feat_dim; ++j) {
                mean += feat[j];
            }
            mean /= static_cast<scl::Real>(feat_dim);
            
            for (scl::Index j = 0; j < feat_dim; ++j) {
                scl::Real diff = feat[j] - mean;
                var += diff * diff;
            }
            var /= static_cast<scl::Real>(feat_dim);
            
            scl::Real std_val = std::sqrt(var + static_cast<scl::Real>(epsilon));
            
            for (scl::Index j = 0; j < feat_dim; ++j) {
                feat[j] = (feat[j] - mean) / std_val;
            }
        }
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_batch_norm(
    scl_real_t* features,
    scl_index_t n_nodes,
    scl_index_t feat_dim,
    const scl_real_t* gamma,
    const scl_real_t* beta,
    scl_real_t epsilon
) {
    if (!features || !gamma || !beta) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const scl::Size N = static_cast<scl::Size>(n_nodes);
        const scl::Size F = static_cast<scl::Size>(feat_dim);
        
        scl::Array<scl::Real> features_arr(
            reinterpret_cast<scl::Real*>(features),
            N * F
        );
        scl::Array<const scl::Real> gamma_arr(
            reinterpret_cast<const scl::Real*>(gamma),
            F
        );
        scl::Array<const scl::Real> beta_arr(
            reinterpret_cast<const scl::Real*>(beta),
            F
        );
        
        for (scl::Index j = 0; j < feat_dim; ++j) {
            scl::Real mean = scl::Real(0);
            scl::Real var = scl::Real(0);
            
            for (scl::Size i = 0; i < N; ++i) {
                mean += features_arr[i * F + j];
            }
            mean /= static_cast<scl::Real>(n_nodes);
            
            for (scl::Size i = 0; i < N; ++i) {
                scl::Real diff = features_arr[i * F + j] - mean;
                var += diff * diff;
            }
            var /= static_cast<scl::Real>(n_nodes);
            
            scl::Real std_val = std::sqrt(var + static_cast<scl::Real>(epsilon));
            
            for (scl::Size i = 0; i < N; ++i) {
                features_arr[i * F + j] = gamma_arr[j] * (features_arr[i * F + j] - mean) / std_val + beta_arr[j];
            }
        }
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gnn_dropout(
    scl_real_t* features,
    scl_real_t dropout_rate,
    uint64_t seed,
    int training,
    scl_size_t n
) {
    if (!features) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    if (training == 0 || dropout_rate <= 0.0 || dropout_rate >= 1.0) {
        return SCL_ERROR_OK;
    }
    
    try {
        scl::Array<scl::Real> features_arr(
            reinterpret_cast<scl::Real*>(features),
            n
        );
        
        std::mt19937 gen(static_cast<unsigned int>(seed));
        std::uniform_real_distribution<scl::Real> dis(0.0, 1.0);
        scl::Real scale = scl::Real(1) / (scl::Real(1) - static_cast<scl::Real>(dropout_rate));
        
        for (scl::Size i = 0; i < n; ++i) {
            if (dis(gen) < static_cast<scl::Real>(dropout_rate)) {
                features_arr[i] = scl::Real(0);
            } else {
                features_arr[i] *= scale;
            }
        }
        
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::gnn::message_passing<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Index,
    scl::Array<scl::Real>,
    scl::kernel::gnn::AggregationType
);

template void scl::kernel::gnn::graph_attention<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Index,
    scl::Array<const scl::Real>,
    scl::Array<scl::Real>,
    bool
);

template void scl::kernel::gnn::multi_head_attention<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Index,
    scl::Index,
    scl::Array<const scl::Real>,
    scl::Array<scl::Real>,
    bool
);

template void scl::kernel::gnn::graph_convolution<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Index,
    scl::Array<const scl::Real>,
    scl::Index,
    scl::Array<scl::Real>,
    bool,
    scl::kernel::gnn::ActivationType
);

template void scl::kernel::gnn::sage_aggregate<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Index,
    scl::Array<scl::Real>,
    scl::kernel::gnn::AggregationType,
    scl::Index
);

template void scl::kernel::gnn::feature_smoothing<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Real>,
    scl::Index,
    scl::Index,
    scl::Real,
    scl::Index
);

template void scl::kernel::gnn::global_pool<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Index,
    scl::Array<scl::Real>,
    scl::kernel::gnn::AggregationType
);

template void scl::kernel::gnn::hierarchical_pool<scl::Real, true>(
    scl::Array<const scl::Real>,
    scl::Index,
    scl::Index,
    scl::Array<const scl::Index>,
    scl::Index,
    scl::Array<scl::Real>,
    scl::kernel::gnn::AggregationType
);

template void scl::kernel::gnn::compute_edge_features<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Index,
    scl::Array<scl::Real>,
    bool
);

} // extern "C"

