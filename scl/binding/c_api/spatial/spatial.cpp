// =============================================================================
// FILE: scl/binding/c_api/spatial/spatial.cpp
// BRIEF: C API implementation for spatial statistics
// =============================================================================

#include "scl/binding/c_api/spatial/spatial.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/spatial.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_spatial_weight_sum(
    scl_sparse_t graph,
    scl_real_t* weight_sum
) {
    if (!graph || !weight_sum) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(graph);
        scl::Real sum = scl::Real(0);

        sparse->visit([&](auto& m) {
            scl::kernel::spatial::weight_sum(m, sum);
        });

        *weight_sum = static_cast<scl_real_t>(sum);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_morans_i(
    scl_sparse_t graph,
    scl_sparse_t features,
    scl_real_t* output
) {
    if (!graph || !features || !output) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* graph_sparse = static_cast<scl_sparse_matrix*>(graph);
        auto* feat_sparse = static_cast<scl_sparse_matrix*>(features);
        scl::Index n_features = feat_sparse->rows();

        graph_sparse->visit([&](auto& graph_m) {
            feat_sparse->visit([&](auto& feat_m) {
                scl::Array<scl::Real> out(reinterpret_cast<scl::Real*>(output),
                                         static_cast<scl::Size>(n_features));
                scl::kernel::spatial::morans_i(graph_m, feat_m, out);
            });
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_gearys_c(
    scl_sparse_t graph,
    scl_sparse_t features,
    scl_real_t* output
) {
    if (!graph || !features || !output) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* graph_sparse = static_cast<scl_sparse_matrix*>(graph);
        auto* feat_sparse = static_cast<scl_sparse_matrix*>(features);
        scl::Index n_features = feat_sparse->rows();

        graph_sparse->visit([&](auto& graph_m) {
            feat_sparse->visit([&](auto& feat_m) {
                scl::Array<scl::Real> out(reinterpret_cast<scl::Real*>(output),
                                         static_cast<scl::Size>(n_features));
                scl::kernel::spatial::gearys_c(graph_m, feat_m, out);
            });
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
