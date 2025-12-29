// =============================================================================
// FILE: scl/binding/c_api/sampling.cpp
// BRIEF: C API implementation for advanced sampling
// =============================================================================

#include "scl/binding/c_api/sampling.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/sampling.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Geometric Sketching
// =============================================================================

SCL_EXPORT scl_error_t scl_sampling_geometric_sketching(
    scl_sparse_t data,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(data, "Data matrix handle is null");
    SCL_C_API_CHECK_NULL(selected_indices, "Selected indices pointer is null");
    SCL_C_API_CHECK_NULL(n_selected, "Number selected pointer is null");

    SCL_C_API_TRY {
        const Size target_size_sz = static_cast<Size>(target_size);
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size_sz);
        
        data->visit([&](auto& m) {
            scl::kernel::sampling::geometric_sketching(m, target_size_sz, indices_arr.ptr, *n_selected, seed);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Density Preserving Sampling - NOT YET IMPLEMENTED
// =============================================================================
//
// TODO: Implement when type-specific sparse matrices are added to C API
//
// The kernel function signature is:
//   void density_preserving(
//       const Sparse<T, IsCSR>& data,
//       const Sparse<Index, IsCSR>& neighbors,  // ← Different value type!
//       Size target_size,
//       Index* selected_indices,
//       Size& n_selected
//   )
//
// Current C API limitation:
//   - scl_sparse_t only wraps Sparse<Real, IsCSR>
//   - Cannot represent Sparse<Index, IsCSR> for neighbor indices
//
// This function is commented out to prevent compilation errors
/*
SCL_EXPORT scl_error_t scl_sampling_density_preserving(...) {
    // Implementation pending type system extension
}
*/

// =============================================================================
// Landmark Selection
// =============================================================================

SCL_EXPORT scl_error_t scl_sampling_landmark_selection(
    scl_sparse_t data,
    scl_size_t n_landmarks,
    scl_index_t* landmark_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(data, "Data matrix handle is null");
    SCL_C_API_CHECK_NULL(landmark_indices, "Landmark indices pointer is null");
    SCL_C_API_CHECK_NULL(n_selected, "Number selected pointer is null");

    SCL_C_API_TRY {
        const Size n_landmarks_sz = static_cast<Size>(n_landmarks);
        Array<Index> indices_arr(reinterpret_cast<Index*>(landmark_indices), n_landmarks_sz);
        
        data->visit([&](auto& m) {
            scl::kernel::sampling::landmark_selection(m, n_landmarks_sz, indices_arr.ptr, *n_selected, seed);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Representative Cells
// =============================================================================

SCL_EXPORT scl_error_t scl_sampling_representative_cells(
    scl_sparse_t data,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_size_t per_cluster,
    scl_index_t* representatives,
    scl_size_t* n_selected,
    scl_size_t max_count,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(data, "Data matrix handle is null");
    SCL_C_API_CHECK_NULL(cluster_labels, "Cluster labels pointer is null");
    SCL_C_API_CHECK_NULL(representatives, "Representatives pointer is null");
    SCL_C_API_CHECK_NULL(n_selected, "Number selected pointer is null");

    SCL_C_API_TRY {
        const Size n_cells_sz = static_cast<Size>(n_cells);
        const Size max_count_sz = static_cast<Size>(max_count);
        
        Array<const Index> labels_arr(reinterpret_cast<const Index*>(cluster_labels), n_cells_sz);
        Array<Index> reps_arr(reinterpret_cast<Index*>(representatives), max_count_sz);
        
        data->visit([&](auto& m) {
            scl::kernel::sampling::representative_cells(m, labels_arr, per_cluster, reps_arr.ptr, *n_selected, seed);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Balanced Sampling
// =============================================================================

SCL_EXPORT scl_error_t scl_sampling_balanced(
    const scl_index_t* labels,
    scl_size_t n,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(labels, "Labels pointer is null");
    SCL_C_API_CHECK_NULL(selected_indices, "Selected indices pointer is null");
    SCL_C_API_CHECK_NULL(n_selected, "Number selected pointer is null");

    SCL_C_API_TRY {
        const Size n_sz = static_cast<Size>(n);
        const Size target_size_sz = static_cast<Size>(target_size);
        
        Array<const Index> labels_arr(reinterpret_cast<const Index*>(labels), n_sz);
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size_sz);
        
        scl::kernel::sampling::balanced_sampling(labels_arr, target_size_sz, indices_arr.ptr, *n_selected, seed);
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Stratified Sampling
// =============================================================================

SCL_EXPORT scl_error_t scl_sampling_stratified(
    const scl_real_t* values,
    scl_size_t n,
    scl_size_t n_strata,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(values, "Values pointer is null");
    SCL_C_API_CHECK_NULL(selected_indices, "Selected indices pointer is null");
    SCL_C_API_CHECK_NULL(n_selected, "Number selected pointer is null");

    SCL_C_API_TRY {
        const Size n_sz = static_cast<Size>(n);
        const Size target_size_sz = static_cast<Size>(target_size);
        
        Array<const Real> values_arr(reinterpret_cast<const Real*>(values), n_sz);
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size_sz);
        
        scl::kernel::sampling::stratified_sampling(values_arr, n_strata, target_size_sz, indices_arr.ptr, *n_selected, seed);
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Uniform Sampling
// =============================================================================

SCL_EXPORT scl_error_t scl_sampling_uniform(
    scl_size_t n,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(selected_indices, "Selected indices pointer is null");
    SCL_C_API_CHECK_NULL(n_selected, "Number selected pointer is null");

    SCL_C_API_TRY {
        const Size target_size_sz = static_cast<Size>(target_size);
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size_sz);
        
        scl::kernel::sampling::uniform_sampling(n, target_size_sz, indices_arr.ptr, *n_selected, seed);
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Importance Sampling
// =============================================================================

SCL_EXPORT scl_error_t scl_sampling_importance(
    const scl_real_t* weights,
    scl_size_t n,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(weights, "Weights pointer is null");
    SCL_C_API_CHECK_NULL(selected_indices, "Selected indices pointer is null");
    SCL_C_API_CHECK_NULL(n_selected, "Number selected pointer is null");

    SCL_C_API_TRY {
        const Size n_sz = static_cast<Size>(n);
        const Size target_size_sz = static_cast<Size>(target_size);
        
        Array<const Real> weights_arr(reinterpret_cast<const Real*>(weights), n_sz);
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size_sz);
        
        scl::kernel::sampling::importance_sampling(weights_arr, target_size_sz, indices_arr.ptr, *n_selected, seed);
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Reservoir Sampling
// =============================================================================

SCL_EXPORT scl_error_t scl_sampling_reservoir(
    scl_size_t stream_size,
    scl_size_t reservoir_size,
    scl_index_t* reservoir,
    scl_size_t* n_selected,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(reservoir, "Reservoir pointer is null");
    SCL_C_API_CHECK_NULL(n_selected, "Number selected pointer is null");

    SCL_C_API_TRY {
        const Size reservoir_size_sz = static_cast<Size>(reservoir_size);
        Array<Index> res_arr(reinterpret_cast<Index*>(reservoir), reservoir_size_sz);
        
        scl::kernel::sampling::reservoir_sampling(stream_size, reservoir_size_sz, res_arr.ptr, *n_selected, seed);
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

} // extern "C"
