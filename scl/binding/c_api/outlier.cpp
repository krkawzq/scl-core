// =============================================================================
// FILE: scl/binding/c_api/outlier/outlier.cpp
// BRIEF: C API implementation for outlier detection
// =============================================================================

#include "scl/binding/c_api/outlier.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/outlier.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Isolation Score
// =============================================================================

SCL_EXPORT scl_error_t scl_outlier_isolation_score(
    scl_sparse_t data,
    scl_real_t* scores) {
    
    SCL_C_API_CHECK_NULL(data, "Data matrix is null");
    SCL_C_API_CHECK_NULL(scores, "Scores array is null");

    SCL_C_API_TRY
        const Index n_cells = data->rows();

        data->visit([&](auto& m) {
            Array<Real> score_arr(reinterpret_cast<Real*>(scores), static_cast<Size>(n_cells));
            scl::kernel::outlier::isolation_score(m, score_arr);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Local Outlier Factor - NOT YET IMPLEMENTED
// =============================================================================
//
// TODO: Implement LOF when type-specific sparse matrices are added to C API
//
// The kernel function signature is:
//   void local_outlier_factor(
//       const Sparse<T, IsCSR>& data,
//       const Sparse<Index, IsCSR>& neighbors,  // ← Different value type!
//       const Sparse<Real, IsCSR>& distances,   // ← Different value type!
//       Array<Real> lof_scores
//   )
//
// Current C API limitation:
//   - scl_sparse_t only wraps Sparse<Real, IsCSR>
//   - Cannot represent Sparse<Index, IsCSR> for neighbor indices
//
// This function is commented out to prevent compilation errors
// Uncomment when type-specific sparse matrix support is added
/*
SCL_EXPORT scl_error_t scl_outlier_local_outlier_factor(
    scl_sparse_t data,
    scl_sparse_t neighbors,  // Would need scl_sparse_index_t
    scl_sparse_t distances,
    scl_real_t* lof_scores
) {
    // Implementation pending type system extension
}
*/

// =============================================================================
// Ambient Detection
// =============================================================================

SCL_EXPORT scl_error_t scl_outlier_ambient_detection(
    scl_sparse_t expression,
    scl_real_t* ambient_scores) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(ambient_scores, "Ambient scores array is null");

    SCL_C_API_TRY
        const Index n_cells = expression->rows();

        expression->visit([&](auto& m) {
            Array<Real> scores(reinterpret_cast<Real*>(ambient_scores), static_cast<Size>(n_cells));
            scl::kernel::outlier::ambient_detection(m, scores);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
