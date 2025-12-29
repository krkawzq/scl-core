// =============================================================================
// FILE: scl/binding/c_api/scoring/scoring.cpp
// BRIEF: C API implementation for gene set scoring
// =============================================================================

#include "scl/binding/c_api/scoring.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/scoring.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

[[nodiscard]] constexpr auto convert_scoring_method(
    scl_scoring_method_t method) noexcept -> scl::kernel::scoring::ScoringMethod {
    using SM = scl::kernel::scoring::ScoringMethod;
    switch (method) {
        case SCL_SCORING_MEAN: return SM::Mean;
        case SCL_SCORING_RANK_BASED: return SM::RankBased;
        case SCL_SCORING_WEIGHTED: return SM::Weighted;
        case SCL_SCORING_SEURAT_MODULE: return SM::SeuratModule;
        case SCL_SCORING_ZSCORE: return SM::ZScore;
        default: return SM::Mean;
    }
}

} // anonymous namespace

extern "C" {

// =============================================================================
// Gene Set Score
// =============================================================================

SCL_EXPORT scl_error_t scl_scoring_gene_set_score(
    scl_sparse_t expression,
    const scl_index_t* gene_set,
    const scl_size_t n_genes_in_set,
    scl_real_t* scores,
    const scl_size_t n_cells,
    const scl_index_t n_genes,
    const scl_scoring_method_t method,
    const scl_real_t quantile) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(gene_set, "Gene set array is null");
    SCL_C_API_CHECK_NULL(scores, "Scores array is null");
    SCL_C_API_CHECK(n_genes_in_set > 0 && n_cells > 0 && n_genes > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");

    SCL_C_API_TRY
        Array<const Index> gene_set_arr(
            reinterpret_cast<const Index*>(gene_set),
            static_cast<Size>(n_genes_in_set)
        );
        Array<Real> scores_arr(reinterpret_cast<Real*>(scores), static_cast<Size>(n_cells));
        
        expression->visit([&](auto& expr) {
            scl::kernel::scoring::gene_set_score(
                expr, gene_set_arr,
                convert_scoring_method(method),
                scores_arr, n_cells, n_genes,
                static_cast<Real>(quantile)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
