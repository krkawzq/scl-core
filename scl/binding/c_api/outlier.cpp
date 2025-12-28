// =============================================================================
// FILE: scl/binding/c_api/outlier/outlier.cpp
// BRIEF: C API implementation for outlier detection
// =============================================================================

#include "scl/binding/c_api/outlier.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/outlier.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_outlier_isolation_score(
    scl_sparse_t data,
    scl_real_t* scores
) {
    if (!data || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(data);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> score_arr(reinterpret_cast<scl::Real*>(scores),
                                           static_cast<scl::Size>(n_cells));
            scl::kernel::outlier::isolation_score(m, score_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_outlier_local_outlier_factor(
    scl_sparse_t data,
    scl_sparse_t neighbors,
    scl_sparse_t distances,
    scl_real_t* lof_scores
) {
    if (!data || !neighbors || !distances || !lof_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* data_sparse = static_cast<scl_sparse_matrix*>(data);
        auto* neigh_sparse = static_cast<scl_sparse_matrix*>(neighbors);
        auto* dist_sparse = static_cast<scl_sparse_matrix*>(distances);
        scl::Index n_cells = data_sparse->rows();

        data_sparse->visit([&](auto& data_m) {
            neigh_sparse->visit([&](auto& neigh_m) {
                dist_sparse->visit([&](auto& dist_m) {
                    scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(lof_scores),
                                                 static_cast<scl::Size>(n_cells));
                    scl::kernel::outlier::local_outlier_factor(data_m, neigh_m, dist_m, scores);
                });
            });
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_outlier_ambient_detection(
    scl_sparse_t expression,
    scl_real_t* ambient_scores
) {
    if (!expression || !ambient_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(ambient_scores),
                                        static_cast<scl::Size>(n_cells));
            scl::kernel::outlier::ambient_detection(m, scores);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_outlier_empty_drops(
    scl_sparse_t raw_counts,
    unsigned char* is_empty,
    scl_real_t fdr_threshold
) {
    if (!raw_counts || !is_empty) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(raw_counts);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<bool> empty_arr(reinterpret_cast<bool*>(is_empty),
                                      static_cast<scl::Size>(n_cells));
            scl::kernel::outlier::empty_drops(m, empty_arr, static_cast<scl::Real>(fdr_threshold));
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_outlier_outlier_genes(
    scl_sparse_t expression,
    scl_index_t* outlier_gene_indices,
    scl_size_t max_outliers,
    scl_size_t* n_outliers,
    scl_real_t threshold
) {
    if (!expression || !outlier_gene_indices || !n_outliers) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Size n_out = 0;

        sparse->visit([&](auto& m) {
            scl::kernel::outlier::outlier_genes(
                m, outlier_gene_indices, n_out, static_cast<scl::Real>(threshold)
            );
        });

        *n_outliers = n_out;
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_outlier_doublet_score(
    scl_sparse_t expression,
    scl_sparse_t neighbors,
    scl_real_t* scores
) {
    if (!expression || !neighbors || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* expr_sparse = static_cast<scl_sparse_matrix*>(expression);
        auto* neigh_sparse = static_cast<scl_sparse_matrix*>(neighbors);
        scl::Index n_cells = expr_sparse->rows();

        expr_sparse->visit([&](auto& expr) {
            neigh_sparse->visit([&](auto& neigh) {
                scl::Array<scl::Real> score_arr(reinterpret_cast<scl::Real*>(scores),
                                               static_cast<scl::Size>(n_cells));
                scl::kernel::outlier::doublet_score(expr, neigh, score_arr);
            });
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_outlier_mitochondrial_outliers(
    scl_sparse_t expression,
    const scl_index_t* mito_genes,
    scl_size_t n_mito_genes,
    scl_real_t* mito_fraction,
    unsigned char* is_outlier,
    scl_real_t threshold
) {
    if (!expression || !mito_genes || !mito_fraction || !is_outlier) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> mito(mito_genes, n_mito_genes);
            scl::Array<scl::Real> frac(reinterpret_cast<scl::Real*>(mito_fraction),
                                      static_cast<scl::Size>(n_cells));
            scl::Array<bool> outlier(reinterpret_cast<bool*>(is_outlier),
                                    static_cast<scl::Size>(n_cells));
            scl::kernel::outlier::mitochondrial_outliers(
                m, mito, frac, outlier, static_cast<scl::Real>(threshold)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_outlier_qc_filter(
    scl_sparse_t expression,
    scl_real_t min_genes,
    scl_real_t max_genes,
    scl_real_t min_counts,
    scl_real_t max_counts,
    scl_real_t max_mito_fraction,
    const scl_index_t* mito_genes,
    scl_size_t n_mito_genes,
    unsigned char* pass_qc
) {
    if (!expression || !pass_qc) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Index> mito;
            if (mito_genes && n_mito_genes > 0) {
                mito = scl::Array<const scl::Index>(mito_genes, n_mito_genes);
            }
            scl::Array<bool> pass(reinterpret_cast<bool*>(pass_qc),
                                 static_cast<scl::Size>(n_cells));
            scl::kernel::outlier::qc_filter(
                m, static_cast<scl::Real>(min_genes), static_cast<scl::Real>(max_genes),
                static_cast<scl::Real>(min_counts), static_cast<scl::Real>(max_counts),
                static_cast<scl::Real>(max_mito_fraction), mito, pass
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
