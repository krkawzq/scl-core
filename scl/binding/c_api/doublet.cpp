#include "doublet.h"
#include "scl/kernel/doublet.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include <cstring>
#include <cmath>

static scl_error_t convert_error(scl::ErrorCode code) {
    return static_cast<scl_error_t>(code);
}

static scl::Sparse<scl::Real, true>* unwrap_matrix(scl_sparse_matrix_t mat) {
    return static_cast<scl::Sparse<scl::Real, true>*>(mat);
}

extern "C" {

scl_error_t scl_doublet_simulate_doublets(
    scl_sparse_matrix_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_doublets,
    scl_real_t* doublet_profiles,
    uint64_t seed
) {
    try {
        auto* mat = unwrap_matrix(X);
        if (!mat || !doublet_profiles) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::kernel::doublet::simulate_doublets(*mat, n_cells, n_genes, n_doublets, doublet_profiles, seed);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_doublet_compute_knn_scores(
    scl_sparse_matrix_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* doublet_profiles,
    scl_index_t n_doublets,
    scl_index_t k_neighbors,
    scl_real_t* doublet_scores
) {
    try {
        auto* mat = unwrap_matrix(X);
        if (!mat || !doublet_profiles || !doublet_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<scl::Real> scores_arr(doublet_scores, static_cast<scl::Size>(n_cells));
        scl::kernel::doublet::compute_knn_doublet_scores(*mat, n_cells, n_genes, doublet_profiles, n_doublets, k_neighbors, scores_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_doublet_compute_knn_scores_pca(
    const scl_real_t* cell_embeddings,
    scl_index_t n_cells,
    scl_index_t n_dims,
    const scl_real_t* doublet_embeddings,
    scl_index_t n_doublets,
    scl_index_t k_neighbors,
    scl_real_t* doublet_scores
) {
    try {
        if (!cell_embeddings || !doublet_embeddings || !doublet_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<scl::Real> scores_arr(doublet_scores, static_cast<scl::Size>(n_cells));
        scl::kernel::doublet::compute_knn_doublet_scores_pca(cell_embeddings, n_cells, n_dims, doublet_embeddings, n_doublets, k_neighbors, scores_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_doublet_scrublet_scores(
    scl_sparse_matrix_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* scores,
    scl_index_t n_simulated,
    scl_index_t k_neighbors,
    uint64_t seed
) {
    try {
        auto* mat = unwrap_matrix(X);
        if (!mat || !scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<scl::Real> scores_arr(scores, static_cast<scl::Size>(n_cells));
        scl::kernel::doublet::scrublet_scores(*mat, n_cells, n_genes, scores_arr, n_simulated, k_neighbors, seed);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_doublet_doubletfinder_pann(
    const scl_real_t* cell_embeddings,
    scl_index_t n_cells,
    scl_index_t n_dims,
    const scl_real_t* doublet_embeddings,
    scl_index_t n_doublets,
    scl_real_t pK,
    scl_real_t* pann_scores
) {
    try {
        if (!cell_embeddings || !doublet_embeddings || !pann_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<scl::Real> scores_arr(pann_scores, static_cast<scl::Size>(n_cells));
        scl::kernel::doublet::doubletfinder_pann(cell_embeddings, n_cells, n_dims, doublet_embeddings, n_doublets, pK, scores_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_real_t scl_doublet_estimate_threshold(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t expected_doublet_rate
) {
    try {
        if (!scores || n_scores == 0) {
            return scl::kernel::doublet::config::DEFAULT_THRESHOLD;
        }
        scl::Array<const scl::Real> scores_arr(scores, n_scores);
        return scl::kernel::doublet::estimate_threshold(scores_arr, expected_doublet_rate);
    } catch (...) {
        return scl::kernel::doublet::config::DEFAULT_THRESHOLD;
    }
}

scl_index_t scl_doublet_call_doublets(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t threshold,
    int* is_doublet
) {
    try {
        if (!scores || !is_doublet) {
            return 0;
        }
        scl::Array<const scl::Real> scores_arr(scores, n_scores);
        scl::Array<bool> is_doublet_arr(reinterpret_cast<bool*>(is_doublet), n_scores);
        return scl::kernel::doublet::call_doublets(scores_arr, threshold, is_doublet_arr);
    } catch (...) {
        return 0;
    }
}

scl_real_t scl_doublet_detect_bimodal_threshold(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_index_t n_bins
) {
    try {
        if (!scores || n_scores == 0) {
            return scl::kernel::doublet::config::DEFAULT_THRESHOLD;
        }
        scl::Array<const scl::Real> scores_arr(scores, n_scores);
        return scl::kernel::doublet::detect_bimodal_threshold(scores_arr, n_bins);
    } catch (...) {
        return scl::kernel::doublet::config::DEFAULT_THRESHOLD;
    }
}

scl_index_t scl_doublet_expected_doublets(
    scl_index_t n_cells,
    scl_real_t doublet_rate
) {
    return scl::kernel::doublet::expected_doublets(n_cells, doublet_rate);
}

scl_real_t scl_doublet_estimate_doublet_rate(
    scl_index_t n_cells_loaded,
    scl_real_t cells_per_droplet_mean
) {
    return scl::kernel::doublet::estimate_doublet_rate(n_cells_loaded, cells_per_droplet_mean);
}

scl_error_t scl_doublet_classify_doublet_types(
    const scl_index_t* cluster_labels,
    const int* is_doublet,
    scl_size_t n_cells,
    scl_index_t n_clusters,
    scl_index_t* doublet_type
) {
    try {
        if (!cluster_labels || !is_doublet || !doublet_type) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Index> labels_arr(cluster_labels, n_cells);
        scl::Array<const bool> is_doublet_arr(reinterpret_cast<const bool*>(is_doublet), n_cells);
        scl::Array<scl::Index> type_arr(doublet_type, n_cells);
        scl::kernel::doublet::classify_doublet_types(labels_arr, is_doublet_arr, n_clusters, type_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_doublet_classify_doublet_types_knn(
    scl_sparse_matrix_t knn_graph,
    const scl_index_t* cluster_labels,
    const int* is_doublet,
    scl_size_t n_cells,
    scl_index_t n_clusters,
    scl_index_t* doublet_type
) {
    try {
        auto* mat = unwrap_matrix(knn_graph);
        if (!mat || !cluster_labels || !is_doublet || !doublet_type) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Index> labels_arr(cluster_labels, n_cells);
        scl::Array<const bool> is_doublet_arr(reinterpret_cast<const bool*>(is_doublet), n_cells);
        scl::Array<scl::Index> type_arr(doublet_type, n_cells);
        scl::kernel::doublet::classify_doublet_types_knn(*mat, labels_arr, is_doublet_arr, n_clusters, type_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_doublet_density_doublet_score(
    scl_sparse_matrix_t knn_graph,
    scl_real_t* density_scores,
    scl_size_t n_cells
) {
    try {
        auto* mat = unwrap_matrix(knn_graph);
        if (!mat || !density_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<scl::Real> scores_arr(density_scores, n_cells);
        scl::kernel::doublet::density_doublet_score(*mat, scores_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_doublet_variance_doublet_score(
    scl_sparse_matrix_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* gene_means,
    scl_real_t* variance_scores
) {
    try {
        auto* mat = unwrap_matrix(X);
        if (!mat || !gene_means || !variance_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Real> means_arr(gene_means, static_cast<scl::Size>(n_genes));
        scl::Array<scl::Real> scores_arr(variance_scores, static_cast<scl::Size>(n_cells));
        scl::kernel::doublet::variance_doublet_score(*mat, n_cells, n_genes, means_arr, scores_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_doublet_combined_doublet_score(
    const scl_real_t* knn_scores,
    const scl_real_t* density_scores,
    const scl_real_t* variance_scores,
    scl_size_t n_cells,
    scl_real_t knn_weight,
    scl_real_t density_weight,
    scl_real_t variance_weight,
    scl_real_t* combined_scores
) {
    try {
        if (!knn_scores || !density_scores || !variance_scores || !combined_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Real> knn_arr(knn_scores, n_cells);
        scl::Array<const scl::Real> density_arr(density_scores, n_cells);
        scl::Array<const scl::Real> variance_arr(variance_scores, n_cells);
        scl::Array<scl::Real> combined_arr(combined_scores, n_cells);
        scl::kernel::doublet::combined_doublet_score(knn_arr, density_arr, variance_arr, combined_arr, knn_weight, density_weight, variance_weight);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_index_t scl_doublet_detect_doublets(
    scl_sparse_matrix_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* scores,
    int* is_doublet,
    scl_real_t expected_rate,
    scl_index_t k_neighbors,
    uint64_t seed
) {
    try {
        auto* mat = unwrap_matrix(X);
        if (!mat || !scores || !is_doublet) {
            return 0;
        }
        scl::Array<scl::Real> scores_arr(scores, static_cast<scl::Size>(n_cells));
        scl::Array<bool> is_doublet_arr(reinterpret_cast<bool*>(is_doublet), static_cast<scl::Size>(n_cells));
        return scl::kernel::doublet::detect_doublets(*mat, n_cells, n_genes, scores_arr, is_doublet_arr, scl::kernel::doublet::DoubletMethod::Scrublet, expected_rate, k_neighbors, seed);
    } catch (...) {
        return 0;
    }
}

scl_index_t scl_doublet_get_singlet_indices(
    const int* is_doublet,
    scl_size_t n_cells,
    scl_index_t* singlet_indices
) {
    try {
        if (!is_doublet || !singlet_indices) {
            return 0;
        }
        scl::Array<const bool> is_doublet_arr(reinterpret_cast<const bool*>(is_doublet), n_cells);
        scl::Array<scl::Index> indices_arr(singlet_indices, n_cells);
        return scl::kernel::doublet::get_singlet_indices(is_doublet_arr, indices_arr);
    } catch (...) {
        return 0;
    }
}

scl_error_t scl_doublet_doublet_score_stats(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t* mean,
    scl_real_t* std_dev,
    scl_real_t* median
) {
    try {
        if (!scores || !mean || !std_dev || !median) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Real> scores_arr(scores, n_scores);
        scl::kernel::doublet::doublet_score_stats(scores_arr, mean, std_dev, median);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_real_t scl_doublet_multiplet_rate_10x(
    scl_index_t n_cells_recovered
) {
    return scl::kernel::doublet::multiplet_rate_10x(n_cells_recovered);
}

scl_error_t scl_doublet_cluster_doublet_enrichment(
    const scl_real_t* doublet_scores,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_index_t n_clusters,
    scl_real_t* cluster_mean_scores,
    scl_real_t* cluster_doublet_fraction
) {
    try {
        if (!doublet_scores || !cluster_labels || !cluster_mean_scores || !cluster_doublet_fraction) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Real> scores_arr(doublet_scores, n_cells);
        scl::Array<const scl::Index> labels_arr(cluster_labels, n_cells);
        scl::Array<scl::Real> mean_arr(cluster_mean_scores, static_cast<scl::Size>(n_clusters));
        scl::Array<scl::Real> frac_arr(cluster_doublet_fraction, static_cast<scl::Size>(n_clusters));
        scl::kernel::doublet::cluster_doublet_enrichment(scores_arr, labels_arr, n_clusters, mean_arr, frac_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

} // extern "C"
