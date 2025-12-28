#include "scl/binding/c_api/impute.h"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/impute.hpp"

#include <cstring>
#include <exception>
#include <cstdint>

extern "C" {

// Type definitions (if not in header)
#ifndef SCL_C_API_TYPES_DEFINED
typedef scl::Real scl_real_t;
typedef scl::Index scl_index_t;
typedef scl::Size scl_size_t;
typedef void* scl_sparse_matrix_t;
typedef int32_t scl_error_t;
typedef int32_t scl_impute_mode_t;
#define SCL_ERROR_OK 0
#define SCL_ERROR_UNKNOWN 1
#define SCL_ERROR_INTERNAL_ERROR 2
#define SCL_ERROR_INVALID_ARGUMENT 10
#define SCL_ERROR_DIMENSION_MISMATCH 11
#endif

static scl_error_t exception_to_error(const std::exception& e) {
    if (auto* scl_err = dynamic_cast<const scl::Exception*>(&e)) {
        return static_cast<scl_error_t>(scl_err->code());
    }
    return SCL_ERROR_UNKNOWN;
}

scl_error_t scl_knn_impute_dense(
    scl_sparse_matrix_t X,
    const scl_index_t* knn_indices,
    const scl_real_t* knn_distances,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t k_neighbors,
    scl_real_t* X_imputed,
    scl_real_t bandwidth,
    scl_real_t threshold
) {
    try {
        if (!X || !knn_indices || !knn_distances || !X_imputed) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(X);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::impute::knn_impute_dense(
            *sparse,
            knn_indices,
            knn_distances,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Index>(k_neighbors),
            X_imputed,
            static_cast<scl::Real>(bandwidth),
            static_cast<scl::Real>(threshold)
        );

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_knn_impute_weighted_dense(
    scl_sparse_matrix_t X,
    const scl_index_t* knn_indices,
    const scl_real_t* knn_weights,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t k_neighbors,
    scl_real_t* X_imputed,
    scl_real_t threshold
) {
    try {
        if (!X || !knn_indices || !knn_weights || !X_imputed) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(X);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::impute::knn_impute_weighted_dense(
            *sparse,
            knn_indices,
            knn_weights,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Index>(k_neighbors),
            X_imputed,
            static_cast<scl::Real>(threshold)
        );

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_diffusion_impute_sparse_transition(
    scl_sparse_matrix_t X,
    scl_sparse_matrix_t transition_matrix,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_steps,
    scl_real_t* X_imputed
) {
    try {
        if (!X || !transition_matrix || !X_imputed) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse_x = static_cast<scl::CSR*>(X);
        auto* sparse_t = static_cast<scl::CSR*>(transition_matrix);
        if (!sparse_x->valid() || !sparse_t->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::impute::diffusion_impute_sparse_transition(
            *sparse_x,
            *sparse_t,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Index>(n_steps),
            X_imputed
        );

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_magic_impute(
    scl_sparse_matrix_t X,
    scl_sparse_matrix_t affinity_matrix,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t t_diffusion,
    scl_real_t* X_imputed
) {
    try {
        if (!X || !affinity_matrix || !X_imputed) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse_x = static_cast<scl::CSR*>(X);
        auto* sparse_a = static_cast<scl::CSR*>(affinity_matrix);
        if (!sparse_x->valid() || !sparse_a->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::impute::magic_impute(
            *sparse_x,
            *sparse_a,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Index>(t_diffusion),
            X_imputed
        );

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_alra_impute(
    const scl_real_t* X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t rank,
    scl_real_t* X_imputed
) {
    try {
        if (!X || !X_imputed) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::impute::alra_impute(
            X,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Index>(rank),
            X_imputed
        );

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_impute_selected_genes(
    scl_sparse_matrix_t X,
    const scl_index_t* knn_indices,
    const scl_real_t* knn_distances,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t k_neighbors,
    const scl_index_t* genes_to_impute,
    scl_index_t n_impute_genes,
    scl_real_t* X_imputed,
    scl_real_t bandwidth
) {
    try {
        if (!X || !knn_indices || !knn_distances || !genes_to_impute || !X_imputed) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(X);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::impute::impute_selected_genes(
            *sparse,
            knn_indices,
            knn_distances,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Index>(k_neighbors),
            scl::Array<const scl::Index>(genes_to_impute, static_cast<scl::Size>(n_impute_genes)),
            X_imputed,
            static_cast<scl::Real>(bandwidth)
        );

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_detect_dropouts(
    scl_sparse_matrix_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t mean_threshold,
    scl_real_t* dropout_probability
) {
    try {
        if (!X || !dropout_probability) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(X);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::impute::detect_dropouts(
            *sparse,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Real>(mean_threshold),
            dropout_probability
        );

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_imputation_quality(
    const scl_real_t* X_original,
    const scl_real_t* X_imputed,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* quality
) {
    try {
        if (!X_original || !X_imputed || !quality) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        *quality = static_cast<scl_real_t>(
            scl::kernel::impute::imputation_quality(
                X_original,
                X_imputed,
                static_cast<scl::Index>(n_cells),
                static_cast<scl::Index>(n_genes)
            )
        );

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_smooth_expression(
    scl_sparse_matrix_t X,
    const scl_index_t* knn_indices,
    const scl_real_t* knn_weights,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t k_neighbors,
    scl_real_t alpha,
    scl_real_t* X_smoothed
) {
    try {
        if (!X || !knn_indices || !knn_weights || !X_smoothed) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(X);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::impute::smooth_expression(
            *sparse,
            knn_indices,
            knn_weights,
            static_cast<scl::Index>(n_cells),
            static_cast<scl::Index>(n_genes),
            static_cast<scl::Index>(k_neighbors),
            static_cast<scl::Real>(alpha),
            X_smoothed
        );

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

} // extern "C"

