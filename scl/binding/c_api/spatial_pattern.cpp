// =============================================================================
// FILE: scl/binding/c_api/spatial_pattern/spatial_pattern.cpp
// BRIEF: C API implementation for spatial pattern detection
// =============================================================================

#include "scl/binding/c_api/spatial_pattern.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/spatial_pattern.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_spatial_pattern_variability(
    scl_sparse_t expression,
    const scl_real_t* coordinates,
    scl_size_t n_dims,
    scl_real_t* variability_scores,
    scl_real_t* p_values,
    scl_size_t n_permutations,
    uint64_t seed
) {
    if (!expression || !coordinates || !variability_scores || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);

        if (!sparse->is_csr_format()) {
            return SCL_ERROR_INVALID_ARGUMENT;  // spatial_variability requires CSR format
        }

        sparse->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::spatial_pattern::spatial_variability(
                    m, reinterpret_cast<const scl::Real*>(coordinates), n_dims,
                    reinterpret_cast<scl::Real*>(variability_scores),
                    reinterpret_cast<scl::Real*>(p_values),
                    n_permutations, seed
                );
            }
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_pattern_gradient(
    const scl_real_t* expression,
    const scl_real_t* coordinates,
    scl_size_t n_cells,
    scl_size_t n_dims,
    scl_real_t* gradient_direction,
    scl_real_t* gradient_strength
) {
    if (!expression || !coordinates || !gradient_direction || !gradient_strength) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto strength = scl::Real(0);
        scl::kernel::spatial_pattern::spatial_gradient(
            reinterpret_cast<const scl::Real*>(expression),
            reinterpret_cast<const scl::Real*>(coordinates),
            n_cells, n_dims,
            reinterpret_cast<scl::Real*>(gradient_direction),
            strength
        );

        *gradient_strength = static_cast<scl_real_t>(strength);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_pattern_periodic(
    scl_sparse_t expression,
    const scl_real_t* coordinates,
    scl_size_t n_dims,
    scl_real_t* periodicity_scores,
    scl_real_t* dominant_wavelengths,
    scl_size_t n_wavelengths,
    const scl_real_t* test_wavelengths
) {
    if (!expression || !coordinates || !periodicity_scores || !dominant_wavelengths || !test_wavelengths) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);

        if (!sparse->is_csr_format()) {
            return SCL_ERROR_INVALID_ARGUMENT;  // periodic_pattern requires CSR format
        }

        sparse->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::spatial_pattern::periodic_pattern(
                    m, reinterpret_cast<const scl::Real*>(coordinates), n_dims,
                    reinterpret_cast<scl::Real*>(periodicity_scores),
                    reinterpret_cast<scl::Real*>(dominant_wavelengths),
                    n_wavelengths,
                    reinterpret_cast<const scl::Real*>(test_wavelengths)
                );
            }
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_pattern_boundary(
    scl_sparse_t expression,
    const scl_real_t* coordinates,
    scl_size_t n_dims,
    scl_real_t* boundary_scores,
    scl_size_t n_neighbors
) {
    if (!expression || !coordinates || !boundary_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        if (!sparse->is_csr_format()) {
            return SCL_ERROR_INVALID_ARGUMENT;  // boundary_detection requires CSR format
        }

        sparse->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(boundary_scores),
                                             static_cast<scl::Size>(n_cells));
                scl::kernel::spatial_pattern::boundary_detection(
                    m, reinterpret_cast<const scl::Real*>(coordinates), n_dims,
                    scores, n_neighbors
                );
            }
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_pattern_domain(
    scl_sparse_t expression,
    const scl_real_t* coordinates,
    scl_size_t n_dims,
    scl_index_t n_domains,
    scl_index_t* domain_labels,
    uint64_t seed
) {
    if (!expression || !coordinates || !domain_labels) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);
        scl::Index n_cells = sparse->rows();

        if (!sparse->is_csr_format()) {
            return SCL_ERROR_INVALID_ARGUMENT;  // spatial_domain requires CSR format
        }

        sparse->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::Array<scl::Index> labels(domain_labels, static_cast<scl::Size>(n_cells));
                scl::kernel::spatial_pattern::spatial_domain(
                    m, reinterpret_cast<const scl::Real*>(coordinates), n_dims,
                    n_domains, labels, seed
                );
            }
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_pattern_hotspot(
    const scl_real_t* values,
    const scl_real_t* coordinates,
    scl_size_t n_cells,
    scl_size_t n_dims,
    scl_real_t bandwidth,
    scl_real_t* gi_scores,
    scl_real_t* z_scores
) {
    if (!values || !coordinates || !gi_scores || !z_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::kernel::spatial_pattern::hotspot_analysis(
            reinterpret_cast<const scl::Real*>(values),
            reinterpret_cast<const scl::Real*>(coordinates),
            n_cells, n_dims, static_cast<scl::Real>(bandwidth),
            reinterpret_cast<scl::Real*>(gi_scores),
            reinterpret_cast<scl::Real*>(z_scores)
        );

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_pattern_autocorrelation(
    scl_sparse_t expression,
    const scl_real_t* coordinates,
    scl_size_t n_dims,
    scl_real_t* morans_i,
    scl_real_t* gearys_c
) {
    if (!expression || !coordinates || !morans_i || !gearys_c) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);

        if (!sparse->is_csr_format()) {
            return SCL_ERROR_INVALID_ARGUMENT;  // spatial_autocorrelation requires CSR format
        }

        sparse->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::spatial_pattern::spatial_autocorrelation(
                    m, reinterpret_cast<const scl::Real*>(coordinates), n_dims,
                    reinterpret_cast<scl::Real*>(morans_i),
                    reinterpret_cast<scl::Real*>(gearys_c)
                );
            }
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_pattern_smoothing(
    scl_sparse_t expression,
    const scl_real_t* coordinates,
    scl_size_t n_dims,
    scl_real_t bandwidth,
    scl_real_t* smoothed
) {
    if (!expression || !coordinates || !smoothed) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);

        if (!sparse->is_csr_format()) {
            return SCL_ERROR_INVALID_ARGUMENT;  // spatial_smoothing requires CSR format
        }

        sparse->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::spatial_pattern::spatial_smoothing(
                    m, reinterpret_cast<const scl::Real*>(coordinates), n_dims,
                    static_cast<scl::Real>(bandwidth),
                    reinterpret_cast<scl::Real*>(smoothed)
                );
            }
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_pattern_coexpression(
    scl_sparse_t expression,
    const scl_real_t* coordinates,
    scl_size_t n_dims,
    const scl_index_t* gene_pairs,
    scl_size_t n_pairs,
    scl_real_t* coexpression_scores
) {
    if (!expression || !coordinates || !gene_pairs || !coexpression_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(expression);

        if (!sparse->is_csr_format()) {
            return SCL_ERROR_INVALID_ARGUMENT;  // spatial_coexpression requires CSR format
        }

        sparse->visit([&](auto& m) {
            if constexpr (std::remove_reference_t<decltype(m)>::is_csr) {
                scl::kernel::spatial_pattern::spatial_coexpression(
                    m, reinterpret_cast<const scl::Real*>(coordinates), n_dims,
                    gene_pairs, n_pairs,
                    reinterpret_cast<scl::Real*>(coexpression_scores)
                );
            }
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_pattern_ripleys_k(
    const scl_real_t* coordinates,
    scl_size_t n_cells,
    scl_size_t n_dims,
    const scl_real_t* radii,
    scl_size_t n_radii,
    scl_real_t* k_values,
    scl_real_t study_area
) {
    if (!coordinates || !radii || !k_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::kernel::spatial_pattern::ripleys_k(
            reinterpret_cast<const scl::Real*>(coordinates),
            n_cells, n_dims,
            reinterpret_cast<const scl::Real*>(radii), n_radii,
            reinterpret_cast<scl::Real*>(k_values),
            static_cast<scl::Real>(study_area)
        );

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_spatial_pattern_entropy(
    const scl_index_t* labels,
    const scl_real_t* coordinates,
    scl_size_t n_cells,
    scl_size_t n_dims,
    scl_real_t bandwidth,
    scl_real_t* entropy_scores
) {
    if (!labels || !coordinates || !entropy_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> label_arr(labels, n_cells);
        scl::Array<scl::Real> scores(reinterpret_cast<scl::Real*>(entropy_scores), n_cells);

        scl::kernel::spatial_pattern::spatial_entropy(
            label_arr,
            reinterpret_cast<const scl::Real*>(coordinates),
            n_dims, static_cast<scl::Real>(bandwidth),
            scores
        );

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
