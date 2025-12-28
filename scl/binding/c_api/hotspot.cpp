#include "scl/binding/c_api/hotspot.h"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/hotspot.hpp"

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
typedef int8_t scl_spatial_pattern_t;
typedef int8_t scl_hotspot_type_t;
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

scl_error_t scl_row_standardize_weights(
    scl_sparse_matrix_t weights,
    scl_real_t* row_standardized
) {
    try {
        if (!weights || !row_standardized) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(weights);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::hotspot::row_standardize_weights(
            *sparse,
            scl::Array<scl::Real>(row_standardized, static_cast<scl::Size>(sparse->nnz()))
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

scl_error_t scl_local_morans_i(
    scl_sparse_matrix_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* local_i,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    scl_index_t n_permutations,
    uint64_t seed
) {
    try {
        if (!spatial_weights || !values || !local_i || !z_scores || !p_values) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(spatial_weights);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::hotspot::local_morans_i(
            *sparse,
            scl::Array<const scl::Real>(values, static_cast<scl::Size>(n)),
            static_cast<scl::Index>(n),
            scl::Array<scl::Real>(local_i, static_cast<scl::Size>(n)),
            scl::Array<scl::Real>(z_scores, static_cast<scl::Size>(n)),
            scl::Array<scl::Real>(p_values, static_cast<scl::Size>(n)),
            static_cast<scl::Index>(n_permutations),
            seed
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

scl_error_t scl_classify_lisa_patterns(
    scl_sparse_matrix_t spatial_weights,
    const scl_real_t* values,
    const scl_real_t* local_i,
    const scl_real_t* p_values,
    scl_index_t n,
    scl_real_t significance_level,
    scl_spatial_pattern_t* patterns
) {
    try {
        if (!spatial_weights || !values || !local_i || !p_values || !patterns) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(spatial_weights);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<scl::kernel::hotspot::SpatialPattern> cpp_patterns(
            reinterpret_cast<scl::kernel::hotspot::SpatialPattern*>(patterns),
            static_cast<scl::Size>(n)
        );

        scl::kernel::hotspot::classify_lisa_patterns(
            *sparse,
            scl::Array<const scl::Real>(values, static_cast<scl::Size>(n)),
            scl::Array<const scl::Real>(local_i, static_cast<scl::Size>(n)),
            scl::Array<const scl::Real>(p_values, static_cast<scl::Size>(n)),
            static_cast<scl::Index>(n),
            static_cast<scl::Real>(significance_level),
            cpp_patterns
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

scl_error_t scl_getis_ord_g_star(
    scl_sparse_matrix_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* g_star,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    int include_self
) {
    try {
        if (!spatial_weights || !values || !g_star || !z_scores || !p_values) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(spatial_weights);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::hotspot::getis_ord_g_star(
            *sparse,
            scl::Array<const scl::Real>(values, static_cast<scl::Size>(n)),
            static_cast<scl::Index>(n),
            scl::Array<scl::Real>(g_star, static_cast<scl::Size>(n)),
            scl::Array<scl::Real>(z_scores, static_cast<scl::Size>(n)),
            scl::Array<scl::Real>(p_values, static_cast<scl::Size>(n)),
            include_self != 0
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

scl_error_t scl_identify_hotspots(
    const scl_real_t* z_scores,
    scl_index_t n,
    scl_real_t significance_level,
    scl_hotspot_type_t* classification
) {
    try {
        if (!z_scores || !classification) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<scl::kernel::hotspot::HotspotType> cpp_classification(
            reinterpret_cast<scl::kernel::hotspot::HotspotType*>(classification),
            static_cast<scl::Size>(n)
        );

        scl::kernel::hotspot::identify_hotspots(
            scl::Array<const scl::Real>(z_scores, static_cast<scl::Size>(n)),
            static_cast<scl::Index>(n),
            static_cast<scl::Real>(significance_level),
            cpp_classification
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

scl_error_t scl_local_gearys_c(
    scl_sparse_matrix_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* local_c,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    scl_index_t n_permutations,
    uint64_t seed
) {
    try {
        if (!spatial_weights || !values || !local_c || !z_scores || !p_values) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(spatial_weights);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::hotspot::local_gearys_c(
            *sparse,
            scl::Array<const scl::Real>(values, static_cast<scl::Size>(n)),
            static_cast<scl::Index>(n),
            scl::Array<scl::Real>(local_c, static_cast<scl::Size>(n)),
            scl::Array<scl::Real>(z_scores, static_cast<scl::Size>(n)),
            scl::Array<scl::Real>(p_values, static_cast<scl::Size>(n)),
            static_cast<scl::Index>(n_permutations),
            seed
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

scl_error_t scl_global_morans_i(
    scl_sparse_matrix_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* moran_i,
    scl_real_t* z_score,
    scl_real_t* p_value,
    scl_index_t n_permutations,
    uint64_t seed
) {
    try {
        if (!spatial_weights || !values || !moran_i || !z_score || !p_value) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(spatial_weights);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Real cpp_moran_i, cpp_z_score, cpp_p_value;

        scl::kernel::hotspot::global_morans_i(
            *sparse,
            scl::Array<const scl::Real>(values, static_cast<scl::Size>(n)),
            static_cast<scl::Index>(n),
            cpp_moran_i,
            cpp_z_score,
            cpp_p_value,
            static_cast<scl::Index>(n_permutations),
            seed
        );

        *moran_i = static_cast<scl_real_t>(cpp_moran_i);
        *z_score = static_cast<scl_real_t>(cpp_z_score);
        *p_value = static_cast<scl_real_t>(cpp_p_value);

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_global_gearys_c(
    scl_sparse_matrix_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* geary_c,
    scl_real_t* z_score,
    scl_real_t* p_value
) {
    try {
        if (!spatial_weights || !values || !geary_c || !z_score || !p_value) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(spatial_weights);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Real cpp_geary_c, cpp_z_score, cpp_p_value;

        scl::kernel::hotspot::global_gearys_c(
            *sparse,
            scl::Array<const scl::Real>(values, static_cast<scl::Size>(n)),
            static_cast<scl::Index>(n),
            cpp_geary_c,
            cpp_z_score,
            cpp_p_value
        );

        *geary_c = static_cast<scl_real_t>(cpp_geary_c);
        *z_score = static_cast<scl_real_t>(cpp_z_score);
        *p_value = static_cast<scl_real_t>(cpp_p_value);

        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return exception_to_error(e);
    } catch (const std::exception& e) {
        return SCL_ERROR_UNKNOWN;
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_compute_spatial_lag(
    scl_sparse_matrix_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* spatial_lag
) {
    try {
        if (!spatial_weights || !values || !spatial_lag) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        auto* sparse = static_cast<scl::CSR*>(spatial_weights);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::hotspot::compute_spatial_lag(
            *sparse,
            scl::Array<const scl::Real>(values, static_cast<scl::Size>(n)),
            static_cast<scl::Index>(n),
            scl::Array<scl::Real>(spatial_lag, static_cast<scl::Size>(n))
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

