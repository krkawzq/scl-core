#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <random>
#include <cstring>

// =============================================================================
/// @file resample_mapped_impl.hpp
/// @brief Resampling for Mapped Sparse Matrices
///
/// Resampling is primarily RNG-bound, not memory-bound.
/// For Mapped matrices, we:
/// - Stream data to apply sampling
/// - Return OwnedSparse with sampled data
///
/// Operations:
/// - downsample_mapped: Downsample with target counts
/// - bootstrap_mapped: Bootstrap resampling
// =============================================================================

namespace scl::kernel::resample::mapped {

// =============================================================================
// Downsample Implementation
// =============================================================================

/// @brief Downsample mapped matrix to target total counts per row (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> downsample_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Real> target_counts,
    uint64_t seed
) {
    // Materialize first, then apply downsampling
    auto owned = matrix.materialize();

    const Index primary_dim = scl::primary_size(owned.matrix);

    SCL_CHECK_DIM(target_counts.len == static_cast<Size>(primary_dim),
                  "Downsample: Target counts size mismatch");

    // Apply downsampling in parallel
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        std::mt19937_64 rng(seed + p);

        Index start = owned.matrix.indptr[p];
        Index end = owned.matrix.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* vals = owned.data.ptr + start;

        // Compute current total
        Real current_total = 0.0;
        for (Index k = 0; k < len; ++k) {
            current_total += static_cast<Real>(vals[k]);
        }

        Real target = target_counts[p];
        if (target >= current_total || current_total <= 0) return;

        // Proportional scaling with noise
        Real scale = target / current_total;

        for (Index k = 0; k < len; ++k) {
            Real v = static_cast<Real>(vals[k]);
            Real expected = v * scale;

            // Binomial-like sampling
            Real sampled = std::floor(expected);
            Real frac = expected - sampled;

            std::uniform_real_distribution<Real> dist(0.0, 1.0);
            if (dist(rng) < frac) {
                sampled += 1.0;
            }

            vals[k] = static_cast<T>(std::max(0.0, sampled));
        }
    });

    return owned;
}

/// @brief Downsample mapped matrix (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> downsample_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Real> target_counts,
    uint64_t seed
) {
    // Materialize first, then apply downsampling
    auto owned = matrix.materialize();

    const Index primary_dim = scl::primary_size(owned.matrix);

    SCL_CHECK_DIM(target_counts.len == static_cast<Size>(primary_dim),
                  "Downsample: Target counts size mismatch");

    // Apply downsampling in parallel
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        std::mt19937_64 rng(seed + p);

        Index start = owned.matrix.indptr[p];
        Index end = owned.matrix.indptr[p + 1];
        Index len = end - start;

        if (len == 0) return;

        T* vals = owned.data.ptr + start;

        // Compute current total
        Real current_total = 0.0;
        for (Index k = 0; k < len; ++k) {
            current_total += static_cast<Real>(vals[k]);
        }

        Real target = target_counts[p];
        if (target >= current_total || current_total <= 0) return;

        // Proportional scaling with noise
        Real scale = target / current_total;

        for (Index k = 0; k < len; ++k) {
            Real v = static_cast<Real>(vals[k]);
            Real expected = v * scale;

            // Binomial-like sampling
            Real sampled = std::floor(expected);
            Real frac = expected - sampled;

            std::uniform_real_distribution<Real> dist(0.0, 1.0);
            if (dist(rng) < frac) {
                sampled += 1.0;
            }

            vals[k] = static_cast<T>(std::max(0.0, sampled));
        }
    });

    return owned;
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
scl::io::OwnedSparse<typename MatrixT::ValueType, IsCSR> downsample_mapped_dispatch(
    const MatrixT& matrix,
    Array<const Real> target_counts,
    uint64_t seed
) {
    return downsample_mapped(matrix, target_counts, seed);
}

} // namespace scl::kernel::resample::mapped

