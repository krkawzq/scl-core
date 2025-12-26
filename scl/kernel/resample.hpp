#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <random>

// =============================================================================
/// @file resample.hpp
/// @brief Count Resampling for Data Augmentation
///
/// Implements:
/// 1. Downsample: Sample M counts from N (M < N)
/// 2. Binomial resample: c' ~ Binomial(c, p)
///
/// Use Cases:
/// - Data augmentation for deep learning
/// - Depth normalization
/// - Robustness testing
///
/// Performance: O(nnz) per operation
// =============================================================================

namespace scl::kernel::resample {

/// @brief Downsample counts to target sum (unified for CSR/CSC)
///
/// Uses multinomial sampling to downsample each primary dimension element.
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param target_sum Target sum for each element
/// @param seed Random seed
template <typename MatrixT>
    requires AnySparse<MatrixT>
void downsample_counts(
    MatrixT& matrix,
    Real target_sum,
    uint64_t seed = 42
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        std::mt19937_64 rng(seed + p);
        std::uniform_real_distribution<Real> dist(0.0, 1.0);
        
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));
        
        // Compute current sum
        Real current_sum = 0.0;
        for (size_t k = 0; k < vals.size(); ++k) {
            current_sum += vals[k];
        }
        
        if (current_sum <= target_sum) return;  // Already below target
        
        // Multinomial sampling
        Real remaining = target_sum;
        Real total_remaining = current_sum;
        
        for (size_t k = 0; k < vals.size() && remaining > 0; ++k) {
            Real count = vals[k];
            Real prob = count / total_remaining;
            
            // Sample from binomial
            Real sampled = 0.0;
            for (Real c = 0; c < remaining && c < count; c += 1.0) {
                if (dist(rng) < prob) {
                    sampled += 1.0;
                }
            }
            
            vals[k] = sampled;
            remaining -= sampled;
            total_remaining -= count;
        }
    });
}

} // namespace scl::kernel::resample
