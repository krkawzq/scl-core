#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <cstring>

// =============================================================================
// FILE: scl/kernel/kernel.hpp
// BRIEF: Sparse kernel methods including KDE, kernel functions, and kernel ops
// =============================================================================

namespace scl::kernel::kernel {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_BANDWIDTH = Real(1.0);
    constexpr Real MIN_BANDWIDTH = Real(1e-10);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Real LOG_MIN = Real(1e-300);
    constexpr Index DEFAULT_K_NEIGHBORS = 15;
}

// =============================================================================
// Kernel Types
// =============================================================================

enum class KernelType {
    Gaussian,
    Epanechnikov,
    Cosine,
    Linear,
    Polynomial,
    Laplacian,
    Cauchy,
    Sigmoid,
    Uniform,
    Triangular
};

// =============================================================================
// Kernel Function Evaluations
// =============================================================================

namespace detail {

// Gaussian kernel: exp(-||x-y||^2 / (2 * h^2))
SCL_FORCE_INLINE Real gaussian_kernel(Real dist_sq, Real bandwidth) {
    Real h2 = bandwidth * bandwidth;
    return std::exp(-dist_sq / (Real(2) * h2));
}

// Epanechnikov kernel: 3/4 * (1 - u^2) for |u| <= 1
SCL_FORCE_INLINE Real epanechnikov_kernel(Real dist, Real bandwidth) {
    Real u = dist / bandwidth;
    if (u > Real(1)) return Real(0);
    return Real(0.75) * (Real(1) - u * u);
}

// Cosine kernel: (pi/4) * cos(pi/2 * u) for |u| <= 1
SCL_FORCE_INLINE Real cosine_kernel(Real dist, Real bandwidth) {
    Real u = dist / bandwidth;
    if (u > Real(1)) return Real(0);
    constexpr Real pi = Real(3.14159265358979323846);
    return (pi / Real(4)) * std::cos(pi * u / Real(2));
}

// Laplacian kernel: exp(-|x-y| / h)
SCL_FORCE_INLINE Real laplacian_kernel(Real dist, Real bandwidth) {
    return std::exp(-dist / bandwidth);
}

// Cauchy kernel: 1 / (1 + (d/h)^2)
SCL_FORCE_INLINE Real cauchy_kernel(Real dist_sq, Real bandwidth) {
    Real h2 = bandwidth * bandwidth;
    return Real(1) / (Real(1) + dist_sq / h2);
}

// Uniform kernel: 0.5 if d < h, else 0
SCL_FORCE_INLINE Real uniform_kernel(Real dist, Real bandwidth) {
    return (dist < bandwidth) ? Real(0.5) : Real(0);
}

// Triangular kernel: (1 - |d|/h) if d < h
SCL_FORCE_INLINE Real triangular_kernel(Real dist, Real bandwidth) {
    if (dist >= bandwidth) return Real(0);
    return Real(1) - dist / bandwidth;
}

// Apply kernel based on type
SCL_FORCE_INLINE Real apply_kernel(
    Real dist_sq,
    Real bandwidth,
    KernelType kernel_type
) {
    Real dist = std::sqrt(dist_sq);

    switch (kernel_type) {
        case KernelType::Gaussian:
            return gaussian_kernel(dist_sq, bandwidth);
        case KernelType::Epanechnikov:
            return epanechnikov_kernel(dist, bandwidth);
        case KernelType::Cosine:
            return cosine_kernel(dist, bandwidth);
        case KernelType::Laplacian:
            return laplacian_kernel(dist, bandwidth);
        case KernelType::Cauchy:
            return cauchy_kernel(dist_sq, bandwidth);
        case KernelType::Uniform:
            return uniform_kernel(dist, bandwidth);
        case KernelType::Triangular:
            return triangular_kernel(dist, bandwidth);
        default:
            return gaussian_kernel(dist_sq, bandwidth);
    }
}

// Simple PRNG for sampling
struct FastRNG {
    uint64_t state;

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept : state(seed) {}

    SCL_FORCE_INLINE uint64_t next() noexcept {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next() >> 11) * Real(0x1.0p-53);
    }

    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        return static_cast<Size>(next() % static_cast<uint64_t>(n));
    }
};

} // namespace detail

// =============================================================================
// Kernel Density Estimation (KDE) from Distance Matrix
// =============================================================================

template <typename T, bool IsCSR>
void kde_from_distances(
    const Sparse<T, IsCSR>& distances,
    Array<Real> density,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();

    SCL_CHECK_DIM(density.len >= static_cast<Size>(n),
                  "KDE: density buffer too small");

    if (n == 0) return;

    bandwidth = scl::algo::max2(bandwidth, config::MIN_BANDWIDTH);

    for (Index i = 0; i < n; ++i) {
        auto indices = distances.primary_indices(i);
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        Real sum = Real(0);
        for (Index k = 0; k < len; ++k) {
            Real d = static_cast<Real>(values[k]);
            Real d_sq = d * d;
            sum += detail::apply_kernel(d_sq, bandwidth, kernel_type);
        }

        // Add self-contribution (distance = 0)
        sum += detail::apply_kernel(Real(0), bandwidth, kernel_type);

        // Normalize by effective number of neighbors + 1
        density[i] = sum / static_cast<Real>(len + 1);
    }
}

// =============================================================================
// Adaptive Bandwidth Estimation (Silverman's rule of thumb)
// =============================================================================

inline Real silverman_bandwidth(
    Array<const Real> data,
    Index n_features = 1
) {
    const Size n = data.len / static_cast<Size>(n_features);
    if (n <= 1) return config::DEFAULT_BANDWIDTH;

    // Compute standard deviation of first feature
    Real mean = Real(0);
    for (Size i = 0; i < n; ++i) {
        mean += data[i * n_features];
    }
    mean /= static_cast<Real>(n);

    Real var = Real(0);
    for (Size i = 0; i < n; ++i) {
        Real d = data[i * n_features] - mean;
        var += d * d;
    }
    var /= static_cast<Real>(n - 1);
    Real std_dev = std::sqrt(var);

    // Silverman's rule: h = 1.06 * std * n^(-1/5)
    Real h = Real(1.06) * std_dev * std::pow(static_cast<Real>(n), Real(-0.2));

    return scl::algo::max2(h, config::MIN_BANDWIDTH);
}

// =============================================================================
// Scott's Rule for Bandwidth
// =============================================================================

inline Real scott_bandwidth(
    Array<const Real> data,
    Index n_features = 1
) {
    const Size n = data.len / static_cast<Size>(n_features);
    if (n <= 1) return config::DEFAULT_BANDWIDTH;

    // Compute standard deviation
    Real mean = Real(0);
    for (Size i = 0; i < n; ++i) {
        mean += data[i * n_features];
    }
    mean /= static_cast<Real>(n);

    Real var = Real(0);
    for (Size i = 0; i < n; ++i) {
        Real d = data[i * n_features] - mean;
        var += d * d;
    }
    var /= static_cast<Real>(n - 1);
    Real std_dev = std::sqrt(var);

    // Scott's rule: h = std * n^(-1/(d+4)) where d = 1 for univariate
    Real h = std_dev * std::pow(static_cast<Real>(n), Real(-0.2));

    return scl::algo::max2(h, config::MIN_BANDWIDTH);
}

// =============================================================================
// Local Bandwidth Estimation (k-NN based)
// =============================================================================

template <typename T, bool IsCSR>
void local_bandwidth(
    const Sparse<T, IsCSR>& distances,
    Array<Real> bandwidths,
    Index k = 0
) {
    const Index n = distances.primary_dim();

    SCL_CHECK_DIM(bandwidths.len >= static_cast<Size>(n),
                  "LocalBandwidth: output buffer too small");

    for (Index i = 0; i < n; ++i) {
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        if (len == 0) {
            bandwidths[i] = config::DEFAULT_BANDWIDTH;
            continue;
        }

        Index effective_k = (k > 0 && k <= len) ? k : len;

        // Use distance to k-th neighbor as bandwidth
        Real max_dist = Real(0);
        for (Index j = 0; j < effective_k; ++j) {
            Real d = static_cast<Real>(values[j]);
            max_dist = scl::algo::max2(max_dist, d);
        }

        bandwidths[i] = scl::algo::max2(max_dist, config::MIN_BANDWIDTH);
    }
}

// =============================================================================
// KDE with Local Bandwidth (Adaptive KDE)
// =============================================================================

template <typename T, bool IsCSR>
void adaptive_kde(
    const Sparse<T, IsCSR>& distances,
    Array<Real> density,
    Array<const Real> bandwidths,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();

    SCL_CHECK_DIM(density.len >= static_cast<Size>(n),
                  "AdaptiveKDE: density buffer too small");
    SCL_CHECK_DIM(bandwidths.len >= static_cast<Size>(n),
                  "AdaptiveKDE: bandwidths buffer too small");

    if (n == 0) return;

    for (Index i = 0; i < n; ++i) {
        auto indices = distances.primary_indices(i);
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        Real h_i = scl::algo::max2(bandwidths[i], config::MIN_BANDWIDTH);
        Real sum = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real d = static_cast<Real>(values[k]);

            // Geometric mean of bandwidths for symmetric kernel
            Real h_j = scl::algo::max2(bandwidths[j], config::MIN_BANDWIDTH);
            Real h = std::sqrt(h_i * h_j);

            Real d_sq = d * d;
            sum += detail::apply_kernel(d_sq, h, kernel_type);
        }

        // Self contribution
        sum += detail::apply_kernel(Real(0), h_i, kernel_type);

        density[i] = sum / static_cast<Real>(len + 1);
    }
}

// =============================================================================
// Kernel Matrix Computation (Sparse Output)
// =============================================================================

template <typename T, bool IsCSR>
void compute_kernel_matrix(
    const Sparse<T, IsCSR>& distances,
    Real* kernel_values,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();

    bandwidth = scl::algo::max2(bandwidth, config::MIN_BANDWIDTH);

    Size val_idx = 0;
    for (Index i = 0; i < n; ++i) {
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        for (Index k = 0; k < len; ++k) {
            Real d = static_cast<Real>(values[k]);
            Real d_sq = d * d;
            kernel_values[val_idx++] = detail::apply_kernel(d_sq, bandwidth, kernel_type);
        }
    }
}

// =============================================================================
// Kernel Sum (Aggregated Kernel Values per Row)
// =============================================================================

template <typename T, bool IsCSR>
void kernel_row_sums(
    const Sparse<T, IsCSR>& distances,
    Array<Real> sums,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();

    SCL_CHECK_DIM(sums.len >= static_cast<Size>(n),
                  "KernelRowSums: output buffer too small");

    bandwidth = scl::algo::max2(bandwidth, config::MIN_BANDWIDTH);

    for (Index i = 0; i < n; ++i) {
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        Real sum = Real(0);
        for (Index k = 0; k < len; ++k) {
            Real d = static_cast<Real>(values[k]);
            Real d_sq = d * d;
            sum += detail::apply_kernel(d_sq, bandwidth, kernel_type);
        }

        sums[i] = sum;
    }
}

// =============================================================================
// Kernel Weighted Mean (Nadaraya-Watson Estimator)
// =============================================================================

template <typename T, bool IsCSR>
void nadaraya_watson(
    const Sparse<T, IsCSR>& distances,
    Array<const Real> y_values,
    Array<Real> predictions,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();

    SCL_CHECK_DIM(predictions.len >= static_cast<Size>(n),
                  "NW: predictions buffer too small");

    bandwidth = scl::algo::max2(bandwidth, config::MIN_BANDWIDTH);

    for (Index i = 0; i < n; ++i) {
        auto indices = distances.primary_indices(i);
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        Real weighted_sum = Real(0);
        Real weight_sum = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real d = static_cast<Real>(values[k]);
            Real d_sq = d * d;
            Real w = detail::apply_kernel(d_sq, bandwidth, kernel_type);

            weighted_sum += w * y_values[j];
            weight_sum += w;
        }

        // Add self contribution
        Real self_weight = detail::apply_kernel(Real(0), bandwidth, kernel_type);
        weighted_sum += self_weight * y_values[i];
        weight_sum += self_weight;

        predictions[i] = (weight_sum > Real(1e-15)) ? weighted_sum / weight_sum : y_values[i];
    }
}

// =============================================================================
// Kernel Smoothing on Graph
// =============================================================================

template <typename T, bool IsCSR>
void kernel_smooth_graph(
    const Sparse<T, IsCSR>& kernel_weights,
    Array<const Real> values,
    Array<Real> smoothed_values
) {
    const Index n = kernel_weights.primary_dim();

    SCL_CHECK_DIM(smoothed_values.len >= static_cast<Size>(n),
                  "KernelSmooth: output buffer too small");

    for (Index i = 0; i < n; ++i) {
        auto indices = kernel_weights.primary_indices(i);
        auto weights = kernel_weights.primary_values(i);
        const Index len = kernel_weights.primary_length(i);

        Real weighted_sum = Real(0);
        Real weight_sum = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real w = static_cast<Real>(weights[k]);
            weighted_sum += w * values[j];
            weight_sum += w;
        }

        // Self contribution with weight 1
        weighted_sum += values[i];
        weight_sum += Real(1);

        smoothed_values[i] = (weight_sum > Real(1e-15)) ? weighted_sum / weight_sum : values[i];
    }
}

// =============================================================================
// Local Linear Regression (Kernel Smoothing)
// =============================================================================

template <typename T, bool IsCSR>
void local_linear_regression(
    const Sparse<T, IsCSR>& distances,
    Array<const Real> X,
    Array<const Real> Y,
    Array<Real> predictions,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();

    SCL_CHECK_DIM(predictions.len >= static_cast<Size>(n),
                  "LLR: predictions buffer too small");

    bandwidth = scl::algo::max2(bandwidth, config::MIN_BANDWIDTH);

    for (Index i = 0; i < n; ++i) {
        auto indices = distances.primary_indices(i);
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        Real x_i = X[i];

        // Compute weighted least squares
        Real s0 = Real(0), s1 = Real(0), s2 = Real(0);
        Real t0 = Real(0), t1 = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real d = static_cast<Real>(values[k]);
            Real d_sq = d * d;
            Real w = detail::apply_kernel(d_sq, bandwidth, kernel_type);

            Real dx = X[j] - x_i;

            s0 += w;
            s1 += w * dx;
            s2 += w * dx * dx;
            t0 += w * Y[j];
            t1 += w * Y[j] * dx;
        }

        // Add self contribution
        Real w0 = detail::apply_kernel(Real(0), bandwidth, kernel_type);
        s0 += w0;
        t0 += w0 * Y[i];

        // Solve 2x2 system
        Real det = s0 * s2 - s1 * s1;
        if (std::abs(det) > Real(1e-15)) {
            Real a = (s2 * t0 - s1 * t1) / det;
            predictions[i] = a;
        } else {
            // Fall back to Nadaraya-Watson
            predictions[i] = (s0 > Real(1e-15)) ? t0 / s0 : Y[i];
        }
    }
}

// =============================================================================
// Kernel PCA Approximation (via Nystrom)
// =============================================================================

template <typename T, bool IsCSR>
void nystrom_approximation(
    const Sparse<T, IsCSR>& landmark_distances,
    Array<Real> embedding,
    Index n_components,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = landmark_distances.rows();

    SCL_CHECK_DIM(embedding.len >= static_cast<Size>(n) * static_cast<Size>(n_components),
                  "Nystrom: embedding buffer too small");

    if (n == 0 || n_components == 0) return;

    bandwidth = scl::algo::max2(bandwidth, config::MIN_BANDWIDTH);

    Size total = static_cast<Size>(n) * static_cast<Size>(n_components);
    Real* Q = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    Real* Q_new = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);

    // Initialize with deterministic values
    for (Size i = 0; i < total; ++i) {
        Q[i] = std::sin(static_cast<Real>(i) * Real(0.1)) + Real(0.5);
    }

    for (Index iter = 0; iter < 50; ++iter) {
        // Apply kernel matrix
        for (Index i = 0; i < n; ++i) {
            auto indices = landmark_distances.primary_indices(i);
            auto values = landmark_distances.primary_values(i);
            const Index len = landmark_distances.primary_length(i);

            for (Index c = 0; c < n_components; ++c) {
                Real sum = Real(0);

                for (Index k = 0; k < len; ++k) {
                    Index j = indices[k];
                    Real d = static_cast<Real>(values[k]);
                    Real kval = detail::apply_kernel(d * d, bandwidth, kernel_type);
                    sum += kval * Q[static_cast<Size>(j) * n_components + c];
                }

                Q_new[static_cast<Size>(i) * n_components + c] = sum;
            }
        }

        // Orthogonalize (Gram-Schmidt)
        for (Index c = 0; c < n_components; ++c) {
            for (Index p = 0; p < c; ++p) {
                Real dot = Real(0), norm_p = Real(0);
                for (Index i = 0; i < n; ++i) {
                    Size ic = static_cast<Size>(i) * n_components + c;
                    Size ip = static_cast<Size>(i) * n_components + p;
                    dot += Q_new[ic] * Q_new[ip];
                    norm_p += Q_new[ip] * Q_new[ip];
                }
                if (norm_p > Real(1e-15)) {
                    Real coeff = dot / norm_p;
                    for (Index i = 0; i < n; ++i) {
                        Size ic = static_cast<Size>(i) * n_components + c;
                        Size ip = static_cast<Size>(i) * n_components + p;
                        Q_new[ic] -= coeff * Q_new[ip];
                    }
                }
            }

            // Normalize
            Real norm = Real(0);
            for (Index i = 0; i < n; ++i) {
                Size ic = static_cast<Size>(i) * n_components + c;
                norm += Q_new[ic] * Q_new[ic];
            }
            if (norm > Real(1e-15)) {
                Real inv_norm = Real(1) / std::sqrt(norm);
                for (Index i = 0; i < n; ++i) {
                    Size ic = static_cast<Size>(i) * n_components + c;
                    Q_new[ic] *= inv_norm;
                }
            }
        }

        // Swap
        Real* tmp = Q;
        Q = Q_new;
        Q_new = tmp;
    }

    // Copy to output
    for (Size i = 0; i < total; ++i) {
        embedding[i] = Q[i];
    }

    scl::memory::aligned_free(Q_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(Q, SCL_ALIGNMENT);
}

// =============================================================================
// Mean Shift Step (Single Iteration)
// =============================================================================

template <typename T, bool IsCSR>
void mean_shift_step(
    const Sparse<T, IsCSR>& distances,
    Array<const Real> current_positions,
    Array<Real> new_positions,
    Index n_dims,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();
    const Size stride = static_cast<Size>(n_dims);

    SCL_CHECK_DIM(new_positions.len >= static_cast<Size>(n) * stride,
                  "MeanShift: output buffer too small");

    bandwidth = scl::algo::max2(bandwidth, config::MIN_BANDWIDTH);

    for (Index i = 0; i < n; ++i) {
        auto indices = distances.primary_indices(i);
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        Real* new_pos = new_positions.ptr + static_cast<Size>(i) * stride;
        const Real* curr_pos = current_positions.ptr + static_cast<Size>(i) * stride;

        // Initialize with zeros
        for (Index d = 0; d < n_dims; ++d) {
            new_pos[d] = Real(0);
        }

        Real weight_sum = Real(0);

        // Weighted sum of neighbor positions
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real dist = static_cast<Real>(values[k]);
            Real w = detail::apply_kernel(dist * dist, bandwidth, kernel_type);

            const Real* neighbor_pos = current_positions.ptr + static_cast<Size>(j) * stride;
            for (Index d = 0; d < n_dims; ++d) {
                new_pos[d] += w * neighbor_pos[d];
            }
            weight_sum += w;
        }

        // Add self contribution
        Real self_w = detail::apply_kernel(Real(0), bandwidth, kernel_type);
        for (Index d = 0; d < n_dims; ++d) {
            new_pos[d] += self_w * curr_pos[d];
        }
        weight_sum += self_w;

        // Normalize
        if (weight_sum > Real(1e-15)) {
            for (Index d = 0; d < n_dims; ++d) {
                new_pos[d] /= weight_sum;
            }
        } else {
            for (Index d = 0; d < n_dims; ++d) {
                new_pos[d] = curr_pos[d];
            }
        }
    }
}

// =============================================================================
// Kernel Entropy Estimation
// =============================================================================

template <typename T, bool IsCSR>
void kernel_entropy(
    const Sparse<T, IsCSR>& distances,
    Array<Real> entropy,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();

    SCL_CHECK_DIM(entropy.len >= static_cast<Size>(n),
                  "Entropy: output buffer too small");

    bandwidth = scl::algo::max2(bandwidth, config::MIN_BANDWIDTH);

    for (Index i = 0; i < n; ++i) {
        auto indices = distances.primary_indices(i);
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        // Compute kernel probabilities
        Real sum = Real(0);
        for (Index k = 0; k < len; ++k) {
            Real d = static_cast<Real>(values[k]);
            sum += detail::apply_kernel(d * d, bandwidth, kernel_type);
        }
        sum += detail::apply_kernel(Real(0), bandwidth, kernel_type);  // Self

        if (sum <= Real(1e-15)) {
            entropy[i] = Real(0);
            continue;
        }

        // Compute entropy: -sum(p * log(p))
        Real H = Real(0);
        for (Index k = 0; k < len; ++k) {
            Real d = static_cast<Real>(values[k]);
            Real kval = detail::apply_kernel(d * d, bandwidth, kernel_type);
            Real p = kval / sum;
            if (p > config::LOG_MIN) {
                H -= p * std::log(p);
            }
        }

        // Self contribution
        Real self_kval = detail::apply_kernel(Real(0), bandwidth, kernel_type);
        Real self_p = self_kval / sum;
        if (self_p > config::LOG_MIN) {
            H -= self_p * std::log(self_p);
        }

        entropy[i] = H;
    }
}

// =============================================================================
// Perplexity-Based Bandwidth Search (for t-SNE style applications)
// =============================================================================

template <typename T, bool IsCSR>
void find_bandwidth_for_perplexity(
    const Sparse<T, IsCSR>& distances,
    Array<Real> bandwidths,
    Real target_perplexity,
    Index max_iter = 50,
    Real tol = Real(1e-4)
) {
    const Index n = distances.primary_dim();

    SCL_CHECK_DIM(bandwidths.len >= static_cast<Size>(n),
                  "Perplexity: bandwidths buffer too small");

    Real target_entropy = std::log(target_perplexity);

    for (Index i = 0; i < n; ++i) {
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        if (len == 0) {
            bandwidths[i] = config::DEFAULT_BANDWIDTH;
            continue;
        }

        // Binary search for bandwidth
        Real lo = config::MIN_BANDWIDTH;
        Real hi = Real(1e10);
        Real beta = Real(1);  // 1 / (2 * sigma^2)

        for (Index iter = 0; iter < max_iter; ++iter) {
            // Compute probabilities and entropy for current beta
            Real sum_p = Real(0);
            for (Index k = 0; k < len; ++k) {
                Real d = static_cast<Real>(values[k]);
                sum_p += std::exp(-beta * d * d);
            }

            if (sum_p < Real(1e-300)) {
                beta /= Real(2);
                continue;
            }

            Real H = Real(0);
            for (Index k = 0; k < len; ++k) {
                Real d = static_cast<Real>(values[k]);
                Real p = std::exp(-beta * d * d) / sum_p;
                if (p > config::LOG_MIN) {
                    H -= p * std::log(p);
                }
            }

            Real diff = H - target_entropy;

            if (std::abs(diff) < tol) break;

            if (diff > Real(0)) {
                // Entropy too high, increase beta
                lo = beta;
                beta = (hi > Real(1e9)) ? beta * Real(2) : (lo + hi) / Real(2);
            } else {
                // Entropy too low, decrease beta
                hi = beta;
                beta = (lo + hi) / Real(2);
            }
        }

        // Convert beta to bandwidth
        bandwidths[i] = (beta > Real(1e-15)) ? Real(1) / std::sqrt(Real(2) * beta) : config::DEFAULT_BANDWIDTH;
    }
}

// =============================================================================
// Kernel Two-Sample Test Statistic (MMD Approximation)
// =============================================================================

template <typename T, bool IsCSR>
Real kernel_mmd_from_groups(
    const Sparse<T, IsCSR>& distances,
    Array<const bool> group_labels,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();

    bandwidth = scl::algo::max2(bandwidth, config::MIN_BANDWIDTH);

    // Count samples in each group
    Size n_A = 0, n_B = 0;
    for (Index i = 0; i < n; ++i) {
        if (group_labels[i]) ++n_A;
        else ++n_B;
    }

    if (n_A == 0 || n_B == 0) return Real(0);

    // Compute kernel sums
    Real k_AA = Real(0);
    Real k_BB = Real(0);
    Real k_AB = Real(0);

    for (Index i = 0; i < n; ++i) {
        auto indices = distances.primary_indices(i);
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        bool in_A = group_labels[i];

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real d = static_cast<Real>(values[k]);
            Real kval = detail::apply_kernel(d * d, bandwidth, kernel_type);

            bool j_in_A = group_labels[j];

            if (in_A && j_in_A) {
                k_AA += kval;
            } else if (!in_A && !j_in_A) {
                k_BB += kval;
            } else {
                k_AB += kval;
            }
        }

        // Self-kernel contribution
        Real self_k = detail::apply_kernel(Real(0), bandwidth, kernel_type);
        if (in_A) k_AA += self_k;
        else k_BB += self_k;
    }

    // MMD^2 = E[K(X,X')] + E[K(Y,Y')] - 2*E[K(X,Y)]
    Real mmd_sq = k_AA / (static_cast<Real>(n_A) * static_cast<Real>(n_A))
                + k_BB / (static_cast<Real>(n_B) * static_cast<Real>(n_B))
                - Real(2) * k_AB / (static_cast<Real>(n_A) * static_cast<Real>(n_B));

    return (mmd_sq > Real(0)) ? std::sqrt(mmd_sq) : Real(0);
}

// =============================================================================
// Evaluate Specific Kernel Function
// =============================================================================

inline Real evaluate_kernel(
    KernelType type,
    Real distance,
    Real bandwidth
) {
    return detail::apply_kernel(distance * distance, bandwidth, type);
}

// =============================================================================
// Kernel Normalization Constant
// =============================================================================

inline Real kernel_normalization(
    KernelType type,
    Index dimension
) {
    constexpr Real pi = Real(3.14159265358979323846);

    switch (type) {
        case KernelType::Gaussian: {
            // (2 * pi)^(-d/2)
            return std::pow(Real(2) * pi, -static_cast<Real>(dimension) / Real(2));
        }
        case KernelType::Epanechnikov: {
            // 3/(4) for d=1, more complex for higher d
            if (dimension == 1) return Real(0.75);
            return Real(0.75);  // Simplified
        }
        case KernelType::Uniform: {
            return Real(0.5);
        }
        case KernelType::Triangular: {
            return Real(1);
        }
        default:
            return Real(1);
    }
}

// =============================================================================
// RBF Kernel on Sparse Distances
// =============================================================================

template <typename T, bool IsCSR>
void rbf_sparse(
    const Sparse<T, IsCSR>& distances,
    Real bandwidth,
    Real* kernel_values
) {
    compute_kernel_matrix(distances, kernel_values, bandwidth, KernelType::Gaussian);
}

// =============================================================================
// Adaptive RBF Kernel
// =============================================================================

template <typename T, bool IsCSR>
void adaptive_rbf(
    const Sparse<T, IsCSR>& distances,
    Array<const Real> bandwidths,
    Real* kernel_values
) {
    const Index n = distances.primary_dim();

    Size val_idx = 0;
    for (Index i = 0; i < n; ++i) {
        auto indices = distances.primary_indices(i);
        auto values = distances.primary_values(i);
        const Index len = distances.primary_length(i);

        Real h_i = scl::algo::max2(bandwidths[i], config::MIN_BANDWIDTH);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real h_j = scl::algo::max2(bandwidths[j], config::MIN_BANDWIDTH);
            Real h = std::sqrt(h_i * h_j);

            Real d = static_cast<Real>(values[k]);
            kernel_values[val_idx++] = detail::gaussian_kernel(d * d, h);
        }
    }
}

} // namespace scl::kernel::kernel
