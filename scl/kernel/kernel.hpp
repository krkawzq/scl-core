#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"

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
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real LOG_MIN = Real(1e-300);
    constexpr Index DEFAULT_K_NEIGHBORS = 15;
    constexpr Index NYSTROM_MAX_ITER = 50;
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
SCL_FORCE_INLINE Real gaussian_kernel(Real dist_sq, Real inv_2h2) {
    return std::exp(-dist_sq * inv_2h2);
}

// Precompute 1/(2*h^2) for Gaussian kernel
SCL_FORCE_INLINE Real gaussian_precompute(Real bandwidth) {
    return Real(1) / (Real(2) * bandwidth * bandwidth);
}

// Epanechnikov kernel: 3/4 * (1 - u^2) for |u| <= 1
SCL_FORCE_INLINE Real epanechnikov_kernel(Real dist_sq, Real inv_h2) {
    Real u2 = dist_sq * inv_h2;
    if (SCL_UNLIKELY(u2 > Real(1))) return Real(0);
    return Real(0.75) * (Real(1) - u2);
}

// Cosine kernel: (pi/4) * cos(pi/2 * u) for |u| <= 1
SCL_FORCE_INLINE Real cosine_kernel(Real dist, Real inv_h) {
    Real u = dist * inv_h;
    if (SCL_UNLIKELY(u > Real(1))) return Real(0);
    constexpr Real pi = Real(3.14159265358979323846);
    return (pi / Real(4)) * std::cos(pi * u / Real(2));
}

// Laplacian kernel: exp(-|x-y| / h)
SCL_FORCE_INLINE Real laplacian_kernel(Real dist, Real inv_h) {
    return std::exp(-dist * inv_h);
}

// Cauchy kernel: 1 / (1 + (d/h)^2)
SCL_FORCE_INLINE Real cauchy_kernel(Real dist_sq, Real inv_h2) {
    return Real(1) / (Real(1) + dist_sq * inv_h2);
}

// Uniform kernel: 0.5 if d < h, else 0
SCL_FORCE_INLINE Real uniform_kernel(Real dist_sq, Real h2) {
    return (dist_sq < h2) ? Real(0.5) : Real(0);
}

// Triangular kernel: (1 - |d|/h) if d < h
SCL_FORCE_INLINE Real triangular_kernel(Real dist, Real inv_h) {
    Real u = dist * inv_h;
    if (SCL_UNLIKELY(u >= Real(1))) return Real(0);
    return Real(1) - u;
}

// Kernel parameters structure for precomputation
struct KernelParams {
    Real inv_h;      // 1/h
    Real inv_h2;     // 1/h^2
    Real inv_2h2;    // 1/(2h^2)
    Real h2;         // h^2
    KernelType type;

    SCL_FORCE_INLINE explicit KernelParams(Real h, KernelType t) noexcept 
        : inv_h(Real(1) / scl::algo::max2(h, config::MIN_BANDWIDTH)),
          inv_h2(inv_h * inv_h),
          inv_2h2(Real(0.5) * inv_h2),
          h2(scl::algo::max2(h, config::MIN_BANDWIDTH) * scl::algo::max2(h, config::MIN_BANDWIDTH)),
          type(t) {
    }
};

// Apply kernel based on type (optimized with precomputed params)
SCL_FORCE_INLINE Real apply_kernel(Real dist_sq, const KernelParams& params) {
    switch (params.type) {
        case KernelType::Gaussian:
            return gaussian_kernel(dist_sq, params.inv_2h2);
        case KernelType::Epanechnikov:
            return epanechnikov_kernel(dist_sq, params.inv_h2);
        case KernelType::Cosine:
            return cosine_kernel(std::sqrt(dist_sq), params.inv_h);
        case KernelType::Laplacian:
            return laplacian_kernel(std::sqrt(dist_sq), params.inv_h);
        case KernelType::Cauchy:
            return cauchy_kernel(dist_sq, params.inv_h2);
        case KernelType::Uniform:
            return uniform_kernel(dist_sq, params.h2);
        case KernelType::Triangular:
            return triangular_kernel(std::sqrt(dist_sq), params.inv_h);
        default:
            return gaussian_kernel(dist_sq, params.inv_2h2);
    }
}

// Self-kernel value (distance = 0)
SCL_FORCE_INLINE Real self_kernel(const KernelParams& params) {
    switch (params.type) {
        case KernelType::Epanechnikov:
            return Real(0.75);
        case KernelType::Cosine:
            return Real(3.14159265358979323846) / Real(4);
        case KernelType::Uniform:
            return Real(0.5);
        case KernelType::Gaussian:
        case KernelType::Laplacian:
        case KernelType::Cauchy:
        case KernelType::Triangular:
        default:
            return Real(1);
    }
}


// SIMD Gaussian kernel batch evaluation
SCL_FORCE_INLINE void gaussian_kernel_batch(
    const Real* SCL_RESTRICT dist_sq,
    Real* SCL_RESTRICT result,
    Size n,
    Real inv_2h2
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);
    auto v_inv_2h2 = s::Set(d, -inv_2h2);
    Size k = 0;

    for (; k + lanes <= n; k += lanes) {
        auto v_dsq = s::Load(d, dist_sq + k);
        auto v_exp_arg = s::Mul(v_dsq, v_inv_2h2);
        auto v_result = s::Exp(d, v_exp_arg);
        s::Store(v_result, d, result + k);
    }

    for (; k < n; ++k) {
        result[k] = std::exp(-dist_sq[k] * inv_2h2);
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

// SIMD-optimized mean computation
SCL_FORCE_INLINE Real compute_mean(const Real* SCL_RESTRICT data, Size n, Size stride = 1) {
    if (SCL_UNLIKELY(n == 0)) return Real(0);

    if (stride == 1) {
        namespace s = scl::simd;
        using SimdTag = s::SimdTagFor<Real>;
        const SimdTag d;
        const size_t lanes = s::Lanes(d);
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        Size i = 0;

        for (; i + 2 * lanes <= n; i += 2 * lanes) {
            v_sum0 = s::Add(v_sum0, s::Load(d, data + i));
            v_sum1 = s::Add(v_sum1, s::Load(d, data + i + lanes));
        }

        auto v_sum = s::Add(v_sum0, v_sum1);
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));

        for (; i < n; ++i) {
            sum += data[i];
        }

        return sum / static_cast<Real>(n);
    } else {
        Real sum = Real(0);
        for (Size i = 0; i < n; ++i) {
            sum += data[i * stride];
        }
        return sum / static_cast<Real>(n);
    }
}

// SIMD-optimized variance computation
SCL_FORCE_INLINE Real compute_variance(const Real* SCL_RESTRICT data, Size n, Real mean, Size stride = 1) {
    if (SCL_UNLIKELY(n <= 1)) return Real(0);

    if (stride == 1) {
        namespace s = scl::simd;
        using SimdTag = s::SimdTagFor<Real>;
        const SimdTag d;
        const size_t lanes = s::Lanes(d);
        auto v_mean = s::Set(d, mean);
        auto v_var0 = s::Zero(d);
        auto v_var1 = s::Zero(d);
        Size i = 0;

        for (; i + 2 * lanes <= n; i += 2 * lanes) {
            auto v0 = s::Sub(s::Load(d, data + i), v_mean);
            auto v1 = s::Sub(s::Load(d, data + i + lanes), v_mean);
            v_var0 = s::MulAdd(v0, v0, v_var0);
            v_var1 = s::MulAdd(v1, v1, v_var1);
        }

        auto v_var = s::Add(v_var0, v_var1);
        Real var = s::GetLane(s::SumOfLanes(d, v_var));

        for (; i < n; ++i) {
            Real diff = data[i] - mean;
            var += diff * diff;
        }

        return var / static_cast<Real>(n - 1);
    } else {
        Real var = Real(0);
        for (Size i = 0; i < n; ++i) {
            Real diff = data[i * stride] - mean;
            var += diff * diff;
        }
        return var / static_cast<Real>(n - 1);
    }
}

} // namespace detail

// =============================================================================
// Kernel Density Estimation (KDE) from Distance Matrix - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void kde_from_distances(
    const Sparse<T, IsCSR>& distances,
    Array<Real> density,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(density.len >= N, "KDE: density buffer too small");

    if (SCL_UNLIKELY(n == 0)) return;

    detail::KernelParams params(bandwidth, kernel_type);
    const Real self_k = detail::self_kernel(params);
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);

        if (SCL_UNLIKELY(len == 0)) {
            density[i] = self_k;
            return;
        }

        Real sum = Real(0);

        // Unrolled accumulation
        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            Real d0 = static_cast<Real>(values[k + 0]);
            Real d1 = static_cast<Real>(values[k + 1]);
            Real d2 = static_cast<Real>(values[k + 2]);
            Real d3 = static_cast<Real>(values[k + 3]);
            sum += detail::apply_kernel(d0 * d0, params);
            sum += detail::apply_kernel(d1 * d1, params);
            sum += detail::apply_kernel(d2 * d2, params);
            sum += detail::apply_kernel(d3 * d3, params);
        }

        for (; k < len; ++k) {
            Real d = static_cast<Real>(values[k]);
            sum += detail::apply_kernel(d * d, params);
        }

        // Add self-contribution
        sum += self_k;
        density[i] = sum / static_cast<Real>(len + 1);
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// Adaptive Bandwidth Estimation (Silverman's rule of thumb) - Optimized
// =============================================================================

inline Real silverman_bandwidth(
    Array<const Real> data,
    Index n_features = 1
) {
    const Size n = data.len / static_cast<Size>(n_features);
    if (SCL_UNLIKELY(n <= 1)) return config::DEFAULT_BANDWIDTH;

    const Size stride = static_cast<Size>(n_features);
    Real mean = detail::compute_mean(data.ptr, n, stride);
    Real var = detail::compute_variance(data.ptr, n, mean, stride);
    Real std_dev = std::sqrt(var);

    // Silverman's rule: h = 1.06 * std * n^(-1/5)
    Real h = Real(1.06) * std_dev * std::pow(static_cast<Real>(n), Real(-0.2));
    return scl::algo::max2(h, config::MIN_BANDWIDTH);
}

// =============================================================================
// Scott's Rule for Bandwidth - Optimized
// =============================================================================

inline Real scott_bandwidth(
    Array<const Real> data,
    Index n_features = 1
) {
    const Size n = data.len / static_cast<Size>(n_features);
    if (SCL_UNLIKELY(n <= 1)) return config::DEFAULT_BANDWIDTH;

    const Size stride = static_cast<Size>(n_features);
    Real mean = detail::compute_mean(data.ptr, n, stride);
    Real var = detail::compute_variance(data.ptr, n, mean, stride);
    Real std_dev = std::sqrt(var);

    // Scott's rule: h = std * n^(-1/(d+4))
    Real h = std_dev * std::pow(static_cast<Real>(n), Real(-0.2));
    return scl::algo::max2(h, config::MIN_BANDWIDTH);
}

// =============================================================================
// Local Bandwidth Estimation (k-NN based) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void local_bandwidth(
    const Sparse<T, IsCSR>& distances,
    Array<Real> bandwidths,
    Index k = 0
) {
    const Index n = distances.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(bandwidths.len >= N, "LocalBandwidth: output buffer too small");

    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);

        if (SCL_UNLIKELY(len == 0)) {
            bandwidths[i] = config::DEFAULT_BANDWIDTH;
            return;
        }

        Index effective_k = (k > 0 && k <= len) ? k : len;

        // Find max distance to k-th neighbor
        Real max_dist = Real(0);
        Index j = 0;

        // Unrolled max search
        for (; j + 4 <= effective_k; j += 4) {
            Real d0 = static_cast<Real>(values[j + 0]);
            Real d1 = static_cast<Real>(values[j + 1]);
            Real d2 = static_cast<Real>(values[j + 2]);
            Real d3 = static_cast<Real>(values[j + 3]);
            max_dist = scl::algo::max2(max_dist, scl::algo::max2(d0, d1));
            max_dist = scl::algo::max2(max_dist, scl::algo::max2(d2, d3));
        }

        for (; j < effective_k; ++j) {
            max_dist = scl::algo::max2(max_dist, static_cast<Real>(values[j]));
        }

        bandwidths[i] = scl::algo::max2(max_dist, config::MIN_BANDWIDTH);
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// KDE with Local Bandwidth (Adaptive KDE) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void adaptive_kde(
    const Sparse<T, IsCSR>& distances,
    Array<Real> density,
    Array<const Real> bandwidths,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(density.len >= N, "AdaptiveKDE: density buffer too small");
    SCL_CHECK_DIM(bandwidths.len >= N, "AdaptiveKDE: bandwidths buffer too small");

    if (SCL_UNLIKELY(n == 0)) return;

    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        auto indices = distances.primary_indices_unsafe(i);
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);

        Real h_i = scl::algo::max2(bandwidths[i], config::MIN_BANDWIDTH);

        if (SCL_UNLIKELY(len == 0)) {
            detail::KernelParams params(h_i, kernel_type);
            density[i] = detail::self_kernel(params);
            return;
        }

        Real sum = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real d = static_cast<Real>(values[k]);

            // Geometric mean of bandwidths
            Real h_j = scl::algo::max2(bandwidths[j], config::MIN_BANDWIDTH);
            Real h = std::sqrt(h_i * h_j);

            detail::KernelParams params(h, kernel_type);
            sum += detail::apply_kernel(d * d, params);
        }

        // Self contribution
        detail::KernelParams self_params(h_i, kernel_type);
        sum += detail::self_kernel(self_params);

        density[i] = sum / static_cast<Real>(len + 1);
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// Kernel Matrix Computation (Sparse Output) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void compute_kernel_matrix(
    const Sparse<T, IsCSR>& distances,
    Real* SCL_RESTRICT kernel_values,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();
    detail::KernelParams params(bandwidth, kernel_type);

    // For Gaussian kernel, use SIMD batch evaluation
    if (kernel_type == KernelType::Gaussian) {
        const Real inv_2h2 = params.inv_2h2;
        Size val_idx = 0;

        for (Index i = 0; i < n; ++i) {
            auto values = distances.primary_values_unsafe(i);
            const Index len = distances.primary_length_unsafe(i);

            // Convert distances to squared distances first
            for (Index k = 0; k < len; ++k) {
                Real d = static_cast<Real>(values[k]);
                kernel_values[val_idx + k] = d * d;
            }

            // Batch Gaussian evaluation
            detail::gaussian_kernel_batch(
                kernel_values + val_idx,
                kernel_values + val_idx,
                static_cast<Size>(len),
                inv_2h2
            );

            val_idx += static_cast<Size>(len);
        }
    } else {
        Size val_idx = 0;

        for (Index i = 0; i < n; ++i) {
            auto values = distances.primary_values_unsafe(i);
            const Index len = distances.primary_length_unsafe(i);

            Index k = 0;
            for (; k + 4 <= len; k += 4) {
                Real d0 = static_cast<Real>(values[k + 0]);
                Real d1 = static_cast<Real>(values[k + 1]);
                Real d2 = static_cast<Real>(values[k + 2]);
                Real d3 = static_cast<Real>(values[k + 3]);
                kernel_values[val_idx++] = detail::apply_kernel(d0 * d0, params);
                kernel_values[val_idx++] = detail::apply_kernel(d1 * d1, params);
                kernel_values[val_idx++] = detail::apply_kernel(d2 * d2, params);
                kernel_values[val_idx++] = detail::apply_kernel(d3 * d3, params);
            }

            for (; k < len; ++k) {
                Real d = static_cast<Real>(values[k]);
                kernel_values[val_idx++] = detail::apply_kernel(d * d, params);
            }
        }
    }
}

// =============================================================================
// Kernel Sum (Aggregated Kernel Values per Row) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void kernel_row_sums(
    const Sparse<T, IsCSR>& distances,
    Array<Real> sums,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(sums.len >= N, "KernelRowSums: output buffer too small");

    detail::KernelParams params(bandwidth, kernel_type);
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);

        Real sum = Real(0);
        Index k = 0;

        for (; k + 4 <= len; k += 4) {
            Real d0 = static_cast<Real>(values[k + 0]);
            Real d1 = static_cast<Real>(values[k + 1]);
            Real d2 = static_cast<Real>(values[k + 2]);
            Real d3 = static_cast<Real>(values[k + 3]);
            sum += detail::apply_kernel(d0 * d0, params);
            sum += detail::apply_kernel(d1 * d1, params);
            sum += detail::apply_kernel(d2 * d2, params);
            sum += detail::apply_kernel(d3 * d3, params);
        }

        for (; k < len; ++k) {
            Real d = static_cast<Real>(values[k]);
            sum += detail::apply_kernel(d * d, params);
        }

        sums[i] = sum;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// Kernel Weighted Mean (Nadaraya-Watson Estimator) - Optimized
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
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(predictions.len >= N, "NW: predictions buffer too small");

    detail::KernelParams params(bandwidth, kernel_type);
    const Real self_k = detail::self_kernel(params);
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        auto indices = distances.primary_indices_unsafe(i);
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);

        Real weighted_sum = Real(0);
        Real weight_sum = Real(0);

        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            for (int off = 0; off < 4; ++off) {
                Index j = indices[k + off];
                Real d = static_cast<Real>(values[k + off]);
                Real w = detail::apply_kernel(d * d, params);
                weighted_sum += w * y_values[j];
                weight_sum += w;
            }
        }

        for (; k < len; ++k) {
            Index j = indices[k];
            Real d = static_cast<Real>(values[k]);
            Real w = detail::apply_kernel(d * d, params);
            weighted_sum += w * y_values[j];
            weight_sum += w;
        }

        // Self contribution
        weighted_sum += self_k * y_values[i];
        weight_sum += self_k;

        predictions[i] = (SCL_LIKELY(weight_sum > Real(1e-15))) ?
            weighted_sum / weight_sum : y_values[i];
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// Kernel Smoothing on Graph - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void kernel_smooth_graph(
    const Sparse<T, IsCSR>& kernel_weights,
    Array<const Real> values,
    Array<Real> smoothed_values
) {
    const Index n = kernel_weights.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(smoothed_values.len >= N, "KernelSmooth: output buffer too small");

    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        auto indices = kernel_weights.primary_indices_unsafe(i);
        auto weights = kernel_weights.primary_values_unsafe(i);
        const Index len = kernel_weights.primary_length_unsafe(i);

        Real weighted_sum = Real(0);
        Real weight_sum = Real(0);

        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            for (int off = 0; off < 4; ++off) {
                Index j = indices[k + off];
                Real w = static_cast<Real>(weights[k + off]);
                weighted_sum += w * values[j];
                weight_sum += w;
            }
        }

        for (; k < len; ++k) {
            Index j = indices[k];
            Real w = static_cast<Real>(weights[k]);
            weighted_sum += w * values[j];
            weight_sum += w;
        }

        // Self contribution
        weighted_sum += values[i];
        weight_sum += Real(1);

        smoothed_values[i] = (SCL_LIKELY(weight_sum > Real(1e-15))) ?
            weighted_sum / weight_sum : values[i];
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// Local Linear Regression (Kernel Smoothing) - Optimized
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
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(predictions.len >= N, "LLR: predictions buffer too small");

    detail::KernelParams params(bandwidth, kernel_type);
    const Real self_k = detail::self_kernel(params);
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        auto indices = distances.primary_indices_unsafe(i);
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);

        Real x_i = X[i];

        // Compute weighted least squares moments
        Real s0 = Real(0), s1 = Real(0), s2 = Real(0);
        Real t0 = Real(0), t1 = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real d = static_cast<Real>(values[k]);
            Real w = detail::apply_kernel(d * d, params);
            Real dx = X[j] - x_i;

            s0 += w;
            s1 += w * dx;
            s2 += w * dx * dx;
            t0 += w * Y[j];
            t1 += w * Y[j] * dx;
        }

        // Self contribution
        s0 += self_k;
        t0 += self_k * Y[i];

        // Solve 2x2 system
        Real det = s0 * s2 - s1 * s1;
        if (SCL_LIKELY(std::abs(det) > Real(1e-15))) {
            predictions[i] = (s2 * t0 - s1 * t1) / det;
        } else {
            // Fall back to Nadaraya-Watson
            predictions[i] = (s0 > Real(1e-15)) ? t0 / s0 : Y[i];
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// Kernel PCA Approximation (via Nystrom) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void nystrom_approximation(
    const Sparse<T, IsCSR>& landmark_distances,
    Array<Real> embedding,
    Index n_components,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = landmark_distances.primary_dim();
    const Size N = static_cast<Size>(n);
    const Size n_comp = static_cast<Size>(n_components);

    SCL_CHECK_DIM(embedding.len >= N * n_comp, "Nystrom: embedding buffer too small");

    if (SCL_UNLIKELY(n == 0 || n_components == 0)) return;

    detail::KernelParams params(bandwidth, kernel_type);
    const Size total = N * n_comp;
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto Q_ptr = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    auto Q_new_ptr = scl::memory::aligned_alloc<Real>(total, SCL_ALIGNMENT);
    Real* Q = Q_ptr.release();
    Real* Q_new = Q_new_ptr.release();

    // Initialize with deterministic values
    for (Size i = 0; i < total; ++i) {
        Q[i] = std::sin(static_cast<Real>(i) * Real(0.1)) + Real(0.5);
    }

    for (Index iter = 0; iter < config::NYSTROM_MAX_ITER; ++iter) {
        // Apply kernel matrix
        if (use_parallel) {
            scl::threading::parallel_for(Size(0), N, [&](size_t i) {
                auto indices = landmark_distances.primary_indices_unsafe(static_cast<Index>(i));
                auto values = landmark_distances.primary_values_unsafe(static_cast<Index>(i));
                const Index len = landmark_distances.primary_length_unsafe(static_cast<Index>(i));

                Real* qi_new = Q_new + i * n_comp;

                for (Index c = 0; c < n_components; ++c) {
                    Real sum = Real(0);
                    for (Index k = 0; k < len; ++k) {
                        Index j = indices[k];
                        Real d = static_cast<Real>(values[k]);
                        Real kval = detail::apply_kernel(d * d, params);
                        sum += kval * Q[static_cast<Size>(j) * n_comp + c];
                    }
                    qi_new[c] = sum;
                }
            });
        } else {
            for (Index i = 0; i < n; ++i) {
                auto indices = landmark_distances.primary_indices_unsafe(i);
                auto values = landmark_distances.primary_values_unsafe(i);
                const Index len = landmark_distances.primary_length_unsafe(i);

                Real* qi_new = Q_new + static_cast<Size>(i) * n_comp;

                for (Index c = 0; c < n_components; ++c) {
                    Real sum = Real(0);
                    for (Index k = 0; k < len; ++k) {
                        Index j = indices[k];
                        Real d = static_cast<Real>(values[k]);
                        Real kval = detail::apply_kernel(d * d, params);
                        sum += kval * Q[static_cast<Size>(j) * n_comp + c];
                    }
                    qi_new[c] = sum;
                }
            }
        }

        // Orthogonalize (Gram-Schmidt) - sequential for correctness
        for (Index c = 0; c < n_components; ++c) {
            // Subtract projections onto previous components
            for (Index p = 0; p < c; ++p) {
                Real dot = Real(0), norm_p = Real(0);

                // SIMD dot product
                namespace s = scl::simd;
                using SimdTag = s::SimdTagFor<Real>;
                const SimdTag d;
                const size_t lanes = s::Lanes(d);
                auto v_dot = s::Zero(d);
                auto v_norm = s::Zero(d);
                Size i = 0;

                for (; i + lanes <= N; i += lanes) {
                    auto vc = s::LoadU(d, Q_new + i * n_comp + c);
                    auto vp = s::LoadU(d, Q_new + i * n_comp + p);
                    v_dot = s::MulAdd(vc, vp, v_dot);
                    v_norm = s::MulAdd(vp, vp, v_norm);
                }

                dot = s::GetLane(s::SumOfLanes(d, v_dot));
                norm_p = s::GetLane(s::SumOfLanes(d, v_norm));

                // Scalar remainder
                for (; i < N; ++i) {
                    Size ic = i * n_comp + c;
                    Size ip = i * n_comp + p;
                    dot += Q_new[ic] * Q_new[ip];
                    norm_p += Q_new[ip] * Q_new[ip];
                }

                if (SCL_LIKELY(norm_p > Real(1e-15))) {
                    Real coeff = dot / norm_p;
                    for (Size idx = 0; idx < N; ++idx) {
                        Q_new[idx * n_comp + c] -= coeff * Q_new[idx * n_comp + p];
                    }
                }
            }

            // Normalize component c
            Real norm = Real(0);
            for (Size i = 0; i < N; ++i) {
                Real v = Q_new[i * n_comp + c];
                norm += v * v;
            }

            if (SCL_LIKELY(norm > Real(1e-15))) {
                Real inv_norm = Real(1) / std::sqrt(norm);
                for (Size i = 0; i < N; ++i) {
                    Q_new[i * n_comp + c] *= inv_norm;
                }
            }
        }

        // Swap buffers
        Real* tmp = Q;
        Q = Q_new;
        Q_new = tmp;
    }

    // Copy to output
    std::memcpy(embedding.ptr, Q, total * sizeof(Real));

    scl::memory::aligned_free(Q_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(Q, SCL_ALIGNMENT);
}

// =============================================================================
// Mean Shift Step (Single Iteration) - Optimized
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
    const Size N = static_cast<Size>(n);
    const Size stride = static_cast<Size>(n_dims);

    SCL_CHECK_DIM(new_positions.len >= N * stride, "MeanShift: output buffer too small");

    detail::KernelParams params(bandwidth, kernel_type);
    const Real self_k = detail::self_kernel(params);
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        auto indices = distances.primary_indices_unsafe(i);
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);

        Real* new_pos = new_positions.ptr + static_cast<Size>(i) * stride;
        const Real* curr_pos = current_positions.ptr + static_cast<Size>(i) * stride;

        // Zero initialize
        std::memset(new_pos, 0, stride * sizeof(Real));

        Real weight_sum = Real(0);

        // Weighted sum of neighbor positions
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real dist = static_cast<Real>(values[k]);
            Real w = detail::apply_kernel(dist * dist, params);
            const Real* neighbor_pos = current_positions.ptr + static_cast<Size>(j) * stride;

            // SIMD accumulation for high dimensions
            if (stride >= config::SIMD_THRESHOLD) {
                namespace s = scl::simd;
                using SimdTag = s::SimdTagFor<Real>;
                const SimdTag d;
                const size_t lanes = s::Lanes(d);
                auto v_w = s::Set(d, w);
                Index dim = 0;

                for (; dim + static_cast<Index>(lanes) <= n_dims; dim += static_cast<Index>(lanes)) {
                    auto v_pos = s::Load(d, new_pos + dim);
                    auto v_neighbor = s::Load(d, neighbor_pos + dim);
                    s::Store(s::MulAdd(v_w, v_neighbor, v_pos), d, new_pos + dim);
                }

                for (; dim < n_dims; ++dim) {
                    new_pos[dim] += w * neighbor_pos[dim];
                }
            } else {
                for (Index dim = 0; dim < n_dims; ++dim) {
                    new_pos[dim] += w * neighbor_pos[dim];
                }
            }

            weight_sum += w;
        }

        // Self contribution
        for (Index dim = 0; dim < n_dims; ++dim) {
            new_pos[dim] += self_k * curr_pos[dim];
        }

        weight_sum += self_k;

        // Normalize
        if (SCL_LIKELY(weight_sum > Real(1e-15))) {
            Real inv_weight = Real(1) / weight_sum;
            for (Index dim = 0; dim < n_dims; ++dim) {
                new_pos[dim] *= inv_weight;
            }
        } else {
            std::memcpy(new_pos, curr_pos, stride * sizeof(Real));
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// Kernel Entropy Estimation - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void kernel_entropy(
    const Sparse<T, IsCSR>& distances,
    Array<Real> entropy,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(entropy.len >= N, "Entropy: output buffer too small");

    detail::KernelParams params(bandwidth, kernel_type);
    const Real self_k = detail::self_kernel(params);
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);

        // First pass: compute normalization sum
        Real sum = self_k;
        for (Index k = 0; k < len; ++k) {
            Real d = static_cast<Real>(values[k]);
            sum += detail::apply_kernel(d * d, params);
        }

        if (SCL_UNLIKELY(sum <= Real(1e-15))) {
            entropy[i] = Real(0);
            return;
        }

        Real inv_sum = Real(1) / sum;

        // Second pass: compute entropy
        Real H = Real(0);
        for (Index k = 0; k < len; ++k) {
            Real d = static_cast<Real>(values[k]);
            Real kval = detail::apply_kernel(d * d, params);
            Real p = kval * inv_sum;

            if (SCL_LIKELY(p > config::LOG_MIN)) {
                H -= p * std::log(p);
            }
        }

        // Self contribution
        Real self_p = self_k * inv_sum;
        if (SCL_LIKELY(self_p > config::LOG_MIN)) {
            H -= self_p * std::log(self_p);
        }

        entropy[i] = H;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// Perplexity-Based Bandwidth Search (for t-SNE style applications) - Optimized
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
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(bandwidths.len >= N, "Perplexity: bandwidths buffer too small");

    const Real target_entropy = std::log(target_perplexity);
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD);

    auto process_node = [&](Index i) {
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);

        if (SCL_UNLIKELY(len == 0)) {
            bandwidths[i] = config::DEFAULT_BANDWIDTH;
            return;
        }

        // Binary search for bandwidth (beta = 1/(2*sigma^2))
        Real lo = config::MIN_BANDWIDTH;
        Real hi = Real(1e10);
        Real beta = Real(1);

        for (Index iter = 0; iter < max_iter; ++iter) {
            // Compute probabilities and entropy
            Real sum_p = Real(0);

            // Fast exponential sum
            for (Index k = 0; k < len; ++k) {
                Real d = static_cast<Real>(values[k]);
                sum_p += std::exp(-beta * d * d);
            }

            if (SCL_UNLIKELY(sum_p < Real(1e-300))) {
                beta *= Real(0.5);
                continue;
            }

            Real inv_sum = Real(1) / sum_p;
            Real H = Real(0);

            for (Index k = 0; k < len; ++k) {
                Real d = static_cast<Real>(values[k]);
                Real p = std::exp(-beta * d * d) * inv_sum;

                if (SCL_LIKELY(p > config::LOG_MIN)) {
                    H -= p * std::log(p);
                }
            }

            Real diff = H - target_entropy;
            if (std::abs(diff) < tol) break;

            if (diff > Real(0)) {
                // Entropy too high, increase beta
                lo = beta;
                beta = (hi > Real(1e9)) ? beta * Real(2) : (lo + hi) * Real(0.5);
            } else {
                // Entropy too low, decrease beta
                hi = beta;
                beta = (lo + hi) * Real(0.5);
            }
        }

        // Convert beta to bandwidth
        bandwidths[i] = (beta > Real(1e-15)) ?
            Real(1) / std::sqrt(Real(2) * beta) : config::DEFAULT_BANDWIDTH;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            process_node(static_cast<Index>(i));
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            process_node(i);
        }
    }
}

// =============================================================================
// Kernel Two-Sample Test Statistic (MMD Approximation) - Optimized
// =============================================================================

template <typename T, bool IsCSR>
Real kernel_mmd_from_groups(
    const Sparse<T, IsCSR>& distances,
    Array<const bool> group_labels,
    Real bandwidth = config::DEFAULT_BANDWIDTH,
    KernelType kernel_type = KernelType::Gaussian
) {
    const Index n = distances.primary_dim();
    detail::KernelParams params(bandwidth, kernel_type);
    const Real self_k = detail::self_kernel(params);

    // Count samples in each group
    Size n_A = 0, n_B = 0;
    for (Index i = 0; i < n; ++i) {
        if (group_labels[i]) ++n_A;
        else ++n_B;
    }

    if (SCL_UNLIKELY(n_A == 0 || n_B == 0)) return Real(0);

    // Compute kernel sums (could parallelize with reduction)
    Real k_AA = Real(0);
    Real k_BB = Real(0);
    Real k_AB = Real(0);

    for (Index i = 0; i < n; ++i) {
        auto indices = distances.primary_indices_unsafe(i);
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);
        bool in_A = group_labels[i];

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real d = static_cast<Real>(values[k]);
            Real kval = detail::apply_kernel(d * d, params);
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
        if (in_A) k_AA += self_k;
        else k_BB += self_k;
    }

    // MMD^2 = E[K(X,X')] + E[K(Y,Y')] - 2*E[K(X,Y)]
    Real n_A_r = static_cast<Real>(n_A);
    Real n_B_r = static_cast<Real>(n_B);
    Real mmd_sq = k_AA / (n_A_r * n_A_r)
                + k_BB / (n_B_r * n_B_r)
                - Real(2) * k_AB / (n_A_r * n_B_r);

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
    detail::KernelParams params(bandwidth, type);
    return detail::apply_kernel(distance * distance, params);
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
            return std::pow(Real(2) * pi, -static_cast<Real>(dimension) * Real(0.5));
        }
        case KernelType::Epanechnikov: {
            return Real(0.75);
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
    Real* SCL_RESTRICT kernel_values
) {
    compute_kernel_matrix(distances, kernel_values, bandwidth, KernelType::Gaussian);
}

// =============================================================================
// Adaptive RBF Kernel - Optimized
// =============================================================================

template <typename T, bool IsCSR>
void adaptive_rbf(
    const Sparse<T, IsCSR>& distances,
    Array<const Real> bandwidths,
    Real* SCL_RESTRICT kernel_values
) {
    const Index n = distances.primary_dim();
    Size val_idx = 0;

    for (Index i = 0; i < n; ++i) {
        auto indices = distances.primary_indices_unsafe(i);
        auto values = distances.primary_values_unsafe(i);
        const Index len = distances.primary_length_unsafe(i);

        Real h_i = scl::algo::max2(bandwidths[i], config::MIN_BANDWIDTH);
        Real h_i_sq = h_i * h_i;

        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            for (int off = 0; off < 4; ++off) {
                Index j = indices[k + off];
                Real h_j = scl::algo::max2(bandwidths[j], config::MIN_BANDWIDTH);
                Real h_sq = h_i_sq * h_j * h_j;  // (h_i * h_j)^2 for geometric mean squared
                Real h = std::sqrt(std::sqrt(h_sq));  // sqrt(h_i * h_j)
                Real d = static_cast<Real>(values[k + off]);
                Real inv_2h2 = Real(0.5) / (h * h);
                kernel_values[val_idx++] = std::exp(-d * d * inv_2h2);
            }
        }

        for (; k < len; ++k) {
            Index j = indices[k];
            Real h_j = scl::algo::max2(bandwidths[j], config::MIN_BANDWIDTH);
            Real h = std::sqrt(h_i * h_j);
            Real d = static_cast<Real>(values[k]);
            Real inv_2h2 = Real(0.5) / (h * h);
            kernel_values[val_idx++] = std::exp(-d * d * inv_2h2);
        }
    }
}

} // namespace scl::kernel::kernel
