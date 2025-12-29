#pragma once

#include "scl/core/sparse.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"

#include <cmath>
#include <cstdint>

// =============================================================================
// FILE: scl/kernel/stat/permutation_stat.hpp
// BRIEF: Optimized permutation testing for statistical kernels
//
// KEY OPTIMIZATION:
//   Pre-sort data ONCE, then permute group_ids (not data) for each permutation.
//   This avoids O(n log n) sort per permutation, reducing to O(n) shuffle.
// =============================================================================

namespace scl::kernel::stat::permutation_stat {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size DEFAULT_N_PERMUTATIONS = 1000;
    constexpr Size MIN_PERMUTATIONS = 100;
    constexpr Size MAX_PERMUTATIONS = 100000;
    constexpr Size EARLY_CHECK_INTERVAL = 100;
    constexpr Real EARLY_STOP_ALPHA = Real(0.001);
    constexpr Real EARLY_STOP_BETA = Real(0.5);
}

// =============================================================================
// Fast PRNG (Xoshiro256++)
// =============================================================================

namespace detail {
class FastRNG {
public:
    explicit FastRNG(uint64_t seed) noexcept : state_{} {
        uint64_t s = seed;
        for (auto& state_elem : state_) {
            s += 0x9e3779b97f4a7c15ULL;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            state_elem = z ^ (z >> 31);
        }
    }

    SCL_FORCE_INLINE uint64_t operator()() noexcept {
        const uint64_t result = rotl(state_[0] + state_[3], 23) + state_[0];
        const uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

    // Lemire's nearly divisionless method
    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        uint64_t x = (*this)();
        __uint128_t m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n);
        auto l = static_cast<uint64_t>(m);
        if (l < n) {
            auto t = -n % n;
            while (l < t) {
                x = (*this)();
                m = static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n);
                l = static_cast<uint64_t>(m);
            }
        }
        return static_cast<Size>(m >> 64);
    }

    void jump() noexcept {
        // NOLINTNEXTLINE(modernize-avoid-c-arrays)
        static const uint64_t JUMP[] = {
            0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
            0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL
        };
        uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (const auto& jump_val : JUMP) {
            for (int b = 0; b < 64; ++b) {
                if (jump_val & (1ULL << b)) {
                    s0 ^= state_[0]; s1 ^= state_[1];
                    s2 ^= state_[2]; s3 ^= state_[3];
                }
                (*this)();
            }
        }
        state_[0] = s0; state_[1] = s1;
        state_[2] = s2; state_[3] = s3;
    }

private:
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    alignas(32) uint64_t state_[4] {0, 0, 0, 0};
    static SCL_FORCE_INLINE uint64_t rotl(uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
};

// Fisher-Yates shuffle for group_ids array
SCL_FORCE_INLINE void shuffle_groups(int32_t* groups, Size n, FastRNG& rng) noexcept {
    Size i = n - 1;
    for (; i >= 4; i -= 4) {
        Size j0 = rng.bounded(i + 1), j1 = rng.bounded(i);
        Size j2 = rng.bounded(i - 1), j3 = rng.bounded(i - 2);
        int32_t t0 = groups[i];     groups[i] = groups[j0];     groups[j0] = t0;
        int32_t t1 = groups[i-1];   groups[i-1] = groups[j1];   groups[j1] = t1;
        int32_t t2 = groups[i-2];   groups[i-2] = groups[j2];   groups[j2] = t2;
        int32_t t3 = groups[i-3];   groups[i-3] = groups[j3];   groups[j3] = t3;
    }
    for (; i > 0; --i) {
        Size j = rng.bounded(i + 1);
        int32_t tmp = groups[i];
        groups[i] = groups[j];
        groups[j] = tmp;
    }
}

// Compute U statistic from pre-sorted data with permuted groups
template <typename T>
SCL_FORCE_INLINE Real compute_U_from_sorted(
    const T* sorted_values,
    const Size* sorted_indices,
    Size n,
    const int32_t* perm_groups,
    Size n1,
    Size n2
) {
    if (n1 == 0 || n2 == 0) return Real(0);

    double R1 = 0.0;
    Size i = 0;
    Size rank = 1;

    while (i < n) {
        T val = sorted_values[i];
        Size tie_start = i;
        while (i < n && sorted_values[i] == val) ++i;
        Size tie_count = i - tie_start;
        double avg_rank = static_cast<double>(rank) + static_cast<double>(tie_count - 1) * 0.5;
        for (Size j = tie_start; j < i; ++j) {
            if (perm_groups[sorted_indices[j]] == 0) {
                R1 += avg_rank;
            }
        }
        rank += tie_count;
    }

    double U1 = R1 - static_cast<double>(n1) * static_cast<double>(n1 + 1) * 0.5;
    return static_cast<Real>(U1);
}

// Compute mean difference from pre-sorted data with permuted groups
template <typename T>
SCL_FORCE_INLINE Real compute_mean_diff_from_sorted(
    const T* values,
    const Size* indices,
    Size n,
    const int32_t* perm_groups,
    [[maybe_unused]] Size n1,
    [[maybe_unused]] Size n2
) {
    double sum1 = 0.0, sum2 = 0.0;
    Size count1 = 0, count2 = 0;

    for (Size i = 0; i < n; ++i) {
        auto v = static_cast<double>(values[i]);
        if (perm_groups[indices[i]] == 0) {
            sum1 += v;
            count1++;
        } else if (perm_groups[indices[i]] == 1) {
            sum2 += v;
            count2++;
        }
    }

    double mean1 = (count1 > 0) ? sum1 / static_cast<double>(count1) : 0.0;
    double mean2 = (count2 > 0) ? sum2 / static_cast<double>(count2) : 0.0;
    return static_cast<Real>(mean1 - mean2);
}

} // namespace detail

// =============================================================================
// Statistic Type
// =============================================================================

enum class PermStatType {
    MWU,       // Mann-Whitney U statistic
    MeanDiff,  // Mean difference (t-test like)
    KS         // Kolmogorov-Smirnov D statistic
};

// =============================================================================
// Batch Permutation Test - Reuse Sort Optimization
// =============================================================================

template <typename T, bool IsCSR>
void batch_permutation_reuse_sort(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_permutations,
    Array<Real> out_p_values,
    PermStatType stat_type = PermStatType::MWU,
    uint64_t seed = 42
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N_features = static_cast<Size>(primary_dim);
    const Size N_samples = group_ids.len;

    n_permutations = scl::algo::max2(n_permutations, config::MIN_PERMUTATIONS);
    n_permutations = scl::algo::min2(n_permutations, config::MAX_PERMUTATIONS);

    Size n1 = 0, n2 = 0;
    for (Size i = 0; i < N_samples; ++i) {
        if (group_ids[static_cast<Index>(i)] == 0) n1++;
        else if (group_ids[static_cast<Index>(i)] == 1) n2++;
    }
    SCL_CHECK_ARG(n1 > 0 && n2 > 0, "Both groups must have members");

    Size max_len = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Size len = static_cast<Size>(matrix.primary_length_unsafe(i));
        if (len > max_len) max_len = len;
    }

    const size_t n_threads = scl::threading::get_num_threads_runtime();

    // Workspace: sorted values, sorted indices, permuted groups, null distribution
    Size workspace_per_thread = max_len + max_len + N_samples + n_permutations;
    scl::threading::WorkspacePool<double> work_pool;
    work_pool.init(n_threads, workspace_per_thread);

    scl::threading::parallel_for(Size(0), N_features, [&](size_t p, size_t thread_rank) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        if (SCL_UNLIKELY(len_sz == 0)) {
            out_p_values[static_cast<Index>(p)] = Real(1);
            return;
        }

        auto values = matrix.primary_values_unsafe(idx);
        auto indices = matrix.primary_indices_unsafe(idx);

        double* workspace = work_pool.get(thread_rank);
        T* sorted_vals = reinterpret_cast<T*>(workspace);
        Size* sorted_idx = reinterpret_cast<Size*>(workspace + max_len);
        auto perm_groups = reinterpret_cast<int32_t*>(workspace + max_len * 2);
        Real* null_dist = reinterpret_cast<Real*>(workspace + max_len * 2 + N_samples);

        // Copy and filter valid samples
        Size total = 0;
        for (Size k = 0; k < len_sz; ++k) {
            Index sec_idx = indices[k];
            int32_t g = group_ids[sec_idx];
            if (g == 0 || g == 1) {
                sorted_vals[total] = values[k];
                sorted_idx[total] = static_cast<Size>(sec_idx);
                total++;
            }
        }

        if (total == 0) {
            out_p_values[static_cast<Index>(p)] = Real(1);
            return;
        }

        // Sort by value (argsort)
        for (Size i = 1; i < total; ++i) {
            T key_val = sorted_vals[i];
            Size key_idx = sorted_idx[i];
            Size j = i;
            while (j > 0 && sorted_vals[j - 1] > key_val) {
                sorted_vals[j] = sorted_vals[j - 1];
                sorted_idx[j] = sorted_idx[j - 1];
                --j;
            }
            sorted_vals[j] = key_val;
            sorted_idx[j] = key_idx;
        }

        // Initialize permuted groups
        for (Size i = 0; i < N_samples; ++i) {
            perm_groups[i] = group_ids[static_cast<Index>(i)];
        }

        // Compute observed statistic
        auto observed = Real(0);
        switch (stat_type) {
            case PermStatType::MWU:
                observed = detail::compute_U_from_sorted(
                    sorted_vals, sorted_idx, total, group_ids.ptr, n1, n2);
                break;
            case PermStatType::MeanDiff:
                observed = detail::compute_mean_diff_from_sorted(
                    sorted_vals, sorted_idx, total, group_ids.ptr, n1, n2);
                break;
            default:
                observed = Real(0);
        }

        // Setup RNG with unique stream per thread
        detail::FastRNG rng(seed);
        for (size_t j = 0; j < thread_rank; ++j) {
            rng.jump();
        }
        for (size_t j = 0; j < (p % 16); ++j) {
            rng();
        }

        Real abs_obs = (observed >= 0) ? observed : -observed;
        Size actual_perms = n_permutations;

        // Generate null distribution
        for (Size perm = 0; perm < n_permutations; ++perm) {
            detail::shuffle_groups(perm_groups, N_samples, rng);

            switch (stat_type) {
                case PermStatType::MWU:
                    null_dist[perm] = detail::compute_U_from_sorted(
                        sorted_vals, sorted_idx, total, perm_groups, n1, n2);
                    break;
                case PermStatType::MeanDiff:
                    null_dist[perm] = detail::compute_mean_diff_from_sorted(
                        sorted_vals, sorted_idx, total, perm_groups, n1, n2);
                    break;
                default:
                    null_dist[perm] = Real(0);
            }

            // Adaptive early stopping
            if ((perm + 1) % config::EARLY_CHECK_INTERVAL == 0) {
                Size checked = perm + 1;
                Size extreme = 0;
                for (Size k = 0; k < checked; ++k) {
                    Real av = (null_dist[k] >= 0) ? null_dist[k] : -null_dist[k];
                    extreme += (av >= abs_obs);
                }
                Real pval = static_cast<Real>(extreme + 1) / static_cast<Real>(checked + 1);
                if (pval < config::EARLY_STOP_ALPHA || pval > config::EARLY_STOP_BETA) {
                    actual_perms = checked;
                    break;
                }
            }
        }

        // Compute two-sided p-value
        Size count = 0;
        for (Size k = 0; k < actual_perms; ++k) {
            Real av = (null_dist[k] >= 0) ? null_dist[k] : -null_dist[k];
            count += (av >= abs_obs);
        }
        out_p_values[static_cast<Index>(p)] = static_cast<Real>(count + 1) / static_cast<Real>(actual_perms + 1);
    });
}

// =============================================================================
// Single Feature Permutation Test
// =============================================================================

template <typename T>
Real permutation_test_single(
    Array<const T> values,
    Array<const int32_t> group_ids,
    Size n_permutations,
    PermStatType stat_type = PermStatType::MWU,
    uint64_t seed = 42
) {
    const Size n = values.len;
    SCL_CHECK_DIM(group_ids.len == n, "Values and groups must have same length");

    n_permutations = scl::algo::max2(n_permutations, config::MIN_PERMUTATIONS);
    n_permutations = scl::algo::min2(n_permutations, config::MAX_PERMUTATIONS);

    Size n1 = 0, n2 = 0;
    for (Size i = 0; i < n; ++i) {
        if (group_ids[static_cast<Index>(i)] == 0) n1++;
        else if (group_ids[static_cast<Index>(i)] == 1) n2++;
    }

    if (n1 == 0 || n2 == 0) return Real(1);

    // Allocate workspace
    auto sorted_vals_ptr = scl::memory::aligned_alloc<T>(n, SCL_ALIGNMENT);
    auto sorted_idx_ptr = scl::memory::aligned_alloc<Size>(n, SCL_ALIGNMENT);
    auto perm_groups_ptr = scl::memory::aligned_alloc<int32_t>(n, SCL_ALIGNMENT);
    auto null_dist_ptr = scl::memory::aligned_alloc<Real>(n_permutations, SCL_ALIGNMENT);
    
    T* sorted_vals = sorted_vals_ptr.get();
    Size* sorted_idx = sorted_idx_ptr.get();
    int32_t* perm_groups = perm_groups_ptr.get();
    Real* null_dist = null_dist_ptr.get();

    // Filter and sort
    Size total = 0;
    for (Size i = 0; i < n; ++i) {
        if (group_ids[static_cast<Index>(i)] == 0 || group_ids[static_cast<Index>(i)] == 1) {
            sorted_vals[total] = values[static_cast<Index>(i)];
            sorted_idx[total] = i;
            total++;
        }
    }

    // Argsort
    for (Size i = 1; i < total; ++i) {
        T key_val = sorted_vals[i];
        Size key_idx = sorted_idx[i];
        Size j = i;
        while (j > 0 && sorted_vals[j - 1] > key_val) {
            sorted_vals[j] = sorted_vals[j - 1];
            sorted_idx[j] = sorted_idx[j - 1];
            --j;
        }
        sorted_vals[j] = key_val;
        sorted_idx[j] = key_idx;
    }

    for (Size i = 0; i < n; ++i) {
        perm_groups[i] = group_ids[static_cast<Index>(i)];
    }

    auto observed = Real(0);
    switch (stat_type) {
        case PermStatType::MWU:
            observed = detail::compute_U_from_sorted(
                sorted_vals, sorted_idx, total, group_ids.ptr, n1, n2);
            break;
        case PermStatType::MeanDiff:
            observed = detail::compute_mean_diff_from_sorted(
                sorted_vals, sorted_idx, total, group_ids.ptr, n1, n2);
            break;
        default:
            observed = Real(0);
    }

    detail::FastRNG rng(seed);
    Real abs_obs = (observed >= 0) ? observed : -observed;

    for (Size perm = 0; perm < n_permutations; ++perm) {
        detail::shuffle_groups(perm_groups, n, rng);

        switch (stat_type) {
            case PermStatType::MWU:
                null_dist[perm] = detail::compute_U_from_sorted(
                    sorted_vals, sorted_idx, total, perm_groups, n1, n2);
                break;
            case PermStatType::MeanDiff:
                null_dist[perm] = detail::compute_mean_diff_from_sorted(
                    sorted_vals, sorted_idx, total, perm_groups, n1, n2);
                break;
            default:
                null_dist[perm] = Real(0);
        }
    }

    Size count = 0;
    for (Size k = 0; k < n_permutations; ++k) {
        Real av = (null_dist[k] >= 0) ? null_dist[k] : -null_dist[k];
        count += (av >= abs_obs);
    }
    Real p_value = static_cast<Real>(count + 1) / static_cast<Real>(n_permutations + 1);

    // Memory will be automatically freed by unique_ptr destructors
    return p_value;
}

} // namespace scl::kernel::stat::permutation_stat
