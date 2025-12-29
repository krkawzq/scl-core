#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"

#include <array>
#include <cmath>

// =============================================================================
// FILE: scl/kernel/sampling.hpp
// BRIEF: Advanced sampling strategies for large single-cell datasets
//
// APPLICATIONS:
// - Geometric sketching for preserving rare populations
// - Density-preserving downsampling
// - Landmark selection for scalable analysis
// - Representative cell selection
// =============================================================================

namespace scl::kernel::sampling {

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size DEFAULT_BINS = 64;
    constexpr Size MAX_ITERATIONS = 1000;
    constexpr Real CONVERGENCE_TOL = Real(1e-6);
    constexpr Size PARALLEL_THRESHOLD = 256;
}

namespace detail {

// Simple hash function for grid-based sampling
SCL_FORCE_INLINE uint64_t hash_combine(uint64_t h1, uint64_t h2) {
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
}

// Fast PRNG (Xoshiro128+) - higher quality than LCG
struct alignas(16) FastRNG {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::array<uint32_t, 4> s{};

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept {
        uint64_t z = seed;
        for (uint32_t& si : s) {
            z += 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            si = static_cast<uint32_t>(z >> 32);
        }
    }

    [[nodiscard]] SCL_FORCE_INLINE uint32_t rotl(uint32_t x, int k) const noexcept {
        return (x << k) | (x >> (32 - k));
    }

    SCL_FORCE_INLINE uint32_t next() noexcept {
        const uint32_t result = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 11);
        return result;
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next()) * Real(2.3283064365386963e-10);
    }

    SCL_FORCE_INLINE Size next_index(Size max_val) noexcept {
        return static_cast<Size>(next() % static_cast<uint32_t>(max_val));
    }
};

// Compute grid cell index for geometric sketching
SCL_FORCE_INLINE uint64_t compute_grid_cell(
    const Real* point,
    Size n_dims,
    const Real* min_vals,
    const Real* cell_sizes,
    Size n_bins
) {
    uint64_t cell_hash = 0;
    for (Size d = 0; d < n_dims; ++d) {
        Size bin = static_cast<Size>((point[d] - min_vals[d]) / (cell_sizes[d] + config::EPSILON));
        bin = scl::algo::min2(bin, n_bins - 1);
        cell_hash = hash_combine(cell_hash, static_cast<uint64_t>(bin));
    }
    return cell_hash;
}

// Weighted reservoir sampling
SCL_FORCE_INLINE void weighted_reservoir(
    const Real* weights,
    Size n,
    Size k,
    Index* selected,
    FastRNG& rng
) {
    // Compute prefix sum of weights
    auto cumsum_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    Real* cumsum = cumsum_ptr.release();
    cumsum[0] = weights[0];
    for (Size i = 1; i < n; ++i) {
        cumsum[i] = cumsum[i - 1] + weights[i];
    }
    Real total = cumsum[n - 1];

    // Sample k items
    for (Size s = 0; s < k; ++s) {
        Real r = rng.uniform() * total;

        // Binary search
        Size lo = 0, hi = n;
        while (lo < hi) {
            Size mid = (lo + hi) / 2;
            if (cumsum[mid] < r) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        selected[s] = static_cast<Index>(lo);
    }

    scl::memory::aligned_free(cumsum);
}

// KMeans++ initialization for landmark selection
template <typename T, bool IsCSR>
void kmeans_pp_init(
    const Sparse<T, IsCSR>& data,
    Size k,
    Index* centers,
    FastRNG& rng
) {
    const Size n = static_cast<Size>(data.rows());

    // Distance to nearest center for each point
    auto min_dist_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    Real* min_dist = min_dist_ptr.release();
    for (Size i = 0; i < n; ++i) {
        min_dist[i] = std::numeric_limits<Real>::max();
    }

    // Choose first center randomly
    centers[0] = static_cast<Index>(rng.next_index(n));

    for (Size c = 1; c < k; ++c) {
        Index last_center = centers[c - 1];

        const Index center_start = data.row_indices_unsafe()[last_center];
        const Index center_end = data.row_indices_unsafe()[last_center + 1];

        // Parallel distance update
        auto update_distance = [&](Size i) {
            Real dist = Real(0.0);

            const Index row_start = data.row_indices_unsafe()[i];
            const Index row_end = data.row_indices_unsafe()[i + 1];

            // Sparse distance computation with merge
            Index ci = center_start, ri = row_start;
            while (ci < center_end && ri < row_end) {
                Index c_col = data.col_indices_unsafe()[ci];
                Index r_col = data.col_indices_unsafe()[ri];

                if (c_col == r_col) {
                    Real diff = static_cast<Real>(data.values()[ci]) -
                               static_cast<Real>(data.values()[ri]);
                    dist += diff * diff;
                    ++ci;
                    ++ri;
                } else if (c_col < r_col) {
                    Real val = static_cast<Real>(data.values()[ci]);
                    dist += val * val;
                    ++ci;
                } else {
                    Real val = static_cast<Real>(data.values()[ri]);
                    dist += val * val;
                    ++ri;
                }
            }
            while (ci < center_end) {
                Real val = static_cast<Real>(data.values()[ci++]);
                dist += val * val;
            }
            while (ri < row_end) {
                Real val = static_cast<Real>(data.values()[ri++]);
                dist += val * val;
            }

            min_dist[i] = scl::algo::min2(min_dist[i], dist);
        };

        if (n >= config::PARALLEL_THRESHOLD) {
            scl::threading::parallel_for(Size(0), n, [&](Size i) {
                update_distance(i);
            });
        } else {
            for (Size i = 0; i < n; ++i) {
                update_distance(i);
            }
        }

        // Parallel reduction for total distance
        Real total_dist = Real(0.0);
        if (n >= config::PARALLEL_THRESHOLD) {
            total_dist = scl::vectorize::sum(Array<const Real>(min_dist, n));
        } else {
            for (Size i = 0; i < n; ++i) {
                total_dist += min_dist[i];
            }
        }

        // Sample next center proportional to squared distance
        Real r = rng.uniform() * total_dist;
        Real cumsum = Real(0.0);
        for (Size i = 0; i < n; ++i) {
            cumsum += min_dist[i];
            if (cumsum >= r) {
                centers[c] = static_cast<Index>(i);
                break;
            }
        }
    }

    scl::memory::aligned_free(min_dist);
}

} // namespace detail

// =============================================================================
// Geometric Sketching
// =============================================================================

template <typename T, bool IsCSR>
void geometric_sketching(
    const Sparse<T, IsCSR>& data,
    Size target_size,
    Index* selected_indices,
    Size& n_selected,
    uint64_t seed
) {
    const Size n_cells = static_cast<Size>(data.rows());
    const Size n_features = static_cast<Size>(data.cols());

    n_selected = 0;
    if (n_cells == 0 || target_size == 0) return;

    target_size = scl::algo::min2(target_size, n_cells);

    detail::FastRNG rng(seed);

    // Compute data bounds for grid
    auto min_vals_ptr = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);

    Real* min_vals = min_vals_ptr.release();
    auto max_vals_ptr = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);

    Real* max_vals = max_vals_ptr.release();
    auto cell_sizes_ptr = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);

    Real* cell_sizes = cell_sizes_ptr.release();

    for (Size d = 0; d < n_features; ++d) {
        min_vals[d] = std::numeric_limits<Real>::max();
        max_vals[d] = std::numeric_limits<Real>::lowest();
    }

    // Find bounds
    for (Size i = 0; i < n_cells; ++i) {
        const Index row_start = data.row_indices_unsafe()[i];
        const Index row_end = data.row_indices_unsafe()[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index feature = data.col_indices_unsafe()[j];
            Real val = static_cast<Real>(data.values()[j]);
            min_vals[feature] = scl::algo::min2(min_vals[feature], val);
            max_vals[feature] = scl::algo::max2(max_vals[feature], val);
        }
    }

    // Compute grid cell sizes
    Size n_bins = config::DEFAULT_BINS;
    for (Size d = 0; d < n_features; ++d) {
        if (min_vals[d] > max_vals[d]) {
            min_vals[d] = Real(0.0);
            max_vals[d] = Real(1.0);
        }
        cell_sizes[d] = (max_vals[d] - min_vals[d]) / static_cast<Real>(n_bins);
        if (cell_sizes[d] < config::EPSILON) {
            cell_sizes[d] = Real(1.0);
        }
    }

    // Assign cells to grid buckets and sample from each
    auto cell_hashes_ptr = scl::memory::aligned_alloc<uint64_t>(n_cells, SCL_ALIGNMENT);

    uint64_t* cell_hashes = cell_hashes_ptr.release();
    auto point_buffer_ptr = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);

    Real* point_buffer = point_buffer_ptr.release();

    for (Size i = 0; i < n_cells; ++i) {
        // Build dense point representation
        for (Size d = 0; d < n_features; ++d) {
            point_buffer[d] = Real(0.0);
        }

        const Index row_start = data.row_indices_unsafe()[i];
        const Index row_end = data.row_indices_unsafe()[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index feature = data.col_indices_unsafe()[j];
            point_buffer[feature] = static_cast<Real>(data.values()[j]);
        }

        cell_hashes[i] = detail::compute_grid_cell(point_buffer, n_features,
            min_vals, cell_sizes, n_bins);
    }

    // Sort by cell hash using sort_pairs
    auto sorted_indices_ptr = scl::memory::aligned_alloc<Index>(n_cells, SCL_ALIGNMENT);

    Index* sorted_indices = sorted_indices_ptr.release();
    for (Size i = 0; i < n_cells; ++i) {
        sorted_indices[i] = static_cast<Index>(i);
    }
    scl::sort::sort_pairs(
        Array<uint64_t>(cell_hashes, n_cells),
        Array<Index>(sorted_indices, n_cells)
    );

    // Count cells per bucket
    Size n_buckets = 1;
    for (Size i = 1; i < n_cells; ++i) {
        if (cell_hashes[i] != cell_hashes[i - 1]) {
            ++n_buckets;
        }
    }

    // Sample proportionally from each bucket
    Size cells_per_bucket = (target_size + n_buckets - 1) / n_buckets;
    cells_per_bucket = scl::algo::max2(Size(1), cells_per_bucket);

    Size bucket_start = 0;
    uint64_t current_hash = cell_hashes[0];

    for (Size i = 0; i <= n_cells && n_selected < target_size; ++i) {
        bool end_bucket = (i == n_cells) ||
            (i > 0 && cell_hashes[i] != current_hash);

        if (end_bucket) {
            Size bucket_size = i - bucket_start;
            Size to_sample = scl::algo::min2(cells_per_bucket, bucket_size);
            to_sample = scl::algo::min2(to_sample, target_size - n_selected);

            // Sample from this bucket
            for (Size s = 0; s < to_sample; ++s) {
                Size idx = bucket_start + rng.next_index(bucket_size);
                selected_indices[n_selected++] = sorted_indices[idx];
            }

            if (i < n_cells) {
                bucket_start = i;
                current_hash = cell_hashes[i];
            }
        }
    }

    scl::memory::aligned_free(min_vals);
    scl::memory::aligned_free(max_vals);
    scl::memory::aligned_free(cell_sizes);
    scl::memory::aligned_free(cell_hashes);
    scl::memory::aligned_free(point_buffer);
    scl::memory::aligned_free(sorted_indices);
}

// =============================================================================
// Density-Preserving Sampling
// =============================================================================

template <typename T, bool IsCSR>
void density_preserving(
    const Sparse<T, IsCSR>& data,
    const Sparse<Index, IsCSR>& neighbors,
    Size target_size,
    Index* selected_indices,
    Size& n_selected
) {
    const Size n_cells = static_cast<Size>(data.rows());

    n_selected = 0;
    if (n_cells == 0 || target_size == 0) return;

    target_size = scl::algo::min2(target_size, n_cells);

    // Compute local density for each cell
    auto density_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* density = density_ptr.release();

    for (Size i = 0; i < n_cells; ++i) {
        const Index row_start = neighbors.row_indices_unsafe()[i];
        const Index row_end = neighbors.row_indices_unsafe()[i + 1];
        density[i] = static_cast<Real>(row_end - row_start + 1);  // +1 for self
    }

    // Compute sampling weights (inverse density)
    auto weights_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* weights = weights_ptr.release();
    Real total_weight = Real(0.0);

    for (Size i = 0; i < n_cells; ++i) {
        weights[i] = Real(1.0) / (density[i] + config::EPSILON);
        total_weight += weights[i];
    }

    // Normalize weights
    for (Size i = 0; i < n_cells; ++i) {
        weights[i] /= total_weight;
    }

    // Sample without replacement using systematic sampling
    auto cumsum_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);

    Real* cumsum = cumsum_ptr.release();
    cumsum[0] = weights[0];
    for (Size i = 1; i < n_cells; ++i) {
        cumsum[i] = cumsum[i - 1] + weights[i];
    }

    // Systematic sampling
    Real step = Real(1.0) / static_cast<Real>(target_size);
    Real start = step * Real(0.5);

    Size ci = 0;
    for (Size s = 0; s < target_size; ++s) {
        Real r = start + step * static_cast<Real>(s);
        while (ci < n_cells - 1 && cumsum[ci] < r) {
            ++ci;
        }
        selected_indices[n_selected++] = static_cast<Index>(ci);
    }

    scl::memory::aligned_free(density);
    scl::memory::aligned_free(weights);
    scl::memory::aligned_free(cumsum);
}

// =============================================================================
// Landmark Selection
// =============================================================================

template <typename T, bool IsCSR>
void landmark_selection(
    const Sparse<T, IsCSR>& data,
    Size n_landmarks,
    Index* landmark_indices,
    Size& n_selected,
    uint64_t seed
) {
    const Size n_cells = static_cast<Size>(data.rows());

    n_selected = 0;
    if (n_cells == 0 || n_landmarks == 0) return;

    n_landmarks = scl::algo::min2(n_landmarks, n_cells);

    detail::FastRNG rng(seed);

    // Use KMeans++ initialization to select diverse landmarks
    detail::kmeans_pp_init(data, n_landmarks, landmark_indices, rng);
    n_selected = n_landmarks;
}

// =============================================================================
// Representative Cells
// =============================================================================

template <typename T, bool IsCSR>
void representative_cells(
    const Sparse<T, IsCSR>& data,
    Array<const Index> cluster_labels,
    Size per_cluster,
    Index* representatives,
    Size& n_selected,
    uint64_t /* seed */
) {
    const Size n_cells = static_cast<Size>(data.rows());
    SCL_CHECK_DIM(n_cells == cluster_labels.len, "Labels length must match data rows");

    n_selected = 0;
    if (n_cells == 0 || per_cluster == 0) return;

    // Find number of clusters
    Index n_clusters = 0;
    for (Size i = 0; i < n_cells; ++i) {
        if (cluster_labels.ptr[i] >= n_clusters) {
            n_clusters = cluster_labels.ptr[i] + 1;
        }
    }

    // Count cells per cluster
    auto cluster_sizes_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);
    Size* cluster_sizes = cluster_sizes_ptr.get();
    for (Index c = 0; c < n_clusters; ++c) {
        cluster_sizes[c] = 0;
    }
    for (Size i = 0; i < n_cells; ++i) {
        ++cluster_sizes[cluster_labels.ptr[i]];
    }

    // Collect indices per cluster
    auto cluster_indices_ptr = scl::memory::aligned_alloc<Index*>(n_clusters, SCL_ALIGNMENT);
    Index** cluster_indices = cluster_indices_ptr.get();
    auto cluster_counts_ptr = scl::memory::aligned_alloc<Size>(n_clusters, SCL_ALIGNMENT);
    Size* cluster_counts = cluster_counts_ptr.get();

    // Each Index* in cluster_indices needs its own unique_ptr for correct aligned_free
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::vector<std::unique_ptr<Index[], scl::memory::AlignedDeleter<Index>>> owned_cluster_indices;
    owned_cluster_indices.reserve(n_clusters);

    for (Index c = 0; c < n_clusters; ++c) {
        auto arr = scl::memory::aligned_alloc<Index>(cluster_sizes[c], SCL_ALIGNMENT);
        cluster_indices[c] = arr.get();
        owned_cluster_indices.push_back(std::move(arr));
        cluster_counts[c] = 0;
    }

    for (Size i = 0; i < n_cells; ++i) {
        Index c = cluster_labels.ptr[i];
        cluster_indices[c][cluster_counts[c]++] = static_cast<Index>(i);
    }

    // Allocate centroid storage
    auto centroids_ptr = scl::memory::aligned_alloc<Real*>(n_clusters, SCL_ALIGNMENT);
    Real** centroids = centroids_ptr.get();
    const Size n_features = static_cast<Size>(data.cols());

    // Each Real* in centroids needs its own unique_ptr for correct aligned_free
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::vector<std::unique_ptr<Real[], scl::memory::AlignedDeleter<Real>>> owned_centroids;
    owned_centroids.reserve(n_clusters);

    for (Index c = 0; c < n_clusters; ++c) {
        auto arr = scl::memory::aligned_alloc<Real>(n_features, SCL_ALIGNMENT);
        centroids[c] = arr.get();
        for (Size d = 0; d < n_features; ++d) {
            centroids[c][d] = Real(0.0);
        }
        owned_centroids.push_back(std::move(arr));
    }

    // Compute centroids
    for (Size i = 0; i < n_cells; ++i) {
        Index c = cluster_labels.ptr[i];
        const Index row_start = data.row_indices_unsafe()[i];
        const Index row_end = data.row_indices_unsafe()[i + 1];

        for (Index j = row_start; j < row_end; ++j) {
            Index feature = data.col_indices_unsafe()[j];
            centroids[c][feature] += static_cast<Real>(data.values()[j]);
        }
    }

    for (Index c = 0; c < n_clusters; ++c) {
        if (cluster_sizes[c] > 0) {
            for (Size d = 0; d < n_features; ++d) {
                centroids[c][d] /= static_cast<Real>(cluster_sizes[c]);
            }
        }
    }

    // Select representatives closest to centroid
    auto distances_ptr = scl::memory::aligned_alloc<Real>(n_cells, SCL_ALIGNMENT);
    Real* distances = distances_ptr.get();

    for (Index c = 0; c < n_clusters; ++c) {
        Size to_select = scl::algo::min2(per_cluster, cluster_sizes[c]);
        // Compute distances to centroid
        for (Size idx = 0; idx < cluster_sizes[c]; ++idx) {
            Index cell = cluster_indices[c][idx];
            Real dist = Real(0.0);

            const Index row_start = data.row_indices_unsafe()[cell];
            const Index row_end = data.row_indices_unsafe()[cell + 1];

            // Sparse distance to centroid
            for (Index j = row_start; j < row_end; ++j) {
                Index feature = data.col_indices_unsafe()[j];
                Real diff = static_cast<Real>(data.values()[j]) - centroids[c][feature];
                dist += diff * diff;
            }
            // Add contribution from zero entries (centroid non-zero)
            for (Size d = 0; d < n_features; ++d) {
                if (centroids[c][d] != Real(0.0)) {
                    bool found = false;
                    for (Index j = row_start; j < row_end; ++j) {
                        if (data.col_indices_unsafe()[j] == static_cast<Index>(d)) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        dist += centroids[c][d] * centroids[c][d];
                    }
                }
            }
            distances[idx] = dist;
        }

        // Select top-k closest using partial_sort with comparator
        auto sorted_ptr = scl::memory::aligned_alloc<Index>(cluster_sizes[c], SCL_ALIGNMENT);
        Index* sorted = sorted_ptr.get();
        for (Size idx = 0; idx < cluster_sizes[c]; ++idx) {
            sorted[idx] = static_cast<Index>(idx);
        }
        scl::algo::partial_sort(sorted, cluster_sizes[c], to_select,
            [&](Index a, Index b) { return distances[a] < distances[b]; });

        for (Size s = 0; s < to_select; ++s) {
            representatives[n_selected++] = cluster_indices[c][sorted[s]];
        }

        // sorted_ptr will free when out of scope
    }

    // Cleanup happens automatically via unique_ptrs (RAII)
    // (for types not owned via unique_ptr, we must still free)

    // Free cluster_sizes, cluster_counts, distances (single-level allocations)
    // cluster_indices_ptr, centroids_ptr (outer arrays) are handled by unique_ptr, 
    // but the internal arrays are in vectors owned_cluster_indices and owned_centroids
    // which RAII-frees.

    // No manual scl::memory::aligned_free needed.
}


// =============================================================================
// Balanced Sampling
// =============================================================================

void balanced_sampling(
    Array<const Index> labels,
    Size target_size,
    Index* selected_indices,
    Size& n_selected,
    uint64_t seed
) {
    const Size n = labels.len;

    n_selected = 0;
    if (n == 0 || target_size == 0) return;

    detail::FastRNG rng(seed);

    // Find number of groups
    Index n_groups = 0;
    for (Size i = 0; i < n; ++i) {
        if (labels.ptr[i] >= n_groups) {
            n_groups = labels.ptr[i] + 1;
        }
    }

    // Count elements per group
    auto group_sizes_ptr = scl::memory::aligned_alloc<Size>(n_groups, SCL_ALIGNMENT);
    Size* group_sizes = group_sizes_ptr.get();
    for (Index g = 0; g < n_groups; ++g) {
        group_sizes[g] = 0;
    }
    for (Size i = 0; i < n; ++i) {
        ++group_sizes[labels.ptr[i]];
    }

    // Collect indices per group arrays
    // Allocate array of Index* (for group_indices)
    auto group_indices_ptr = scl::memory::aligned_alloc<Index*>(n_groups, SCL_ALIGNMENT);
    Index** group_indices = group_indices_ptr.get();

    // Track ownership of each group_indices[g] so we can free later
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::vector<std::unique_ptr<Index[], scl::memory::AlignedDeleter<Index>>> owned_group_arrays;
    owned_group_arrays.reserve(static_cast<size_t>(n_groups));

    auto group_counts_ptr = scl::memory::aligned_alloc<Size>(n_groups, SCL_ALIGNMENT);
    Size* group_counts = group_counts_ptr.get();

    for (Index g = 0; g < n_groups; ++g) {
        auto array_ptr = scl::memory::aligned_alloc<Index>(group_sizes[g], SCL_ALIGNMENT);
        group_indices[g] = array_ptr.get();
        owned_group_arrays.push_back(std::move(array_ptr));
        group_counts[g] = 0;
    }

    for (Size i = 0; i < n; ++i) {
        Index g = labels.ptr[i];
        group_indices[g][group_counts[g]++] = static_cast<Index>(i);
    }

    // Calculate samples per group (balanced)
    Size non_empty_groups = 0;
    for (Index g = 0; g < n_groups; ++g) {
        if (group_sizes[g] > 0) ++non_empty_groups;
    }

    Size per_group = target_size / non_empty_groups;
    Size remainder = target_size % non_empty_groups;

    // Sample from each group
    for (Index g = 0; g < n_groups && n_selected < target_size; ++g) {
        if (group_sizes[g] == 0) continue;

        Size to_sample = per_group;
        if (remainder > 0) {
            ++to_sample;
            --remainder;
        }
        to_sample = scl::algo::min2(to_sample, group_sizes[g]);
        to_sample = scl::algo::min2(to_sample, target_size - n_selected);

        // Shuffle and take first to_sample
        for (Size i = 0; i < group_sizes[g]; ++i) {
            Size j = i + rng.next_index(group_sizes[g] - i);
            scl::algo::swap(group_indices[g][i], group_indices[g][j]);
        }

        for (Size s = 0; s < to_sample; ++s) {
            selected_indices[n_selected++] = group_indices[g][s];
        }
    }

    // Cleanup is automatic via unique_ptr's (owned_group_arrays, group_sizes_ptr, group_counts_ptr, group_indices_ptr)
}


// =============================================================================
// Additional sampling methods
// =============================================================================

// Stratified sampling based on continuous variable
void stratified_sampling(
    Array<const Real> values,
    Size n_strata,
    Size target_size,
    Index* selected_indices,
    Size& n_selected,
    uint64_t seed
) {
    const Size n = values.len;

    n_selected = 0;
    if (n == 0 || target_size == 0 || n_strata == 0) return;

    // Find min/max values
    Real min_val = values.ptr[0], max_val = values.ptr[0];
    for (Size i = 1; i < n; ++i) {
        if (values.ptr[i] < min_val) min_val = values.ptr[i];
        if (values.ptr[i] > max_val) max_val = values.ptr[i];
    }

    // Create strata
    Real range = max_val - min_val;
    Real stratum_width = range / static_cast<Real>(n_strata);
    if (stratum_width < config::EPSILON) stratum_width = Real(1.0);

    // Assign to strata
    auto strata_ptr = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    Index* strata = strata_ptr.release();
    for (Size i = 0; i < n; ++i) {
        auto s = static_cast<Index>((values.ptr[i] - min_val) / stratum_width);
        s = scl::algo::min2(s, static_cast<Index>(n_strata - 1));
        strata[i] = s;
    }

    // Use balanced sampling with strata as labels
    Array<const Index> strata_arr = {strata, n};
    balanced_sampling(strata_arr, target_size, selected_indices, n_selected, seed);

    scl::memory::aligned_free(strata);
}

// Uniform random sampling
void uniform_sampling(
    Size n,
    Size target_size,
    Index* selected_indices,
    Size& n_selected,
    uint64_t seed
) {
    n_selected = 0;
    if (n == 0 || target_size == 0) return;

    target_size = scl::algo::min2(target_size, n);

    detail::FastRNG rng(seed);

    // Fisher-Yates shuffle approach
    auto indices_ptr = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    Index* indices = indices_ptr.release();
    for (Size i = 0; i < n; ++i) {
        indices[i] = static_cast<Index>(i);
    }

    for (Size i = 0; i < target_size; ++i) {
        Size j = i + rng.next_index(n - i);
        scl::algo::swap(indices[i], indices[j]);
        selected_indices[n_selected++] = indices[i];
    }

    scl::memory::aligned_free(indices);
}

// Importance sampling based on weights
void importance_sampling(
    Array<const Real> weights,
    Size target_size,
    Index* selected_indices,
    Size& n_selected,
    uint64_t seed
) {
    const Size n = weights.len;

    n_selected = 0;
    if (n == 0 || target_size == 0) return;

    detail::FastRNG rng(seed);

    // Normalize weights
    Real total = Real(0.0);
    for (Size i = 0; i < n; ++i) {
        total += weights.ptr[i];
    }

    if (total < config::EPSILON) {
        uniform_sampling(n, target_size, selected_indices, n_selected, seed);
        return;
    }

    // Sample with replacement proportional to weights
    auto cumsum_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    Real* cumsum = cumsum_ptr.release();
    cumsum[0] = weights.ptr[0] / total;
    for (Size i = 1; i < n; ++i) {
        cumsum[i] = cumsum[i - 1] + weights.ptr[i] / total;
    }

    for (Size s = 0; s < target_size; ++s) {
        Real r = rng.uniform();

        // Binary search
        Size lo = 0, hi = n;
        while (lo < hi) {
            Size mid = (lo + hi) / 2;
            if (cumsum[mid] < r) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        selected_indices[n_selected++] = static_cast<Index>(lo);
    }

    scl::memory::aligned_free(cumsum);
}

// Reservoir sampling for streaming data
void reservoir_sampling(
    Size stream_size,
    Size reservoir_size,
    Index* reservoir,
    Size& n_selected,
    uint64_t seed
) {
    n_selected = 0;
    if (stream_size == 0 || reservoir_size == 0) return;

    reservoir_size = scl::algo::min2(reservoir_size, stream_size);

    detail::FastRNG rng(seed);

    // Fill reservoir with first k elements
    for (Size i = 0; i < reservoir_size; ++i) {
        reservoir[n_selected++] = static_cast<Index>(i);
    }

    // Process remaining elements
    for (Size i = reservoir_size; i < stream_size; ++i) {
        Size j = rng.next_index(i + 1);
        if (j < reservoir_size) {
            reservoir[j] = static_cast<Index>(i);
        }
    }
}

} // namespace scl::kernel::sampling
