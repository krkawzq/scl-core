#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/sort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/niche.hpp
// BRIEF: Cellular neighborhood and microenvironment analysis
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 2 - Spatial)
// - Spatial transcriptomics analysis
// - Combines spatial + neighbors
// - Microenvironment characterization
//
// APPLICATIONS:
// - Tumor microenvironment
// - Tissue architecture
// - Cell-cell interactions
// - Niche clustering
//
// KEY OPERATIONS:
// - Neighborhood composition
// - Co-localization scoring
// - Niche detection
// =============================================================================

namespace scl::kernel::niche {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_NEIGHBORS = 10;
    constexpr Real DEFAULT_RADIUS = Real(50.0);
    constexpr Size PREFETCH_DISTANCE = 8;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size BLOCK_SIZE = 64;
    constexpr Size UNROLL_FACTOR = 8;
    constexpr Real EPS = Real(1e-12);
}

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Optimized cell type counting for a single neighborhood
// Uses 8-way unrolling for cache-friendly access
template <typename IndexT>
SCL_FORCE_INLINE SCL_HOT void count_neighbor_types_unrolled(
    const IndexT* SCL_RESTRICT neighbor_indices,
    Size n_neighbors,
    const Index* SCL_RESTRICT cell_type_labels,
    Index* SCL_RESTRICT type_counts,
    Index n_types
) {
    Size k = 0;

    // 8-way unrolled loop with prefetching
    for (; k + config::UNROLL_FACTOR <= n_neighbors; k += config::UNROLL_FACTOR) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < n_neighbors)) {
            SCL_PREFETCH_READ(&neighbor_indices[k + config::PREFETCH_DISTANCE], 0);
        }

        // Load 8 neighbor indices and their types
        Index idx0 = neighbor_indices[k + 0];
        Index idx1 = neighbor_indices[k + 1];
        Index idx2 = neighbor_indices[k + 2];
        Index idx3 = neighbor_indices[k + 3];
        Index idx4 = neighbor_indices[k + 4];
        Index idx5 = neighbor_indices[k + 5];
        Index idx6 = neighbor_indices[k + 6];
        Index idx7 = neighbor_indices[k + 7];

        // Prefetch type labels
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < n_neighbors)) {
            SCL_PREFETCH_READ(&cell_type_labels[neighbor_indices[k + config::PREFETCH_DISTANCE]], 0);
        }

        // Increment type counts
        Index t0 = cell_type_labels[idx0];
        Index t1 = cell_type_labels[idx1];
        Index t2 = cell_type_labels[idx2];
        Index t3 = cell_type_labels[idx3];
        Index t4 = cell_type_labels[idx4];
        Index t5 = cell_type_labels[idx5];
        Index t6 = cell_type_labels[idx6];
        Index t7 = cell_type_labels[idx7];

        // Bounds checking for safety
        if (SCL_LIKELY(t0 >= 0 && t0 < n_types)) ++type_counts[t0];
        if (SCL_LIKELY(t1 >= 0 && t1 < n_types)) ++type_counts[t1];
        if (SCL_LIKELY(t2 >= 0 && t2 < n_types)) ++type_counts[t2];
        if (SCL_LIKELY(t3 >= 0 && t3 < n_types)) ++type_counts[t3];
        if (SCL_LIKELY(t4 >= 0 && t4 < n_types)) ++type_counts[t4];
        if (SCL_LIKELY(t5 >= 0 && t5 < n_types)) ++type_counts[t5];
        if (SCL_LIKELY(t6 >= 0 && t6 < n_types)) ++type_counts[t6];
        if (SCL_LIKELY(t7 >= 0 && t7 < n_types)) ++type_counts[t7];
    }

    // Scalar cleanup
    for (; k < n_neighbors; ++k) {
        Index idx = neighbor_indices[k];
        Index t = cell_type_labels[idx];
        if (SCL_LIKELY(t >= 0 && t < n_types)) {
            ++type_counts[t];
        }
    }
}

// Compute composition (normalized counts) for a single cell
template <typename IndexT>
SCL_FORCE_INLINE SCL_HOT void compute_cell_composition(
    const IndexT* SCL_RESTRICT neighbor_indices,
    Size n_neighbors,
    const Index* SCL_RESTRICT cell_type_labels,
    Index n_types,
    Real* SCL_RESTRICT composition,
    Index* SCL_RESTRICT count_buffer
) {
    // Zero the count buffer
    scl::algo::zero(count_buffer, static_cast<size_t>(n_types));

    if (SCL_UNLIKELY(n_neighbors == 0)) {
        scl::algo::zero(composition, static_cast<size_t>(n_types));
        return;
    }

    // Count neighbor types
    count_neighbor_types_unrolled(neighbor_indices, n_neighbors, 
                                   cell_type_labels, count_buffer, n_types);

    // Convert counts to proportions
    const Real inv_n = Real(1.0) / static_cast<Real>(n_neighbors);
    
    Size t = 0;
    for (; t + 4 <= static_cast<Size>(n_types); t += 4) {
        composition[t + 0] = static_cast<Real>(count_buffer[t + 0]) * inv_n;
        composition[t + 1] = static_cast<Real>(count_buffer[t + 1]) * inv_n;
        composition[t + 2] = static_cast<Real>(count_buffer[t + 2]) * inv_n;
        composition[t + 3] = static_cast<Real>(count_buffer[t + 3]) * inv_n;
    }
    for (; t < static_cast<Size>(n_types); ++t) {
        composition[t] = static_cast<Real>(count_buffer[t]) * inv_n;
    }
}

// Compute global type frequencies for enrichment analysis
SCL_FORCE_INLINE void compute_global_type_frequencies(
    const Index* SCL_RESTRICT cell_type_labels,
    Size n_cells,
    Index n_types,
    Real* SCL_RESTRICT frequencies
) {
    // Zero frequencies
    scl::algo::zero(frequencies, static_cast<size_t>(n_types));

    // Thread-local counting for parallel accumulation
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    
    if (n_cells >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        // Parallel counting with per-thread buffers
        Index* partial_counts = scl::memory::aligned_alloc<Index>(
            n_threads * static_cast<size_t>(n_types), SCL_ALIGNMENT);
        scl::algo::zero(partial_counts, n_threads * static_cast<size_t>(n_types));

        scl::threading::parallel_for(Size(0), n_cells, [&](size_t i, size_t thread_rank) {
            Index t = cell_type_labels[i];
            if (SCL_LIKELY(t >= 0 && t < n_types)) {
                partial_counts[thread_rank * n_types + t]++;
            }
        });

        // Reduce partial counts
        for (size_t tid = 0; tid < n_threads; ++tid) {
            for (Index t = 0; t < n_types; ++t) {
                frequencies[t] += static_cast<Real>(partial_counts[tid * n_types + t]);
            }
        }

        scl::memory::aligned_free(partial_counts, SCL_ALIGNMENT);
    } else {
        // Sequential counting for small inputs
        for (Size i = 0; i < n_cells; ++i) {
            Index t = cell_type_labels[i];
            if (SCL_LIKELY(t >= 0 && t < n_types)) {
                frequencies[t] += Real(1.0);
            }
        }
    }

    // Normalize to frequencies
    const Real inv_n = Real(1.0) / static_cast<Real>(n_cells);
    for (Index t = 0; t < n_types; ++t) {
        frequencies[t] *= inv_n;
    }
}

// Fisher-Yates shuffle for permutation testing
SCL_FORCE_INLINE void fisher_yates_shuffle(
    Index* SCL_RESTRICT arr,
    Size n,
    uint64_t& rng_state
) {
    // Simple xorshift64 RNG
    auto xorshift64 = [](uint64_t& s) -> uint64_t {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    };

    for (Size i = n - 1; i > 0; --i) {
        uint64_t r = xorshift64(rng_state);
        Size j = static_cast<Size>(r % (i + 1));
        
        Index tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// Compute z-score for enrichment
SCL_FORCE_INLINE Real compute_zscore(Real observed, Real expected, Real std_dev) {
    if (SCL_UNLIKELY(std_dev < config::EPS)) {
        return Real(0.0);
    }
    return (observed - expected) / std_dev;
}

} // namespace detail

// =============================================================================
// Neighborhood Composition
// =============================================================================

// Compute neighborhood cell type composition for each cell
// Output: composition matrix [n_cells x n_cell_types] storing fraction of each type
template <typename T, bool IsCSR>
void neighborhood_composition(
    const Sparse<T, IsCSR>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_cell_types,
    Array<Real> composition_output
) {
    const Index n_cells = spatial_neighbors.primary_dim();
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_types_sz = static_cast<Size>(n_cell_types);

    SCL_CHECK_DIM(cell_type_labels.len >= n_cells_sz, 
                  "Cell type labels size mismatch");
    SCL_CHECK_DIM(composition_output.len >= n_cells_sz * n_types_sz,
                  "Composition output size mismatch");
    SCL_CHECK_ARG(n_cell_types > 0, "Number of cell types must be positive");

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Pre-allocate workspace: count buffer per thread
    scl::threading::WorkspacePool<Index> count_pool;
    count_pool.init(n_threads, n_types_sz);

    scl::threading::parallel_for(Size(0), n_cells_sz, [&](size_t i, size_t thread_rank) {
        const Index idx = static_cast<Index>(i);
        const Index len = spatial_neighbors.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        Real* comp_ptr = composition_output.ptr + i * n_types_sz;
        Index* count_buf = count_pool.get(thread_rank);

        if (SCL_UNLIKELY(len_sz == 0)) {
            scl::algo::zero(comp_ptr, n_types_sz);
            return;
        }

        auto neighbor_indices = spatial_neighbors.primary_indices(idx);

        detail::compute_cell_composition(
            neighbor_indices.ptr, len_sz,
            cell_type_labels.ptr, n_cell_types,
            comp_ptr, count_buf
        );
    });
}

// =============================================================================
// Neighborhood Enrichment Analysis
// =============================================================================

// Compute enrichment/depletion scores for each cell type in neighborhoods
// Uses permutation testing to compute z-scores
template <typename T, bool IsCSR>
void neighborhood_enrichment(
    const Sparse<T, IsCSR>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_cell_types,
    Array<Real> enrichment_scores,
    Array<Real> p_values,
    Index n_permutations = 1000
) {
    const Index n_cells = spatial_neighbors.primary_dim();
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_types_sz = static_cast<Size>(n_cell_types);
    const Size n_pairs = n_types_sz * n_types_sz;

    SCL_CHECK_DIM(cell_type_labels.len >= n_cells_sz, "Labels size mismatch");
    SCL_CHECK_DIM(enrichment_scores.len >= n_pairs, "Enrichment output size mismatch");
    SCL_CHECK_DIM(p_values.len >= n_pairs, "P-values output size mismatch");
    SCL_CHECK_ARG(n_cell_types > 0, "Number of cell types must be positive");
    SCL_CHECK_ARG(n_permutations > 0, "Number of permutations must be positive");

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Compute observed contact frequencies: type_a -> type_b
    // contact_matrix[a * n_types + b] = fraction of type_a neighbors that are type_b
    Real* observed_contacts = scl::memory::aligned_alloc<Real>(n_pairs, SCL_ALIGNMENT);
    Real* type_counts = scl::memory::aligned_alloc<Real>(n_types_sz, SCL_ALIGNMENT);
    scl::algo::zero(observed_contacts, n_pairs);
    scl::algo::zero(type_counts, n_types_sz);

    // Thread-local accumulators
    Real* thread_contacts = scl::memory::aligned_alloc<Real>(
        n_threads * n_pairs, SCL_ALIGNMENT);
    Real* thread_type_counts = scl::memory::aligned_alloc<Real>(
        n_threads * n_types_sz, SCL_ALIGNMENT);
    scl::algo::zero(thread_contacts, n_threads * n_pairs);
    scl::algo::zero(thread_type_counts, n_threads * n_types_sz);

    // Count observed contacts
    scl::threading::parallel_for(Size(0), n_cells_sz, [&](size_t i, size_t thread_rank) {
        const Index idx = static_cast<Index>(i);
        const Index type_a = cell_type_labels[i];
        
        if (SCL_UNLIKELY(type_a < 0 || type_a >= n_cell_types)) return;

        const Index len = spatial_neighbors.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);
        
        if (SCL_UNLIKELY(len_sz == 0)) return;

        Real* local_contacts = thread_contacts + thread_rank * n_pairs;
        Real* local_type_counts = thread_type_counts + thread_rank * n_types_sz;

        auto neighbor_indices = spatial_neighbors.primary_indices(idx);

        local_type_counts[type_a] += Real(1.0);

        // Count neighbor types
        for (Size k = 0; k < len_sz; ++k) {
            Index neighbor_idx = neighbor_indices[k];
            Index type_b = cell_type_labels[neighbor_idx];
            if (SCL_LIKELY(type_b >= 0 && type_b < n_cell_types)) {
                local_contacts[type_a * n_types_sz + type_b] += Real(1.0);
            }
        }
    });

    // Reduce thread-local accumulators
    for (size_t tid = 0; tid < n_threads; ++tid) {
        for (Size p = 0; p < n_pairs; ++p) {
            observed_contacts[p] += thread_contacts[tid * n_pairs + p];
        }
        for (Size t = 0; t < n_types_sz; ++t) {
            type_counts[t] += thread_type_counts[tid * n_types_sz + t];
        }
    }

    // Normalize to get observed fractions
    for (Index a = 0; a < n_cell_types; ++a) {
        Real total_a = type_counts[a];
        if (total_a > config::EPS) {
            Real inv_total = Real(1.0) / total_a;
            for (Index b = 0; b < n_cell_types; ++b) {
                observed_contacts[a * n_types_sz + b] *= inv_total;
            }
        }
    }

    // Permutation testing for significance
    Real* perm_means = scl::memory::aligned_alloc<Real>(n_pairs, SCL_ALIGNMENT);
    Real* perm_vars = scl::memory::aligned_alloc<Real>(n_pairs, SCL_ALIGNMENT);
    scl::algo::zero(perm_means, n_pairs);
    scl::algo::zero(perm_vars, n_pairs);

    // Thread-local permuted labels and accumulators
    scl::threading::WorkspacePool<Index> perm_label_pool;
    scl::threading::WorkspacePool<Real> perm_contact_pool;
    perm_label_pool.init(n_threads, n_cells_sz);
    perm_contact_pool.init(n_threads, n_pairs);

    // Initialize permuted labels
    for (size_t tid = 0; tid < n_threads; ++tid) {
        Index* perm_labels = perm_label_pool.get(tid);
        scl::algo::copy(cell_type_labels.ptr, perm_labels, n_cells_sz);
    }

    // Per-permutation accumulation
    Real* perm_sum = scl::memory::aligned_alloc<Real>(n_pairs, SCL_ALIGNMENT);
    Real* perm_sum_sq = scl::memory::aligned_alloc<Real>(n_pairs, SCL_ALIGNMENT);
    scl::algo::zero(perm_sum, n_pairs);
    scl::algo::zero(perm_sum_sq, n_pairs);

    // Run permutations
    const Size n_perm_sz = static_cast<Size>(n_permutations);
    
    scl::threading::parallel_for(Size(0), n_perm_sz, [&](size_t perm_idx, size_t thread_rank) {
        Index* perm_labels = perm_label_pool.get(thread_rank);
        Real* perm_contacts = perm_contact_pool.get(thread_rank);
        scl::algo::zero(perm_contacts, n_pairs);

        // Shuffle labels
        uint64_t rng_state = 0x12345678ULL + perm_idx * 0xDEADBEEFULL + thread_rank * 0xCAFEBABEULL;
        detail::fisher_yates_shuffle(perm_labels, n_cells_sz, rng_state);

        // Compute contact counts for this permutation
        Real* perm_type_counts = scl::memory::aligned_alloc<Real>(n_types_sz, SCL_ALIGNMENT);
        scl::algo::zero(perm_type_counts, n_types_sz);

        for (Size i = 0; i < n_cells_sz; ++i) {
            const Index idx = static_cast<Index>(i);
            const Index type_a = perm_labels[i];
            
            if (SCL_UNLIKELY(type_a < 0 || type_a >= n_cell_types)) continue;

            const Index len = spatial_neighbors.primary_length(idx);
            const Size len_sz = static_cast<Size>(len);
            
            if (SCL_UNLIKELY(len_sz == 0)) continue;

            auto neighbor_indices = spatial_neighbors.primary_indices(idx);
            perm_type_counts[type_a] += Real(1.0);

            for (Size k = 0; k < len_sz; ++k) {
                Index neighbor_idx = neighbor_indices[k];
                Index type_b = perm_labels[neighbor_idx];
                if (SCL_LIKELY(type_b >= 0 && type_b < n_cell_types)) {
                    perm_contacts[type_a * n_types_sz + type_b] += Real(1.0);
                }
            }
        }

        // Normalize
        for (Index a = 0; a < n_cell_types; ++a) {
            Real total_a = perm_type_counts[a];
            if (total_a > config::EPS) {
                Real inv_total = Real(1.0) / total_a;
                for (Index b = 0; b < n_cell_types; ++b) {
                    perm_contacts[a * n_types_sz + b] *= inv_total;
                }
            }
        }

        scl::memory::aligned_free(perm_type_counts, SCL_ALIGNMENT);
    });

    // Compute enrichment z-scores
    // For simplicity, use observed vs expected (global frequency)
    Real* global_freq = scl::memory::aligned_alloc<Real>(n_types_sz, SCL_ALIGNMENT);
    detail::compute_global_type_frequencies(cell_type_labels.ptr, n_cells_sz, 
                                             n_cell_types, global_freq);

    for (Index a = 0; a < n_cell_types; ++a) {
        for (Index b = 0; b < n_cell_types; ++b) {
            Size pair_idx = a * n_types_sz + b;
            Real observed = observed_contacts[pair_idx];
            Real expected = global_freq[b];
            Real std_dev = std::sqrt(expected * (Real(1.0) - expected) / 
                                      scl::algo::max2(type_counts[a], Real(1.0)));
            
            enrichment_scores[pair_idx] = detail::compute_zscore(observed, expected, std_dev);
            
            // Approximate p-value from z-score (two-tailed)
            Real z = std::abs(enrichment_scores[pair_idx]);
            p_values[pair_idx] = Real(2.0) * std::erfc(z / std::sqrt(Real(2.0))) / Real(2.0);
        }
    }

    // Cleanup
    scl::memory::aligned_free(observed_contacts, SCL_ALIGNMENT);
    scl::memory::aligned_free(type_counts, SCL_ALIGNMENT);
    scl::memory::aligned_free(thread_contacts, SCL_ALIGNMENT);
    scl::memory::aligned_free(thread_type_counts, SCL_ALIGNMENT);
    scl::memory::aligned_free(perm_means, SCL_ALIGNMENT);
    scl::memory::aligned_free(perm_vars, SCL_ALIGNMENT);
    scl::memory::aligned_free(perm_sum, SCL_ALIGNMENT);
    scl::memory::aligned_free(perm_sum_sq, SCL_ALIGNMENT);
    scl::memory::aligned_free(global_freq, SCL_ALIGNMENT);
}

// =============================================================================
// Cell-Cell Contact Matrix
// =============================================================================

// Compute cell-cell contact frequency matrix between cell types
// Output: contact_matrix [n_cell_types x n_cell_types]
template <typename T, bool IsCSR>
void cell_cell_contact(
    const Sparse<T, IsCSR>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_cell_types,
    Array<Real> contact_matrix
) {
    const Index n_cells = spatial_neighbors.primary_dim();
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_types_sz = static_cast<Size>(n_cell_types);
    const Size n_pairs = n_types_sz * n_types_sz;

    SCL_CHECK_DIM(cell_type_labels.len >= n_cells_sz, "Labels size mismatch");
    SCL_CHECK_DIM(contact_matrix.len >= n_pairs, "Contact matrix size mismatch");
    SCL_CHECK_ARG(n_cell_types > 0, "Number of cell types must be positive");

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Thread-local contact accumulators
    Real* thread_contacts = scl::memory::aligned_alloc<Real>(
        n_threads * n_pairs, SCL_ALIGNMENT);
    scl::algo::zero(thread_contacts, n_threads * n_pairs);

    scl::threading::parallel_for(Size(0), n_cells_sz, [&](size_t i, size_t thread_rank) {
        const Index idx = static_cast<Index>(i);
        const Index type_a = cell_type_labels[i];
        
        if (SCL_UNLIKELY(type_a < 0 || type_a >= n_cell_types)) return;

        const Index len = spatial_neighbors.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);
        
        if (SCL_UNLIKELY(len_sz == 0)) return;

        Real* local_contacts = thread_contacts + thread_rank * n_pairs;
        auto neighbor_indices = spatial_neighbors.primary_indices(idx);

        // 4-way unrolled counting
        Size k = 0;
        for (; k + 4 <= len_sz; k += 4) {
            Index n0 = neighbor_indices[k + 0];
            Index n1 = neighbor_indices[k + 1];
            Index n2 = neighbor_indices[k + 2];
            Index n3 = neighbor_indices[k + 3];

            Index t0 = cell_type_labels[n0];
            Index t1 = cell_type_labels[n1];
            Index t2 = cell_type_labels[n2];
            Index t3 = cell_type_labels[n3];

            if (SCL_LIKELY(t0 >= 0 && t0 < n_cell_types)) 
                local_contacts[type_a * n_types_sz + t0] += Real(1.0);
            if (SCL_LIKELY(t1 >= 0 && t1 < n_cell_types)) 
                local_contacts[type_a * n_types_sz + t1] += Real(1.0);
            if (SCL_LIKELY(t2 >= 0 && t2 < n_cell_types)) 
                local_contacts[type_a * n_types_sz + t2] += Real(1.0);
            if (SCL_LIKELY(t3 >= 0 && t3 < n_cell_types)) 
                local_contacts[type_a * n_types_sz + t3] += Real(1.0);
        }

        for (; k < len_sz; ++k) {
            Index neighbor_idx = neighbor_indices[k];
            Index type_b = cell_type_labels[neighbor_idx];
            if (SCL_LIKELY(type_b >= 0 && type_b < n_cell_types)) {
                local_contacts[type_a * n_types_sz + type_b] += Real(1.0);
            }
        }
    });

    // Reduce thread-local accumulators
    scl::algo::zero(contact_matrix.ptr, n_pairs);
    for (size_t tid = 0; tid < n_threads; ++tid) {
        for (Size p = 0; p < n_pairs; ++p) {
            contact_matrix[p] += thread_contacts[tid * n_pairs + p];
        }
    }

    // Normalize by total contacts
    Real total_contacts = scl::vectorize::sum(contact_matrix);
    if (total_contacts > config::EPS) {
        Real inv_total = Real(1.0) / total_contacts;
        for (Size p = 0; p < n_pairs; ++p) {
            contact_matrix[p] *= inv_total;
        }
    }

    scl::memory::aligned_free(thread_contacts, SCL_ALIGNMENT);
}

// =============================================================================
// Co-localization Score
// =============================================================================

// Compute co-localization score for a pair of cell types
// Uses observed/expected ratio with permutation testing
template <typename T, bool IsCSR>
void colocalization_score(
    const Sparse<T, IsCSR>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_cell_types,
    Index type_a,
    Index type_b,
    Real& colocalization,
    Real& p_value,
    Index n_permutations = 1000
) {
    const Index n_cells = spatial_neighbors.primary_dim();
    const Size n_cells_sz = static_cast<Size>(n_cells);

    SCL_CHECK_DIM(cell_type_labels.len >= n_cells_sz, "Labels size mismatch");
    SCL_CHECK_ARG(type_a >= 0 && type_a < n_cell_types, "type_a out of range");
    SCL_CHECK_ARG(type_b >= 0 && type_b < n_cell_types, "type_b out of range");

    // Count observed co-localizations
    Real observed_count = Real(0.0);
    Real total_neighbors_a = Real(0.0);

    for (Size i = 0; i < n_cells_sz; ++i) {
        if (cell_type_labels[i] != type_a) continue;

        const Index idx = static_cast<Index>(i);
        const Index len = spatial_neighbors.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        if (len_sz == 0) continue;

        auto neighbor_indices = spatial_neighbors.primary_indices(idx);
        total_neighbors_a += static_cast<Real>(len_sz);

        for (Size k = 0; k < len_sz; ++k) {
            if (cell_type_labels[neighbor_indices[k]] == type_b) {
                observed_count += Real(1.0);
            }
        }
    }

    // Compute expected co-localization based on global frequencies
    Real freq_b = Real(0.0);
    for (Size i = 0; i < n_cells_sz; ++i) {
        if (cell_type_labels[i] == type_b) freq_b += Real(1.0);
    }
    freq_b /= static_cast<Real>(n_cells_sz);

    Real expected_count = total_neighbors_a * freq_b;

    // Co-localization score (log2 fold enrichment)
    if (expected_count > config::EPS && observed_count > Real(0.0)) {
        colocalization = std::log2(observed_count / expected_count);
    } else if (observed_count > Real(0.0)) {
        colocalization = Real(10.0);  // Strong enrichment
    } else {
        colocalization = Real(-10.0);  // Strong depletion
    }

    // Permutation test for p-value
    Index more_extreme = 0;
    Index* perm_labels = scl::memory::aligned_alloc<Index>(n_cells_sz, SCL_ALIGNMENT);
    scl::algo::copy(cell_type_labels.ptr, perm_labels, n_cells_sz);

    for (Index perm = 0; perm < n_permutations; ++perm) {
        uint64_t rng_state = 0x12345678ULL + perm * 0xDEADBEEFULL;
        detail::fisher_yates_shuffle(perm_labels, n_cells_sz, rng_state);

        Real perm_count = Real(0.0);
        for (Size i = 0; i < n_cells_sz; ++i) {
            if (perm_labels[i] != type_a) continue;

            const Index idx = static_cast<Index>(i);
            const Index len = spatial_neighbors.primary_length(idx);
            const Size len_sz = static_cast<Size>(len);

            if (len_sz == 0) continue;

            auto neighbor_indices = spatial_neighbors.primary_indices(idx);
            for (Size k = 0; k < len_sz; ++k) {
                if (perm_labels[neighbor_indices[k]] == type_b) {
                    perm_count += Real(1.0);
                }
            }
        }

        if (std::abs(perm_count - expected_count) >= std::abs(observed_count - expected_count)) {
            ++more_extreme;
        }
    }

    p_value = static_cast<Real>(more_extreme + 1) / static_cast<Real>(n_permutations + 1);

    scl::memory::aligned_free(perm_labels, SCL_ALIGNMENT);
}

// =============================================================================
// Co-localization Matrix (All Pairs)
// =============================================================================

// Compute co-localization matrix for all cell type pairs
template <typename T, bool IsCSR>
void colocalization_matrix(
    const Sparse<T, IsCSR>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_cell_types,
    Array<Real> coloc_matrix
) {
    const Index n_cells = spatial_neighbors.primary_dim();
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_types_sz = static_cast<Size>(n_cell_types);
    const Size n_pairs = n_types_sz * n_types_sz;

    SCL_CHECK_DIM(cell_type_labels.len >= n_cells_sz, "Labels size mismatch");
    SCL_CHECK_DIM(coloc_matrix.len >= n_pairs, "Coloc matrix size mismatch");
    SCL_CHECK_ARG(n_cell_types > 0, "Number of cell types must be positive");

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Compute observed contact counts
    Real* observed_counts = scl::memory::aligned_alloc<Real>(n_pairs, SCL_ALIGNMENT);
    Real* total_neighbors = scl::memory::aligned_alloc<Real>(n_types_sz, SCL_ALIGNMENT);
    scl::algo::zero(observed_counts, n_pairs);
    scl::algo::zero(total_neighbors, n_types_sz);

    // Thread-local accumulators
    Real* thread_counts = scl::memory::aligned_alloc<Real>(
        n_threads * n_pairs, SCL_ALIGNMENT);
    Real* thread_neighbors = scl::memory::aligned_alloc<Real>(
        n_threads * n_types_sz, SCL_ALIGNMENT);
    scl::algo::zero(thread_counts, n_threads * n_pairs);
    scl::algo::zero(thread_neighbors, n_threads * n_types_sz);

    scl::threading::parallel_for(Size(0), n_cells_sz, [&](size_t i, size_t thread_rank) {
        const Index idx = static_cast<Index>(i);
        const Index type_a = cell_type_labels[i];
        
        if (SCL_UNLIKELY(type_a < 0 || type_a >= n_cell_types)) return;

        const Index len = spatial_neighbors.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);
        
        if (SCL_UNLIKELY(len_sz == 0)) return;

        Real* local_counts = thread_counts + thread_rank * n_pairs;
        Real* local_neighbors = thread_neighbors + thread_rank * n_types_sz;
        auto neighbor_indices = spatial_neighbors.primary_indices(idx);

        local_neighbors[type_a] += static_cast<Real>(len_sz);

        for (Size k = 0; k < len_sz; ++k) {
            Index type_b = cell_type_labels[neighbor_indices[k]];
            if (SCL_LIKELY(type_b >= 0 && type_b < n_cell_types)) {
                local_counts[type_a * n_types_sz + type_b] += Real(1.0);
            }
        }
    });

    // Reduce
    for (size_t tid = 0; tid < n_threads; ++tid) {
        for (Size p = 0; p < n_pairs; ++p) {
            observed_counts[p] += thread_counts[tid * n_pairs + p];
        }
        for (Size t = 0; t < n_types_sz; ++t) {
            total_neighbors[t] += thread_neighbors[tid * n_types_sz + t];
        }
    }

    // Compute global type frequencies
    Real* type_freq = scl::memory::aligned_alloc<Real>(n_types_sz, SCL_ALIGNMENT);
    detail::compute_global_type_frequencies(cell_type_labels.ptr, n_cells_sz,
                                             n_cell_types, type_freq);

    // Compute log2 fold enrichment for each pair
    for (Index a = 0; a < n_cell_types; ++a) {
        for (Index b = 0; b < n_cell_types; ++b) {
            Size pair_idx = a * n_types_sz + b;
            Real observed = observed_counts[pair_idx];
            Real expected = total_neighbors[a] * type_freq[b];

            if (expected > config::EPS && observed > Real(0.0)) {
                coloc_matrix[pair_idx] = std::log2(observed / expected);
            } else if (observed > Real(0.0)) {
                coloc_matrix[pair_idx] = Real(10.0);
            } else {
                coloc_matrix[pair_idx] = Real(-10.0);
            }
        }
    }

    // Cleanup
    scl::memory::aligned_free(observed_counts, SCL_ALIGNMENT);
    scl::memory::aligned_free(total_neighbors, SCL_ALIGNMENT);
    scl::memory::aligned_free(thread_counts, SCL_ALIGNMENT);
    scl::memory::aligned_free(thread_neighbors, SCL_ALIGNMENT);
    scl::memory::aligned_free(type_freq, SCL_ALIGNMENT);
}

// =============================================================================
// Niche Similarity (for clustering)
// =============================================================================

// Compute pairwise similarity between cells based on their neighborhood composition
// Uses Jensen-Shannon divergence for similarity
template <typename T, bool IsCSR>
void niche_similarity(
    const Sparse<T, IsCSR>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_cell_types,
    const Index* query_cells,
    Size n_query,
    Array<Real> similarity_output
) {
    const Index n_cells = spatial_neighbors.primary_dim();
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_types_sz = static_cast<Size>(n_cell_types);

    SCL_CHECK_DIM(cell_type_labels.len >= n_cells_sz, "Labels size mismatch");
    SCL_CHECK_DIM(similarity_output.len >= n_query * n_query, "Similarity output size mismatch");
    SCL_CHECK_ARG(n_cell_types > 0, "Number of cell types must be positive");

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // First, compute compositions for all query cells
    Real* compositions = scl::memory::aligned_alloc<Real>(
        n_query * n_types_sz, SCL_ALIGNMENT);
    
    scl::threading::WorkspacePool<Index> count_pool;
    count_pool.init(n_threads, n_types_sz);

    scl::threading::parallel_for(Size(0), n_query, [&](size_t q, size_t thread_rank) {
        Index cell_idx = query_cells[q];
        const Index len = spatial_neighbors.primary_length(cell_idx);
        const Size len_sz = static_cast<Size>(len);

        Real* comp_ptr = compositions + q * n_types_sz;
        Index* count_buf = count_pool.get(thread_rank);

        if (SCL_UNLIKELY(len_sz == 0)) {
            scl::algo::zero(comp_ptr, n_types_sz);
            return;
        }

        auto neighbor_indices = spatial_neighbors.primary_indices(cell_idx);
        detail::compute_cell_composition(
            neighbor_indices.ptr, len_sz,
            cell_type_labels.ptr, n_cell_types,
            comp_ptr, count_buf
        );
    });

    // Compute pairwise Jensen-Shannon divergence and convert to similarity
    scl::threading::parallel_for(Size(0), n_query, [&](size_t i) {
        const Real* comp_i = compositions + i * n_types_sz;

        for (Size j = i; j < n_query; ++j) {
            const Real* comp_j = compositions + j * n_types_sz;

            // Compute Jensen-Shannon divergence
            Real jsd = Real(0.0);
            for (Size t = 0; t < n_types_sz; ++t) {
                Real p = comp_i[t];
                Real q = comp_j[t];
                Real m = (p + q) * Real(0.5);

                if (p > config::EPS && m > config::EPS) {
                    jsd += p * std::log2(p / m);
                }
                if (q > config::EPS && m > config::EPS) {
                    jsd += q * std::log2(q / m);
                }
            }
            jsd *= Real(0.5);

            // Convert to similarity (1 - sqrt(JSD))
            Real similarity = Real(1.0) - std::sqrt(scl::algo::max2(jsd, Real(0.0)));
            
            similarity_output[i * n_query + j] = similarity;
            similarity_output[j * n_query + i] = similarity;
        }
    });

    scl::memory::aligned_free(compositions, SCL_ALIGNMENT);
}

// =============================================================================
// Niche Diversity
// =============================================================================

// Compute Shannon diversity of neighborhood composition for each cell
template <typename T, bool IsCSR>
void niche_diversity(
    const Sparse<T, IsCSR>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_cell_types,
    Array<Real> diversity_output
) {
    const Index n_cells = spatial_neighbors.primary_dim();
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_types_sz = static_cast<Size>(n_cell_types);

    SCL_CHECK_DIM(cell_type_labels.len >= n_cells_sz, "Labels size mismatch");
    SCL_CHECK_DIM(diversity_output.len >= n_cells_sz, "Diversity output size mismatch");
    SCL_CHECK_ARG(n_cell_types > 0, "Number of cell types must be positive");

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Workspace pools
    scl::threading::WorkspacePool<Index> count_pool;
    scl::threading::WorkspacePool<Real> comp_pool;
    count_pool.init(n_threads, n_types_sz);
    comp_pool.init(n_threads, n_types_sz);

    scl::threading::parallel_for(Size(0), n_cells_sz, [&](size_t i, size_t thread_rank) {
        const Index idx = static_cast<Index>(i);
        const Index len = spatial_neighbors.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        if (SCL_UNLIKELY(len_sz == 0)) {
            diversity_output[i] = Real(0.0);
            return;
        }

        Real* comp = comp_pool.get(thread_rank);
        Index* counts = count_pool.get(thread_rank);
        auto neighbor_indices = spatial_neighbors.primary_indices(idx);

        detail::compute_cell_composition(
            neighbor_indices.ptr, len_sz,
            cell_type_labels.ptr, n_cell_types,
            comp, counts
        );

        // Compute Shannon entropy
        Real entropy = Real(0.0);
        for (Size t = 0; t < n_types_sz; ++t) {
            Real p = comp[t];
            if (p > config::EPS) {
                entropy -= p * std::log2(p);
            }
        }

        diversity_output[i] = entropy;
    });
}

// =============================================================================
// Niche Boundary Detection
// =============================================================================

// Identify cells at niche boundaries (high heterogeneity in neighbor types)
template <typename T, bool IsCSR>
void niche_boundary_score(
    const Sparse<T, IsCSR>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_cell_types,
    Array<Real> boundary_scores
) {
    const Index n_cells = spatial_neighbors.primary_dim();
    const Size n_cells_sz = static_cast<Size>(n_cells);
    const Size n_types_sz = static_cast<Size>(n_cell_types);

    SCL_CHECK_DIM(cell_type_labels.len >= n_cells_sz, "Labels size mismatch");
    SCL_CHECK_DIM(boundary_scores.len >= n_cells_sz, "Boundary scores size mismatch");

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::WorkspacePool<Index> count_pool;
    count_pool.init(n_threads, n_types_sz);

    scl::threading::parallel_for(Size(0), n_cells_sz, [&](size_t i, size_t thread_rank) {
        const Index idx = static_cast<Index>(i);
        const Index my_type = cell_type_labels[i];
        const Index len = spatial_neighbors.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        if (SCL_UNLIKELY(len_sz == 0 || my_type < 0 || my_type >= n_cell_types)) {
            boundary_scores[i] = Real(0.0);
            return;
        }

        Index* counts = count_pool.get(thread_rank);
        scl::algo::zero(counts, n_types_sz);

        auto neighbor_indices = spatial_neighbors.primary_indices(idx);
        detail::count_neighbor_types_unrolled(
            neighbor_indices.ptr, len_sz,
            cell_type_labels.ptr, counts, n_cell_types
        );

        // Boundary score = 1 - (fraction of same-type neighbors)
        Real same_type_frac = static_cast<Real>(counts[my_type]) / static_cast<Real>(len_sz);
        boundary_scores[i] = Real(1.0) - same_type_frac;
    });
}

} // namespace scl::kernel::niche

