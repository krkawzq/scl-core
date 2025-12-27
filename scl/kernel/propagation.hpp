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
// FILE: scl/kernel/propagation.hpp
// BRIEF: Label propagation for semi-supervised learning on graphs
// =============================================================================

namespace scl::kernel::propagation {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_ALPHA = Real(0.99);
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real DEFAULT_TOLERANCE = Real(1e-6);
    constexpr Index UNLABELED = -1;
    constexpr Size PARALLEL_THRESHOLD = 500;
}

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Simple PRNG for random node order
struct FastRNG {
    uint64_t state;

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept : state(seed) {}

    SCL_FORCE_INLINE uint64_t next() noexcept {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }

    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        return static_cast<Size>(next() % static_cast<uint64_t>(n));
    }
};

// Fisher-Yates shuffle
SCL_FORCE_INLINE void shuffle(Index* arr, Size n, FastRNG& rng) {
    for (Size i = n - 1; i > 0; --i) {
        Size j = rng.bounded(i + 1);
        Index tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// Check convergence
SCL_FORCE_INLINE bool check_convergence(
    const Index* labels_old,
    const Index* labels_new,
    Size n
) {
    for (Size i = 0; i < n; ++i) {
        if (labels_old[i] != labels_new[i]) return false;
    }
    return true;
}

SCL_FORCE_INLINE bool check_convergence_real(
    const Real* vals_old,
    const Real* vals_new,
    Size n,
    Real tol
) {
    Real diff = Real(0);
    for (Size i = 0; i < n; ++i) {
        Real d = vals_new[i] - vals_old[i];
        diff += (d >= Real(0)) ? d : -d;
    }
    return diff < tol;
}

// Compute row sums for normalization
template <typename T, bool IsCSR>
void compute_row_sums(
    const Sparse<T, IsCSR>& adj,
    Real* row_sums
) {
    const Index n = adj.primary_dim();
    for (Index i = 0; i < n; ++i) {
        auto values = adj.primary_values(i);
        const Index len = adj.primary_length(i);
        Real sum = Real(0);
        for (Index k = 0; k < len; ++k) {
            sum += static_cast<Real>(values[k]);
        }
        row_sums[i] = sum;
    }
}

} // namespace detail

// =============================================================================
// Label Propagation (Hard Labels)
// =============================================================================

template <typename T, bool IsCSR>
void label_propagation(
    const Sparse<T, IsCSR>& adjacency,
    Array<Index> labels,
    Index max_iter = config::DEFAULT_MAX_ITER,
    uint64_t seed = 42
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(labels.len >= static_cast<Size>(n),
                  "Propagation: labels buffer too small");

    if (n == 0) return;

    // Find number of classes and count labeled nodes
    Index max_label = 0;
    Index n_labeled = 0;
    for (Index i = 0; i < n; ++i) {
        if (labels[i] != config::UNLABELED) {
            max_label = scl::algo::max2(max_label, labels[i]);
            ++n_labeled;
        }
    }

    if (n_labeled == 0) return;  // No labels to propagate
    if (n_labeled == n) return;  // All labeled

    Index n_classes = max_label + 1;

    // Allocate workspace
    Index* labels_new = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* order = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* class_votes = scl::memory::aligned_alloc<Real>(n_classes, SCL_ALIGNMENT);

    for (Index i = 0; i < n; ++i) {
        labels_new[i] = labels[i];
        order[i] = i;
    }

    detail::FastRNG rng(seed);

    // Iterate
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Random node order
        detail::shuffle(order, n, rng);

        bool changed = false;

        for (Index idx = 0; idx < n; ++idx) {
            Index i = order[idx];

            // Skip labeled nodes (keep original labels)
            // In standard LP, we propagate to all nodes
            // For semi-supervised, we could skip initially labeled nodes

            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            if (len == 0) continue;

            // Count weighted votes for each class
            scl::algo::zero(class_votes, static_cast<Size>(n_classes));

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Index lbl = labels_new[j];
                if (lbl != config::UNLABELED && lbl < n_classes) {
                    class_votes[lbl] += static_cast<Real>(values[k]);
                }
            }

            // Find majority class
            Index best_class = labels_new[i];
            Real best_votes = Real(0);

            for (Index c = 0; c < n_classes; ++c) {
                if (class_votes[c] > best_votes) {
                    best_votes = class_votes[c];
                    best_class = c;
                }
            }

            if (best_class != labels_new[i]) {
                labels_new[i] = best_class;
                changed = true;
            }
        }

        // Copy back
        for (Index i = 0; i < n; ++i) {
            labels[i] = labels_new[i];
        }

        if (!changed) break;
    }

    scl::memory::aligned_free(class_votes, SCL_ALIGNMENT);
    scl::memory::aligned_free(order, SCL_ALIGNMENT);
    scl::memory::aligned_free(labels_new, SCL_ALIGNMENT);
}

// =============================================================================
// Label Spreading (Regularized, Soft Labels)
// =============================================================================

template <typename T, bool IsCSR>
void label_spreading(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> label_probs,  // n_nodes * n_classes, row-major
    Index n_classes,
    const bool* is_labeled,   // Which nodes have initial labels
    Real alpha = config::DEFAULT_ALPHA,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    const Index n = adjacency.primary_dim();
    const Size total_probs = static_cast<Size>(n) * static_cast<Size>(n_classes);

    SCL_CHECK_DIM(label_probs.len >= total_probs,
                  "Spreading: label_probs buffer too small");

    if (n == 0 || n_classes == 0) return;

    // Compute normalized graph (D^(-1/2) * A * D^(-1/2))
    Real* row_sums = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    detail::compute_row_sums(adjacency, row_sums);

    // Compute D^(-1/2)
    Real* d_inv_sqrt = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    for (Index i = 0; i < n; ++i) {
        d_inv_sqrt[i] = (row_sums[i] > Real(1e-15)) ? Real(1) / std::sqrt(row_sums[i]) : Real(0);
    }

    // Store initial labels (Y_0)
    Real* Y0 = scl::memory::aligned_alloc<Real>(total_probs, SCL_ALIGNMENT);
    Real* Y_new = scl::memory::aligned_alloc<Real>(total_probs, SCL_ALIGNMENT);

    for (Size i = 0; i < total_probs; ++i) {
        Y0[i] = label_probs[i];
    }

    // Iterate: Y = alpha * S * Y + (1-alpha) * Y_0
    // where S = D^(-1/2) * A * D^(-1/2)
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Compute S * Y
        for (Index i = 0; i < n; ++i) {
            Real* yi_new = Y_new + static_cast<Size>(i) * n_classes;

            // Initialize with (1-alpha) * Y0[i]
            if (is_labeled[i]) {
                Real* y0i = Y0 + static_cast<Size>(i) * n_classes;
                for (Index c = 0; c < n_classes; ++c) {
                    yi_new[c] = (Real(1) - alpha) * y0i[c];
                }
            } else {
                for (Index c = 0; c < n_classes; ++c) {
                    yi_new[c] = Real(0);
                }
            }

            // Add alpha * sum_j(S[i,j] * Y[j])
            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Real w = static_cast<Real>(values[k]);
                Real s_ij = d_inv_sqrt[i] * w * d_inv_sqrt[j];

                Real* yj = label_probs.ptr + static_cast<Size>(j) * n_classes;
                for (Index c = 0; c < n_classes; ++c) {
                    yi_new[c] += alpha * s_ij * yj[c];
                }
            }
        }

        // Normalize rows to sum to 1
        for (Index i = 0; i < n; ++i) {
            Real* yi = Y_new + static_cast<Size>(i) * n_classes;
            Real sum = Real(0);
            for (Index c = 0; c < n_classes; ++c) {
                sum += yi[c];
            }
            if (sum > Real(1e-15)) {
                for (Index c = 0; c < n_classes; ++c) {
                    yi[c] /= sum;
                }
            }
        }

        // Check convergence
        if (detail::check_convergence_real(label_probs.ptr, Y_new, total_probs, tol)) {
            for (Size i = 0; i < total_probs; ++i) {
                label_probs[i] = Y_new[i];
            }
            break;
        }

        // Copy
        for (Size i = 0; i < total_probs; ++i) {
            label_probs[i] = Y_new[i];
        }
    }

    scl::memory::aligned_free(Y_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(Y0, SCL_ALIGNMENT);
    scl::memory::aligned_free(d_inv_sqrt, SCL_ALIGNMENT);
    scl::memory::aligned_free(row_sums, SCL_ALIGNMENT);
}

// =============================================================================
// Inductive Label Transfer (Reference to Query)
// =============================================================================

template <typename T, bool IsCSR>
void inductive_transfer(
    const Sparse<T, IsCSR>& ref_to_query,  // Rows: query, Cols: reference (similarity)
    Array<const Index> reference_labels,
    Array<Index> query_labels,
    Index n_classes,
    Real confidence_threshold = Real(0.5)
) {
    const Index n_query = ref_to_query.rows();

    SCL_CHECK_DIM(query_labels.len >= static_cast<Size>(n_query),
                  "Transfer: query_labels buffer too small");

    if (n_query == 0 || n_classes == 0) return;

    Real* class_scores = scl::memory::aligned_alloc<Real>(n_classes, SCL_ALIGNMENT);

    for (Index q = 0; q < n_query; ++q) {
        auto indices = ref_to_query.row_indices(q);
        auto values = ref_to_query.row_values(q);
        const Index len = ref_to_query.row_length(q);

        if (len == 0) {
            query_labels[q] = config::UNLABELED;
            continue;
        }

        // Weighted voting from reference neighbors
        scl::algo::zero(class_scores, static_cast<Size>(n_classes));
        Real total_weight = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index ref_idx = indices[k];
            Index ref_label = reference_labels[ref_idx];
            Real w = static_cast<Real>(values[k]);

            if (ref_label >= 0 && ref_label < n_classes) {
                class_scores[ref_label] += w;
                total_weight += w;
            }
        }

        // Find best class and confidence
        Index best_class = config::UNLABELED;
        Real best_score = Real(0);

        for (Index c = 0; c < n_classes; ++c) {
            if (class_scores[c] > best_score) {
                best_score = class_scores[c];
                best_class = c;
            }
        }

        // Apply confidence threshold
        Real confidence = (total_weight > Real(1e-15)) ? best_score / total_weight : Real(0);
        query_labels[q] = (confidence >= confidence_threshold) ? best_class : config::UNLABELED;
    }

    scl::memory::aligned_free(class_scores, SCL_ALIGNMENT);
}

// =============================================================================
// Confidence-Weighted Label Propagation
// =============================================================================

template <typename T, bool IsCSR>
void confidence_propagation(
    const Sparse<T, IsCSR>& adjacency,
    Array<Index> labels,
    Array<Real> confidence,  // In/Out: confidence per node
    Index n_classes,
    Real alpha = config::DEFAULT_ALPHA,
    Index max_iter = config::DEFAULT_MAX_ITER
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(labels.len >= static_cast<Size>(n),
                  "Propagation: labels buffer too small");
    SCL_CHECK_DIM(confidence.len >= static_cast<Size>(n),
                  "Propagation: confidence buffer too small");

    if (n == 0 || n_classes == 0) return;

    Index* labels_new = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* conf_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* class_votes = scl::memory::aligned_alloc<Real>(n_classes, SCL_ALIGNMENT);

    for (Index i = 0; i < n; ++i) {
        labels_new[i] = labels[i];
        conf_new[i] = confidence[i];
    }

    for (Index iter = 0; iter < max_iter; ++iter) {
        bool changed = false;

        for (Index i = 0; i < n; ++i) {
            auto indices = adjacency.primary_indices(i);
            auto values = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            if (len == 0) continue;

            // Confidence-weighted voting
            scl::algo::zero(class_votes, static_cast<Size>(n_classes));
            Real total_conf = Real(0);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Index lbl = labels[j];
                Real w = static_cast<Real>(values[k]);
                Real c = confidence[j];

                if (lbl >= 0 && lbl < n_classes) {
                    class_votes[lbl] += w * c;
                    total_conf += w * c;
                }
            }

            // Include own label with its confidence
            if (labels[i] >= 0 && labels[i] < n_classes) {
                class_votes[labels[i]] += alpha * confidence[i];
                total_conf += alpha * confidence[i];
            }

            // Find best class
            Index best_class = labels[i];
            Real best_votes = Real(0);

            for (Index c = 0; c < n_classes; ++c) {
                if (class_votes[c] > best_votes) {
                    best_votes = class_votes[c];
                    best_class = c;
                }
            }

            // Update label and confidence
            if (best_class != labels_new[i]) {
                labels_new[i] = best_class;
                changed = true;
            }

            // New confidence: fraction of votes for winning class
            if (total_conf > Real(1e-15)) {
                conf_new[i] = best_votes / total_conf;
            } else {
                conf_new[i] = Real(0);
            }
        }

        // Copy back
        for (Index i = 0; i < n; ++i) {
            labels[i] = labels_new[i];
            confidence[i] = conf_new[i];
        }

        if (!changed) break;
    }

    scl::memory::aligned_free(class_votes, SCL_ALIGNMENT);
    scl::memory::aligned_free(conf_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(labels_new, SCL_ALIGNMENT);
}

// =============================================================================
// Harmonic Function Solution (Exact for Small Graphs)
// =============================================================================

template <typename T, bool IsCSR>
void harmonic_function(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> values,       // In/Out: known values fixed, unknown computed
    const bool* is_known,     // Which nodes have known values
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(values.len >= static_cast<Size>(n),
                  "Harmonic: values buffer too small");

    if (n == 0) return;

    Real* row_sums = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* values_new = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);

    detail::compute_row_sums(adjacency, row_sums);

    for (Index i = 0; i < n; ++i) {
        values_new[i] = values[i];
    }

    // Gauss-Seidel iteration: f(u) = sum_v(w_uv * f(v)) / sum_v(w_uv)
    for (Index iter = 0; iter < max_iter; ++iter) {
        Real max_change = Real(0);

        for (Index i = 0; i < n; ++i) {
            if (is_known[i]) continue;  // Skip known nodes

            auto indices = adjacency.primary_indices(i);
            auto weights = adjacency.primary_values(i);
            const Index len = adjacency.primary_length(i);

            if (len == 0 || row_sums[i] <= Real(1e-15)) continue;

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                sum += static_cast<Real>(weights[k]) * values_new[j];
            }

            Real new_val = sum / row_sums[i];
            Real change = (new_val - values_new[i]);
            change = (change >= Real(0)) ? change : -change;
            max_change = scl::algo::max2(max_change, change);
            values_new[i] = new_val;
        }

        // Copy back
        for (Index i = 0; i < n; ++i) {
            values[i] = values_new[i];
        }

        if (max_change < tol) break;
    }

    scl::memory::aligned_free(values_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(row_sums, SCL_ALIGNMENT);
}

// =============================================================================
// Get Predicted Labels from Soft Probabilities
// =============================================================================

inline void get_hard_labels(
    Array<const Real> probs,  // n_nodes * n_classes, row-major
    Index n_nodes,
    Index n_classes,
    Array<Index> labels,
    Array<Real> max_probs = Array<Real>(nullptr, 0)
) {
    SCL_CHECK_DIM(labels.len >= static_cast<Size>(n_nodes),
                  "Labels: output buffer too small");

    bool output_probs = (max_probs.ptr != nullptr && max_probs.len >= static_cast<Size>(n_nodes));

    for (Index i = 0; i < n_nodes; ++i) {
        const Real* pi = probs.ptr + static_cast<Size>(i) * n_classes;

        Index best_class = 0;
        Real best_prob = pi[0];

        for (Index c = 1; c < n_classes; ++c) {
            if (pi[c] > best_prob) {
                best_prob = pi[c];
                best_class = c;
            }
        }

        labels[i] = best_class;
        if (output_probs) {
            max_probs[i] = best_prob;
        }
    }
}

// =============================================================================
// Initialize Soft Labels from Hard Labels
// =============================================================================

inline void init_soft_labels(
    Array<const Index> hard_labels,
    Index n_classes,
    Array<Real> soft_labels,  // n_nodes * n_classes, row-major
    Real labeled_confidence = Real(1.0),
    Real unlabeled_prior = Real(0.0)  // 0 for uniform, >0 for specific prior
) {
    const Size n = hard_labels.len;

    SCL_CHECK_DIM(soft_labels.len >= n * static_cast<Size>(n_classes),
                  "SoftLabels: output buffer too small");

    Real uniform_prob = (unlabeled_prior > Real(0)) ? unlabeled_prior : (Real(1) / static_cast<Real>(n_classes));

    for (Size i = 0; i < n; ++i) {
        Real* pi = soft_labels.ptr + i * static_cast<Size>(n_classes);
        Index label = hard_labels[i];

        if (label >= 0 && label < n_classes) {
            // Labeled node
            for (Index c = 0; c < n_classes; ++c) {
                pi[c] = (c == label) ? labeled_confidence : (Real(1) - labeled_confidence) / static_cast<Real>(n_classes - 1);
            }
        } else {
            // Unlabeled node
            for (Index c = 0; c < n_classes; ++c) {
                pi[c] = uniform_prob;
            }
        }
    }
}

} // namespace scl::kernel::propagation
