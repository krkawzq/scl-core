#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/softmax_fast_impl.hpp"
#include "scl/kernel/softmax_mapped_impl.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cmath>
#include <limits>

// =============================================================================
/// @file softmax.hpp
/// @brief Softmax and Log-Softmax Transformations
///
/// Unified entry point for sparse matrix softmax with automatic backend dispatch:
///
/// - CustomSparse / VirtualSparse: Uses softmax_fast_impl.hpp (in-place)
/// - MappedCustomSparse / MappedVirtualSparse: Uses softmax_mapped_impl.hpp
///
/// ## Operations
///
/// 1. softmax: softmax(x)_i = exp(x_i - max) / sum(exp(x_j - max))
///    - Numerically stable via max-subtraction
///    - Per-row (CSR) or per-column (CSC)
///
/// 2. log_softmax: log(softmax(x)) = x - max - log(sum(exp(x - max)))
///    - More numerically stable for cross-entropy
///    - Avoids division
///
/// ## Performance Optimizations
///
/// 1. SIMD Max Finding: Vectorized horizontal max reduction
/// 2. 8-Way Unrolled SIMD: Maximum instruction-level parallelism
/// 3. Fused Operations: exp + sum in single pass
/// 4. Adaptive Dispatch: Short/medium/long row strategies
/// 5. Fused Materialize + Transform: For mapped matrices
///
/// ## Numerical Stability
///
/// The max-subtraction trick prevents numerical overflow:
/// - exp(x - max) is always in [0, 1]
/// - sum(exp(x - max)) is always >= 1
/// - Result is valid probability distribution
// =============================================================================

namespace scl::kernel::softmax {

// =============================================================================
// SECTION 1: Generic Implementation (Fallback)
// =============================================================================

namespace detail {

/// @brief Generic softmax for any mutable values array
template <typename T>
SCL_FORCE_INLINE void softmax_generic(T* SCL_RESTRICT vals, Size len) {
    if (len == 0) return;

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    // Pass 1: Find max
    auto v_max = s::Set(d, -std::numeric_limits<T>::infinity());
    Size k = 0;
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_max = s::Max(v_max, v);
    }
    T max_val = s::GetLane(s::MaxOfLanes(d, v_max));
    for (; k < len; ++k) {
        if (vals[k] > max_val) max_val = vals[k];
    }

    // Pass 2: Exp + Sum
    const auto v_max_bc = s::Set(d, max_val);
    auto v_sum = s::Zero(d);
    k = 0;
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v = s::Exp(d, s::Sub(v, v_max_bc));
        s::Store(v, d, vals + k);
        v_sum = s::Add(v_sum, v);
    }
    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    for (; k < len; ++k) {
        T v = std::exp(vals[k] - max_val);
        vals[k] = v;
        sum += v;
    }

    // Pass 3: Normalize
    if (sum > T(0)) {
        T inv_sum = T(1) / sum;
        const auto v_inv_sum = s::Set(d, inv_sum);
        k = 0;
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Mul(v, v_inv_sum), d, vals + k);
        }
        for (; k < len; ++k) {
            vals[k] *= inv_sum;
        }
    }
}

/// @brief Generic log_softmax for any mutable values array
template <typename T>
SCL_FORCE_INLINE void log_softmax_generic(T* SCL_RESTRICT vals, Size len) {
    if (len == 0) return;

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    // Find max
    auto v_max = s::Set(d, -std::numeric_limits<T>::infinity());
    Size k = 0;
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_max = s::Max(v_max, v);
    }
    T max_val = s::GetLane(s::MaxOfLanes(d, v_max));
    for (; k < len; ++k) {
        if (vals[k] > max_val) max_val = vals[k];
    }

    // Sum(exp(x - max))
    const auto v_max_bc = s::Set(d, max_val);
    auto v_sum = s::Zero(d);
    k = 0;
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, s::Exp(d, s::Sub(v, v_max_bc)));
    }
    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    for (; k < len; ++k) {
        sum += std::exp(vals[k] - max_val);
    }

    // log_softmax = x - max - log(sum)
    T offset = max_val + std::log(sum);
    const auto v_offset = s::Set(d, offset);
    k = 0;
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        s::Store(s::Sub(v, v_offset), d, vals + k);
    }
    for (; k < len; ++k) {
        vals[k] -= offset;
    }
}

} // namespace detail

// =============================================================================
// SECTION 2: In-Place Softmax (CustomSparse / VirtualSparse)
// =============================================================================

/// @brief Softmax transformation (in-place)
///
/// Applies softmax independently to each primary dimension element.
/// For CSR: softmax over each row. For CSC: softmax over each column.
///
/// @tparam MatrixT Sparse matrix type (must satisfy AnySparse)
/// @param matrix Input sparse matrix (modified in-place)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void softmax_inplace(MatrixT& matrix) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        fast::softmax_inplace_custom_fast(matrix);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        fast::softmax_inplace_virtual_fast(matrix);
    } else {
        // Generic fallback
        const Index primary_dim = scl::primary_size(matrix);

        scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
            auto vals = scl::primary_values(matrix, p);
            if (vals.len == 0) return;
            detail::softmax_generic(vals.ptr, vals.len);
        });
    }
}

/// @brief Log-softmax transformation (in-place)
///
/// Computes log(softmax(x)) = x - max - log(sum(exp(x - max))).
/// More numerically stable than log(softmax(x)) computed separately.
///
/// @param matrix Input sparse matrix (modified in-place)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void log_softmax_inplace(MatrixT& matrix) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        fast::log_softmax_inplace_custom_fast(matrix);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        fast::log_softmax_inplace_virtual_fast(matrix);
    } else {
        const Index primary_dim = scl::primary_size(matrix);

        scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
            auto vals = scl::primary_values(matrix, p);
            if (vals.len == 0) return;
            detail::log_softmax_generic(vals.ptr, vals.len);
        });
    }
}

// =============================================================================
// SECTION 3: Mapped Softmax (Returns OwnedSparse)
// =============================================================================

/// @brief Softmax for mapped matrix (MappedCustomSparse)
///
/// Materializes to OwnedSparse with softmax applied.
/// Uses fused materialize + transform for efficiency.
///
/// @param matrix Input mapped sparse matrix (read-only)
/// @return OwnedSparse with softmax applied
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> softmax(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    return mapped::softmax_mapped_custom(matrix);
}

/// @brief Softmax for mapped matrix (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> softmax(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    return mapped::softmax_mapped_virtual(matrix);
}

/// @brief Log-softmax for mapped matrix (MappedCustomSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log_softmax(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix
) {
    return mapped::log_softmax_mapped_custom(matrix);
}

/// @brief Log-softmax for mapped matrix (MappedVirtualSparse)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
scl::io::OwnedSparse<T, IsCSR> log_softmax(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix
) {
    return mapped::log_softmax_mapped_virtual(matrix);
}

// =============================================================================
// SECTION 4: Temperature-Scaled Softmax
// =============================================================================

/// @brief Temperature-scaled softmax
///
/// Computes softmax(x / temperature) for controlling distribution sharpness.
/// - temperature < 1: sharper distribution (more confident)
/// - temperature > 1: flatter distribution (more uniform)
/// - temperature = 1: standard softmax
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param temperature Temperature parameter (must be > 0)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void softmax_with_temperature(MatrixT& matrix, typename MatrixT::ValueType temperature) {
    using T = typename MatrixT::ValueType;

    SCL_CHECK_ARG(temperature > T(0), "Temperature must be positive");

    const Index primary_dim = scl::primary_size(matrix);
    T inv_temp = T(1) / temperature;

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_inv_temp = s::Set(d, inv_temp);

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        auto vals = scl::primary_values(matrix, p);
        if (vals.len == 0) return;

        // Scale by 1/temperature
        Size k = 0;
        for (; k + lanes <= vals.len; k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            s::Store(s::Mul(v, v_inv_temp), d, vals.ptr + k);
        }
        for (; k < vals.len; ++k) {
            vals[k] *= inv_temp;
        }

        // Apply standard softmax
        detail::softmax_generic(vals.ptr, vals.len);
    });
}

// =============================================================================
// SECTION 5: Sparse Attention Softmax
// =============================================================================

/// @brief Softmax with masking (for sparse attention)
///
/// Applies softmax with optional mask. Masked positions are set to 0
/// in the output (not included in normalization).
///
/// @param matrix Input sparse matrix (modified in-place)
/// @param mask Boolean mask (true = keep, false = mask out)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void softmax_masked(
    MatrixT& matrix,
    Array<const bool> mask
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(matrix);

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        auto vals = scl::primary_values(matrix, p);
        auto inds = scl::primary_indices(matrix, p);
        if (vals.len == 0) return;

        // Find max of unmasked values
        T max_val = -std::numeric_limits<T>::infinity();
        for (Size k = 0; k < vals.len; ++k) {
            Index idx = inds[k];
            if (idx < static_cast<Index>(mask.len) && mask[idx]) {
                if (vals[k] > max_val) max_val = vals[k];
            }
        }

        // If all masked, zero out
        if (max_val == -std::numeric_limits<T>::infinity()) {
            for (Size k = 0; k < vals.len; ++k) {
                vals[k] = T(0);
            }
            return;
        }

        // Exp + sum for unmasked
        T sum = T(0);
        for (Size k = 0; k < vals.len; ++k) {
            Index idx = inds[k];
            if (idx < static_cast<Index>(mask.len) && mask[idx]) {
                T v = std::exp(vals[k] - max_val);
                vals[k] = v;
                sum += v;
            } else {
                vals[k] = T(0);
            }
        }

        // Normalize unmasked
        if (sum > T(0)) {
            T inv_sum = T(1) / sum;
            for (Size k = 0; k < vals.len; ++k) {
                Index idx = inds[k];
                if (idx < static_cast<Index>(mask.len) && mask[idx]) {
                    vals[k] *= inv_sum;
                }
            }
        }
    });
}

} // namespace scl::kernel::softmax
