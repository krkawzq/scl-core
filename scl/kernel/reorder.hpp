#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cstring>

// =============================================================================
// FILE: scl/kernel/reorder.hpp
// BRIEF: Sparse matrix reordering with adaptive processing
// =============================================================================

namespace scl::kernel::reorder {

namespace config {
    constexpr Size SHORT_THRESHOLD = 32;
    constexpr Size MEDIUM_THRESHOLD = 256;
    constexpr Size PREFETCH_DISTANCE = 16;
}

namespace detail {

SCL_FORCE_INLINE bool is_valid_index(Index new_idx, Index new_dim) {
    return static_cast<uint64_t>(new_idx) < static_cast<uint64_t>(new_dim);
}

template <typename IndexT>
SCL_FORCE_INLINE Size count_valid_short(
    const IndexT* SCL_RESTRICT indices,
    Size len,
    const IndexT* SCL_RESTRICT index_map,
    IndexT new_dim
) {
    Size count = 0;
    for (Size k = 0; k < len; ++k) {
        IndexT new_idx = index_map[indices[k]];
        count += is_valid_index(new_idx, new_dim);
    }
    return count;
}

template <typename IndexT>
SCL_FORCE_INLINE Size count_valid_unrolled(
    const IndexT* SCL_RESTRICT indices,
    Size len,
    const IndexT* SCL_RESTRICT index_map,
    IndexT new_dim
) {
    Size count = 0;

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        if (k + config::PREFETCH_DISTANCE < len) {
            SCL_PREFETCH_READ(&index_map[indices[k + config::PREFETCH_DISTANCE]], 0);
        }

        IndexT new0 = index_map[indices[k + 0]];
        IndexT new1 = index_map[indices[k + 1]];
        IndexT new2 = index_map[indices[k + 2]];
        IndexT new3 = index_map[indices[k + 3]];

        count += is_valid_index(new0, new_dim);
        count += is_valid_index(new1, new_dim);
        count += is_valid_index(new2, new_dim);
        count += is_valid_index(new3, new_dim);
    }

    for (; k < len; ++k) {
        IndexT new_idx = index_map[indices[k]];
        count += is_valid_index(new_idx, new_dim);
    }

    return count;
}

template <typename IndexT>
SCL_FORCE_INLINE Size count_valid_adaptive(
    const IndexT* SCL_RESTRICT indices,
    Size len,
    const IndexT* SCL_RESTRICT index_map,
    IndexT new_dim
) {
    if (len < config::SHORT_THRESHOLD) {
        return count_valid_short(indices, len, index_map, new_dim);
    } else {
        return count_valid_unrolled(indices, len, index_map, new_dim);
    }
}

template <typename T, typename IndexT>
SCL_FORCE_INLINE Size remap_compact_short(
    IndexT* SCL_RESTRICT indices,
    T* SCL_RESTRICT values,
    Size len,
    const IndexT* SCL_RESTRICT index_map,
    IndexT new_dim
) {
    Size write_pos = 0;

    for (Size k = 0; k < len; ++k) {
        IndexT old_idx = indices[k];
        IndexT new_idx = index_map[old_idx];

        if (SCL_LIKELY(is_valid_index(new_idx, new_dim))) {
            indices[write_pos] = new_idx;
            values[write_pos] = values[k];
            ++write_pos;
        }
    }

    return write_pos;
}

template <typename T, typename IndexT>
SCL_FORCE_INLINE Size remap_compact_medium(
    IndexT* SCL_RESTRICT indices,
    T* SCL_RESTRICT values,
    Size len,
    const IndexT* SCL_RESTRICT index_map,
    IndexT new_dim
) {
    Size write_pos = 0;

    for (Size k = 0; k < len; ++k) {
        if (k + config::PREFETCH_DISTANCE < len) {
            SCL_PREFETCH_READ(&index_map[indices[k + config::PREFETCH_DISTANCE]], 0);
        }

        IndexT old_idx = indices[k];
        IndexT new_idx = index_map[old_idx];

        if (is_valid_index(new_idx, new_dim)) {
            indices[write_pos] = new_idx;
            values[write_pos] = values[k];
            ++write_pos;
        }
    }

    return write_pos;
}

template <typename T, typename IndexT>
SCL_FORCE_INLINE Size remap_compact_long(
    IndexT* SCL_RESTRICT indices,
    T* SCL_RESTRICT values,
    Size len,
    const IndexT* SCL_RESTRICT index_map,
    IndexT new_dim
) {
    Size write_pos = 0;
    constexpr Size PREFETCH_DIST1 = 16;
    constexpr Size PREFETCH_DIST2 = 32;

    for (Size k = 0; k < len; ++k) {
        if (k + PREFETCH_DIST1 < len) {
            SCL_PREFETCH_READ(&index_map[indices[k + PREFETCH_DIST1]], 0);
        }
        if (k + PREFETCH_DIST2 < len) {
            SCL_PREFETCH_READ(&index_map[indices[k + PREFETCH_DIST2]], 1);
        }

        IndexT old_idx = indices[k];
        IndexT new_idx = index_map[old_idx];

        if (is_valid_index(new_idx, new_dim)) {
            indices[write_pos] = new_idx;
            values[write_pos] = values[k];
            ++write_pos;
        }
    }

    return write_pos;
}

template <typename T, typename IndexT>
SCL_FORCE_INLINE Size remap_compact_adaptive(
    IndexT* SCL_RESTRICT indices,
    T* SCL_RESTRICT values,
    Size len,
    const IndexT* SCL_RESTRICT index_map,
    IndexT new_dim
) {
    if (len < config::SHORT_THRESHOLD) {
        return remap_compact_short(indices, values, len, index_map, new_dim);
    } else if (len < config::MEDIUM_THRESHOLD) {
        return remap_compact_medium(indices, values, len, index_map, new_dim);
    } else {
        return remap_compact_long(indices, values, len, index_map, new_dim);
    }
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool IsCSR>
void align_secondary(
    Sparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Array<Index> out_lengths,
    Index new_secondary_dim
) {
    const Index primary_dim = matrix.primary_dim();

    SCL_CHECK_DIM(out_lengths.len >= static_cast<Size>(primary_dim),
                  "Reorder: Output lengths too small");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        if (len_sz == 0) {
            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
            // PERFORMANCE: Safe narrowing - p is bounded by primary_dim which fits in Index
            out_lengths[static_cast<Index>(p)] = 0;
            return;
        }

        auto indices_arr = matrix.primary_indices_unsafe(idx);
        auto values_arr = matrix.primary_values_unsafe(idx);

        // PERFORMANCE: RAII memory management with unique_ptr
        // Using aligned_alloc returns unique_ptr for automatic cleanup
        auto indices_copy_ptr = scl::memory::aligned_alloc<Index>(len_sz, SCL_ALIGNMENT);
        auto values_copy_ptr = scl::memory::aligned_alloc<T>(len_sz, SCL_ALIGNMENT);
        Index* indices_copy = indices_copy_ptr.get();
        T* values_copy = values_copy_ptr.get();

        scl::algo::copy(indices_arr.ptr, indices_copy, len_sz);
        scl::algo::copy(values_arr.ptr, values_copy, len_sz);

        Size new_len = detail::remap_compact_adaptive(
            indices_copy, values_copy, len_sz,
            index_map.ptr, new_secondary_dim
        );

        if (new_len > 1) {
            scl::sort::sort_pairs(
                Array<Index>(indices_copy, new_len),
                Array<T>(values_copy, new_len)
            );
        }

        for (Size k = 0; k < new_len; ++k) {
            indices_arr.ptr[k] = indices_copy[k];
            values_arr.ptr[k] = values_copy[k];
        }

        // unique_ptr automatically frees memory when going out of scope

        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - p and new_len are bounded and fit in Index
        out_lengths[static_cast<Index>(p)] = static_cast<Index>(new_len);
    });
}

template <typename T, bool IsCSR>
Size compute_filtered_nnz(
    const Sparse<T, IsCSR>& matrix,
    Array<const Index> index_map,
    Index new_secondary_dim
) {
    const Index primary_dim = matrix.primary_dim();
    const Size primary_dim_sz = static_cast<Size>(primary_dim);

    // PERFORMANCE: RAII memory management with unique_ptr
    // Using aligned_alloc returns unique_ptr for automatic cleanup
    auto partial_sums_ptr = scl::memory::aligned_alloc<Size>(primary_dim_sz, SCL_ALIGNMENT);
    Size* partial_sums = partial_sums_ptr.get();

    scl::threading::parallel_for(Size(0), primary_dim_sz, [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        if (len_sz == 0) {
            partial_sums[p] = 0;
            return;
        }

        auto indices = matrix.primary_indices_unsafe(idx);
        partial_sums[p] = detail::count_valid_adaptive(
            indices.ptr, len_sz, index_map.ptr, new_secondary_dim
        );
    });

    Size total = 0;
    for (Size i = 0; i < primary_dim_sz; ++i) {
        total += partial_sums[i];
    }

    // unique_ptr automatically frees memory when going out of scope
    return total;
}

inline void build_inverse_permutation(
    Array<const Index> permutation,
    Array<Index> inverse
) {
    Size n = permutation.len;
    SCL_CHECK_DIM(inverse.len >= n, "Inverse buffer too small");

    scl::threading::parallel_for(Size(0), n, [&](Size i) {
        // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions)
        // PERFORMANCE: Safe narrowing - i and permutation[i] are bounded and fit in Index
        const auto idx = static_cast<Index>(i);
        inverse[permutation[idx]] = idx;
    });
}

} // namespace scl::kernel::reorder

