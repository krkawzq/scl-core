#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cstring>

// =============================================================================
// FILE: scl/kernel/merge.hpp
// BRIEF: Matrix merging operations with SIMD optimization
// =============================================================================

namespace scl::kernel::merge {

// =============================================================================
// SIMD Utilities
// =============================================================================

namespace detail {

SCL_FORCE_INLINE void add_offset_simd(
    const Index* SCL_RESTRICT src,
    Index* SCL_RESTRICT dst,
    Size count,
    Index offset
) {
    if (offset == 0) {
        std::memcpy(dst, src, count * sizeof(Index));
        return;
    }

    namespace s = scl::simd;
    const s::IndexTag d;
    const size_t lanes = s::Lanes(d);

    const auto v_offset = s::Set(d, offset);

    Size i = 0;
    const Size simd_end = count - (count % (lanes * 2));

    for (; i < simd_end; i += lanes * 2) {
        auto v0 = s::Load(d, src + i);
        auto v1 = s::Load(d, src + i + lanes);
        s::Store(s::Add(v0, v_offset), d, dst + i);
        s::Store(s::Add(v1, v_offset), d, dst + i + lanes);
    }

    for (; i + lanes <= count; i += lanes) {
        auto v = s::Load(d, src + i);
        s::Store(s::Add(v, v_offset), d, dst + i);
    }

    for (; i < count; ++i) {
        dst[i] = src[i] + offset;
    }
}

template <typename T>
inline void parallel_memcpy(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Size count,
    Size chunk_size = 65536
) {
    if (count < chunk_size) {
        std::memcpy(dst, src, count * sizeof(T));
        return;
    }

    const Size n_chunks = (count + chunk_size - 1) / chunk_size;

    scl::threading::parallel_for(Size(0), n_chunks, [&](size_t c) {
        Size start = c * chunk_size;
        Size end = scl::algo::min2(start + chunk_size, count);
        Size len = end - start;

        if (c + 1 < n_chunks) {
            SCL_PREFETCH_READ(src + end, 0);
        }

        std::memcpy(dst + start, src + start, len * sizeof(T));
    });
}

} // namespace detail

// =============================================================================
// Merge Operations
// =============================================================================

template <typename T, bool IsCSR>
Sparse<T, IsCSR> vstack(
    const Sparse<T, IsCSR>& matrix1,
    const Sparse<T, IsCSR>& matrix2,
    BlockStrategy strategy = BlockStrategy::adaptive()
) {
    const Index primary1 = matrix1.primary_dim();
    const Index primary2 = matrix2.primary_dim();
    const Index secondary1 = matrix1.secondary_dim();
    const Index secondary2 = matrix2.secondary_dim();

    const Index total_primary = primary1 + primary2;
    const Index total_secondary = scl::algo::max2(secondary1, secondary2);

    Index* nnzs = scl::memory::aligned_alloc<Index>(total_primary, SCL_ALIGNMENT);
    for (Index i = 0; i < primary1; ++i) {
        nnzs[i] = matrix1.primary_length_unsafe(i);
    }
    for (Index i = 0; i < primary2; ++i) {
        nnzs[primary1 + i] = matrix2.primary_length_unsafe(i);
    }

    Sparse<T, IsCSR> result = Sparse<T, IsCSR>::create(
        IsCSR ? total_primary : total_secondary,
        IsCSR ? total_secondary : total_primary,
        Array<const Index>(nnzs, static_cast<Size>(total_primary)),
        strategy
    );

    scl::memory::aligned_free(nnzs, SCL_ALIGNMENT);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary1), [&](size_t i) {
        const Index idx = static_cast<Index>(i);
        const Index len = matrix1.primary_length_unsafe(idx);
        if (len == 0) return;

        const auto src_values = matrix1.primary_values_unsafe(idx);
        const auto src_indices = matrix1.primary_indices_unsafe(idx);
        auto dst_values = result.primary_values_unsafe(idx);
        auto dst_indices = result.primary_indices_unsafe(idx);

        std::memcpy(dst_values.ptr, src_values.ptr, len * sizeof(T));
        std::memcpy(dst_indices.ptr, src_indices.ptr, len * sizeof(Index));
    });

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary2), [&](size_t i) {
        const Index src_idx = static_cast<Index>(i);
        const Index dst_idx = primary1 + src_idx;
        const Index len = matrix2.primary_length_unsafe(src_idx);
        if (len == 0) return;

        const auto src_values = matrix2.primary_values_unsafe(src_idx);
        const auto src_indices = matrix2.primary_indices_unsafe(src_idx);
        auto dst_values = result.primary_values_unsafe(dst_idx);
        auto dst_indices = result.primary_indices_unsafe(dst_idx);

        std::memcpy(dst_values.ptr, src_values.ptr, len * sizeof(T));
        std::memcpy(dst_indices.ptr, src_indices.ptr, len * sizeof(Index));
    });

    return result;
}

template <typename T, bool IsCSR>
Sparse<T, IsCSR> hstack(
    const Sparse<T, IsCSR>& matrix1,
    const Sparse<T, IsCSR>& matrix2,
    BlockStrategy strategy = BlockStrategy::adaptive()
) {
    const Index primary1 = matrix1.primary_dim();
    const Index primary2 = matrix2.primary_dim();
    const Index secondary1 = matrix1.secondary_dim();
    const Index secondary2 = matrix2.secondary_dim();

    SCL_CHECK_DIM(primary1 == primary2, "hstack: Primary dimension mismatch");

    const Index primary_dim = primary1;
    const Index total_secondary = secondary1 + secondary2;

    Index* nnzs = scl::memory::aligned_alloc<Index>(primary_dim, SCL_ALIGNMENT);
    for (Index i = 0; i < primary_dim; ++i) {
        nnzs[i] = matrix1.primary_length_unsafe(i) + matrix2.primary_length_unsafe(i);
    }

    Sparse<T, IsCSR> result = Sparse<T, IsCSR>::create(
        IsCSR ? primary_dim : total_secondary,
        IsCSR ? total_secondary : primary_dim,
        Array<const Index>(nnzs, static_cast<Size>(primary_dim)),
        strategy
    );

    scl::memory::aligned_free(nnzs, SCL_ALIGNMENT);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);

        const Index len1 = matrix1.primary_length_unsafe(idx);
        const Index len2 = matrix2.primary_length_unsafe(idx);

        auto dst_values = result.primary_values_unsafe(idx);
        auto dst_indices = result.primary_indices_unsafe(idx);

        if (len1 > 0) {
            const auto src1_values = matrix1.primary_values_unsafe(idx);
            const auto src1_indices = matrix1.primary_indices_unsafe(idx);

            std::memcpy(dst_values.ptr, src1_values.ptr, len1 * sizeof(T));
            std::memcpy(dst_indices.ptr, src1_indices.ptr, len1 * sizeof(Index));
        }

        if (len2 > 0) {
            const auto src2_values = matrix2.primary_values_unsafe(idx);
            const auto src2_indices = matrix2.primary_indices_unsafe(idx);

            std::memcpy(dst_values.ptr + len1, src2_values.ptr, len2 * sizeof(T));
            detail::add_offset_simd(
                src2_indices.ptr,
                dst_indices.ptr + len1,
                static_cast<Size>(len2),
                secondary1
            );
        }
    });

    return result;
}

} // namespace scl::kernel::merge
