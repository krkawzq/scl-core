#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/macros.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cstddef>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// =============================================================================
/// @file mapped_common.hpp
/// @brief Mapped Backend Common Definitions (Extreme Performance)
///
/// Defines:
/// - MappedSparseLike concept
/// - Unified dispatchers for Custom/Virtual/Mapped
/// - SIMD-optimized common utilities
/// - Load-balanced chunking strategies
///
/// Design Philosophy:
/// - One set of operators, all backends unified
/// - Compile-time dispatch via concepts
/// - Zero overhead abstraction
/// - Vectorized hot paths
// =============================================================================

namespace scl::kernel::mapped {

// =============================================================================
// MappedSparseLike Concept
// =============================================================================

namespace detail {

/// @brief Detect if type has prefetch method
template <typename T>
concept HasPrefetch = requires(const T& t) {
    { t.prefetch() } -> std::same_as<void>;
};

/// @brief Detect if type has drop_cache method
template <typename T>
concept HasDropCache = requires(const T& t) {
    { t.drop_cache() } -> std::same_as<void>;
};

/// @brief Detect if type is MappedCustomSparse
template <typename T>
concept IsMappedCustomSparse = requires {
    typename T::ValueType;
    typename T::Tag;
} && HasPrefetch<T> && HasDropCache<T>;

/// @brief Detect if type is MappedVirtualSparse
template <typename T>
concept IsMappedVirtualSparse = requires(const T& t) {
    typename T::ValueType;
    typename T::Tag;
    { t.nnz() } -> std::convertible_to<Index>;
};

} // namespace detail

/// @brief Concept for memory-mapped sparse matrices
///
/// Extends SparseLike with memory-mapped specific operations:
/// - prefetch(): Advise kernel about upcoming access
/// - drop_cache(): Release cached pages
///
/// Satisfied by:
/// - MappedCustomSparse<T, IsCSR>
/// - MappedVirtualSparse<T, IsCSR>
template <typename M, bool IsCSR>
concept MappedSparseLike = SparseLike<M, IsCSR> && (
    detail::IsMappedCustomSparse<M> || detail::IsMappedVirtualSparse<M>
);

// =============================================================================
// Backend Classification
// =============================================================================

/// @brief Backend type enumeration
enum class BackendType {
    Custom,     ///< Contiguous arrays (CustomSparse)
    Virtual,    ///< Pointer arrays (VirtualSparse)
    Mapped,     ///< Memory-mapped (MappedCustomSparse)
    Interface   ///< Virtual interface (ISparse)
};

/// @brief Detect backend type at compile time
template <typename M, bool IsCSR>
    requires SparseLike<M, IsCSR>
constexpr BackendType detect_backend() {
    if constexpr (MappedSparseLike<M, IsCSR>) {
        return BackendType::Mapped;
    } else if constexpr (CustomSparseLike<M, IsCSR>) {
        return BackendType::Custom;
    } else if constexpr (VirtualSparseLike<M, IsCSR>) {
        return BackendType::Virtual;
    } else {
        return BackendType::Interface;
    }
}

// =============================================================================
// Streaming Configuration
// =============================================================================

/// @brief Configuration for streaming operations
struct StreamConfig {
    Size chunk_rows = 256;          ///< Rows per chunk for streaming
    Size prefetch_chunks = 2;       ///< Number of chunks to prefetch ahead
    Size cache_size_mb = 64;        ///< Maximum cache size in MB
    bool enable_prefetch = true;    ///< Whether to use prefetch hints
};

/// @brief Default streaming configuration
inline constexpr StreamConfig default_stream_config() {
    return StreamConfig{};
}

// =============================================================================
// Memory Hints
// =============================================================================

/// @brief Issue prefetch hint for mapped data
template <typename M>
SCL_FORCE_INLINE void hint_prefetch(const M& matrix) {
    if constexpr (detail::HasPrefetch<M>) {
        matrix.prefetch();
    }
}

/// @brief Release cache hint for mapped data
template <typename M>
SCL_FORCE_INLINE void hint_drop_cache(const M& matrix) {
    if constexpr (detail::HasDropCache<M>) {
        matrix.drop_cache();
    }
}

/// @brief Prefetch memory address for read
SCL_FORCE_INLINE void prefetch_read(const void* addr) {
    SCL_PREFETCH_READ(addr, 0);
}

/// @brief Prefetch memory address for write
SCL_FORCE_INLINE void prefetch_write(void* addr) {
    SCL_PREFETCH_WRITE(addr, 0);
}

// =============================================================================
// Chunk Iterator (Basic)
// =============================================================================

/// @brief Iterator for chunk-based processing
///
/// Enables efficient streaming over large mapped matrices
/// by processing rows in chunks.
struct ChunkIterator {
    Index start;        ///< First row in chunk
    Index end;          ///< One past last row in chunk
    Index chunk_id;     ///< Chunk index

    [[nodiscard]] SCL_FORCE_INLINE Index size() const { return end - start; }
    [[nodiscard]] SCL_FORCE_INLINE bool empty() const { return start >= end; }
};

/// @brief Generate uniform chunk iterators for streaming
inline std::vector<ChunkIterator> make_chunks(
    Index total_rows,
    Size chunk_size
) {
    if (total_rows <= 0 || chunk_size == 0) return {};

    const Index cs = static_cast<Index>(chunk_size);
    const Index n_chunks = (total_rows + cs - 1) / cs;

    std::vector<ChunkIterator> chunks;
    chunks.reserve(static_cast<Size>(n_chunks));

    for (Index i = 0; i < n_chunks; ++i) {
        Index start = i * cs;
        Index end = std::min(start + cs, total_rows);
        chunks.push_back({start, end, i});
    }

    return chunks;
}

// =============================================================================
// Load-Balanced Chunk Iterator (NNZ-Weighted)
// =============================================================================

/// @brief Balanced work range for parallel processing
struct BalancedRange {
    Index start;        ///< First row
    Index end;          ///< One past last row
    Index nnz_start;    ///< NNZ offset for this range
    Index total_nnz;    ///< Total NNZ in this range
};

/// @brief Compute load-balanced ranges based on NNZ distribution
///
/// Partitions rows so each range has approximately equal NNZ,
/// ensuring balanced work distribution for parallel processing.
///
/// @param row_lengths Array of NNZ per row
/// @param n_rows Total number of rows
/// @param n_partitions Number of partitions (typically num_threads)
/// @return Vector of balanced ranges
inline std::vector<BalancedRange> compute_balanced_ranges(
    const Index* row_lengths,
    Index n_rows,
    Size n_partitions
) {
    if (n_rows <= 0 || n_partitions == 0) return {};

    // Compute prefix sums
    std::vector<Index> prefix(static_cast<Size>(n_rows) + 1);
    prefix[0] = 0;
    for (Index i = 0; i < n_rows; ++i) {
        prefix[i + 1] = prefix[i] + row_lengths[i];
    }

    const Index total_nnz = prefix[n_rows];
    if (total_nnz == 0) {
        // All empty rows - just divide evenly
        std::vector<BalancedRange> ranges(n_partitions);
        const Index rows_per = (n_rows + static_cast<Index>(n_partitions) - 1) / static_cast<Index>(n_partitions);
        for (Size p = 0; p < n_partitions; ++p) {
            ranges[p].start = static_cast<Index>(p) * rows_per;
            ranges[p].end = std::min(ranges[p].start + rows_per, n_rows);
            ranges[p].nnz_start = 0;
            ranges[p].total_nnz = 0;
        }
        return ranges;
    }

    const Index target_nnz = (total_nnz + static_cast<Index>(n_partitions) - 1) / static_cast<Index>(n_partitions);

    std::vector<BalancedRange> ranges;
    ranges.reserve(n_partitions);

    Index current_start = 0;
    for (Size p = 0; p < n_partitions && current_start < n_rows; ++p) {
        const Index target_end_nnz = std::min(prefix[current_start] + target_nnz, total_nnz);

        // Binary search for row where prefix sum exceeds target
        Index lo = current_start;
        Index hi = n_rows;
        while (lo < hi) {
            Index mid = lo + (hi - lo) / 2;
            if (prefix[mid + 1] <= target_end_nnz) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // Ensure at least one row per partition
        Index current_end = std::max(lo, current_start + 1);
        if (p == n_partitions - 1) {
            current_end = n_rows;  // Last partition gets all remaining
        }

        ranges.push_back({
            current_start,
            current_end,
            prefix[current_start],
            prefix[current_end] - prefix[current_start]
        });

        current_start = current_end;
    }

    return ranges;
}

/// @brief Compute balanced ranges from indptr array (CSR/CSC format)
template <typename MatrixT>
inline std::vector<BalancedRange> compute_balanced_ranges_from_indptr(
    const MatrixT& matrix,
    Size n_partitions
) {
    const Index n_primary = scl::primary_size(matrix);
    if (n_primary <= 0) return {};

    // Extract row lengths from indptr
    std::vector<Index> lengths(static_cast<Size>(n_primary));
    for (Index i = 0; i < n_primary; ++i) {
        lengths[i] = matrix.indptr[i + 1] - matrix.indptr[i];
    }

    return compute_balanced_ranges(lengths.data(), n_primary, n_partitions);
}

// =============================================================================
// SIMD-Optimized Statistics
// =============================================================================

namespace detail {

/// @brief SIMD accumulate sum and sum-of-squares (4-way unroll)
template <typename T>
SCL_FORCE_INLINE void accumulate_simd(
    const T* SCL_RESTRICT values,
    Size count,
    T& sum,
    T& sum_sq
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sq0 = s::Zero(d);
    auto v_sq1 = s::Zero(d);

    Size i = 0;
    const Size simd_end = count - (count % (lanes * 4));

    // 4-way unrolled SIMD loop
    for (; i < simd_end; i += lanes * 4) {
        auto v0 = s::Load(d, values + i);
        auto v1 = s::Load(d, values + i + lanes);
        auto v2 = s::Load(d, values + i + lanes * 2);
        auto v3 = s::Load(d, values + i + lanes * 3);

        v_sum0 = s::Add(v_sum0, v0);
        v_sum1 = s::Add(v_sum1, v1);
        v_sum0 = s::Add(v_sum0, v2);
        v_sum1 = s::Add(v_sum1, v3);

        v_sq0 = s::MulAdd(v0, v0, v_sq0);
        v_sq1 = s::MulAdd(v1, v1, v_sq1);
        v_sq0 = s::MulAdd(v2, v2, v_sq0);
        v_sq1 = s::MulAdd(v3, v3, v_sq1);
    }

    // Handle remaining SIMD vectors
    for (; i + lanes <= count; i += lanes) {
        auto v = s::Load(d, values + i);
        v_sum0 = s::Add(v_sum0, v);
        v_sq0 = s::MulAdd(v, v, v_sq0);
    }

    // Reduce SIMD to scalars
    v_sum0 = s::Add(v_sum0, v_sum1);
    v_sq0 = s::Add(v_sq0, v_sq1);
    sum += s::ReduceSum(d, v_sum0);
    sum_sq += s::ReduceSum(d, v_sq0);

    // Scalar tail
    for (; i < count; ++i) {
        T v = values[i];
        sum += v;
        sum_sq += v * v;
    }
}

/// @brief SIMD sum only (4-way unroll)
template <typename T>
SCL_FORCE_INLINE T sum_simd(const T* SCL_RESTRICT values, Size count) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);

    Size i = 0;
    const Size simd_end = count - (count % (lanes * 4));

    for (; i < simd_end; i += lanes * 4) {
        v_sum0 = s::Add(v_sum0, s::Load(d, values + i));
        v_sum1 = s::Add(v_sum1, s::Load(d, values + i + lanes));
        v_sum0 = s::Add(v_sum0, s::Load(d, values + i + lanes * 2));
        v_sum1 = s::Add(v_sum1, s::Load(d, values + i + lanes * 3));
    }

    for (; i + lanes <= count; i += lanes) {
        v_sum0 = s::Add(v_sum0, s::Load(d, values + i));
    }

    v_sum0 = s::Add(v_sum0, v_sum1);
    T result = s::ReduceSum(d, v_sum0);

    for (; i < count; ++i) {
        result += values[i];
    }

    return result;
}

/// @brief SIMD add offset to index array
SCL_FORCE_INLINE void add_offset_simd(
    const Index* SCL_RESTRICT src,
    Index* SCL_RESTRICT dst,
    Size count,
    Index offset
) {
    namespace s = scl::simd;
    const s::IndexTag d;
    const size_t lanes = s::Lanes(d);

    const auto v_offset = s::Set(d, offset);

    Size i = 0;
    const Size simd_end = count - (count % (lanes * 2));

    // 2-way unrolled SIMD
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

    // Scalar tail
    for (; i < count; ++i) {
        dst[i] = src[i] + offset;
    }
}

/// @brief SIMD copy with prefetch
template <typename T>
SCL_FORCE_INLINE void copy_prefetch(
    const T* SCL_RESTRICT src,
    T* SCL_RESTRICT dst,
    Size count,
    Size prefetch_distance = 8
) {
    constexpr Size cache_line = 64 / sizeof(T);

    for (Size i = 0; i < count; ++i) {
        if ((i % cache_line) == 0 && i + prefetch_distance * cache_line < count) {
            SCL_PREFETCH_READ(src + i + prefetch_distance * cache_line, 0);
            SCL_PREFETCH_WRITE(dst + i + prefetch_distance * cache_line, 0);
        }
        dst[i] = src[i];
    }
}

} // namespace detail

// =============================================================================
// Welford Online Algorithm (Optimized)
// =============================================================================

/// @brief Welford's online algorithm state
///
/// Computes mean and variance in a single pass with O(1) memory.
/// Numerically stable for large datasets.
template <typename T>
struct WelfordState {
    T mean = 0;
    T m2 = 0;      // Sum of squared differences from current mean
    Index n = 0;   // Count

    /// @brief Update with new value
    SCL_FORCE_INLINE void update(T value) {
        ++n;
        T delta = value - mean;
        mean += delta / static_cast<T>(n);
        T delta2 = value - mean;
        m2 += delta * delta2;
    }

    /// @brief Batch update with array (more efficient than repeated single updates)
    void update_batch(const T* values, Size count) {
        for (Size i = 0; i < count; ++i) {
            update(values[i]);
        }
    }

    /// @brief Get population variance
    [[nodiscard]] SCL_FORCE_INLINE T variance() const {
        return (n > 0) ? m2 / static_cast<T>(n) : T(0);
    }

    /// @brief Get sample variance (with ddof)
    [[nodiscard]] SCL_FORCE_INLINE T variance(int ddof) const {
        T denom = static_cast<T>(n - ddof);
        return (denom > T(0)) ? m2 / denom : T(0);
    }

    /// @brief Get standard deviation
    [[nodiscard]] SCL_FORCE_INLINE T std() const {
        return std::sqrt(variance());
    }

    /// @brief Merge two Welford states (parallel reduction)
    void merge(const WelfordState& other) {
        if (other.n == 0) return;
        if (n == 0) {
            *this = other;
            return;
        }

        Index combined_n = n + other.n;
        T delta = other.mean - mean;
        T new_mean = mean + delta * static_cast<T>(other.n) / static_cast<T>(combined_n);

        // Chan's parallel algorithm for variance
        m2 += other.m2 + delta * delta *
              static_cast<T>(n) * static_cast<T>(other.n) / static_cast<T>(combined_n);
        mean = new_mean;
        n = combined_n;
    }
};

// =============================================================================
// Sparse Row Statistics (SIMD Optimized)
// =============================================================================

/// @brief Single-pass row statistics for sparse matrices (SIMD optimized)
///
/// Computes sum, mean, variance, nnz in one pass.
/// Uses SIMD for hot accumulation loops.
template <typename T>
struct RowStats {
    T sum = 0;
    T sum_sq = 0;
    Index nnz_count = 0;

    /// @brief Accumulate values (SIMD optimized)
    SCL_FORCE_INLINE void accumulate(const T* values, Size count) {
        if (count == 0) return;

        detail::accumulate_simd(values, count, sum, sum_sq);
        nnz_count += static_cast<Index>(count);
    }

    /// @brief Accumulate single value
    SCL_FORCE_INLINE void accumulate_one(T value) {
        sum += value;
        sum_sq += value * value;
        ++nnz_count;
    }

    [[nodiscard]] SCL_FORCE_INLINE T mean(Index total_cols) const {
        return (total_cols > 0) ? sum / static_cast<T>(total_cols) : T(0);
    }

    [[nodiscard]] T variance(Index total_cols, int ddof = 0) const {
        if (total_cols <= ddof) return T(0);
        T mu = mean(total_cols);
        T denom = static_cast<T>(total_cols - ddof);
        // Var = E[X^2] - E[X]^2, adjusted for sparse zeros
        T ex2 = sum_sq / static_cast<T>(total_cols);
        T var = ex2 - mu * mu;
        // Apply Bessel's correction
        var = var * static_cast<T>(total_cols) / denom;
        return (var < T(0)) ? T(0) : var;
    }

    [[nodiscard]] SCL_FORCE_INLINE T std(Index total_cols, int ddof = 0) const {
        return std::sqrt(variance(total_cols, ddof));
    }

    /// @brief Merge two RowStats (for parallel reduction)
    void merge(const RowStats& other) {
        sum += other.sum;
        sum_sq += other.sum_sq;
        nnz_count += other.nnz_count;
    }
};

// =============================================================================
// Parallel Reduction Utilities
// =============================================================================

/// @brief Parallel sum of array
template <typename T>
inline T parallel_sum(const T* values, Size count) {
    if (count < 10000) {
        return detail::sum_simd(values, count);
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    const Size chunk = (count + n_threads - 1) / n_threads;

    std::vector<T> partial(n_threads, T(0));

    scl::threading::parallel_for(0, n_threads, [&](size_t tid) {
        Size start = tid * chunk;
        Size end = std::min(start + chunk, count);
        if (start < end) {
            partial[tid] = detail::sum_simd(values + start, end - start);
        }
    });

    T total = T(0);
    for (size_t i = 0; i < n_threads; ++i) {
        total += partial[i];
    }
    return total;
}

/// @brief Parallel prefix sum (exclusive scan)
inline void parallel_prefix_sum(
    const Index* input,
    Index* output,
    Size count
) {
    if (count == 0) return;

    // For small arrays, use serial
    if (count < 10000) {
        output[0] = 0;
        for (Size i = 0; i < count; ++i) {
            output[i + 1] = output[i] + input[i];
        }
        return;
    }

    // Two-pass parallel prefix sum
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    const Size chunk = (count + n_threads - 1) / n_threads;

    std::vector<Index> block_sums(n_threads + 1, 0);

    // Pass 1: Local prefix sums and block totals
    scl::threading::parallel_for(0, n_threads, [&](size_t tid) {
        Size start = tid * chunk;
        Size end = std::min(start + chunk, count);

        Index local_sum = 0;
        for (Size i = start; i < end; ++i) {
            output[i] = local_sum;
            local_sum += input[i];
        }
        block_sums[tid + 1] = local_sum;
    });

    // Sequential: Compute block offsets
    for (size_t i = 1; i <= n_threads; ++i) {
        block_sums[i] += block_sums[i - 1];
    }

    // Pass 2: Add block offsets
    scl::threading::parallel_for(0, n_threads, [&](size_t tid) {
        Size start = tid * chunk;
        Size end = std::min(start + chunk, count);
        Index offset = block_sums[tid];

        for (Size i = start; i < end; ++i) {
            output[i] += offset;
        }
    });

    output[count] = block_sums[n_threads];
}

} // namespace scl::kernel::mapped

