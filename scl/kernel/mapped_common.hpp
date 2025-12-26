#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/io/mmatrix.hpp"

#include <cstddef>
#include <vector>
#include <algorithm>

// =============================================================================
/// @file mapped_common.hpp
/// @brief Mapped Backend Common Definitions
///
/// Defines:
/// - MappedSparseLike concept
/// - Unified dispatchers for Custom/Virtual/Mapped
/// - Common utilities for mapped processing
///
/// Design Philosophy:
/// - One set of operators, all backends unified
/// - Compile-time dispatch via concepts
/// - Zero overhead abstraction
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
inline StreamConfig default_stream_config() {
    return StreamConfig{};
}

// =============================================================================
// Memory Hints
// =============================================================================

/// @brief Issue prefetch hint for mapped data
template <typename M>
inline void hint_prefetch(const M& matrix) {
    if constexpr (detail::HasPrefetch<M>) {
        matrix.prefetch();
    }
}

/// @brief Release cache hint for mapped data
template <typename M>
inline void hint_drop_cache(const M& matrix) {
    if constexpr (detail::HasDropCache<M>) {
        matrix.drop_cache();
    }
}

// =============================================================================
// Chunk Iterator
// =============================================================================

/// @brief Iterator for chunk-based processing
///
/// Enables efficient streaming over large mapped matrices
/// by processing rows in chunks.
struct ChunkIterator {
    Index start;        ///< First row in chunk
    Index end;          ///< One past last row in chunk
    Index chunk_id;     ///< Chunk index

    [[nodiscard]] Index size() const { return end - start; }
    [[nodiscard]] bool empty() const { return start >= end; }
};

/// @brief Generate chunk iterators for streaming
inline std::vector<ChunkIterator> make_chunks(
    Index total_rows,
    Size chunk_size
) {
    std::vector<ChunkIterator> chunks;
    Index n_chunks = (total_rows + static_cast<Index>(chunk_size) - 1) / static_cast<Index>(chunk_size);
    chunks.reserve(static_cast<Size>(n_chunks));

    for (Index i = 0; i < n_chunks; ++i) {
        Index start = i * static_cast<Index>(chunk_size);
        Index end = std::min(start + static_cast<Index>(chunk_size), total_rows);
        chunks.push_back({start, end, i});
    }

    return chunks;
}

// =============================================================================
// Welford Online Algorithm
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
    void update(T value) {
        ++n;
        T delta = value - mean;
        mean += delta / static_cast<T>(n);
        T delta2 = value - mean;
        m2 += delta * delta2;
    }

    /// @brief Get population variance
    [[nodiscard]] T variance() const {
        return (n > 0) ? m2 / static_cast<T>(n) : 0;
    }

    /// @brief Get sample variance (with ddof)
    [[nodiscard]] T variance(int ddof) const {
        T denom = static_cast<T>(n - ddof);
        return (denom > 0) ? m2 / denom : 0;
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
// Sparse Row Statistics (Single Pass)
// =============================================================================

/// @brief Single-pass row statistics for sparse matrices
///
/// Computes sum, mean, variance, nnz in one pass.
/// Optimized for mapped data where multiple passes are expensive.
template <typename T>
struct RowStats {
    T sum = 0;
    T sum_sq = 0;
    Index nnz_count = 0;

    void accumulate(const T* values, Size count) {
        for (Size i = 0; i < count; ++i) {
            T v = values[i];
            sum += v;
            sum_sq += v * v;
        }
        nnz_count += static_cast<Index>(count);
    }

    [[nodiscard]] T mean(Index total_cols) const {
        return sum / static_cast<T>(total_cols);
    }

    [[nodiscard]] T variance(Index total_cols, int ddof = 0) const {
        T mu = mean(total_cols);
        T denom = static_cast<T>(total_cols - ddof);
        if (denom <= 0) return 0;
        // Var = E[X^2] - E[X]^2, adjusted for sparse zeros
        T ex2 = sum_sq / static_cast<T>(total_cols);
        T var = ex2 - mu * mu;
        // Apply Bessel's correction
        var = var * static_cast<T>(total_cols) / denom;
        return (var < 0) ? 0 : var;
    }
};

} // namespace scl::kernel::mapped
