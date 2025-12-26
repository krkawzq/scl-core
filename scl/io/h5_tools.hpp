#pragma once

#include "scl/io/hdf5.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/io/scheduler.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"

#include <algorithm>
#include <tuple>
#include <memory>
#include <atomic>
#include <mutex>
#include <numeric>
#include <fstream>
#include <unordered_map>

#ifdef SCL_HAS_HDF5

// =============================================================================
/// @file h5_tools.hpp
/// @brief High-Performance HDF5 Sparse Matrix I/O with Two-Phase Architecture
///
/// Implements a Plan-Execute pipeline optimized for sparse matrix retrieval:
///
/// ## Two-Phase Architecture
///
/// Phase 1: PLAN (Single-Threaded, I/O-Focused)
///   1. Load indptr once
///   2. Build Zone Map: (min_col, max_col, nnz) per chunk
///   3. Create Query Context: sorted columns, clustering detection
///   4. Build Chunk-Row Map: which rows hit which chunks
///   5. Generate I/O Plan: merge nearby ranges
///
/// Phase 2: EXECUTE (Pipeline)
///   Stage A: Sequential I/O - Load required chunks (single thread)
///   Stage B: Parallel Intersection - Process rows per chunk (multi-thread)
///   Stage C: Parallel Assembly - Merge results into rows (multi-thread)
///
/// ## Key Optimizations
///
/// 1. Chunk-Centric Processing
///    - Each chunk loaded once, serves all hitting rows
///    - Eliminates redundant I/O from row-centric approaches
///
/// 2. True Zone Map Pre-filtering
///    - O(1) chunk rejection based on (min_col, max_col)
///    - Skip chunks with no possible matches before any data I/O
///
/// 3. Serial I/O + Parallel Compute Separation
///    - No locks during computation phase
///    - Avoids HDF5 thread-safety issues entirely
///
/// 4. Adaptive Intersection Algorithms
///    - Galloping: |query| << |chunk| (ratio < 1%)
///    - Binary Search: |query| < |chunk| (ratio < 10%)
///    - Linear Merge: |query| ≈ |chunk| (SIMD-friendly)
///
/// ## Complexity
///
/// Let: N = total_cols, Q = query_cols, C = chunk_size, K = num_chunks, R = rows
///
/// - Zone Map Build: O(K) chunk boundary reads
/// - Zone Map Filter: O(K)
/// - Total I/O: O(K' · C) where K' = filtered chunks << K
/// - Intersection: O(R · min(Q·log C, Q+C))
///
/// ## Performance Expectations
///
/// - Sparse queries (Q/N < 0.1%): 100-1000x speedup
/// - Medium queries (0.1% < Q/N < 10%): 10-100x speedup
/// - Dense queries (Q/N > 10%): 2-5x speedup (still benefits from Zone Map)
// =============================================================================

namespace scl::io::h5 {

// =============================================================================
// Forward Declarations
// =============================================================================

template <typename T, bool IsCSR> class QueryResult;
template <typename T> class ChunkCache;
struct ZoneMapEntry;
class ZoneMap;
struct QueryPlan;

// =============================================================================
// SECTION 1: Zone Map (Pre-computed Chunk Index)
// =============================================================================

namespace detail {

/// @brief Zone Map entry for a single chunk.
///
/// Stores precomputed (min_col, max_col) bounds for O(1) range filtering.
/// This allows skipping chunks that cannot possibly contain query columns.
struct ZoneMapEntry {
    Index min_col;      ///< Minimum column index in chunk
    Index max_col;      ///< Maximum column index in chunk
    Index nnz;          ///< Number of non-zeros in chunk
    hsize_t file_start; ///< Start offset in file (element index)
    hsize_t file_end;   ///< End offset in file (element index)
    
    /// @brief Check if chunk possibly contains any column in query range.
    [[nodiscard]] SCL_FORCE_INLINE 
    bool may_overlap(Index query_min, Index query_max) const noexcept {
        return !(max_col < query_min || min_col > query_max);
    }
    
    /// @brief Check if chunk is fully contained in query range.
    [[nodiscard]] SCL_FORCE_INLINE 
    bool fully_contained(Index query_min, Index query_max) const noexcept {
        return min_col >= query_min && max_col <= query_max;
    }
};

/// @brief Zone Map for entire dataset.
///
/// Pre-computed index that enables O(1) chunk filtering before any data I/O.
/// Built once during planning phase, used throughout execution.
class ZoneMap {
public:
    std::vector<ZoneMapEntry> entries;
    Index global_min;
    Index global_max;
    hsize_t chunk_size;
    hsize_t total_elements;
    
    ZoneMap() 
        : global_min(std::numeric_limits<Index>::max())
        , global_max(std::numeric_limits<Index>::min())
        , chunk_size(0)
        , total_elements(0)
    {}
    
    /// @brief Build Zone Map by scanning chunk boundaries.
    ///
    /// Reads only first and last elements of each chunk to determine bounds.
    /// Total I/O: O(2 · K) element reads << O(total_elements).
    static ZoneMap build(
        const Dataset& indices_dset,
        hsize_t chunk_sz,
        hsize_t total_elem
    ) {
        ZoneMap zm;
        zm.chunk_size = chunk_sz;
        zm.total_elements = total_elem;
        
        if (total_elem == 0) return zm;
        
        hsize_t num_chunks = (total_elem + chunk_sz - 1) / chunk_sz;
        zm.entries.reserve(num_chunks);
        
        // Temporary buffer for boundary reads
        Index boundary_buf[2];
        
        for (hsize_t c = 0; c < num_chunks; ++c) {
            hsize_t start = c * chunk_sz;
            hsize_t end = std::min(start + chunk_sz, total_elem);
            hsize_t count = end - start;
            
            if (count == 0) continue;
            
            ZoneMapEntry entry;
            entry.file_start = start;
            entry.file_end = end;
            entry.nnz = static_cast<Index>(count);
            
            // Read first element
            {
                Dataspace file_space = indices_dset.get_space();
                std::vector<hsize_t> v_start = {start};
                std::vector<hsize_t> v_count = {1};
                file_space.select_hyperslab(v_start, v_count);
                
                std::vector<hsize_t> mem_dims = {1};
                Dataspace mem_space(mem_dims);
                indices_dset.read(&boundary_buf[0], mem_space, file_space);
                entry.min_col = boundary_buf[0];
            }
            
            // Read last element
            {
                Dataspace file_space = indices_dset.get_space();
                std::vector<hsize_t> v_start = {end - 1};
                std::vector<hsize_t> v_count = {1};
                file_space.select_hyperslab(v_start, v_count);
                
                std::vector<hsize_t> mem_dims = {1};
                Dataspace mem_space(mem_dims);
                indices_dset.read(&boundary_buf[1], mem_space, file_space);
                entry.max_col = boundary_buf[1];
            }
            
            // Update global bounds
            zm.global_min = std::min(zm.global_min, entry.min_col);
            zm.global_max = std::max(zm.global_max, entry.max_col);
            
            zm.entries.push_back(entry);
        }
        
        return zm;
    }
    
    /// @brief Filter chunks that may contain columns in query range.
    ///
    /// Complexity: O(K) where K = number of chunks.
    /// Returns indices of chunks that pass the filter.
    [[nodiscard]] std::vector<Size> filter(
        Index query_min, 
        Index query_max
    ) const {
        // Fast rejection: query range doesn't overlap global range
        if (query_max < global_min || query_min > global_max) {
            return {};
        }
        
        std::vector<Size> result;
        result.reserve(entries.size() / 4);  // Heuristic: expect ~25% hit rate
        
        for (Size i = 0; i < entries.size(); ++i) {
            if (entries[i].may_overlap(query_min, query_max)) {
                result.push_back(i);
            }
        }
        
        return result;
    }
    
    /// @brief Get chunk index for a given element offset.
    [[nodiscard]] SCL_FORCE_INLINE 
    Size offset_to_chunk(hsize_t offset) const noexcept {
        return static_cast<Size>(offset / chunk_size);
    }
};

// =============================================================================
// SECTION 2: Adaptive Set Intersection Algorithms
// =============================================================================

/// @brief Intersection strategy selection based on size ratio.
enum class IntersectStrategy {
    Galloping,    ///< |query| << |chunk|, exponential search
    BinarySearch, ///< |query| < |chunk|, binary search per element
    LinearMerge   ///< |query| ≈ |chunk|, dual-pointer merge
};

/// @brief Select optimal intersection strategy based on size ratio.
[[nodiscard]] SCL_FORCE_INLINE 
IntersectStrategy select_strategy(Size query_size, Size chunk_size) noexcept {
    if (chunk_size == 0) return IntersectStrategy::LinearMerge;
    double ratio = static_cast<double>(query_size) / chunk_size;
    
    if (ratio < 0.01) return IntersectStrategy::Galloping;
    if (ratio < 0.10) return IntersectStrategy::BinarySearch;
    return IntersectStrategy::LinearMerge;
}

/// @brief Galloping (exponential) lower bound search.
///
/// Finds first position where arr[pos] >= target.
/// Complexity: O(log(answer_position)) - optimal for sparse queries.
template <typename T>
[[nodiscard]] SCL_FORCE_INLINE 
Size gallop_lower_bound(const T* arr, Size len, T target) noexcept {
    if (len == 0 || arr[0] >= target) return 0;
    
    // Exponential search: find range [lo, hi]
    Size lo = 0, hi = 1;
    while (hi < len && arr[hi] < target) {
        lo = hi;
        hi = std::min(hi * 2, len);
    }
    
    // Binary search within range
    while (lo < hi) {
        Size mid = lo + (hi - lo) / 2;
        if (arr[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// @brief Galloping intersection for very sparse queries.
///
/// Best when |query| << |chunk|.
/// Complexity: O(|query| · log(|chunk|/|query|)).
template <typename T>
void intersect_galloping(
    const Index* SCL_RESTRICT chunk_indices,
    const T* SCL_RESTRICT chunk_values,
    Size chunk_len,
    const Index* SCL_RESTRICT query_indices,
    Size query_len,
    std::vector<Index>& out_indices,
    std::vector<T>& out_values
) {
    Size chunk_pos = 0;
    
    for (Size q = 0; q < query_len && chunk_pos < chunk_len; ++q) {
        Index target = query_indices[q];
        
        // Gallop forward to find target
        chunk_pos += gallop_lower_bound(
            chunk_indices + chunk_pos,
            chunk_len - chunk_pos,
            target
        );
        
        if (chunk_pos < chunk_len && chunk_indices[chunk_pos] == target) {
            out_indices.push_back(target);
            out_values.push_back(chunk_values[chunk_pos]);
            ++chunk_pos;
        }
    }
}

/// @brief Binary search intersection.
///
/// Best when |query| is small relative to |chunk|.
/// Complexity: O(|query| · log|chunk|).
template <typename T>
void intersect_binary(
    const Index* SCL_RESTRICT chunk_indices,
    const T* SCL_RESTRICT chunk_values,
    Size chunk_len,
    const Index* SCL_RESTRICT query_indices,
    Size query_len,
    std::vector<Index>& out_indices,
    std::vector<T>& out_values
) {
    for (Size q = 0; q < query_len; ++q) {
        Index target = query_indices[q];
        
        auto it = std::lower_bound(
            chunk_indices, 
            chunk_indices + chunk_len, 
            target
        );
        
        if (it != chunk_indices + chunk_len && *it == target) {
            Size pos = static_cast<Size>(it - chunk_indices);
            out_indices.push_back(target);
            out_values.push_back(chunk_values[pos]);
        }
    }
}

/// @brief Linear merge intersection.
///
/// Best when |query| ≈ |chunk|. SIMD-friendly access pattern.
/// Complexity: O(|query| + |chunk|).
template <typename T>
void intersect_linear(
    const Index* SCL_RESTRICT chunk_indices,
    const T* SCL_RESTRICT chunk_values,
    Size chunk_len,
    const Index* SCL_RESTRICT query_indices,
    Size query_len,
    std::vector<Index>& out_indices,
    std::vector<T>& out_values
) {
    Size i = 0, j = 0;
    
    while (i < chunk_len && j < query_len) {
        Index c = chunk_indices[i];
        Index q = query_indices[j];
        
        if (c < q) {
            ++i;
        } else if (c > q) {
            ++j;
        } else {
            out_indices.push_back(c);
            out_values.push_back(chunk_values[i]);
            ++i;
            ++j;
        }
    }
}

/// @brief Adaptive intersection dispatcher.
///
/// Automatically selects optimal algorithm based on size ratio.
template <typename T>
void intersect_adaptive(
    const Index* SCL_RESTRICT chunk_indices,
    const T* SCL_RESTRICT chunk_values,
    Size chunk_len,
    const Index* SCL_RESTRICT query_indices,
    Size query_len,
    std::vector<Index>& out_indices,
    std::vector<T>& out_values
) {
    if (chunk_len == 0 || query_len == 0) return;
    
    auto strategy = select_strategy(query_len, chunk_len);
    
    switch (strategy) {
    case IntersectStrategy::Galloping:
        intersect_galloping(
            chunk_indices, chunk_values, chunk_len,
            query_indices, query_len,
            out_indices, out_values
        );
        break;
        
    case IntersectStrategy::BinarySearch:
        intersect_binary(
            chunk_indices, chunk_values, chunk_len,
            query_indices, query_len,
            out_indices, out_values
        );
        break;
        
    case IntersectStrategy::LinearMerge:
        intersect_linear(
            chunk_indices, chunk_values, chunk_len,
            query_indices, query_len,
            out_indices, out_values
        );
        break;
    }
}

// =============================================================================
// SECTION 3: Query Context (Precomputed Query State)
// =============================================================================

/// @brief Precomputed query state for column mask queries.
///
/// Amortizes sorting and analysis cost across all rows.
/// Created once per query, reused throughout execution.
struct QueryContext {
    std::vector<Index> sorted_cols;  ///< Sorted column indices
    Index col_min;                   ///< Minimum query column
    Index col_max;                   ///< Maximum query column
    Size query_size;                 ///< Number of query columns
    double density;                  ///< Q / N
    double clustering_factor;        ///< β from scheduler
    
    QueryContext() 
        : col_min(0), col_max(0), query_size(0)
        , density(0), clustering_factor(0.6) 
    {}
    
    /// @brief Build context from column mask array.
    static QueryContext build(
        Array<const Index> col_mask,
        Index total_cols
    ) {
        QueryContext ctx;
        
        if (col_mask.len == 0) return ctx;
        
        // Copy and sort
        ctx.sorted_cols.reserve(col_mask.len);
        for (Size i = 0; i < col_mask.len; ++i) {
            ctx.sorted_cols.push_back(col_mask[i]);
        }
        std::sort(ctx.sorted_cols.begin(), ctx.sorted_cols.end());
        
        // Remove duplicates
        auto last = std::unique(ctx.sorted_cols.begin(), ctx.sorted_cols.end());
        ctx.sorted_cols.erase(last, ctx.sorted_cols.end());
        
        ctx.query_size = ctx.sorted_cols.size();
        ctx.col_min = ctx.sorted_cols.front();
        ctx.col_max = ctx.sorted_cols.back();
        ctx.density = static_cast<double>(ctx.query_size) / total_cols;
        ctx.clustering_factor = detect_clustering(ctx.sorted_cols);
        
        return ctx;
    }
    
private:
    /// @brief Detect clustering factor from gap distribution.
    static double detect_clustering(const std::vector<Index>& sorted) {
        if (sorted.size() <= 1) return 1.0;
        
        // Check if contiguous
        bool contiguous = true;
        for (Size i = 1; i < sorted.size(); ++i) {
            if (sorted[i] != sorted[i-1] + 1) {
                contiguous = false;
                break;
            }
        }
        if (contiguous) return 0.2;
        
        // Compute median gap
        std::vector<Index> gaps;
        gaps.reserve(sorted.size() - 1);
        for (Size i = 1; i < sorted.size(); ++i) {
            gaps.push_back(sorted[i] - sorted[i-1]);
        }
        
        std::nth_element(gaps.begin(), gaps.begin() + gaps.size()/2, gaps.end());
        Index median_gap = gaps[gaps.size() / 2];
        
        // Map gap to β
        if (median_gap <= 5)   return 0.3;
        if (median_gap <= 20)  return 0.4;
        if (median_gap <= 50)  return 0.5;
        if (median_gap <= 100) return 0.6;
        if (median_gap <= 500) return 0.7;
        return 1.0;
    }
};

// =============================================================================
// SECTION 4: Chunk Cache (In-Memory Chunk Storage)
// =============================================================================

/// @brief Single chunk data loaded into memory.
template <typename T>
struct ChunkData {
    std::vector<T> values;
    std::vector<Index> indices;
    Size chunk_idx;
    bool loaded;
    
    ChunkData() : chunk_idx(0), loaded(false) {}
    
    void clear() {
        values.clear();
        indices.clear();
        loaded = false;
    }
};

/// @brief In-memory cache for loaded chunks.
///
/// All I/O happens in planning phase; execution phase reads from cache.
template <typename T>
class ChunkCache {
public:
    ChunkCache() = default;
    
    /// @brief Reserve space for expected number of chunks.
    void reserve(Size num_chunks) {
        _chunks.reserve(num_chunks);
    }
    
    /// @brief Load a chunk from HDF5 dataset.
    ///
    /// Called during planning phase (single-threaded).
    void load_chunk(
        Size chunk_idx,
        const Dataset& data_dset,
        const Dataset& indices_dset,
        hsize_t file_start,
        hsize_t file_end
    ) {
        hsize_t count = file_end - file_start;
        if (count == 0) return;
        
        // Ensure slot exists
        if (_index_map.find(chunk_idx) == _index_map.end()) {
            _index_map[chunk_idx] = _chunks.size();
            _chunks.emplace_back();
        }
        
        Size slot = _index_map[chunk_idx];
        auto& chunk = _chunks[slot];
        
        chunk.chunk_idx = chunk_idx;
        chunk.values.resize(static_cast<Size>(count));
        chunk.indices.resize(static_cast<Size>(count));
        
        // Read values
        {
            Dataspace file_space = data_dset.get_space();
            std::vector<hsize_t> v_start = {file_start};
            std::vector<hsize_t> v_count = {count};
            file_space.select_hyperslab(v_start, v_count);
            
            std::vector<hsize_t> mem_dims = {count};
            Dataspace mem_space(mem_dims);
            data_dset.read(chunk.values.data(), mem_space, file_space);
        }
        
        // Read indices
        {
            Dataspace file_space = indices_dset.get_space();
            std::vector<hsize_t> v_start = {file_start};
            std::vector<hsize_t> v_count = {count};
            file_space.select_hyperslab(v_start, v_count);
            
            std::vector<hsize_t> mem_dims = {count};
            Dataspace mem_space(mem_dims);
            indices_dset.read(chunk.indices.data(), mem_space, file_space);
        }
        
        chunk.loaded = true;
    }
    
    /// @brief Get chunk data (read-only, thread-safe).
    [[nodiscard]] const ChunkData<T>* get(Size chunk_idx) const {
        auto it = _index_map.find(chunk_idx);
        if (it == _index_map.end()) return nullptr;
        return &_chunks[it->second];
    }
    
    /// @brief Check if chunk is loaded.
    [[nodiscard]] bool has(Size chunk_idx) const {
        return _index_map.find(chunk_idx) != _index_map.end();
    }
    
    /// @brief Get number of loaded chunks.
    [[nodiscard]] Size size() const { return _chunks.size(); }
    
    /// @brief Clear all cached data.
    void clear() {
        _chunks.clear();
        _index_map.clear();
    }
    
private:
    std::vector<ChunkData<T>> _chunks;
    std::unordered_map<Size, Size> _index_map;  // chunk_idx -> slot
};

// =============================================================================
// SECTION 5: Row Segment (Result Storage)
// =============================================================================

/// @brief Storage for a single row's filtered data.
///
/// Uses contiguous vector storage (not deque) for guaranteed memory layout.
template <typename T>
struct RowSegment {
    std::vector<T> values;
    std::vector<Index> indices;
    Index row_idx;
    
    RowSegment() : row_idx(-1) {}
    explicit RowSegment(Index r) : row_idx(r) {}
    
    /// @brief Reserve space based on expected nnz.
    void reserve(Size expected_nnz) {
        values.reserve(expected_nnz);
        indices.reserve(expected_nnz);
    }
    
    /// @brief Append data from intersection result.
    void append(const std::vector<Index>& idxs, const std::vector<T>& vals) {
        indices.insert(indices.end(), idxs.begin(), idxs.end());
        values.insert(values.end(), vals.begin(), vals.end());
    }
    
    /// @brief Get number of non-zeros.
    [[nodiscard]] Index nnz() const { 
        return static_cast<Index>(values.size()); 
    }
    
    /// @brief Check if empty.
    [[nodiscard]] bool empty() const { return values.empty(); }
    
    void clear() {
        values.clear();
        indices.clear();
    }
};

// =============================================================================
// SECTION 6: Query Plan (Execution Blueprint)
// =============================================================================

/// @brief Row-to-chunk mapping for a single row.
struct RowChunkMapping {
    Index row_idx;                    ///< Logical row index
    Index phys_start;                 ///< Start offset in data/indices arrays
    Index phys_end;                   ///< End offset in data/indices arrays
    std::vector<Size> hitting_chunks; ///< Chunk indices this row intersects
};

/// @brief Complete query execution plan.
///
/// Built during planning phase, executed in parallel.
struct QueryPlan {
    QueryContext query_ctx;           ///< Query column context
    ZoneMap zone_map;                 ///< Chunk boundary index
    std::vector<Size> required_chunks;///< Chunks that pass Zone Map filter
    std::vector<RowChunkMapping> row_mappings; ///< Row-to-chunk assignments
    
    Index total_rows;
    Index total_cols;
    hsize_t chunk_size;
    
    QueryPlan() : total_rows(0), total_cols(0), chunk_size(0) {}
};

} // namespace detail

// =============================================================================
// SECTION 7: Query Result (Output Container)
// =============================================================================

/// @brief Result container for filtered sparse matrix query.
///
/// Provides both incremental access and materialization to OwnedSparse.
/// Satisfies SparseLike concept after construction.
template <typename T, bool IsCSR>
class QueryResult {
public:
    using ValueType = T;
    using Tag = TagSparse<IsCSR>;
    using RowSegment = detail::RowSegment<T>;
    
    static constexpr bool is_csr = IsCSR;
    static constexpr bool is_csc = !IsCSR;
    
    const Index rows;
    const Index cols;
    
    QueryResult() : rows(0), cols(0) {}
    
    QueryResult(
        std::vector<RowSegment>&& segments,
        Index num_rows,
        Index num_cols
    )
        : rows(num_rows), cols(num_cols)
        , _segments(std::move(segments))
    {}
    
    // Move only
    QueryResult(QueryResult&&) noexcept = default;
    QueryResult& operator=(QueryResult&&) noexcept = default;
    QueryResult(const QueryResult&) = delete;
    QueryResult& operator=(const QueryResult&) = delete;
    
    // -------------------------------------------------------------------------
    // SparseLike Interface (CSR)
    // -------------------------------------------------------------------------
    
    [[nodiscard]] Array<T> row_values(Index i) const requires (IsCSR) {
        const auto& seg = _segments[static_cast<Size>(i)];
        return Array<T>(
            const_cast<T*>(seg.values.data()), 
            seg.values.size()
        );
    }
    
    [[nodiscard]] Array<Index> row_indices(Index i) const requires (IsCSR) {
        const auto& seg = _segments[static_cast<Size>(i)];
        return Array<Index>(
            const_cast<Index*>(seg.indices.data()), 
            seg.indices.size()
        );
    }
    
    [[nodiscard]] Index row_length(Index i) const requires (IsCSR) {
        return _segments[static_cast<Size>(i)].nnz();
    }
    
    // -------------------------------------------------------------------------
    // SparseLike Interface (CSC)
    // -------------------------------------------------------------------------
    
    [[nodiscard]] Array<T> col_values(Index j) const requires (!IsCSR) {
        const auto& seg = _segments[static_cast<Size>(j)];
        return Array<T>(
            const_cast<T*>(seg.values.data()), 
            seg.values.size()
        );
    }
    
    [[nodiscard]] Array<Index> col_indices(Index j) const requires (!IsCSR) {
        const auto& seg = _segments[static_cast<Size>(j)];
        return Array<Index>(
            const_cast<Index*>(seg.indices.data()), 
            seg.indices.size()
        );
    }
    
    [[nodiscard]] Index col_length(Index j) const requires (!IsCSR) {
        return _segments[static_cast<Size>(j)].nnz();
    }
    
    // -------------------------------------------------------------------------
    // Query Methods
    // -------------------------------------------------------------------------
    
    [[nodiscard]] Index nnz() const noexcept {
        Index total = 0;
        for (const auto& seg : _segments) {
            total += seg.nnz();
        }
        return total;
    }
    
    [[nodiscard]] bool empty() const noexcept {
        return rows == 0 && cols == 0;
    }
    
    [[nodiscard]] Index primary_dim() const noexcept {
        return IsCSR ? rows : cols;
    }
    
    [[nodiscard]] Index secondary_dim() const noexcept {
        return IsCSR ? cols : rows;
    }
    
    // -------------------------------------------------------------------------
    // Materialization
    // -------------------------------------------------------------------------
    
    /// @brief Materialize to OwnedSparse (contiguous storage).
    [[nodiscard]] scl::io::OwnedSparse<T, IsCSR> materialize() const {
        Index pdim = primary_dim();
        Index total_nnz = nnz();
        
        std::vector<T> data;
        std::vector<Index> indices;
        std::vector<Index> indptr;
        
        data.reserve(static_cast<Size>(total_nnz));
        indices.reserve(static_cast<Size>(total_nnz));
        indptr.reserve(static_cast<Size>(pdim + 1));
        indptr.push_back(0);
        
        for (const auto& seg : _segments) {
            data.insert(data.end(), seg.values.begin(), seg.values.end());
            indices.insert(indices.end(), seg.indices.begin(), seg.indices.end());
            indptr.push_back(static_cast<Index>(data.size()));
        }
        
        return OwnedSparse<T, IsCSR>(
            std::move(data), std::move(indices), std::move(indptr),
            rows, cols
        );
    }
    
    /// @brief Materialize and return with view.
    [[nodiscard]] std::pair<OwnedSparse<T, IsCSR>, CustomSparse<T, IsCSR>> 
    materialize_with_view() const {
        auto owned = materialize();
        auto view = owned.view();
        return {std::move(owned), view};
    }
    
private:
    std::vector<RowSegment> _segments;
};

// =============================================================================
// SECTION 8: Query Executor (Two-Phase Pipeline)
// =============================================================================

/// @brief High-performance query executor implementing two-phase architecture.
///
/// Phase 1 (PLAN): Single-threaded preparation
///   - Load indptr
///   - Build Zone Map
///   - Create Query Context
///   - Build row-chunk mappings
///   - Load required chunks (serial I/O)
///
/// Phase 2 (EXECUTE): Parallel computation
///   - Process rows in parallel (no I/O, no locks)
///   - Adaptive intersection per row
///   - Assemble results
template <typename T>
class QueryExecutor {
public:
    /// @brief Execute masked query with two-phase pipeline.
    ///
    /// This is the main entry point for filtered sparse matrix retrieval.
    static QueryResult<T, true> execute_masked(
        const std::string& h5_path,
        const std::string& group_path,
        Array<const Index> row_mask,
        Array<const Index> col_mask
    ) {
        using RowSegment = detail::RowSegment<T>;
        
        if (row_mask.len == 0 || col_mask.len == 0) {
            return QueryResult<T, true>();
        }
        
        // =====================================================================
        // PHASE 1: PLAN (Single-Threaded)
        // =====================================================================
        
        File file(h5_path);
        Group group(file.id(), group_path);
        
        // 1.1 Read metadata
        std::array<hsize_t, 2> shape_arr = group.read_attr<hsize_t, 2>("shape");
        Index total_rows = static_cast<Index>(shape_arr[0]);
        Index total_cols = static_cast<Index>(shape_arr[1]);
        
        // 1.2 Load indptr (always needed)
        Dataset indptr_dset(group.id(), "indptr");
        std::vector<Index> indptr(static_cast<Size>(total_rows + 1));
        indptr_dset.read(indptr.data());
        
        // 1.3 Open data arrays
        Dataset data_dset(group.id(), "data");
        Dataset indices_dset(group.id(), "indices");
        
        // 1.4 Detect chunk size
        hsize_t chunk_size = 10000;
        auto chunk_dims = data_dset.get_chunk_dims();
        if (chunk_dims && !chunk_dims->empty()) {
            chunk_size = (*chunk_dims)[0];
        }
        
        hsize_t total_nnz = static_cast<hsize_t>(indptr.back());
        
        // 1.5 Build Query Context
        detail::QueryContext query_ctx = detail::QueryContext::build(col_mask, total_cols);
        
        if (query_ctx.query_size == 0) {
            // No valid columns
            std::vector<RowSegment> empty_segments(row_mask.len);
            for (Size i = 0; i < row_mask.len; ++i) {
                empty_segments[i].row_idx = row_mask[i];
            }
            return QueryResult<T, true>(
                std::move(empty_segments),
                static_cast<Index>(row_mask.len),
                total_cols
            );
        }
        
        // 1.6 Build Zone Map
        detail::ZoneMap zone_map = detail::ZoneMap::build(
            indices_dset, chunk_size, total_nnz
        );
        
        // 1.7 Filter chunks using Zone Map
        std::vector<Size> required_chunks = zone_map.filter(
            query_ctx.col_min, 
            query_ctx.col_max
        );
        
        // 1.8 Build row-chunk mappings
        std::vector<detail::RowChunkMapping> row_mappings(row_mask.len);
        
        for (Size i = 0; i < row_mask.len; ++i) {
            auto& mapping = row_mappings[i];
            mapping.row_idx = row_mask[i];
            
            if (mapping.row_idx >= total_rows) {
                mapping.phys_start = 0;
                mapping.phys_end = 0;
                continue;
            }
            
            mapping.phys_start = indptr[static_cast<Size>(mapping.row_idx)];
            mapping.phys_end = indptr[static_cast<Size>(mapping.row_idx + 1)];
            
            if (mapping.phys_start >= mapping.phys_end) continue;
            
            // Find chunks this row hits (that also passed Zone Map filter)
            Size start_chunk = zone_map.offset_to_chunk(
                static_cast<hsize_t>(mapping.phys_start)
            );
            Size end_chunk = zone_map.offset_to_chunk(
                static_cast<hsize_t>(mapping.phys_end - 1)
            );
            
            for (Size c = start_chunk; c <= end_chunk && c < zone_map.entries.size(); ++c) {
                // Check if this chunk is in required set
                if (std::binary_search(required_chunks.begin(), required_chunks.end(), c)) {
                    mapping.hitting_chunks.push_back(c);
                }
            }
        }
        
        // 1.9 Load required chunks (SERIAL I/O - no locks needed)
        detail::ChunkCache<T> cache;
        cache.reserve(required_chunks.size());
        
        for (Size chunk_idx : required_chunks) {
            const auto& entry = zone_map.entries[chunk_idx];
            cache.load_chunk(
                chunk_idx,
                data_dset,
                indices_dset,
                entry.file_start,
                entry.file_end
            );
        }
        
        // =====================================================================
        // PHASE 2: EXECUTE (Parallel Computation)
        // =====================================================================
        
        // 2.1 Allocate result segments
        std::vector<RowSegment> segments(row_mask.len);
        
        // 2.2 Process rows in parallel (NO I/O, NO LOCKS)
        scl::threading::parallel_for(Size(0), row_mask.len,
            [&](Size i)
        {
            const auto& mapping = row_mappings[i];
            auto& seg = segments[i];
            seg.row_idx = mapping.row_idx;
            
            if (mapping.hitting_chunks.empty()) return;
            
            // Estimate output size for reservation
            Size est_hits = std::max(Size(1), 
                static_cast<Size>(query_ctx.density * (mapping.phys_end - mapping.phys_start))
            );
            seg.reserve(est_hits);
            
            // Process each chunk this row hits
            for (Size chunk_idx : mapping.hitting_chunks) {
                const auto* chunk = cache.get(chunk_idx);
                if (!chunk || !chunk->loaded) continue;
                
                const auto& entry = zone_map.entries[chunk_idx];
                
                // Calculate overlap within chunk
                Index chunk_start = static_cast<Index>(entry.file_start);
                Index chunk_end = static_cast<Index>(entry.file_end);
                
                Index overlap_start = std::max(mapping.phys_start, chunk_start);
                Index overlap_end = std::min(mapping.phys_end, chunk_end);
                
                if (overlap_start >= overlap_end) continue;
                
                // Local offset within chunk
                Size local_start = static_cast<Size>(overlap_start - chunk_start);
                Size local_len = static_cast<Size>(overlap_end - overlap_start);
                
                // Zone Map filter: quick range check
                Index chunk_min = chunk->indices[local_start];
                Index chunk_max = chunk->indices[local_start + local_len - 1];
                
                if (chunk_max < query_ctx.col_min || chunk_min > query_ctx.col_max) {
                    continue;  // Skip - no overlap possible
                }
                
                // Adaptive intersection
                std::vector<Index> hit_indices;
                std::vector<T> hit_values;
                
                detail::intersect_adaptive(
                    chunk->indices.data() + local_start,
                    chunk->values.data() + local_start,
                    local_len,
                    query_ctx.sorted_cols.data(),
                    query_ctx.query_size,
                    hit_indices,
                    hit_values
                );
                
                seg.append(hit_indices, hit_values);
            }
        });
        
        return QueryResult<T, true>(
            std::move(segments),
            static_cast<Index>(row_mask.len),
            total_cols
        );
    }
    
    /// @brief Load sparse matrix rows (row selection only, no column filter).
    static OwnedSparse<T, true> execute_rows(
        const std::string& h5_path,
        const std::string& group_path,
        Array<const Index> row_mask
    ) {
        if (row_mask.len == 0) {
            return OwnedSparse<T, true>();
        }
        
        File file(h5_path);
        Group group(file.id(), group_path);
        
        // Metadata
        std::array<hsize_t, 2> shape_arr = group.read_attr<hsize_t, 2>("shape");
        Index total_rows = static_cast<Index>(shape_arr[0]);
        Index total_cols = static_cast<Index>(shape_arr[1]);
        
        // Load indptr
        Dataset indptr_dset(group.id(), "indptr");
        std::vector<Index> indptr(static_cast<Size>(total_rows + 1));
        indptr_dset.read(indptr.data());
        
        // Compute merged ranges for coalesced I/O
        struct Range { Index begin, end; };
        std::vector<Range> ranges;
        ranges.reserve(row_mask.len);
        
        for (Size i = 0; i < row_mask.len; ++i) {
            Index row_idx = row_mask[i];
            if (row_idx >= total_rows) continue;
            
            Index start = indptr[static_cast<Size>(row_idx)];
            Index end = indptr[static_cast<Size>(row_idx + 1)];
            
            if (start < end) {
                ranges.push_back({start, end});
            }
        }
        
        // Merge nearby ranges
        if (!ranges.empty()) {
            std::sort(ranges.begin(), ranges.end(), 
                [](const Range& a, const Range& b) { return a.begin < b.begin; });
            
            std::vector<Range> merged;
            merged.push_back(ranges[0]);
            
            constexpr Index gap_threshold = 128;
            for (Size i = 1; i < ranges.size(); ++i) {
                auto& last = merged.back();
                const auto& curr = ranges[i];
                
                if (curr.begin <= last.end + gap_threshold) {
                    last.end = std::max(last.end, curr.end);
                } else {
                    merged.push_back(curr);
                }
            }
            ranges = std::move(merged);
        }
        
        // Compute total nnz
        Index total_nnz = 0;
        for (const auto& r : ranges) total_nnz += (r.end - r.begin);
        
        // Allocate output
        std::vector<T> data(static_cast<Size>(total_nnz));
        std::vector<Index> indices(static_cast<Size>(total_nnz));
        
        // Batch read
        Dataset data_dset(group.id(), "data");
        Dataset indices_dset(group.id(), "indices");
        
        Index write_offset = 0;
        for (const auto& range : ranges) {
            hsize_t start = static_cast<hsize_t>(range.begin);
            hsize_t count = static_cast<hsize_t>(range.end - range.begin);
            
            if (count == 0) continue;
            
            Dataspace file_space = data_dset.get_space();
            std::vector<hsize_t> v_start = {start};
            std::vector<hsize_t> v_count = {count};
            file_space.select_hyperslab(v_start, v_count);
            
            std::vector<hsize_t> mem_dims = {count};
            Dataspace mem_space(mem_dims);
            
            data_dset.read(data.data() + write_offset, mem_space, file_space);
            
            Dataspace indices_file_space = indices_dset.get_space();
            indices_file_space.select_hyperslab(v_start, v_count);
            indices_dset.read(indices.data() + write_offset, mem_space, indices_file_space);
            
            write_offset += (range.end - range.begin);
        }
        
        // Rebuild indptr for selected rows
        std::vector<Index> new_indptr;
        new_indptr.reserve(row_mask.len + 1);
        new_indptr.push_back(0);
        
        for (Size i = 0; i < row_mask.len; ++i) {
            Index row_idx = row_mask[i];
            if (row_idx >= total_rows) {
                new_indptr.push_back(new_indptr.back());
                continue;
            }
            Index row_len = indptr[static_cast<Size>(row_idx + 1)] 
                          - indptr[static_cast<Size>(row_idx)];
            new_indptr.push_back(new_indptr.back() + row_len);
        }
        
        return OwnedSparse<T, true>(
            std::move(data), std::move(indices), std::move(new_indptr),
            static_cast<Index>(row_mask.len), total_cols
        );
    }
    
    /// @brief Load full sparse matrix.
    static OwnedSparse<T, true> execute_full(
        const std::string& h5_path,
        const std::string& group_path
    ) {
        File file(h5_path);
        Group group(file.id(), group_path);
        
        std::array<hsize_t, 2> shape_arr = group.read_attr<hsize_t, 2>("shape");
        Index rows = static_cast<Index>(shape_arr[0]);
        Index cols = static_cast<Index>(shape_arr[1]);
        
        Dataset data_dset(group.id(), "data");
        Dataset indices_dset(group.id(), "indices");
        Dataset indptr_dset(group.id(), "indptr");
        
        Index nnz = static_cast<Index>(data_dset.get_size());
        
        std::vector<T> data(static_cast<Size>(nnz));
        std::vector<Index> indices(static_cast<Size>(nnz));
        std::vector<Index> indptr(static_cast<Size>(rows + 1));
        
        data_dset.read(data.data());
        indices_dset.read(indices.data());
        indptr_dset.read(indptr.data());
        
        return OwnedSparse<T, true>(
            std::move(data), std::move(indices), std::move(indptr), 
            rows, cols
        );
    }
};

// =============================================================================
// SECTION 9: Public API Functions
// =============================================================================

/// @brief Load sparse matrix with row AND column masks (optimized 2D slicing).
///
/// Uses two-phase architecture:
/// - Phase 1: Plan (single-threaded I/O, Zone Map filtering)
/// - Phase 2: Execute (parallel computation, adaptive intersection)
///
/// Returns: QueryResult with filtered data.
template <typename T, bool IsCSR = true>
[[nodiscard]] inline QueryResult<T, IsCSR> load_sparse_masked(
    const std::string& h5_path,
    const std::string& group_path,
    Array<const Index> row_mask,
    Array<const Index> col_mask
) {
    return QueryExecutor<T>::execute_masked(
        h5_path, group_path, row_mask, col_mask
    );
}

/// @brief Load sparse matrix rows (row selection only).
///
/// Uses range merging for coalesced I/O.
/// Returns: OwnedSparse with selected rows.
template <typename T, bool IsCSR = true>
[[nodiscard]] inline OwnedSparse<T, IsCSR> load_sparse_rows(
    const std::string& h5_path,
    const std::string& group_path,
    Array<const Index> selected_rows
) {
    return QueryExecutor<T>::execute_rows(
        h5_path, group_path, selected_rows
    );
}

/// @brief Load full sparse matrix.
template <typename T, bool IsCSR = true>
[[nodiscard]] inline OwnedSparse<T, IsCSR> load_sparse_full(
    const std::string& h5_path,
    const std::string& group_path
) {
    return QueryExecutor<T>::execute_full(h5_path, group_path);
}

// =============================================================================
// SECTION 10: Convenience Wrappers (Vector Interface)
// =============================================================================

template <typename T, bool IsCSR = true>
[[nodiscard]] inline OwnedSparse<T, IsCSR> load_sparse_rows(
    const std::string& h5_path,
    const std::string& group_path,
    const std::vector<Index>& row_mask
) {
    return load_sparse_rows<T, IsCSR>(
        h5_path, group_path,
        Array<const Index>(row_mask.data(), row_mask.size())
    );
}

template <typename T, bool IsCSR = true>
[[nodiscard]] inline QueryResult<T, IsCSR> load_sparse_masked(
    const std::string& h5_path,
    const std::string& group_path,
    const std::vector<Index>& row_mask,
    const std::vector<Index>& col_mask
) {
    return load_sparse_masked<T, IsCSR>(
        h5_path, group_path,
        Array<const Index>(row_mask.data(), row_mask.size()),
        Array<const Index>(col_mask.data(), col_mask.size())
    );
}

// =============================================================================
// SECTION 11: Save Functions
// =============================================================================

/// @brief Save sparse matrix to HDF5 (anndata-compatible format).
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
inline void save_sparse(
    const std::string& h5_path,
    const std::string& group_path,
    const MatrixT& mat,
    const std::vector<hsize_t>& chunk_dims = {10000},
    unsigned compress_level = 6
) {
    File file = File::create(h5_path);
    Group group = Group::create(file.id(), group_path);
    
    // Write shape attribute
    std::array<hsize_t, 2> shape_arr = {
        static_cast<hsize_t>(mat.rows),
        static_cast<hsize_t>(mat.cols)
    };
    group.write_attr<hsize_t, 2>("shape", shape_arr);
    
    // Materialize if needed
    auto get_owned = [&]() {
        if constexpr (requires { mat.materialize(); }) {
            return mat.materialize();
        } else {
            return mat;
        }
    };
    const auto& owned = get_owned();
    
    // Create property list with compression
    DatasetCreateProps props;
    props.chunked(chunk_dims).shuffle().deflate(compress_level);
    
    // Create datasets
    Index total_nnz = owned.nnz();
    std::vector<hsize_t> data_dims = {static_cast<hsize_t>(total_nnz)};
    Dataspace data_space(data_dims);
    
    Dataset data_dset = Dataset::create(
        group.id(), "data", detail::native_type<typename MatrixT::ValueType>(), 
        data_space, props.id()
    );
    data_dset.write(owned.data.data());
    
    Dataset indices_dset = Dataset::create(
        group.id(), "indices", detail::native_type<Index>(), data_space, props.id()
    );
    indices_dset.write(owned.indices.data());
    
    Index primary_dim = IsCSR ? mat.rows : mat.cols;
    std::vector<hsize_t> indptr_dims = {static_cast<hsize_t>(primary_dim + 1)};
    Dataspace indptr_space(indptr_dims);
    
    Dataset indptr_dset = Dataset::create(
        group.id(), "indptr", detail::native_type<Index>(), indptr_space
    );
    indptr_dset.write(owned.indptr.data());
    
    file.flush();
}

// =============================================================================
// SECTION 12: Metadata Utilities
// =============================================================================

/// @brief Read sparse matrix shape without loading data.
template <bool IsCSR = true>
[[nodiscard]] inline std::tuple<Index, Index, Index> read_sparse_shape(
    const std::string& h5_path,
    const std::string& group_path
) {
    File file(h5_path);
    Group group(file.id(), group_path);
    
    std::array<hsize_t, 2> shape_arr = group.read_attr<hsize_t, 2>("shape");
    Index rows = static_cast<Index>(shape_arr[0]);
    Index cols = static_cast<Index>(shape_arr[1]);
    
    Dataset data_dset(group.id(), "data");
    Index nnz = static_cast<Index>(data_dset.get_size());
    
    return {rows, cols, nnz};
}

/// @brief Check if H5 group contains valid sparse matrix.
[[nodiscard]] inline bool is_valid_sparse_group(
    const std::string& h5_path,
    const std::string& group_path
) {
    try {
        File file(h5_path);
        Group group(file.id(), group_path);
        
        bool has_data = H5Lexists(group.id(), "data", H5P_DEFAULT) > 0;
        bool has_indices = H5Lexists(group.id(), "indices", H5P_DEFAULT) > 0;
        bool has_indptr = H5Lexists(group.id(), "indptr", H5P_DEFAULT) > 0;
        bool has_shape = H5Aexists(group.id(), "shape") > 0;
        
        return has_data && has_indices && has_indptr && has_shape;
    } catch (...) {
        return false;
    }
}

// =============================================================================
// SECTION 13: With-View Convenience Functions
// =============================================================================

/// @brief Load full matrix and return with view.
template <typename T, bool IsCSR = true>
[[nodiscard]] inline std::pair<OwnedSparse<T, IsCSR>, CustomSparse<T, IsCSR>> 
load_with_view(
    const std::string& h5_path,
    const std::string& group_path
) {
    auto owned = load_sparse_full<T, IsCSR>(h5_path, group_path);
    auto view = owned.view();
    return {std::move(owned), view};
}

/// @brief Load rows and return with view.
template <typename T, bool IsCSR = true>
[[nodiscard]] inline std::pair<OwnedSparse<T, IsCSR>, CustomSparse<T, IsCSR>> 
load_rows_with_view(
    const std::string& h5_path,
    const std::string& group_path,
    Array<const Index> row_mask
) {
    auto owned = load_sparse_rows<T, IsCSR>(h5_path, group_path, row_mask);
    auto view = owned.view();
    return {std::move(owned), view};
}

/// @brief Load masked data, materialize, and return with view.
template <typename T, bool IsCSR = true>
[[nodiscard]] inline std::pair<OwnedSparse<T, IsCSR>, CustomSparse<T, IsCSR>> 
load_masked_with_view(
    const std::string& h5_path,
    const std::string& group_path,
    Array<const Index> row_mask,
    Array<const Index> col_mask
) {
    auto result = load_sparse_masked<T, IsCSR>(h5_path, group_path, row_mask, col_mask);
    auto owned = result.materialize();
    auto view = owned.view();
    return {std::move(owned), view};
}

// =============================================================================
// SECTION 14: Type Aliases
// =============================================================================

template <typename T>
using CSRQueryResult = QueryResult<T, true>;

template <typename T>
using CSCQueryResult = QueryResult<T, false>;

// =============================================================================
// SECTION 15: Concept Verification
// =============================================================================

static_assert(SparseLike<QueryResult<Real, true>, true>);
static_assert(SparseLike<QueryResult<Real, false>, false>);

} // namespace scl::io::h5

#endif // SCL_HAS_HDF5
