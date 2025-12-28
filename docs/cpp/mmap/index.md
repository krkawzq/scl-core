# Memory-Mapped Arrays (Experimental)

::: warning EXPERIMENTAL FEATURE
The mmap module is an experimental feature under active development. APIs may change without notice. Use with caution in production environments.
:::

## Overview

The mmap module provides memory-mapped virtual arrays and sparse matrices with transparent paging, event-driven prefetch scheduling, and out-of-core processing capabilities. It enables efficient processing of datasets larger than available RAM through intelligent page caching and prefetching.

**Location**: `scl/mmap/`

## Design Purpose

### The Problem

Bioinformatics datasets (single-cell RNA-seq, spatial transcriptomics) often exceed available RAM:

- A 10 million cell dataset with 30,000 genes requires 1.2 TB as dense matrix
- Even sparse formats can exceed 100 GB
- Traditional approaches require expensive high-memory machines or complex out-of-core algorithms

### The Solution

The mmap module provides a **transparent paging layer** that:

1. **Makes disk data feel like RAM** - Access elements with `array[i]` syntax
2. **Predicts access patterns** - Event-driven prefetching loads pages before they are needed
3. **Manages memory automatically** - Clock algorithm evicts least-recently-used pages
4. **Enables parallel I/O** - Worker threads perform asynchronous page loading

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│   MmapArray<T>  /  MmapSparse<T, IsCSR>  /  MmapSparseSlice<T>  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     PrefetchScheduler                            │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │SchedulePolicy│  │ Worker Pool  │  │  CacheEntry Array     │  │
│  │(LookaheadPolicy)│(async I/O)   │  │  (page table)         │  │
│  └──────────────┘  └──────────────┘  └───────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     GlobalPagePool                               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Sharded Hash Table (N shards = CPU cores)                 │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐        ┌─────────┐   │ │
│  │  │ Shard 0 │ │ Shard 1 │ │ Shard 2 │  ...   │ Shard N │   │ │
│  │  └─────────┘ └─────────┘ └─────────┘        └─────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Free List (memory reclamation)                            │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Storage Backends                             │
│   LocalFileBackend  /  CompressedBackend  /  NetworkBackend     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Innovations

### 1. Event-Driven Prefetch Scheduling

Traditional prefetching predicts future accesses based on past patterns. The mmap module takes a different approach: **the computation tells the scheduler what it needs**.

```cpp
// The scheduler hooks into your computation loop
scheduler.on_computation_begin(total_rows);
for (size_t row = 0; row < total_rows; ++row) {
    scheduler.on_row_begin(row);  // "I'm about to process row N"
    // Process row...
    scheduler.on_row_end(row);    // "Done with row N"
}
scheduler.on_computation_end();
```

**Why this is clever**:
- The scheduler knows exactly where computation is in the data
- LookaheadPolicy prefetches pages N rows ahead
- Latency feedback adjusts lookahead depth automatically:
  - High latency → increase lookahead (prefetch more aggressively)
  - Low latency → decrease lookahead (save memory)

### 2. Sharded Hash Table with Lock Striping

The GlobalPagePool must handle concurrent access from multiple threads. A single lock would become a bottleneck. Instead:

```cpp
// N shards based on CPU cores (clamped to [8, 128])
struct PageShard {
    ConcurrentPageMap<PageKey, Page*> page_map;
    std::mutex mutex;
};

// Pages are distributed across shards by hash
PageShard& shard_for(const PageKey& key) {
    return shards_[hash(key) % num_shards_];
}
```

**Why this is clever**:
- Lock contention reduced by factor of N
- Each shard has independent hash table + mutex
- FNV-1a hashing provides good distribution
- Quadratic probing avoids clustering

### 3. Fast Address Translation via Bit Operations

Page size is constrained to powers of 2, enabling fast division and modulo:

```cpp
// Page size must be power of 2
constexpr size_t kPageSize = 1048576;  // 1 MB
constexpr size_t kPageShift = 20;       // log2(kPageSize)
constexpr size_t kPageMask = kPageSize - 1;

// Fast division: byte_offset / kPageSize
size_t page_idx = byte_offset >> kPageShift;  // ~10x faster than division

// Fast modulo: byte_offset % kPageSize
size_t page_offset = byte_offset & kPageMask;  // ~10x faster than modulo
```

**Why this is clever**:
- Address translation happens on every element access
- Bit operations are single-cycle on modern CPUs
- Compile-time constants enable further optimizations

### 4. Clock Algorithm for Page Eviction

The Clock algorithm provides near-LRU behavior with minimal overhead:

```cpp
size_t select_victim() {
    for (size_t iter = 0; iter < num_pages_ * 2; ++iter) {
        size_t idx = clock_hand_++ % num_pages_;
        auto& entry = entries_[idx];

        if (!entry.is_loaded()) continue;
        if (entry.is_pinned()) continue;

        // "Second chance" - clear access bit, skip if was set
        if (!entry.clear_access()) {
            return idx;  // Found victim
        }
    }
    return SIZE_MAX;  // All pages pinned or recently accessed
}
```

**Why this is clever**:
- O(1) amortized eviction cost
- Recently accessed pages get a "second chance"
- Pin count prevents evicting pages in use
- No complex priority queue or LRU list maintenance

### 5. PagedBuffer for Cross-Page Access

Sparse matrix rows can span multiple pages. PagedBuffer encapsulates this complexity:

```cpp
template <typename T>
class PagedBuffer {
    std::array<PageHandle, kMaxHandles> handles_;  // Hold pages
    std::array<PageInfo, kMaxHandles> pages_;      // Data pointers

    // Element access handles page boundaries transparently
    T operator[](size_t i) const {
        if (is_contiguous()) {
            return base_ptr_[i];  // Fast path: single page
        }
        // Slow path: find correct page
        for (size_t p = 0; p < num_handles_; ++p) {
            if (i < offset + pages_[p].count) {
                return pages_[p].data[i - offset];
            }
            offset += pages_[p].count;
        }
    }
};
```

**Why this is clever**:
- RAII handles (PageHandle) keep pages pinned automatically
- Fast path for common case (data fits in one page)
- Iterator interface for range-based for loops
- Maximum 8 pages limits memory for long rows

### 6. CRTP-Based Storage Backends

Zero-overhead polymorphism via Curiously Recurring Template Pattern:

```cpp
template <typename Derived>
class StorageBackend {
public:
    size_t load_page(size_t idx, std::byte* dest) {
        return derived().load_page_impl(idx, dest);
    }

private:
    Derived& derived() { return static_cast<Derived&>(*this); }
};

// No virtual function overhead
class LocalFileBackend : public StorageBackend<LocalFileBackend> {
    size_t load_page_impl(size_t idx, std::byte* dest);
};
```

**Why this is clever**:
- Virtual dispatch adds 10-20ns per call
- With CRTP, calls are inlined at compile time
- Different backends have different optimal I/O sizes
- BackendCapabilities describes what each backend supports

## Components Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| `MmapArray<T>` | `array.hpp` | Virtual array with transparent paging |
| `MmapSparse<T, IsCSR>` | `sparse.hpp` | Memory-mapped CSR/CSC sparse matrix |
| `PrefetchScheduler` | `scheduler.hpp` | Event-driven prefetch and cache management |
| `GlobalPagePool` | `page.hpp` | System-wide page allocation and deduplication |
| `MmapConfig` | `configuration.hpp` | Runtime configuration presets |
| `StorageBackend<D>` | `backend/backend.hpp` | CRTP base for storage backends |

## Configuration Presets

```cpp
// High prefetch depth for sequential scan
auto cfg = MmapConfig::sequential(32);

// Large pool, minimal prefetch for random access
auto cfg = MmapConfig::random_access(128);

// Small pool, huge pages for streaming
auto cfg = MmapConfig::streaming(16);

// Writeback enabled for modifications
auto cfg = MmapConfig::read_write(64);
```

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Cache Hit | ~10 ns | Atomic load + pointer arithmetic |
| Cache Miss | 100 us - 10 ms | Page load + possible eviction |
| Prefetch | 50-200 us | Depends on storage backend |
| Address Translation | < 1 ns | Bit operations, no division |

## Future Enhancements

These features are planned but not yet implemented:

- **Compression support** (zstd, lz4) via `CompressedBackend`
- **Network storage** (HTTP, S3) via `NetworkBackend`
- **NUMA-aware allocation** via `memory/numa.hpp`
- **GPU memory mapping** via `memory/gpu.hpp`
- **Tiered caching** (SSD → RAM → GPU) via `cache/tiered.hpp`

## Usage Notes

### Memory Overhead

- ~48 bytes per MmapArray instance
- ~256 bytes per CacheEntry (one per page in working set)
- `max_resident_pages * kPageSize` for cached data

### Thread Safety

All components are designed for concurrent access:
- GlobalPagePool: Fully thread-safe with sharded locking
- PrefetchScheduler: All public methods thread-safe
- MmapArray: Concurrent reads safe, writes need external sync

### Best Practices

1. **Match access pattern to config**: Use `MmapConfig::sequential()` for row-by-row processing
2. **Use computation hooks**: Tell the scheduler where you are in the computation
3. **Batch operations**: `read_range()` is faster than repeated `operator[]`
4. **Pin hot pages**: Use PageHandle to keep frequently accessed pages resident

## See Also

- [Architecture Overview](/cpp/architecture/) - Design principles
- [Memory Model](/cpp/architecture/memory-model) - Lifetime management
- [Threading](/cpp/threading/) - Parallel processing infrastructure
