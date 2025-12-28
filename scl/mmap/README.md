# Memory-Mapped (MMap) Module

## Overview

The mmap module provides memory-mapped virtual arrays and sparse matrices with transparent paging, event-driven prefetch scheduling, and out-of-core processing capabilities. It enables efficient processing of datasets larger than available RAM through intelligent page caching and prefetching.

## Design Philosophy

The module follows a zero-copy architecture where data is accessed directly from memory-mapped files without full loading into RAM. Pages are loaded on-demand with automatic prefetching based on computation events, providing array-like interface while managing memory efficiently.

## Core Components

### 1. MmapArray<T>

Virtual array view with transparent paging and event-driven prefetch.

**Location**: `scl/mmap/array.h`

**Features**:
- Transparent element access via operator[]
- Event-driven prefetch based on computation hooks
- Batch read/write operations with SIMD optimization
- Write-through with optional dirty tracking
- Pluggable prefetch policies

**Usage Pattern**:
```
MmapArray<Real> arr(num_elements, loader, writer, max_resident, num_workers);
arr.begin_computation(total_rows);
for (size_t row = 0; row < total_rows; ++row) {
    ScopedRow guard(arr, row);
    Real value = arr[element_idx];  // Transparent paging
}
arr.end_computation();
```

**Memory Overhead**: ~48 bytes per MmapArray + (resident_pages * kPageSize)

### 2. GlobalPagePool

System-wide singleton page pool with concurrent-safe deduplication.

**Location**: `scl/mmap/page.h`

**Features**:
- Deduplication: Same (file_id, page_offset) shares one Page
- Reference counting: Pages freed when refcount reaches 0
- Memory reclamation: Free pages reused via free-list
- Sharded hash table: Reduces lock contention by factor of N
- No capacity limit: Only limited by system RAM

**Architecture**:
- Sharded hash table (N shards = CPU cores, clamped [8, 128])
- Each shard has independent ConcurrentPageMap + mutex
- Open addressing with quadratic probing
- Tombstone deletion (preserves probe chains)
- Chunked allocation (64-page chunks)

**Thread Safety**: Fully thread-safe with atomic operations and double-check locking

### 3. PrefetchScheduler

Event-driven prefetch scheduler with pluggable policies and worker threads.

**Location**: `scl/mmap/scheduler.h`

**Features**:
- Automatic prefetch based on computation events
- Worker threads for asynchronous I/O
- Clock algorithm for eviction
- Pluggable scheduling policies (LookaheadPolicy, custom)
- Blocking modes: SpinWait, ConditionWait, Hybrid, Callback

**Architecture**:
- Cache entries: Array of CacheEntry (one per page index)
- Task queue: Priority queue for prefetch tasks
- Worker threads: Configurable number (default: hw_concurrency / 4)
- Eviction: Clock algorithm with access bits

**Integration with Computation**:
```
PrefetchScheduler scheduler(total_pages, max_resident, num_workers);
scheduler.on_computation_begin(total_rows);
for (size_t row = 0; row < total_rows; ++row) {
    scheduler.on_row_begin(row);
    // Process row...
    scheduler.on_row_end(row);
}
scheduler.on_computation_end();
```

### 4. PageStore

I/O backend abstraction for page data storage.

**Location**: `scl/mmap/page.h`

**Features**:
- Abstracts storage backend (memory-mapped files, in-memory buffers, network storage)
- Load/write callbacks for custom I/O
- Thread-safe callback invocation

**Usage**:
```
LoadCallback loader = [](size_t page_idx, std::byte* dest) {
    // Load page data from storage
};
WriteCallback writer = [](size_t page_idx, const std::byte* src) {
    // Write page data to storage
};
PageStore store(file_id, total_bytes, loader, writer);
```

### 5. MmapConfig

Runtime configuration for memory-mapped arrays.

**Location**: `scl/mmap/configuration.h`

**Features**:
- Memory limits: max_resident_pages
- Prefetch tuning: depth, threads, pattern hints
- Feature flags: writeback, huge pages, auto-tuning
- Predefined configurations: sequential, random_access, streaming, read_write, strided

**Configuration Presets**:
- Sequential: High prefetch depth (8 pages), moderate pool (32 pages)
- Random: Large pool (128 pages), minimal prefetch (depth=1)
- Streaming: Small pool (16 pages), moderate prefetch (depth=2), huge pages enabled
- ReadWrite: Writeback enabled, balanced pool (64 pages)

### 6. Address Translation

Fast address translation using bit operations (no division/modulo).

**Location**: `scl/mmap/configuration.h`

**Functions**:
- bytes_to_pages: Compute pages needed for byte count (ceiling division)
- byte_to_page_idx: Convert byte offset to page index (fast division)
- byte_to_page_offset: Convert byte offset to offset within page (fast modulo)
- element_to_page_idx<T>: Convert element index to page index
- element_to_page_offset<T>: Convert element index to page offset

**Performance**: All functions are constexpr and inline, approximately 10x faster than division/modulo

## Page Management

### Page Structure

Each Page represents a fixed-size block (typically 1MB) with:
- Unique identity (file_id, page_offset) for deduplication
- Atomic reference counting for safe shared ownership
- Dirty flag for write-back tracking
- Cache-line alignment (64 bytes) for concurrent access

**Memory Layout**:
```
[0 - kPageSize-1]:     data buffer
[kPageSize]:           file_id (8 bytes)
[kPageSize+8]:         page_offset (8 bytes)
[kPageSize+16]:        refcount (atomic, 4 bytes)
[kPageSize+20]:        dirty flag (atomic, 1 byte)
[kPageSize+21]:        next_free pointer (8 bytes)
```

### Page Lifecycle

1. **Creation**: Page allocated from GlobalPagePool
2. **Loading**: Page data loaded from PageStore via LoadCallback
3. **Caching**: Page cached in PrefetchScheduler cache entries
4. **Pinning**: PageHandle pins page during use (prevents eviction)
5. **Eviction**: Clock algorithm selects page for eviction when cache full
6. **Writeback**: Dirty pages written to storage via WriteCallback
7. **Release**: Page released to GlobalPagePool when refcount reaches 0
8. **Reclamation**: Zero-refcount pages added to free-list for reuse

## Thread Safety

All components are designed for concurrent access:

**MmapArray**:
- Read operations: Thread-safe (concurrent access allowed)
- Write operations: Thread-safe (atomic dirty tracking)
- Prefetch: Thread-safe (multi-threaded prefetch workers)

**GlobalPagePool**:
- get_or_create: Double-check locking (TOCTOU-safe)
- release: CAS loop (ABA-safe)
- Hash table: Tombstone deletion (probe chain safe)

**PrefetchScheduler**:
- All public methods thread-safe
- Atomic operations for counters and flags
- Mutexes for task queue and policy access
- Condition variables for worker thread coordination

## Performance Characteristics

**Cache Hit**: ~10ns (atomic load + pointer arithmetic)
**Cache Miss**: ~100us - 10ms (page load + possible eviction)
**Prefetch Latency**: Typically 50-200us per page
**Memory Efficiency**: 40-60% savings for long-running processes (free-list reuse)

**Optimizations**:
- SIMD-optimized batch read/write operations (4-way unrolled)
- Fast address translation (bit operations, no division)
- Prefetch hints reduce cache misses by 30-50%
- Sharded hash table reduces contention by factor of N

## Configuration Tuning

### Page Size Selection

- 64KB: Best for random access (many small reads)
- 256KB: Balanced for mixed workloads
- 1MB: Default, best for sequential scan (bioinformatics)
- 4MB+: Best for streaming large datasets

Override at compile time: `-DSCL_MMAP_PAGE_SIZE=262144`

### Prefetch Depth

- Sequential: depth = 8-32 (high prefetch)
- Random: depth = 1-2 (minimal prefetch)
- Strided: depth = 2-4 (moderate prefetch)
- Adaptive: Auto-adjusted based on fetch latency

### Worker Threads

Default: `hw_concurrency / 4`
- Sequential: `cores / 4`
- Random: `cores / 2`
- Adaptive: `cores / 3`

## Python Bindings

The module provides Python bindings for memory-mapped sparse matrix operations:

**Location**: `src/scl/_kernel/mmap.py`

**Key Functions**:
- mmap_create_csr_from_ptr: Create mmap CSR from existing pointers
- mmap_open_csr_file: Open mmap CSR from file
- mmap_csr_load_full: Load full matrix into memory
- mmap_csr_load_masked: Load masked subset
- mmap_csr_spmv: Sparse matrix-vector multiply
- mmap_csr_normalize_l1/l2: Normalization operations
- mmap_csr_to_csc/dense: Format conversion

**Usage**:
```python
from scl._kernel import mmap

handle = mmap.mmap_open_csr_file("matrix.bin", max_pages=64)
rows, cols, nnz = mmap.mmap_csr_shape(handle)
# ... use handle for operations ...
mmap.mmap_release(handle)
```

## Use Cases

1. **Large Datasets**: Process datasets larger than available RAM
2. **Out-of-Core Processing**: Efficiently handle sparse matrices on disk
3. **Lazy Loading**: Load data on-demand without full materialization
4. **Memory-Efficient Operations**: Shared pages across multiple arrays
5. **Streaming Algorithms**: One-pass algorithms over large datasets

## Integration with Sparse Matrices

The module integrates with sparse matrix operations through:

- Memory-mapped CSR/CSC formats
- Transparent paging for sparse matrix rows/columns
- Efficient loading of masked subsets
- Format conversion (CSR to CSC, CSR to dense)
- Sparse matrix operations (SpMV, normalization, statistics)

**Location**: `src/scl/sparse/_mapped.py` for Python interface

## Error Handling

The module uses defensive programming with:
- Bounds checking on all array access
- Integer overflow detection
- Graceful fallback (returns zero-initialized values)
- Error logging for debugging
- RAII guarantees (no resource leaks)

## Compile-Time Configuration

**Constants** (defined in `configuration.h`):
- kPageSize: Page size in bytes (default: 1MB, must be power of 2)
- kPageShift: log2(kPageSize) for fast division
- kPageMask: kPageSize - 1 for fast modulo
- kDefaultPoolSize: Default cache capacity (64 pages)
- kMaxPageSize: Maximum supported (16MB)
- kMinPageSize: Minimum supported (4KB)

**Validation**: Compile-time checks ensure page size is power of 2 and within bounds.

## Examples

See:
- `tests/python/test_dist.py`: TestMappedCSR class
- `src/scl/sparse/_mapped.py`: MappedCustomCSR implementation
- `scl/mmap/array.hpp`: MmapArray implementation details

## Future Enhancements

Potential improvements:
- Compression support (zstd, lz4)
- Network storage backends (HTTP, S3)
- Distributed page pools (multi-process)
- NUMA-aware page allocation
- GPU memory mapping integration

