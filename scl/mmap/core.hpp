#pragma once

// =============================================================================
/// @file core.hpp
/// @brief SCL MMAP Module - Main Header
///
/// Memory-mapped sparse matrix module for out-of-core computation on large
/// datasets that exceed available RAM.
///
/// Core Components:
///
/// - Page/PageTable/PageHandle: Page management primitives
/// - PagePool/Scheduler: Memory scheduling and caching
/// - VirtualArray: Virtual array abstraction with automatic paging
/// - MappedSparse/MappedDense: Memory-mapped matrix types
/// - MappedVirtualSparse: Zero-copy masked views
///
/// Performance Features:
///
/// - Clock algorithm for O(1) amortized eviction
/// - Bit-shift address translation (no division)
/// - Lock-free fast paths for cache hits
/// - Parallel operations via scl::threading::parallel_for
/// - SIMD-friendly loop structures
///
/// Usage Example:
///
///   #include "scl/mmap/core.hpp"
///
///   // Open mapped sparse matrix from file
///   auto mat = scl::mmap::open_sparse_file<float, true>("data.sclm");
///
///   // Create zero-copy masked view
///   uint8_t row_mask[1000] = {...};
///   auto view = MappedVirtualCSR<float>(mat, row_mask, nullptr);
///
///   // Load to contiguous memory
///   std::vector<float> data(view.nnz());
///   std::vector<int32_t> indices(view.nnz());
///   std::vector<int32_t> indptr(view.rows() + 1);
///   load_full(view, data.data(), indices.data(), indptr.data());
///
/// Configuration:
///
/// - SCL_MMAP_PAGE_SIZE: Page size in bytes (default 1MB)
/// - SCL_MMAP_DEFAULT_POOL_SIZE: Default page pool size (default 64)
// =============================================================================

#include "scl/mmap/configuration.hpp"
#include "scl/mmap/table.hpp"
#include "scl/mmap/scheduler.hpp"
#include "scl/mmap/array.hpp"
#include "scl/mmap/matrix.hpp"
#include "scl/mmap/convert.hpp"
#include "scl/mmap/math.hpp"
