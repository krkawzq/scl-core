#pragma once

// =============================================================================
/// @file core.hpp
/// @brief Unified Kernel Interface - All Kernel Operations
///
/// This header provides a single include point for all kernel operations.
/// All kernel functions are organized in the scl::kernel namespace.
///
/// Usage:
///   #include "scl/kernel/core.hpp"
///
/// All kernel operations are now available through their respective
/// sub-namespaces (e.g., scl::kernel::algebra, scl::kernel::gram, etc.)
// =============================================================================

// Kernel operations (alphabetical order)
#include "scl/kernel/algebra.hpp"
#include "scl/kernel/algebra_fast_impl.hpp"
#include "scl/kernel/algebra_mapped_impl.hpp"
#include "scl/kernel/bbknn.hpp"
#include "scl/kernel/bbknn_fast_impl.hpp"
#include "scl/kernel/bbknn_mapped_impl.hpp"
#include "scl/kernel/correlation.hpp"
#include "scl/kernel/correlation_fast_impl.hpp"
#include "scl/kernel/correlation_mapped_impl.hpp"
#include "scl/kernel/feature.hpp"
#include "scl/kernel/feature_fast_impl.hpp"
#include "scl/kernel/feature_mapped_impl.hpp"
#include "scl/kernel/gram.hpp"
#include "scl/kernel/gram_fast_impl.hpp"
#include "scl/kernel/gram_mapped_impl.hpp"
#include "scl/kernel/group.hpp"
#include "scl/kernel/group_fast_impl.hpp"
#include "scl/kernel/group_mapped_impl.hpp"
#include "scl/kernel/hvg.hpp"
#include "scl/kernel/hvg_fast_impl.hpp"
#include "scl/kernel/hvg_mapped_impl.hpp"
#include "scl/kernel/log1p.hpp"
#include "scl/kernel/log1p_fast_impl.hpp"
#include "scl/kernel/log1p_mapped_impl.hpp"
#include "scl/kernel/merge.hpp"
#include "scl/kernel/merge_fast_impl.hpp"
#include "scl/kernel/merge_mapped_impl.hpp"
#include "scl/kernel/mmd.hpp"
#include "scl/kernel/mmd_fast_impl.hpp"
#include "scl/kernel/mmd_mapped_impl.hpp"
#include "scl/kernel/mwu.hpp"
#include "scl/kernel/mwu_fast_impl.hpp"
#include "scl/kernel/mwu_mapped_impl.hpp"
#include "scl/kernel/neighbors.hpp"
#include "scl/kernel/neighbors_fast_impl.hpp"
#include "scl/kernel/neighbors_mapped_impl.hpp"
#include "scl/kernel/normalize.hpp"
#include "scl/kernel/normalize_fast_impl.hpp"
#include "scl/kernel/normalize_mapped_impl.hpp"
#include "scl/kernel/qc.hpp"
#include "scl/kernel/qc_fast_impl.hpp"
#include "scl/kernel/qc_mapped_impl.hpp"
#include "scl/kernel/reorder.hpp"
#include "scl/kernel/reorder_fast_impl.hpp"
#include "scl/kernel/reorder_mapped_impl.hpp"
#include "scl/kernel/resample.hpp"
#include "scl/kernel/resample_fast_impl.hpp"
#include "scl/kernel/resample_mapped_impl.hpp"
#include "scl/kernel/scale.hpp"
#include "scl/kernel/scale_fast_impl.hpp"
#include "scl/kernel/scale_mapped_impl.hpp"
#include "scl/kernel/slice.hpp"
#include "scl/kernel/slice_fast_impl.hpp"
#include "scl/kernel/slice_mapped_impl.hpp"
#include "scl/kernel/softmax.hpp"
#include "scl/kernel/softmax_fast_impl.hpp"
#include "scl/kernel/softmax_mapped_impl.hpp"
#include "scl/kernel/sparse.hpp"
#include "scl/kernel/sparse_fast_impl.hpp"
#include "scl/kernel/sparse_mapped_impl.hpp"
#include "scl/kernel/spatial.hpp"
#include "scl/kernel/spatial_fast_impl.hpp"
#include "scl/kernel/spatial_mapped_impl.hpp"
#include "scl/kernel/ttest.hpp"
#include "scl/kernel/ttest_fast_impl.hpp"
#include "scl/kernel/ttest_mapped_impl.hpp"
