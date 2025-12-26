// =============================================================================
/// @file memory.cpp
/// @brief Memory Management and Helper Functions
///
/// Provides memory allocation, deallocation, and utility functions.
// =============================================================================

#include "error.hpp"
#include "scl/core/lifetime.hpp"
#include "scl/core/sparse.hpp"

#include <cstring>

extern "C" {

// =============================================================================
// Memory Management
// =============================================================================

int scl_malloc(scl::Size bytes, void** out_ptr) {
    SCL_C_API_WRAPPER(
        if (!out_ptr) {
            throw scl::ValueError("scl_malloc: out_ptr is null");
        }
        auto handle = scl::core::mem::alloc(bytes);
        *out_ptr = handle.release();
    )
}

int scl_calloc(scl::Size bytes, void** out_ptr) {
    SCL_C_API_WRAPPER(
        if (!out_ptr) {
            throw scl::ValueError("scl_calloc: out_ptr is null");
        }
        auto handle = scl::core::mem::alloc_zero(bytes);
        *out_ptr = handle.release();
    )
}

int scl_malloc_aligned(scl::Size bytes, scl::Size alignment, void** out_ptr) {
    SCL_C_API_WRAPPER(
        if (!out_ptr) {
            throw scl::ValueError("scl_malloc_aligned: out_ptr is null");
        }
        auto handle = scl::core::mem::alloc_aligned(bytes, alignment);
        *out_ptr = handle.release();
    )
}

void scl_free(void* ptr) {
    if (ptr) std::free(ptr);
}

void scl_free_aligned(void* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

void scl_memzero(void* ptr, scl::Size bytes) {
    if (ptr && bytes > 0) {
        std::memset(ptr, 0, bytes);
    }
}

int scl_memcpy(const void* src, void* dst, scl::Size bytes) {
    SCL_C_API_WRAPPER(
        if (!src || !dst) {
            throw scl::ValueError("scl_memcpy: null pointer");
        }
        std::memcpy(dst, src, bytes);
    )
}

// =============================================================================
// Helper Functions
// =============================================================================

bool scl_is_valid_value(scl::Real value) {
    return std::isfinite(value);
}

scl::Size scl_sizeof_real() {
    return sizeof(scl::Real);
}

scl::Size scl_sizeof_index() {
    return sizeof(scl::Index);
}

scl::Size scl_alignment() {
    return 64;
}

// =============================================================================
// Workspace Size Calculation Helpers
// =============================================================================

scl::Size scl_ttest_workspace_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * n_groups * 2 * sizeof(scl::Real);
}

scl::Size scl_diff_expr_output_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * (n_groups - 1);
}

scl::Size scl_group_stats_output_size(scl::Size n_features, scl::Size n_groups) {
    return n_features * n_groups;
}

scl::Size scl_gram_output_size(scl::Size n) {
    return n * n;
}

scl::Size scl_correlation_workspace_size(scl::Size n) {
    return n * 2 * sizeof(scl::Real);  // means + inv_stds
}

} // extern "C"
