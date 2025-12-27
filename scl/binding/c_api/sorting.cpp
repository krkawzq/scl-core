// =============================================================================
/// @file sorting.cpp
/// @brief C ABI Wrapper for SCL Sorting Algorithms
///
/// Provides C-compatible exports for:
/// - VQSort: Ultra-fast SIMD sorting (10-20x faster than std::sort)
/// - Argsort: Indirect sorting (returns sorted indices)
/// - Pair sorting: Sort keys and reorder values accordingly
///
/// All functions follow the standard error protocol:
/// - Return 0 on success
/// - Return -1 on failure (error message in thread-local buffer)
// =============================================================================

#include "scl/binding/c_api/scl_c_api.h"
#include "scl/binding/c_api/error.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/argsort.hpp"
#include "scl/core/type.hpp"

#include <algorithm>

extern "C" {

// =============================================================================
// SECTION 1: VQSort - Ultra-Fast SIMD Sorting
// =============================================================================

/// @brief Sort real array in ascending order (SIMD optimized)
///
/// Uses Google Highway VQSort for maximum performance.
/// Typically 10-20x faster than std::sort.
///
/// @param data Array to sort [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_vqsort_real_ascending(scl_real_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Real> arr(data, static_cast<scl::Size>(size));
        scl::sort::sort(arr);
    );
}

/// @brief Sort real array in descending order (SIMD optimized)
///
/// @param data Array to sort [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_vqsort_real_descending(scl_real_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Real> arr(data, static_cast<scl::Size>(size));
        scl::sort::sort_descending(arr);
    );
}

/// @brief Sort index array in ascending order (SIMD optimized)
///
/// @param data Array to sort [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_vqsort_index_ascending(scl_index_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Index> arr(data, static_cast<scl::Size>(size));
        scl::sort::sort(arr);
    );
}

/// @brief Sort index array in descending order (SIMD optimized)
///
/// @param data Array to sort [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_vqsort_index_descending(scl_index_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Index> arr(data, static_cast<scl::Size>(size));
        scl::sort::sort_descending(arr);
    );
}

/// @brief Sort int32 array in ascending order (SIMD optimized)
///
/// @param data Array to sort [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_vqsort_int32_ascending(int32_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<int32_t> arr(data, size);
        scl::sort::sort(arr);
    );
}

/// @brief Sort int32 array in descending order (SIMD optimized)
///
/// @param data Array to sort [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_vqsort_int32_descending(int32_t* data, scl_size_t size) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<int32_t> arr(data, size);
        scl::sort::sort_descending(arr);
    );
}

// =============================================================================
// SECTION 2: Argsort - Indirect Sorting
// =============================================================================

/// @brief Argsort real array in ascending order
///
/// Returns indices that would sort the array.
/// Does NOT modify the input array.
///
/// @param keys Input array (not modified)
/// @param size Number of elements
/// @param indices Output sorted indices [pre-allocated, size = size]
/// @return 0 on success, -1 on failure
int scl_argsort_real_ascending(
    const scl_real_t* keys,
    scl_size_t size,
    scl_index_t* indices
) {
    SCL_C_API_WRAPPER(
        if (!keys || !indices) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<const scl::Real> keys_arr(keys, static_cast<scl::Size>(size));
        scl::Array<scl::Index> indices_arr(indices, static_cast<scl::Size>(size));

        scl::sort::argsort_indirect(keys_arr, indices_arr);
    );
}

/// @brief Argsort real array in descending order
///
/// @param keys Input array (not modified)
/// @param size Number of elements
/// @param indices Output sorted indices [pre-allocated]
/// @return 0 on success, -1 on failure
int scl_argsort_real_descending(
    const scl_real_t* keys,
    scl_size_t size,
    scl_index_t* indices
) {
    SCL_C_API_WRAPPER(
        if (!keys || !indices) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<const scl::Real> keys_arr(keys, static_cast<scl::Size>(size));
        scl::Array<scl::Index> indices_arr(indices, static_cast<scl::Size>(size));

        // Initialize indices
        scl::sort::detail::iota_serial(indices_arr);

        // Sort indices by keys (descending)
        std::sort(indices_arr.ptr, indices_arr.ptr + indices_arr.len,
            [&](scl::Index a, scl::Index b) {
                return keys_arr[a] > keys_arr[b];
            }
        );
    );
}

/// @brief Argsort with buffer (modifies keys in buffer, not original)
///
/// More efficient than indirect argsort for large arrays.
///
/// @param keys Input array (not modified)
/// @param size Number of elements
/// @param indices Output sorted indices [pre-allocated]
/// @param buffer Scratch buffer [pre-allocated, size = size * sizeof(real)]
/// @return 0 on success, -1 on failure
int scl_argsort_real_buffered(
    const scl_real_t* keys,
    scl_size_t size,
    scl_index_t* indices,
    scl_byte_t* buffer
) {
    SCL_C_API_WRAPPER(
        if (!keys || !indices || !buffer) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<const scl::Real> keys_arr(keys, static_cast<scl::Size>(size));
        scl::Array<scl::Index> indices_arr(indices, static_cast<scl::Size>(size));
        scl::Array<scl::Byte> buffer_arr(buffer, size * sizeof(scl::Real));

        scl::sort::argsort_buffered(keys_arr, indices_arr, buffer_arr);
    );
}

/// @brief Argsort with buffer (descending)
///
/// @param keys Input array (not modified)
/// @param size Number of elements
/// @param indices Output sorted indices [pre-allocated]
/// @param buffer Scratch buffer [pre-allocated]
/// @return 0 on success, -1 on failure
int scl_argsort_real_buffered_descending(
    const scl_real_t* keys,
    scl_size_t size,
    scl_index_t* indices,
    scl_byte_t* buffer
) {
    SCL_C_API_WRAPPER(
        if (!keys || !indices || !buffer) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<const scl::Real> keys_arr(keys, static_cast<scl::Size>(size));
        scl::Array<scl::Index> indices_arr(indices, static_cast<scl::Size>(size));
        scl::Array<scl::Byte> buffer_arr(buffer, size * sizeof(scl::Real));

        scl::sort::argsort_buffered_descending(keys_arr, indices_arr, buffer_arr);
    );
}

// =============================================================================
// SECTION 3: Pair Sorting - Sort Keys and Reorder Values
// =============================================================================

/// @brief Sort real keys and reorder real values accordingly (ascending)
///
/// Both keys and values are modified in-place.
///
/// @param keys Key array [modified in-place]
/// @param values Value array [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_sort_pairs_real_real_ascending(
    scl_real_t* keys,
    scl_real_t* values,
    scl_size_t size
) {
    SCL_C_API_WRAPPER(
        if (!keys || !values) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Real> keys_arr(keys, size);
        scl::Array<scl::Real> values_arr(values, size);

        scl::sort::sort_pairs(keys_arr, values_arr);
    );
}

/// @brief Sort real keys and reorder real values accordingly (descending)
///
/// @param keys Key array [modified in-place]
/// @param values Value array [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_sort_pairs_real_real_descending(
    scl_real_t* keys,
    scl_real_t* values,
    scl_size_t size
) {
    SCL_C_API_WRAPPER(
        if (!keys || !values) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Real> keys_arr(keys, size);
        scl::Array<scl::Real> values_arr(values, size);

        scl::sort::sort_pairs_descending(keys_arr, values_arr);
    );
}

/// @brief Sort real keys and reorder index values accordingly (ascending)
///
/// @param keys Key array [modified in-place]
/// @param values Value array [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_sort_pairs_real_index_ascending(
    scl_real_t* keys,
    scl_index_t* values,
    scl_size_t size
) {
    SCL_C_API_WRAPPER(
        if (!keys || !values) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Real> keys_arr(keys, size);
        scl::Array<scl::Index> values_arr(values, size);

        scl::sort::sort_pairs(keys_arr, values_arr);
    );
}

/// @brief Sort real keys and reorder index values accordingly (descending)
///
/// @param keys Key array [modified in-place]
/// @param values Value array [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_sort_pairs_real_index_descending(
    scl_real_t* keys,
    scl_index_t* values,
    scl_size_t size
) {
    SCL_C_API_WRAPPER(
        if (!keys || !values) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Real> keys_arr(keys, size);
        scl::Array<scl::Index> values_arr(values, size);

        scl::sort::sort_pairs_descending(keys_arr, values_arr);
    );
}

/// @brief Sort index keys and reorder real values accordingly (ascending)
///
/// @param keys Key array [modified in-place]
/// @param values Value array [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_sort_pairs_index_real_ascending(
    scl_index_t* keys,
    scl_real_t* values,
    scl_size_t size
) {
    SCL_C_API_WRAPPER(
        if (!keys || !values) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Index> keys_arr(keys, size);
        scl::Array<scl::Real> values_arr(values, size);

        scl::sort::sort_pairs(keys_arr, values_arr);
    );
}

/// @brief Sort index keys and reorder index values accordingly (ascending)
///
/// @param keys Key array [modified in-place]
/// @param values Value array [modified in-place]
/// @param size Number of elements
/// @return 0 on success, -1 on failure
int scl_sort_pairs_index_index_ascending(
    scl_index_t* keys,
    scl_index_t* values,
    scl_size_t size
) {
    SCL_C_API_WRAPPER(
        if (!keys || !values) {
            throw scl::ValueError("Null pointer argument");
        }

        scl::Array<scl::Index> keys_arr(keys, size);
        scl::Array<scl::Index> values_arr(values, size);

        scl::sort::sort_pairs(keys_arr, values_arr);
    );
}

// =============================================================================
// SECTION 4: Top-K Selection (Partial Sort)
// =============================================================================

/// @brief Find top K largest elements (partial sort)
///
/// Reorders array so that the K largest elements are in the first K positions.
/// The first K elements are sorted, the rest are unordered.
///
/// Uses std::partial_sort for O(N log K) complexity.
///
/// @param data Array to partially sort [modified in-place]
/// @param size Total number of elements
/// @param k Number of top elements to find
/// @return 0 on success, -1 on failure
int scl_topk_real(scl_real_t* data, scl_size_t size, scl_size_t k) {
    SCL_C_API_WRAPPER(
        if (!data) {
            throw scl::ValueError("Null pointer argument");
        }

        if (k > size) {
            throw scl::ValueError("k cannot exceed array size");
        }

        std::partial_sort(
            data, data + k, data + size,
            [](scl_real_t a, scl_real_t b) { return a > b; }
        );
    );
}

/// @brief Find top K largest elements with indices
///
/// Returns both the top K values and their original indices.
///
/// @param keys Input array (not modified)
/// @param size Total number of elements
/// @param k Number of top elements to find
/// @param out_values Output top K values [pre-allocated, size = k]
/// @param out_indices Output indices of top K values [pre-allocated, size = k]
/// @return 0 on success, -1 on failure
int scl_topk_real_with_indices(
    const scl_real_t* keys,
    scl_size_t size,
    scl_size_t k,
    scl_real_t* out_values,
    scl_index_t* out_indices
) {
    SCL_C_API_WRAPPER(
        if (!keys || !out_values || !out_indices) {
            throw scl::ValueError("Null pointer argument");
        }

        if (k > size) {
            throw scl::ValueError("k cannot exceed array size");
        }

        // Initialize indices
        scl::Array<scl::Index> indices_arr(out_indices, size);
        scl::sort::detail::iota_serial(indices_arr);

        // Partial sort indices by keys (descending)
        std::partial_sort(
            out_indices, out_indices + k, out_indices + size,
            [keys](scl_index_t a, scl_index_t b) {
                return keys[a] > keys[b];
            }
        );

        // Copy top K values
        for (scl_size_t i = 0; i < k; ++i) {
            out_values[i] = keys[out_indices[i]];
        }
    );
}

// =============================================================================
// SECTION 5: Utility Functions
// =============================================================================

/// @brief Check if array is sorted in ascending order
///
/// @param data Array to check
/// @param size Number of elements
/// @param out_is_sorted Output boolean (1 = sorted, 0 = not sorted)
/// @return 0 on success, -1 on failure
int scl_is_sorted_real_ascending(
    const scl_real_t* data,
    scl_size_t size,
    int* out_is_sorted
) {
    SCL_C_API_WRAPPER(
        if (!data || !out_is_sorted) {
            throw scl::ValueError("Null pointer argument");
        }

        bool sorted = std::is_sorted(data, data + size);
        *out_is_sorted = sorted ? 1 : 0;
    );
}

/// @brief Check if array is sorted in descending order
///
/// @param data Array to check
/// @param size Number of elements
/// @param out_is_sorted Output boolean (1 = sorted, 0 = not sorted)
/// @return 0 on success, -1 on failure
int scl_is_sorted_real_descending(
    const scl_real_t* data,
    scl_size_t size,
    int* out_is_sorted
) {
    SCL_C_API_WRAPPER(
        if (!data || !out_is_sorted) {
            throw scl::ValueError("Null pointer argument");
        }

        bool sorted = std::is_sorted(data, data + size, std::greater<scl_real_t>());
        *out_is_sorted = sorted ? 1 : 0;
    );
}

/// @brief Get buffer size needed for argsort_buffered
///
/// @param size Number of elements
/// @return Buffer size in bytes
scl_size_t scl_argsort_buffer_size(scl_size_t size) {
    return size * sizeof(scl_real_t);
}

} // extern "C"

