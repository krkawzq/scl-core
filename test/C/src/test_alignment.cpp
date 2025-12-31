/// @file test_alignment.cpp
/// @brief Tests for memory alignment correctness
///
/// Verifies that SCL_ALLOCA_ALIGNED properly aligns stack memory

#include "test.hpp"
#include "scl/core/macro.hpp"
#include <cstdint>

SCL_TEST_BEGIN

SCL_TEST_SUITE(alignment_verification)

SCL_TEST_TAGGED(alloca_aligned_16_bytes, "alignment", "quick") {
    void* ptr = SCL_ALLOCA_ALIGNED(100, 16);
    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    SCL_ASSERT_EQ(addr % 16, 0);
}

SCL_TEST_TAGGED(alloca_aligned_32_bytes, "alignment", "quick") {
    void* ptr = SCL_ALLOCA_ALIGNED(100, 32);
    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    SCL_ASSERT_EQ(addr % 32, 0);
}

SCL_TEST_TAGGED(alloca_aligned_64_bytes, "alignment", "quick") {
    void* ptr = SCL_ALLOCA_ALIGNED(100, 64);
    auto addr = reinterpret_cast<std::uintptr_t>(ptr);
    SCL_ASSERT_EQ(addr % 64, 0);
}

SCL_TEST_TAGGED(alloca_aligned_various_sizes, "alignment") {
    // Test different sizes with 64-byte alignment
    for (std::size_t size : {1, 10, 100, 1000, 8192}) {
        void* ptr = SCL_ALLOCA_ALIGNED(size, 64);
        auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        SCL_ASSERT_EQ(addr % 64, 0);
        
        // Write to memory to ensure it's accessible
        std::memset(ptr, 0, size);
    }
}

SCL_TEST_TAGGED(sort_buffer_alignment, "alignment", "quick") {
    // Create matrices and verify from_coo works (indirectly tests alignment)
    std::vector<std::int64_t> row_indices = {0, 1, 2, 3, 4, 5};
    std::vector<std::int64_t> col_indices = {0, 1, 2, 3, 4, 5};
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    auto handle = scl_sparse_from_coo(
        10, 10,
        row_indices.data(),
        col_indices.data(),
        values.data(),
        6,
        SCL_REAL64,
        SCL_INDEX64,
        SCL_LAYOUT_CSR
    );
    
    SCL_ASSERT_SPARSE_VALID(handle);
    SCL_ASSERT_EQ(scl_sparse_nnz(handle), 6);
    
    scl_sparse_destroy(handle);
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

