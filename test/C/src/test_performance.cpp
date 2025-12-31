/// @file test_performance.cpp
/// @brief Performance benchmarks for SCL C-API
///
/// Compares performance across different value types and operations

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

SCL_TEST_SUITE(performance_benchmarks)

SCL_TEST_TAGGED(benchmark_creation_real64, "benchmark", "slow") {
    auto result = benchmark([&]() {
        auto m = scl_sparse_identity(1000, SCL_REAL64, SCL_INDEX64);
        scl_sparse_destroy(m);
    }, 100, 10);
    
    print_benchmark("identity_1000_real64", result);
    SCL_ASSERT_LT(result.avg_ns, 50000000.0);  // < 50ms average
}

SCL_TEST_TAGGED(benchmark_creation_int32, "benchmark", "slow") {
    auto result = benchmark([&]() {
        auto m = scl_sparse_identity(1000, SCL_INT32, SCL_INDEX64);
        scl_sparse_destroy(m);
    }, 100, 10);
    
    print_benchmark("identity_1000_int32", result);
    SCL_ASSERT_LT(result.avg_ns, 50000000.0);
}

SCL_TEST_TAGGED(benchmark_creation_uint32, "benchmark", "slow") {
    auto result = benchmark([&]() {
        auto m = scl_sparse_identity(1000, SCL_UINT32, SCL_INDEX64);
        scl_sparse_destroy(m);
    }, 100, 10);
    
    print_benchmark("identity_1000_uint32", result);
    SCL_ASSERT_LT(result.avg_ns, 50000000.0);
}

SCL_TEST_TAGGED(benchmark_transpose, "benchmark", "slow") {
    auto m = scl_sparse_identity(500, SCL_REAL64, SCL_INDEX64);
    
    auto result = benchmark([&]() {
        auto t = scl_sparse_transpose(m);
        scl_sparse_destroy(t);
    }, 100, 10);
    
    print_benchmark("transpose_500", result);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(benchmark_clone, "benchmark", "slow") {
    auto m = scl_sparse_identity(500, SCL_REAL64, SCL_INDEX64);
    
    auto result = benchmark([&]() {
        auto c = scl_sparse_clone(m);
        scl_sparse_destroy(c);
    }, 100, 10);
    
    print_benchmark("clone_500", result);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(benchmark_scale, "benchmark") {
    auto m = scl_sparse_identity(1000, SCL_REAL64, SCL_INDEX64);
    
    auto result = benchmark([&]() {
        scl_sparse_scale(m, 1.5);
    }, 1000, 100);
    
    print_benchmark("scale_1000", result);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(memory_usage_by_type, "memory") {
    // Compare memory usage of different types
    std::size_t sizes[] = {
        sizeof(float),    // Real32
        sizeof(double),   // Real64
        sizeof(std::int8_t),   // Int8
        sizeof(std::int32_t),  // Int32
        sizeof(std::uint32_t), // Uint32
    };
    
    // Just verify sizes are as expected
    SCL_ASSERT_EQ(sizes[0], 4);   // Real32
    SCL_ASSERT_EQ(sizes[1], 8);   // Real64
    SCL_ASSERT_EQ(sizes[2], 1);   // Int8
    SCL_ASSERT_EQ(sizes[3], 4);   // Int32
    SCL_ASSERT_EQ(sizes[4], 4);   // Uint32
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

