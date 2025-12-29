// =============================================================================
// SCL Core - Test Framework Demonstration
// =============================================================================
//
// This file demonstrates and tests all test framework features.
//
// =============================================================================

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// Basic Assertions
// =============================================================================

SCL_TEST_UNIT(basic_assertions) {
    SCL_ASSERT(true);
    SCL_ASSERT_TRUE(1 == 1);
    SCL_ASSERT_FALSE(1 == 2);
}

SCL_TEST_UNIT(equality_assertions) {
    SCL_ASSERT_EQ(42, 42);
    SCL_ASSERT_NE(42, 43);
    
    SCL_ASSERT_EQ(3.14, 3.14);
    SCL_ASSERT_NE(3.14, 2.71);
}

SCL_TEST_UNIT(comparison_assertions) {
    SCL_ASSERT_LT(1, 2);
    SCL_ASSERT_LE(1, 1);
    SCL_ASSERT_LE(1, 2);
    
    SCL_ASSERT_GT(2, 1);
    SCL_ASSERT_GE(2, 2);
    SCL_ASSERT_GE(2, 1);
}

SCL_TEST_UNIT(floating_point_near) {
    SCL_ASSERT_NEAR(3.14159, 3.14, 0.01);
    SCL_ASSERT_NEAR(1.0, 1.0001, 0.001);
}

SCL_TEST_UNIT(null_pointer_checks) {
    int* ptr = nullptr;
    SCL_ASSERT_NULL(ptr);
    
    int value = 42;
    int* non_null = &value;
    SCL_ASSERT_NOT_NULL(non_null);
}

SCL_TEST_UNIT(string_assertions) {
    SCL_ASSERT_STR_EQ("hello", "hello");
    SCL_ASSERT_STR_CONTAINS("hello world", "world");
}

// =============================================================================
// Test Status Examples
// =============================================================================

SCL_TEST_UNIT(passing_test) {
    SCL_ASSERT_EQ(1 + 1, 2);
}

SCL_TEST_SKIP(intentionally_skipped, "This test is skipped for demo purposes")
{
    // This code won't run
    SCL_ASSERT(false);
}

// =============================================================================
// Test Suites
// =============================================================================

SCL_TEST_SUITE(math_operations)

SCL_TEST_CASE(addition) {
    SCL_ASSERT_EQ(2 + 2, 4);
    SCL_ASSERT_EQ(-1 + 1, 0);
}

SCL_TEST_CASE(subtraction) {
    SCL_ASSERT_EQ(5 - 3, 2);
    SCL_ASSERT_EQ(0 - 1, -1);
}

SCL_TEST_CASE(multiplication) {
    SCL_ASSERT_EQ(3 * 4, 12);
    SCL_ASSERT_EQ(5 * 0, 0);
}

SCL_TEST_SUITE_END

// =============================================================================
// Tagged Tests
// =============================================================================

SCL_TEST_TAGGED(fast_test, "unit", "fast")
{
    SCL_ASSERT_EQ(1, 1);
}

SCL_TEST_TAGGED(slow_test, "integration", "slow")
{
    // Simulate slow test
    int sum = 0;
    for (int i = 0; i < 100000; ++i) {
        sum += i;
    }
    (void)sum;
    SCL_ASSERT_EQ(1, 1);
}

// =============================================================================
// Exception Tests
// =============================================================================

SCL_TEST_UNIT(exception_handling) {
    SCL_ASSERT_THROWS(
        throw std::runtime_error("test error"),
        std::runtime_error
    );
    
    SCL_ASSERT_NO_THROW({
        int x = 1 + 1;
        (void)x;
    });
}

// =============================================================================
// Conditional Skip Example
// =============================================================================

SCL_TEST_UNIT(conditional_skip_example) {
    bool should_skip = false;
    SCL_SKIP_IF(should_skip, "Skipping because condition is false");
    
    SCL_ASSERT_EQ(1, 1);
}

// =============================================================================
// Test Fixture Example
// =============================================================================

SCL_TEST_FIXTURE(MathFixture) {
    int value = 0;
    
    MathFixture() : value(42) {
        // Setup
    }
    
    ~MathFixture() {
        // Teardown
    }
};

SCL_TEST_F(MathFixture, use_fixture) {
    SCL_ASSERT_EQ(fixture.value, 42);
    fixture.value = 100;
    SCL_ASSERT_EQ(fixture.value, 100);
}

SCL_TEST_F(MathFixture, another_fixture_test) {
    SCL_ASSERT_EQ(fixture.value, 42);
}

// =============================================================================
// Benchmark Example
// =============================================================================

SCL_TEST_UNIT(benchmark_example) {
    auto result = scl::test::benchmark([]() {
        int sum = 0;
        for (int i = 0; i < 100; ++i) {
            sum += i;
        }
        (void)sum;
    }, 1000, 100);
    
    scl::test::print_benchmark("simple_sum", result);
    
    SCL_ASSERT_GT(result.avg_ns, 0);
}

// =============================================================================
// Intentional Failure (to test error reporting)
// =============================================================================

// Uncomment to see failure reporting in action:
// SCL_TEST_UNIT(failing_test) {
//     SCL_ASSERT_EQ(1, 2);
// }

SCL_TEST_END

SCL_TEST_MAIN()
