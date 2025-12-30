/// @file test_debug.cpp
/// @brief Test file for scl/core/debug.hpp enhancements
///
/// Compile with:
///   g++ -std=c++20 -I. test_debug.cpp -o test_debug
///   g++ -std=c++20 -DNDEBUG -I. test_debug.cpp -o test_debug_release

#include "scl/core/debug.hpp"
#include <cstdint>
#include <vector>

// =============================================================================
// Test 1: Type Name Extraction
// =============================================================================

void test_type_names() {
    SCL_DEBUG_PRINT("=== Testing Type Name Extraction ===");

    // Basic types
    constexpr auto int_name = scl::debug::type_name<int>();
    constexpr auto double_name = scl::debug::type_name<double>();
    constexpr auto ptr_name = scl::debug::type_name<void*>();

    SCL_DEBUG_PRINT("Type names:");
    SCL_DEBUG_PRINT("  int: %s", int_name.data());
    SCL_DEBUG_PRINT("  double: %s", double_name.data());
    SCL_DEBUG_PRINT("  void*: %s", ptr_name.data());

    // Complex types
    constexpr auto vec_name = scl::debug::type_name<std::vector<int>>();
    SCL_DEBUG_PRINT("  std::vector<int>: %s", vec_name.data());
}

// =============================================================================
// Test 2: Type Inspection
// =============================================================================

void test_type_inspection() {
    SCL_DEBUG_PRINT("\n=== Testing Type Inspection ===");

    // Check various type properties
    static_assert(scl::debug::is_arithmetic_type<int>(), "int should be arithmetic");
    static_assert(scl::debug::is_integral_type<int>(), "int should be integral");
    static_assert(!scl::debug::is_floating_point_type<int>(), "int should not be float");
    static_assert(scl::debug::is_floating_point_type<double>(), "double should be float");

    SCL_DEBUG_PRINT("Type checks passed (compile-time)");

    // POD checks
    struct POD {
        int x;
        double y;
    };

    struct NonPOD {
        NonPOD() {}
        virtual ~NonPOD() {}
        std::vector<int> data;
    };

    static_assert(scl::debug::is_pod_type<POD>(), "POD should be POD");
    static_assert(!scl::debug::is_pod_type<NonPOD>(), "NonPOD should not be POD");

    SCL_DEBUG_PRINT("POD checks passed");
}

// =============================================================================
// Test 3: Debug Print Macros
// =============================================================================

void test_debug_prints() {
    SCL_DEBUG_PRINT("\n=== Testing Debug Print Macros ===");

    int x = 42;
    double pi = 3.14159;
    const char* str = "Hello";
    void* ptr = &x;
    bool flag = true;

    SCL_DEBUG_VAR(x);
    SCL_DEBUG_VAR(pi);
    SCL_DEBUG_VAR(ptr);
    SCL_DEBUG_VAR(flag);

    SCL_DEBUG_PRINT_LOC("This print includes source location");
}

// =============================================================================
// Test 4: Debug Markers
// =============================================================================

void test_debug_markers() {
    SCL_DEBUG_PRINT("\n=== Testing Debug Markers ===");

    SCL_DEBUG_TODO("Implement optimized SIMD version");
    SCL_DEBUG_FIXME("Handle edge case for empty input");
    SCL_DEBUG_HACK("Workaround for compiler limitation");

    // Multiple markers in same scope (should not conflict)
    SCL_DEBUG_TODO("Task 1");
    SCL_DEBUG_TODO("Task 2");
    SCL_DEBUG_TODO("Task 3");

    SCL_DEBUG_PRINT("Multiple markers work without conflicts");
}

// =============================================================================
// Test 5: Performance Timing
// =============================================================================

void expensive_computation() {
    SCL_DEBUG_TIMER("expensive_computation");

    // Simulate work
    volatile int sum = 0;
    for (int i = 0; i < 1000000; ++i) {
        sum += i;
    }
}

void test_performance_timing() {
    SCL_DEBUG_PRINT("\n=== Testing Performance Timing ===");

    {
        SCL_DEBUG_TIMER("quick operation");
        // Quick work
        volatile int x = 42 + 58;
    }

    expensive_computation();
}

// =============================================================================
// Test 6: Memory Inspection
// =============================================================================

void test_memory_inspection() {
    SCL_DEBUG_PRINT("\n=== Testing Memory Inspection ===");

    // Test memory dump
    const char message[] = "Hello, World!";
    SCL_DEBUG_DUMP_MEMORY(message, sizeof(message));

    // Test array printing
    int values[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    SCL_DEBUG_PRINT_ARRAY(values, 10);

    double floats[5] = {1.1, 2.2, 3.3, 4.4, 5.5};
    SCL_DEBUG_PRINT_ARRAY(floats, 5);

    // Test with large array
    std::vector<int> large_array(100);
    for (int i = 0; i < 100; ++i) {
        large_array[i] = i * i;
    }
    SCL_DEBUG_PRINT_ARRAY(large_array.data(), large_array.size());
}

// =============================================================================
// Test 7: Debug Assertions
// =============================================================================

void test_debug_assertions() {
    SCL_DEBUG_PRINT("\n=== Testing Debug Assertions ===");

    int size = 10;
    SCL_DEBUG_ASSERT(size > 0);
    SCL_DEBUG_ASSERT_MSG(size <= 100, "Size exceeds maximum");

    SCL_DEBUG_PRINT("All assertions passed");

    // Uncomment to test failure:
    // SCL_DEBUG_ASSERT_MSG(size > 100, "This should fail");
}

// =============================================================================
// Test 8: Unreachable Code Marker
// =============================================================================

enum class TestEnum { A, B, C };

int handle_enum(TestEnum e) {
    switch (e) {
        case TestEnum::A: return 1;
        case TestEnum::B: return 2;
        case TestEnum::C: return 3;
    }
    SCL_DEBUG_UNREACHABLE("Invalid enum value");
    return -1;
}

void test_unreachable() {
    SCL_DEBUG_PRINT("\n=== Testing Unreachable Marker ===");

    int result = handle_enum(TestEnum::B);
    SCL_DEBUG_VAR(result);

    SCL_DEBUG_PRINT("Unreachable marker compiles correctly");
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    SCL_DEBUG_PRINT("╔════════════════════════════════════════════════════════════╗");
    SCL_DEBUG_PRINT("║  SCL Core Debug Utilities - Comprehensive Test Suite      ║");
    SCL_DEBUG_PRINT("╚════════════════════════════════════════════════════════════╝");

    test_type_names();
    test_type_inspection();
    test_debug_prints();
    test_debug_markers();
    test_performance_timing();
    test_memory_inspection();
    test_debug_assertions();
    test_unreachable();

    SCL_DEBUG_PRINT("\n╔════════════════════════════════════════════════════════════╗");
    SCL_DEBUG_PRINT("║  All tests completed successfully!                        ║");
    SCL_DEBUG_PRINT("╚════════════════════════════════════════════════════════════╝");

    return 0;
}
