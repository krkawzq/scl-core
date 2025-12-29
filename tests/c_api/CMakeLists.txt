# =============================================================================
# SCL Core - C API Tests (Standalone Build)
# =============================================================================
#
# Tests the C API by dynamically linking to PRE-COMPILED libraries.
# Does NOT trigger recompilation of main project (lazy/incremental).
#
# Prerequisites:
#   1. Run 'make compile-cpp' to build libraries first
#   2. Libraries must exist in python/scl/libs/
#
# =============================================================================

cmake_minimum_required(VERSION 3.15)

# -----------------------------------------------------------------------------
# Standalone Configuration (No Recompilation of Main Project)
# -----------------------------------------------------------------------------

# This prevents CMake from trying to build the main project
# Tests only consume pre-built libraries from python/scl/libs/

message(STATUS "")
message(STATUS "========================================")
message(STATUS "SCL C API Tests (Standalone Build)")
message(STATUS "========================================")

# Test binary output directory
set(TEST_UNITS_DIR "${CMAKE_SOURCE_DIR}/tests/c_api/units")
file(MAKE_DIRECTORY ${TEST_UNITS_DIR})

message(STATUS "  Output directory: ${TEST_UNITS_DIR}")
message(STATUS "  Library path: ${CMAKE_SOURCE_DIR}/python/scl/libs")

# Check if libraries exist
set(LIB_PATH "${CMAKE_SOURCE_DIR}/python/scl/libs/libscl_f64_i64.so")
if(NOT EXISTS ${LIB_PATH})
    message(WARNING "")
    message(WARNING "⚠️  Library not found: ${LIB_PATH}")
    message(WARNING "   Please run: make compile-cpp")
    message(WARNING "")
endif()

# -----------------------------------------------------------------------------
# Helper Function: Add Test Executable
# -----------------------------------------------------------------------------

function(add_scl_test test_name source_file)
    set(target_name "test_${test_name}")
    
    add_executable(${target_name} ${source_file})
    
    # Include directories
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}                      # For scl/ includes
        ${CMAKE_CURRENT_SOURCE_DIR}/include      # For test utilities
    )
    
    # C++20 standard
    target_compile_features(${target_name} PRIVATE cxx_std_20)
    
    # Compiler flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name} PRIVATE
            -Wall -Wextra -Wpedantic
            -Wno-unused-parameter
            -Wno-unused-variable
            -Wno-sign-compare
        )
    endif()
    
    # Link to PRE-BUILT library (not a CMake target)
    target_link_directories(${target_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/python/scl/libs
    )
    
    target_link_libraries(${target_name} PRIVATE
        scl_f64_i64  # Links to libscl_f64_i64.so
        ${CMAKE_DL_LIBS}
    )
    
    # Link Eigen3 if available
    if(TARGET Eigen3::Eigen)
        target_link_libraries(${target_name} PRIVATE Eigen3::Eigen)
    endif()
    
    # Set output directory
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${TEST_UNITS_DIR}"
    )
    
    # Set RPATH to find libraries at runtime
    if(UNIX AND NOT APPLE)
        set_target_properties(${target_name} PROPERTIES
            INSTALL_RPATH "${CMAKE_SOURCE_DIR}/python/scl/libs"
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/python/scl/libs"
            BUILD_WITH_INSTALL_RPATH ON
        )
    elseif(APPLE)
        set_target_properties(${target_name} PROPERTIES
            INSTALL_RPATH "@loader_path/../../../../python/scl/libs"
            BUILD_RPATH "${CMAKE_SOURCE_DIR}/python/scl/libs"
            BUILD_WITH_INSTALL_RPATH ON
        )
    endif()
    
    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${target_name})
    
    # Set test environment
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_SOURCE_DIR}/python/scl/libs:$ENV{LD_LIBRARY_PATH}"
    )
    
    message(STATUS "  ✓ ${test_name}")
endfunction()

# -----------------------------------------------------------------------------
# Register Test Files
# -----------------------------------------------------------------------------

message(STATUS "")
message(STATUS "Registering test files:")

# Core tests (always built)
add_scl_test(demo         src/test_demo.cpp)
add_scl_test(guards       src/test_guards.cpp)
add_scl_test(core_error   src/test_core_error.cpp)
add_scl_test(sparse_basic src/test_sparse_basic.cpp)
add_scl_test(sparse_ops   src/test_sparse_ops.cpp)
add_scl_test(dense        src/test_dense.cpp)
add_scl_test(tools        src/test_tools.cpp)

# -----------------------------------------------------------------------------
# Custom Targets
# -----------------------------------------------------------------------------

# Run all tests
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS 
        test_demo test_guards test_core_error 
        test_sparse_basic test_sparse_ops test_dense test_tools
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all C API tests..."
)

# Run tests with verbose output
add_custom_target(run_all_tests_verbose
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose
    DEPENDS 
        test_demo test_guards test_core_error 
        test_sparse_basic test_sparse_ops test_dense test_tools
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running tests (verbose)..."
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

message(STATUS "")
message(STATUS "Test configuration complete:")
message(STATUS "  Binaries: ${TEST_UNITS_DIR}/test_*")
message(STATUS "  Libraries: ${CMAKE_SOURCE_DIR}/python/scl/libs/libscl_*.so")
message(STATUS "  Eigen3: ${Eigen3_FOUND}")
message(STATUS "")
message(STATUS "Build tests: ninja")
message(STATUS "Run tests:   ctest OR ninja run_all_tests")
message(STATUS "")
message(STATUS "========================================")
message(STATUS "")
