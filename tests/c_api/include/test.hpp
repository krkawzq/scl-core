#pragma once

// =============================================================================
// SCL Core - C API Test Framework (Master Include)
// =============================================================================
//
// Single include for all test utilities.
//
// Components:
//   - core.hpp      : Test registration, runner, assertions, reporters
//   - guard.hpp     : RAII wrappers for C API handles
//   - oracle.hpp    : Eigen reference implementation and verification
//   - data.hpp      : Random and structured test data generators
//   - precision.hpp : Numerical precision comparison utilities
//   - blas.hpp      : BLAS reference implementation (optional)
//
// Usage:
//   #include "test.hpp"
//
//   SCL_TEST_BEGIN
//
//   SCL_TEST_RETRY(my_test, 5) {  // Retry 5 times
//       Random rng(42);
//       auto mat = random_sparse_random_shape(10, 100, 0.01, 0.1, rng);
//       
//       // Test implementation...
//       
//       // Compare with reference
//       using precision::Tolerance;
//       SCL_ASSERT_TRUE(matrices_equal(result, expected, Tolerance::normal()));
//   }
//
//   SCL_TEST_END
//   SCL_TEST_MAIN()
//
// =============================================================================

// Core testing framework
#include "core.hpp"

// RAII guards for C API handles
#include "guard.hpp"

// Eigen reference implementation
#include "oracle.hpp"

// Test data generators (with random shapes)
#include "data.hpp"

// Numerical precision comparison
#include "precision.hpp"

// BLAS reference (optional, needs libblas)
// Uncomment if BLAS is available:
// #include "blas.hpp"
