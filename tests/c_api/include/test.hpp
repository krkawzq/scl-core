#pragma once

// =============================================================================
// SCL Core - C API Test Framework (Master Include)
// =============================================================================
//
// Single include for all test utilities.
//
// Components:
//   - core.hpp   : Test registration, runner, assertions, reporters
//   - guard.hpp  : RAII wrappers for C API handles
//   - oracle.hpp : Eigen reference implementation and verification
//   - data.hpp   : Random and structured test data generators
//
// Usage:
//   #include "test.hpp"
//
//   SCL_TEST_BEGIN
//
//   SCL_TEST_UNIT(my_test) {
//       auto mat = scl::test::fixture::medium_sparse();
//       auto csr = scl::test::from_eigen_csr(mat);
//       
//       scl::test::Sparse result;
//       SCL_ASSERT_EQ(scl_sparse_transpose(csr.get(), result.ptr()), SCL_OK);
//       
//       auto eigen_result = scl::test::to_eigen_csr(result);
//       auto expected = scl::test::oracle::transpose_csr_to_csc(mat);
//       
//       SCL_ASSERT_TRUE(scl::test::matrices_equal(eigen_result, expected));
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

// Test data generators
#include "data.hpp"
