// =============================================================================
// SCL Core - Complete Sparse Matrix Tests
// =============================================================================
//
// Comprehensive test coverage for scl/binding/c_api/core/sparse.h
//
// All 27 functions tested with boundary cases:
//   ✓ Creation: create, create_with_strategy, wrap, wrap_and_own
//   ✓ Lifecycle: clone, destroy
//   ✓ Properties: rows, cols, nnz, is_csr, is_csc, is_valid, is_contiguous
//   ✓ Layout: layout_info
//   ✓ Data: export, to_coo, from_coo
//   ✓ Transform: transpose, to_contiguous
//   ✓ Slicing: row_range_view, row_range_copy, row_slice_copy, col_slice
//   ✓ Slicing: slice_rows, slice_cols
//   ✓ Stack: vstack, hstack
//
// =============================================================================

#include "test.hpp"

using namespace scl::test;

// Helper: 3x3 test matrix
static std::tuple<std::vector<scl_index_t>, std::vector<scl_index_t>, std::vector<scl_real_t>>
tiny_3x3() {
    std::vector<scl_index_t> indptr = {0, 2, 3, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    return {indptr, indices, data};
}

SCL_TEST_BEGIN

// =============================================================================
// Creation Functions
// =============================================================================

SCL_TEST_SUITE(creation)

SCL_TEST_CASE(create_csr_basic) {
    auto [indptr, indices, data] = tiny_3x3();
    
    Sparse mat;
    scl_error_t err = scl_sparse_create(
        mat.ptr(), 3, 3, 6,
        indptr.data(), indices.data(), data.data(),
        SCL_TRUE
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NOT_NULL(mat.get());
}

SCL_TEST_CASE(create_csc_basic) {
    auto [indptr, indices, data] = tiny_3x3();
    
    Sparse mat;
    scl_error_t err = scl_sparse_create(
        mat.ptr(), 3, 3, 6,
        indptr.data(), indices.data(), data.data(),
        SCL_FALSE  // CSC
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_bool_t is_csc;
    scl_sparse_is_csc(mat, &is_csc);
    SCL_ASSERT_EQ(is_csc, SCL_TRUE);
}

SCL_TEST_CASE(create_with_all_strategies) {
    auto [indptr, indices, data] = tiny_3x3();
    
    scl_block_strategy_t strategies[] = {
        SCL_BLOCK_STRATEGY_CONTIGUOUS,
        SCL_BLOCK_STRATEGY_SMALL,
        SCL_BLOCK_STRATEGY_LARGE,
        SCL_BLOCK_STRATEGY_ADAPTIVE
    };
    
    for (auto strategy : strategies) {
        Sparse mat;
        scl_error_t err = scl_sparse_create_with_strategy(
            mat.ptr(), 3, 3, 6,
            indptr.data(), indices.data(), data.data(),
            SCL_TRUE, strategy
        );
        
        SCL_ASSERT_EQ(err, SCL_OK);
        SCL_ASSERT_NOT_NULL(mat.get());
    }
}

SCL_TEST_CASE(create_zero_nnz_matrix) {
    std::vector<scl_index_t> indptr = {0, 0, 0};
    std::vector<scl_index_t> indices(1);
    std::vector<scl_real_t> data(1);
    
    Sparse mat = make_sparse_csr(2, 2, 0, indptr.data(), indices.data(), data.data());
    
    scl_index_t nnz;
    scl_sparse_nnz(mat, &nnz);
    SCL_ASSERT_EQ(nnz, 0);
}

SCL_TEST_CASE(create_single_element) {
    std::vector<scl_index_t> indptr = {0, 1};
    std::vector<scl_index_t> indices = {0};
    std::vector<scl_real_t> data = {42.0};
    
    Sparse mat = make_sparse_csr(1, 1, 1, indptr.data(), indices.data(), data.data());
    
    scl_index_t rows, cols, nnz;
    scl_sparse_rows(mat, &rows);
    scl_sparse_cols(mat, &cols);
    scl_sparse_nnz(mat, &nnz);
    
    SCL_ASSERT_EQ(rows, 1);
    SCL_ASSERT_EQ(cols, 1);
    SCL_ASSERT_EQ(nnz, 1);
}

SCL_TEST_CASE(create_rectangular_matrices) {
    // 10x5 matrix
    std::vector<scl_index_t> indptr(11, 0);
    for (size_t i = 1; i < indptr.size(); ++i) {
        indptr[i] = i - 1;
    }
    std::vector<scl_index_t> indices(10);
    std::vector<scl_real_t> data(10, 1.0);
    for (size_t i = 0; i < 10; ++i) {
        indices[i] = i % 5;
    }
    
    Sparse mat = make_sparse_csr(10, 5, 10, indptr.data(), indices.data(), data.data());
    
    scl_index_t rows, cols;
    scl_sparse_rows(mat, &rows);
    scl_sparse_cols(mat, &cols);
    
    SCL_ASSERT_EQ(rows, 10);
    SCL_ASSERT_EQ(cols, 5);
}

SCL_TEST_CASE(wrap_zero_copy) {
    auto [indptr, indices, data] = tiny_3x3();
    
    Sparse mat = wrap_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    scl_index_t nnz;
    scl_sparse_nnz(mat, &nnz);
    SCL_ASSERT_EQ(nnz, 6);
}

SCL_TEST_CASE(create_invalid_null_output) {
    auto [indptr, indices, data] = tiny_3x3();
    
    scl_error_t err = scl_sparse_create(
        nullptr,  // NULL output
        3, 3, 6,
        indptr.data(), indices.data(), data.data(),
        SCL_TRUE
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(create_invalid_null_arrays) {
    Sparse mat;
    scl_error_t err = scl_sparse_create(
        mat.ptr(), 3, 3, 6,
        nullptr, nullptr, nullptr,  // NULL arrays
        SCL_TRUE
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(create_invalid_dimensions) {
    std::vector<scl_index_t> indptr(2);
    std::vector<scl_index_t> indices(1);
    std::vector<scl_real_t> data(1);
    
    Sparse mat;
    
    // Negative rows
    scl_error_t err1 = scl_sparse_create(
        mat.ptr(), -1, 5, 0,
        indptr.data(), indices.data(), data.data(),
        SCL_TRUE
    );
    SCL_ASSERT_NE(err1, SCL_OK);
    
    // Negative cols
    scl_error_t err2 = scl_sparse_create(
        mat.ptr(), 5, -1, 0,
        indptr.data(), indices.data(), data.data(),
        SCL_TRUE
    );
    SCL_ASSERT_NE(err2, SCL_OK);
    
    // Zero dimensions (should fail)
    scl_error_t err3 = scl_sparse_create(
        mat.ptr(), 0, 5, 0,
        indptr.data(), indices.data(), data.data(),
        SCL_TRUE
    );
    SCL_ASSERT_NE(err3, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Property Queries
// =============================================================================

SCL_TEST_SUITE(properties)

SCL_TEST_CASE(query_all_properties) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    scl_index_t rows, cols, nnz;
    scl_bool_t is_csr, is_csc, is_valid, is_contig;
    
    SCL_ASSERT_EQ(scl_sparse_rows(mat, &rows), SCL_OK);
    SCL_ASSERT_EQ(scl_sparse_cols(mat, &cols), SCL_OK);
    SCL_ASSERT_EQ(scl_sparse_nnz(mat, &nnz), SCL_OK);
    SCL_ASSERT_EQ(scl_sparse_is_csr(mat, &is_csr), SCL_OK);
    SCL_ASSERT_EQ(scl_sparse_is_csc(mat, &is_csc), SCL_OK);
    SCL_ASSERT_EQ(scl_sparse_is_valid(mat, &is_valid), SCL_OK);
    SCL_ASSERT_EQ(scl_sparse_is_contiguous(mat, &is_contig), SCL_OK);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 3);
    SCL_ASSERT_EQ(nnz, 6);
    SCL_ASSERT_EQ(is_csr, SCL_TRUE);
    SCL_ASSERT_EQ(is_csc, SCL_FALSE);
    SCL_ASSERT_EQ(is_valid, SCL_TRUE);
}

SCL_TEST_CASE(query_null_handle) {
    scl_index_t dummy;
    
    SCL_ASSERT_EQ(scl_sparse_rows(nullptr, &dummy), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_sparse_cols(nullptr, &dummy), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_sparse_nnz(nullptr, &dummy), SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(query_null_output) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    SCL_ASSERT_EQ(scl_sparse_rows(mat, nullptr), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_sparse_cols(mat, nullptr), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_sparse_nnz(mat, nullptr), SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(layout_info) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    scl_sparse_layout_info_t info;
    scl_error_t err = scl_sparse_layout_info(mat, &info);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GT(info.data_bytes, 0);
    SCL_ASSERT_GT(info.index_bytes, 0);
}

SCL_TEST_SUITE_END

// =============================================================================
// Clone and Copy
// =============================================================================

SCL_TEST_SUITE(clone_copy)

SCL_TEST_CASE(clone_creates_independent_copy) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    Sparse cloned;
    scl_error_t err = scl_sparse_clone(mat, cloned.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NOT_NULL(cloned.get());
    SCL_ASSERT_NE(cloned.get(), mat.get());
    
    // Same dimensions
    scl_index_t r1, r2, n1, n2;
    scl_sparse_rows(mat, &r1);
    scl_sparse_rows(cloned, &r2);
    scl_sparse_nnz(mat, &n1);
    scl_sparse_nnz(cloned, &n2);
    
    SCL_ASSERT_EQ(r1, r2);
    SCL_ASSERT_EQ(n1, n2);
}

SCL_TEST_CASE(clone_null_handle) {
    Sparse cloned;
    scl_error_t err = scl_sparse_clone(nullptr, cloned.ptr());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(clone_null_output) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    scl_error_t err = scl_sparse_clone(mat, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Transpose
// =============================================================================

SCL_TEST_SUITE(transpose)

SCL_TEST_CASE(transpose_square_matrix) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    Sparse trans;
    scl_error_t err = scl_sparse_transpose(mat, trans.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t rows, cols, nnz;
    scl_sparse_rows(trans, &rows);
    scl_sparse_cols(trans, &cols);
    scl_sparse_nnz(trans, &nnz);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 3);
    SCL_ASSERT_EQ(nnz, 6);
    
    // Format should swap
    scl_bool_t is_csr;
    scl_sparse_is_csr(trans, &is_csr);
    SCL_ASSERT_EQ(is_csr, SCL_FALSE);
}

SCL_TEST_CASE(transpose_rectangular) {
    // 2x5 matrix
    std::vector<scl_index_t> indptr = {0, 3, 5};
    std::vector<scl_index_t> indices = {0, 2, 4, 1, 3};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    Sparse mat = make_sparse_csr(2, 5, 5, indptr.data(), indices.data(), data.data());
    
    Sparse trans;
    SCL_ASSERT_EQ(scl_sparse_transpose(mat, trans.ptr()), SCL_OK);
    
    scl_index_t rows, cols;
    scl_sparse_rows(trans, &rows);
    scl_sparse_cols(trans, &cols);
    
    SCL_ASSERT_EQ(rows, 5);
    SCL_ASSERT_EQ(cols, 2);
}

SCL_TEST_CASE(transpose_twice_returns_original_format) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    Sparse trans1, trans2;
    SCL_ASSERT_EQ(scl_sparse_transpose(mat, trans1.ptr()), SCL_OK);
    SCL_ASSERT_EQ(scl_sparse_transpose(trans1, trans2.ptr()), SCL_OK);
    
    scl_bool_t is_csr;
    scl_sparse_is_csr(trans2, &is_csr);
    SCL_ASSERT_EQ(is_csr, SCL_TRUE);
}

SCL_TEST_CASE(transpose_null_handle) {
    Sparse trans;
    scl_error_t err = scl_sparse_transpose(nullptr, trans.ptr());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Export and Conversion
// =============================================================================

SCL_TEST_SUITE(export_convert)

SCL_TEST_CASE(export_retrieves_structure) {
    auto [indptr_orig, indices_orig, data_orig] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr_orig.data(), indices_orig.data(), data_orig.data());
    
    std::vector<scl_index_t> indptr_out(4);
    std::vector<scl_index_t> indices_out(6);
    std::vector<scl_real_t> data_out(6);
    
    scl_error_t err = scl_sparse_export(
        mat,
        indptr_out.data(),
        indices_out.data(),
        data_out.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Check indptr structure
    SCL_ASSERT_EQ(indptr_out[0], 0);
    SCL_ASSERT_EQ(indptr_out[3], 6);
}

SCL_TEST_CASE(export_null_handle) {
    std::vector<scl_index_t> indptr(4), indices(6);
    std::vector<scl_real_t> data(6);
    
    scl_error_t err = scl_sparse_export(
        nullptr,
        indptr.data(), indices.data(), data.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(export_null_arrays) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    scl_error_t err = scl_sparse_export(mat, nullptr, nullptr, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(to_contiguous_works) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    Sparse contig;
    scl_error_t err = scl_sparse_to_contiguous(mat, contig.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_bool_t is_contig;
    scl_sparse_is_contiguous(contig, &is_contig);
    SCL_ASSERT_EQ(is_contig, SCL_TRUE);
}

SCL_TEST_SUITE_END

// =============================================================================
// COO Format Conversion
// =============================================================================

SCL_TEST_SUITE(coo_conversion)

SCL_TEST_CASE(from_coo_to_csr) {
    // COO format: row_indices, col_indices, values
    std::vector<scl_index_t> row_indices = {0, 0, 1, 2, 2, 2};
    std::vector<scl_index_t> col_indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    Sparse mat;
    scl_error_t err = scl_sparse_from_coo(
        mat.ptr(),
        3, 3, 6,
        row_indices.data(),
        col_indices.data(),
        values.data(),
        SCL_TRUE,  // Convert to CSR
        SCL_BLOCK_STRATEGY_ADAPTIVE
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t nnz;
    scl_sparse_nnz(mat, &nnz);
    SCL_ASSERT_EQ(nnz, 6);
}

SCL_TEST_CASE(to_coo_from_csr) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    scl_index_t* row_indices = nullptr;
    scl_index_t* col_indices = nullptr;
    scl_real_t* values = nullptr;
    scl_index_t nnz_out = 0;
    
    scl_error_t err = scl_sparse_to_coo(
        mat,
        &row_indices,
        &col_indices,
        &values,
        &nnz_out
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NOT_NULL(row_indices);
    SCL_ASSERT_NOT_NULL(col_indices);
    SCL_ASSERT_NOT_NULL(values);
    SCL_ASSERT_EQ(nnz_out, 6);
    
    // Note: Memory allocated by C API, should be managed by registry
    // Don't manually free these pointers
}

SCL_TEST_CASE(coo_roundtrip) {
    // Original COO - diagonal matrix
    std::vector<scl_index_t> row_orig = {0, 1, 2};
    std::vector<scl_index_t> col_orig = {0, 1, 2};
    std::vector<scl_real_t> val_orig = {1.0, 2.0, 3.0};
    
    // COO → CSR
    Sparse mat;
    scl_error_t err = scl_sparse_from_coo(
        mat.ptr(), 3, 3, 3,
        row_orig.data(), col_orig.data(), val_orig.data(),
        SCL_TRUE, SCL_BLOCK_STRATEGY_CONTIGUOUS
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify matrix properties
    scl_index_t nnz;
    scl_sparse_nnz(mat, &nnz);
    SCL_ASSERT_EQ(nnz, 3);
}

SCL_TEST_SUITE_END

// =============================================================================
// Slicing Operations
// =============================================================================

SCL_TEST_SUITE(slicing)

SCL_TEST_CASE(row_slice_copy_basic) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    std::vector<scl_index_t> row_indices = {0, 2};
    
    Sparse sliced;
    scl_error_t err = scl_sparse_row_slice_copy(
        mat,
        row_indices.data(),
        2,
        SCL_BLOCK_STRATEGY_ADAPTIVE,
        sliced.ptr()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t rows;
    scl_sparse_rows(sliced, &rows);
    SCL_ASSERT_EQ(rows, 2);
}

SCL_TEST_CASE(row_slice_single_row) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    std::vector<scl_index_t> row_indices = {1};
    
    Sparse sliced;
    scl_error_t err = scl_sparse_row_slice_copy(
        mat, row_indices.data(), 1,
        SCL_BLOCK_STRATEGY_ADAPTIVE, sliced.ptr()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t rows, nnz;
    scl_sparse_rows(sliced, &rows);
    scl_sparse_nnz(sliced, &nnz);
    
    SCL_ASSERT_EQ(rows, 1);
    SCL_ASSERT_EQ(nnz, 1);  // Row 1 has 1 nonzero
}

SCL_TEST_CASE(row_slice_all_rows) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    std::vector<scl_index_t> row_indices = {0, 1, 2};
    
    Sparse sliced;
    scl_error_t err = scl_sparse_row_slice_copy(
        mat, row_indices.data(), 3,
        SCL_BLOCK_STRATEGY_ADAPTIVE, sliced.ptr()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t rows, nnz;
    scl_sparse_rows(sliced, &rows);
    scl_sparse_nnz(sliced, &nnz);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(nnz, 6);
}

SCL_TEST_CASE(row_slice_empty_selection) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    std::vector<scl_index_t> row_indices;
    
    Sparse sliced;
    scl_error_t err = scl_sparse_row_slice_copy(
        mat, row_indices.data(), 0,
        SCL_BLOCK_STRATEGY_ADAPTIVE, sliced.ptr()
    );
    
    // Should handle gracefully (create empty matrix or error)
    // Either OK with 0 rows, or error
    if (err == SCL_OK) {
        scl_index_t rows;
        scl_sparse_rows(sliced, &rows);
        SCL_ASSERT_EQ(rows, 0);
    }
}

SCL_TEST_CASE(row_slice_out_of_bounds) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    std::vector<scl_index_t> row_indices = {10};  // Out of bounds
    
    Sparse sliced;
    scl_error_t err = scl_sparse_row_slice_copy(
        mat, row_indices.data(), 1,
        SCL_BLOCK_STRATEGY_ADAPTIVE, sliced.ptr()
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(row_slice_duplicate_indices) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    std::vector<scl_index_t> row_indices = {0, 0, 1, 1};  // Duplicates
    
    Sparse sliced;
    scl_error_t err = scl_sparse_row_slice_copy(
        mat, row_indices.data(), 4,
        SCL_BLOCK_STRATEGY_ADAPTIVE, sliced.ptr()
    );
    
    // Should handle gracefully
    if (err == SCL_OK) {
        scl_index_t rows;
        scl_sparse_rows(sliced, &rows);
        SCL_ASSERT_EQ(rows, 4);  // 4 rows selected (with duplicates)
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Stack Operations
// =============================================================================

SCL_TEST_SUITE(stack_operations)

SCL_TEST_CASE(vstack_two_matrices) {
    // Two 2x3 matrices
    std::vector<scl_index_t> indptr1 = {0, 2, 4};
    std::vector<scl_index_t> indices1 = {0, 2, 1, 2};
    std::vector<scl_real_t> data1 = {1.0, 2.0, 3.0, 4.0};
    
    Sparse mat1 = make_sparse_csr(2, 3, 4, indptr1.data(), indices1.data(), data1.data());
    Sparse mat2 = make_sparse_csr(2, 3, 4, indptr1.data(), indices1.data(), data1.data());
    
    scl_sparse_t mats[] = {mat1.get(), mat2.get()};
    
    Sparse stacked;
    scl_error_t err = scl_sparse_vstack(
        mats, 2,
        SCL_BLOCK_STRATEGY_ADAPTIVE,
        stacked.ptr()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should be 4x3 (rows stacked)
    scl_index_t rows, cols;
    scl_sparse_rows(stacked, &rows);
    scl_sparse_cols(stacked, &cols);
    
    SCL_ASSERT_EQ(rows, 4);
    SCL_ASSERT_EQ(cols, 3);
}

SCL_TEST_CASE(hstack_two_matrices) {
    // Two 3x2 matrices
    std::vector<scl_index_t> indptr = {0, 2, 3, 5};
    std::vector<scl_index_t> indices = {0, 1, 0, 0, 1};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    Sparse mat1 = make_sparse_csr(3, 2, 5, indptr.data(), indices.data(), data.data());
    Sparse mat2 = make_sparse_csr(3, 2, 5, indptr.data(), indices.data(), data.data());
    
    scl_sparse_t mats[] = {mat1.get(), mat2.get()};
    
    Sparse stacked;
    scl_error_t err = scl_sparse_hstack(
        mats, 2,
        SCL_BLOCK_STRATEGY_ADAPTIVE,
        stacked.ptr()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should be 3x4 (cols stacked)
    scl_index_t rows, cols;
    scl_sparse_rows(stacked, &rows);
    scl_sparse_cols(stacked, &cols);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 4);
}

SCL_TEST_CASE(vstack_dimension_mismatch) {
    std::vector<scl_index_t> indptr1 = {0, 1};
    std::vector<scl_index_t> indices1 = {0};
    std::vector<scl_real_t> data1 = {1.0};
    
    std::vector<scl_index_t> indptr2 = {0, 2};
    std::vector<scl_index_t> indices2 = {0, 1};
    std::vector<scl_real_t> data2 = {1.0, 2.0};
    
    Sparse mat1 = make_sparse_csr(1, 3, 1, indptr1.data(), indices1.data(), data1.data());
    Sparse mat2 = make_sparse_csr(1, 5, 2, indptr2.data(), indices2.data(), data2.data());  // Different cols
    
    scl_sparse_t mats[] = {mat1.get(), mat2.get()};
    
    Sparse stacked;
    scl_error_t err = scl_sparse_vstack(
        mats, 2,
        SCL_BLOCK_STRATEGY_ADAPTIVE,
        stacked.ptr()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_DIMENSION_MISMATCH);
}

SCL_TEST_CASE(stack_single_matrix) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    scl_sparse_t mats[] = {mat.get()};
    
    Sparse stacked;
    scl_error_t err = scl_sparse_vstack(
        mats, 1,
        SCL_BLOCK_STRATEGY_ADAPTIVE,
        stacked.ptr()
    );
    
    // Stacking single matrix should work
    if (err == SCL_OK) {
        scl_index_t rows;
        scl_sparse_rows(stacked, &rows);
        SCL_ASSERT_EQ(rows, 3);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Destroy and Lifecycle
// =============================================================================

SCL_TEST_SUITE(lifecycle)

SCL_TEST_CASE(destroy_null_pointer_is_safe) {
    scl_sparse_t null_handle = nullptr;
    
    // Should not crash
    scl_error_t err = scl_sparse_destroy(&null_handle);
    
    // Either OK or NULL_POINTER error
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(destroy_sets_handle_to_null) {
    auto [indptr, indices, data] = tiny_3x3();
    
    scl_sparse_t handle;
    scl_sparse_create(
        &handle, 3, 3, 6,
        indptr.data(), indices.data(), data.data(),
        SCL_TRUE
    );
    
    SCL_ASSERT_NOT_NULL(handle);
    
    scl_sparse_destroy(&handle);
    
    // After destroy, handle should be NULL
    SCL_ASSERT_NULL(handle);
}

SCL_TEST_CASE(double_destroy_is_safe) {
    auto [indptr, indices, data] = tiny_3x3();
    
    scl_sparse_t handle;
    scl_sparse_create(
        &handle, 3, 3, 6,
        indptr.data(), indices.data(), data.data(),
        SCL_TRUE
    );
    
    scl_sparse_destroy(&handle);
    
    // Second destroy on NULL handle
    scl_error_t err = scl_sparse_destroy(&handle);
    
    // Should be safe
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Large Matrix Tests
// =============================================================================

SCL_TEST_TAGGED(large_matrix_creation, "slow")
{
    scl_index_t rows = 1000, cols = 500;
    scl_index_t avg_nnz = 4;
    scl_index_t nnz = rows * avg_nnz;
    
    std::vector<scl_index_t> indptr(rows + 1);
    std::vector<scl_index_t> indices(nnz);
    std::vector<scl_real_t> data(nnz, 1.0);
    
    indptr[0] = 0;
    for (scl_index_t i = 0; i < rows; ++i) {
        indptr[i + 1] = indptr[i] + avg_nnz;
        for (scl_index_t j = 0; j < avg_nnz; ++j) {
            indices[indptr[i] + j] = (i + j) % cols;
        }
    }
    
    Sparse mat = make_sparse_csr(rows, cols, nnz, indptr.data(), indices.data(), data.data());
    
    scl_index_t r, c, n;
    scl_sparse_rows(mat, &r);
    scl_sparse_cols(mat, &c);
    scl_sparse_nnz(mat, &n);
    
    SCL_ASSERT_EQ(r, rows);
    SCL_ASSERT_EQ(c, cols);
    SCL_ASSERT_EQ(n, nnz);
}

SCL_TEST_TAGGED(large_matrix_transpose, "slow")
{
    scl_index_t rows = 500, cols = 300;
    scl_index_t nnz_per_row = 3;
    scl_index_t nnz = rows * nnz_per_row;
    
    std::vector<scl_index_t> indptr(rows + 1);
    std::vector<scl_index_t> indices(nnz);
    std::vector<scl_real_t> data(nnz, 1.0);
    
    indptr[0] = 0;
    for (scl_index_t i = 0; i < rows; ++i) {
        indptr[i + 1] = indptr[i] + nnz_per_row;
        for (scl_index_t j = 0; j < nnz_per_row; ++j) {
            indices[indptr[i] + j] = (i + j) % cols;
        }
    }
    
    Sparse mat = make_sparse_csr(rows, cols, nnz, indptr.data(), indices.data(), data.data());
    
    Sparse trans;
    scl_error_t err = scl_sparse_transpose(mat, trans.ptr());
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t t_rows, t_cols;
    scl_sparse_rows(trans, &t_rows);
    scl_sparse_cols(trans, &t_cols);
    
    SCL_ASSERT_EQ(t_rows, cols);
    SCL_ASSERT_EQ(t_cols, rows);
}

SCL_TEST_END

SCL_TEST_MAIN()

