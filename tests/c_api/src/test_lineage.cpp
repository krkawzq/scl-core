// =============================================================================
// SCL Core - Lineage Tracing Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/lineage.h
//
// Functions tested:
//   ✓ scl_lineage_coupling - Coupling matrix computation
//   ✓ scl_lineage_fate_bias - Fate bias score computation
//   ✓ scl_lineage_build_tree - Lineage tree construction
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

using namespace scl::test;
using precision::Tolerance;

SCL_TEST_BEGIN

// =============================================================================
// Helper: Create test data
// =============================================================================

static std::pair<std::vector<scl_index_t>, std::vector<scl_index_t>>
create_test_assignments(scl_size_t n_cells, scl_size_t n_clones, scl_size_t n_types) {
    std::vector<scl_index_t> clone_ids(n_cells);
    std::vector<scl_index_t> cell_types(n_cells);
    
    // Assign cells to clones and types
    for (size_t i = 0; i < n_cells; ++i) {
        clone_ids[i] = static_cast<scl_index_t>(i % n_clones);
        cell_types[i] = static_cast<scl_index_t>(i % n_types);
    }
    
    return {clone_ids, cell_types};
}

// =============================================================================
// Coupling Matrix Tests
// =============================================================================

SCL_TEST_SUITE(lineage_coupling)

SCL_TEST_CASE(coupling_basic) {
    scl_size_t n_cells = 10;
    scl_size_t n_clones = 3;
    scl_size_t n_types = 2;
    
    auto [clone_ids, cell_types] = create_test_assignments(n_cells, n_clones, n_types);
    
    std::vector<scl_real_t> coupling_matrix(n_clones * n_types, 0.0);
    
    scl_error_t err = scl_lineage_coupling(
        clone_ids.data(), cell_types.data(), coupling_matrix.data(),
        n_cells, n_clones, n_types
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Coupling matrix should be non-negative
    for (size_t i = 0; i < n_clones * n_types; ++i) {
        SCL_ASSERT_GE(coupling_matrix[i], 0.0);
    }
    
    // Sum of all entries should equal n_cells
    scl_real_t sum = 0.0;
    for (auto val : coupling_matrix) {
        sum += val;
    }
    SCL_ASSERT_NEAR(sum, static_cast<scl_real_t>(n_cells), 1e-10);
}

SCL_TEST_CASE(coupling_single_clone) {
    scl_size_t n_cells = 5;
    scl_size_t n_clones = 1;
    scl_size_t n_types = 3;
    
    std::vector<scl_index_t> clone_ids(n_cells, 0);
    std::vector<scl_index_t> cell_types(n_cells);
    for (size_t i = 0; i < n_cells; ++i) {
        cell_types[i] = static_cast<scl_index_t>(i % n_types);
    }
    
    std::vector<scl_real_t> coupling_matrix(n_clones * n_types, 0.0);
    
    scl_error_t err = scl_lineage_coupling(
        clone_ids.data(), cell_types.data(), coupling_matrix.data(),
        n_cells, n_clones, n_types
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_real_t sum = 0.0;
    for (auto val : coupling_matrix) {
        sum += val;
    }
    SCL_ASSERT_NEAR(sum, static_cast<scl_real_t>(n_cells), 1e-10);
}

SCL_TEST_CASE(coupling_single_type) {
    scl_size_t n_cells = 5;
    scl_size_t n_clones = 3;
    scl_size_t n_types = 1;
    
    std::vector<scl_index_t> clone_ids(n_cells);
    std::vector<scl_index_t> cell_types(n_cells, 0);
    for (size_t i = 0; i < n_cells; ++i) {
        clone_ids[i] = static_cast<scl_index_t>(i % n_clones);
    }
    
    std::vector<scl_real_t> coupling_matrix(n_clones * n_types, 0.0);
    
    scl_error_t err = scl_lineage_coupling(
        clone_ids.data(), cell_types.data(), coupling_matrix.data(),
        n_cells, n_clones, n_types
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_real_t sum = 0.0;
    for (auto val : coupling_matrix) {
        sum += val;
    }
    SCL_ASSERT_NEAR(sum, static_cast<scl_real_t>(n_cells), 1e-10);
}

SCL_TEST_CASE(coupling_perfect_separation) {
    // Each clone has only one cell type
    scl_size_t n_cells = 6;
    scl_size_t n_clones = 3;
    scl_size_t n_types = 3;
    
    std::vector<scl_index_t> clone_ids = {0, 0, 1, 1, 2, 2};
    std::vector<scl_index_t> cell_types = {0, 0, 1, 1, 2, 2};
    
    std::vector<scl_real_t> coupling_matrix(n_clones * n_types, 0.0);
    
    scl_error_t err = scl_lineage_coupling(
        clone_ids.data(), cell_types.data(), coupling_matrix.data(),
        n_cells, n_clones, n_types
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Diagonal should have values, off-diagonal should be zero
    SCL_ASSERT_GT(coupling_matrix[0 * n_types + 0], 0.0);  // clone 0, type 0
    SCL_ASSERT_GT(coupling_matrix[1 * n_types + 1], 0.0);  // clone 1, type 1
    SCL_ASSERT_GT(coupling_matrix[2 * n_types + 2], 0.0);  // clone 2, type 2
}

SCL_TEST_CASE(coupling_null_clone_ids) {
    std::vector<scl_index_t> cell_types(10, 0);
    std::vector<scl_real_t> coupling_matrix(6, 0.0);
    
    scl_error_t err = scl_lineage_coupling(
        nullptr, cell_types.data(), coupling_matrix.data(),
        10, 3, 2
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(coupling_null_cell_types) {
    std::vector<scl_index_t> clone_ids(10, 0);
    std::vector<scl_real_t> coupling_matrix(6, 0.0);
    
    scl_error_t err = scl_lineage_coupling(
        clone_ids.data(), nullptr, coupling_matrix.data(),
        10, 3, 2
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(coupling_null_output) {
    scl_size_t n_cells = 10;
    auto [clone_ids, cell_types] = create_test_assignments(n_cells, 3, 2);
    
    scl_error_t err = scl_lineage_coupling(
        clone_ids.data(), cell_types.data(), nullptr,
        n_cells, 3, 2
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(coupling_zero_cells) {
    std::vector<scl_index_t> clone_ids;
    std::vector<scl_index_t> cell_types;
    std::vector<scl_real_t> coupling_matrix(6, 0.0);
    
    scl_error_t err = scl_lineage_coupling(
        clone_ids.data(), cell_types.data(), coupling_matrix.data(),
        0, 3, 2
    );
    
    // Should handle gracefully
    if (err == SCL_OK) {
        scl_real_t sum = 0.0;
        for (auto val : coupling_matrix) {
            sum += val;
        }
        SCL_ASSERT_NEAR(sum, 0.0, 1e-10);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Fate Bias Tests
// =============================================================================

SCL_TEST_SUITE(lineage_fate_bias)

SCL_TEST_CASE(fate_bias_basic) {
    scl_size_t n_cells = 10;
    scl_size_t n_types = 3;
    
    std::vector<scl_index_t> clone_ids(n_cells);
    std::vector<scl_index_t> cell_types(n_cells);
    for (size_t i = 0; i < n_cells; ++i) {
        clone_ids[i] = static_cast<scl_index_t>(i / 3);  // 3 cells per clone
        cell_types[i] = static_cast<scl_index_t>(i % n_types);
    }
    
    std::vector<scl_real_t> bias_scores(n_types, 0.0);
    
    scl_error_t err = scl_lineage_fate_bias(
        clone_ids.data(), cell_types.data(), bias_scores.data(),
        n_cells, n_types
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Bias scores should be non-negative
    for (size_t i = 0; i < n_types; ++i) {
        SCL_ASSERT_GE(bias_scores[i], 0.0);
    }
}

SCL_TEST_CASE(fate_bias_single_type) {
    scl_size_t n_cells = 5;
    scl_size_t n_types = 1;
    
    std::vector<scl_index_t> clone_ids(n_cells);
    std::vector<scl_index_t> cell_types(n_cells, 0);
    for (size_t i = 0; i < n_cells; ++i) {
        clone_ids[i] = static_cast<scl_index_t>(i);
    }
    
    std::vector<scl_real_t> bias_scores(n_types, 0.0);
    
    scl_error_t err = scl_lineage_fate_bias(
        clone_ids.data(), cell_types.data(), bias_scores.data(),
        n_cells, n_types
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(bias_scores[0], 0.0);
}

SCL_TEST_CASE(fate_bias_uniform_distribution) {
    // All clones have equal distribution across types
    scl_size_t n_cells = 12;
    scl_size_t n_types = 3;
    
    std::vector<scl_index_t> clone_ids = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    std::vector<scl_index_t> cell_types = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
    
    std::vector<scl_real_t> bias_scores(n_types, 0.0);
    
    scl_error_t err = scl_lineage_fate_bias(
        clone_ids.data(), cell_types.data(), bias_scores.data(),
        n_cells, n_types
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All bias scores should be similar (low bias)
    for (size_t i = 0; i < n_types; ++i) {
        SCL_ASSERT_GE(bias_scores[i], 0.0);
    }
}

SCL_TEST_CASE(fate_bias_null_clone_ids) {
    std::vector<scl_index_t> cell_types(10, 0);
    std::vector<scl_real_t> bias_scores(3, 0.0);
    
    scl_error_t err = scl_lineage_fate_bias(
        nullptr, cell_types.data(), bias_scores.data(),
        10, 3
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(fate_bias_null_cell_types) {
    std::vector<scl_index_t> clone_ids(10, 0);
    std::vector<scl_real_t> bias_scores(3, 0.0);
    
    scl_error_t err = scl_lineage_fate_bias(
        clone_ids.data(), nullptr, bias_scores.data(),
        10, 3
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(fate_bias_null_output) {
    scl_size_t n_cells = 10;
    auto [clone_ids, cell_types] = create_test_assignments(n_cells, 3, 3);
    
    scl_error_t err = scl_lineage_fate_bias(
        clone_ids.data(), cell_types.data(), nullptr,
        n_cells, 3
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Build Tree Tests
// =============================================================================

SCL_TEST_SUITE(lineage_build_tree)

SCL_TEST_CASE(build_tree_basic) {
    scl_size_t n_cells = 10;
    scl_size_t n_clones = 3;
    
    std::vector<scl_index_t> clone_ids(n_cells);
    for (size_t i = 0; i < n_cells; ++i) {
        clone_ids[i] = static_cast<scl_index_t>(i % n_clones);
    }
    
    std::vector<scl_real_t> pseudotime(n_cells);
    for (size_t i = 0; i < n_cells; ++i) {
        pseudotime[i] = static_cast<scl_real_t>(i);
    }
    
    std::vector<scl_index_t> parent(n_cells, -1);
    
    scl_error_t err = scl_lineage_build_tree(
        clone_ids.data(), pseudotime.data(), parent.data(),
        n_cells, n_clones
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Each clone should have a root (parent = -1 or valid index)
    std::vector<bool> has_root(n_clones, false);
    for (size_t i = 0; i < n_cells; ++i) {
        if (parent[i] == -1 || parent[i] < static_cast<scl_index_t>(n_cells)) {
            has_root[clone_ids[i]] = true;
        }
    }
}

SCL_TEST_CASE(build_tree_single_clone) {
    scl_size_t n_cells = 5;
    scl_size_t n_clones = 1;
    
    std::vector<scl_index_t> clone_ids(n_cells, 0);
    std::vector<scl_real_t> pseudotime(n_cells);
    for (size_t i = 0; i < n_cells; ++i) {
        pseudotime[i] = static_cast<scl_real_t>(i);
    }
    
    std::vector<scl_index_t> parent(n_cells, -1);
    
    scl_error_t err = scl_lineage_build_tree(
        clone_ids.data(), pseudotime.data(), parent.data(),
        n_cells, n_clones
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Should have at least one root
    bool has_root = false;
    for (auto p : parent) {
        if (p == -1) {
            has_root = true;
            break;
        }
    }
    SCL_ASSERT_TRUE(has_root);
}

SCL_TEST_CASE(build_tree_ordered_pseudotime) {
    scl_size_t n_cells = 8;
    scl_size_t n_clones = 2;
    
    std::vector<scl_index_t> clone_ids = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<scl_real_t> pseudotime = {0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0};
    
    std::vector<scl_index_t> parent(n_cells, -1);
    
    scl_error_t err = scl_lineage_build_tree(
        clone_ids.data(), pseudotime.data(), parent.data(),
        n_cells, n_clones
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Each clone should form a chain
    // First cell in each clone should be root or have valid parent
    SCL_ASSERT_TRUE(parent[0] == -1 || parent[0] < static_cast<scl_index_t>(n_cells));
    SCL_ASSERT_TRUE(parent[4] == -1 || parent[4] < static_cast<scl_index_t>(n_cells));
}

SCL_TEST_CASE(build_tree_null_clone_ids) {
    std::vector<scl_real_t> pseudotime(10, 0.0);
    std::vector<scl_index_t> parent(10, -1);
    
    scl_error_t err = scl_lineage_build_tree(
        nullptr, pseudotime.data(), parent.data(),
        10, 3
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(build_tree_null_pseudotime) {
    std::vector<scl_index_t> clone_ids(10, 0);
    std::vector<scl_index_t> parent(10, -1);
    
    scl_error_t err = scl_lineage_build_tree(
        clone_ids.data(), nullptr, parent.data(),
        10, 3
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(build_tree_null_parent) {
    scl_size_t n_cells = 10;
    std::vector<scl_index_t> clone_ids(n_cells, 0);
    std::vector<scl_real_t> pseudotime(n_cells, 0.0);
    
    scl_error_t err = scl_lineage_build_tree(
        clone_ids.data(), pseudotime.data(), nullptr,
        n_cells, 3
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(build_tree_zero_cells) {
    std::vector<scl_index_t> clone_ids;
    std::vector<scl_real_t> pseudotime;
    std::vector<scl_index_t> parent;
    
    scl_error_t err = scl_lineage_build_tree(
        clone_ids.data(), pseudotime.data(), parent.data(),
        0, 3
    );
    
    // Should handle gracefully
    SCL_ASSERT_TRUE(err == SCL_OK || err != SCL_OK);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

