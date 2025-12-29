// =============================================================================
// SCL Core - Communication Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/communication.h
//
// Functions tested (13):
//   ✓ scl_comm_lr_score_matrix
//   ✓ scl_comm_lr_score_batch
//   ✓ scl_comm_lr_permutation_test
//   ✓ scl_comm_probability
//   ✓ scl_comm_filter_significant
//   ✓ scl_comm_aggregate_to_network
//   ✓ scl_comm_sender_score
//   ✓ scl_comm_receiver_score
//   ✓ scl_comm_network_centrality
//   ✓ scl_comm_spatial_score
//   ✓ scl_comm_expression_specificity
//   ✓ scl_comm_natmi_edge_weight
//
// =============================================================================

#include "test.hpp"

using namespace scl::test;

// Helper: Create small expression matrix (5 genes x 8 cells)
static auto small_expression_5x8() {
    std::vector<scl_index_t> indptr = {0, 3, 5, 7, 9, 12};
    std::vector<scl_index_t> indices = {0, 2, 4, 1, 3, 5, 6, 0, 7, 1, 2, 3};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    return std::make_tuple(indptr, indices, data);
}

SCL_TEST_BEGIN

// =============================================================================
// L-R Score Matrix Tests
// =============================================================================

SCL_TEST_SUITE(lr_score_matrix)

SCL_TEST_CASE(lr_score_matrix_basic) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    // 8 cells, 2 cell types
    std::vector<scl_index_t> cell_type_labels = {0, 0, 0, 0, 1, 1, 1, 1};

    std::vector<scl_real_t> score_matrix(4);  // 2x2 matrix

    scl_error_t err = scl_comm_lr_score_matrix(
        expr, cell_type_labels.data(),
        0, 1,  // ligand=0, receptor=1
        8, 2,
        score_matrix.data(),
        SCL_COMM_SCORE_MEAN_PRODUCT
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Scores should be non-negative
    for (scl_real_t score : score_matrix) {
        SCL_ASSERT_GE(score, 0.0);
        SCL_ASSERT_TRUE(std::isfinite(score));
    }
}

SCL_TEST_CASE(lr_score_matrix_all_methods) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<scl_real_t> score_matrix(4);

    scl_comm_score_method_t methods[] = {
        SCL_COMM_SCORE_MEAN_PRODUCT,
        SCL_COMM_SCORE_GEOMETRIC_MEAN,
        SCL_COMM_SCORE_MIN_MEAN,
        SCL_COMM_SCORE_PRODUCT,
        SCL_COMM_SCORE_NATMI
    };

    for (auto method : methods) {
        scl_error_t err = scl_comm_lr_score_matrix(
            expr, cell_type_labels.data(),
            0, 1, 8, 2,
            score_matrix.data(), method
        );

        SCL_ASSERT_EQ(err, SCL_OK);
    }
}

SCL_TEST_CASE(lr_score_matrix_null_inputs) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels(8);
    std::vector<scl_real_t> score_matrix(4);

    // NULL expression
    scl_error_t err = scl_comm_lr_score_matrix(
        nullptr, cell_type_labels.data(),
        0, 1, 8, 2, score_matrix.data(),
        SCL_COMM_SCORE_MEAN_PRODUCT
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL cell_type_labels
    err = scl_comm_lr_score_matrix(
        expr, nullptr,
        0, 1, 8, 2, score_matrix.data(),
        SCL_COMM_SCORE_MEAN_PRODUCT
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL score_matrix
    err = scl_comm_lr_score_matrix(
        expr, cell_type_labels.data(),
        0, 1, 8, 2, nullptr,
        SCL_COMM_SCORE_MEAN_PRODUCT
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(lr_score_matrix_invalid_gene_indices) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels(8);
    std::vector<scl_real_t> score_matrix(4);

    // Negative ligand
    scl_error_t err = scl_comm_lr_score_matrix(
        expr, cell_type_labels.data(),
        -1, 1, 8, 2, score_matrix.data(),
        SCL_COMM_SCORE_MEAN_PRODUCT
    );
    SCL_ASSERT_NE(err, SCL_OK);

    // Ligand >= n_genes
    err = scl_comm_lr_score_matrix(
        expr, cell_type_labels.data(),
        5, 1, 8, 2, score_matrix.data(),
        SCL_COMM_SCORE_MEAN_PRODUCT
    );
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// L-R Score Batch Tests
// =============================================================================

SCL_TEST_SUITE(lr_score_batch)

SCL_TEST_CASE(lr_score_batch_basic) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels = {0, 0, 0, 0, 1, 1, 1, 1};

    // 2 L-R pairs
    std::vector<scl_index_t> ligand_genes = {0, 1};
    std::vector<scl_index_t> receptor_genes = {2, 3};

    std::vector<scl_real_t> scores(8);  // 2 pairs * 2 types * 2 types

    scl_error_t err = scl_comm_lr_score_batch(
        expr, cell_type_labels.data(),
        ligand_genes.data(), receptor_genes.data(),
        2, 8, 2, scores.data(),
        SCL_COMM_SCORE_MEAN_PRODUCT
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Scores should be non-negative
    for (scl_real_t score : scores) {
        SCL_ASSERT_GE(score, 0.0);
    }
}

SCL_TEST_CASE(lr_score_batch_null_inputs) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels(8);
    std::vector<scl_index_t> ligand_genes = {0};
    std::vector<scl_index_t> receptor_genes = {1};
    std::vector<scl_real_t> scores(4);

    // NULL expression
    scl_error_t err = scl_comm_lr_score_batch(
        nullptr, cell_type_labels.data(),
        ligand_genes.data(), receptor_genes.data(),
        1, 8, 2, scores.data(), SCL_COMM_SCORE_MEAN_PRODUCT
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL ligand_genes
    err = scl_comm_lr_score_batch(
        expr, cell_type_labels.data(),
        nullptr, receptor_genes.data(),
        1, 8, 2, scores.data(), SCL_COMM_SCORE_MEAN_PRODUCT
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL receptor_genes
    err = scl_comm_lr_score_batch(
        expr, cell_type_labels.data(),
        ligand_genes.data(), nullptr,
        1, 8, 2, scores.data(), SCL_COMM_SCORE_MEAN_PRODUCT
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL scores
    err = scl_comm_lr_score_batch(
        expr, cell_type_labels.data(),
        ligand_genes.data(), receptor_genes.data(),
        1, 8, 2, nullptr, SCL_COMM_SCORE_MEAN_PRODUCT
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(lr_score_batch_zero_pairs) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels(8);
    std::vector<scl_index_t> ligand_genes(1);
    std::vector<scl_index_t> receptor_genes(1);
    std::vector<scl_real_t> scores(1);

    scl_error_t err = scl_comm_lr_score_batch(
        expr, cell_type_labels.data(),
        ligand_genes.data(), receptor_genes.data(),
        0, 8, 2, scores.data(), SCL_COMM_SCORE_MEAN_PRODUCT
    );
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Permutation Test Tests
// =============================================================================

SCL_TEST_SUITE(permutation_test)

SCL_TEST_CASE(permutation_test_basic) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels = {0, 0, 0, 0, 1, 1, 1, 1};

    scl_real_t observed_score, p_value;

    scl_error_t err = scl_comm_lr_permutation_test(
        expr, cell_type_labels.data(),
        0, 1, 0, 1,  // ligand=0, receptor=1, sender=0, receiver=1
        8, 100,  // 100 permutations
        &observed_score, &p_value,
        SCL_COMM_SCORE_MEAN_PRODUCT,
        12345
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_TRUE(std::isfinite(observed_score));
    SCL_ASSERT_GE(p_value, 0.0);
    SCL_ASSERT_LE(p_value, 1.0);
}

SCL_TEST_CASE(permutation_test_null_inputs) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels(8);
    scl_real_t observed_score, p_value;

    // NULL expression
    scl_error_t err = scl_comm_lr_permutation_test(
        nullptr, cell_type_labels.data(),
        0, 1, 0, 1, 8, 100,
        &observed_score, &p_value,
        SCL_COMM_SCORE_MEAN_PRODUCT, 12345
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL observed_score
    err = scl_comm_lr_permutation_test(
        expr, cell_type_labels.data(),
        0, 1, 0, 1, 8, 100,
        nullptr, &p_value,
        SCL_COMM_SCORE_MEAN_PRODUCT, 12345
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL p_value
    err = scl_comm_lr_permutation_test(
        expr, cell_type_labels.data(),
        0, 1, 0, 1, 8, 100,
        &observed_score, nullptr,
        SCL_COMM_SCORE_MEAN_PRODUCT, 12345
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(permutation_test_invalid_cell_types) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels(8);
    scl_real_t observed_score, p_value;

    // Negative sender
    scl_error_t err = scl_comm_lr_permutation_test(
        expr, cell_type_labels.data(),
        0, 1, -1, 1, 8, 100,
        &observed_score, &p_value,
        SCL_COMM_SCORE_MEAN_PRODUCT, 12345
    );
    SCL_ASSERT_NE(err, SCL_OK);

    // Receiver >= n_types
    err = scl_comm_lr_permutation_test(
        expr, cell_type_labels.data(),
        0, 1, 0, 5, 8, 100,
        &observed_score, &p_value,
        SCL_COMM_SCORE_MEAN_PRODUCT, 12345
    );
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Communication Probability Tests
// =============================================================================

SCL_TEST_SUITE(communication_probability)

SCL_TEST_CASE(comm_probability_basic) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels = {0, 0, 0, 0, 1, 1, 1, 1};

    std::vector<scl_index_t> ligand_genes = {0, 1};
    std::vector<scl_index_t> receptor_genes = {2, 3};

    std::vector<scl_real_t> p_values(8);  // 2 pairs * 2 types * 2 types
    std::vector<scl_real_t> scores(8);

    scl_error_t err = scl_comm_probability(
        expr, cell_type_labels.data(),
        ligand_genes.data(), receptor_genes.data(),
        2, 8, 2,
        p_values.data(), scores.data(),
        50, SCL_COMM_SCORE_MEAN_PRODUCT, 12345
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Check p-values
    for (scl_real_t p : p_values) {
        SCL_ASSERT_GE(p, 0.0);
        SCL_ASSERT_LE(p, 1.0);
    }

    // Check scores
    for (scl_real_t s : scores) {
        SCL_ASSERT_GE(s, 0.0);
    }
}

SCL_TEST_CASE(comm_probability_null_inputs) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels(8);
    std::vector<scl_index_t> ligand_genes = {0};
    std::vector<scl_index_t> receptor_genes = {1};
    std::vector<scl_real_t> p_values(4);
    std::vector<scl_real_t> scores(4);

    // NULL p_values
    scl_error_t err = scl_comm_probability(
        expr, cell_type_labels.data(),
        ligand_genes.data(), receptor_genes.data(),
        1, 8, 2, nullptr, scores.data(),
        50, SCL_COMM_SCORE_MEAN_PRODUCT, 12345
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL scores
    err = scl_comm_probability(
        expr, cell_type_labels.data(),
        ligand_genes.data(), receptor_genes.data(),
        1, 8, 2, p_values.data(), nullptr,
        50, SCL_COMM_SCORE_MEAN_PRODUCT, 12345
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Filter Significant Tests
// =============================================================================

SCL_TEST_SUITE(filter_significant)

SCL_TEST_CASE(filter_significant_basic) {
    // 2 pairs, 2 types
    std::vector<scl_real_t> p_values = {0.01, 0.5, 0.02, 0.9};

    std::vector<scl_index_t> pair_indices(4);
    std::vector<scl_index_t> sender_types(4);
    std::vector<scl_index_t> receiver_types(4);
    std::vector<scl_real_t> filtered_pvalues(4);
    scl_index_t n_results;

    scl_error_t err = scl_comm_filter_significant(
        p_values.data(), 2, 2, 0.05,
        pair_indices.data(), sender_types.data(), receiver_types.data(),
        filtered_pvalues.data(), 4, &n_results
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_LE(n_results, 4);

    // Filtered p-values should be <= threshold
    for (scl_index_t i = 0; i < n_results; ++i) {
        SCL_ASSERT_LE(filtered_pvalues[i], 0.05);
    }
}

SCL_TEST_CASE(filter_significant_null_inputs) {
    std::vector<scl_real_t> p_values(4);
    std::vector<scl_index_t> pair_indices(4);
    std::vector<scl_index_t> sender_types(4);
    std::vector<scl_index_t> receiver_types(4);
    std::vector<scl_real_t> filtered_pvalues(4);
    scl_index_t n_results;

    // NULL p_values
    scl_error_t err = scl_comm_filter_significant(
        nullptr, 2, 2, 0.05,
        pair_indices.data(), sender_types.data(), receiver_types.data(),
        filtered_pvalues.data(), 4, &n_results
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL n_results
    err = scl_comm_filter_significant(
        p_values.data(), 2, 2, 0.05,
        pair_indices.data(), sender_types.data(), receiver_types.data(),
        filtered_pvalues.data(), 4, nullptr
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Aggregate to Network Tests
// =============================================================================

SCL_TEST_SUITE(aggregate_to_network)

SCL_TEST_CASE(aggregate_to_network_basic) {
    // 2 pairs, 2 types
    std::vector<scl_real_t> scores = {1.0, 2.0, 3.0, 4.0};
    std::vector<scl_real_t> p_values = {0.01, 0.5, 0.02, 0.9};

    std::vector<scl_real_t> network_weights(4);  // 2x2
    std::vector<scl_index_t> network_counts(4);

    scl_error_t err = scl_comm_aggregate_to_network(
        scores.data(), p_values.data(),
        2, 2, 0.05,
        network_weights.data(), network_counts.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Weights should be non-negative
    for (scl_real_t w : network_weights) {
        SCL_ASSERT_GE(w, 0.0);
    }

    // Counts should be non-negative
    for (scl_index_t c : network_counts) {
        SCL_ASSERT_GE(c, 0);
    }
}

SCL_TEST_CASE(aggregate_to_network_null_inputs) {
    std::vector<scl_real_t> scores(4);
    std::vector<scl_real_t> p_values(4);
    std::vector<scl_real_t> network_weights(4);
    std::vector<scl_index_t> network_counts(4);

    // NULL scores
    scl_error_t err = scl_comm_aggregate_to_network(
        nullptr, p_values.data(), 2, 2, 0.05,
        network_weights.data(), network_counts.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL p_values
    err = scl_comm_aggregate_to_network(
        scores.data(), nullptr, 2, 2, 0.05,
        network_weights.data(), network_counts.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL network_weights
    err = scl_comm_aggregate_to_network(
        scores.data(), p_values.data(), 2, 2, 0.05,
        nullptr, network_counts.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL network_counts
    err = scl_comm_aggregate_to_network(
        scores.data(), p_values.data(), 2, 2, 0.05,
        network_weights.data(), nullptr
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Sender/Receiver Score Tests
// =============================================================================

SCL_TEST_SUITE(sender_receiver_scores)

SCL_TEST_CASE(sender_score_basic) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<scl_index_t> ligand_genes = {0, 1};

    std::vector<scl_real_t> scores(2);  // 2 types

    scl_error_t err = scl_comm_sender_score(
        expr, cell_type_labels.data(),
        ligand_genes.data(), 2,
        8, 2, scores.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    for (scl_real_t s : scores) {
        SCL_ASSERT_GE(s, 0.0);
        SCL_ASSERT_TRUE(std::isfinite(s));
    }
}

SCL_TEST_CASE(receiver_score_basic) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<scl_index_t> receptor_genes = {2, 3};

    std::vector<scl_real_t> scores(2);  // 2 types

    scl_error_t err = scl_comm_receiver_score(
        expr, cell_type_labels.data(),
        receptor_genes.data(), 2,
        8, 2, scores.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    for (scl_real_t s : scores) {
        SCL_ASSERT_GE(s, 0.0);
        SCL_ASSERT_TRUE(std::isfinite(s));
    }
}

SCL_TEST_CASE(sender_receiver_null_inputs) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels(8);
    std::vector<scl_index_t> genes = {0};
    std::vector<scl_real_t> scores(2);

    // NULL expression
    scl_error_t err = scl_comm_sender_score(
        nullptr, cell_type_labels.data(),
        genes.data(), 1, 8, 2, scores.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL scores
    err = scl_comm_sender_score(
        expr, cell_type_labels.data(),
        genes.data(), 1, 8, 2, nullptr
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Network Centrality Tests
// =============================================================================

SCL_TEST_SUITE(network_centrality)

SCL_TEST_CASE(network_centrality_basic) {
    // 3x3 network
    std::vector<scl_real_t> network_weights = {
        0.0, 1.0, 2.0,
        1.0, 0.0, 3.0,
        2.0, 3.0, 0.0
    };

    std::vector<scl_real_t> in_degree(3);
    std::vector<scl_real_t> out_degree(3);
    std::vector<scl_real_t> betweenness(3);

    scl_error_t err = scl_comm_network_centrality(
        network_weights.data(), 3,
        in_degree.data(), out_degree.data(), betweenness.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // All centrality measures should be non-negative
    for (scl_real_t d : in_degree) {
        SCL_ASSERT_GE(d, 0.0);
    }
    for (scl_real_t d : out_degree) {
        SCL_ASSERT_GE(d, 0.0);
    }
    for (scl_real_t b : betweenness) {
        SCL_ASSERT_GE(b, 0.0);
    }
}

SCL_TEST_CASE(network_centrality_null_inputs) {
    std::vector<scl_real_t> network_weights(9);
    std::vector<scl_real_t> in_degree(3);
    std::vector<scl_real_t> out_degree(3);
    std::vector<scl_real_t> betweenness(3);

    // NULL network_weights
    scl_error_t err = scl_comm_network_centrality(
        nullptr, 3,
        in_degree.data(), out_degree.data(), betweenness.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL in_degree
    err = scl_comm_network_centrality(
        network_weights.data(), 3,
        nullptr, out_degree.data(), betweenness.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL out_degree
    err = scl_comm_network_centrality(
        network_weights.data(), 3,
        in_degree.data(), nullptr, betweenness.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL betweenness
    err = scl_comm_network_centrality(
        network_weights.data(), 3,
        in_degree.data(), out_degree.data(), nullptr
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Spatial Communication Tests
// =============================================================================

SCL_TEST_SUITE(spatial_communication)

SCL_TEST_CASE(spatial_score_basic) {
    auto [indptr_expr, indices_expr, data_expr] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr_expr.data(), indices_expr.data(), data_expr.data());

    // Simple spatial graph: 0-1, 2-3, etc.
    std::vector<scl_index_t> indptr_spatial = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<scl_index_t> indices_spatial = {1, 0, 3, 2, 5, 4, 7, 6};
    std::vector<scl_real_t> data_spatial(8, 1.0);

    Sparse spatial_graph = make_sparse_csr(8, 8, 8,
        indptr_spatial.data(), indices_spatial.data(), data_spatial.data());

    std::vector<scl_real_t> cell_scores(8);

    scl_error_t err = scl_comm_spatial_score(
        expr, spatial_graph,
        0, 1,  // ligand=0, receptor=1
        8, cell_scores.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    for (scl_real_t s : cell_scores) {
        SCL_ASSERT_GE(s, 0.0);
        SCL_ASSERT_TRUE(std::isfinite(s));
    }
}

SCL_TEST_CASE(spatial_score_null_inputs) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> cell_scores(8);

    // NULL expression
    scl_error_t err = scl_comm_spatial_score(
        nullptr, expr, 0, 1, 8, cell_scores.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL spatial_graph
    err = scl_comm_spatial_score(
        expr, nullptr, 0, 1, 8, cell_scores.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL cell_scores
    err = scl_comm_spatial_score(
        expr, expr, 0, 1, 8, nullptr
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Expression Specificity Tests
// =============================================================================

SCL_TEST_SUITE(expression_specificity)

SCL_TEST_CASE(expression_specificity_basic) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels = {0, 0, 0, 0, 1, 1, 1, 1};

    std::vector<scl_real_t> specificity(2);  // 2 types

    scl_error_t err = scl_comm_expression_specificity(
        expr, cell_type_labels.data(),
        0, 8, 2, specificity.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Specificity should be in [0, 1]
    for (scl_real_t s : specificity) {
        SCL_ASSERT_GE(s, 0.0);
        SCL_ASSERT_LE(s, 1.0 + 1e-6);
    }
}

SCL_TEST_CASE(expression_specificity_null_inputs) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels(8);
    std::vector<scl_real_t> specificity(2);

    // NULL expression
    scl_error_t err = scl_comm_expression_specificity(
        nullptr, cell_type_labels.data(),
        0, 8, 2, specificity.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL specificity
    err = scl_comm_expression_specificity(
        expr, cell_type_labels.data(),
        0, 8, 2, nullptr
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// NATMI Edge Weight Tests
// =============================================================================

SCL_TEST_SUITE(natmi_edge_weight)

SCL_TEST_CASE(natmi_edge_weight_basic) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels = {0, 0, 0, 0, 1, 1, 1, 1};

    std::vector<scl_real_t> edge_weights(4);  // 2x2

    scl_error_t err = scl_comm_natmi_edge_weight(
        expr, cell_type_labels.data(),
        0, 1, 8, 2, edge_weights.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Edge weights should be non-negative
    for (scl_real_t w : edge_weights) {
        SCL_ASSERT_GE(w, 0.0);
        SCL_ASSERT_TRUE(std::isfinite(w));
    }
}

SCL_TEST_CASE(natmi_edge_weight_null_inputs) {
    auto [indptr, indices, data] = small_expression_5x8();
    Sparse expr = make_sparse_csr(5, 8, 12, indptr.data(), indices.data(), data.data());

    std::vector<scl_index_t> cell_type_labels(8);
    std::vector<scl_real_t> edge_weights(4);

    // NULL expression
    scl_error_t err = scl_comm_natmi_edge_weight(
        nullptr, cell_type_labels.data(),
        0, 1, 8, 2, edge_weights.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL cell_type_labels
    err = scl_comm_natmi_edge_weight(
        expr, nullptr,
        0, 1, 8, 2, edge_weights.data()
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);

    // NULL edge_weights
    err = scl_comm_natmi_edge_weight(
        expr, cell_type_labels.data(),
        0, 1, 8, 2, nullptr
    );
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
