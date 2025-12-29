// =============================================================================
// SCL Core - Comprehensive alignment.h Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/alignment.h
//
// Functions tested (11 total):
//   ✓ scl_alignment_mnn_pairs
//   ✓ scl_alignment_find_anchors
//   ✓ scl_alignment_transfer_labels
//   ✓ scl_alignment_integration_score
//   ✓ scl_alignment_batch_mixing
//   ✓ scl_alignment_kbet_score
//   ✓ scl_alignment_compute_correction_vectors
//   ✓ scl_alignment_smooth_correction_vectors
//   ✓ scl_alignment_cca_projection
//
// =============================================================================

#include "test.hpp"

extern "C" {
#include "scl/binding/c_api/alignment.h"
}

using namespace scl::test;

// Helper: Create small dataset for alignment tests
static Sparse make_small_dataset(scl_index_t n_cells, scl_index_t n_genes, Random& rng) {
    return random_sparse_csr(n_cells, n_genes, 0.3, rng);
}

SCL_TEST_BEGIN

// =============================================================================
// Mutual Nearest Neighbors (MNN)
// =============================================================================

SCL_TEST_SUITE(mnn_pairs)

SCL_TEST_CASE(mnn_pairs_basic) {
    Random rng(42);

    auto data1 = make_small_dataset(20, 50, rng);
    auto data2 = make_small_dataset(25, 50, rng);

    std::vector<scl_index_t> mnn_cell1(100);  // Pre-allocate
    std::vector<scl_index_t> mnn_cell2(100);
    scl_size_t n_pairs = 0;

    scl_error_t err = scl_alignment_mnn_pairs(
        data1, data2, 5,
        mnn_cell1.data(), mnn_cell2.data(), &n_pairs
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Should find some MNN pairs (but not too many)
    SCL_ASSERT_GT(n_pairs, 0);
    SCL_ASSERT_LE(n_pairs, 100);

    // Verify indices are in valid range
    for (scl_size_t i = 0; i < n_pairs; ++i) {
        SCL_ASSERT_GE(mnn_cell1[i], 0);
        SCL_ASSERT_LT(mnn_cell1[i], 20);
        SCL_ASSERT_GE(mnn_cell2[i], 0);
        SCL_ASSERT_LT(mnn_cell2[i], 25);
    }
}

SCL_TEST_CASE(mnn_pairs_null_checks) {
    Random rng(42);
    auto data1 = make_small_dataset(10, 20, rng);
    auto data2 = make_small_dataset(10, 20, rng);

    std::vector<scl_index_t> mnn1(50), mnn2(50);
    scl_size_t n_pairs = 0;

    SCL_ASSERT_EQ(
        scl_alignment_mnn_pairs(nullptr, data2, 5, mnn1.data(), mnn2.data(), &n_pairs),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_mnn_pairs(data1, nullptr, 5, mnn1.data(), mnn2.data(), &n_pairs),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_mnn_pairs(data1, data2, 5, nullptr, mnn2.data(), &n_pairs),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_mnn_pairs(data1, data2, 5, mnn1.data(), nullptr, &n_pairs),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_mnn_pairs(data1, data2, 5, mnn1.data(), mnn2.data(), nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(mnn_pairs_invalid_k) {
    Random rng(42);
    auto data1 = make_small_dataset(10, 20, rng);
    auto data2 = make_small_dataset(10, 20, rng);

    std::vector<scl_index_t> mnn1(50), mnn2(50);
    scl_size_t n_pairs = 0;

    // k = 0 should fail
    scl_error_t err = scl_alignment_mnn_pairs(
        data1, data2, 0, mnn1.data(), mnn2.data(), &n_pairs
    );
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Anchor Finding
// =============================================================================

SCL_TEST_SUITE(anchors)

SCL_TEST_CASE(find_anchors_basic) {
    Random rng(123);

    auto data1 = make_small_dataset(30, 50, rng);
    auto data2 = make_small_dataset(30, 50, rng);

    std::vector<scl_index_t> anchor_cell1(100);
    std::vector<scl_index_t> anchor_cell2(100);
    std::vector<scl_real_t> anchor_scores(100);
    scl_size_t n_anchors = 0;

    scl_error_t err = scl_alignment_find_anchors(
        data1, data2, 10,
        anchor_cell1.data(), anchor_cell2.data(),
        anchor_scores.data(), &n_anchors
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Should find some anchors
    SCL_ASSERT_GT(n_anchors, 0);
    SCL_ASSERT_LE(n_anchors, 100);

    // Verify indices and scores
    for (scl_size_t i = 0; i < n_anchors; ++i) {
        SCL_ASSERT_GE(anchor_cell1[i], 0);
        SCL_ASSERT_LT(anchor_cell1[i], 30);
        SCL_ASSERT_GE(anchor_cell2[i], 0);
        SCL_ASSERT_LT(anchor_cell2[i], 30);
        SCL_ASSERT_GE(anchor_scores[i], 0.0);
        SCL_ASSERT_TRUE(std::isfinite(anchor_scores[i]));
    }
}

SCL_TEST_CASE(find_anchors_null_checks) {
    Random rng(123);
    auto data1 = make_small_dataset(10, 20, rng);
    auto data2 = make_small_dataset(10, 20, rng);

    std::vector<scl_index_t> anchor1(50), anchor2(50);
    std::vector<scl_real_t> scores(50);
    scl_size_t n_anchors = 0;

    SCL_ASSERT_EQ(
        scl_alignment_find_anchors(nullptr, data2, 5, anchor1.data(), anchor2.data(), scores.data(), &n_anchors),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_find_anchors(data1, data2, 5, nullptr, anchor2.data(), scores.data(), &n_anchors),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_find_anchors(data1, data2, 5, anchor1.data(), anchor2.data(), scores.data(), nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Label Transfer
// =============================================================================

SCL_TEST_SUITE(label_transfer)

SCL_TEST_CASE(transfer_labels_basic) {
    // Setup: 10 source cells, 15 target cells, 5 anchors
    std::vector<scl_index_t> anchor_cell1 = {0, 2, 4, 6, 8};
    std::vector<scl_index_t> anchor_cell2 = {1, 3, 5, 7, 9};
    std::vector<scl_real_t> anchor_weights = {1.0, 0.9, 0.8, 0.85, 0.95};

    std::vector<scl_index_t> source_labels = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
    std::vector<scl_index_t> target_labels(15);
    std::vector<scl_real_t> transfer_confidence(15);

    scl_error_t err = scl_alignment_transfer_labels(
        anchor_cell1.data(), anchor_cell2.data(), anchor_weights.data(), 5,
        source_labels.data(), 10, 15,
        target_labels.data(), transfer_confidence.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify labels are in valid range
    for (scl_size_t i = 0; i < 15; ++i) {
        SCL_ASSERT_GE(target_labels[i], 0);
        SCL_ASSERT_GE(transfer_confidence[i], 0.0);
        SCL_ASSERT_LE(transfer_confidence[i], 1.0);
        SCL_ASSERT_TRUE(std::isfinite(transfer_confidence[i]));
    }
}

SCL_TEST_CASE(transfer_labels_null_checks) {
    std::vector<scl_index_t> anchor1 = {0, 1};
    std::vector<scl_index_t> anchor2 = {0, 1};
    std::vector<scl_real_t> weights = {1.0, 1.0};
    std::vector<scl_index_t> source = {0, 1};
    std::vector<scl_index_t> target(5);
    std::vector<scl_real_t> confidence(5);

    SCL_ASSERT_EQ(
        scl_alignment_transfer_labels(nullptr, anchor2.data(), weights.data(), 2, source.data(), 2, 5, target.data(), confidence.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_transfer_labels(anchor1.data(), anchor2.data(), weights.data(), 2, nullptr, 2, 5, target.data(), confidence.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_transfer_labels(anchor1.data(), anchor2.data(), weights.data(), 2, source.data(), 2, 5, nullptr, confidence.data()),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Integration Quality
// =============================================================================

SCL_TEST_SUITE(integration_quality)

SCL_TEST_CASE(integration_score_basic) {
    Random rng(42);

    auto integrated_data = make_small_dataset(50, 100, rng);
    auto neighbors = random_sparse_csr(50, 50, 0.2, rng);

    std::vector<scl_index_t> batch_labels(50);
    for (scl_index_t i = 0; i < 50; ++i) {
        batch_labels[i] = i % 3;  // 3 batches
    }

    scl_real_t score = 0.0;

    scl_error_t err = scl_alignment_integration_score(
        integrated_data, batch_labels.data(), 50, neighbors, &score
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Score should be between 0 and 1
    SCL_ASSERT_GE(score, 0.0);
    SCL_ASSERT_LE(score, 1.0);
    SCL_ASSERT_TRUE(std::isfinite(score));
}

SCL_TEST_CASE(batch_mixing_basic) {
    Random rng(123);

    auto neighbors = random_sparse_csr(40, 40, 0.25, rng);

    std::vector<scl_index_t> batch_labels(40);
    for (scl_index_t i = 0; i < 40; ++i) {
        batch_labels[i] = i % 2;  // 2 batches
    }

    std::vector<scl_real_t> mixing_scores(40);

    scl_error_t err = scl_alignment_batch_mixing(
        batch_labels.data(), 40, neighbors, mixing_scores.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // All scores should be valid
    for (scl_index_t i = 0; i < 40; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(mixing_scores[i]));
    }
}

SCL_TEST_CASE(kbet_score_basic) {
    Random rng(456);

    auto neighbors = random_sparse_csr(30, 30, 0.3, rng);

    std::vector<scl_index_t> batch_labels(30);
    for (scl_index_t i = 0; i < 30; ++i) {
        batch_labels[i] = i % 4;  // 4 batches
    }

    scl_real_t score = 0.0;

    scl_error_t err = scl_alignment_kbet_score(
        neighbors, batch_labels.data(), 30, &score
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // kBET score should be between 0 and 1
    SCL_ASSERT_GE(score, 0.0);
    SCL_ASSERT_LE(score, 1.0);
    SCL_ASSERT_TRUE(std::isfinite(score));
}

SCL_TEST_CASE(integration_quality_null_checks) {
    Random rng(42);
    auto data = make_small_dataset(20, 30, rng);
    auto neighbors = random_sparse_csr(20, 20, 0.2, rng);
    std::vector<scl_index_t> labels(20, 0);
    scl_real_t score = 0.0;

    SCL_ASSERT_EQ(
        scl_alignment_integration_score(nullptr, labels.data(), 20, neighbors, &score),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_integration_score(data, nullptr, 20, neighbors, &score),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_integration_score(data, labels.data(), 20, nullptr, &score),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_integration_score(data, labels.data(), 20, neighbors, nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Correction Vectors
// =============================================================================

SCL_TEST_SUITE(correction_vectors)

SCL_TEST_CASE(compute_correction_vectors_basic) {
    Random rng(789);

    scl_index_t n1 = 20, n2 = 25, n_features = 50;
    auto data1 = make_small_dataset(n1, n_features, rng);
    auto data2 = make_small_dataset(n2, n_features, rng);

    // Some MNN pairs
    std::vector<scl_index_t> mnn_cell1 = {0, 5, 10, 15};
    std::vector<scl_index_t> mnn_cell2 = {2, 7, 12, 17};
    scl_size_t n_pairs = 4;

    std::vector<scl_real_t> correction_vectors(n2 * n_features, 0.0);

    scl_error_t err = scl_alignment_compute_correction_vectors(
        data1, data2,
        mnn_cell1.data(), mnn_cell2.data(), n_pairs,
        correction_vectors.data(), n_features
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify all corrections are finite
    for (scl_size_t i = 0; i < n2 * n_features; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(correction_vectors[i]));
    }
}

SCL_TEST_CASE(smooth_correction_vectors_basic) {
    Random rng(321);

    scl_index_t n2 = 20, n_features = 30;
    auto data2 = make_small_dataset(n2, n_features, rng);

    std::vector<scl_real_t> correction_vectors(n2 * n_features);
    for (auto& v : correction_vectors) {
        v = rng.uniform(-1.0, 1.0);
    }

    scl_error_t err = scl_alignment_smooth_correction_vectors(
        data2, correction_vectors.data(), n_features, 1.0
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify all corrections are finite after smoothing
    for (scl_size_t i = 0; i < n2 * n_features; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(correction_vectors[i]));
    }
}

SCL_TEST_CASE(correction_vectors_null_checks) {
    Random rng(42);
    auto data1 = make_small_dataset(10, 20, rng);
    auto data2 = make_small_dataset(10, 20, rng);

    std::vector<scl_index_t> mnn1 = {0, 1};
    std::vector<scl_index_t> mnn2 = {0, 1};
    std::vector<scl_real_t> corrections(200);

    SCL_ASSERT_EQ(
        scl_alignment_compute_correction_vectors(nullptr, data2, mnn1.data(), mnn2.data(), 2, corrections.data(), 20),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_compute_correction_vectors(data1, data2, nullptr, mnn2.data(), 2, corrections.data(), 20),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_smooth_correction_vectors(nullptr, corrections.data(), 20, 1.0),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_smooth_correction_vectors(data2, nullptr, 20, 1.0),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// CCA Projection
// =============================================================================

SCL_TEST_SUITE(cca_projection)

SCL_TEST_CASE(cca_projection_basic) {
    Random rng(999);

    scl_index_t n1 = 30, n2 = 35, n_genes = 50, n_components = 10;
    auto data1 = make_small_dataset(n1, n_genes, rng);
    auto data2 = make_small_dataset(n2, n_genes, rng);

    std::vector<scl_real_t> projection1(n1 * n_components);
    std::vector<scl_real_t> projection2(n2 * n_components);

    scl_error_t err = scl_alignment_cca_projection(
        data1, data2, n_components,
        projection1.data(), projection2.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify projections are finite
    for (scl_size_t i = 0; i < n1 * n_components; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(projection1[i]));
    }

    for (scl_size_t i = 0; i < n2 * n_components; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(projection2[i]));
    }
}

SCL_TEST_CASE(cca_projection_null_checks) {
    Random rng(42);
    auto data1 = make_small_dataset(10, 20, rng);
    auto data2 = make_small_dataset(10, 20, rng);

    std::vector<scl_real_t> proj1(50), proj2(50);

    SCL_ASSERT_EQ(
        scl_alignment_cca_projection(nullptr, data2, 5, proj1.data(), proj2.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_cca_projection(data1, nullptr, 5, proj1.data(), proj2.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_cca_projection(data1, data2, 5, nullptr, proj2.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_alignment_cca_projection(data1, data2, 5, proj1.data(), nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(cca_projection_invalid_components) {
    Random rng(42);
    auto data1 = make_small_dataset(10, 20, rng);
    auto data2 = make_small_dataset(10, 20, rng);

    std::vector<scl_real_t> proj1(1), proj2(1);

    // n_components = 0 should fail
    scl_error_t err = scl_alignment_cca_projection(
        data1, data2, 0, proj1.data(), proj2.data()
    );

    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Random Integration Tests
// =============================================================================

SCL_TEST_SUITE(random_tests)

SCL_TEST_RETRY(mnn_random_datasets, 3) {
    Random rng(12345);

    auto [n1, n_genes1] = random_shape(20, 50, rng);
    auto [n2, n_genes2] = random_shape(20, 50, rng);

    // Make sure they have same number of genes
    scl_index_t n_genes = std::min(n_genes1, n_genes2);

    auto data1 = random_sparse_csr(n1, n_genes, 0.15, rng);
    auto data2 = random_sparse_csr(n2, n_genes, 0.15, rng);

    std::vector<scl_index_t> mnn1(200), mnn2(200);
    scl_size_t n_pairs = 0;

    scl_error_t err = scl_alignment_mnn_pairs(
        data1, data2, 5, mnn1.data(), mnn2.data(), &n_pairs
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_LE(n_pairs, 200);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
