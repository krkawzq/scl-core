// =============================================================================
// SCL Core - Comprehensive annotation.h Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/annotation.h
//
// Functions tested (14 total):
//   ✓ scl_annotation_reference_mapping
//   ✓ scl_annotation_correlation_assignment
//   ✓ scl_annotation_build_reference_profiles
//   ✓ scl_annotation_marker_gene_score
//   ✓ scl_annotation_assign_from_marker_scores
//   ✓ scl_annotation_consensus_annotation
//   ✓ scl_annotation_detect_novel_types
//   ✓ scl_annotation_detect_novel_by_distance
//   ✓ scl_annotation_label_propagation
//   ✓ scl_annotation_quality_metrics
//   ✓ scl_annotation_confusion_matrix
//   ✓ scl_annotation_entropy
//
// =============================================================================

#include "test.hpp"

extern "C" {
#include "scl/binding/c_api/annotation.h"
}

using namespace scl::test;

// Helper: Convert Eigen CSR to Sparse guard
static Sparse eigen_to_sparse(const EigenCSR& eigen_mat) {
    auto arrays = from_eigen_csr(eigen_mat);
    return make_sparse_csr(
        arrays.rows, arrays.cols, arrays.nnz,
        arrays.indptr.data(), arrays.indices.data(), arrays.data.data()
    );
}

// Helper: Create small expression dataset
static Sparse make_expression_data(scl_index_t n_cells, scl_index_t n_genes, Random& rng) {
    return eigen_to_sparse(random_sparse_csr(n_cells, n_genes, 0.2, rng));
}

SCL_TEST_BEGIN

// =============================================================================
// Reference Mapping (KNN-based)
// =============================================================================

SCL_TEST_SUITE(reference_mapping)

SCL_TEST_CASE(reference_mapping_basic) {
    Random rng(42);

    scl_index_t n_query = 30, n_ref = 50, n_genes = 100, n_types = 5;

    auto query_expr = make_expression_data(n_query, n_genes, rng);
    auto ref_expr = make_expression_data(n_ref, n_genes, rng);
    auto query_to_ref = random_sparse_csr(n_query, n_ref, 0.1, rng);

    std::vector<scl_index_t> ref_labels(n_ref);
    for (scl_index_t i = 0; i < n_ref; ++i) {
        ref_labels[i] = i % n_types;
    }

    std::vector<scl_index_t> query_labels(n_query);
    std::vector<scl_real_t> confidence(n_query);

    scl_error_t err = scl_annotation_reference_mapping(
        query_expr, ref_expr, ref_labels.data(), n_ref,
        query_to_ref, n_query, n_types,
        query_labels.data(), confidence.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify labels and confidence
    for (scl_index_t i = 0; i < n_query; ++i) {
        SCL_ASSERT_GE(query_labels[i], 0);
        SCL_ASSERT_LT(query_labels[i], n_types);
        SCL_ASSERT_GE(confidence[i], 0.0);
        SCL_ASSERT_LE(confidence[i], 1.0);
        SCL_ASSERT_TRUE(std::isfinite(confidence[i]));
    }
}

SCL_TEST_CASE(reference_mapping_null_checks) {
    Random rng(42);
    auto query = make_expression_data(10, 20, rng);
    auto ref = make_expression_data(20, 20, rng);
    auto neighbors = random_sparse_csr(10, 20, 0.1, rng);

    std::vector<scl_index_t> ref_labels(20, 0);
    std::vector<scl_index_t> query_labels(10);
    std::vector<scl_real_t> confidence(10);

    SCL_ASSERT_EQ(
        scl_annotation_reference_mapping(nullptr, ref, ref_labels.data(), 20, neighbors, 10, 3, query_labels.data(), confidence.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_reference_mapping(query, ref, nullptr, 20, neighbors, 10, 3, query_labels.data(), confidence.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_reference_mapping(query, ref, ref_labels.data(), 20, neighbors, 10, 3, nullptr, confidence.data()),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Correlation-Based Assignment
// =============================================================================

SCL_TEST_SUITE(correlation_assignment)

SCL_TEST_CASE(correlation_assignment_basic) {
    Random rng(123);

    scl_index_t n_query = 40, n_types = 6, n_genes = 80;

    auto query_expr = make_expression_data(n_query, n_genes, rng);
    auto ref_profiles = make_expression_data(n_types, n_genes, rng);

    std::vector<scl_index_t> assigned_labels(n_query);
    std::vector<scl_real_t> correlation_scores(n_query);
    std::vector<scl_real_t> all_correlations(n_query * n_types);

    scl_error_t err = scl_annotation_correlation_assignment(
        query_expr, ref_profiles, n_query, n_types, n_genes,
        assigned_labels.data(), correlation_scores.data(),
        all_correlations.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify labels and scores
    for (scl_index_t i = 0; i < n_query; ++i) {
        SCL_ASSERT_GE(assigned_labels[i], 0);
        SCL_ASSERT_LT(assigned_labels[i], n_types);
        SCL_ASSERT_TRUE(std::isfinite(correlation_scores[i]));
    }

    // Verify all correlations are finite
    for (scl_size_t i = 0; i < n_query * n_types; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(all_correlations[i]));
    }
}

SCL_TEST_CASE(correlation_assignment_without_all_correlations) {
    Random rng(456);

    scl_index_t n_query = 20, n_types = 4, n_genes = 50;

    auto query_expr = make_expression_data(n_query, n_genes, rng);
    auto ref_profiles = make_expression_data(n_types, n_genes, rng);

    std::vector<scl_index_t> assigned_labels(n_query);
    std::vector<scl_real_t> correlation_scores(n_query);

    // Pass nullptr for all_correlations (optional)
    scl_error_t err = scl_annotation_correlation_assignment(
        query_expr, ref_profiles, n_query, n_types, n_genes,
        assigned_labels.data(), correlation_scores.data(),
        nullptr  // Don't compute all correlations
    );

    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(correlation_assignment_null_checks) {
    Random rng(42);
    auto query = make_expression_data(10, 20, rng);
    auto profiles = make_expression_data(3, 20, rng);

    std::vector<scl_index_t> labels(10);
    std::vector<scl_real_t> scores(10);

    SCL_ASSERT_EQ(
        scl_annotation_correlation_assignment(nullptr, profiles, 10, 3, 20, labels.data(), scores.data(), nullptr),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_correlation_assignment(query, nullptr, 10, 3, 20, labels.data(), scores.data(), nullptr),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_correlation_assignment(query, profiles, 10, 3, 20, nullptr, scores.data(), nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Build Reference Profiles
// =============================================================================

SCL_TEST_SUITE(reference_profiles)

SCL_TEST_CASE(build_reference_profiles_basic) {
    Random rng(789);

    scl_index_t n_cells = 60, n_genes = 100, n_types = 5;

    auto expression = make_expression_data(n_cells, n_genes, rng);

    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i % n_types;
    }

    std::vector<scl_real_t> profiles(n_types * n_genes);

    scl_error_t err = scl_annotation_build_reference_profiles(
        expression, labels.data(), n_cells, n_genes, n_types,
        profiles.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify all profiles are finite and non-negative
    for (scl_size_t i = 0; i < n_types * n_genes; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(profiles[i]));
        SCL_ASSERT_GE(profiles[i], 0.0);
    }
}

SCL_TEST_CASE(build_reference_profiles_null_checks) {
    Random rng(42);
    auto expr = make_expression_data(20, 30, rng);
    std::vector<scl_index_t> labels(20, 0);
    std::vector<scl_real_t> profiles(90);

    SCL_ASSERT_EQ(
        scl_annotation_build_reference_profiles(nullptr, labels.data(), 20, 30, 3, profiles.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_build_reference_profiles(expr, nullptr, 20, 30, 3, profiles.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_build_reference_profiles(expr, labels.data(), 20, 30, 3, nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Marker Gene Scoring
// =============================================================================

SCL_TEST_SUITE(marker_gene_score)

SCL_TEST_CASE(marker_gene_score_basic) {
    Random rng(321);

    scl_index_t n_cells = 50, n_genes = 100, n_types = 4;

    auto expression = make_expression_data(n_cells, n_genes, rng);

    // Define marker genes for each type
    std::vector<scl_index_t> markers_type0 = {0, 1, 2, 3, 4};
    std::vector<scl_index_t> markers_type1 = {10, 11, 12, 13};
    std::vector<scl_index_t> markers_type2 = {20, 21, 22};
    std::vector<scl_index_t> markers_type3 = {30, 31, 32, 33, 34, 35};

    std::vector<const scl_index_t*> marker_genes = {
        markers_type0.data(),
        markers_type1.data(),
        markers_type2.data(),
        markers_type3.data()
    };

    std::vector<scl_index_t> marker_counts = {5, 4, 3, 6};

    std::vector<scl_real_t> scores(n_cells * n_types);

    scl_error_t err = scl_annotation_marker_gene_score(
        expression, marker_genes.data(), marker_counts.data(),
        n_cells, n_genes, n_types, scores.data(), SCL_TRUE
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify scores are finite
    for (scl_size_t i = 0; i < n_cells * n_types; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(scores[i]));
    }
}

SCL_TEST_CASE(marker_gene_score_without_normalization) {
    Random rng(654);

    scl_index_t n_cells = 30, n_genes = 60, n_types = 3;

    auto expression = make_expression_data(n_cells, n_genes, rng);

    std::vector<scl_index_t> markers0 = {0, 1, 2};
    std::vector<scl_index_t> markers1 = {10, 11};
    std::vector<scl_index_t> markers2 = {20, 21, 22, 23};

    std::vector<const scl_index_t*> marker_genes = {markers0.data(), markers1.data(), markers2.data()};
    std::vector<scl_index_t> marker_counts = {3, 2, 4};

    std::vector<scl_real_t> scores(n_cells * n_types);

    scl_error_t err = scl_annotation_marker_gene_score(
        expression, marker_genes.data(), marker_counts.data(),
        n_cells, n_genes, n_types, scores.data(), SCL_FALSE  // No normalization
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    for (scl_size_t i = 0; i < n_cells * n_types; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(scores[i]));
    }
}

SCL_TEST_CASE(assign_from_marker_scores_basic) {
    scl_index_t n_cells = 20, n_types = 4;

    // Create some marker scores
    std::vector<scl_real_t> scores(n_cells * n_types);
    Random rng(111);
    for (auto& s : scores) {
        s = rng.uniform(0.0, 1.0);
    }

    std::vector<scl_index_t> labels(n_cells);
    std::vector<scl_real_t> confidence(n_cells);

    scl_error_t err = scl_annotation_assign_from_marker_scores(
        scores.data(), n_cells, n_types, labels.data(), confidence.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify labels
    for (scl_index_t i = 0; i < n_cells; ++i) {
        SCL_ASSERT_GE(labels[i], 0);
        SCL_ASSERT_LT(labels[i], n_types);
        SCL_ASSERT_GE(confidence[i], 0.0);
        SCL_ASSERT_TRUE(std::isfinite(confidence[i]));
    }
}

SCL_TEST_CASE(marker_gene_score_null_checks) {
    Random rng(42);
    auto expr = make_expression_data(10, 20, rng);

    std::vector<scl_index_t> markers = {0, 1, 2};
    const scl_index_t* marker_ptrs[] = {markers.data()};
    scl_index_t counts[] = {3};
    std::vector<scl_real_t> scores(30);

    SCL_ASSERT_EQ(
        scl_annotation_marker_gene_score(nullptr, marker_ptrs, counts, 10, 20, 3, scores.data(), SCL_TRUE),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_marker_gene_score(expr, nullptr, counts, 10, 20, 3, scores.data(), SCL_TRUE),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_marker_gene_score(expr, marker_ptrs, nullptr, 10, 20, 3, scores.data(), SCL_TRUE),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_marker_gene_score(expr, marker_ptrs, counts, 10, 20, 3, nullptr, SCL_TRUE),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Consensus Annotation
// =============================================================================

SCL_TEST_SUITE(consensus_annotation)

SCL_TEST_CASE(consensus_annotation_basic) {
    scl_index_t n_methods = 3, n_cells = 40, n_types = 5;

    // Create predictions from 3 methods
    std::vector<scl_index_t> pred1(n_cells), pred2(n_cells), pred3(n_cells);
    std::vector<scl_real_t> conf1(n_cells), conf2(n_cells), conf3(n_cells);

    Random rng(987);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        pred1[i] = i % n_types;
        pred2[i] = (i + 1) % n_types;
        pred3[i] = (i + 2) % n_types;

        conf1[i] = rng.uniform(0.5, 1.0);
        conf2[i] = rng.uniform(0.5, 1.0);
        conf3[i] = rng.uniform(0.5, 1.0);
    }

    const scl_index_t* predictions[] = {pred1.data(), pred2.data(), pred3.data()};
    const scl_real_t* confidences[] = {conf1.data(), conf2.data(), conf3.data()};

    std::vector<scl_index_t> consensus_labels(n_cells);
    std::vector<scl_real_t> consensus_confidence(n_cells);

    scl_error_t err = scl_annotation_consensus_annotation(
        predictions, confidences, n_methods, n_cells, n_types,
        consensus_labels.data(), consensus_confidence.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify consensus results
    for (scl_index_t i = 0; i < n_cells; ++i) {
        SCL_ASSERT_GE(consensus_labels[i], 0);
        SCL_ASSERT_LT(consensus_labels[i], n_types);
        SCL_ASSERT_GE(consensus_confidence[i], 0.0);
        SCL_ASSERT_LE(consensus_confidence[i], 1.0);
        SCL_ASSERT_TRUE(std::isfinite(consensus_confidence[i]));
    }
}

SCL_TEST_CASE(consensus_annotation_without_confidences) {
    scl_index_t n_methods = 2, n_cells = 30, n_types = 4;

    std::vector<scl_index_t> pred1(n_cells), pred2(n_cells);
    Random rng(222);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        pred1[i] = i % n_types;
        pred2[i] = (i + 1) % n_types;
    }

    const scl_index_t* predictions[] = {pred1.data(), pred2.data()};

    std::vector<scl_index_t> consensus_labels(n_cells);
    std::vector<scl_real_t> consensus_confidence(n_cells);

    // Pass nullptr for confidences (optional)
    scl_error_t err = scl_annotation_consensus_annotation(
        predictions, nullptr, n_methods, n_cells, n_types,
        consensus_labels.data(), consensus_confidence.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(consensus_annotation_null_checks) {
    scl_index_t pred[] = {0, 1, 2};
    const scl_index_t* preds[] = {pred};
    std::vector<scl_index_t> labels(3);
    std::vector<scl_real_t> conf(3);

    SCL_ASSERT_EQ(
        scl_annotation_consensus_annotation(nullptr, nullptr, 1, 3, 3, labels.data(), conf.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_consensus_annotation(preds, nullptr, 1, 3, 3, nullptr, conf.data()),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Detect Novel Types
// =============================================================================

SCL_TEST_SUITE(novel_detection)

SCL_TEST_CASE(detect_novel_types_basic) {
    Random rng(555);

    scl_index_t n_query = 50;
    auto query_expr = make_expression_data(n_query, 80, rng);

    std::vector<scl_real_t> confidence_scores(n_query);
    for (scl_index_t i = 0; i < n_query; ++i) {
        confidence_scores[i] = rng.uniform(0.0, 1.0);
    }

    std::vector<scl_bool_t> is_novel(n_query);

    scl_error_t err = scl_annotation_detect_novel_types(
        query_expr, confidence_scores.data(), n_query, 0.5,
        is_novel.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify flags are boolean
    scl_index_t novel_count = 0;
    for (scl_index_t i = 0; i < n_query; ++i) {
        SCL_ASSERT_TRUE(is_novel[i] == SCL_TRUE || is_novel[i] == SCL_FALSE);
        if (confidence_scores[i] < 0.5) {
            SCL_ASSERT_EQ(is_novel[i], SCL_TRUE);
            novel_count++;
        }
    }
}

SCL_TEST_CASE(detect_novel_by_distance_basic) {
    Random rng(666);

    scl_index_t n_query = 40, n_types = 5, n_genes = 60;

    auto query_expr = make_expression_data(n_query, n_genes, rng);

    std::vector<scl_real_t> ref_profiles(n_types * n_genes);
    for (auto& v : ref_profiles) {
        v = rng.uniform(0.0, 5.0);
    }

    std::vector<scl_index_t> assigned_labels(n_query);
    for (scl_index_t i = 0; i < n_query; ++i) {
        assigned_labels[i] = i % n_types;
    }

    std::vector<scl_bool_t> is_novel(n_query);
    std::vector<scl_real_t> distances(n_query);

    scl_error_t err = scl_annotation_detect_novel_by_distance(
        query_expr, ref_profiles.data(), assigned_labels.data(),
        n_query, n_types, n_genes, 10.0,
        is_novel.data(), distances.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify results
    for (scl_index_t i = 0; i < n_query; ++i) {
        SCL_ASSERT_TRUE(is_novel[i] == SCL_TRUE || is_novel[i] == SCL_FALSE);
        SCL_ASSERT_GE(distances[i], 0.0);
        SCL_ASSERT_TRUE(std::isfinite(distances[i]));
    }
}

SCL_TEST_CASE(detect_novel_null_checks) {
    Random rng(42);
    auto query = make_expression_data(10, 20, rng);
    std::vector<scl_real_t> conf(10, 0.5);
    std::vector<scl_bool_t> is_novel(10);

    SCL_ASSERT_EQ(
        scl_annotation_detect_novel_types(nullptr, conf.data(), 10, 0.5, is_novel.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_detect_novel_types(query, nullptr, 10, 0.5, is_novel.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_detect_novel_types(query, conf.data(), 10, 0.5, nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Label Propagation
// =============================================================================

SCL_TEST_SUITE(label_propagation)

SCL_TEST_CASE(label_propagation_basic) {
    Random rng(777);

    scl_index_t n_cells = 60, n_types = 6;

    auto neighbor_graph = random_sparse_csr(n_cells, n_cells, 0.15, rng);

    // Initial labels: some labeled, some unlabeled (-1)
    std::vector<scl_index_t> initial_labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        initial_labels[i] = (i < 30) ? (i % n_types) : -1;  // First 30 labeled
    }

    std::vector<scl_index_t> final_labels(n_cells);
    std::vector<scl_real_t> label_confidence(n_cells);

    scl_error_t err = scl_annotation_label_propagation(
        neighbor_graph, initial_labels.data(), n_cells, n_types, 100,
        final_labels.data(), label_confidence.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify all cells now have labels
    for (scl_index_t i = 0; i < n_cells; ++i) {
        SCL_ASSERT_GE(final_labels[i], 0);
        SCL_ASSERT_LT(final_labels[i], n_types);
        SCL_ASSERT_GE(label_confidence[i], 0.0);
        SCL_ASSERT_LE(label_confidence[i], 1.0);
        SCL_ASSERT_TRUE(std::isfinite(label_confidence[i]));
    }

    // Initially labeled cells should keep their labels
    for (scl_index_t i = 0; i < 30; ++i) {
        SCL_ASSERT_EQ(final_labels[i], initial_labels[i]);
    }
}

SCL_TEST_CASE(label_propagation_null_checks) {
    Random rng(42);
    auto graph = random_sparse_csr(20, 20, 0.1, rng);
    std::vector<scl_index_t> initial(20, 0);
    std::vector<scl_index_t> final(20);
    std::vector<scl_real_t> conf(20);

    SCL_ASSERT_EQ(
        scl_annotation_label_propagation(nullptr, initial.data(), 20, 3, 50, final.data(), conf.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_label_propagation(graph, nullptr, 20, 3, 50, final.data(), conf.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_label_propagation(graph, initial.data(), 20, 3, 50, nullptr, conf.data()),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Quality Metrics
// =============================================================================

SCL_TEST_SUITE(quality_metrics)

SCL_TEST_CASE(quality_metrics_basic) {
    scl_index_t n_cells = 100, n_types = 5;

    std::vector<scl_index_t> predicted(n_cells);
    std::vector<scl_index_t> true_labels(n_cells);

    Random rng(888);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        true_labels[i] = i % n_types;
        // 80% accuracy
        predicted[i] = (rng.uniform(0.0, 1.0) < 0.8) ? true_labels[i] : ((true_labels[i] + 1) % n_types);
    }

    scl_real_t accuracy = 0.0, macro_f1 = 0.0;
    std::vector<scl_real_t> per_class_f1(n_types);

    scl_error_t err = scl_annotation_quality_metrics(
        predicted.data(), true_labels.data(), n_cells, n_types,
        &accuracy, &macro_f1, per_class_f1.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify metrics
    SCL_ASSERT_GE(accuracy, 0.0);
    SCL_ASSERT_LE(accuracy, 1.0);
    SCL_ASSERT_GE(macro_f1, 0.0);
    SCL_ASSERT_LE(macro_f1, 1.0);
    SCL_ASSERT_TRUE(std::isfinite(accuracy));
    SCL_ASSERT_TRUE(std::isfinite(macro_f1));

    for (scl_index_t i = 0; i < n_types; ++i) {
        SCL_ASSERT_GE(per_class_f1[i], 0.0);
        SCL_ASSERT_LE(per_class_f1[i], 1.0);
        SCL_ASSERT_TRUE(std::isfinite(per_class_f1[i]));
    }
}

SCL_TEST_CASE(confusion_matrix_basic) {
    scl_index_t n_cells = 50, n_types = 4;

    std::vector<scl_index_t> predicted(n_cells);
    std::vector<scl_index_t> true_labels(n_cells);

    for (scl_index_t i = 0; i < n_cells; ++i) {
        true_labels[i] = i % n_types;
        predicted[i] = (i / 10) % n_types;  // Some pattern
    }

    std::vector<scl_index_t> confusion(n_types * n_types);

    scl_error_t err = scl_annotation_confusion_matrix(
        predicted.data(), true_labels.data(), n_cells, n_types,
        confusion.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify confusion matrix sums to n_cells
    scl_index_t total = 0;
    for (scl_size_t i = 0; i < n_types * n_types; ++i) {
        SCL_ASSERT_GE(confusion[i], 0);
        total += confusion[i];
    }
    SCL_ASSERT_EQ(total, n_cells);
}

SCL_TEST_CASE(entropy_basic) {
    scl_index_t n_cells = 40, n_types = 5;

    std::vector<scl_real_t> type_probabilities(n_cells * n_types);

    Random rng(999);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        // Generate random probabilities that sum to 1
        scl_real_t sum = 0.0;
        for (scl_index_t t = 0; t < n_types; ++t) {
            type_probabilities[i * n_types + t] = rng.uniform(0.0, 1.0);
            sum += type_probabilities[i * n_types + t];
        }
        // Normalize
        for (scl_index_t t = 0; t < n_types; ++t) {
            type_probabilities[i * n_types + t] /= sum;
        }
    }

    std::vector<scl_real_t> entropy(n_cells);

    scl_error_t err = scl_annotation_entropy(
        type_probabilities.data(), n_cells, n_types, entropy.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify entropy values are non-negative and finite
    for (scl_index_t i = 0; i < n_cells; ++i) {
        SCL_ASSERT_GE(entropy[i], 0.0);
        SCL_ASSERT_TRUE(std::isfinite(entropy[i]));
    }
}

SCL_TEST_CASE(quality_metrics_null_checks) {
    std::vector<scl_index_t> pred(10, 0);
    std::vector<scl_index_t> true_val(10, 0);
    scl_real_t acc = 0.0, f1 = 0.0;

    SCL_ASSERT_EQ(
        scl_annotation_quality_metrics(nullptr, true_val.data(), 10, 3, &acc, &f1, nullptr),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_quality_metrics(pred.data(), nullptr, 10, 3, &acc, &f1, nullptr),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_annotation_quality_metrics(pred.data(), true_val.data(), 10, 3, nullptr, &f1, nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Random Integration Tests
// =============================================================================

SCL_TEST_SUITE(random_tests)

SCL_TEST_RETRY(annotation_pipeline_random, 3) {
    Random rng(54321);

    auto [n_query, n_genes] = random_shape(30, 80, rng);
    auto [n_ref, _] = random_shape(40, 100, rng);

    scl_index_t n_types = 4;

    auto query = random_sparse_csr(n_query, n_genes, 0.2, rng);
    auto ref = random_sparse_csr(n_ref, n_genes, 0.2, rng);

    std::vector<scl_index_t> ref_labels(n_ref);
    for (scl_index_t i = 0; i < n_ref; ++i) {
        ref_labels[i] = i % n_types;
    }

    std::vector<scl_real_t> profiles(n_types * n_genes);

    scl_error_t err = scl_annotation_build_reference_profiles(
        ref, ref_labels.data(), n_ref, n_genes, n_types, profiles.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify profiles
    for (scl_size_t i = 0; i < n_types * n_genes; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(profiles[i]));
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
