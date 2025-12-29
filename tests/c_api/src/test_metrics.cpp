// =============================================================================
// SCL Core - Quality Metrics Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/metrics.h
//
// Functions tested:
//   - scl_metrics_silhouette_score
//   - scl_metrics_silhouette_samples
//   - scl_metrics_adjusted_rand_index
//   - scl_metrics_normalized_mutual_information
//   - scl_metrics_homogeneity_score
//   - scl_metrics_completeness_score
//   - scl_metrics_v_measure
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/metrics.h"

using namespace scl::test;
using precision::Tolerance;

// Helper: Generate symmetric distance matrix
static EigenCSR make_distance_matrix(scl_index_t n_cells, Random& rng) {
    EigenCSR mat(n_cells, n_cells);
    mat.reserve(Eigen::VectorXi::Constant(n_cells, n_cells));
    
    // Distance matrix: symmetric, zero diagonal
    for (scl_index_t i = 0; i < n_cells; ++i) {
        for (scl_index_t j = i; j < n_cells; ++j) {
            if (i == j) {
                mat.insert(i, j) = 0.0;
            } else {
                scl_real_t dist = rng.uniform(0.1, 10.0);
                mat.insert(i, j) = dist;
                mat.insert(j, i) = dist;  // Symmetric
            }
        }
    }
    
    mat.makeCompressed();
    return mat;
}

// Helper: Convert Eigen CSR to SCL Sparse
static Sparse eigen_to_scl_sparse(const EigenCSR& eigen_mat) {
    auto csr = from_eigen_csr(eigen_mat);
    return make_sparse_csr(
        csr.rows, csr.cols, csr.nnz,
        csr.indptr.data(), csr.indices.data(), csr.data.data()
    );
}

SCL_TEST_BEGIN

// =============================================================================
// Silhouette Score
// =============================================================================

SCL_TEST_SUITE(silhouette_score)

SCL_TEST_RETRY(silhouette_score_basic, 3)
{
    Random rng(42);
    
    scl_index_t n_cells = 20;
    auto dist_eigen = make_distance_matrix(n_cells, rng);
    Sparse dist = eigen_to_scl_sparse(dist_eigen);
    
    // Generate labels (3 clusters)
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i / (n_cells / 3);
    }
    
    scl_real_t score = 0.0;
    scl_error_t err = scl_metrics_silhouette_score(
        dist, labels.data(), n_cells, &score
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Silhouette score should be in [-1, 1] range
    SCL_ASSERT_GE(score, -1.0);
    SCL_ASSERT_LE(score, 1.0);
}

SCL_TEST_CASE(silhouette_score_perfect_clustering)
{
    Random rng(123);
    scl_index_t n_cells = 10;
    
    // Create two well-separated clusters
    EigenCSR dist_eigen(n_cells, n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        for (scl_index_t j = 0; j < n_cells; ++j) {
            if (i == j) {
                dist_eigen.insert(i, j) = 0.0;
            } else {
                scl_index_t cluster_i = i < 5 ? 0 : 1;
                scl_index_t cluster_j = j < 5 ? 0 : 1;
                
                if (cluster_i == cluster_j) {
                    dist_eigen.insert(i, j) = 0.1;  // Small within-cluster distance
                } else {
                    dist_eigen.insert(i, j) = 10.0;  // Large between-cluster distance
                }
            }
        }
    }
    dist_eigen.makeCompressed();
    Sparse dist = eigen_to_scl_sparse(dist_eigen);
    
    // Perfect clustering
    std::vector<scl_index_t> labels = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    
    scl_real_t score = 0.0;
    scl_error_t err = scl_metrics_silhouette_score(
        dist, labels.data(), n_cells, &score
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Well-separated clusters should have high silhouette score
    SCL_ASSERT_GT(score, 0.5);
}

SCL_TEST_CASE(silhouette_score_single_cluster)
{
    Random rng(456);
    scl_index_t n_cells = 10;
    auto dist_eigen = make_distance_matrix(n_cells, rng);
    Sparse dist = eigen_to_scl_sparse(dist_eigen);
    
    // All same cluster
    std::vector<scl_index_t> labels(n_cells, 0);
    
    scl_real_t score = 0.0;
    scl_error_t err = scl_metrics_silhouette_score(
        dist, labels.data(), n_cells, &score
    );
    
    // Single cluster may return 0 or error
    if (err == SCL_OK) {
        SCL_ASSERT_GE(score, -1.0);
        SCL_ASSERT_LE(score, 1.0);
    }
}

SCL_TEST_CASE(silhouette_score_null_distances)
{
    std::vector<scl_index_t> labels(10);
    
    scl_real_t score = 0.0;
    scl_error_t err = scl_metrics_silhouette_score(
        nullptr, labels.data(), 10, &score
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(silhouette_score_null_labels)
{
    Random rng(789);
    scl_index_t n_cells = 10;
    auto dist_eigen = make_distance_matrix(n_cells, rng);
    Sparse dist = eigen_to_scl_sparse(dist_eigen);
    
    scl_real_t score = 0.0;
    scl_error_t err = scl_metrics_silhouette_score(
        dist, nullptr, n_cells, &score
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(silhouette_score_null_output)
{
    Random rng(999);
    scl_index_t n_cells = 10;
    auto dist_eigen = make_distance_matrix(n_cells, rng);
    Sparse dist = eigen_to_scl_sparse(dist_eigen);
    
    std::vector<scl_index_t> labels(n_cells);
    
    scl_error_t err = scl_metrics_silhouette_score(
        dist, labels.data(), n_cells, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Silhouette Samples
// =============================================================================

SCL_TEST_SUITE(silhouette_samples)

SCL_TEST_RETRY(silhouette_samples_basic, 3)
{
    Random rng(111);
    
    scl_index_t n_cells = 20;
    auto dist_eigen = make_distance_matrix(n_cells, rng);
    Sparse dist = eigen_to_scl_sparse(dist_eigen);
    
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i / (n_cells / 3);
    }
    
    std::vector<scl_real_t> scores(n_cells);
    
    scl_error_t err = scl_metrics_silhouette_samples(
        dist, labels.data(), n_cells, scores.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All scores should be in [-1, 1] range
    for (scl_index_t i = 0; i < n_cells; ++i) {
        SCL_ASSERT_GE(scores[i], -1.0);
        SCL_ASSERT_LE(scores[i], 1.0);
    }
}

SCL_TEST_CASE(silhouette_samples_null_distances)
{
    std::vector<scl_index_t> labels(10);
    std::vector<scl_real_t> scores(10);
    
    scl_error_t err = scl_metrics_silhouette_samples(
        nullptr, labels.data(), 10, scores.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(silhouette_samples_null_labels)
{
    Random rng(222);
    scl_index_t n_cells = 10;
    auto dist_eigen = make_distance_matrix(n_cells, rng);
    Sparse dist = eigen_to_scl_sparse(dist_eigen);
    
    std::vector<scl_real_t> scores(n_cells);
    
    scl_error_t err = scl_metrics_silhouette_samples(
        dist, nullptr, n_cells, scores.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(silhouette_samples_null_output)
{
    Random rng(333);
    scl_index_t n_cells = 10;
    auto dist_eigen = make_distance_matrix(n_cells, rng);
    Sparse dist = eigen_to_scl_sparse(dist_eigen);
    
    std::vector<scl_index_t> labels(n_cells);
    
    scl_error_t err = scl_metrics_silhouette_samples(
        dist, labels.data(), n_cells, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Adjusted Rand Index (ARI)
// =============================================================================

SCL_TEST_SUITE(adjusted_rand_index)

SCL_TEST_CASE(ari_perfect_match)
{
    scl_index_t n_cells = 10;
    
    // Same labels
    std::vector<scl_index_t> labels1 = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
    std::vector<scl_index_t> labels2 = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
    
    scl_real_t ari = 0.0;
    scl_error_t err = scl_metrics_adjusted_rand_index(
        labels1.data(), labels2.data(), n_cells, &ari
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Perfect match should have ARI = 1.0
    SCL_ASSERT_NEAR(ari, 1.0, 1e-9);
}

SCL_TEST_CASE(ari_completely_different)
{
    scl_index_t n_cells = 10;
    
    // Completely different labels
    std::vector<scl_index_t> labels1 = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    std::vector<scl_index_t> labels2 = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    
    scl_real_t ari = 0.0;
    scl_error_t err = scl_metrics_adjusted_rand_index(
        labels1.data(), labels2.data(), n_cells, &ari
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // ARI should be in [-1, 1] range
    SCL_ASSERT_GE(ari, -1.0);
    SCL_ASSERT_LE(ari, 1.0);
}

SCL_TEST_RETRY(ari_random_labels, 3)
{
    Random rng(444);
    
    scl_index_t n_cells = rng.uniform_int(20, 50);
    scl_index_t n_clusters1 = rng.uniform_int(2, 8);
    scl_index_t n_clusters2 = rng.uniform_int(2, 8);
    
    std::vector<scl_index_t> labels1(n_cells);
    std::vector<scl_index_t> labels2(n_cells);
    
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels1[i] = rng.uniform_int(0, n_clusters1 - 1);
        labels2[i] = rng.uniform_int(0, n_clusters2 - 1);
    }
    
    scl_real_t ari = 0.0;
    scl_error_t err = scl_metrics_adjusted_rand_index(
        labels1.data(), labels2.data(), n_cells, &ari
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(ari, -1.0);
    SCL_ASSERT_LE(ari, 1.0);
}

SCL_TEST_CASE(ari_null_labels1)
{
    std::vector<scl_index_t> labels2(10);
    
    scl_real_t ari = 0.0;
    scl_error_t err = scl_metrics_adjusted_rand_index(
        nullptr, labels2.data(), 10, &ari
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(ari_null_labels2)
{
    std::vector<scl_index_t> labels1(10);
    
    scl_real_t ari = 0.0;
    scl_error_t err = scl_metrics_adjusted_rand_index(
        labels1.data(), nullptr, 10, &ari
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(ari_null_output)
{
    std::vector<scl_index_t> labels1 = {0, 0, 1, 1};
    std::vector<scl_index_t> labels2 = {0, 1, 0, 1};
    
    scl_error_t err = scl_metrics_adjusted_rand_index(
        labels1.data(), labels2.data(), 4, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Normalized Mutual Information (NMI)
// =============================================================================

SCL_TEST_SUITE(normalized_mutual_information)

SCL_TEST_CASE(nmi_perfect_match)
{
    scl_index_t n_cells = 10;
    
    // Same labels
    std::vector<scl_index_t> labels1 = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
    std::vector<scl_index_t> labels2 = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
    
    scl_real_t nmi = 0.0;
    scl_error_t err = scl_metrics_normalized_mutual_information(
        labels1.data(), labels2.data(), n_cells, &nmi
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Perfect match should have NMI = 1.0
    SCL_ASSERT_NEAR(nmi, 1.0, 1e-9);
}

SCL_TEST_CASE(nmi_completely_different)
{
    scl_index_t n_cells = 10;
    
    // Completely different labels
    std::vector<scl_index_t> labels1 = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    std::vector<scl_index_t> labels2 = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    
    scl_real_t nmi = 0.0;
    scl_error_t err = scl_metrics_normalized_mutual_information(
        labels1.data(), labels2.data(), n_cells, &nmi
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // NMI should be in [0, 1] range
    SCL_ASSERT_GE(nmi, 0.0);
    SCL_ASSERT_LE(nmi, 1.0);
}

SCL_TEST_RETRY(nmi_random_labels, 3)
{
    Random rng(555);
    
    scl_index_t n_cells = rng.uniform_int(20, 50);
    scl_index_t n_clusters1 = rng.uniform_int(2, 8);
    scl_index_t n_clusters2 = rng.uniform_int(2, 8);
    
    std::vector<scl_index_t> labels1(n_cells);
    std::vector<scl_index_t> labels2(n_cells);
    
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels1[i] = rng.uniform_int(0, n_clusters1 - 1);
        labels2[i] = rng.uniform_int(0, n_clusters2 - 1);
    }
    
    scl_real_t nmi = 0.0;
    scl_error_t err = scl_metrics_normalized_mutual_information(
        labels1.data(), labels2.data(), n_cells, &nmi
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(nmi, 0.0);
    SCL_ASSERT_LE(nmi, 1.0);
}

SCL_TEST_CASE(nmi_null_labels1)
{
    std::vector<scl_index_t> labels2(10);
    
    scl_real_t nmi = 0.0;
    scl_error_t err = scl_metrics_normalized_mutual_information(
        nullptr, labels2.data(), 10, &nmi
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(nmi_null_labels2)
{
    std::vector<scl_index_t> labels1(10);
    
    scl_real_t nmi = 0.0;
    scl_error_t err = scl_metrics_normalized_mutual_information(
        labels1.data(), nullptr, 10, &nmi
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(nmi_null_output)
{
    std::vector<scl_index_t> labels1 = {0, 0, 1, 1};
    std::vector<scl_index_t> labels2 = {0, 1, 0, 1};
    
    scl_error_t err = scl_metrics_normalized_mutual_information(
        labels1.data(), labels2.data(), 4, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Homogeneity Score
// =============================================================================

SCL_TEST_SUITE(homogeneity)

SCL_TEST_CASE(homogeneity_perfect)
{
    scl_index_t n_cells = 10;
    
    // Perfect homogeneity: true labels are subsets of predicted
    std::vector<scl_index_t> labels_true = {0, 0, 1, 1, 2, 2};
    std::vector<scl_index_t> labels_pred = {0, 0, 1, 1, 2, 2};
    
    scl_real_t homogeneity = 0.0;
    scl_error_t err = scl_metrics_homogeneity_score(
        labels_true.data(), labels_pred.data(), n_cells, &homogeneity
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Perfect match should have homogeneity = 1.0
    SCL_ASSERT_NEAR(homogeneity, 1.0, 1e-9);
}

SCL_TEST_RETRY(homogeneity_random, 3)
{
    Random rng(666);
    
    scl_index_t n_cells = rng.uniform_int(20, 50);
    scl_index_t n_clusters = rng.uniform_int(2, 8);
    
    std::vector<scl_index_t> labels_true(n_cells);
    std::vector<scl_index_t> labels_pred(n_cells);
    
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels_true[i] = rng.uniform_int(0, n_clusters - 1);
        labels_pred[i] = rng.uniform_int(0, n_clusters - 1);
    }
    
    scl_real_t homogeneity = 0.0;
    scl_error_t err = scl_metrics_homogeneity_score(
        labels_true.data(), labels_pred.data(), n_cells, &homogeneity
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Homogeneity should be in [0, 1] range
    SCL_ASSERT_GE(homogeneity, 0.0);
    SCL_ASSERT_LE(homogeneity, 1.0);
}

SCL_TEST_CASE(homogeneity_null_labels_true)
{
    std::vector<scl_index_t> labels_pred(10);
    
    scl_real_t homogeneity = 0.0;
    scl_error_t err = scl_metrics_homogeneity_score(
        nullptr, labels_pred.data(), 10, &homogeneity
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(homogeneity_null_labels_pred)
{
    std::vector<scl_index_t> labels_true(10);
    
    scl_real_t homogeneity = 0.0;
    scl_error_t err = scl_metrics_homogeneity_score(
        labels_true.data(), nullptr, 10, &homogeneity
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(homogeneity_null_output)
{
    std::vector<scl_index_t> labels_true = {0, 0, 1, 1};
    std::vector<scl_index_t> labels_pred = {0, 1, 0, 1};
    
    scl_error_t err = scl_metrics_homogeneity_score(
        labels_true.data(), labels_pred.data(), 4, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Completeness Score
// =============================================================================

SCL_TEST_SUITE(completeness)

SCL_TEST_CASE(completeness_perfect)
{
    scl_index_t n_cells = 10;
    
    // Perfect completeness: predicted labels are subsets of true
    std::vector<scl_index_t> labels_true = {0, 0, 1, 1, 2, 2};
    std::vector<scl_index_t> labels_pred = {0, 0, 1, 1, 2, 2};
    
    scl_real_t completeness = 0.0;
    scl_error_t err = scl_metrics_completeness_score(
        labels_true.data(), labels_pred.data(), n_cells, &completeness
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Perfect match should have completeness = 1.0
    SCL_ASSERT_NEAR(completeness, 1.0, 1e-9);
}

SCL_TEST_RETRY(completeness_random, 3)
{
    Random rng(777);
    
    scl_index_t n_cells = rng.uniform_int(20, 50);
    scl_index_t n_clusters = rng.uniform_int(2, 8);
    
    std::vector<scl_index_t> labels_true(n_cells);
    std::vector<scl_index_t> labels_pred(n_cells);
    
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels_true[i] = rng.uniform_int(0, n_clusters - 1);
        labels_pred[i] = rng.uniform_int(0, n_clusters - 1);
    }
    
    scl_real_t completeness = 0.0;
    scl_error_t err = scl_metrics_completeness_score(
        labels_true.data(), labels_pred.data(), n_cells, &completeness
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Completeness should be in [0, 1] range
    SCL_ASSERT_GE(completeness, 0.0);
    SCL_ASSERT_LE(completeness, 1.0);
}

SCL_TEST_CASE(completeness_null_labels_true)
{
    std::vector<scl_index_t> labels_pred(10);
    
    scl_real_t completeness = 0.0;
    scl_error_t err = scl_metrics_completeness_score(
        nullptr, labels_pred.data(), 10, &completeness
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(completeness_null_labels_pred)
{
    std::vector<scl_index_t> labels_true(10);
    
    scl_real_t completeness = 0.0;
    scl_error_t err = scl_metrics_completeness_score(
        labels_true.data(), nullptr, 10, &completeness
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(completeness_null_output)
{
    std::vector<scl_index_t> labels_true = {0, 0, 1, 1};
    std::vector<scl_index_t> labels_pred = {0, 1, 0, 1};
    
    scl_error_t err = scl_metrics_completeness_score(
        labels_true.data(), labels_pred.data(), 4, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// V-Measure Score
// =============================================================================

SCL_TEST_SUITE(v_measure)

SCL_TEST_CASE(v_measure_perfect)
{
    scl_index_t n_cells = 10;
    
    // Perfect clustering
    std::vector<scl_index_t> labels_true = {0, 0, 1, 1, 2, 2};
    std::vector<scl_index_t> labels_pred = {0, 0, 1, 1, 2, 2};
    
    scl_real_t v_measure = 0.0;
    scl_error_t err = scl_metrics_v_measure(
        labels_true.data(), labels_pred.data(), n_cells, &v_measure
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // Perfect match should have v_measure = 1.0
    SCL_ASSERT_NEAR(v_measure, 1.0, 1e-9);
}

SCL_TEST_RETRY(v_measure_random, 3)
{
    Random rng(888);
    
    scl_index_t n_cells = rng.uniform_int(20, 50);
    scl_index_t n_clusters = rng.uniform_int(2, 8);
    
    std::vector<scl_index_t> labels_true(n_cells);
    std::vector<scl_index_t> labels_pred(n_cells);
    
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels_true[i] = rng.uniform_int(0, n_clusters - 1);
        labels_pred[i] = rng.uniform_int(0, n_clusters - 1);
    }
    
    scl_real_t v_measure = 0.0;
    scl_error_t err = scl_metrics_v_measure(
        labels_true.data(), labels_pred.data(), n_cells, &v_measure
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // V-measure should be in [0, 1] range
    SCL_ASSERT_GE(v_measure, 0.0);
    SCL_ASSERT_LE(v_measure, 1.0);
}

SCL_TEST_CASE(v_measure_null_labels_true)
{
    std::vector<scl_index_t> labels_pred(10);
    
    scl_real_t v_measure = 0.0;
    scl_error_t err = scl_metrics_v_measure(
        nullptr, labels_pred.data(), 10, &v_measure
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(v_measure_null_labels_pred)
{
    std::vector<scl_index_t> labels_true(10);
    
    scl_real_t v_measure = 0.0;
    scl_error_t err = scl_metrics_v_measure(
        labels_true.data(), nullptr, 10, &v_measure
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(v_measure_null_output)
{
    std::vector<scl_index_t> labels_true = {0, 0, 1, 1};
    std::vector<scl_index_t> labels_pred = {0, 1, 0, 1};
    
    scl_error_t err = scl_metrics_v_measure(
        labels_true.data(), labels_pred.data(), 4, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

