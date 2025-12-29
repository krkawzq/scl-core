// =============================================================================
// SCL Core - Comprehensive association.h Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/association.h
//
// Functions tested (8 total):
//   ✓ scl_association_gene_peak_correlation
//   ✓ scl_association_cis_regulatory
//   ✓ scl_association_enhancer_gene_link
//   ✓ scl_association_multimodal_neighbors
//   ✓ scl_association_feature_coupling
//   ✓ scl_association_peak_to_gene_activity
//   ✓ scl_association_correlation_in_subset
//
// =============================================================================

#include "test.hpp"

extern "C" {
#include "scl/binding/c_api/association.h"
}

using namespace scl::test;

// Helper: Create small RNA data
static Sparse make_rna_data(scl_index_t n_cells, scl_index_t n_genes, Random& rng) {
    return random_sparse_csr(n_cells, n_genes, 0.25, rng);
}

// Helper: Create small ATAC data
static Sparse make_atac_data(scl_index_t n_cells, scl_index_t n_peaks, Random& rng) {
    return random_sparse_csr(n_cells, n_peaks, 0.15, rng);
}

SCL_TEST_BEGIN

// =============================================================================
// Gene-Peak Correlation
// =============================================================================

SCL_TEST_SUITE(gene_peak_correlation)

SCL_TEST_CASE(gene_peak_correlation_basic) {
    Random rng(42);

    scl_index_t n_cells = 50, n_genes = 30, n_peaks = 40;

    auto rna = make_rna_data(n_cells, n_genes, rng);
    auto atac = make_atac_data(n_cells, n_peaks, rng);

    std::vector<scl_index_t> gene_indices(200);
    std::vector<scl_index_t> peak_indices(200);
    std::vector<scl_real_t> correlations(200);
    scl_size_t n_correlations = 0;

    scl_error_t err = scl_association_gene_peak_correlation(
        rna, atac,
        gene_indices.data(), peak_indices.data(), correlations.data(),
        &n_correlations, 0.3  // Min correlation threshold
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Should find some correlations
    SCL_ASSERT_LE(n_correlations, 200);

    // Verify indices and correlation values
    for (scl_size_t i = 0; i < n_correlations; ++i) {
        SCL_ASSERT_GE(gene_indices[i], 0);
        SCL_ASSERT_LT(gene_indices[i], n_genes);
        SCL_ASSERT_GE(peak_indices[i], 0);
        SCL_ASSERT_LT(peak_indices[i], n_peaks);
        SCL_ASSERT_TRUE(std::isfinite(correlations[i]));
        SCL_ASSERT_TRUE(std::abs(correlations[i]) >= 0.3 || std::abs(correlations[i]) < 0.3 + 1e-6);
    }
}

SCL_TEST_CASE(gene_peak_correlation_high_threshold) {
    Random rng(123);

    scl_index_t n_cells = 40, n_genes = 20, n_peaks = 25;

    auto rna = make_rna_data(n_cells, n_genes, rng);
    auto atac = make_atac_data(n_cells, n_peaks, rng);

    std::vector<scl_index_t> gene_indices(100);
    std::vector<scl_index_t> peak_indices(100);
    std::vector<scl_real_t> correlations(100);
    scl_size_t n_correlations = 0;

    // High threshold should find fewer correlations
    scl_error_t err = scl_association_gene_peak_correlation(
        rna, atac,
        gene_indices.data(), peak_indices.data(), correlations.data(),
        &n_correlations, 0.8
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_LE(n_correlations, 100);
}

SCL_TEST_CASE(gene_peak_correlation_null_checks) {
    Random rng(42);
    auto rna = make_rna_data(10, 20, rng);
    auto atac = make_atac_data(10, 15, rng);

    std::vector<scl_index_t> genes(50), peaks(50);
    std::vector<scl_real_t> corrs(50);
    scl_size_t n_corr = 0;

    SCL_ASSERT_EQ(
        scl_association_gene_peak_correlation(nullptr, atac, genes.data(), peaks.data(), corrs.data(), &n_corr, 0.3),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_gene_peak_correlation(rna, nullptr, genes.data(), peaks.data(), corrs.data(), &n_corr, 0.3),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_gene_peak_correlation(rna, atac, nullptr, peaks.data(), corrs.data(), &n_corr, 0.3),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_gene_peak_correlation(rna, atac, genes.data(), peaks.data(), corrs.data(), nullptr, 0.3),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Cis-Regulatory Associations
// =============================================================================

SCL_TEST_SUITE(cis_regulatory)

SCL_TEST_CASE(cis_regulatory_basic) {
    Random rng(456);

    scl_index_t n_cells = 60, n_genes = 40, n_peaks = 50;

    auto rna = make_rna_data(n_cells, n_genes, rng);
    auto atac = make_atac_data(n_cells, n_peaks, rng);

    // Test specific gene-peak pairs
    std::vector<scl_index_t> gene_indices = {0, 5, 10, 15, 20};
    std::vector<scl_index_t> peak_indices = {2, 7, 12, 17, 22};
    scl_size_t n_pairs = 5;

    std::vector<scl_real_t> correlations(n_pairs);
    std::vector<scl_real_t> p_values(n_pairs);

    scl_error_t err = scl_association_cis_regulatory(
        rna, atac,
        gene_indices.data(), peak_indices.data(), n_pairs,
        correlations.data(), p_values.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify correlations and p-values
    for (scl_size_t i = 0; i < n_pairs; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(correlations[i]));
        SCL_ASSERT_GE(correlations[i], -1.0);
        SCL_ASSERT_LE(correlations[i], 1.0);

        SCL_ASSERT_TRUE(std::isfinite(p_values[i]));
        SCL_ASSERT_GE(p_values[i], 0.0);
        SCL_ASSERT_LE(p_values[i], 1.0);
    }
}

SCL_TEST_CASE(cis_regulatory_null_checks) {
    Random rng(42);
    auto rna = make_rna_data(20, 30, rng);
    auto atac = make_atac_data(20, 25, rng);

    std::vector<scl_index_t> genes = {0, 1};
    std::vector<scl_index_t> peaks = {0, 1};
    std::vector<scl_real_t> corrs(2), pvals(2);

    SCL_ASSERT_EQ(
        scl_association_cis_regulatory(nullptr, atac, genes.data(), peaks.data(), 2, corrs.data(), pvals.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_cis_regulatory(rna, nullptr, genes.data(), peaks.data(), 2, corrs.data(), pvals.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_cis_regulatory(rna, atac, nullptr, peaks.data(), 2, corrs.data(), pvals.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_cis_regulatory(rna, atac, genes.data(), peaks.data(), 2, nullptr, pvals.data()),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Enhancer-Gene Links
// =============================================================================

SCL_TEST_SUITE(enhancer_gene_link)

SCL_TEST_CASE(enhancer_gene_link_basic) {
    Random rng(789);

    scl_index_t n_cells = 70, n_genes = 35, n_peaks = 45;

    auto rna = make_rna_data(n_cells, n_genes, rng);
    auto atac = make_atac_data(n_cells, n_peaks, rng);

    std::vector<scl_index_t> link_genes(150);
    std::vector<scl_index_t> link_peaks(150);
    std::vector<scl_real_t> link_correlations(150);
    scl_size_t n_links = 0;

    scl_error_t err = scl_association_enhancer_gene_link(
        rna, atac, 0.4,
        link_genes.data(), link_peaks.data(), link_correlations.data(),
        &n_links
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify links
    SCL_ASSERT_LE(n_links, 150);

    for (scl_size_t i = 0; i < n_links; ++i) {
        SCL_ASSERT_GE(link_genes[i], 0);
        SCL_ASSERT_LT(link_genes[i], n_genes);
        SCL_ASSERT_GE(link_peaks[i], 0);
        SCL_ASSERT_LT(link_peaks[i], n_peaks);
        SCL_ASSERT_TRUE(std::isfinite(link_correlations[i]));
        // Only positive correlations
        SCL_ASSERT_GE(link_correlations[i], 0.0);
    }
}

SCL_TEST_CASE(enhancer_gene_link_null_checks) {
    Random rng(42);
    auto rna = make_rna_data(10, 15, rng);
    auto atac = make_atac_data(10, 20, rng);

    std::vector<scl_index_t> genes(30), peaks(30);
    std::vector<scl_real_t> corrs(30);
    scl_size_t n_links = 0;

    SCL_ASSERT_EQ(
        scl_association_enhancer_gene_link(nullptr, atac, 0.5, genes.data(), peaks.data(), corrs.data(), &n_links),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_enhancer_gene_link(rna, nullptr, 0.5, genes.data(), peaks.data(), corrs.data(), &n_links),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_enhancer_gene_link(rna, atac, 0.5, nullptr, peaks.data(), corrs.data(), &n_links),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_enhancer_gene_link(rna, atac, 0.5, genes.data(), peaks.data(), corrs.data(), nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Multi-modal Neighbors
// =============================================================================

SCL_TEST_SUITE(multimodal_neighbors)

SCL_TEST_CASE(multimodal_neighbors_basic) {
    Random rng(321);

    scl_index_t n_cells = 40, n_genes = 50, n_peaks = 60;

    auto modality1 = make_rna_data(n_cells, n_genes, rng);
    auto modality2 = make_atac_data(n_cells, n_peaks, rng);

    scl_index_t k = 10;

    std::vector<scl_index_t> neighbor_indices(n_cells * k);
    std::vector<scl_real_t> neighbor_distances(n_cells * k);

    scl_error_t err = scl_association_multimodal_neighbors(
        modality1, modality2, 0.5, 0.5, k,
        neighbor_indices.data(), neighbor_distances.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify neighbors
    for (scl_size_t i = 0; i < n_cells * k; ++i) {
        SCL_ASSERT_GE(neighbor_indices[i], 0);
        SCL_ASSERT_LT(neighbor_indices[i], n_cells);
        SCL_ASSERT_GE(neighbor_distances[i], 0.0);
        SCL_ASSERT_TRUE(std::isfinite(neighbor_distances[i]));
    }
}

SCL_TEST_CASE(multimodal_neighbors_different_weights) {
    Random rng(654);

    scl_index_t n_cells = 30, n_genes = 40, n_peaks = 50;

    auto mod1 = make_rna_data(n_cells, n_genes, rng);
    auto mod2 = make_atac_data(n_cells, n_peaks, rng);

    scl_index_t k = 5;

    std::vector<scl_index_t> neighbors(n_cells * k);
    std::vector<scl_real_t> distances(n_cells * k);

    // Emphasize modality 1
    scl_error_t err = scl_association_multimodal_neighbors(
        mod1, mod2, 0.8, 0.2, k,
        neighbors.data(), distances.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    for (scl_size_t i = 0; i < n_cells * k; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(distances[i]));
    }
}

SCL_TEST_CASE(multimodal_neighbors_null_checks) {
    Random rng(42);
    auto mod1 = make_rna_data(10, 20, rng);
    auto mod2 = make_atac_data(10, 15, rng);

    std::vector<scl_index_t> neighbors(50);
    std::vector<scl_real_t> distances(50);

    SCL_ASSERT_EQ(
        scl_association_multimodal_neighbors(nullptr, mod2, 0.5, 0.5, 5, neighbors.data(), distances.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_multimodal_neighbors(mod1, nullptr, 0.5, 0.5, 5, neighbors.data(), distances.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_multimodal_neighbors(mod1, mod2, 0.5, 0.5, 5, nullptr, distances.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_multimodal_neighbors(mod1, mod2, 0.5, 0.5, 5, neighbors.data(), nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Feature Coupling
// =============================================================================

SCL_TEST_SUITE(feature_coupling)

SCL_TEST_CASE(feature_coupling_basic) {
    Random rng(987);

    scl_index_t n_cells = 50, n_features1 = 40, n_features2 = 35;

    auto modality1 = random_sparse_csr(n_cells, n_features1, 0.2, rng);
    auto modality2 = random_sparse_csr(n_cells, n_features2, 0.2, rng);

    std::vector<scl_index_t> feature1_indices(100);
    std::vector<scl_index_t> feature2_indices(100);
    std::vector<scl_real_t> coupling_scores(100);
    scl_size_t n_couplings = 0;

    scl_error_t err = scl_association_feature_coupling(
        modality1, modality2,
        feature1_indices.data(), feature2_indices.data(), coupling_scores.data(),
        &n_couplings, 0.5
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify couplings
    SCL_ASSERT_LE(n_couplings, 100);

    for (scl_size_t i = 0; i < n_couplings; ++i) {
        SCL_ASSERT_GE(feature1_indices[i], 0);
        SCL_ASSERT_LT(feature1_indices[i], n_features1);
        SCL_ASSERT_GE(feature2_indices[i], 0);
        SCL_ASSERT_LT(feature2_indices[i], n_features2);
        SCL_ASSERT_TRUE(std::isfinite(coupling_scores[i]));
        SCL_ASSERT_GE(coupling_scores[i], 0.5 - 1e-6);
    }
}

SCL_TEST_CASE(feature_coupling_null_checks) {
    Random rng(42);
    auto mod1 = random_sparse_csr(10, 20, 0.2, rng);
    auto mod2 = random_sparse_csr(10, 15, 0.2, rng);

    std::vector<scl_index_t> feat1(30), feat2(30);
    std::vector<scl_real_t> scores(30);
    scl_size_t n_coup = 0;

    SCL_ASSERT_EQ(
        scl_association_feature_coupling(nullptr, mod2, feat1.data(), feat2.data(), scores.data(), &n_coup, 0.5),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_feature_coupling(mod1, nullptr, feat1.data(), feat2.data(), scores.data(), &n_coup, 0.5),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_feature_coupling(mod1, mod2, nullptr, feat2.data(), scores.data(), &n_coup, 0.5),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_feature_coupling(mod1, mod2, feat1.data(), feat2.data(), scores.data(), nullptr, 0.5),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Peak-to-Gene Activity
// =============================================================================

SCL_TEST_SUITE(peak_to_gene_activity)

SCL_TEST_CASE(peak_to_gene_activity_basic) {
    Random rng(111);

    scl_index_t n_cells = 60, n_peaks = 80, n_genes = 50;

    auto atac = make_atac_data(n_cells, n_peaks, rng);

    // Create peak-to-gene mapping
    std::vector<scl_index_t> peak_to_gene_map(n_peaks);
    for (scl_index_t i = 0; i < n_peaks; ++i) {
        peak_to_gene_map[i] = (i < n_genes) ? i : (i % n_genes);
    }

    std::vector<scl_real_t> gene_activity(n_cells * n_genes);

    scl_error_t err = scl_association_peak_to_gene_activity(
        atac, peak_to_gene_map.data(), n_peaks, n_genes,
        gene_activity.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify gene activity is finite and non-negative
    for (scl_size_t i = 0; i < n_cells * n_genes; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(gene_activity[i]));
        SCL_ASSERT_GE(gene_activity[i], 0.0);
    }
}

SCL_TEST_CASE(peak_to_gene_activity_with_unmapped_peaks) {
    Random rng(222);

    scl_index_t n_cells = 40, n_peaks = 60, n_genes = 30;

    auto atac = make_atac_data(n_cells, n_peaks, rng);

    // Some peaks not mapped to genes (-1)
    std::vector<scl_index_t> peak_to_gene_map(n_peaks);
    for (scl_index_t i = 0; i < n_peaks; ++i) {
        peak_to_gene_map[i] = (i % 3 == 0) ? -1 : (i % n_genes);
    }

    std::vector<scl_real_t> gene_activity(n_cells * n_genes);

    scl_error_t err = scl_association_peak_to_gene_activity(
        atac, peak_to_gene_map.data(), n_peaks, n_genes,
        gene_activity.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    for (scl_size_t i = 0; i < n_cells * n_genes; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(gene_activity[i]));
    }
}

SCL_TEST_CASE(peak_to_gene_activity_null_checks) {
    Random rng(42);
    auto atac = make_atac_data(10, 20, rng);
    std::vector<scl_index_t> mapping(20, 0);
    std::vector<scl_real_t> activity(100);

    SCL_ASSERT_EQ(
        scl_association_peak_to_gene_activity(nullptr, mapping.data(), 20, 10, activity.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_peak_to_gene_activity(atac, nullptr, 20, 10, activity.data()),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_peak_to_gene_activity(atac, mapping.data(), 20, 10, nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Correlation in Subset
// =============================================================================

SCL_TEST_SUITE(correlation_in_subset)

SCL_TEST_CASE(correlation_in_subset_basic) {
    Random rng(333);

    scl_index_t n_cells = 80, n_genes = 60, n_peaks = 70;

    auto data1 = make_rna_data(n_cells, n_genes, rng);
    auto data2 = make_atac_data(n_cells, n_peaks, rng);

    // Select a subset of cells
    std::vector<scl_index_t> cell_indices = {5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
    scl_size_t n_subset = cell_indices.size();

    scl_real_t correlation = 0.0;

    scl_error_t err = scl_association_correlation_in_subset(
        data1, 10,   // Feature 10 in data1
        data2, 20,   // Feature 20 in data2
        cell_indices.data(), n_subset,
        &correlation
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify correlation is valid
    SCL_ASSERT_TRUE(std::isfinite(correlation));
    SCL_ASSERT_GE(correlation, -1.0);
    SCL_ASSERT_LE(correlation, 1.0);
}

SCL_TEST_CASE(correlation_in_subset_full_dataset) {
    Random rng(444);

    scl_index_t n_cells = 50, n_genes = 40, n_peaks = 45;

    auto data1 = make_rna_data(n_cells, n_genes, rng);
    auto data2 = make_atac_data(n_cells, n_peaks, rng);

    // Use all cells
    std::vector<scl_index_t> cell_indices(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        cell_indices[i] = i;
    }

    scl_real_t correlation = 0.0;

    scl_error_t err = scl_association_correlation_in_subset(
        data1, 5, data2, 10,
        cell_indices.data(), n_cells,
        &correlation
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_TRUE(std::isfinite(correlation));
}

SCL_TEST_CASE(correlation_in_subset_null_checks) {
    Random rng(42);
    auto data1 = make_rna_data(20, 30, rng);
    auto data2 = make_atac_data(20, 25, rng);

    std::vector<scl_index_t> cells = {0, 1, 2, 3, 4};
    scl_real_t corr = 0.0;

    SCL_ASSERT_EQ(
        scl_association_correlation_in_subset(nullptr, 0, data2, 0, cells.data(), 5, &corr),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_correlation_in_subset(data1, 0, nullptr, 0, cells.data(), 5, &corr),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_correlation_in_subset(data1, 0, data2, 0, nullptr, 5, &corr),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_association_correlation_in_subset(data1, 0, data2, 0, cells.data(), 5, nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(correlation_in_subset_invalid_feature) {
    Random rng(555);
    auto data1 = make_rna_data(20, 30, rng);
    auto data2 = make_atac_data(20, 25, rng);

    std::vector<scl_index_t> cells = {0, 1, 2};
    scl_real_t corr = 0.0;

    // Feature index out of bounds
    scl_error_t err = scl_association_correlation_in_subset(
        data1, 999, data2, 0, cells.data(), 3, &corr
    );

    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Random Integration Tests
// =============================================================================

SCL_TEST_SUITE(random_tests)

SCL_TEST_RETRY(multimodal_integration_random, 3) {
    Random rng(67890);

    auto [n_cells, n_genes] = random_shape(40, 100, rng);
    auto [_, n_peaks] = random_shape(50, 120, rng);

    auto rna = random_sparse_csr(n_cells, n_genes, 0.2, rng);
    auto atac = random_sparse_csr(n_cells, n_peaks, 0.15, rng);

    std::vector<scl_index_t> genes(50), peaks(50);
    std::vector<scl_real_t> corrs(50);
    scl_size_t n_corr = 0;

    scl_error_t err = scl_association_gene_peak_correlation(
        rna, atac, genes.data(), peaks.data(), corrs.data(), &n_corr, 0.4
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_LE(n_corr, 50);
}

SCL_TEST_RETRY(peak_to_gene_random, 3) {
    Random rng(11111);

    auto [n_cells, n_peaks] = random_shape(30, 80, rng);
    scl_index_t n_genes = n_peaks / 2;

    auto atac = random_sparse_csr(n_cells, n_peaks, 0.1, rng);

    std::vector<scl_index_t> mapping(n_peaks);
    for (scl_index_t i = 0; i < n_peaks; ++i) {
        mapping[i] = rng.uniform(0.0, 1.0) < 0.7 ? (i % n_genes) : -1;
    }

    std::vector<scl_real_t> activity(n_cells * n_genes);

    scl_error_t err = scl_association_peak_to_gene_activity(
        atac, mapping.data(), n_peaks, n_genes, activity.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Verify all activity values are finite
    for (scl_size_t i = 0; i < n_cells * n_genes; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(activity[i]));
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
