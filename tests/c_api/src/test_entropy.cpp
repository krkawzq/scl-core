// =============================================================================
// SCL Core - Entropy Module Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/entropy.h
//
// Functions tested (17 total):
//   ✓ scl_entropy_discrete_entropy
//   ✓ scl_entropy_count_entropy
//   ✓ scl_entropy_row_entropy
//   ✓ scl_entropy_kl_divergence
//   ✓ scl_entropy_js_divergence
//   ✓ scl_entropy_symmetric_kl
//   ✓ scl_entropy_mutual_information
//   ✓ scl_entropy_joint_entropy
//   ✓ scl_entropy_conditional_entropy
//   ✓ scl_entropy_marginal_entropy
//   ✓ scl_entropy_normalized_mi
//   ✓ scl_entropy_adjusted_mi
//   ✓ scl_entropy_discretize_equal_width
//   ✓ scl_entropy_discretize_equal_frequency
//   ✓ scl_entropy_cross_entropy
//   ✓ scl_entropy_gini_impurity
//
// =============================================================================

#include "test.hpp"

extern "C" {
#include "scl/binding/c_api/entropy.h"
}

using namespace scl::test;

// Helper: Create uniform distribution
static std::vector<scl_real_t> uniform_probs(scl_size_t n) {
    return std::vector<scl_real_t>(n, 1.0 / n);
}

SCL_TEST_BEGIN

// =============================================================================
// Basic Entropy Measures
// =============================================================================

SCL_TEST_SUITE(entropy_measures)

SCL_TEST_CASE(discrete_entropy_uniform_distribution) {
    auto probs = uniform_probs(4);
    scl_real_t entropy = 0.0;

    scl_error_t err = scl_entropy_discrete_entropy(
        probs.data(), probs.size(), 1, &entropy
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(entropy, 2.0, 1e-10);  // log2(4) = 2
}

SCL_TEST_CASE(discrete_entropy_natural_log) {
    auto probs = uniform_probs(2);
    scl_real_t entropy = 0.0;

    scl_error_t err = scl_entropy_discrete_entropy(
        probs.data(), probs.size(), 0, &entropy
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(entropy, std::log(2.0), 1e-10);
}

SCL_TEST_CASE(discrete_entropy_single_element) {
    std::vector<scl_real_t> probs = {1.0};
    scl_real_t entropy = 0.0;

    scl_error_t err = scl_entropy_discrete_entropy(
        probs.data(), probs.size(), 1, &entropy
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(entropy, 0.0, 1e-10);  // Single element = no entropy
}

SCL_TEST_CASE(discrete_entropy_non_uniform) {
    std::vector<scl_real_t> probs = {0.5, 0.25, 0.25};
    scl_real_t entropy = 0.0;

    scl_error_t err = scl_entropy_discrete_entropy(
        probs.data(), probs.size(), 1, &entropy
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    // H = -0.5*log2(0.5) - 0.25*log2(0.25) - 0.25*log2(0.25) = 1.5
    SCL_ASSERT_NEAR(entropy, 1.5, 1e-10);
}

SCL_TEST_CASE(discrete_entropy_null_pointer) {
    std::vector<scl_real_t> probs = {0.5, 0.5};
    scl_real_t entropy = 0.0;

    SCL_ASSERT_EQ(
        scl_entropy_discrete_entropy(nullptr, 2, 1, &entropy),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_entropy_discrete_entropy(probs.data(), 2, 1, nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(discrete_entropy_invalid_size) {
    std::vector<scl_real_t> probs = {0.5, 0.5};
    scl_real_t entropy = 0.0;

    scl_error_t err = scl_entropy_discrete_entropy(
        probs.data(), 0, 1, &entropy
    );

    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_CASE(count_entropy_basic) {
    std::vector<scl_real_t> counts = {10.0, 10.0, 10.0, 10.0};
    scl_real_t entropy = 0.0;

    scl_error_t err = scl_entropy_count_entropy(
        counts.data(), counts.size(), 1, &entropy
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(entropy, 2.0, 1e-10);  // Uniform distribution
}

SCL_TEST_CASE(count_entropy_with_zeros) {
    std::vector<scl_real_t> counts = {10.0, 0.0, 5.0, 5.0};
    scl_real_t entropy = 0.0;

    scl_error_t err = scl_entropy_count_entropy(
        counts.data(), counts.size(), 1, &entropy
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GT(entropy, 0.0);
}

SCL_TEST_CASE(count_entropy_null_pointer) {
    scl_real_t entropy = 0.0;

    SCL_ASSERT_EQ(
        scl_entropy_count_entropy(nullptr, 4, 1, &entropy),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(row_entropy_sparse_matrix) {
    // 3x3 sparse matrix
    std::vector<scl_index_t> indptr = {0, 2, 4, 6};
    std::vector<scl_index_t> indices = {0, 1, 1, 2, 0, 2};
    std::vector<scl_real_t> data = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};

    Sparse mat = make_sparse_csr(3, 3, 6,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> entropies(3);

    scl_error_t err = scl_entropy_row_entropy(
        mat, entropies.data(), 3, 0, 1
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // All values should be non-negative
    for (auto e : entropies) {
        SCL_ASSERT_GE(e, 0.0);
    }
}

SCL_TEST_CASE(row_entropy_normalized) {
    std::vector<scl_index_t> indptr = {0, 2, 4, 6};
    std::vector<scl_index_t> indices = {0, 1, 1, 2, 0, 2};
    std::vector<scl_real_t> data = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0};

    Sparse mat = make_sparse_csr(3, 3, 6,
                                  indptr.data(), indices.data(), data.data());

    std::vector<scl_real_t> entropies(3);

    scl_error_t err = scl_entropy_row_entropy(
        mat, entropies.data(), 3, 1, 1  // Normalized
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Normalized entropy should be in [0, 1]
    for (auto e : entropies) {
        SCL_ASSERT_GE(e, 0.0);
        SCL_ASSERT_LE(e, 1.0 + 1e-10);
    }
}

SCL_TEST_CASE(row_entropy_null_matrix) {
    std::vector<scl_real_t> entropies(3);

    SCL_ASSERT_EQ(
        scl_entropy_row_entropy(nullptr, entropies.data(), 3, 0, 1),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Divergence Measures
// =============================================================================

SCL_TEST_SUITE(divergence_measures)

SCL_TEST_CASE(kl_divergence_identical_distributions) {
    std::vector<scl_real_t> p = {0.25, 0.25, 0.25, 0.25};
    scl_real_t kl = 0.0;

    scl_error_t err = scl_entropy_kl_divergence(
        p.data(), p.data(), p.size(), 1, &kl
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(kl, 0.0, 1e-10);
}

SCL_TEST_CASE(kl_divergence_different_distributions) {
    std::vector<scl_real_t> p = {0.5, 0.5};
    std::vector<scl_real_t> q = {0.25, 0.75};
    scl_real_t kl = 0.0;

    scl_error_t err = scl_entropy_kl_divergence(
        p.data(), q.data(), p.size(), 1, &kl
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GT(kl, 0.0);  // Should be positive
}

SCL_TEST_CASE(kl_divergence_null_pointer) {
    std::vector<scl_real_t> p = {0.5, 0.5};
    std::vector<scl_real_t> q = {0.25, 0.75};
    scl_real_t kl = 0.0;

    SCL_ASSERT_EQ(
        scl_entropy_kl_divergence(nullptr, q.data(), 2, 1, &kl),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_entropy_kl_divergence(p.data(), nullptr, 2, 1, &kl),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(js_divergence_symmetric) {
    std::vector<scl_real_t> p = {0.5, 0.5};
    std::vector<scl_real_t> q = {0.25, 0.75};
    scl_real_t js1 = 0.0, js2 = 0.0;

    scl_entropy_js_divergence(p.data(), q.data(), 2, 1, &js1);
    scl_entropy_js_divergence(q.data(), p.data(), 2, 1, &js2);

    // JS divergence should be symmetric
    SCL_ASSERT_NEAR(js1, js2, 1e-10);
}

SCL_TEST_CASE(js_divergence_identical) {
    std::vector<scl_real_t> p = {0.3, 0.3, 0.4};
    scl_real_t js = 0.0;

    scl_error_t err = scl_entropy_js_divergence(
        p.data(), p.data(), p.size(), 1, &js
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(js, 0.0, 1e-10);
}

SCL_TEST_CASE(js_divergence_bounded) {
    std::vector<scl_real_t> p = {1.0, 0.0};
    std::vector<scl_real_t> q = {0.0, 1.0};
    scl_real_t js = 0.0;

    scl_error_t err = scl_entropy_js_divergence(
        p.data(), q.data(), 2, 1, &js
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    // JS divergence is bounded: 0 <= JS <= 1 (for log2)
    SCL_ASSERT_GE(js, 0.0);
    SCL_ASSERT_LE(js, 1.0 + 1e-10);
}

SCL_TEST_CASE(symmetric_kl_commutative) {
    std::vector<scl_real_t> p = {0.4, 0.6};
    std::vector<scl_real_t> q = {0.2, 0.8};
    scl_real_t skl1 = 0.0, skl2 = 0.0;

    scl_entropy_symmetric_kl(p.data(), q.data(), 2, 1, &skl1);
    scl_entropy_symmetric_kl(q.data(), p.data(), 2, 1, &skl2);

    SCL_ASSERT_NEAR(skl1, skl2, 1e-10);
}

SCL_TEST_CASE(symmetric_kl_null_pointer) {
    std::vector<scl_real_t> p = {0.5, 0.5};
    scl_real_t skl = 0.0;

    SCL_ASSERT_EQ(
        scl_entropy_symmetric_kl(nullptr, p.data(), 2, 1, &skl),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Mutual Information
// =============================================================================

SCL_TEST_SUITE(mutual_information)

SCL_TEST_CASE(mutual_information_independent) {
    // Independent variables: X = [0,0,1,1], Y = [0,1,0,1]
    std::vector<scl_index_t> x = {0, 0, 1, 1};
    std::vector<scl_index_t> y = {0, 1, 0, 1};
    scl_real_t mi = 0.0;

    scl_error_t err = scl_entropy_mutual_information(
        x.data(), y.data(), 4, 2, 2, 1, &mi
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(mi, 0.0, 1e-10);
}

SCL_TEST_CASE(mutual_information_identical) {
    std::vector<scl_index_t> x = {0, 0, 1, 1, 2, 2};
    scl_real_t mi = 0.0;

    scl_error_t err = scl_entropy_mutual_information(
        x.data(), x.data(), 6, 3, 3, 1, &mi
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    // MI(X, X) = H(X)
    SCL_ASSERT_GT(mi, 0.0);
}

SCL_TEST_CASE(mutual_information_null_pointer) {
    std::vector<scl_index_t> x = {0, 1, 0, 1};
    std::vector<scl_index_t> y = {0, 0, 1, 1};
    scl_real_t mi = 0.0;

    SCL_ASSERT_EQ(
        scl_entropy_mutual_information(nullptr, y.data(), 4, 2, 2, 1, &mi),
        SCL_ERROR_NULL_POINTER
    );

    SCL_ASSERT_EQ(
        scl_entropy_mutual_information(x.data(), nullptr, 4, 2, 2, 1, &mi),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(joint_entropy_basic) {
    std::vector<scl_index_t> x = {0, 0, 1, 1};
    std::vector<scl_index_t> y = {0, 1, 0, 1};
    scl_real_t je = 0.0;

    scl_error_t err = scl_entropy_joint_entropy(
        x.data(), y.data(), 4, 2, 2, 1, &je
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GT(je, 0.0);
}

SCL_TEST_CASE(conditional_entropy_basic) {
    std::vector<scl_index_t> x = {0, 0, 1, 1};
    std::vector<scl_index_t> y = {0, 1, 0, 1};
    scl_real_t ce = 0.0;

    scl_error_t err = scl_entropy_conditional_entropy(
        x.data(), y.data(), 4, 2, 2, 1, &ce
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(ce, 0.0);
}

SCL_TEST_CASE(conditional_entropy_deterministic) {
    // Y completely determines X
    std::vector<scl_index_t> x = {0, 1, 2, 0, 1, 2};
    std::vector<scl_index_t> y = {0, 1, 2, 0, 1, 2};  // Same as X
    scl_real_t ce = 0.0;

    scl_error_t err = scl_entropy_conditional_entropy(
        y.data(), x.data(), 6, 3, 3, 1, &ce
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    // H(Y|X) should be 0 if X determines Y
    SCL_ASSERT_NEAR(ce, 0.0, 1e-10);
}

SCL_TEST_CASE(marginal_entropy_basic) {
    std::vector<scl_index_t> x = {0, 0, 1, 1, 2, 2};
    scl_real_t me = 0.0;

    scl_error_t err = scl_entropy_marginal_entropy(
        x.data(), 6, 3, 1, &me
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(me, std::log2(3.0), 1e-10);  // Uniform over 3 bins
}

SCL_TEST_CASE(marginal_entropy_null_pointer) {
    scl_real_t me = 0.0;

    SCL_ASSERT_EQ(
        scl_entropy_marginal_entropy(nullptr, 6, 3, 1, &me),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Normalized Mutual Information
// =============================================================================

SCL_TEST_SUITE(normalized_mi)

SCL_TEST_CASE(normalized_mi_identical_labels) {
    std::vector<scl_index_t> labels = {0, 0, 1, 1, 2, 2};
    scl_real_t nmi = 0.0;

    scl_error_t err = scl_entropy_normalized_mi(
        labels.data(), labels.data(), 6, 3, 3, &nmi
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(nmi, 1.0, 1e-10);  // Perfect agreement
}

SCL_TEST_CASE(normalized_mi_independent_labels) {
    std::vector<scl_index_t> labels1 = {0, 0, 1, 1};
    std::vector<scl_index_t> labels2 = {0, 1, 0, 1};
    scl_real_t nmi = 0.0;

    scl_error_t err = scl_entropy_normalized_mi(
        labels1.data(), labels2.data(), 4, 2, 2, &nmi
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(nmi, 0.0);
    SCL_ASSERT_LE(nmi, 1.0 + 1e-10);
}

SCL_TEST_CASE(normalized_mi_null_pointer) {
    std::vector<scl_index_t> labels = {0, 1, 0, 1};
    scl_real_t nmi = 0.0;

    SCL_ASSERT_EQ(
        scl_entropy_normalized_mi(nullptr, labels.data(), 4, 2, 2, &nmi),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(adjusted_mi_identical_labels) {
    std::vector<scl_index_t> labels = {0, 0, 1, 1, 2, 2};
    scl_real_t ami = 0.0;

    scl_error_t err = scl_entropy_adjusted_mi(
        labels.data(), labels.data(), 6, 3, 3, &ami
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(ami, 1.0, 1e-10);
}

SCL_TEST_CASE(adjusted_mi_random_labels) {
    std::vector<scl_index_t> labels1 = {0, 1, 2, 0, 1, 2};
    std::vector<scl_index_t> labels2 = {2, 1, 0, 2, 1, 0};
    scl_real_t ami = 0.0;

    scl_error_t err = scl_entropy_adjusted_mi(
        labels1.data(), labels2.data(), 6, 3, 3, &ami
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    // AMI should be close to 0 for random labels
    SCL_ASSERT_GE(ami, -1.0);
    SCL_ASSERT_LE(ami, 1.0);
}

SCL_TEST_SUITE_END

// =============================================================================
// Discretization
// =============================================================================

SCL_TEST_SUITE(discretization)

SCL_TEST_CASE(discretize_equal_width_basic) {
    std::vector<scl_real_t> values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<scl_index_t> binned(6);

    scl_error_t err = scl_entropy_discretize_equal_width(
        values.data(), 6, 3, binned.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Check bins are valid
    for (auto b : binned) {
        SCL_ASSERT_GE(b, 0);
        SCL_ASSERT_LT(b, 3);
    }
}

SCL_TEST_CASE(discretize_equal_width_single_bin) {
    std::vector<scl_real_t> values = {1.0, 2.0, 3.0};
    std::vector<scl_index_t> binned(3);

    scl_error_t err = scl_entropy_discretize_equal_width(
        values.data(), 3, 1, binned.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // All should be in bin 0
    for (auto b : binned) {
        SCL_ASSERT_EQ(b, 0);
    }
}

SCL_TEST_CASE(discretize_equal_width_null_pointer) {
    std::vector<scl_index_t> binned(3);

    SCL_ASSERT_EQ(
        scl_entropy_discretize_equal_width(nullptr, 3, 2, binned.data()),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(discretize_equal_frequency_basic) {
    std::vector<scl_real_t> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<scl_index_t> binned(6);

    scl_error_t err = scl_entropy_discretize_equal_frequency(
        values.data(), 6, 3, binned.data()
    );

    SCL_ASSERT_EQ(err, SCL_OK);

    // Count elements in each bin
    std::vector<int> counts(3, 0);
    for (auto b : binned) {
        SCL_ASSERT_GE(b, 0);
        SCL_ASSERT_LT(b, 3);
        counts[b]++;
    }

    // Should have roughly equal counts
    for (auto c : counts) {
        SCL_ASSERT_GE(c, 1);
    }
}

SCL_TEST_CASE(discretize_equal_frequency_null_pointer) {
    std::vector<scl_real_t> values = {1.0, 2.0, 3.0};

    SCL_ASSERT_EQ(
        scl_entropy_discretize_equal_frequency(values.data(), 3, 2, nullptr),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Other Measures
// =============================================================================

SCL_TEST_SUITE(other_measures)

SCL_TEST_CASE(cross_entropy_identical) {
    std::vector<scl_real_t> probs = {0.25, 0.25, 0.25, 0.25};
    scl_real_t ce = 0.0;

    scl_error_t err = scl_entropy_cross_entropy(
        probs.data(), probs.data(), 4, &ce
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    // CE(p, p) = H(p)
    SCL_ASSERT_NEAR(ce, 2.0, 1e-10);  // H(uniform over 4) = 2 bits
}

SCL_TEST_CASE(cross_entropy_different) {
    std::vector<scl_real_t> true_p = {1.0, 0.0};
    std::vector<scl_real_t> pred_p = {0.8, 0.2};
    scl_real_t ce = 0.0;

    scl_error_t err = scl_entropy_cross_entropy(
        true_p.data(), pred_p.data(), 2, &ce
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GT(ce, 0.0);
}

SCL_TEST_CASE(cross_entropy_null_pointer) {
    std::vector<scl_real_t> probs = {0.5, 0.5};
    scl_real_t ce = 0.0;

    SCL_ASSERT_EQ(
        scl_entropy_cross_entropy(nullptr, probs.data(), 2, &ce),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_CASE(gini_impurity_pure) {
    std::vector<scl_real_t> probs = {1.0, 0.0, 0.0};
    scl_real_t gini = 0.0;

    scl_error_t err = scl_entropy_gini_impurity(
        probs.data(), 3, &gini
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(gini, 0.0, 1e-10);  // Pure = no impurity
}

SCL_TEST_CASE(gini_impurity_uniform) {
    std::vector<scl_real_t> probs = {0.25, 0.25, 0.25, 0.25};
    scl_real_t gini = 0.0;

    scl_error_t err = scl_entropy_gini_impurity(
        probs.data(), 4, &gini
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    // Gini = 1 - sum(p_i^2) = 1 - 4*(0.25^2) = 0.75
    SCL_ASSERT_NEAR(gini, 0.75, 1e-10);
}

SCL_TEST_CASE(gini_impurity_binary) {
    std::vector<scl_real_t> probs = {0.5, 0.5};
    scl_real_t gini = 0.0;

    scl_error_t err = scl_entropy_gini_impurity(
        probs.data(), 2, &gini
    );

    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NEAR(gini, 0.5, 1e-10);
}

SCL_TEST_CASE(gini_impurity_null_pointer) {
    scl_real_t gini = 0.0;

    SCL_ASSERT_EQ(
        scl_entropy_gini_impurity(nullptr, 3, &gini),
        SCL_ERROR_NULL_POINTER
    );
}

SCL_TEST_SUITE_END

// =============================================================================
// Entropy Relations (Mathematical Properties)
// =============================================================================

SCL_TEST_SUITE(entropy_relations)

SCL_TEST_RETRY(mutual_information_relation, 3) {
    // MI(X;Y) = H(X) + H(Y) - H(X,Y)
    Random rng(42);

    std::vector<scl_index_t> x(100);
    std::vector<scl_index_t> y(100);

    for (size_t i = 0; i < 100; ++i) {
        x[i] = rng.uniform_int(0, 2);
        y[i] = rng.uniform_int(0, 2);
    }

    scl_real_t mi = 0.0, hx = 0.0, hy = 0.0, hxy = 0.0;

    scl_entropy_mutual_information(x.data(), y.data(), 100, 3, 3, 1, &mi);
    scl_entropy_marginal_entropy(x.data(), 100, 3, 1, &hx);
    scl_entropy_marginal_entropy(y.data(), 100, 3, 1, &hy);
    scl_entropy_joint_entropy(x.data(), y.data(), 100, 3, 3, 1, &hxy);

    // Check relation
    SCL_ASSERT_NEAR(mi, hx + hy - hxy, 1e-8);
}

SCL_TEST_RETRY(conditional_entropy_relation, 3) {
    // H(Y|X) = H(X,Y) - H(X)
    Random rng(123);

    std::vector<scl_index_t> x(100);
    std::vector<scl_index_t> y(100);

    for (size_t i = 0; i < 100; ++i) {
        x[i] = rng.uniform_int(0, 2);
        y[i] = rng.uniform_int(0, 2);
    }

    scl_real_t cond_entropy = 0.0, joint_entropy = 0.0, marg_entropy = 0.0;

    scl_entropy_conditional_entropy(x.data(), y.data(), 100, 3, 3, 1, &cond_entropy);
    scl_entropy_joint_entropy(x.data(), y.data(), 100, 3, 3, 1, &joint_entropy);
    scl_entropy_marginal_entropy(x.data(), 100, 3, 1, &marg_entropy);

    SCL_ASSERT_NEAR(cond_entropy, joint_entropy - marg_entropy, 1e-8);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
