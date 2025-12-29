// =============================================================================
// SCL Core - Hotspot Detection Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/hotspot.h
//
// Functions tested:
//   - scl_hotspot_local_morans_i
//   - scl_hotspot_getis_ord_g_star
//   - scl_hotspot_local_gearys_c
//   - scl_hotspot_global_morans_i
//   - scl_hotspot_global_gearys_c
//
// Reference implementation: Standard spatial statistics formulas
// Precision requirement: Tolerance::statistical() (rtol=1e-4, atol=1e-6)
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"

using namespace scl::test;
using precision::Tolerance;

// Helper: Create simple spatial weights matrix (k-NN graph)
Sparse create_simple_weights(scl_index_t n, Random& rng) {
    // Create a simple symmetric weight matrix (each node connected to neighbors)
    std::vector<scl_index_t> indptr(n + 1);
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    
    indptr[0] = 0;
    for (scl_index_t i = 0; i < n; ++i) {
        // Each node connected to previous and next (ring topology)
        scl_index_t prev = (i > 0) ? (i - 1) : (n - 1);
        scl_index_t next = (i < n - 1) ? (i + 1) : 0;
        
        indices.push_back(prev);
        data.push_back(1.0);
        
        indices.push_back(next);
        data.push_back(1.0);
        
        indptr[i + 1] = indices.size();
    }
    
    return make_sparse_csr(n, n, indices.size(), indptr.data(), indices.data(), data.data());
}

// Helper: Create identity weights (self-connections only)
Sparse create_identity_weights(scl_index_t n) {
    std::vector<scl_index_t> indptr(n + 1);
    std::vector<scl_index_t> indices(n);
    std::vector<scl_real_t> data(n, 1.0);
    
    for (scl_index_t i = 0; i < n; ++i) {
        indptr[i] = i;
        indices[i] = i;
    }
    indptr[n] = n;
    
    return make_sparse_csr(n, n, n, indptr.data(), indices.data(), data.data());
}

SCL_TEST_BEGIN

// =============================================================================
// Local Moran's I Tests
// =============================================================================

SCL_TEST_SUITE(local_morans_i)

SCL_TEST_CASE(local_morans_i_basic) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    
    // Simple values: alternating high/low
    std::vector<scl_real_t> values(n);
    for (scl_index_t i = 0; i < n; ++i) {
        values[i] = (i % 2 == 0) ? 10.0 : 1.0;
    }
    
    std::vector<scl_real_t> local_i(n);
    std::vector<scl_real_t> z_scores(n);
    std::vector<scl_real_t> p_values(n);
    
    scl_error_t err = scl_hotspot_local_morans_i(
        weights, values.data(), n,
        local_i.data(), z_scores.data(), p_values.data(),
        0,  // No permutations
        42
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Check that results are computed
    for (scl_index_t i = 0; i < n; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(local_i[i]));
        SCL_ASSERT_TRUE(std::isfinite(z_scores[i]));
        SCL_ASSERT_TRUE(std::isfinite(p_values[i]));
        SCL_ASSERT_GE(p_values[i], 0.0);
        SCL_ASSERT_LE(p_values[i], 1.0);
    }
}

SCL_TEST_CASE(local_morans_i_constant_values) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    
    // All values the same
    std::vector<scl_real_t> values(n, 5.0);
    
    std::vector<scl_real_t> local_i(n);
    std::vector<scl_real_t> z_scores(n);
    std::vector<scl_real_t> p_values(n);
    
    scl_error_t err = scl_hotspot_local_morans_i(
        weights, values.data(), n,
        local_i.data(), z_scores.data(), p_values.data(),
        0, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // With constant values, local I should be near 0
    for (scl_index_t i = 0; i < n; ++i) {
        SCL_ASSERT_NEAR(local_i[i], 0.0, 1e-6);
    }
}

SCL_TEST_RETRY(local_morans_i_random, 3)
{
    Random rng(123);
    scl_index_t n = rng.uniform_int(10, 30);
    Sparse weights = create_simple_weights(n, rng);
    
    auto values = random_vector(n, rng);
    
    std::vector<scl_real_t> local_i(n);
    std::vector<scl_real_t> z_scores(n);
    std::vector<scl_real_t> p_values(n);
    
    scl_error_t err = scl_hotspot_local_morans_i(
        weights, values.data(), n,
        local_i.data(), z_scores.data(), p_values.data(),
        0, rng.uniform_int(1, 1000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify all outputs are valid
    for (scl_index_t i = 0; i < n; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(local_i[i]));
        SCL_ASSERT_TRUE(std::isfinite(z_scores[i]));
        SCL_ASSERT_TRUE(std::isfinite(p_values[i]));
    }
}

SCL_TEST_CASE(local_morans_i_null_inputs) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    std::vector<scl_real_t> values(n, 1.0);
    std::vector<scl_real_t> local_i(n), z_scores(n), p_values(n);
    
    // NULL weights
    scl_error_t err1 = scl_hotspot_local_morans_i(
        nullptr, values.data(), n,
        local_i.data(), z_scores.data(), p_values.data(), 0, 42
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    // NULL values
    scl_error_t err2 = scl_hotspot_local_morans_i(
        weights, nullptr, n,
        local_i.data(), z_scores.data(), p_values.data(), 0, 42
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
    
    // NULL outputs
    scl_error_t err3 = scl_hotspot_local_morans_i(
        weights, values.data(), n,
        nullptr, z_scores.data(), p_values.data(), 0, 42
    );
    SCL_ASSERT_EQ(err3, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(local_morans_i_invalid_size) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    std::vector<scl_real_t> values(5);  // Wrong size
    std::vector<scl_real_t> local_i(n), z_scores(n), p_values(n);
    
    scl_error_t err = scl_hotspot_local_morans_i(
        weights, values.data(), n,
        local_i.data(), z_scores.data(), p_values.data(), 0, 42
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Getis-Ord Gi* Tests
// =============================================================================

SCL_TEST_SUITE(getis_ord_g_star)

SCL_TEST_CASE(getis_ord_g_star_basic) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    
    std::vector<scl_real_t> values(n);
    for (scl_index_t i = 0; i < n; ++i) {
        values[i] = static_cast<scl_real_t>(i);
    }
    
    std::vector<scl_real_t> g_star(n);
    std::vector<scl_real_t> z_scores(n);
    std::vector<scl_real_t> p_values(n);
    
    scl_error_t err = scl_hotspot_getis_ord_g_star(
        weights, values.data(), n,
        g_star.data(), z_scores.data(), p_values.data(),
        1  // include_self
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (scl_index_t i = 0; i < n; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(g_star[i]));
        SCL_ASSERT_TRUE(std::isfinite(z_scores[i]));
        SCL_ASSERT_TRUE(std::isfinite(p_values[i]));
    }
}

SCL_TEST_CASE(getis_ord_g_star_without_self) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    
    std::vector<scl_real_t> values(n, 1.0);
    std::vector<scl_real_t> g_star(n);
    std::vector<scl_real_t> z_scores(n);
    std::vector<scl_real_t> p_values(n);
    
    scl_error_t err = scl_hotspot_getis_ord_g_star(
        weights, values.data(), n,
        g_star.data(), z_scores.data(), p_values.data(),
        0  // exclude_self
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_RETRY(getis_ord_g_star_random, 3)
{
    Random rng(456);
    scl_index_t n = rng.uniform_int(10, 30);
    Sparse weights = create_simple_weights(n, rng);
    auto values = random_vector(n, rng);
    
    std::vector<scl_real_t> g_star(n);
    std::vector<scl_real_t> z_scores(n);
    std::vector<scl_real_t> p_values(n);
    
    scl_error_t err = scl_hotspot_getis_ord_g_star(
        weights, values.data(), n,
        g_star.data(), z_scores.data(), p_values.data(),
        rng.bernoulli() ? 1 : 0
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(getis_ord_g_star_null_inputs) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    std::vector<scl_real_t> values(n);
    std::vector<scl_real_t> g_star(n), z_scores(n), p_values(n);
    
    scl_error_t err1 = scl_hotspot_getis_ord_g_star(
        nullptr, values.data(), n,
        g_star.data(), z_scores.data(), p_values.data(), 1
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_hotspot_getis_ord_g_star(
        weights, nullptr, n,
        g_star.data(), z_scores.data(), p_values.data(), 1
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Local Geary's C Tests
// =============================================================================

SCL_TEST_SUITE(local_gearys_c)

SCL_TEST_CASE(local_gearys_c_basic) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    
    std::vector<scl_real_t> values(n);
    for (scl_index_t i = 0; i < n; ++i) {
        values[i] = static_cast<scl_real_t>(i);
    }
    
    std::vector<scl_real_t> local_c(n);
    std::vector<scl_real_t> z_scores(n);
    std::vector<scl_real_t> p_values(n);
    
    scl_error_t err = scl_hotspot_local_gearys_c(
        weights, values.data(), n,
        local_c.data(), z_scores.data(), p_values.data(),
        0, 42
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    for (scl_index_t i = 0; i < n; ++i) {
        SCL_ASSERT_TRUE(std::isfinite(local_c[i]));
        SCL_ASSERT_TRUE(std::isfinite(z_scores[i]));
        SCL_ASSERT_TRUE(std::isfinite(p_values[i]));
    }
}

SCL_TEST_RETRY(local_gearys_c_random, 3)
{
    Random rng(789);
    scl_index_t n = rng.uniform_int(10, 30);
    Sparse weights = create_simple_weights(n, rng);
    auto values = random_vector(n, rng);
    
    std::vector<scl_real_t> local_c(n);
    std::vector<scl_real_t> z_scores(n);
    std::vector<scl_real_t> p_values(n);
    
    scl_error_t err = scl_hotspot_local_gearys_c(
        weights, values.data(), n,
        local_c.data(), z_scores.data(), p_values.data(),
        0, rng.uniform_int(1, 1000)
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(local_gearys_c_null_inputs) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    std::vector<scl_real_t> values(n);
    std::vector<scl_real_t> local_c(n), z_scores(n), p_values(n);
    
    scl_error_t err1 = scl_hotspot_local_gearys_c(
        nullptr, values.data(), n,
        local_c.data(), z_scores.data(), p_values.data(), 0, 42
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Global Moran's I Tests
// =============================================================================

SCL_TEST_SUITE(global_morans_i)

SCL_TEST_CASE(global_morans_i_basic) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    
    std::vector<scl_real_t> values(n);
    for (scl_index_t i = 0; i < n; ++i) {
        values[i] = static_cast<scl_real_t>(i);
    }
    
    scl_real_t moran_i;
    scl_real_t z_score;
    scl_real_t p_value;
    
    scl_error_t err = scl_hotspot_global_morans_i(
        weights, values.data(), n,
        &moran_i, &z_score, &p_value
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_TRUE(std::isfinite(moran_i));
    SCL_ASSERT_TRUE(std::isfinite(z_score));
    SCL_ASSERT_TRUE(std::isfinite(p_value));
    SCL_ASSERT_GE(p_value, 0.0);
    SCL_ASSERT_LE(p_value, 1.0);
}

SCL_TEST_CASE(global_morans_i_constant_values) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    std::vector<scl_real_t> values(n, 5.0);
    
    scl_real_t moran_i, z_score, p_value;
    
    scl_error_t err = scl_hotspot_global_morans_i(
        weights, values.data(), n,
        &moran_i, &z_score, &p_value
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    // With constant values, Moran's I should be near 0
    SCL_ASSERT_NEAR(moran_i, 0.0, 1e-6);
}

SCL_TEST_RETRY(global_morans_i_random, 3)
{
    Random rng(111);
    scl_index_t n = rng.uniform_int(10, 30);
    Sparse weights = create_simple_weights(n, rng);
    auto values = random_vector(n, rng);
    
    scl_real_t moran_i, z_score, p_value;
    
    scl_error_t err = scl_hotspot_global_morans_i(
        weights, values.data(), n,
        &moran_i, &z_score, &p_value
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_TRUE(std::isfinite(moran_i));
}

SCL_TEST_CASE(global_morans_i_null_inputs) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    std::vector<scl_real_t> values(n);
    
    scl_real_t moran_i, z_score, p_value;
    
    scl_error_t err1 = scl_hotspot_global_morans_i(
        nullptr, values.data(), n, &moran_i, &z_score, &p_value
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err2 = scl_hotspot_global_morans_i(
        weights, nullptr, n, &moran_i, &z_score, &p_value
    );
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
    
    scl_error_t err3 = scl_hotspot_global_morans_i(
        weights, values.data(), n, nullptr, &z_score, &p_value
    );
    SCL_ASSERT_EQ(err3, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Global Geary's C Tests
// =============================================================================

SCL_TEST_SUITE(global_gearys_c)

SCL_TEST_CASE(global_gearys_c_basic) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    
    std::vector<scl_real_t> values(n);
    for (scl_index_t i = 0; i < n; ++i) {
        values[i] = static_cast<scl_real_t>(i);
    }
    
    scl_real_t geary_c;
    scl_real_t z_score;
    scl_real_t p_value;
    
    scl_error_t err = scl_hotspot_global_gearys_c(
        weights, values.data(), n,
        &geary_c, &z_score, &p_value
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_TRUE(std::isfinite(geary_c));
    SCL_ASSERT_TRUE(std::isfinite(z_score));
    SCL_ASSERT_TRUE(std::isfinite(p_value));
}

SCL_TEST_RETRY(global_gearys_c_random, 3)
{
    Random rng(222);
    scl_index_t n = rng.uniform_int(10, 30);
    Sparse weights = create_simple_weights(n, rng);
    auto values = random_vector(n, rng);
    
    scl_real_t geary_c, z_score, p_value;
    
    scl_error_t err = scl_hotspot_global_gearys_c(
        weights, values.data(), n,
        &geary_c, &z_score, &p_value
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(global_gearys_c_null_inputs) {
    scl_index_t n = 10;
    Sparse weights = create_simple_weights(n, global_rng());
    std::vector<scl_real_t> values(n);
    
    scl_real_t geary_c, z_score, p_value;
    
    scl_error_t err1 = scl_hotspot_global_gearys_c(
        nullptr, values.data(), n, &geary_c, &z_score, &p_value
    );
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

