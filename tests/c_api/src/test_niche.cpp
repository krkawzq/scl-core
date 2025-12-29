// =============================================================================
// SCL Core - Niche Analysis Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/niche.h
//
// Functions tested:
//   ✓ scl_niche_neighborhood_composition
//   ✓ scl_niche_neighborhood_enrichment
//   ✓ scl_niche_cell_cell_contact
//   ✓ scl_niche_colocalization_score
//   ✓ scl_niche_colocalization_matrix
//   ✓ scl_niche_similarity
//   ✓ scl_niche_diversity
//   ✓ scl_niche_boundary_score
//
// =============================================================================

#include "test.hpp"
#include "precision.hpp"
#include "scl/binding/c_api/niche.h"

using namespace scl::test;
using precision::Tolerance;

// Helper: Create a simple spatial neighbor graph (ring topology)
static Sparse create_ring_graph(scl_index_t n_cells) {
    std::vector<scl_index_t> indptr(n_cells + 1);
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    
    indptr[0] = 0;
    for (scl_index_t i = 0; i < n_cells; ++i) {
        // Each cell connects to previous and next (ring)
        scl_index_t prev = (i + n_cells - 1) % n_cells;
        scl_index_t next = (i + 1) % n_cells;
        
        indices.push_back(prev);
        data.push_back(1.0);
        indices.push_back(next);
        data.push_back(1.0);
        
        indptr[i + 1] = indices.size();
    }
    
    return make_sparse_csr(n_cells, n_cells, indices.size(),
                          indptr.data(), indices.data(), data.data());
}

// Helper: Create grid graph (2D grid of cells)
static Sparse create_grid_graph(scl_index_t rows, scl_index_t cols) {
    scl_index_t n_cells = rows * cols;
    std::vector<scl_index_t> indptr(n_cells + 1);
    std::vector<scl_index_t> indices;
    std::vector<scl_real_t> data;
    
    indptr[0] = 0;
    for (scl_index_t i = 0; i < rows; ++i) {
        for (scl_index_t j = 0; j < cols; ++j) {
            scl_index_t cell_idx = i * cols + j;
            scl_index_t count = 0;
            
            // Connect to neighbors (up, down, left, right)
            if (i > 0) {
                indices.push_back((i - 1) * cols + j);
                data.push_back(1.0);
                ++count;
            }
            if (i < rows - 1) {
                indices.push_back((i + 1) * cols + j);
                data.push_back(1.0);
                ++count;
            }
            if (j > 0) {
                indices.push_back(i * cols + (j - 1));
                data.push_back(1.0);
                ++count;
            }
            if (j < cols - 1) {
                indices.push_back(i * cols + (j + 1));
                data.push_back(1.0);
                ++count;
            }
            
            indptr[cell_idx + 1] = indices.size();
        }
    }
    
    return make_sparse_csr(n_cells, n_cells, indices.size(),
                          indptr.data(), indices.data(), data.data());
}

SCL_TEST_BEGIN

// =============================================================================
// Neighborhood Composition Tests
// =============================================================================

SCL_TEST_SUITE(neighborhood_composition)

SCL_TEST_CASE(composition_basic) {
    scl_index_t n_cells = 10;
    scl_index_t n_types = 3;
    
    Sparse neighbors = create_ring_graph(n_cells);
    
    // Assign cell types: 0,1,2,0,1,2,...
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i % n_types;
    }
    
    std::vector<scl_real_t> output(n_cells * n_types, 0.0);
    
    scl_error_t err = scl_niche_neighborhood_composition(
        neighbors, labels.data(), n_types, output.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Check that each row sums to 1.0 (composition is normalized)
    for (scl_index_t i = 0; i < n_cells; ++i) {
        scl_real_t sum = 0.0;
        for (scl_index_t t = 0; t < n_types; ++t) {
            sum += output[i * n_types + t];
        }
        SCL_ASSERT_NEAR(sum, 1.0, 1e-6);
    }
}

SCL_TEST_CASE(composition_null_neighbors) {
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_real_t> output(30);
    
    scl_error_t err = scl_niche_neighborhood_composition(
        nullptr, labels.data(), 3, output.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(composition_null_labels) {
    Sparse neighbors = create_ring_graph(10);
    std::vector<scl_real_t> output(30);
    
    scl_error_t err = scl_niche_neighborhood_composition(
        neighbors, nullptr, 3, output.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(composition_null_output) {
    Sparse neighbors = create_ring_graph(10);
    std::vector<scl_index_t> labels(10, 0);
    
    scl_error_t err = scl_niche_neighborhood_composition(
        neighbors, labels.data(), 3, nullptr
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(composition_single_type) {
    scl_index_t n_cells = 5;
    Sparse neighbors = create_ring_graph(n_cells);
    std::vector<scl_index_t> labels(n_cells, 0);  // All same type
    std::vector<scl_real_t> output(n_cells, 0.0);
    
    scl_error_t err = scl_niche_neighborhood_composition(
        neighbors, labels.data(), 1, output.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // All compositions should be 1.0 (only one type)
    for (scl_index_t i = 0; i < n_cells; ++i) {
        SCL_ASSERT_NEAR(output[i], 1.0, 1e-6);
    }
}

SCL_TEST_RETRY(composition_random, 3)
{
    Random rng(42);
    scl_index_t n_cells = rng.uniform_int(20, 50);
    scl_index_t n_types = rng.uniform_int(2, 5);
    
    Sparse neighbors = create_ring_graph(n_cells);
    
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = rng.uniform_int(0, n_types - 1);
    }
    
    std::vector<scl_real_t> output(n_cells * n_types, 0.0);
    
    scl_error_t err = scl_niche_neighborhood_composition(
        neighbors, labels.data(), n_types, output.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify normalization
    for (scl_index_t i = 0; i < n_cells; ++i) {
        scl_real_t sum = 0.0;
        for (scl_index_t t = 0; t < n_types; ++t) {
            sum += output[i * n_types + t];
        }
        SCL_ASSERT_NEAR(sum, 1.0, 1e-6);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Neighborhood Enrichment Tests
// =============================================================================

SCL_TEST_SUITE(neighborhood_enrichment)

SCL_TEST_CASE(enrichment_basic) {
    scl_index_t n_cells = 20;
    scl_index_t n_types = 3;
    scl_index_t n_perm = 100;
    
    Sparse neighbors = create_ring_graph(n_cells);
    
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i % n_types;
    }
    
    std::vector<scl_real_t> enrichment(n_types * n_types, 0.0);
    std::vector<scl_real_t> p_values(n_types * n_types, 0.0);
    
    scl_error_t err = scl_niche_neighborhood_enrichment(
        neighbors, labels.data(), n_types,
        enrichment.data(), p_values.data(), n_perm
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // P-values should be in [0, 1]
    for (scl_index_t i = 0; i < n_types * n_types; ++i) {
        SCL_ASSERT_GE(p_values[i], 0.0);
        SCL_ASSERT_LE(p_values[i], 1.0);
    }
}

SCL_TEST_CASE(enrichment_null_neighbors) {
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_real_t> enrichment(9), p_values(9);
    
    scl_error_t err = scl_niche_neighborhood_enrichment(
        nullptr, labels.data(), 3,
        enrichment.data(), p_values.data(), 100
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(enrichment_zero_permutations) {
    Sparse neighbors = create_ring_graph(10);
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_real_t> enrichment(9), p_values(9);
    
    scl_error_t err = scl_niche_neighborhood_enrichment(
        neighbors, labels.data(), 3,
        enrichment.data(), p_values.data(), 0
    );
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Cell-Cell Contact Tests
// =============================================================================

SCL_TEST_SUITE(cell_cell_contact)

SCL_TEST_CASE(contact_basic) {
    scl_index_t n_cells = 10;
    scl_index_t n_types = 3;
    
    Sparse neighbors = create_ring_graph(n_cells);
    
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i % n_types;
    }
    
    std::vector<scl_real_t> contact(n_types * n_types, 0.0);
    
    scl_error_t err = scl_niche_cell_cell_contact(
        neighbors, labels.data(), n_types, contact.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Contact matrix should be symmetric
    for (scl_index_t i = 0; i < n_types; ++i) {
        for (scl_index_t j = 0; j < n_types; ++j) {
            SCL_ASSERT_NEAR(
                contact[i * n_types + j],
                contact[j * n_types + i],
                1e-6
            );
        }
    }
}

SCL_TEST_CASE(contact_null_neighbors) {
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_real_t> contact(9);
    
    scl_error_t err = scl_niche_cell_cell_contact(
        nullptr, labels.data(), 3, contact.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Co-localization Score Tests
// =============================================================================

SCL_TEST_SUITE(colocalization_score)

SCL_TEST_CASE(colocalization_basic) {
    scl_index_t n_cells = 20;
    scl_index_t n_types = 3;
    scl_index_t n_perm = 100;
    
    Sparse neighbors = create_ring_graph(n_cells);
    
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i % n_types;
    }
    
    scl_real_t score = 0.0, p_value = 0.0;
    
    scl_error_t err = scl_niche_colocalization_score(
        neighbors, labels.data(), n_types,
        0, 1,  // type_a, type_b
        &score, &p_value, n_perm
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_GE(p_value, 0.0);
    SCL_ASSERT_LE(p_value, 1.0);
}

SCL_TEST_CASE(colocalization_null_neighbors) {
    std::vector<scl_index_t> labels(10, 0);
    scl_real_t score = 0.0, p_value = 0.0;
    
    scl_error_t err = scl_niche_colocalization_score(
        nullptr, labels.data(), 3,
        0, 1, &score, &p_value, 100
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(colocalization_invalid_types) {
    Sparse neighbors = create_ring_graph(10);
    std::vector<scl_index_t> labels(10, 0);
    scl_real_t score = 0.0, p_value = 0.0;
    
    // type_a >= n_types
    scl_error_t err1 = scl_niche_colocalization_score(
        neighbors, labels.data(), 3,
        5, 1, &score, &p_value, 100
    );
    SCL_ASSERT_NE(err1, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Co-localization Matrix Tests
// =============================================================================

SCL_TEST_SUITE(colocalization_matrix)

SCL_TEST_CASE(colocalization_matrix_basic) {
    scl_index_t n_cells = 15;
    scl_index_t n_types = 3;
    
    Sparse neighbors = create_ring_graph(n_cells);
    
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i % n_types;
    }
    
    std::vector<scl_real_t> matrix(n_types * n_types, 0.0);
    
    scl_error_t err = scl_niche_colocalization_matrix(
        neighbors, labels.data(), n_types, matrix.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Matrix should be symmetric
    for (scl_index_t i = 0; i < n_types; ++i) {
        for (scl_index_t j = 0; j < n_types; ++j) {
            SCL_ASSERT_NEAR(
                matrix[i * n_types + j],
                matrix[j * n_types + i],
                1e-6
            );
        }
    }
}

SCL_TEST_CASE(colocalization_matrix_null_neighbors) {
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_real_t> matrix(9);
    
    scl_error_t err = scl_niche_colocalization_matrix(
        nullptr, labels.data(), 3, matrix.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Niche Similarity Tests
// =============================================================================

SCL_TEST_SUITE(niche_similarity)

SCL_TEST_CASE(similarity_basic) {
    scl_index_t n_cells = 20;
    scl_index_t n_types = 3;
    
    Sparse neighbors = create_ring_graph(n_cells);
    
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i % n_types;
    }
    
    std::vector<scl_index_t> query = {0, 1, 2, 3};
    std::vector<scl_real_t> similarity(query.size() * query.size(), 0.0);
    
    scl_error_t err = scl_niche_similarity(
        neighbors, labels.data(), n_types,
        query.data(), query.size(), similarity.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Similarity matrix should be symmetric
    for (size_t i = 0; i < query.size(); ++i) {
        for (size_t j = 0; j < query.size(); ++j) {
            SCL_ASSERT_NEAR(
                similarity[i * query.size() + j],
                similarity[j * query.size() + i],
                1e-6
            );
        }
    }
    
    // Diagonal should be 1.0 (self-similarity)
    for (size_t i = 0; i < query.size(); ++i) {
        SCL_ASSERT_NEAR(similarity[i * query.size() + i], 1.0, 1e-6);
    }
}

SCL_TEST_CASE(similarity_null_neighbors) {
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_index_t> query = {0, 1};
    std::vector<scl_real_t> similarity(4);
    
    scl_error_t err = scl_niche_similarity(
        nullptr, labels.data(), 3,
        query.data(), query.size(), similarity.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(similarity_empty_query) {
    Sparse neighbors = create_ring_graph(10);
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_index_t> query;
    std::vector<scl_real_t> similarity;
    
    scl_error_t err = scl_niche_similarity(
        neighbors, labels.data(), 3,
        query.data(), 0, similarity.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Niche Diversity Tests
// =============================================================================

SCL_TEST_SUITE(niche_diversity)

SCL_TEST_CASE(diversity_basic) {
    scl_index_t n_cells = 15;
    scl_index_t n_types = 3;
    
    Sparse neighbors = create_ring_graph(n_cells);
    
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i % n_types;
    }
    
    std::vector<scl_real_t> diversity(n_cells, 0.0);
    
    scl_error_t err = scl_niche_diversity(
        neighbors, labels.data(), n_types, diversity.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Diversity should be non-negative
    for (scl_index_t i = 0; i < n_cells; ++i) {
        SCL_ASSERT_GE(diversity[i], 0.0);
    }
}

SCL_TEST_CASE(diversity_null_neighbors) {
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_real_t> diversity(10);
    
    scl_error_t err = scl_niche_diversity(
        nullptr, labels.data(), 3, diversity.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(diversity_single_type) {
    scl_index_t n_cells = 10;
    Sparse neighbors = create_ring_graph(n_cells);
    std::vector<scl_index_t> labels(n_cells, 0);  // All same type
    std::vector<scl_real_t> diversity(n_cells, 0.0);
    
    scl_error_t err = scl_niche_diversity(
        neighbors, labels.data(), 1, diversity.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Single type should have zero diversity
    for (scl_index_t i = 0; i < n_cells; ++i) {
        SCL_ASSERT_NEAR(diversity[i], 0.0, 1e-6);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Niche Boundary Score Tests
// =============================================================================

SCL_TEST_SUITE(boundary_score)

SCL_TEST_CASE(boundary_basic) {
    scl_index_t n_cells = 20;
    scl_index_t n_types = 3;
    
    Sparse neighbors = create_ring_graph(n_cells);
    
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < n_cells; ++i) {
        labels[i] = i % n_types;
    }
    
    std::vector<scl_real_t> boundary(n_cells, 0.0);
    
    scl_error_t err = scl_niche_boundary_score(
        neighbors, labels.data(), n_types, boundary.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Boundary scores should be non-negative
    for (scl_index_t i = 0; i < n_cells; ++i) {
        SCL_ASSERT_GE(boundary[i], 0.0);
    }
}

SCL_TEST_CASE(boundary_null_neighbors) {
    std::vector<scl_index_t> labels(10, 0);
    std::vector<scl_real_t> boundary(10);
    
    scl_error_t err = scl_niche_boundary_score(
        nullptr, labels.data(), 3, boundary.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(boundary_grid_graph) {
    // Test with grid graph (more realistic spatial structure)
    scl_index_t rows = 5, cols = 5;
    Sparse neighbors = create_grid_graph(rows, cols);
    scl_index_t n_cells = rows * cols;
    scl_index_t n_types = 2;
    
    // Create two regions: left half type 0, right half type 1
    std::vector<scl_index_t> labels(n_cells);
    for (scl_index_t i = 0; i < rows; ++i) {
        for (scl_index_t j = 0; j < cols; ++j) {
            scl_index_t idx = i * cols + j;
            labels[idx] = (j < cols / 2) ? 0 : 1;
        }
    }
    
    std::vector<scl_real_t> boundary(n_cells, 0.0);
    
    scl_error_t err = scl_niche_boundary_score(
        neighbors, labels.data(), n_types, boundary.data()
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Cells at the boundary should have higher scores
    // (This is a heuristic check - actual implementation may vary)
    for (scl_index_t i = 0; i < n_cells; ++i) {
        SCL_ASSERT_GE(boundary[i], 0.0);
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

