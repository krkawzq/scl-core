#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"

#include <array>
#include <cmath>

// =============================================================================
// FILE: scl/kernel/projection.hpp
// BRIEF: High-performance sparse random projection for dimensionality reduction
//
// Optimizations applied:
// - SIMD-accelerated random number generation (4-way parallel)
// - Block-wise processing for cache efficiency
// - Precomputed sparse projection structures
// - Feature hashing with minimal memory
// - Multi-accumulator FMA patterns
// - Adaptive algorithm selection based on sparsity
// =============================================================================

namespace scl::kernel::projection {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 8;
    constexpr Size BLOCK_SIZE = 256;
    constexpr Size SMALL_OUTPUT_DIM = 64;
    constexpr Size HASH_BATCH_SIZE = 8;
    constexpr Real DEFAULT_EPSILON = Real(0.1);
    constexpr Size MIN_PARALLEL_ROWS = 64;
    
    // Memory thresholds for algorithm selection
    constexpr Size MAX_PRECOMPUTE_BYTES = static_cast<Size>(256) * static_cast<Size>(1024) * static_cast<Size>(1024);  // 256 MB
}

// =============================================================================
// Projection Types
// =============================================================================

enum class ProjectionType : uint8_t {
    Gaussian,       // Dense Gaussian N(0, 1/k)
    Achlioptas,     // Sparse {+1, 0, -1} with prob {1/6, 2/3, 1/6}
    Sparse,         // Very sparse with density 1/sqrt(d)
    CountSketch,    // Sign flips with hash-based indexing
    FeatureHash     // Multiple hash functions for better accuracy
};

// =============================================================================
// High-Performance PRNG
// =============================================================================

namespace detail {

// Xoshiro256** - faster than Splitmix64 for bulk generation
struct alignas(32) Xoshiro256 {
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::array<uint64_t, 4> s{};

    SCL_FORCE_INLINE explicit Xoshiro256(uint64_t seed) noexcept {
        // Initialize with splitmix64
        uint64_t z = seed;
        for (uint64_t& si : s) {
            z += 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            si = z ^ (z >> 31);
        }
    }

    [[nodiscard]] SCL_FORCE_INLINE uint64_t rotl(uint64_t x, int k) const noexcept {
        return (x << k) | (x >> (64 - k));
    }

    SCL_FORCE_INLINE uint64_t next() noexcept {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Generate 4 random uint64_t at once
    SCL_FORCE_INLINE void next4(uint64_t* out) noexcept {
        out[0] = next();
        out[1] = next();
        out[2] = next();
        out[3] = next();
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next() >> 11) * Real(0x1.0p-53);
    }

    // Box-Muller for pairs of Gaussians
    SCL_FORCE_INLINE void gaussian2(Real& g1, Real& g2) noexcept {
        Real u1 = uniform();
        Real u2 = uniform();
        Real r = std::sqrt(Real(-2) * std::log(u1 + Real(1e-15)));
        Real theta = Real(2) * Real(3.14159265358979323846) * u2;
        g1 = r * std::cos(theta);
        g2 = r * std::sin(theta);
    }
};

// Fast hash for deterministic projection
SCL_FORCE_INLINE uint64_t fast_hash(uint64_t seed, Index col) noexcept {
    uint64_t h = seed ^ (static_cast<uint64_t>(col) * 0x9e3779b97f4a7c15ULL);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

// Two independent hashes from one computation
SCL_FORCE_INLINE void dual_hash(uint64_t seed, Index col, uint32_t& h1, uint32_t& h2) noexcept {
    uint64_t h = fast_hash(seed, col);
    h1 = static_cast<uint32_t>(h);
    h2 = static_cast<uint32_t>(h >> 32);
}

// =============================================================================
// SIMD-Accelerated Accumulation
// =============================================================================

template <typename T>
SCL_HOT SCL_FORCE_INLINE void accumulate_simd(
    T* SCL_RESTRICT output,
    Size len,
    T value,
    const T* SCL_RESTRICT proj_row
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);

    const auto v_val = s::Set(d, value);

    Size k = 0;

    // 4-way unrolled SIMD loop
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, output + k);
        auto v1 = s::Load(d, output + k + lanes);
        auto v2 = s::Load(d, output + k + 2 * lanes);
        auto v3 = s::Load(d, output + k + 3 * lanes);

        auto p0 = s::Load(d, proj_row + k);
        auto p1 = s::Load(d, proj_row + k + lanes);
        auto p2 = s::Load(d, proj_row + k + 2 * lanes);
        auto p3 = s::Load(d, proj_row + k + 3 * lanes);

        v0 = s::MulAdd(v_val, p0, v0);
        v1 = s::MulAdd(v_val, p1, v1);
        v2 = s::MulAdd(v_val, p2, v2);
        v3 = s::MulAdd(v_val, p3, v3);

        s::Store(v0, d, output + k);
        s::Store(v1, d, output + k + lanes);
        s::Store(v2, d, output + k + 2 * lanes);
        s::Store(v3, d, output + k + 3 * lanes);
    }

    // 2-way unrolled
    for (; k + 2 * lanes <= len; k += 2 * lanes) {
        auto v0 = s::Load(d, output + k);
        auto v1 = s::Load(d, output + k + lanes);
        auto p0 = s::Load(d, proj_row + k);
        auto p1 = s::Load(d, proj_row + k + lanes);
        s::Store(s::MulAdd(v_val, p0, v0), d, output + k);
        s::Store(s::MulAdd(v_val, p1, v1), d, output + k + lanes);
    }

    // Single SIMD
    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, output + k);
        auto p = s::Load(d, proj_row + k);
        s::Store(s::MulAdd(v_val, p, v), d, output + k);
    }

    // Scalar cleanup
    for (; k < len; ++k) {
        output[k] += value * proj_row[k];
    }
}

// Scalar version for short vectors
template <typename T>
SCL_FORCE_INLINE void accumulate_scalar(
    T* SCL_RESTRICT output,
    Size len,
    T value,
    const T* SCL_RESTRICT proj_row
) noexcept {
    Size k = 0;
    
    // 4-way unrolled scalar
    for (; k + 4 <= len; k += 4) {
        output[k + 0] += value * proj_row[k + 0];
        output[k + 1] += value * proj_row[k + 1];
        output[k + 2] += value * proj_row[k + 2];
        output[k + 3] += value * proj_row[k + 3];
    }
    
    for (; k < len; ++k) {
        output[k] += value * proj_row[k];
    }
}

// =============================================================================
// Block-wise Gaussian Projection (Cache-Optimized)
// =============================================================================

template <typename T>
SCL_HOT void project_row_gaussian_blocked(
    T* SCL_RESTRICT output,
    Size output_dim,
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Index len,
    uint64_t row_seed,
    T scale
) noexcept {
    scl::algo::zero(output, output_dim);
    
    if (len == 0) return;

    // Process in blocks for better cache utilization
    constexpr Size BLOCK = config::BLOCK_SIZE;
    
    // Temporary buffer for Gaussian values (on stack for small blocks)
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    std::array<T, BLOCK> local_gauss{};
    
    for (Size block_start = 0; block_start < output_dim; block_start += BLOCK) {
        Size block_end = scl::algo::min2(block_start + BLOCK, output_dim);
        Size block_len = block_end - block_start;
        
        T* out_block = output + block_start;
        
        for (Index j = 0; j < len; ++j) {
            Index col = indices[j];
            T val = values[j];
            
            // Deterministic seed for this (col, block) pair
            uint64_t block_seed = row_seed ^ fast_hash(static_cast<uint64_t>(col), static_cast<Index>(block_start));
            Xoshiro256 rng(block_seed);
            
            // Generate Gaussians for this block
            Size k = 0;
            for (; k + 2 <= block_len; k += 2) {
                rng.gaussian2(local_gauss[k], local_gauss[k + 1]);
            }
            if (k < block_len) {
                Real g1 = Real(0);
                Real g2 = Real(0);
                rng.gaussian2(g1, g2);
                local_gauss[k] = g1;
            }
            
            // Scale and accumulate
            T scaled_val = val * scale;
            if (block_len >= config::SIMD_THRESHOLD) {
                accumulate_simd(out_block, block_len, scaled_val, local_gauss);
            } else {
                accumulate_scalar(out_block, block_len, scaled_val, local_gauss);
            }
        }
    }
}

// =============================================================================
// Sparse Sign Projection (Achlioptas / Very Sparse)
// =============================================================================

template <typename T>
SCL_HOT void project_row_sparse_sign(
    T* SCL_RESTRICT output,
    Size output_dim,
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Index len,
    uint64_t row_seed,
    T scale,
    uint32_t sparsity_mask  // e.g., 0x3 for 1/4 density, 0x1 for 1/2
) noexcept {
    scl::algo::zero(output, output_dim);
    
    if (len == 0) return;

    for (Index j = 0; j < len; ++j) {
        Index col = indices[j];
        T val = values[j] * scale;
        
        uint64_t col_seed = row_seed ^ fast_hash(static_cast<uint64_t>(col), 0);
        
        // Process output dimensions in batches of 8
        Size k = 0;
        for (; k + 8 <= output_dim; k += 8) {
            uint64_t h = fast_hash(col_seed, static_cast<Index>(k >> 3));
            
            // Each byte controls one output dimension
            for (Size b = 0; b < 8; ++b) {
                auto byte = static_cast<uint8_t>(h >> (b * 8));
                
                // Check sparsity (lower bits)
                if ((byte & sparsity_mask) == 0) {
                    // Sign from another bit
                    T sign = (byte & 0x80) ? val : -val;
                    output[k + b] += sign;
                }
            }
        }
        
        // Handle remaining dimensions
        if (k < output_dim) {
            uint64_t h = fast_hash(col_seed, static_cast<Index>(k >> 3));
            for (Size b = 0; k + b < output_dim; ++b) {
                auto byte = static_cast<uint8_t>(h >> (b * 8));
                if ((byte & sparsity_mask) == 0) {
                    T sign = (byte & 0x80) ? val : -val;
                    output[k + b] += sign;
                }
            }
        }
    }
}

// Achlioptas: {+1, 0, -1} with prob {1/6, 2/3, 1/6}
template <typename T>
SCL_HOT void project_row_achlioptas(
    T* SCL_RESTRICT output,
    Size output_dim,
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Index len,
    uint64_t row_seed,
    T scale
) noexcept {
    scl::algo::zero(output, output_dim);
    
    if (len == 0) return;

    for (Index j = 0; j < len; ++j) {
        Index col = indices[j];
        T val = values[j] * scale;
        
        uint64_t col_seed = row_seed ^ fast_hash(static_cast<uint64_t>(col), 0);
        Xoshiro256 rng(col_seed);
        
        Size k = 0;
        
        // Process 4 at a time using batch random generation
        // NOLINTNEXTLINE(modernize-avoid-c-arrays)
        std::array<uint64_t, 4> rand_buf{};
        for (; k + 16 <= output_dim; k += 16) {
            rng.next4(rand_buf.data());
            
            for (Size i = 0; i < 4; ++i) {
                uint64_t r = rand_buf[i];
                for (Size b = 0; b < 4; ++b) {
                    // Use 16 bits per decision: mod 6 approximation
                    auto v = static_cast<uint16_t>(r >> (b * 16));
                    auto mod6 = (static_cast<uint32_t>(v) * 6) >> 16;
                    
                    if (mod6 == 0) {
                        output[k + i * 4 + b] += val;
                    } else if (mod6 == 5) {
                        output[k + i * 4 + b] -= val;
                    }
                }
            }
        }
        
        // Cleanup
        for (; k < output_dim; ++k) {
            uint64_t r = rng.next();
            auto v = static_cast<uint32_t>(r % 6);
            if (v == 0) {
                output[k] += val;
            } else if (v == 5) {
                output[k] -= val;
            }
        }
    }
}

// =============================================================================
// Count-Sketch (Hash-based, O(nnz) per row)
// =============================================================================

template <typename T>
SCL_HOT void project_row_countsketch(
    T* SCL_RESTRICT output,
    Size output_dim,
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Index len,
    uint64_t hash_seed,
    uint64_t /* sign_seed */
) noexcept {
    scl::algo::zero(output, output_dim);
    
    if (len == 0) return;

    const Size k = output_dim;
    
    // Process in batches for better instruction-level parallelism
    Index j = 0;
    for (; j + 4 <= len; j += 4) {
        Index col0 = indices[j + 0];
        Index col1 = indices[j + 1];
        Index col2 = indices[j + 2];
        Index col3 = indices[j + 3];
        
        T val0 = values[j + 0];
        T val1 = values[j + 1];
        T val2 = values[j + 2];
        T val3 = values[j + 3];
        
        // Compute hashes in parallel
        uint32_t h0 = 0;
        uint32_t s0 = 0;
        uint32_t h1 = 0;
        uint32_t s1 = 0;
        uint32_t h2 = 0;
        uint32_t s2 = 0;
        uint32_t h3 = 0;
        uint32_t s3 = 0;
        dual_hash(hash_seed, col0, h0, s0);
        dual_hash(hash_seed, col1, h1, s1);
        dual_hash(hash_seed, col2, h2, s2);
        dual_hash(hash_seed, col3, h3, s3);
        
        // Compute buckets and signs
        Size bucket0 = static_cast<Size>(h0) % k;
        Size bucket1 = static_cast<Size>(h1) % k;
        Size bucket2 = static_cast<Size>(h2) % k;
        Size bucket3 = static_cast<Size>(h3) % k;
        
        T sign0 = (s0 & 1) ? val0 : -val0;
        T sign1 = (s1 & 1) ? val1 : -val1;
        T sign2 = (s2 & 1) ? val2 : -val2;
        T sign3 = (s3 & 1) ? val3 : -val3;
        
        output[bucket0] += sign0;
        output[bucket1] += sign1;
        output[bucket2] += sign2;
        output[bucket3] += sign3;
    }
    
    // Cleanup
    for (; j < len; ++j) {
        Index col = indices[j];
        T val = values[j];
        
        uint32_t h = 0;
        uint32_t s = 0;
        dual_hash(hash_seed, col, h, s);
        
        Size bucket = static_cast<Size>(h) % k;
        T sign = (s & 1) ? val : -val;
        
        output[bucket] += sign;
    }
}

// =============================================================================
// Feature Hashing with Multiple Hash Functions
// =============================================================================

template <typename T>
SCL_HOT void project_row_feature_hash(
    T* SCL_RESTRICT output,
    Size output_dim,
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Index len,
    uint64_t seed,
    Size n_hashes  // Number of hash functions (typically 2-4)
) noexcept {
    scl::algo::zero(output, output_dim);
    
    if (len == 0) return;

    const Size k = output_dim;
    const T scale = T(1) / std::sqrt(static_cast<T>(n_hashes));
    
    for (Index j = 0; j < len; ++j) {
        Index col = indices[j];
        T val = values[j] * scale;
        
        // Multiple independent hash functions
        for (Size h = 0; h < n_hashes; ++h) {
            uint64_t h_seed = seed ^ (h * 0xc4ceb9fe1a85ec53ULL);
            
            uint32_t bucket_hash = 0;
            uint32_t sign_hash = 0;
            dual_hash(h_seed, col, bucket_hash, sign_hash);
            
            Size bucket = static_cast<Size>(bucket_hash) % k;
            T sign = (sign_hash & 1) ? val : -val;
            
            output[bucket] += sign;
        }
    }
}

} // namespace detail

// =============================================================================
// Precomputed Projection Matrix (for repeated use)
// =============================================================================

template <typename T>
struct alignas(64) ProjectionMatrix {
    T* data = nullptr;
    Size input_dim = 0;
    Size output_dim = 0;
    Size stride = 0;  // Row stride for alignment
    bool owns_data = false;

    ProjectionMatrix() noexcept = default;

    ~ProjectionMatrix() {
        if (owns_data && data) {
            scl::memory::aligned_free(data, SCL_ALIGNMENT);
        }
    }

    ProjectionMatrix(ProjectionMatrix&& other) noexcept
        : data(other.data), input_dim(other.input_dim),
          output_dim(other.output_dim), stride(other.stride), owns_data(other.owns_data)
    {
        other.data = nullptr;
        other.owns_data = false;
    }

    ProjectionMatrix& operator=(ProjectionMatrix&& other) noexcept {
        if (this != &other) {
            if (owns_data && data) {
                scl::memory::aligned_free(data, SCL_ALIGNMENT);
            }
            data = other.data;
            input_dim = other.input_dim;
            output_dim = other.output_dim;
            stride = other.stride;
            owns_data = other.owns_data;
            other.data = nullptr;
            other.owns_data = false;
        }
        return *this;
    }

    ProjectionMatrix(const ProjectionMatrix&) = delete;
    ProjectionMatrix& operator=(const ProjectionMatrix&) = delete;

    SCL_FORCE_INLINE const T* row(Size col_idx) const noexcept {
        return data + col_idx * stride;
    }

    [[nodiscard]] SCL_FORCE_INLINE bool valid() const noexcept {
        return data != nullptr && input_dim > 0 && output_dim > 0;
    }
    
    [[nodiscard]] Size memory_bytes() const noexcept {
        return input_dim * stride * sizeof(T);
    }
};

// =============================================================================
// Sparse Projection Matrix (for very high-dimensional inputs)
// =============================================================================

template <typename T>
struct SparseProjectionMatrix {
    // For each input dimension: list of (output_idx, sign)
    Index* output_indices = nullptr;
    int8_t* signs = nullptr;
    Size* offsets = nullptr;  // offsets[i+1] - offsets[i] = nnz for input dim i
    
    Size input_dim = 0;
    Size output_dim = 0;
    Size total_nnz = 0;
    T scale = T(1);
    bool owns_data = false;

    SparseProjectionMatrix() noexcept = default;

    ~SparseProjectionMatrix() {
        if (owns_data) {
            if (output_indices) scl::memory::aligned_free(output_indices, SCL_ALIGNMENT);
            if (signs) scl::memory::aligned_free(signs, SCL_ALIGNMENT);
            if (offsets) scl::memory::aligned_free(offsets, SCL_ALIGNMENT);
        }
    }

    SparseProjectionMatrix(SparseProjectionMatrix&& other) noexcept
        : output_indices(other.output_indices), signs(other.signs), offsets(other.offsets),
          input_dim(other.input_dim), output_dim(other.output_dim),
          total_nnz(other.total_nnz), scale(other.scale), owns_data(other.owns_data)
    {
        other.output_indices = nullptr;
        other.signs = nullptr;
        other.offsets = nullptr;
        other.owns_data = false;
    }

    SparseProjectionMatrix& operator=(SparseProjectionMatrix&& other) noexcept {
        if (this != &other) {
            if (owns_data) {
                if (output_indices) scl::memory::aligned_free(output_indices, SCL_ALIGNMENT);
                if (signs) scl::memory::aligned_free(signs, SCL_ALIGNMENT);
                if (offsets) scl::memory::aligned_free(offsets, SCL_ALIGNMENT);
            }
            output_indices = other.output_indices;
            signs = other.signs;
            offsets = other.offsets;
            input_dim = other.input_dim;
            output_dim = other.output_dim;
            total_nnz = other.total_nnz;
            scale = other.scale;
            owns_data = other.owns_data;
            other.output_indices = nullptr;
            other.signs = nullptr;
            other.offsets = nullptr;
            other.owns_data = false;
        }
        return *this;
    }

    SparseProjectionMatrix(const SparseProjectionMatrix&) = delete;
    SparseProjectionMatrix& operator=(const SparseProjectionMatrix&) = delete;

    [[nodiscard]] SCL_FORCE_INLINE bool valid() const noexcept {
        return offsets != nullptr && input_dim > 0 && output_dim > 0;
    }
    
    [[nodiscard]] Size memory_bytes() const noexcept {
        return total_nnz * (sizeof(Index) + sizeof(int8_t)) + (input_dim + 1) * sizeof(Size);
    }
};

// =============================================================================
// Factory Functions
// =============================================================================

template <typename T>
ProjectionMatrix<T> create_gaussian_matrix(
    Size input_dim,
    Size output_dim,
    uint64_t seed = 42
) {
    ProjectionMatrix<T> result;
    result.input_dim = input_dim;
    result.output_dim = output_dim;
    
    // Align stride to cache line
    result.stride = (output_dim + 15) & ~Size(15);
    result.owns_data = true;

    Size total = input_dim * result.stride;
    result.data = scl::memory::aligned_alloc<T>(total, SCL_ALIGNMENT);
    if (!result.data) return {};

    // Zero padding
    scl::algo::zero(result.data, total);

    T scale = T(1) / std::sqrt(static_cast<T>(output_dim));
    
    scl::threading::parallel_for(Size(0), input_dim, [&](size_t i) {
        detail::Xoshiro256 rng(seed ^ (i * 0x9e3779b97f4a7c15ULL));
        T* row = result.data + i * result.stride;
        
        Size k = 0;
        for (; k + 2 <= output_dim; k += 2) {
            rng.gaussian2(row[k], row[k + 1]);
            row[k] *= scale;
            row[k + 1] *= scale;
        }
        if (k < output_dim) {
            Real g1 = Real(0);
            Real g2 = Real(0);
            rng.gaussian2(g1, g2);
            row[k] = g1 * scale;
        }
    });

    return result;
}

template <typename T>
ProjectionMatrix<T> create_achlioptas_matrix(
    Size input_dim,
    Size output_dim,
    uint64_t seed = 42
) {
    ProjectionMatrix<T> result;
    result.input_dim = input_dim;
    result.output_dim = output_dim;
    result.stride = (output_dim + 15) & ~Size(15);
    result.owns_data = true;

    Size total = input_dim * result.stride;
    result.data = scl::memory::aligned_alloc<T>(total, SCL_ALIGNMENT);
    if (!result.data) return {};

    scl::algo::zero(result.data, total);

    T scale = std::sqrt(T(3) / static_cast<T>(output_dim));

    scl::threading::parallel_for(Size(0), input_dim, [&](size_t i) {
        detail::Xoshiro256 rng(seed ^ (i * 0x9e3779b97f4a7c15ULL));
        T* row = result.data + i * result.stride;

        for (Size k = 0; k < output_dim; ++k) {
            uint64_t r = rng.next();
            auto v = static_cast<uint32_t>(r % 6);
            if (v == 0) {
                row[k] = scale;
            } else if (v == 5) {
                row[k] = -scale;
            }
            // else row[k] = 0 (already zeroed)
        }
    });

    return result;
}

template <typename T>
SparseProjectionMatrix<T> create_sparse_matrix(
    Size input_dim,
    Size output_dim,
    Real density,
    uint64_t seed = 42
) {
    SparseProjectionMatrix<T> result;
    result.input_dim = input_dim;
    result.output_dim = output_dim;
    result.owns_data = true;

    result.offsets = scl::memory::aligned_alloc<Size>(input_dim + 1, SCL_ALIGNMENT);
    
    // First pass: count nnz per input dimension
    scl::threading::parallel_for(Size(0), input_dim, [&](size_t i) {
        detail::Xoshiro256 rng(seed ^ (i * 0x9e3779b97f4a7c15ULL));
        Size count = 0;
        for (Size k = 0; k < output_dim; ++k) {
            if (rng.uniform() < density) {
                ++count;
            }
        }
        result.offsets[i] = count;
    });
    
    // Prefix sum
    Size total_nnz = 0;
    for (Size i = 0; i < input_dim; ++i) {
        Size count = result.offsets[i];
        result.offsets[i] = total_nnz;
        total_nnz += count;
    }
    result.offsets[input_dim] = total_nnz;
    result.total_nnz = total_nnz;
    
    // Allocate
    result.output_indices = scl::memory::aligned_alloc<Index>(total_nnz, SCL_ALIGNMENT);
    result.signs = scl::memory::aligned_alloc<int8_t>(total_nnz, SCL_ALIGNMENT);
    result.scale = std::sqrt(T(1) / (static_cast<T>(output_dim) * density));
    
    // Second pass: fill data
    scl::threading::parallel_for(Size(0), input_dim, [&](size_t i) {
        detail::Xoshiro256 rng(seed ^ (i * 0x9e3779b97f4a7c15ULL));
        
        Size offset = result.offsets[i];
        
        for (Size k = 0; k < output_dim; ++k) {
            Real u = rng.uniform();
            if (u < density) {
                result.output_indices[offset] = static_cast<Index>(k);
                result.signs[offset] = (rng.next() & 1) ? int8_t(1) : int8_t(-1);
                ++offset;
            }
        }
    });
    
    return result;
}

// =============================================================================
// Transform with Precomputed Dense Matrix
// =============================================================================

template <typename T, bool IsCSR>
void project_with_dense_matrix(
    const Sparse<T, IsCSR>& matrix,
    const ProjectionMatrix<T>& proj,
    Array<T> output
) {
    static_assert(IsCSR, "project_with_dense_matrix requires CSR format");

    const Index n_rows = matrix.rows();
    const Size N = static_cast<Size>(n_rows);
    const Size output_dim = proj.output_dim;
    const Size total_output = N * output_dim;

    SCL_CHECK_DIM(proj.input_dim == static_cast<Size>(matrix.cols()),
                  "Projection: dimension mismatch");
    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        const auto idx = static_cast<Index>(i);
        const Index len = matrix.row_length_unsafe(idx);

        T* out_row = output.ptr + i * output_dim;
        scl::algo::zero(out_row, output_dim);

        if (len == 0) return;

        auto indices = matrix.row_indices_unsafe(idx);
        auto values = matrix.row_values_unsafe(idx);

        for (Index j = 0; j < len; ++j) {
            // Prefetch next projection row
            if (SCL_LIKELY(j + config::PREFETCH_DISTANCE < len)) {
                SCL_PREFETCH_READ(proj.row(indices[j + config::PREFETCH_DISTANCE]), 0);
            }

            Index col = indices[j];
            T val = values[j];
            const T* proj_row = proj.row(col);

            if (output_dim >= config::SIMD_THRESHOLD) {
                detail::accumulate_simd(out_row, output_dim, val, proj_row);
            } else {
                detail::accumulate_scalar(out_row, output_dim, val, proj_row);
            }
        }
    });
}

// =============================================================================
// Transform with Precomputed Sparse Matrix
// =============================================================================

template <typename T, bool IsCSR>
void project_with_sparse_matrix(
    const Sparse<T, IsCSR>& matrix,
    const SparseProjectionMatrix<T>& proj,
    Array<T> output
) {
    static_assert(IsCSR, "project_with_sparse_matrix requires CSR format");

    const Index n_rows = matrix.rows();
    const Size N = static_cast<Size>(n_rows);
    const Size output_dim = proj.output_dim;
    const Size total_output = N * output_dim;

    SCL_CHECK_DIM(proj.input_dim == static_cast<Size>(matrix.cols()),
                  "Projection: dimension mismatch");
    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    const T scale = proj.scale;

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        const auto idx = static_cast<Index>(i);
        const Index len = matrix.row_length_unsafe(idx);

        T* out_row = output.ptr + i * output_dim;
        scl::algo::zero(out_row, output_dim);

        if (len == 0) return;

        auto mat_indices = matrix.row_indices_unsafe(idx);
        auto mat_values = matrix.row_values_unsafe(idx);

        for (Index j = 0; j < len; ++j) {
            Index col = mat_indices[j];
            T val = mat_values[j] * scale;

            Size proj_start = proj.offsets[col];
            Size proj_end = proj.offsets[col + 1];

            const Index* proj_idx = proj.output_indices + proj_start;
            const int8_t* proj_sign = proj.signs + proj_start;
            Size proj_len = proj_end - proj_start;

            // Unrolled accumulation
            Size k = 0;
            for (; k + 4 <= proj_len; k += 4) {
                out_row[proj_idx[k + 0]] += val * static_cast<T>(proj_sign[k + 0]);
                out_row[proj_idx[k + 1]] += val * static_cast<T>(proj_sign[k + 1]);
                out_row[proj_idx[k + 2]] += val * static_cast<T>(proj_sign[k + 2]);
                out_row[proj_idx[k + 3]] += val * static_cast<T>(proj_sign[k + 3]);
            }
            for (; k < proj_len; ++k) {
                out_row[proj_idx[k]] += val * static_cast<T>(proj_sign[k]);
            }
        }
    });
}

// =============================================================================
// On-the-Fly Projections (Memory Efficient)
// =============================================================================

template <typename T, bool IsCSR>
void project_gaussian_otf(
    const Sparse<T, IsCSR>& matrix,
    Size output_dim,
    Array<T> output,
    uint64_t seed = 42
) {
    static_assert(IsCSR, "project_gaussian_otf requires CSR format");

    const Index n_rows = matrix.rows();
    const Size N = static_cast<Size>(n_rows);
    const Size total_output = N * output_dim;

    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    T scale = T(1) / std::sqrt(static_cast<T>(output_dim));

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        const auto idx = static_cast<Index>(i);
        const Index len = matrix.row_length_unsafe(idx);

        T* out_row = output.ptr + i * output_dim;

        if (len == 0) {
            scl::algo::zero(out_row, output_dim);
            return;
        }

        auto indices = matrix.row_indices_unsafe(idx);
        auto values = matrix.row_values_unsafe(idx);

        uint64_t row_seed = seed ^ (i * 0x9e3779b97f4a7c15ULL);

        detail::project_row_gaussian_blocked(
            out_row, output_dim,
            indices.ptr, values.ptr, len,
            row_seed, scale
        );
    });
}

template <typename T, bool IsCSR>
void project_achlioptas_otf(
    const Sparse<T, IsCSR>& matrix,
    Size output_dim,
    Array<T> output,
    uint64_t seed = 42
) {
    static_assert(IsCSR, "project_achlioptas_otf requires CSR format");

    const Index n_rows = matrix.rows();
    const Size N = static_cast<Size>(n_rows);
    const Size total_output = N * output_dim;

    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    T scale = std::sqrt(T(3) / static_cast<T>(output_dim));

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        const auto idx = static_cast<Index>(i);
        const Index len = matrix.row_length_unsafe(idx);

        T* out_row = output.ptr + i * output_dim;

        if (len == 0) {
            scl::algo::zero(out_row, output_dim);
            return;
        }

        auto indices = matrix.row_indices_unsafe(idx);
        auto values = matrix.row_values_unsafe(idx);

        uint64_t row_seed = seed ^ (i * 0x9e3779b97f4a7c15ULL);

        detail::project_row_achlioptas(
            out_row, output_dim,
            indices.ptr, values.ptr, len,
            row_seed, scale
        );
    });
}

template <typename T, bool IsCSR>
void project_sparse_otf(
    const Sparse<T, IsCSR>& matrix,
    Size output_dim,
    Array<T> output,
    Real density,
    uint64_t seed = 42
) {
    static_assert(IsCSR, "project_sparse_otf requires CSR format");

    const Index n_rows = matrix.rows();
    const Size N = static_cast<Size>(n_rows);
    const Size total_output = N * output_dim;

    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    T scale = std::sqrt(T(1) / (static_cast<T>(output_dim) * density));
    
    // Convert density to sparsity mask
    // density = 1/2 -> mask = 0x1 (1 bit check)
    // density = 1/4 -> mask = 0x3 (2 bit check)
    // density = 1/8 -> mask = 0x7 (3 bit check)
    uint32_t sparsity_mask = 0;
    if (density >= 0.5) {
        sparsity_mask = 0x1;
    } else if (density >= 0.25) {
        sparsity_mask = 0x3;
    } else if (density >= 0.125) {
        sparsity_mask = 0x7;
    } else {
        sparsity_mask = 0xF;
    }

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        const auto idx = static_cast<Index>(i);
        const Index len = matrix.row_length_unsafe(idx);

        T* out_row = output.ptr + i * output_dim;

        if (len == 0) {
            scl::algo::zero(out_row, output_dim);
            return;
        }

        auto indices = matrix.row_indices_unsafe(idx);
        auto values = matrix.row_values_unsafe(idx);

        uint64_t row_seed = seed ^ (i * 0x9e3779b97f4a7c15ULL);

        detail::project_row_sparse_sign(
            out_row, output_dim,
            indices.ptr, values.ptr, len,
            row_seed, scale, sparsity_mask
        );
    });
}

template <typename T, bool IsCSR>
void project_countsketch(
    const Sparse<T, IsCSR>& matrix,
    Size output_dim,
    Array<T> output,
    uint64_t seed = 42
) {
    static_assert(IsCSR, "project_countsketch requires CSR format");

    const Index n_rows = matrix.rows();
    const Size N = static_cast<Size>(n_rows);
    const Size total_output = N * output_dim;

    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    uint64_t hash_seed = seed;
    uint64_t sign_seed = seed ^ 0xdeadbeefcafebabeULL;

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        const auto idx = static_cast<Index>(i);
        const Index len = matrix.row_length_unsafe(idx);

        T* out_row = output.ptr + i * output_dim;

        if (len == 0) {
            scl::algo::zero(out_row, output_dim);
            return;
        }

        auto indices = matrix.row_indices_unsafe(idx);
        auto values = matrix.row_values_unsafe(idx);

        detail::project_row_countsketch(
            out_row, output_dim,
            indices.ptr, values.ptr, len,
            hash_seed, sign_seed
        );
    });
}

template <typename T, bool IsCSR>
void project_feature_hash(
    const Sparse<T, IsCSR>& matrix,
    Size output_dim,
    Array<T> output,
    Size n_hashes = 3,
    uint64_t seed = 42
) {
    static_assert(IsCSR, "project_feature_hash requires CSR format");

    const Index n_rows = matrix.rows();
    const Size N = static_cast<Size>(n_rows);
    const Size total_output = N * output_dim;

    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        const auto idx = static_cast<Index>(i);
        const Index len = matrix.row_length_unsafe(idx);

        T* out_row = output.ptr + i * output_dim;

        if (len == 0) {
            scl::algo::zero(out_row, output_dim);
            return;
        }

        auto indices = matrix.row_indices_unsafe(idx);
        auto values = matrix.row_values_unsafe(idx);

        detail::project_row_feature_hash(
            out_row, output_dim,
            indices.ptr, values.ptr, len,
            seed, n_hashes
        );
    });
}

// =============================================================================
// High-Level Interface with Auto-Selection
// =============================================================================

template <typename T, bool IsCSR>
void project(
    const Sparse<T, IsCSR>& matrix,
    Size output_dim,
    Array<T> output,
    ProjectionType type = ProjectionType::Sparse,
    uint64_t seed = 42
) {
    Real density = Real(1) / std::sqrt(static_cast<Real>(matrix.cols()));
    density = scl::algo::max2(density, Real(0.01));
    density = scl::algo::min2(density, Real(0.5));

    switch (type) {
        case ProjectionType::Gaussian:
            project_gaussian_otf(matrix, output_dim, output, seed);
            break;
        case ProjectionType::Achlioptas:
            project_achlioptas_otf(matrix, output_dim, output, seed);
            break;
        case ProjectionType::Sparse:
            project_sparse_otf(matrix, output_dim, output, density, seed);
            break;
        case ProjectionType::CountSketch:
            project_countsketch(matrix, output_dim, output, seed);
            break;
        case ProjectionType::FeatureHash:
            project_feature_hash(matrix, output_dim, output, 3, seed);
            break;
    }
}

// Auto-select best projection method
template <typename T, bool IsCSR>
void project_auto(
    const Sparse<T, IsCSR>& matrix,
    Size output_dim,
    Array<T> output,
    uint64_t seed = 42
) {
    const Size input_dim = static_cast<Size>(matrix.cols());
    
    // Heuristic selection based on dimensions
    if (output_dim <= 64) {
        // For very small output, CountSketch is fastest
        project_countsketch(matrix, output_dim, output, seed);
    } else if (input_dim > 100000) {
        // For very high dimensional input, use sparse projection
        Real density = Real(1) / std::sqrt(static_cast<Real>(input_dim));
        project_sparse_otf(matrix, output_dim, output, density, seed);
    } else if (output_dim >= 512) {
        // For large output, Achlioptas is good balance
        project_achlioptas_otf(matrix, output_dim, output, seed);
    } else {
        // Default: sparse projection
        Real density = Real(1) / std::sqrt(static_cast<Real>(input_dim));
        density = scl::algo::max2(density, Real(0.05));
        project_sparse_otf(matrix, output_dim, output, density, seed);
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

// Johnson-Lindenstrauss optimal dimension
inline Size compute_jl_dimension(Size n_samples, Real epsilon = config::DEFAULT_EPSILON) {
    Real ln_n = std::log(static_cast<Real>(n_samples));
    Real eps2 = epsilon * epsilon;
    Real denom = eps2 / Real(2) - eps2 * epsilon / Real(3);
    Real k = Real(4) * ln_n / denom;
    return static_cast<Size>(std::ceil(k));
}

// Recommend projection type based on problem characteristics
inline ProjectionType recommend_projection(
    Size input_dim,
    Size output_dim,
    Size /* n_rows */,
    bool preserve_distances
) {
    if (!preserve_distances) {
        // Just need dimensionality reduction
        return (output_dim <= 128) ? ProjectionType::CountSketch : ProjectionType::FeatureHash;
    }
    
    // Need to preserve distances (JL property)
    if (input_dim > 50000) {
        return ProjectionType::Sparse;  // Memory efficient
    } else if (output_dim >= 256) {
        return ProjectionType::Achlioptas;  // Good accuracy
    } else {
        return ProjectionType::Gaussian;  // Best accuracy
    }
}

// Estimate memory usage for precomputed matrix
inline Size estimate_dense_matrix_memory(Size input_dim, Size output_dim) {
    Size stride = (output_dim + 15) & ~Size(15);
    return input_dim * stride * sizeof(Real);
}

inline Size estimate_sparse_matrix_memory(Size input_dim, Size output_dim, Real density) {
    Size expected_nnz = static_cast<Size>(static_cast<Real>(input_dim) * static_cast<Real>(output_dim) * density);
    return expected_nnz * (sizeof(Index) + sizeof(int8_t)) + (input_dim + 1) * sizeof(Size);
}

} // namespace scl::kernel::projection
