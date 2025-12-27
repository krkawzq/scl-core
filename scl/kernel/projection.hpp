#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/projection.hpp
// BRIEF: Sparse random projection for dimensionality reduction
// =============================================================================

namespace scl::kernel::projection {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size SIMD_THRESHOLD = 64;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size SMALL_OUTPUT_DIM = 32;
    constexpr Real DEFAULT_EPSILON = Real(0.1);
}

// =============================================================================
// Projection Types
// =============================================================================

enum class ProjectionType {
    Gaussian,       // Dense Gaussian N(0, 1/k)
    Achlioptas,     // Sparse {+1, 0, -1} with prob {1/6, 2/3, 1/6}
    Sparse,         // Very sparse with density 1/sqrt(d)
    CountSketch     // Sign flips with hash-based indexing
};

// =============================================================================
// PRNG: Splitmix64 for fast reproducible random numbers
// =============================================================================

namespace detail {

struct Splitmix64 {
    uint64_t state;

    SCL_FORCE_INLINE explicit Splitmix64(uint64_t seed) noexcept : state(seed) {}

    SCL_FORCE_INLINE uint64_t next() noexcept {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next() >> 11) * Real(0x1.0p-53);
    }

    SCL_FORCE_INLINE Real gaussian() noexcept {
        Real u1 = uniform();
        Real u2 = uniform();
        return std::sqrt(Real(-2) * std::log(u1 + Real(1e-15))) *
               std::cos(Real(2) * Real(3.14159265358979323846) * u2);
    }

    SCL_FORCE_INLINE int8_t achlioptas() noexcept {
        uint64_t r = next();
        uint32_t v = static_cast<uint32_t>(r % 6);
        if (v == 0) return 1;
        if (v == 5) return -1;
        return 0;
    }

    SCL_FORCE_INLINE int8_t sparse_sign(Real density) noexcept {
        if (uniform() >= density) return 0;
        return (next() & 1) ? 1 : -1;
    }
};

// =============================================================================
// SIMD-Optimized Dense Accumulation
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void accumulate_scalar_simd(
    T* SCL_RESTRICT output,
    Size output_dim,
    T value,
    const T* SCL_RESTRICT proj_row
) {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    const auto v_val = s::Set(d, value);

    Size k = 0;
    for (; k + 4 * lanes <= output_dim; k += 4 * lanes) {
        auto v0 = s::Load(d, output + k);
        auto v1 = s::Load(d, output + k + lanes);
        auto v2 = s::Load(d, output + k + 2 * lanes);
        auto v3 = s::Load(d, output + k + 3 * lanes);

        auto p0 = s::Load(d, proj_row + k);
        auto p1 = s::Load(d, proj_row + k + lanes);
        auto p2 = s::Load(d, proj_row + k + 2 * lanes);
        auto p3 = s::Load(d, proj_row + k + 3 * lanes);

        s::Store(s::MulAdd(v_val, p0, v0), d, output + k);
        s::Store(s::MulAdd(v_val, p1, v1), d, output + k + lanes);
        s::Store(s::MulAdd(v_val, p2, v2), d, output + k + 2 * lanes);
        s::Store(s::MulAdd(v_val, p3, v3), d, output + k + 3 * lanes);
    }

    for (; k + lanes <= output_dim; k += lanes) {
        auto v = s::Load(d, output + k);
        auto p = s::Load(d, proj_row + k);
        s::Store(s::MulAdd(v_val, p, v), d, output + k);
    }

    for (; k < output_dim; ++k) {
        output[k] += value * proj_row[k];
    }
}

template <typename T>
SCL_FORCE_INLINE void accumulate_scalar_short(
    T* SCL_RESTRICT output,
    Size output_dim,
    T value,
    const T* SCL_RESTRICT proj_row
) {
    for (Size k = 0; k < output_dim; ++k) {
        output[k] += value * proj_row[k];
    }
}

// =============================================================================
// On-the-fly Projection (No Explicit Matrix)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void project_row_gaussian_otf(
    T* SCL_RESTRICT output,
    Size output_dim,
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Index len,
    uint64_t row_seed,
    T scale
) {
    scl::algo::zero(output, output_dim);

    for (Index j = 0; j < len; ++j) {
        Index col = indices[j];
        T val = values[j];

        Splitmix64 rng(row_seed ^ (static_cast<uint64_t>(col) * 0x9e3779b97f4a7c15ULL));

        for (Size k = 0; k < output_dim; ++k) {
            T r = rng.gaussian() * scale;
            output[k] += val * r;
        }
    }
}

template <typename T>
SCL_FORCE_INLINE void project_row_achlioptas_otf(
    T* SCL_RESTRICT output,
    Size output_dim,
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Index len,
    uint64_t row_seed,
    T scale
) {
    scl::algo::zero(output, output_dim);

    for (Index j = 0; j < len; ++j) {
        Index col = indices[j];
        T val = values[j];

        Splitmix64 rng(row_seed ^ (static_cast<uint64_t>(col) * 0x9e3779b97f4a7c15ULL));

        for (Size k = 0; k < output_dim; ++k) {
            int8_t r = rng.achlioptas();
            if (r != 0) {
                output[k] += val * static_cast<T>(r) * scale;
            }
        }
    }
}

template <typename T>
SCL_FORCE_INLINE void project_row_sparse_otf(
    T* SCL_RESTRICT output,
    Size output_dim,
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Index len,
    uint64_t row_seed,
    T scale,
    Real density
) {
    scl::algo::zero(output, output_dim);

    for (Index j = 0; j < len; ++j) {
        Index col = indices[j];
        T val = values[j];

        Splitmix64 rng(row_seed ^ (static_cast<uint64_t>(col) * 0x9e3779b97f4a7c15ULL));

        for (Size k = 0; k < output_dim; ++k) {
            int8_t r = rng.sparse_sign(density);
            if (r != 0) {
                output[k] += val * static_cast<T>(r) * scale;
            }
        }
    }
}

// =============================================================================
// Count-Sketch Projection (Hash-based)
// =============================================================================

SCL_FORCE_INLINE uint32_t hash_combine(uint64_t seed, Index col) noexcept {
    uint64_t h = seed ^ (static_cast<uint64_t>(col) * 0x9e3779b97f4a7c15ULL);
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    return static_cast<uint32_t>(h >> 32);
}

template <typename T>
SCL_FORCE_INLINE void project_row_countsketch(
    T* SCL_RESTRICT output,
    Size output_dim,
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Index len,
    uint64_t hash_seed,
    uint64_t sign_seed
) {
    scl::algo::zero(output, output_dim);

    const Size k = output_dim;

    for (Index j = 0; j < len; ++j) {
        Index col = indices[j];
        T val = values[j];

        uint32_t h = hash_combine(hash_seed, col);
        Size bucket = static_cast<Size>(h) % k;

        uint32_t s = hash_combine(sign_seed, col);
        T sign = (s & 1) ? T(1) : T(-1);

        output[bucket] += val * sign;
    }
}

} // namespace detail

// =============================================================================
// Pre-computed Projection Matrix
// =============================================================================

template <typename T>
struct ProjectionMatrix {
    T* data;
    Size input_dim;
    Size output_dim;
    bool owns_data;

    ProjectionMatrix() : data(nullptr), input_dim(0), output_dim(0), owns_data(false) {}

    ~ProjectionMatrix() {
        if (owns_data && data) {
            scl::memory::aligned_free(data, SCL_ALIGNMENT);
        }
    }

    ProjectionMatrix(ProjectionMatrix&& other) noexcept
        : data(other.data), input_dim(other.input_dim),
          output_dim(other.output_dim), owns_data(other.owns_data)
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
            owns_data = other.owns_data;
            other.data = nullptr;
            other.owns_data = false;
        }
        return *this;
    }

    ProjectionMatrix(const ProjectionMatrix&) = delete;
    ProjectionMatrix& operator=(const ProjectionMatrix&) = delete;

    SCL_FORCE_INLINE const T* row(Size col_idx) const noexcept {
        return data + col_idx * output_dim;
    }

    SCL_FORCE_INLINE bool valid() const noexcept {
        return data != nullptr && input_dim > 0 && output_dim > 0;
    }
};

// =============================================================================
// Factory Functions for Projection Matrix
// =============================================================================

template <typename T>
ProjectionMatrix<T> create_gaussian_projection(
    Size input_dim,
    Size output_dim,
    uint64_t seed = 42
) {
    ProjectionMatrix<T> result;
    result.input_dim = input_dim;
    result.output_dim = output_dim;
    result.owns_data = true;

    Size total = input_dim * output_dim;
    result.data = scl::memory::aligned_alloc<T>(total, SCL_ALIGNMENT);

    if (!result.data) return {};

    T scale = T(1) / std::sqrt(static_cast<T>(output_dim));
    detail::Splitmix64 rng(seed);

    for (Size i = 0; i < total; ++i) {
        result.data[i] = rng.gaussian() * scale;
    }

    return result;
}

template <typename T>
ProjectionMatrix<T> create_achlioptas_projection(
    Size input_dim,
    Size output_dim,
    uint64_t seed = 42
) {
    ProjectionMatrix<T> result;
    result.input_dim = input_dim;
    result.output_dim = output_dim;
    result.owns_data = true;

    Size total = input_dim * output_dim;
    result.data = scl::memory::aligned_alloc<T>(total, SCL_ALIGNMENT);

    if (!result.data) return {};

    T scale = std::sqrt(T(3) / static_cast<T>(output_dim));
    detail::Splitmix64 rng(seed);

    for (Size i = 0; i < total; ++i) {
        int8_t v = rng.achlioptas();
        result.data[i] = static_cast<T>(v) * scale;
    }

    return result;
}

template <typename T>
ProjectionMatrix<T> create_sparse_projection(
    Size input_dim,
    Size output_dim,
    Real density,
    uint64_t seed = 42
) {
    ProjectionMatrix<T> result;
    result.input_dim = input_dim;
    result.output_dim = output_dim;
    result.owns_data = true;

    Size total = input_dim * output_dim;
    result.data = scl::memory::aligned_alloc<T>(total, SCL_ALIGNMENT);

    if (!result.data) return {};

    T scale = std::sqrt(T(1) / (static_cast<T>(output_dim) * density));
    detail::Splitmix64 rng(seed);

    for (Size i = 0; i < total; ++i) {
        int8_t v = rng.sparse_sign(density);
        result.data[i] = static_cast<T>(v) * scale;
    }

    return result;
}

// =============================================================================
// Transform: Sparse Matrix x Projection Matrix
// =============================================================================

template <typename T, bool IsCSR>
void project_with_matrix(
    const Sparse<T, IsCSR>& matrix,
    const ProjectionMatrix<T>& proj,
    Array<T> output
) {
    static_assert(IsCSR, "project_with_matrix requires CSR format");

    const Index n_rows = matrix.rows();
    const Size output_dim = proj.output_dim;
    const Size total_output = static_cast<Size>(n_rows) * output_dim;

    SCL_CHECK_DIM(proj.input_dim == static_cast<Size>(matrix.cols()),
                  "Projection: dimension mismatch");
    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    scl::memory::zero(output);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_rows), [&](size_t i) {
        const Index idx = static_cast<Index>(i);
        const Index len = matrix.row_length(idx);

        if (len == 0) return;

        T* out_row = output.ptr + i * output_dim;
        auto indices = matrix.row_indices(idx);
        auto values = matrix.row_values(idx);

        if (output_dim >= config::SIMD_THRESHOLD) {
            for (Index j = 0; j < len; ++j) {
                if (j + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
                    SCL_PREFETCH_READ(proj.row(indices[j + config::PREFETCH_DISTANCE]), 0);
                }

                Index col = indices[j];
                T val = values[j];
                const T* proj_row = proj.row(col);

                detail::accumulate_scalar_simd(out_row, output_dim, val, proj_row);
            }
        } else {
            for (Index j = 0; j < len; ++j) {
                Index col = indices[j];
                T val = values[j];
                const T* proj_row = proj.row(col);

                detail::accumulate_scalar_short(out_row, output_dim, val, proj_row);
            }
        }
    });
}

// =============================================================================
// On-the-Fly Projection (Memory Efficient)
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
    const Size total_output = static_cast<Size>(n_rows) * output_dim;

    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    T scale = T(1) / std::sqrt(static_cast<T>(output_dim));

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_rows), [&](size_t i) {
        const Index idx = static_cast<Index>(i);
        const Index len = matrix.row_length(idx);

        T* out_row = output.ptr + i * output_dim;

        if (len == 0) {
            scl::algo::zero(out_row, output_dim);
            return;
        }

        auto indices = matrix.row_indices(idx);
        auto values = matrix.row_values(idx);

        uint64_t row_seed = seed ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL);

        detail::project_row_gaussian_otf(
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
    const Size total_output = static_cast<Size>(n_rows) * output_dim;

    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    T scale = std::sqrt(T(3) / static_cast<T>(output_dim));

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_rows), [&](size_t i) {
        const Index idx = static_cast<Index>(i);
        const Index len = matrix.row_length(idx);

        T* out_row = output.ptr + i * output_dim;

        if (len == 0) {
            scl::algo::zero(out_row, output_dim);
            return;
        }

        auto indices = matrix.row_indices(idx);
        auto values = matrix.row_values(idx);

        uint64_t row_seed = seed ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL);

        detail::project_row_achlioptas_otf(
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
    const Size total_output = static_cast<Size>(n_rows) * output_dim;

    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    T scale = std::sqrt(T(1) / (static_cast<T>(output_dim) * density));

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_rows), [&](size_t i) {
        const Index idx = static_cast<Index>(i);
        const Index len = matrix.row_length(idx);

        T* out_row = output.ptr + i * output_dim;

        if (len == 0) {
            scl::algo::zero(out_row, output_dim);
            return;
        }

        auto indices = matrix.row_indices(idx);
        auto values = matrix.row_values(idx);

        uint64_t row_seed = seed ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL);

        detail::project_row_sparse_otf(
            out_row, output_dim,
            indices.ptr, values.ptr, len,
            row_seed, scale, density
        );
    });
}

// =============================================================================
// Count-Sketch Projection
// =============================================================================

template <typename T, bool IsCSR>
void project_countsketch(
    const Sparse<T, IsCSR>& matrix,
    Size output_dim,
    Array<T> output,
    uint64_t seed = 42
) {
    static_assert(IsCSR, "project_countsketch requires CSR format");

    const Index n_rows = matrix.rows();
    const Size total_output = static_cast<Size>(n_rows) * output_dim;

    SCL_CHECK_DIM(output.len >= total_output,
                  "Projection: output buffer too small");

    uint64_t hash_seed = seed;
    uint64_t sign_seed = seed ^ 0xdeadbeefcafebabeULL;

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_rows), [&](size_t i) {
        const Index idx = static_cast<Index>(i);
        const Index len = matrix.row_length(idx);

        T* out_row = output.ptr + i * output_dim;

        if (len == 0) {
            scl::algo::zero(out_row, output_dim);
            return;
        }

        auto indices = matrix.row_indices(idx);
        auto values = matrix.row_values(idx);

        detail::project_row_countsketch(
            out_row, output_dim,
            indices.ptr, values.ptr, len,
            hash_seed, sign_seed
        );
    });
}

// =============================================================================
// High-Level Interface
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
    }
}

// =============================================================================
// Utility: Compute Optimal Output Dimension (JL Lemma)
// =============================================================================

inline Size compute_jl_dimension(Size n_samples, Real epsilon = config::DEFAULT_EPSILON) {
    Real ln_n = std::log(static_cast<Real>(n_samples));
    Real eps2 = epsilon * epsilon;
    Real denom = eps2 / Real(2) - eps2 * epsilon / Real(3);
    Real k = Real(4) * ln_n / denom;
    return static_cast<Size>(std::ceil(k));
}

} // namespace scl::kernel::projection
