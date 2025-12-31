#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/core/error.hpp"
#include "scl/math/stat_base.hpp"

#include <algorithm>
#include <numeric>
#include <vector>
#include <span>
#include <cmath>

// =============================================================================
// FILE: scl/math/multiple_testing.hpp
// DESCRIPTION: Multiple testing correction methods for p-value adjustment
//              Core FDR control and family-wise error rate procedures
// =============================================================================

namespace scl::math {

// =============================================================================
// Multiple Testing Correction Types
// =============================================================================

/// @brief Multiple testing correction methods
enum class MultipleTestingMethod {
    Bonferroni,          ///< Bonferroni correction (FWER control)
    BenjaminiHochberg,   ///< Benjamini-Hochberg FDR (independent/positive dependence)
    BenjaminiYekutieli   ///< Benjamini-Yekutieli FDR (arbitrary dependence)
};

// =============================================================================
// Helper Functions
// =============================================================================

namespace detail {

/// @brief Computes harmonic number H(n) = sum(1/i) for i=1 to n
/// @param[in] n Number of terms
/// @return H(n) = 1 + 1/2 + 1/3 + ... + 1/n
/// @note Used in Benjamini-Yekutieli correction factor
/// @note For large n, H(n) ≈ ln(n) + γ where γ ≈ 0.5772 (Euler-Mascheroni constant)
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_harmonic_number(Size n) -> double {
    if (n == 0) [[unlikely]] {
        return 0.0;
    }

    double sum = 0.0;
    for (Size i = 1; i <= n; ++i) {
        sum += 1.0 / static_cast<double>(i);
    }
    return sum;
}

/// @brief Clamps a p-value to valid range [0, 1]
/// @tparam Real Floating-point type
/// @param[in] p P-value to clamp
/// @return Clamped p-value in [0, 1]
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto clamp_pvalue(Real p) -> Real {
    if (p < Real(0)) [[unlikely]] {
        return Real(0);
    }
    if (p > Real(1)) [[unlikely]] {
        return Real(1);
    }
    return p;
}

} // namespace detail

// =============================================================================
// Bonferroni Correction
// =============================================================================

/// @brief Applies Bonferroni correction to p-values
/// @tparam Real Floating-point type
/// @param[in] p_values Input p-values (read-only view)
/// @param[out] adjusted Output adjusted p-values (must have same size as p_values)
/// @note Formula: p_adj = min(p * n, 1.0)
/// @note Controls family-wise error rate (FWER)
/// @note Most conservative correction; suitable for small number of tests
/// @note Does NOT require sorting (order-independent)
/// @pre p_values and adjusted must have the same size
/// @throws std::invalid_argument if sizes don't match
template<Arithmetic Real>
SCL_FORCE_INLINE
auto bonferroni_correction(
    std::span<const Real> p_values,
    std::span<Real> adjusted
) -> void {
    SCL_CHECK(p_values.size() == adjusted.size(),
        "bonferroni_correction: p_values and adjusted must have same size");

    const Size n = p_values.size();
    if (n == 0) [[unlikely]] {
        return;
    }

    const double n_tests = static_cast<double>(n);

    for (Size i = 0; i < n; ++i) {
        const Real p_adj = static_cast<Real>(static_cast<double>(p_values[i]) * n_tests);
        adjusted[i] = detail::clamp_pvalue(p_adj);
    }
}

// =============================================================================
// Benjamini-Hochberg FDR Correction
// =============================================================================

/// @brief Applies Benjamini-Hochberg FDR correction to p-values
/// @tparam Real Floating-point type
/// @param[in] p_values Input p-values (read-only view)
/// @param[out] adjusted Output adjusted p-values (must have same size as p_values)
/// @note Formula: p_adj[i] = p[i] * n / rank[i], enforced monotonicity
/// @note Controls false discovery rate (FDR)
/// @note Assumes independence or positive dependence among tests
/// @note Less conservative than Bonferroni; more powerful
/// @note Requires sorting by p-value internally
/// @pre p_values and adjusted must have the same size
/// @throws std::invalid_argument if sizes don't match
template<Arithmetic Real>
SCL_FORCE_INLINE
auto benjamini_hochberg_correction(
    std::span<const Real> p_values,
    std::span<Real> adjusted
) -> void {
    SCL_CHECK(p_values.size() == adjusted.size(),
        "benjamini_hochberg_correction: p_values and adjusted must have same size");

    const Size n = p_values.size();
    if (n == 0) [[unlikely]] {
        return;
    }

    // Create index array for sorting
    std::vector<Size> indices(n);
    std::iota(indices.begin(), indices.end(), Size(0));

    // Sort indices by p-values (ascending)
    std::sort(indices.begin(), indices.end(),
        [&p_values](Size i, Size j) {
            return p_values[i] < p_values[j];
        }
    );

    // Temporary storage for sorted adjusted p-values
    std::vector<Real> sorted_adjusted(n);

    const double n_tests = static_cast<double>(n);

    // Compute adjusted p-values: p_adj[i] = p[i] * n / rank
    for (Size i = 0; i < n; ++i) {
        const Size idx = indices[i];
        const double rank = static_cast<double>(i + 1);
        const double p_adj = static_cast<double>(p_values[idx]) * n_tests / rank;
        sorted_adjusted[i] = static_cast<Real>(p_adj);
    }

    // Enforce monotonicity: cumulative minimum from right to left
    sorted_adjusted[n - 1] = detail::clamp_pvalue(sorted_adjusted[n - 1]);
    for (Size i = n - 1; i > 0; --i) {
        sorted_adjusted[i - 1] = std::min(sorted_adjusted[i - 1], sorted_adjusted[i]);
        sorted_adjusted[i - 1] = detail::clamp_pvalue(sorted_adjusted[i - 1]);
    }

    // Map back to original order
    for (Size i = 0; i < n; ++i) {
        adjusted[indices[i]] = sorted_adjusted[i];
    }
}

// =============================================================================
// Benjamini-Yekutieli FDR Correction
// =============================================================================

/// @brief Applies Benjamini-Yekutieli FDR correction to p-values
/// @tparam Real Floating-point type
/// @param[in] p_values Input p-values (read-only view)
/// @param[out] adjusted Output adjusted p-values (must have same size as p_values)
/// @note Formula: p_adj[i] = p[i] * n * c(n) / rank[i], where c(n) = sum(1/j) for j=1..n
/// @note Controls false discovery rate (FDR)
/// @note Assumes arbitrary dependence among tests (most general)
/// @note More conservative than Benjamini-Hochberg
/// @note Correction factor c(n) ≈ ln(n) + 0.5772 for large n
/// @note Requires sorting by p-value internally
/// @pre p_values and adjusted must have the same size
/// @throws std::invalid_argument if sizes don't match
template<Arithmetic Real>
SCL_FORCE_INLINE
auto benjamini_yekutieli_correction(
    std::span<const Real> p_values,
    std::span<Real> adjusted
) -> void {
    SCL_CHECK(p_values.size() == adjusted.size(),
        "benjamini_yekutieli_correction: p_values and adjusted must have same size");

    const Size n = p_values.size();
    if (n == 0) [[unlikely]] {
        return;
    }

    // Compute harmonic number c(n) = sum(1/i) for i=1 to n
    const double c_n = detail::compute_harmonic_number(n);

    // Create index array for sorting
    std::vector<Size> indices(n);
    std::iota(indices.begin(), indices.end(), Size(0));

    // Sort indices by p-values (ascending)
    std::sort(indices.begin(), indices.end(),
        [&p_values](Size i, Size j) {
            return p_values[i] < p_values[j];
        }
    );

    // Temporary storage for sorted adjusted p-values
    std::vector<Real> sorted_adjusted(n);

    const double n_tests = static_cast<double>(n);

    // Compute adjusted p-values: p_adj[i] = p[i] * n * c(n) / rank
    for (Size i = 0; i < n; ++i) {
        const Size idx = indices[i];
        const double rank = static_cast<double>(i + 1);
        const double p_adj = static_cast<double>(p_values[idx]) * n_tests * c_n / rank;
        sorted_adjusted[i] = static_cast<Real>(p_adj);
    }

    // Enforce monotonicity: cumulative minimum from right to left
    sorted_adjusted[n - 1] = detail::clamp_pvalue(sorted_adjusted[n - 1]);
    for (Size i = n - 1; i > 0; --i) {
        sorted_adjusted[i - 1] = std::min(sorted_adjusted[i - 1], sorted_adjusted[i]);
        sorted_adjusted[i - 1] = detail::clamp_pvalue(sorted_adjusted[i - 1]);
    }

    // Map back to original order
    for (Size i = 0; i < n; ++i) {
        adjusted[indices[i]] = sorted_adjusted[i];
    }
}

// =============================================================================
// Generic Multiple Testing Correction
// =============================================================================

/// @brief Applies multiple testing correction using specified method
/// @tparam Real Floating-point type
/// @param[in] p_values Input p-values (read-only view)
/// @param[out] adjusted Output adjusted p-values (must have same size as p_values)
/// @param[in] method Correction method to use
/// @note Dispatches to appropriate correction function
/// @pre p_values and adjusted must have the same size
/// @throws std::invalid_argument if sizes don't match or method is invalid
template<Arithmetic Real>
SCL_FORCE_INLINE
auto correct_pvalues(
    std::span<const Real> p_values,
    std::span<Real> adjusted,
    MultipleTestingMethod method
) -> void {
    switch (method) {
        case MultipleTestingMethod::Bonferroni:
            bonferroni_correction(p_values, adjusted);
            break;
        case MultipleTestingMethod::BenjaminiHochberg:
            benjamini_hochberg_correction(p_values, adjusted);
            break;
        case MultipleTestingMethod::BenjaminiYekutieli:
            benjamini_yekutieli_correction(p_values, adjusted);
            break;
        default:
            SCL_CHECK(false, "correct_pvalues: invalid method");
    }
}

// =============================================================================
// Convenience Overloads (Vector Interface)
// =============================================================================

/// @brief Applies Bonferroni correction to p-values (vector interface)
/// @tparam Real Floating-point type
/// @param[in,out] p_values Input p-values; will be modified in-place
/// @note In-place version: overwrites input with adjusted p-values
template<Arithmetic Real>
SCL_FORCE_INLINE
auto bonferroni_correction_inplace(std::vector<Real>& p_values) -> void {
    bonferroni_correction<Real>(std::span<const Real>(p_values), std::span<Real>(p_values));
}

/// @brief Applies Benjamini-Hochberg FDR correction (vector interface)
/// @tparam Real Floating-point type
/// @param[in,out] p_values Input p-values; will be modified in-place
/// @note In-place version: overwrites input with adjusted p-values
template<Arithmetic Real>
SCL_FORCE_INLINE
auto benjamini_hochberg_correction_inplace(std::vector<Real>& p_values) -> void {
    benjamini_hochberg_correction<Real>(std::span<const Real>(p_values), std::span<Real>(p_values));
}

/// @brief Applies Benjamini-Yekutieli FDR correction (vector interface)
/// @tparam Real Floating-point type
/// @param[in,out] p_values Input p-values; will be modified in-place
/// @note In-place version: overwrites input with adjusted p-values
template<Arithmetic Real>
SCL_FORCE_INLINE
auto benjamini_yekutieli_correction_inplace(std::vector<Real>& p_values) -> void {
    benjamini_yekutieli_correction<Real>(std::span<const Real>(p_values), std::span<Real>(p_values));
}

} // namespace scl::math
