// =============================================================================
// FILE: scl/core/simd.h
// BRIEF: API reference for SCL SIMD Wrapper (Google Highway)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include <hwy/highway.h>
#include <cstddef>

namespace scl::simd {

// =============================================================================
// OVERVIEW
// =============================================================================

/* -----------------------------------------------------------------------------
 * MODULE: SIMD Wrapper
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Architecture-agnostic SIMD abstraction layer using Google Highway.
 *
 * PURPOSE:
 *     This module provides a unified interface for SIMD operations across
 *     different hardware architectures (AVX2, AVX-512, NEON, etc.). It wraps
 *     Google Highway and provides SCL-specific type aliases and utilities.
 *
 * DESIGN PRINCIPLES:
 *     - Zero runtime overhead: all abstractions compile away
 *     - Type safety: tag-based dispatch prevents type mismatches
 *     - Architecture portability: same code runs on x86, ARM, etc.
 *     - Scalability: automatically uses best vector width for hardware
 *
 * CONFIGURATION:
 *     - SCL_ONLY_SCALAR: Disable SIMD, use scalar fallback only
 *     - HWY_COMPILE_ONLY_SCALAR: Propagated from SCL_ONLY_SCALAR
 *
 * NAMESPACE INJECTION:
 *     All Highway functions (Load, Store, Add, Mul, etc.) are imported
 *     into scl::simd namespace, allowing direct use without hwy:: prefix.
 *
 * TYPICAL USAGE PATTERN:
 *     const scl::simd::Tag d;
 *     auto v1 = scl::simd::Load(d, ptr);
 *     auto v2 = scl::simd::Add(v1, v1);
 *     scl::simd::Store(v2, d, output);
 * -------------------------------------------------------------------------- */

// =============================================================================
// NAMESPACE INJECTION
// =============================================================================

/* -----------------------------------------------------------------------------
 * USING DIRECTIVE: Highway Namespace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Import all Highway SIMD functions into scl::simd namespace.
 *
 * PURPOSE:
 *     This using directive imports the architecture-specific Highway namespace
 *     (e.g., hwy::N_AVX2, hwy::N_NEON) into scl::simd, making all Highway
 *     operations directly available as scl::simd::Load, scl::simd::Add, etc.
 *
 * IMPORTED FUNCTIONS (partial list):
 *     - Load, Store, LoadU, StoreU: Memory operations
 *     - Add, Sub, Mul, Div: Arithmetic operations
 *     - And, Or, Xor, Not: Logical operations
 *     - Min, Max, Abs: Comparison and absolute value
 *     - Set, Zero, Iota: Vector initialization
 *     - IfThenElse: Conditional selection
 *     - And many more (see Highway documentation)
 *
 * THREAD SAFETY:
 *     Safe - SIMD operations are thread-local
 *
 * PERFORMANCE:
 *     Zero overhead - using directive is compile-time only
 * -------------------------------------------------------------------------- */
using namespace hwy::HWY_NAMESPACE;

// =============================================================================
// TYPE ALIASES (SMART TAGS)
// =============================================================================

/* -----------------------------------------------------------------------------
 * TYPE ALIAS: Tag
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Primary SIMD descriptor tag for scl::Real type.
 *
 * PURPOSE:
 *     Provides a convenient alias for ScalableTag<scl::Real>. This tag
 *     automatically selects the optimal vector width for the current
 *     hardware and scl::Real type (float or double).
 *
 * DEFINITION:
 *     using Tag = ScalableTag<scl::Real>;
 *
 * PRECONDITIONS:
 *     - scl::Real must be defined (float or double)
 *
 * POSTCONDITIONS:
 *     - Tag represents the SIMD descriptor for scl::Real
 *     - Lanes(Tag()) returns the number of elements in a vector
 *
 * THREAD SAFETY:
 *     Safe - type alias is compile-time
 *
 * WHEN TO USE:
 *     - Processing floating-point arrays
 *     - Mathematical operations on scl::Real data
 *     - Default choice for most SCL kernels
 *
 * USAGE PATTERN:
 *     const Tag d;
 *     auto v = Load(d, float_ptr);
 * -------------------------------------------------------------------------- */
using Tag = ScalableTag<scl::Real>;  // SIMD tag for scl::Real

/* -----------------------------------------------------------------------------
 * TYPE ALIAS: IndexTag
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     SIMD descriptor tag for scl::Index type (int64_t).
 *
 * PURPOSE:
 *     Provides a convenient alias for ScalableTag<scl::Index>. Used for
 *     vectorized index operations, gather/scatter, and integer arithmetic.
 *
 * DEFINITION:
 *     using IndexTag = ScalableTag<scl::Index>;
 *
 * PRECONDITIONS:
 *     - scl::Index must be defined (typically int64_t)
 *
 * POSTCONDITIONS:
 *     - IndexTag represents the SIMD descriptor for scl::Index
 *
 * THREAD SAFETY:
 *     Safe - type alias is compile-time
 *
 * WHEN TO USE:
 *     - Vectorized index calculations
 *     - Gather/scatter operations with 64-bit indices
 *     - Integer arithmetic on scl::Index arrays
 * -------------------------------------------------------------------------- */
using IndexTag = ScalableTag<scl::Index>;  // SIMD tag for scl::Index

/* -----------------------------------------------------------------------------
 * TYPE ALIAS: ReinterpretTag
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     SIMD descriptor tag for unsigned integers matching scl::Real size.
 *
 * PURPOSE:
 *     Used for bitwise operations on floating-point data without type
 *     conversion. Allows manipulation of float/double bit patterns while
 *     preserving vector layout.
 *
 * DEFINITION:
 *     using ReinterpretTag = RebindToUnsigned<Tag>;
 *
 * PRECONDITIONS:
 *     - Tag must be defined
 *
 * POSTCONDITIONS:
 *     - ReinterpretTag has same vector width as Tag
 *     - Element type is unsigned integer with same size as scl::Real
 *
 * THREAD SAFETY:
 *     Safe - type alias is compile-time
 *
 * WHEN TO USE:
 *     - Bitwise masking of floating-point values
 *     - Sign bit manipulation
 *     - Fast absolute value (clear sign bit)
 *     - NaN/Inf detection via bit patterns
 * -------------------------------------------------------------------------- */
using ReinterpretTag = RebindToUnsigned<Tag>;  // Unsigned int tag for bitwise ops

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: lanes
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Get the number of elements (lanes) in a SIMD vector for scl::Real.
 *
 * PARAMETERS:
 *     None
 *
 * PRECONDITIONS:
 *     - Tag must be defined
 *
 * POSTCONDITIONS:
 *     - Returns the number of scl::Real elements in a vector register
 *     - Value depends on hardware and scl::Real size
 *
 * RETURN VALUE:
 *     Number of lanes (typically 4, 8, 16, etc. for floats on modern CPUs)
 *
 * COMPLEXITY:
 *     Time:  O(1) - compile-time constant
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - constexpr evaluation
 *
 * PERFORMANCE:
 *     Zero runtime cost - inlined to a compile-time constant
 *
 * USAGE NOTES:
 *     - Useful for loop bounds and buffer allocation
 *     - Value varies by architecture (AVX2 vs AVX-512 vs NEON)
 *     - For scl::Real = float on AVX2: returns 8
 *     - For scl::Real = double on AVX2: returns 4
 * -------------------------------------------------------------------------- */
inline size_t lanes();  // Returns Lanes(Tag())

// =============================================================================
// TYPE-BASED TAG SELECTION
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: GetSimdTag
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Get the appropriate SIMD tag for a given type T.
 *
 * SIGNATURE:
 *     template <typename T>
 *     auto GetSimdTag()
 *
 * PARAMETERS:
 *     T [template] - Type for which to get SIMD tag
 *
 * PRECONDITIONS:
 *     - T must be a valid SIMD element type
 *
 * POSTCONDITIONS:
 *     - Returns Tag if T == scl::Real
 *     - Returns IndexTag if T == scl::Index
 *     - Returns ScalableTag<T> otherwise
 *
 * RETURN VALUE:
 *     SIMD descriptor tag appropriate for type T
 *
 * COMPLEXITY:
 *     Time:  O(1) - compile-time dispatch
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - constexpr function
 *
 * PERFORMANCE:
 *     Zero runtime cost - all dispatch is compile-time
 *
 * USAGE PATTERN:
 *     using SimdTag = decltype(GetSimdTag<T>());
 *     const SimdTag d;
 *     auto v = Load(d, ptr);
 *
 * WHEN TO USE:
 *     - Generic template functions that need SIMD tags
 *     - Type-agnostic kernel implementations
 *     - Avoiding manual tag selection in templates
 * -------------------------------------------------------------------------- */
template <typename T>
inline auto GetSimdTag();  // Returns Tag, IndexTag, or ScalableTag<T>

/* -----------------------------------------------------------------------------
 * TYPE ALIAS: SimdTagFor
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Type alias for SIMD tag corresponding to type T.
 *
 * SIGNATURE:
 *     template <typename T>
 *     using SimdTagFor = [conditional type selection]
 *
 * PURPOSE:
 *     Provides a compile-time type alias for the SIMD tag of type T.
 *     Alternative to decltype(GetSimdTag<T>()) for use in type declarations.
 *
 * DEFINITION:
 *     std::conditional_t selecting Tag, IndexTag, or ScalableTag<T>
 *
 * PRECONDITIONS:
 *     - T must be a valid SIMD element type
 *
 * POSTCONDITIONS:
 *     - SimdTagFor<scl::Real> is Tag
 *     - SimdTagFor<scl::Index> is IndexTag
 *     - SimdTagFor<OtherType> is ScalableTag<OtherType>
 *
 * THREAD SAFETY:
 *     Safe - type alias is compile-time
 *
 * USAGE PATTERN:
 *     using SimdTag = SimdTagFor<T>;
 *     const SimdTag d;
 *     auto v = Load(d, ptr);
 *
 * WHEN TO USE:
 *     - Same as GetSimdTag but for type declarations
 *     - Template parameter deduction
 *     - Type traits and metaprogramming
 * -------------------------------------------------------------------------- */
template <typename T>
using SimdTagFor = std::conditional_t<
    std::is_same_v<T, Real>, Tag,
    std::conditional_t<std::is_same_v<T, Index>, IndexTag,
        ScalableTag<T>>>;

// =============================================================================
// COMMON HIGHWAY OPERATIONS (REFERENCE)
// =============================================================================

/* -----------------------------------------------------------------------------
 * NOTE: Highway Operations Reference
 * -----------------------------------------------------------------------------
 * The following operations are available via the namespace injection.
 * This is not an exhaustive list - see Highway documentation for complete API.
 *
 * MEMORY OPERATIONS:
 *     Load(d, ptr)           - Load aligned vector from memory
 *     LoadU(d, ptr)          - Load unaligned vector from memory
 *     Store(v, d, ptr)       - Store vector to aligned memory
 *     StoreU(v, d, ptr)      - Store vector to unaligned memory
 *     MaskedLoad(mask, d, ptr) - Conditional load
 *
 * ARITHMETIC:
 *     Add(v1, v2)            - Element-wise addition
 *     Sub(v1, v2)            - Element-wise subtraction
 *     Mul(v1, v2)            - Element-wise multiplication
 *     Div(v1, v2)            - Element-wise division
 *     Neg(v)                 - Element-wise negation
 *     Abs(v)                 - Element-wise absolute value
 *     Sqrt(v)                - Element-wise square root
 *     MulAdd(a, b, c)        - Fused multiply-add: a*b + c
 *     NegMulAdd(a, b, c)     - Fused neg-multiply-add: -(a*b) + c
 *
 * COMPARISON:
 *     Eq(v1, v2)             - Element-wise equality
 *     Ne(v1, v2)             - Element-wise inequality
 *     Lt(v1, v2)             - Element-wise less than
 *     Le(v1, v2)             - Element-wise less or equal
 *     Gt(v1, v2)             - Element-wise greater than
 *     Ge(v1, v2)             - Element-wise greater or equal
 *
 * MIN/MAX:
 *     Min(v1, v2)            - Element-wise minimum
 *     Max(v1, v2)            - Element-wise maximum
 *
 * LOGICAL:
 *     And(v1, v2)            - Bitwise AND
 *     Or(v1, v2)             - Bitwise OR
 *     Xor(v1, v2)            - Bitwise XOR
 *     Not(v)                 - Bitwise NOT
 *     AndNot(v1, v2)         - Bitwise AND-NOT: v1 & ~v2
 *
 * INITIALIZATION:
 *     Zero(d)                - Create vector of zeros
 *     Set(d, value)          - Create vector with all elements = value
 *     Iota(d, start)         - Create vector [start, start+1, start+2, ...]
 *
 * CONDITIONAL:
 *     IfThenElse(mask, true_val, false_val) - Conditional selection
 *     IfThenElseZero(mask, true_val)        - Select or zero
 *     IfThenZeroElse(mask, false_val)       - Zero or select
 *
 * REDUCTIONS:
 *     SumOfLanes(d, v)       - Horizontal sum of all lanes
 *     MinOfLanes(d, v)       - Horizontal minimum
 *     MaxOfLanes(d, v)       - Horizontal maximum
 *
 * ADVANCED:
 *     TableLookupLanes(v, indices) - Permute vector by lane indices
 *     Broadcast<N>(v)              - Broadcast lane N to all lanes
 *     Reverse(d, v)                - Reverse lane order
 *
 * MATHEMATICAL (hwy/contrib/math):
 *     Exp(d, v)              - Element-wise e^x
 *     Log(d, v)              - Element-wise natural logarithm
 *     Sin(d, v)              - Element-wise sine
 *     Cos(d, v)              - Element-wise cosine
 *     Tanh(d, v)             - Element-wise hyperbolic tangent
 *
 * TYPE CONVERSION:
 *     ConvertTo(d_to, v)     - Convert vector to different element type
 *     PromoteTo(d_to, v)     - Promote to wider type (int32->int64, etc.)
 *     DemoteTo(d_to, v)      - Demote to narrower type
 *     BitCast(d_to, v)       - Reinterpret bits as different type
 *
 * USAGE NOTES:
 *     - All operations are architecture-agnostic
 *     - Highway automatically selects best instruction set
 *     - Operations compile to single CPU instructions when possible
 *     - Fallback to scalar code when SIMD unavailable
 * -------------------------------------------------------------------------- */

} // namespace scl::simd
