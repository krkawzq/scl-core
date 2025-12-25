#include "scl/math/saturation.hpp"
#include "scl/threading.hpp"
#include "scl/common.hpp"
#include <algorithm>
#include <cmath>

namespace scl {
namespace math {

void saturation(
    const Float* input,
    Float* output,
    Size size,
    Float min_val,
    Float max_val
) {
    // [Owner: AI]
    // Simple clamping operation
    
    threading::parallel_for(0, size, [&](Size i) {
        output[i] = std::max(min_val, std::min(max_val, input[i]));
    });
}

void smooth_saturation(
    const Float* input,
    Float* output,
    Size size,
    Float min_val,
    Float max_val,
    Float center
) {
    // [Owner: AI]
    // Smooth saturation using tanh
    
    const Float range = max_val - min_val;
    const Float scale = range / static_cast<Float>(2.0);
    
    threading::parallel_for(0, size, [&](Size i) {
        const Float x_normalized = (input[i] - center) / scale;
        output[i] = scale * static_cast<Float>(std::tanh(static_cast<double>(x_normalized))) + center;
    });
}

} // namespace math
} // namespace scl

