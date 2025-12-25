#include "scl/math/diffusion.hpp"
#include "scl/threading.hpp"
#include "scl/common.hpp"
#include <algorithm>
#include <cmath>

namespace scl {
namespace math {

void diffusion_2d(
    const Float* input,
    Float* output,
    Size width,
    Size height,
    Float alpha,
    Int boundary_type
) {
    // [Owner: AI]
    // Simple 5-point stencil diffusion operator
    
    // Copy input to output first
    const Size total_size = width * height;
    std::copy(input, input + total_size, output);
    
    // Apply diffusion kernel
    threading::parallel_for(1, height - 1, [&](Size j) {
        for (Size i = 1; i < width - 1; ++i) {
            const Size idx = j * width + i;
            
            // 5-point stencil: center + 4 neighbors
            Float laplacian = static_cast<Float>(
                input[(j-1) * width + i] +  // top
                input[(j+1) * width + i] +  // bottom
                input[j * width + (i-1)] +  // left
                input[j * width + (i+1)] -  // right
                4.0 * input[idx]            // center
            );
            
            output[idx] = static_cast<Float>(input[idx] + alpha * laplacian);
        }
    });
    
    // Handle boundaries (simplified: zero-padding)
    // TODO: Implement proper boundary conditions based on boundary_type
}

void anisotropic_diffusion_2d(
    const Float* input,
    Float* output,
    Size width,
    Size height,
    Float kappa,
    Int iterations
) {
    // [Owner: AI]
    // Perona-Malik anisotropic diffusion
    
    // Allocate temporary buffer
    std::vector<Float> temp(width * height);
    Float* current = output;
    Float* next = temp.data();
    
    // Copy input to current
    std::copy(input, input + width * height, current);
    
    for (Int iter = 0; iter < iterations; ++iter) {
        threading::parallel_for(1, height - 1, [&](Size j) {
            for (Size i = 1; i < width - 1; ++i) {
                const Size idx = j * width + i;
                
                // Compute gradients
                Float grad_n = std::abs(current[(j-1) * width + i] - current[idx]);
                Float grad_s = std::abs(current[(j+1) * width + i] - current[idx]);
                Float grad_e = std::abs(current[j * width + (i+1)] - current[idx]);
                Float grad_w = std::abs(current[j * width + (i-1)] - current[idx]);
                
                // Perona-Malik diffusion coefficient
                auto g = [kappa](Float grad) {
                    const Float kappa_sq = kappa * kappa;
                    return static_cast<Float>(1.0) / (static_cast<Float>(1.0) + (grad * grad) / kappa_sq);
                };
                
                // Update
                Float c_n = g(grad_n);
                Float c_s = g(grad_s);
                Float c_e = g(grad_e);
                Float c_w = g(grad_w);
                
                const Float factor = static_cast<Float>(0.25);
                next[idx] = current[idx] + factor * (
                    c_n * (current[(j-1) * width + i] - current[idx]) +
                    c_s * (current[(j+1) * width + i] - current[idx]) +
                    c_e * (current[j * width + (i+1)] - current[idx]) +
                    c_w * (current[j * width + (i-1)] - current[idx])
                );
            }
        });
        
        // Swap buffers
        std::swap(current, next);
    }
    
    // If odd number of iterations, copy back to output
    if (iterations % 2 == 1) {
        std::copy(temp.begin(), temp.end(), output);
    }
}

} // namespace math
} // namespace scl

