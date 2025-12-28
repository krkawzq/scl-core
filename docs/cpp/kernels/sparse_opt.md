# sparse_opt.hpp

> scl/kernel/sparse_opt.hpp · Sparse optimization methods with SIMD optimization

## Overview

This file provides high-performance sparse optimization methods for regularized regression problems. It implements coordinate descent, proximal gradient methods, and iterative thresholding algorithms with SIMD optimization for L1, L2, and elastic net regularization.

This file provides:
- Lasso regression via coordinate descent
- Elastic net regression via coordinate descent
- Proximal gradient descent (ISTA)
- Accelerated proximal gradient (FISTA)
- Iterative hard thresholding (IHT)
- Group Lasso for grouped sparsity
- Sparse logistic regression
- Regularization path computation
- SIMD-optimized proximal operators

**Header**: `#include "scl/kernel/sparse_opt.hpp"`

---

## Main APIs

### lasso_coordinate_descent

**SUMMARY:**
Solve Lasso regression via coordinate descent with warm restarts.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void lasso_coordinate_descent(
    const Sparse<T, IsCSR>& X,       // Design matrix [n_samples x n_features]
    Array<const Real> y,              // Target vector [n_samples]
    Real alpha,                       // L1 regularization strength
    Array<Real> coefficients,         // Output coefficients [n_features]
    Index max_iter = 1000,            // Maximum iterations
    Real tol = 1e-4                   // Convergence tolerance
);
```

**PARAMETERS:**
- X            [in]     Sparse design matrix, shape (n_samples, n_features)
- y            [in]     Target vector, size n_samples
- alpha        [in]     L1 regularization strength (lambda)
- coefficients [out]    Pre-allocated output, size n_features
- max_iter     [in]     Maximum number of iterations
- tol          [in]     Relative tolerance for convergence

**PRECONDITIONS:**
- X must be valid sparse matrix
- y.len >= X.rows()
- coefficients.len >= X.cols()
- alpha >= 0
- max_iter > 0

**POSTCONDITIONS:**
- coefficients contains solution to: min (1/2n)||y - X*coef||^2 + alpha*||coef||_1
- Sparse solution (many coefficients exactly zero)

**ALGORITHM:**
1. Initialize residuals r = y, coefficients = 0
2. Precompute column squared norms: ||X[:, j]||^2
3. For each iteration until convergence:
   - For each coordinate j:
     - Compute rho = X[:, j]^T * r + ||X[:, j]||^2 * coef[j]
     - Apply soft thresholding: coef[j] = S(rho, lambda) / ||X[:, j]||^2
     - Update residuals: r -= delta * X[:, j]
4. Check convergence via relative coefficient change

**COMPLEXITY:**
- Time:  O(max_iter * n_features * avg_col_nnz)
- Space: O(n_samples + n_features) for residuals and column norms

**THREAD SAFETY:**
Safe - uses thread-local memory allocation

---

### elastic_net_coordinate_descent

**SUMMARY:**
Solve elastic net regression via coordinate descent.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void elastic_net_coordinate_descent(
    const Sparse<T, IsCSR>& X,       // Design matrix
    Array<const Real> y,              // Target vector
    Real alpha,                       // Overall regularization strength
    Real l1_ratio,                    // L1/L2 mixing ratio [0, 1]
    Array<Real> coefficients,         // Output coefficients
    Index max_iter = 1000,
    Real tol = 1e-4
);
```

**PARAMETERS:**
- X            [in]     Sparse design matrix
- y            [in]     Target vector
- alpha        [in]     Overall regularization strength
- l1_ratio     [in]     Mixing parameter: 1.0 = Lasso, 0.0 = Ridge
- coefficients [out]    Output coefficients
- max_iter     [in]     Maximum iterations
- tol          [in]     Convergence tolerance

**PRECONDITIONS:**
- l1_ratio in [0, 1]
- Other preconditions same as lasso_coordinate_descent

**POSTCONDITIONS:**
- coefficients contains solution to:
  min (1/2n)||y - X*coef||^2 + alpha*(l1_ratio*||coef||_1 + (1-l1_ratio)/2*||coef||_2^2)

**ALGORITHM:**
Same as Lasso but with modified update:
- coef[j] = S(rho, l1_lambda) / (||X[:, j]||^2 + l2_lambda)

**COMPLEXITY:**
- Time:  O(max_iter * n_features * avg_col_nnz)
- Space: O(n_samples + n_features)

**THREAD SAFETY:**
Safe

---

### proximal_gradient

**SUMMARY:**
Solve regularized regression via proximal gradient descent (ISTA).

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void proximal_gradient(
    const Sparse<T, IsCSR>& X,       // Design matrix
    Array<const Real> y,              // Target vector
    Real alpha,                       // Regularization strength
    RegularizationType reg_type,      // L1, L2, ELASTIC_NET, SCAD, MCP
    Array<Real> coefficients,         // Output coefficients
    Index max_iter = 1000,
    Real tol = 1e-4
);
```

**PARAMETERS:**
- X            [in]     Sparse design matrix
- y            [in]     Target vector
- alpha        [in]     Regularization strength
- reg_type     [in]     Type of regularization penalty
- coefficients [out]    Output coefficients
- max_iter     [in]     Maximum iterations
- tol          [in]     Convergence tolerance

**PRECONDITIONS:**
- Same as lasso_coordinate_descent

**POSTCONDITIONS:**
- coefficients contains regularized solution
- Sparsity depends on reg_type (L1, SCAD, MCP induce sparsity)

**ALGORITHM:**
1. Estimate Lipschitz constant L via power iteration
2. Set step size t = 1/L
3. For each iteration:
   - Compute gradient: grad = X^T * (X*coef - y)
   - Gradient step: z = coef - t * grad
   - Proximal step: coef = prox_{t*alpha}(z)
4. Check convergence

**COMPLEXITY:**
- Time:  O(max_iter * nnz) where nnz is number of non-zeros in X
- Space: O(n_samples + n_features)

**THREAD SAFETY:**
Safe

---

### fista

**SUMMARY:**
Solve Lasso via accelerated proximal gradient (FISTA).

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void fista(
    const Sparse<T, IsCSR>& X,       // Design matrix
    Array<const Real> y,              // Target vector
    Real alpha,                       // L1 regularization strength
    Array<Real> coefficients,         // Output coefficients
    Index max_iter = 1000,
    Real tol = 1e-4
);
```

**PARAMETERS:**
- X            [in]     Sparse design matrix
- y            [in]     Target vector
- alpha        [in]     L1 regularization strength
- coefficients [out]    Output coefficients
- max_iter     [in]     Maximum iterations
- tol          [in]     Convergence tolerance

**PRECONDITIONS:**
- Same as lasso_coordinate_descent

**POSTCONDITIONS:**
- Same as lasso_coordinate_descent

**ALGORITHM:**
FISTA with Nesterov momentum:
1. Initialize z = coef = 0, t = 1
2. For each iteration:
   - Gradient step on z
   - Proximal step (soft thresholding)
   - Momentum update: t_new = (1 + sqrt(1 + 4t^2)) / 2
   - z = coef + (t-1)/t_new * (coef - coef_old)

**COMPLEXITY:**
- Time:  O(max_iter * nnz) - converges O(1/k^2) vs O(1/k) for ISTA
- Space: O(n_samples + 2 * n_features)

**NUMERICAL NOTES:**
- Faster convergence than ISTA for ill-conditioned problems
- May oscillate near optimum

**THREAD SAFETY:**
Safe

---

### iht

**SUMMARY:**
Iterative hard thresholding for fixed-sparsity recovery.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void iht(
    const Sparse<T, IsCSR>& X,       // Design matrix
    Array<const Real> y,              // Target vector
    Index sparsity_level,             // Number of non-zeros to keep
    Array<Real> coefficients,         // Output coefficients
    Index max_iter = 1000
);
```

**PARAMETERS:**
- X              [in]     Sparse design matrix
- y              [in]     Target vector
- sparsity_level [in]     Desired number of non-zero coefficients
- coefficients   [out]    Output coefficients
- max_iter       [in]     Maximum iterations

**PRECONDITIONS:**
- sparsity_level in (0, n_features]
- Other preconditions same as lasso_coordinate_descent

**POSTCONDITIONS:**
- coefficients has exactly sparsity_level non-zero entries (or fewer if converged)
- Non-zeros are largest magnitude coefficients

**ALGORITHM:**
1. For each iteration:
   - Gradient step: coef -= (1/L) * X^T * (X*coef - y)
   - Hard thresholding: keep only top-k by magnitude
2. Repeat until convergence

**COMPLEXITY:**
- Time:  O(max_iter * (nnz + n_features * log(n_features)))
- Space: O(n_samples + n_features)

**THREAD SAFETY:**
Safe

---

### group_lasso

**SUMMARY:**
Group Lasso for grouped feature selection.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void group_lasso(
    const Sparse<T, IsCSR>& X,       // Design matrix
    Array<const Real> y,              // Target vector
    const Index* group_indices,       // Feature indices per group (concatenated)
    const Index* group_offsets,       // Offsets into group_indices [n_groups + 1]
    Index n_groups,                   // Number of groups
    Real alpha,                       // Regularization strength
    Array<Real> coefficients,         // Output coefficients
    Index max_iter = 1000
);
```

**PARAMETERS:**
- X              [in]     Sparse design matrix
- y              [in]     Target vector
- group_indices  [in]     Feature indices for each group (contiguous)
- group_offsets  [in]     Start/end offsets for each group
- n_groups       [in]     Number of feature groups
- alpha          [in]     Regularization strength
- coefficients   [out]    Output coefficients
- max_iter       [in]     Maximum iterations

**PRECONDITIONS:**
- group_offsets has length n_groups + 1
- group_indices covers all features

**POSTCONDITIONS:**
- Entire groups are zeroed out together (group sparsity)
- Minimizes: (1/2n)||y - X*coef||^2 + alpha * sum_g(sqrt(|g|) * ||coef_g||_2)

**ALGORITHM:**
Block coordinate descent:
1. For each group:
   - Compute group gradient
   - Apply group soft thresholding (L2 norm threshold)
   - Zero entire group if below threshold

**COMPLEXITY:**
- Time:  O(max_iter * n_groups * avg_group_nnz)
- Space: O(n_samples + max_group_size)

**THREAD SAFETY:**
Safe

---

### sparse_logistic_regression

**SUMMARY:**
L1-regularized logistic regression via coordinate descent.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void sparse_logistic_regression(
    const Sparse<T, IsCSR>& X,       // Design matrix
    Array<const Index> y_binary,      // Binary labels {0, 1}
    Real alpha,                       // L1 regularization strength
    Array<Real> coefficients,         // Output coefficients
    Index max_iter = 1000
);
```

**PARAMETERS:**
- X            [in]     Sparse design matrix
- y_binary     [in]     Binary class labels (0 or 1)
- alpha        [in]     L1 regularization strength
- coefficients [out]    Output coefficients
- max_iter     [in]     Maximum iterations

**PRECONDITIONS:**
- y_binary contains only 0 and 1 values
- Other preconditions same as lasso_coordinate_descent

**POSTCONDITIONS:**
- coefficients contains sparse logistic regression solution
- Maximizes: sum(y*log(p) + (1-y)*log(1-p)) - alpha*||coef||_1

**ALGORITHM:**
Iteratively reweighted least squares with L1 penalty:
1. Compute probabilities: p = sigmoid(X*coef + intercept)
2. Compute weights: w = p*(1-p)
3. Compute working response: z = eta + (y-p)/w
4. Solve weighted L1-regularized regression
5. Repeat until convergence

**COMPLEXITY:**
- Time:  O(max_iter * n_features * avg_col_nnz)
- Space: O(n_samples + n_features)

**THREAD SAFETY:**
Safe

---

### lasso_path

**SUMMARY:**
Compute Lasso solution path for multiple regularization values.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void lasso_path(
    const Sparse<T, IsCSR>& X,       // Design matrix
    Array<const Real> y,              // Target vector
    Array<const Real> alphas,         // Regularization values (decreasing order)
    Real* coefficient_paths,          // Output [n_alphas x n_features]
    Index max_iter = 1000
);
```

**PARAMETERS:**
- X                  [in]     Sparse design matrix
- y                  [in]     Target vector
- alphas             [in]     Array of regularization values
- coefficient_paths  [out]    Matrix of solutions, size n_alphas * n_features
- max_iter           [in]     Maximum iterations per alpha

**PRECONDITIONS:**
- alphas should be in decreasing order for efficient warm starts
- coefficient_paths has capacity >= alphas.len * X.cols()

**POSTCONDITIONS:**
- coefficient_paths[a * n_features + j] = coefficient j at alpha[a]

**ALGORITHM:**
1. Start with largest alpha (most regularization)
2. Use solution as warm start for next smaller alpha
3. Repeat for all alphas

**COMPLEXITY:**
- Time:  O(n_alphas * avg_iters * n_features * avg_col_nnz)
- Space: O(n_samples + n_features)

**NUMERICAL NOTES:**
- Warm starts significantly reduce total iterations
- Order alphas from large to small for best performance

**THREAD SAFETY:**
Safe

---

## Proximal Operators

### prox_l1

**SUMMARY:**
SIMD-optimized soft thresholding operator.

**SIGNATURE:**
```cpp
void prox_l1(
    Array<Real> x,     // Input/output vector
    Real lambda        // Threshold
);
```

**POSTCONDITIONS:**
- x[i] = sign(x[i]) * max(|x[i]| - lambda, 0)

**COMPLEXITY:**
- Time:  O(n) with SIMD acceleration
- Space: O(1)

---

### prox_elastic_net

**SUMMARY:**
SIMD-optimized elastic net proximal operator.

**SIGNATURE:**
```cpp
void prox_elastic_net(
    Array<Real> x,     // Input/output vector
    Real lambda,       // Regularization strength
    Real l1_ratio      // L1/L2 mixing ratio
);
```

**POSTCONDITIONS:**
- Applies soft thresholding then L2 shrinkage

**COMPLEXITY:**
- Time:  O(n) with SIMD acceleration
- Space: O(1)

---

## Configuration

Default parameters in `scl::kernel::sparse_opt::config`:

- `DEFAULT_ALPHA = 1.0`: Default regularization strength
- `DEFAULT_L1_RATIO = 1.0`: Default mixing ratio (pure Lasso)
- `DEFAULT_MAX_ITER = 1000`: Default maximum iterations
- `DEFAULT_TOL = 1e-4`: Default convergence tolerance
- `PREFETCH_DISTANCE = 8`: Cache prefetch distance
- `SIMD_THRESHOLD = 16`: Minimum size for SIMD path
- `EPS = 1e-12`: Numerical stability constant
- `LIPSCHITZ_SCALING = 1.5`: Safety factor for step size

---

## Regularization Types

```cpp
enum class RegularizationType {
    L1,              // Lasso (sparsity-inducing)
    L2,              // Ridge (shrinkage)
    ELASTIC_NET,     // L1 + L2 combination
    SCAD,            // Smoothly Clipped Absolute Deviation
    MCP              // Minimax Concave Penalty
};
```

---

## Performance Notes

### SIMD Optimization

- Soft thresholding uses SIMD masked operations
- Sparse matrix-vector products use 4-way unrolling
- Dot products use multi-accumulator pattern

### Memory Efficiency

- Pre-allocated buffers for residuals and gradients
- Column norms computed once and cached
- Warm starts reduce memory pressure

### Convergence

- Coordinate descent: O(1/k) convergence rate
- FISTA: O(1/k^2) convergence rate
- IHT: Linear convergence for well-conditioned problems

---

## Use Cases

### Feature Selection

```cpp
// Lasso for automatic feature selection
scl::kernel::sparse_opt::lasso_coordinate_descent(
    X, y, alpha, coefficients, 1000, 1e-4);

// Non-zero coefficients indicate selected features
for (Index j = 0; j < n_features; ++j) {
    if (std::abs(coefficients[j]) > 1e-8) {
        // Feature j is selected
    }
}
```

### Regularization Path

```cpp
// Compute solution path for model selection
Real alphas[] = {1.0, 0.1, 0.01, 0.001};
Real* paths = new Real[4 * n_features];

scl::kernel::sparse_opt::lasso_path(
    X, y, Array<const Real>(alphas, 4), paths, 1000);

// Cross-validate to select best alpha
```

### Sparse Classification

```cpp
// L1-regularized logistic regression
scl::kernel::sparse_opt::sparse_logistic_regression(
    X, y_binary, alpha, coefficients, 1000);

// Sparse model for interpretability
```

---

## See Also

- [Sparse Matrix Operations](sparse.md) - General sparse matrix utilities
- [Normalization](normalization.md) - Data preprocessing
- [Scaling](scale.md) - Feature scaling
