# Sparse Optimization Kernels

Sparse optimization methods with SIMD acceleration for L1/L2 regularization, Lasso, Elastic Net, and advanced sparse regression.

**Location**: `scl/kernel/sparse_opt.hpp`

**Strategic Position**: Tier 4 - Advanced Foundation (Sparse + Nonlinear)

**Applications**:
- Sparse regression (Lasso, Elastic Net)
- Feature selection
- Sparse PCA
- Compressed sensing
- L1-regularized logistic regression

---

## RegularizationType

**SUMMARY:**
Enumeration of supported regularization types.

**SIGNATURE:**
```cpp
enum class RegularizationType {
    L1,           // L1 norm (sum of absolute values)
    L2,           // L2 norm (sum of squares)
    ELASTIC_NET,  // Combination of L1 and L2
    SCAD,         // Smoothly Clipped Absolute Deviation
    MCP           // Minimax Concave Penalty
};
```

**VALUES:**
- L1: Lasso penalty, induces sparsity
- L2: Ridge penalty, shrinkage only
- ELASTIC_NET: Hybrid L1+L2 penalty
- SCAD: Non-convex penalty with oracle properties
- MCP: Non-convex penalty with strong sparsity

---

## prox_l1

**SUMMARY:**
Proximal operator for L1 penalty (SIMD-optimized soft thresholding).

**SIGNATURE:**
```cpp
void prox_l1(
    Array<Real> x,      // Values to threshold [in/out]
    Real lambda         // Threshold parameter
);
```

**PARAMETERS:**
- x      [in,out] Array of coefficients to apply soft thresholding
- lambda [in]     Threshold value (must be non-negative)

**PRECONDITIONS:**
- x.len > 0
- lambda >= 0

**POSTCONDITIONS:**
- x[i] = sign(x[i]) * max(|x[i]| - lambda, 0) for all i
- Applied in-place with SIMD optimization

**ALGORITHM:**
SIMD soft thresholding:
1. For x > lambda: result = x - lambda
2. For x < -lambda: result = x + lambda
3. Otherwise: result = 0

**COMPLEXITY:**
- Time:  O(n) with SIMD acceleration
- Space: O(1)

**THREAD SAFETY:**
Unsafe - modifies x in-place

**MUTABILITY:**
INPLACE - modifies x array

**NUMERICAL NOTES:**
Uses SIMD masking for branch-free thresholding

---

## prox_elastic_net

**SUMMARY:**
Proximal operator for Elastic Net penalty (L1 + L2 combined).

**SIGNATURE:**
```cpp
void prox_elastic_net(
    Array<Real> x,
    Real lambda,
    Real l1_ratio       // Mixing parameter in [0,1]
);
```

**PARAMETERS:**
- x        [in,out] Coefficients to threshold
- lambda   [in]     Overall regularization strength
- l1_ratio [in]     L1 mixing ratio (0=pure L2, 1=pure L1)

**PRECONDITIONS:**
- x.len > 0
- lambda >= 0
- l1_ratio in [0, 1]

**POSTCONDITIONS:**
- Applies elastic net proximal operator: soft threshold then L2 scaling
- Formula: prox(x) = soft_threshold(x, l1_lambda) / (1 + l2_lambda)
- Where l1_lambda = lambda * l1_ratio, l2_lambda = lambda * (1 - l1_ratio)

**ALGORITHM:**
1. Compute l1_lambda = lambda * l1_ratio
2. Compute l2_scale = 1 / (1 + lambda * (1 - l1_ratio))
3. Apply soft threshold with l1_lambda
4. Scale by l2_scale

**COMPLEXITY:**
- Time:  O(n) with SIMD
- Space: O(1)

**THREAD SAFETY:**
Unsafe - modifies x in-place

**MUTABILITY:**
INPLACE

---

## lasso_coordinate_descent

**SUMMARY:**
Lasso regression via coordinate descent (minimize squared loss + L1 penalty).

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void lasso_coordinate_descent(
    const Sparse<T, IsCSR>& X,          // Feature matrix [n_samples x n_features]
    Array<const Real> y,                 // Target values [n_samples]
    Real alpha,                          // Regularization strength
    Array<Real> coefficients,            // Output coefficients [n_features]
    Index max_iter = 1000,               // Maximum iterations
    Real tol = 1e-4                      // Convergence tolerance
);
```

**PARAMETERS:**
- X            [in]     Feature matrix (n_samples x n_features)
- y            [in]     Target vector [n_samples]
- alpha        [in]     L1 regularization strength (must be >= 0)
- coefficients [out]    Solution coefficients [n_features]
- max_iter     [in]     Maximum coordinate descent iterations
- tol          [in]     Convergence tolerance for coefficient changes

**PRECONDITIONS:**
- X.valid() must be true
- X.rows() == y.len
- coefficients.len >= X.cols()
- alpha >= 0
- max_iter > 0

**POSTCONDITIONS:**
- coefficients contains Lasso solution
- Minimizes: (1/2n) * ||y - X*coef||^2 + alpha * ||coef||_1
- Sparse solution (many coefficients are exactly zero)

**ALGORITHM:**
Coordinate descent with soft thresholding:
1. Initialize coefficients to zero, residuals r = y
2. Precompute column squared norms
3. For each iteration:
   a. For each feature j:
      - Compute rho = X[:,j]^T * r + norm_sq * coef[j]
      - Update coef[j] = soft_threshold(rho, lambda) / norm_sq
      - Update residuals if coefficient changed
   b. Check convergence
4. Return when converged or max_iter reached

**COMPLEXITY:**
- Time:  O(max_iter * nnz) for sparse X
- Space: O(n_samples + n_features)

**THREAD SAFETY:**
Safe - no concurrent modification

**MUTABILITY:**
CONST for X and y, writes to coefficients

**NUMERICAL NOTES:**
- Uses efficient sparse-dense dot products
- Exploits sparsity for CSC format (fast column access)
- CSR format requires row iteration (slower but still efficient)

---

## elastic_net_coordinate_descent

**SUMMARY:**
Elastic Net regression via coordinate descent (L1 + L2 penalty).

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void elastic_net_coordinate_descent(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    Real l1_ratio,                       // L1 mixing ratio in [0,1]
    Array<Real> coefficients,
    Index max_iter = 1000,
    Real tol = 1e-4
);
```

**PARAMETERS:**
- X            [in]  Feature matrix
- y            [in]  Target vector
- alpha        [in]  Overall regularization strength
- l1_ratio     [in]  L1 mixing parameter (0=Ridge, 1=Lasso)
- coefficients [out] Solution coefficients
- max_iter     [in]  Maximum iterations
- tol          [in]  Convergence tolerance

**PRECONDITIONS:**
- X.valid() and y.len == X.rows()
- coefficients.len >= X.cols()
- alpha >= 0
- l1_ratio in [0, 1]

**POSTCONDITIONS:**
- Minimizes: (1/2n) * ||y - X*coef||^2 + alpha * (l1_ratio * ||coef||_1 + (1-l1_ratio)/2 * ||coef||_2^2)
- Combines sparsity (L1) and grouping (L2) effects

**ALGORITHM:**
Similar to Lasso but with modified update:
- l1_lambda = alpha * l1_ratio * n_samples
- l2_lambda = alpha * (1 - l1_ratio) * n_samples
- Update: coef[j] = soft_threshold(rho, l1_lambda) / (norm_sq + l2_lambda)

**COMPLEXITY:**
- Time:  O(max_iter * nnz)
- Space: O(n_samples + n_features)

**THREAD SAFETY:**
Safe

**NUMERICAL NOTES:**
- When l1_ratio=0: pure Ridge (no sparsity)
- When l1_ratio=1: pure Lasso
- Intermediate values: sparse + grouped solutions

---

## proximal_gradient

**SUMMARY:**
Proximal gradient descent (ISTA) for sparse regression with various penalties.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void proximal_gradient(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    RegularizationType reg_type,         // L1, L2, ELASTIC_NET, SCAD, or MCP
    Array<Real> coefficients,
    Index max_iter = 1000,
    Real tol = 1e-4
);
```

**PARAMETERS:**
- X            [in]  Feature matrix
- y            [in]  Target vector
- alpha        [in]  Regularization strength
- reg_type     [in]  Type of regularization
- coefficients [out] Solution coefficients
- max_iter     [in]  Maximum iterations
- tol          [in]  Convergence tolerance

**PRECONDITIONS:**
- X.valid() and y.len == X.rows()
- coefficients.len >= X.cols()
- alpha >= 0

**POSTCONDITIONS:**
- coefficients contains solution for specified regularization type
- Supports non-convex penalties (SCAD, MCP) for stronger sparsity

**ALGORITHM:**
Iterative Shrinkage-Thresholding Algorithm (ISTA):
1. Estimate Lipschitz constant L via power iteration
2. Set step_size = 1/L
3. For each iteration:
   a. Compute residuals r = X*coef - y
   b. Compute gradient grad = X^T * r
   c. Gradient step: coef = coef - step_size * grad
   d. Proximal step: apply regularization operator
4. Return when converged

**COMPLEXITY:**
- Time:  O(max_iter * nnz) plus O(power_iter * nnz) for initialization
- Space: O(n_samples + n_features)

**THREAD SAFETY:**
Safe

**NUMERICAL NOTES:**
- Uses power iteration to estimate largest singular value
- Step size chosen for guaranteed convergence
- Non-convex penalties (SCAD/MCP) may converge to local minimum

---

## fista

**SUMMARY:**
Fast Iterative Shrinkage-Thresholding Algorithm (accelerated proximal gradient for Lasso).

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void fista(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter = 1000,
    Real tol = 1e-4
);
```

**PARAMETERS:**
- X            [in]  Feature matrix
- y            [in]  Target vector
- alpha        [in]  L1 regularization strength
- coefficients [out] Solution coefficients
- max_iter     [in]  Maximum iterations
- tol          [in]  Convergence tolerance

**PRECONDITIONS:**
- X.valid() and y.len == X.rows()
- coefficients.len >= X.cols()
- alpha >= 0

**POSTCONDITIONS:**
- Solves Lasso problem with accelerated convergence
- Typically faster than ISTA by factor of 2-10x

**ALGORITHM:**
Nesterov-accelerated proximal gradient:
1. Initialize z = coef = 0, t = 1
2. For each iteration:
   a. Compute gradient at z (momentum point)
   b. Gradient step: coef = z - step_size * grad
   c. Proximal L1 step on coef
   d. Compute momentum: t_new = (1 + sqrt(1 + 4t^2)) / 2
   e. Update z = coef + ((t-1)/t_new) * (coef - coef_old)
   f. Set t = t_new
3. Return when converged

**COMPLEXITY:**
- Time:  O(max_iter * nnz)
- Space: O(n_samples + 2*n_features) (extra storage for momentum)

**THREAD SAFETY:**
Safe

**NUMERICAL NOTES:**
- Optimal O(1/k^2) convergence rate vs O(1/k) for ISTA
- Requires extra momentum variable z

---

## iht

**SUMMARY:**
Iterative Hard Thresholding for sparse recovery with fixed sparsity level.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void iht(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Index sparsity_level,                // Number of non-zero coefficients
    Array<Real> coefficients,
    Index max_iter = 1000
);
```

**PARAMETERS:**
- X               [in]  Feature matrix
- y               [in]  Target vector
- sparsity_level  [in]  Desired number of non-zero coefficients
- coefficients    [out] Solution coefficients
- max_iter        [in]  Maximum iterations

**PRECONDITIONS:**
- X.valid() and y.len == X.rows()
- coefficients.len >= X.cols()
- sparsity_level > 0 and sparsity_level <= X.cols()

**POSTCONDITIONS:**
- coefficients has exactly sparsity_level non-zero entries
- Minimizes ||y - X*coef||^2 subject to ||coef||_0 <= sparsity_level

**ALGORITHM:**
Hard thresholding operator:
1. Estimate step size from Lipschitz constant
2. For each iteration:
   a. Compute gradient of squared loss
   b. Gradient step
   c. Keep only top-k coefficients by absolute value (hard threshold)
3. Return after max_iter

**COMPLEXITY:**
- Time:  O(max_iter * (nnz + n_features*log(sparsity_level)))
- Space: O(n_samples + 2*n_features)

**THREAD SAFETY:**
Safe

**NUMERICAL NOTES:**
- Solves non-convex L0 problem (NP-hard)
- No convergence guarantees, but works well in practice
- Uses partial sort (O(n + k*log(k))) for top-k selection

---

## lasso_path

**SUMMARY:**
Compute Lasso solution path for multiple regularization values.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void lasso_path(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    Array<const Real> alphas,            // Sorted regularization values
    Real* coefficient_paths,             // Output paths [n_alphas x n_features]
    Index max_iter = 1000
);
```

**PARAMETERS:**
- X                  [in]  Feature matrix
- y                  [in]  Target vector
- alphas             [in]  Array of alpha values (typically decreasing)
- coefficient_paths  [out] Solution for each alpha [n_alphas x n_features]
- max_iter           [in]  Maximum iterations per alpha

**PRECONDITIONS:**
- X.valid() and y.len == X.rows()
- alphas.len > 0, all alphas[i] >= 0
- coefficient_paths has size >= alphas.len * X.cols()

**POSTCONDITIONS:**
- coefficient_paths[i*n_features : (i+1)*n_features] contains solution for alphas[i]
- Uses warm starts for efficiency

**ALGORITHM:**
1. Start with zero coefficients
2. For each alpha value:
   a. Use previous solution as warm start
   b. Run coordinate descent
   c. Store solution
3. Return full path

**COMPLEXITY:**
- Time:  O(n_alphas * max_iter * nnz)
- Space: O(n_samples + 2*n_features)

**THREAD SAFETY:**
Safe

**NUMERICAL NOTES:**
- Warm starts significantly reduce computation
- Typically compute path from large to small alpha (less sparse to more sparse)

---

## group_lasso

**SUMMARY:**
Group Lasso via block coordinate descent (promotes group sparsity).

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void group_lasso(
    const Sparse<T, IsCSR>& X,
    Array<const Real> y,
    const Index* group_indices,          // Feature assignments [n_features]
    const Index* group_offsets,          // Group boundaries [n_groups+1]
    Index n_groups,
    Real alpha,
    Array<Real> coefficients,
    Index max_iter = 1000
);
```

**PARAMETERS:**
- X              [in]  Feature matrix
- y              [in]  Target vector
- group_indices  [in]  Feature indices grouped [n_features]
- group_offsets  [in]  Group start positions [n_groups+1]
- n_groups       [in]  Number of groups
- alpha          [in]  Regularization strength
- coefficients   [out] Solution coefficients
- max_iter       [in]  Maximum iterations

**PRECONDITIONS:**
- X.valid() and y.len == X.rows()
- coefficients.len >= X.cols()
- group_offsets[0] = 0, group_offsets[n_groups] = X.cols()
- group_offsets is non-decreasing
- alpha >= 0

**POSTCONDITIONS:**
- Entire groups are selected or zeroed together
- Penalty: alpha * sum_g sqrt(|g|) * ||coef_g||_2

**ALGORITHM:**
Block coordinate descent with group soft thresholding:
1. For each iteration:
   a. For each group g:
      - Compute group gradient
      - Compute group L2 norm
      - Apply group soft thresholding
      - Update all features in group
2. Repeat until convergence

**COMPLEXITY:**
- Time:  O(max_iter * nnz * avg_group_size)
- Space: O(n_samples + n_features + max_group_size)

**THREAD SAFETY:**
Safe

**NUMERICAL NOTES:**
- Promotes sparsity at group level
- Within selected groups, all coefficients are non-zero
- sqrt(|g|) factor accounts for group size

---

## sparse_logistic_regression

**SUMMARY:**
L1-regularized logistic regression via coordinate descent.

**SIGNATURE:**
```cpp
template <typename T, bool IsCSR>
void sparse_logistic_regression(
    const Sparse<T, IsCSR>& X,
    Array<const Index> y_binary,         // Binary labels (0 or 1)
    Real alpha,
    Array<Real> coefficients,
    Index max_iter = 1000
);
```

**PARAMETERS:**
- X            [in]  Feature matrix
- y_binary     [in]  Binary labels [n_samples], values in {0, 1}
- alpha        [in]  L1 regularization strength
- coefficients [out] Solution coefficients
- max_iter     [in]  Maximum iterations

**PRECONDITIONS:**
- X.valid() and y_binary.len == X.rows()
- coefficients.len >= X.cols()
- All y_binary[i] in {0, 1}
- alpha >= 0

**POSTCONDITIONS:**
- Minimizes: -log_likelihood + alpha * ||coef||_1
- where log_likelihood = sum_i [y_i*log(p_i) + (1-y_i)*log(1-p_i)]
- p_i = sigmoid(X[i,:]*coef + intercept)
- Produces sparse coefficient vector

**ALGORITHM:**
Iteratively Reweighted Least Squares (IRLS) with L1 penalty:
1. Initialize coefficients and intercept to zero
2. For each iteration:
   a. Compute probabilities: p[i] = sigmoid(X[i,:]*coef + intercept)
   b. Compute weights: w[i] = p[i] * (1 - p[i])
   c. Compute working response: z[i] = linear_pred[i] + (y[i] - p[i])/w[i]
   d. Update intercept (unpenalized): weighted least squares
   e. Coordinate descent on features with weighted soft thresholding
3. Return when converged

**COMPLEXITY:**
- Time:  O(max_iter * nnz * inner_iter)
- Space: O(n_samples * 4 + n_features)

**THREAD SAFETY:**
Safe

**MUTABILITY:**
CONST for X and y, writes to coefficients

**NUMERICAL NOTES:**
- Uses numerically stable sigmoid computation
- Clips probabilities away from 0/1 for stability
- Intercept is not penalized
- Handles CSR and CSC formats efficiently
