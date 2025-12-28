# Pseudotime

Pseudotime inference kernels for trajectory analysis and developmental ordering.

## Overview

Pseudotime provides:

- **Shortest Path Pseudotime** - Graph distance from root cell
- **Diffusion Pseudotime** - Diffusion map distance (DPT)
- **Root Selection** - Marker-based or peripheral root selection
- **Branch Detection** - Identify branch points in trajectory
- **Trajectory Segmentation** - Assign cells to trajectory segments
- **Pseudotime Smoothing** - Neighborhood-based smoothing
- **Gene Correlation** - Correlation with pseudotime
- **Velocity Integration** - RNA velocity-weighted pseudotime

## Shortest Path Pseudotime

### dijkstra_shortest_path

Compute shortest path distances from a single source node using Dijkstra's algorithm:

```cpp
#include "scl/kernel/pseudotime.hpp"

Sparse<Real, true> adjacency = /* ... */;  // Graph adjacency matrix
Array<Real> distances(n_nodes);

scl::kernel::pseudotime::dijkstra_shortest_path(
    adjacency,
    source,      // Source node index
    distances
);
```

**Parameters:**
- `adjacency`: Graph adjacency matrix (edge weights as distances)
- `source`: Source node index
- `distances`: Shortest distances from source to all nodes

**Postconditions:**
- `distances[i]` = shortest path distance from source to i
- `distances[source]` = 0
- Unreachable nodes have distance = INF_DISTANCE

**Algorithm:**
4-ary heap Dijkstra:
1. Initialize all distances to INF, source to 0
2. Pop minimum from heap, relax neighbors
3. Continue until heap empty

**Complexity:**
- Time: O((V + E) * log_4(V))
- Space: O(V) for heap and distance arrays

**Use cases:**
- Shortest path computation
- Graph distance analysis
- Root-based pseudotime

### graph_pseudotime

Compute pseudotime as normalized shortest path distance from root cell:

```cpp
Array<Real> pseudotime(n_nodes);

scl::kernel::pseudotime::graph_pseudotime(
    adjacency,
    root_cell,   // Starting cell for trajectory
    pseudotime
);
```

**Parameters:**
- `adjacency`: Cell neighborhood graph
- `root_cell`: Starting cell for trajectory
- `pseudotime`: Normalized pseudotime values [0, 1]

**Postconditions:**
- `pseudotime[root_cell]` = 0
- `pseudotime[i]` in [0, 1] for all reachable cells
- Unreachable cells have pseudotime = 1

**Algorithm:**
1. Run Dijkstra from root_cell
2. Normalize distances to [0, 1] by dividing by max
3. Set unreachable cells to 1

**Complexity:**
- Time: O((V + E) * log_4(V))
- Space: O(V) auxiliary

**Use cases:**
- Simple pseudotime inference
- Root-based ordering
- Developmental trajectories

## Diffusion Pseudotime

### diffusion_pseudotime

Compute diffusion pseudotime (DPT) using diffusion map distance:

```cpp
Sparse<Real, true> transition_matrix = /* ... */;  // Markov transition matrix
Array<Real> pseudotime(n_nodes);

scl::kernel::pseudotime::diffusion_pseudotime(
    transition_matrix,
    root_cell,
    pseudotime,
    config::DEFAULT_N_DCS,        // n_dcs = 10
    config::DEFAULT_N_ITERATIONS  // n_iterations = 100
);
```

**Parameters:**
- `transition_matrix`: Markov transition matrix (row-stochastic)
- `root_cell`: Starting cell for trajectory
- `pseudotime`: Output DPT values [0, 1]
- `n_dcs`: Number of diffusion components
- `n_iterations`: Power iteration iterations

**Postconditions:**
- Pseudotime based on diffusion distance from root
- Values normalized to [0, 1]
- Captures connectivity structure beyond graph distance

**Algorithm:**
1. Initialize random diffusion components [n x n_dcs]
2. Power iteration: DC = T * DC (apply transition)
3. Orthonormalize with modified Gram-Schmidt
4. Compute Euclidean distance from root in DC space
5. Normalize to [0, 1]

**Complexity:**
- Time: O(n_iterations * nnz * n_dcs)
- Space: O(n * n_dcs) for diffusion components

**Use cases:**
- Robust pseudotime inference
- Noise-resistant ordering
- Complex trajectory structures

## Root Selection

### select_root_cell

Select root cell as the one with minimum marker gene expression:

```cpp
Array<const Real> marker_expression = /* ... */;  // Stem/early marker

Index root = scl::kernel::pseudotime::select_root_cell(
    adjacency,
    marker_expression
);
```

**Returns:** Index of cell with minimum marker expression

**Complexity:**
- Time: O(n)
- Space: O(1) auxiliary

**Use cases:**
- Marker-based root selection
- Stem cell identification
- Early developmental stage

### select_root_peripheral

Select root cell as the most peripheral node:

```cpp
Index root = scl::kernel::pseudotime::select_root_peripheral(
    adjacency
);
```

**Returns:** Index of most peripheral cell

**Algorithm:**
1. For each cell: compute average edge weight to neighbors
2. Return cell with maximum average (most isolated)

**Complexity:**
- Time: O(nnz)
- Space: O(n) auxiliary

**Use cases:**
- Peripheral root selection
- Trajectory endpoints
- When markers unavailable

## Branch Detection

### detect_branch_points

Identify branch points in trajectory based on pseudotime topology:

```cpp
Array<const Real> pseudotime = /* ... */;
Array<Index> branch_points(n_nodes);

Index n_branches = scl::kernel::pseudotime::detect_branch_points(
    adjacency,
    pseudotime,
    branch_points,
    config::DEFAULT_THRESHOLD  // threshold = 0.1
);
```

**Parameters:**
- `adjacency`: Cell neighborhood graph
- `pseudotime`: Pre-computed pseudotime values
- `branch_points`: Indices of detected branch point cells
- `threshold`: Pseudotime difference threshold for neighbor classification

**Returns:** Number of branch points detected

**Postconditions:**
- `branch_points[0..return_value)` contains branch cell indices
- Branch defined as: (>=1 earlier and >=2 later) or (>=2 earlier and >=1 later)

**Algorithm:**
For each cell in parallel:
1. Count neighbors with pseudotime < (pt_i - threshold) (earlier)
2. Count neighbors with pseudotime > (pt_i + threshold) (later)
3. Mark as branch if asymmetric neighbor distribution

**Complexity:**
- Time: O(nnz)
- Space: O(n * n_threads) for thread-local buffers

**Use cases:**
- Trajectory branching analysis
- Cell fate decision points
- Developmental bifurcations

### segment_trajectory

Assign cells to trajectory segments based on branch points:

```cpp
Array<Index> segment_labels(n_nodes);

scl::kernel::pseudotime::segment_trajectory(
    adjacency,
    pseudotime,
    branch_points,
    n_branch_points,
    segment_labels
);
```

**Parameters:**
- `adjacency`: Cell neighborhood graph
- `pseudotime`: Pseudotime values
- `branch_points`: Detected branch point indices
- `n_branch_points`: Number of branch points
- `segment_labels`: Segment assignment for each cell

**Postconditions:**
- `segment_labels[i]` in [0, n_branch_points]
- Cells before first branch: segment 0
- Cells between branch k and k+1: segment k+1

**Algorithm:**
1. Sort branch points by pseudotime
2. For each cell: find segment by comparing to branch pseudotimes

**Complexity:**
- Time: O(n * n_branch_points)
- Space: O(n_branch_points) auxiliary

**Use cases:**
- Trajectory segmentation
- Branch-specific analysis
- Developmental stage assignment

## Pseudotime Smoothing

### smooth_pseudotime

Smooth pseudotime values using neighborhood averaging:

```cpp
Array<Real> pseudotime = /* ... */;

scl::kernel::pseudotime::smooth_pseudotime(
    adjacency,
    pseudotime,
    10,         // n_iterations
    Real(0.5)   // alpha (smoothing strength)
);
```

**Parameters:**
- `adjacency`: Cell neighborhood graph
- `pseudotime`: Pseudotime values to smooth, modified in-place
- `n_iterations`: Number of smoothing iterations
- `alpha`: Smoothing strength [0, 1]

**Postconditions:**
- Pseudotime smoothed: pt = (1-alpha)*pt + alpha*avg(neighbors)
- Repeated n_iterations times

**Complexity:**
- Time: O(n_iterations * nnz)
- Space: O(n) auxiliary

**Use cases:**
- Noise reduction
- Smooth trajectory ordering
- Local consistency

## Gene Correlation

### pseudotime_correlation

Compute Pearson correlation between pseudotime and each gene:

```cpp
Sparse<Real, true> X = /* ... */;  // Expression matrix (cells x genes)
Array<const Real> pseudotime = /* ... */;
Array<Real> correlations(n_genes);

scl::kernel::pseudotime::pseudotime_correlation(
    X,
    pseudotime,
    n_cells,
    n_genes,
    correlations
);
```

**Parameters:**
- `X`: Gene expression matrix (cells x genes, CSR)
- `pseudotime`: Pseudotime values
- `n_cells`: Number of cells
- `n_genes`: Number of genes
- `correlations`: Per-gene correlations with pseudotime

**Postconditions:**
- `correlations[g]` = Pearson(pseudotime, gene_g_expression)
- Accounts for sparse zeros in variance computation

**Algorithm:**
Two-pass algorithm:
1. First pass: compute gene sums and covariances with pseudotime
2. Second pass: compute gene variances
3. Parallel correlation computation

**Complexity:**
- Time: O(nnz + n_genes)
- Space: O(n_genes) auxiliary

**Use cases:**
- Trajectory-associated genes
- Developmental markers
- Time-dependent expression

## Velocity Integration

### velocity_weighted_pseudotime

Refine pseudotime using RNA velocity direction information:

```cpp
Array<const Real> initial_pseudotime = /* ... */;
Array<const Real> velocity_field = /* ... */;  // Per-cell velocity
Array<Real> refined_pseudotime(n_nodes);

scl::kernel::pseudotime::velocity_weighted_pseudotime(
    adjacency,
    initial_pseudotime,
    velocity_field,
    refined_pseudotime,
    20  // n_iterations
);
```

**Parameters:**
- `adjacency`: Cell neighborhood graph
- `initial_pseudotime`: Initial pseudotime estimate
- `velocity_field`: Per-cell velocity magnitude/direction
- `refined_pseudotime`: Velocity-refined pseudotime
- `n_iterations`: Refinement iterations

**Postconditions:**
- Refined pseudotime incorporates velocity information
- Normalized to [0, 1]

**Algorithm:**
For each iteration:
1. For each cell: weighted average with velocity-adjusted weights
2. Neighbors earlier in pseudotime: weight by 1/(1+velocity)
3. Neighbors later: weight by (1+velocity)
4. Renormalize to [0, 1]

**Complexity:**
- Time: O(n_iterations * nnz)
- Space: O(n) auxiliary

**Use cases:**
- RNA velocity integration
- Direction-aware pseudotime
- Improved trajectory ordering

## Utility Functions

### find_terminal_states

Identify terminal (end) states as cells above pseudotime percentile:

```cpp
Array<Index> terminal_cells(n_nodes);

Index n_terminal = scl::kernel::pseudotime::find_terminal_states(
    adjacency,
    pseudotime,
    terminal_cells,
    Real(0.95)  // percentile
);
```

**Returns:** Number of terminal cells identified

**Postconditions:**
- `terminal_cells[0..return_value)` contains cells with pt >= threshold
- threshold = percentile of pseudotime distribution

**Complexity:**
- Time: O(n log n) for percentile, O(n) for selection
- Space: O(n) auxiliary

**Use cases:**
- Terminal state identification
- Endpoint analysis
- Mature cell types

### compute_backbone

Select representative backbone cells evenly spaced along pseudotime:

```cpp
Array<Index> backbone_indices(n_backbone_cells);

Index n_backbone = scl::kernel::pseudotime::compute_backbone(
    adjacency,
    pseudotime,
    n_backbone_cells,
    backbone_indices
);
```

**Returns:** Actual number of backbone cells selected

**Postconditions:**
- `backbone_indices` contains cells uniformly sampled in pseudotime
- Covers full range from earliest to latest pseudotime

**Algorithm:**
1. Sort cells by pseudotime
2. Select cells at uniform pseudotime intervals

**Complexity:**
- Time: O(n log n) for sorting
- Space: O(n) auxiliary

**Use cases:**
- Representative cell selection
- Trajectory visualization
- Reduced dataset for analysis

### compute_pseudotime

Generic pseudotime computation with method selection:

```cpp
Array<Real> pseudotime(n_nodes);

scl::kernel::pseudotime::compute_pseudotime(
    adjacency,
    root_cell,
    pseudotime,
    PseudotimeMethod::DiffusionPseudotime,  // method
    config::DEFAULT_N_DCS  // n_dcs (for DPT)
);
```

**Parameters:**
- `adjacency`: Cell neighborhood graph or transition matrix
- `root_cell`: Root cell index
- `pseudotime`: Computed pseudotime values
- `method`: Algorithm to use
- `n_dcs`: Diffusion components (for DPT method)

**Postconditions:**
- Pseudotime computed using specified method
- Values normalized to [0, 1]

**Use cases:**
- Unified interface
- Method comparison
- Quick pseudotime inference

## Configuration

Default parameters in `scl::kernel::pseudotime::config`:

```cpp
namespace config {
    constexpr Index DEFAULT_N_DCS = 10;
    constexpr Index DEFAULT_N_ITERATIONS = 100;
    constexpr Real DEFAULT_THRESHOLD = 0.1;
    constexpr Real DEFAULT_DAMPING = 0.85;
    constexpr Real CONVERGENCE_TOL = 1e-6;
    constexpr Real INF_DISTANCE = 1e30;
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr Index HEAP_ARITY = 4;
}
```

## Performance Considerations

### Parallelization

- `dijkstra_multi_source`: Parallel over sources
- `diffusion_pseudotime`: Parallel SpMM and distance computation
- `detect_branch_points`: Parallel over cells
- `pseudotime_correlation`: Parallel correlation computation

### Memory Efficiency

- 4-ary heap for efficient Dijkstra
- WorkspacePool for thread-local buffers
- Minimal temporary allocations

## Best Practices

### 1. Choose Appropriate Method

```cpp
// For simple trajectories
scl::kernel::pseudotime::graph_pseudotime(adjacency, root, pseudotime);

// For complex/noisy trajectories
scl::kernel::pseudotime::diffusion_pseudotime(
    transition_matrix, root, pseudotime
);
```

### 2. Select Root Appropriately

```cpp
// Marker-based
Index root = scl::kernel::pseudotime::select_root_cell(
    adjacency, marker_expression
);

// Peripheral
Index root = scl::kernel::pseudotime::select_root_peripheral(adjacency);
```

### 3. Smooth for Better Results

```cpp
// After initial computation
scl::kernel::pseudotime::smooth_pseudotime(
    adjacency, pseudotime, 10, 0.5
);
```

## Examples

### Complete Pseudotime Analysis

```cpp
// 1. Select root
Index root = scl::kernel::pseudotime::select_root_cell(
    adjacency, marker_expression
);

// 2. Compute pseudotime
Array<Real> pseudotime(n_nodes);
scl::kernel::pseudotime::graph_pseudotime(adjacency, root, pseudotime);

// 3. Smooth
scl::kernel::pseudotime::smooth_pseudotime(adjacency, pseudotime, 10);

// 4. Detect branches
Array<Index> branches(n_nodes);
Index n_branches = scl::kernel::pseudotime::detect_branch_points(
    adjacency, pseudotime, branches
);

// 5. Segment trajectory
Array<Index> segments(n_nodes);
scl::kernel::pseudotime::segment_trajectory(
    adjacency, pseudotime, branches, n_branches, segments
);

// 6. Find trajectory genes
Array<Real> correlations(n_genes);
scl::kernel::pseudotime::pseudotime_correlation(
    expression, pseudotime, n_cells, n_genes, correlations
);
```

---

::: tip Method Selection
Use graph_pseudotime for simple trajectories and diffusion_pseudotime for complex or noisy data.
:::

::: warning Root Selection
Root selection significantly affects pseudotime ordering. Use marker-based selection when possible.
:::

