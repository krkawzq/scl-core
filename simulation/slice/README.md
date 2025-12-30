# Sparse Matrix Slice Strategy Simulator

A self-contained tool for simulating and optimizing sparse matrix slice operations.

## Quick Start

```bash
# Run simulation with default parameters
python3 slice_simulator.py --run-simulation

# Generate optimized C++ decision code
python3 slice_simulator.py --generate-code --output ../../../scl/kernel/sparse_slice_strategy.hpp

# Full workflow: simulate + generate code
python3 slice_simulator.py --run-simulation --generate-code
```

## Overview

This tool simulates three strategies for sparse matrix slice operations and generates optimized C++ code for runtime strategy selection.

### Three Strategies

| Strategy | Name | Use Case | Complexity |
|----------|------|----------|------------|
| PROBE | Point-wise probing | Low density, small segments | O(m) |
| MERGE | Two-pointer merge | Medium density, sequential | O(m + Δp) |
| RANGE | Range query (RLE) | High density, block patterns | O(m log q) |

### Key Features

- **Zero table lookup**: Generates branch-based decision code
- **Branch prediction hints**: Optimized for typical sparse matrices (p < 0.1)
- **Bayesian online learning**: Adapts to runtime density patterns
- **Comprehensive cost model**: Multi-level cache hierarchy modeling

## Command-Line Parameters

### Actions

```bash
--run-simulation          Run performance simulation
--generate-code           Generate C++ decision code
```

### Simulation Parameters

```bash
--mask-length N          Mask array length (default: 100000)
--density, -p P          Mask density 0.0-1.0 (default: 0.1)
--n-segments N           Number of segments (default: 100)
--segment-size-min M     Min segment size (default: 50)
--segment-size-max M     Max segment size (default: 500)
--mask-pattern PATTERN   random|block|periodic (default: random)
```

### Bayesian Learning

```bash
--no-bayesian            Disable Bayesian learning
--bayes-alpha ALPHA      Beta prior alpha (default: 1.0)
--bayes-beta BETA        Beta prior beta (default: 1.0)
--bayes-warmup N         Warmup segments (default: 10)
```

### Hardware Parameters

```bash
--cache-l1 BYTES         L1 cache size (default: 32KB)
--cache-l2 BYTES         L2 cache size (default: 256KB)
--cache-l3 BYTES         L3 cache size (default: 8MB)
--latency-l1 CYCLES      L1 latency (default: 4.0)
--latency-l2 CYCLES      L2 latency (default: 12.0)
--latency-l3 CYCLES      L3 latency (default: 40.0)
--latency-dram CYCLES    DRAM latency (default: 200.0)
--simd-width BYTES       SIMD width (default: 64 for AVX-512)
--branch-penalty CYCLES  Misprediction penalty (default: 15.0)
```

### Output

```bash
--output, -o FILE        Output file (default: sparse_slice_strategy.hpp)
--verbose, -v            Verbose output
```

## Examples

### Example 1: Simulate Very Sparse Matrix (Typical Case)

```bash
python3 slice_simulator.py --run-simulation \
    --density 0.01 \
    --n-segments 200 \
    --mask-pattern random
```

Expected output:
```
Configuration:
  Mask length    : 100,000
  Actual density : 0.0098
  Segments       : 200

Results:
  Regret ratio   : 0.0234 (2.34%)

Strategy Distribution:
  PROBE      :  178 ( 89.0%)
  MERGE      :   22 ( 11.0%)

Assessment: EXCELLENT
```

### Example 2: Medium Density Simulation

```bash
python3 slice_simulator.py --run-simulation \
    --density 0.3 \
    --segment-size-min 200 \
    --segment-size-max 1000
```

### Example 3: Custom Hardware (ARM Graviton3)

```bash
python3 slice_simulator.py --run-simulation \
    --cache-l1 65536 \
    --cache-l2 1048576 \
    --cache-l3 33554432 \
    --latency-l3 35 \
    --simd-width 16
```

### Example 4: Generate Code for Production

```bash
python3 slice_simulator.py --generate-code \
    --output ../../../scl/kernel/sparse_slice_strategy.hpp
```

This generates an optimized header like:

```cpp
namespace scl::sparse {

enum class SliceStrategy : std::uint8_t {
    PROBE = 0,
    MERGE = 1,
    RANGE = 2
};

[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto select_slice_strategy(float p, int m) noexcept -> SliceStrategy {
    if (p < 0.1f) [[likely]] {
        if (m < 100) [[likely]] {
            return SliceStrategy::PROBE;
        }
        const int threshold = static_cast<int>(50.0f / p);
        return (m < threshold) ? SliceStrategy::PROBE : SliceStrategy::MERGE;
    }
    // ... more logic
}

}  // namespace scl::sparse
```

### Example 5: Full Workflow

```bash
# Simulate, analyze, and generate production code
python3 slice_simulator.py \
    --run-simulation \
    --generate-code \
    --density 0.05 \
    --n-segments 500 \
    --output strategy.hpp \
    --verbose
```

## Performance Insights

### Typical Sparse Matrix Distributions

Real-world sparse matrices follow these patterns:

| Domain | Typical Density | Dominant Strategy |
|--------|----------------|-------------------|
| Graph adjacency | p ~ 0.001-0.01 | PROBE (95%+) |
| Text features | p ~ 0.01-0.05 | PROBE (80%), MERGE (20%) |
| Scientific computing | p ~ 0.001-0.1 | PROBE (70%), MERGE (30%) |
| ML features | p ~ 0.01-0.2 | PROBE (50%), MERGE (50%) |

**Key insight**: 90%+ of real sparse operations have p < 0.1, so the generated code heavily optimizes this case with `[[likely]]` branch hints.

### Cost Model Summary

The simulator models three main factors:

1. **Cache hierarchy**: L1/L2/L3/DRAM with realistic latencies
2. **Branch prediction**: 2p(1-p) misprediction rate for PROBE
3. **SIMD vectorization**: Block-level processing (AVX-512: 16×int32)

### Decision Boundaries (Approximate)

```
Density (p)
  1.0  ┌─────────────────────┐
       │                     │
  0.5  │      MERGE          │
       │                     │
  0.1  ├─────────────────────┤
       │  PROBE  │   MERGE   │
  0.01 │         │           │
       └─────────────────────┘
         100    1000   10000
              Segment Size (m)
```

## Integration with SCL

The generated strategy decision code integrates with `scl::sparse`:

```cpp
#include "scl/kernel/sparse_slice_strategy.hpp"

// In your sparse slice implementation
template<typename T, typename IndexT>
auto col_slice(const SharedSpan<IndexT>& col_indices,
               const SharedSpan<bool>& mask,
               float estimated_density) {
    const int m = static_cast<int>(col_indices.size());
    
    auto strategy = scl::sparse::select_slice_strategy(estimated_density, m);
    
    switch (strategy) {
        case scl::sparse::SliceStrategy::PROBE:
            return probe_impl(col_indices, mask);
        case scl::sparse::SliceStrategy::MERGE:
            return merge_impl(col_indices, mask);
        case scl::sparse::SliceStrategy::RANGE:
            return range_impl(col_indices, mask);
    }
}
```

## Algorithm Details

See `RESEARCH_SUMMARY.md` for:
- Comparison with GraphBLAS `GrB_extract`
- Detailed cost model derivation
- Bayesian online learning theory
- Performance validation

## Dependencies

- Python 3.8+
- NumPy

No configuration files needed—all parameters via command line.

## Files

- `slice_simulator.py`: Complete self-contained tool (1200 lines)
- `README.md`: This documentation
- `RESEARCH_SUMMARY.md`: Detailed analysis and comparison with GraphBLAS

## License

MIT License - Part of SCL-Core Project

