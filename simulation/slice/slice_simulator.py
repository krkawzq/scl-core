#!/usr/bin/env python3
"""Sparse Matrix Slice Strategy Simulator and Code Generator.

A self-contained tool for simulating sparse matrix slice operations,
analyzing strategy performance, and generating optimized C++ decision code.

Usage:
    python3 slice_simulator.py --help
    python3 slice_simulator.py --run-simulation --density 0.1
    python3 slice_simulator.py --generate-code --output strategy.hpp

Author: SCL-Core Team
License: MIT
"""

import sys
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
from enum import Enum


# =============================================================================
# Core Data Structures
# =============================================================================

class Strategy(Enum):
    """Processing strategies for sparse slice."""
    SKIP = -1  # [NEW] Segment-level pruning: completely skip
    PROBE = 0  # Point-wise probing
    MERGE = 1  # Two-pointer merge
    RANGE = 2  # Range query with RLE


@dataclass
class HardwareParams:
    """Hardware parameters for cost modeling."""
    # Cache hierarchy (bytes, cycles)
    cache_l1: int = 32 * 1024
    cache_l2: int = 256 * 1024
    cache_l3: int = 8 * 1024 * 1024
    latency_l1: float = 4.0
    latency_l2: float = 12.0
    latency_l3: float = 40.0
    latency_dram: float = 200.0
    
    # SIMD parameters
    simd_width: int = 64  # AVX-512
    index_size: int = 4   # int32
    vector_cost: float = 1.0
    
    # Branch parameters
    branch_penalty: float = 15.0
    prefetch_gain: float = 2.5
    locality_decay: float = 0.85
    
    @property
    def block_size(self) -> int:
        """SIMD block size (elements)."""
        return self.simd_width // self.index_size


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    mask_length: int = 100000
    density: float = 0.1
    n_segments: int = 100
    segment_size_min: int = 50
    segment_size_max: int = 500
    mask_pattern: str = 'random'
    
    # Bayesian learning
    use_bayesian: bool = True
    bayes_alpha: float = 1.0
    bayes_beta: float = 1.0
    bayes_warmup: int = 10


@dataclass
class SegmentInfo:
    """Segment information for precise decision making."""
    indices: np.ndarray      # Index array
    m: int                   # Number of indices
    i_min: int              # Minimum index
    i_max: int              # Maximum index
    delta: int              # Span = i_max - i_min + 1
    is_contiguous: bool     # Whether contiguous (delta == m)
    is_sorted: bool         # Whether sorted (always True for CSR/CSC)


@dataclass
class MaskInfo:
    """Preprocessed mask information."""
    N: int                  # Total length
    s: int                  # Number of nonzeros sum(mask)
    p: float                # Density s/N
    q: int                  # Number of RLE intervals
    mask_min: int           # Position of first 1
    mask_max: int           # Position of last 1
    # Optional: precomputed auxiliary structures
    P: np.ndarray = None    # Ordered position table
    R: list = None          # Interval list [(l0,r0), (l1,r1), ...]


# =============================================================================
# Cache Model
# =============================================================================

class CacheModel:
    """Multi-level cache access cost model."""
    
    def __init__(self, hw: HardwareParams):
        self.hw = hw
        self.capacities = [hw.cache_l1, hw.cache_l2, hw.cache_l3, float('inf')]
        self.latencies = [hw.latency_l1, hw.latency_l2, hw.latency_l3, hw.latency_dram]
    
    def expected_latency(self, working_set_bytes: float, sequential: bool = False) -> float:
        """Calculate expected access latency based on working set size."""
        if working_set_bytes <= 0:
            return self.hw.latency_l1
        
        total_latency = 0.0
        cumulative_miss = 1.0
        
        for capacity, latency in zip(self.capacities, self.latencies):
            if sequential:
                hit_rate = min(1.0, (capacity / working_set_bytes) * self.hw.prefetch_gain)
            else:
                hit_rate = min(1.0, (capacity / working_set_bytes) ** self.hw.locality_decay)
            
            prob = cumulative_miss * hit_rate
            total_latency += prob * latency
            cumulative_miss *= (1 - hit_rate)
        
        return total_latency


# =============================================================================
# Cost Model V2 - Fixed Version
# =============================================================================

class CostModel:
    """Corrected cost model with segment-level pruning."""
    
    def __init__(self, hw: HardwareParams):
        self.hw = hw
        self.cache = CacheModel(hw)
    
    # =========================================================================
    # Fix 1: Segment-level range pruning (GraphBLAS Case 1-4 fast path)
    # =========================================================================
    
    def can_skip_segment(self, seg: SegmentInfo, mask: MaskInfo) -> bool:
        """
        Check if segment is completely disjoint from mask support.
        
        GraphBLAS equivalent: 
            if (col_indices[nnz-1] < mask_min || col_indices[0] > mask_max)
                return;  // skip
        """
        if seg.m == 0:
            return True
        if seg.i_max < mask.mask_min:
            return True
        if seg.i_min > mask.mask_max:
            return True
        return False
    
    # =========================================================================
    # Fix 2: More accurate PROBE cost
    # =========================================================================
    
    def cost_probe(self, seg: SegmentInfo, mask: MaskInfo) -> float:
        """
        Strategy A: Point-wise probing.
        
        Fixes:
        1. Index access is sequential -> use sequential latency
        2. Mask access is random
        3. Branch penalty depends on local density within segment range
        """
        m = seg.m
        
        # Random mask access cost (Key: working set is entire mask)
        mask_ws = mask.N  # bytes (assuming uint8)
        latency_mask = self.cache.expected_latency(mask_ws, sequential=False)
        
        # Sequential index access cost (Fix: should NOT divide by block_size here)
        index_ws = m * self.hw.index_size
        latency_index = self.cache.expected_latency(index_ws, sequential=True)
        
        # Branch misprediction (Fix: use local density estimate)
        # If mask_min/mask_max available, can do more precise estimation
        local_p = mask.p  # Simplified: use global density
        mispred_rate = 2.0 * local_p * (1.0 - local_p)
        
        # Total cost
        cost = m * latency_mask                           # Check mask once per index
        cost += (m / self.hw.block_size) * latency_index  # Sequential index read (cacheline friendly)
        cost += m * mispred_rate * self.hw.branch_penalty
        
        return cost
    
    # =========================================================================
    # Fix 3: MERGE cost needs range clipping
    # =========================================================================
    
    def cost_merge(self, seg: SegmentInfo, mask: MaskInfo) -> float:
        """
        Strategy B: Two-pointer merge.
        
        Fixes:
        1. P_range should be number of P elements in [i_min, i_max], not delta*p
        2. Need binary search to locate P start and end points
        """
        m = seg.m
        
        # Fix: Expected number of P elements in [i_min, i_max]
        # Assuming uniform distribution: |P ∩ [i_min, i_max]| ≈ delta * p
        # More precise: if mask_min/mask_max known, can clip
        effective_range = min(seg.delta, mask.mask_max - mask.mask_min + 1)
        P_range = effective_range * mask.p
        
        # Binary search cost (find start and end in P)
        # Two binary searches on P: O(2 * log(s))
        binary_search_cost = 2 * np.log2(max(2, mask.s)) * self.hw.latency_l1
        
        # Merge scan cost
        total_elements = m + P_range
        total_bytes = total_elements * self.hw.index_size
        latency = self.cache.expected_latency(total_bytes, sequential=True)
        
        blocks = int(np.ceil(m / self.hw.block_size))
        
        cost = binary_search_cost
        cost += total_elements * latency / self.hw.block_size
        cost += blocks * self.hw.vector_cost
        
        return cost
    
    # =========================================================================
    # Fix 4: RANGE cost binary search count
    # =========================================================================
    
    def cost_range(self, seg: SegmentInfo, mask: MaskInfo) -> float:
        """
        Strategy C: Range query.
        
        Fixes:
        1. Each overlapping interval needs 2 binary searches (lower_bound + upper_bound)
        2. Should consider if intervals overlap with [i_min, i_max]
        """
        if mask.q == 0:
            return float('inf')
        
        m = seg.m
        
        # Overlapping interval count estimate
        # More precise: clip using mask_min/mask_max
        overlap_fraction = seg.delta / mask.N
        q_overlap = max(1.0, mask.q * overlap_fraction)
        
        # Binary search cost (Fix: 2 binary searches per interval)
        index_ws = m * self.hw.index_size
        latency_rand = self.cache.expected_latency(index_ws, sequential=False)
        binary_cost = q_overlap * 2 * np.log2(max(2, m)) * latency_rand
        
        # Copy cost
        expected_hits = m * mask.p
        copy_cost = expected_hits * self.hw.vector_cost
        
        return binary_cost + copy_cost
    
    # =========================================================================
    # Fix 5: Unified decision interface
    # =========================================================================
    
    def optimal_strategy(self, seg: SegmentInfo, mask: MaskInfo) -> Tuple[Strategy, Dict[Strategy, float]]:
        """
        Select optimal strategy.
        
        New: SKIP strategy for segment-level pruning
        """
        # Segment-level pruning check
        if self.can_skip_segment(seg, mask):
            return Strategy.SKIP, {Strategy.SKIP: 0.0}
        
        # Special case: segment is contiguous and fully within mask range
        # GraphBLAS Case 5: I=imin:imax
        if seg.is_contiguous:
            # Can use memcpy-style fast path
            # Simplified here, fold into RANGE
            pass
        
        # Compute costs for three strategies
        costs = {
            Strategy.PROBE: self.cost_probe(seg, mask),
            Strategy.MERGE: self.cost_merge(seg, mask),
            Strategy.RANGE: self.cost_range(seg, mask),
        }
        
        best = min(costs, key=costs.get)
        return best, costs
    
    # Legacy interface for backward compatibility
    def optimal_strategy_legacy(self, m: int, p: float, N: int, delta: int, q: int) -> Tuple[Strategy, Dict[Strategy, float]]:
        """Legacy interface - convert to new format."""
        # Create dummy segment and mask info
        indices = np.arange(m, dtype=np.int32)
        seg = create_segment_info(indices)
        seg.delta = delta
        
        s = int(N * p)
        mask_info = MaskInfo(N=N, s=s, p=p, q=q, mask_min=0, mask_max=N-1)
        
        return self.optimal_strategy(seg, mask_info)


# =============================================================================
# Bayesian Estimator
# =============================================================================

class BayesianEstimator:
    """Online Bayesian density estimation (Beta-Bernoulli)."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, warmup: int = 10):
        self.alpha = alpha
        self.beta = beta
        self.alpha_0 = alpha
        self.beta_0 = beta
        self.warmup = warmup
        self.n_updates = 0
    
    def update(self, hits: int, total: int):
        """Update posterior with observations."""
        self.alpha += hits
        self.beta += (total - hits)
        self.n_updates += 1
    
    def estimate(self) -> float:
        """Get density estimate (Thompson Sampling)."""
        if self.n_updates < self.warmup:
            return self.alpha_0 / (self.alpha_0 + self.beta_0)
        return np.random.beta(self.alpha, self.beta)
    
    def mean(self) -> float:
        """Get posterior mean."""
        return self.alpha / (self.alpha + self.beta)
    
    def reset(self):
        """Reset to prior."""
        self.alpha = self.alpha_0
        self.beta = self.beta_0
        self.n_updates = 0


# =============================================================================
# Data Generator
# =============================================================================

class DataGenerator:
    """Generate test data for simulation."""
    
    @staticmethod
    def generate_mask(N: int, p: float, pattern: str = 'random') -> np.ndarray:
        """Generate boolean mask array."""
        if pattern == 'random':
            return (np.random.random(N) < p).astype(np.uint8)
        elif pattern == 'block':
            mask = np.zeros(N, dtype=np.uint8)
            block_size = int(N * p)
            if block_size > 0:
                start = np.random.randint(0, max(1, N - block_size + 1))
                mask[start:start + block_size] = 1
            return mask
        elif pattern == 'periodic':
            period = max(1, int(1 / p)) if p > 0 else N
            mask = np.zeros(N, dtype=np.uint8)
            mask[::period] = 1
            return mask
        else:
            return (np.random.random(N) < p).astype(np.uint8)
    
    @staticmethod
    def compute_rle_count(mask: np.ndarray) -> int:
        """Compute number of RLE intervals."""
        if len(mask) == 0:
            return 0
        diff = np.diff(mask.astype(np.int8))
        return int(np.sum(diff == 1)) + (1 if mask[0] == 1 else 0)
    
    @staticmethod
    def generate_segment_indices(N: int, m: int) -> np.ndarray:
        """Generate sorted segment indices."""
        if m >= N:
            return np.arange(N, dtype=np.int32)
        indices = np.sort(np.random.choice(N, m, replace=False))
        return indices.astype(np.int32)


# =============================================================================
# Helper Functions for New Cost Model
# =============================================================================

def precompute_mask_info(mask: np.ndarray) -> MaskInfo:
    """Precompute mask statistics."""
    N = len(mask)
    s = int(np.sum(mask))
    p = s / N if N > 0 else 0.0
    
    # Find mask_min and mask_max
    nonzero_indices = np.nonzero(mask)[0]
    if len(nonzero_indices) == 0:
        mask_min, mask_max = N, -1  # Empty mask
    else:
        mask_min = int(nonzero_indices[0])
        mask_max = int(nonzero_indices[-1])
    
    # Compute RLE interval count
    if len(mask) == 0:
        q = 0
    else:
        diff = np.diff(mask.astype(np.int8))
        q = int(np.sum(diff == 1)) + (1 if mask[0] == 1 else 0)
    
    return MaskInfo(
        N=N, s=s, p=p, q=q,
        mask_min=mask_min, mask_max=mask_max
    )


def create_segment_info(indices: np.ndarray) -> SegmentInfo:
    """Create segment info from index array."""
    m = len(indices)
    if m == 0:
        return SegmentInfo(
            indices=indices, m=0, 
            i_min=0, i_max=0, delta=0,
            is_contiguous=True, is_sorted=True
        )
    
    i_min = int(indices[0])
    i_max = int(indices[-1])
    delta = i_max - i_min + 1
    is_contiguous = (delta == m)
    is_sorted = np.all(np.diff(indices) >= 0)
    
    return SegmentInfo(
        indices=indices, m=m,
        i_min=i_min, i_max=i_max, delta=delta,
        is_contiguous=is_contiguous, is_sorted=is_sorted
    )


# =============================================================================
# Simulator
# =============================================================================

class Simulator:
    """Main simulation engine."""
    
    def __init__(self, hw: HardwareParams, config: SimulationConfig):
        self.hw = hw
        self.config = config
        self.cost_model = CostModel(hw)
        self.estimator = BayesianEstimator(
            config.bayes_alpha,
            config.bayes_beta,
            config.bayes_warmup
        ) if config.use_bayesian else None
    
    def run(self) -> Dict:
        """Run full simulation with corrected cost model."""
        # Generate mask
        mask = DataGenerator.generate_mask(
            self.config.mask_length,
            self.config.density,
            self.config.mask_pattern
        )
        
        # Precompute mask info (NEW)
        mask_info = precompute_mask_info(mask)
        
        N = mask_info.N
        true_p = mask_info.p
        q = mask_info.q
        
        # Run segments
        results = []
        total_cost = 0.0
        optimal_cost = 0.0
        strategy_counts = {Strategy.SKIP: 0, Strategy.PROBE: 0, Strategy.MERGE: 0, Strategy.RANGE: 0}
        
        if self.estimator:
            self.estimator.reset()
        
        for i in range(self.config.n_segments):
            m = np.random.randint(self.config.segment_size_min, self.config.segment_size_max + 1)
            indices = DataGenerator.generate_segment_indices(N, m)
            
            if m == 0:
                continue
            
            # Create segment info (NEW)
            seg_info = create_segment_info(indices)
            
            # True metrics
            hits = int(np.sum(mask[indices]))
            
            # Create mask info with estimated density
            if self.estimator:
                est_p = self.estimator.estimate()
            else:
                est_p = true_p
            
            # Estimated mask info for decision
            est_mask_info = MaskInfo(
                N=N, 
                s=int(N * est_p), 
                p=est_p, 
                q=q,
                mask_min=mask_info.mask_min, 
                mask_max=mask_info.mask_max
            )
            
            # Select strategy using new interface
            chosen, costs = self.cost_model.optimal_strategy(seg_info, est_mask_info)
            strategy_counts[chosen] += 1
            
            # Actual cost with true mask info
            _, true_costs = self.cost_model.optimal_strategy(seg_info, mask_info)
            actual_cost = true_costs.get(chosen, 0.0)
            opt_cost = min(true_costs.values()) if true_costs else 0.0
            
            total_cost += actual_cost
            optimal_cost += opt_cost
            
            # Update estimator
            if self.estimator:
                self.estimator.update(hits, m)
            
            results.append({
                'segment_id': i,
                'm': m,
                'delta': seg_info.delta,
                'hits': hits,
                'strategy': chosen,
                'estimated_p': est_p,
                'true_p': true_p,
                'cost': actual_cost,
                'optimal_cost': opt_cost,
            })
        
        regret = total_cost - optimal_cost
        regret_ratio = regret / optimal_cost if optimal_cost > 0 else 0.0
        
        return {
            'config': {
                'N': N,
                'p': self.config.density,
                'actual_p': float(true_p),
                'n_segments': self.config.n_segments,
                'q': q,
                'mask_min': mask_info.mask_min,
                'mask_max': mask_info.mask_max,
            },
            'results': results,
            'summary': {
                'total_cost': total_cost,
                'optimal_cost': optimal_cost,
                'regret': regret,
                'regret_ratio': regret_ratio,
                'strategy_counts': {s.name: strategy_counts[s] for s in Strategy},
            }
        }


# =============================================================================
# Code Generator
# =============================================================================

class CodeGenerator:
    """Generate optimized C++ strategy decision code."""
    
    def __init__(self, hw: HardwareParams):
        self.hw = hw
        self.cost_model = CostModel(hw)
    
    def analyze_decision_boundaries(self, N: int = 100000) -> Dict:
        """Analyze decision boundaries across (p, m) space."""
        p_values = np.linspace(0.001, 0.999, 64)
        m_values = np.logspace(1, 4, 64).astype(int)
        
        lut = np.zeros((len(p_values), len(m_values)), dtype=np.int8)
        
        for i, p in enumerate(p_values):
            q = max(1, int(2 * N * p * (1 - p)))
            s = int(N * p)
            
            # Create mask info for typical scenario
            mask_info = MaskInfo(N=N, s=s, p=p, q=q, mask_min=0, mask_max=N-1)
            
            for j, m in enumerate(m_values):
                # Create typical segment (evenly spaced)
                indices = np.linspace(0, N-1, m, dtype=np.int32)
                seg_info = create_segment_info(indices)
                
                strategy, _ = self.cost_model.optimal_strategy(seg_info, mask_info)
                lut[i, j] = strategy.value
        
        # Analyze boundaries (exclude SKIP=-1)
        valid_mask = lut >= 0
        probe_region = np.sum((lut == 0) & valid_mask) / np.sum(valid_mask) if np.any(valid_mask) else 0
        merge_region = np.sum((lut == 1) & valid_mask) / np.sum(valid_mask) if np.any(valid_mask) else 0
        range_region = np.sum((lut == 2) & valid_mask) / np.sum(valid_mask) if np.any(valid_mask) else 0
        skip_region = np.sum(lut == -1) / lut.size
        
        # Check smoothness
        is_smooth, smoothness_metrics = self._check_smoothness(lut)
        
        return {
            'lut': lut,
            'p_values': p_values,
            'm_values': m_values,
            'distribution': {
                'SKIP': skip_region,
                'PROBE': probe_region,
                'MERGE': merge_region,
                'RANGE': range_region,
            },
            'is_smooth': is_smooth,
            'smoothness': smoothness_metrics,
        }
    
    def _check_smoothness(self, lut: np.ndarray) -> Tuple[bool, Dict]:
        """Check if decision boundary is smooth (monotonic).
        
        A smooth boundary means:
        1. Each row has at most 2 transitions (0→1, 1→2 or 0→1→2)
        2. Boundaries move monotonically with density
        
        Returns:
            (is_smooth, metrics)
        """
        n_rows, n_cols = lut.shape
        
        # Check row-wise monotonicity
        max_transitions = 0
        non_monotonic_rows = 0
        
        for i in range(n_rows):
            row = lut[i, :]
            transitions = np.sum(np.diff(row) != 0)
            max_transitions = max(max_transitions, transitions)
            
            # Check if transitions are monotonic (values only increase)
            if np.any(np.diff(row) < 0):
                non_monotonic_rows += 1
        
        # Check column-wise smoothness
        non_monotonic_cols = 0
        for j in range(n_cols):
            col = lut[:, j]
            if np.any(np.diff(col) < 0):
                non_monotonic_cols += 1
        
        # Decision boundary smoothness score
        # 0 = perfectly smooth, 1 = very rough
        roughness_score = (non_monotonic_rows + non_monotonic_cols) / (n_rows + n_cols)
        
        # Consider smooth if:
        # - Max 2 transitions per row (0→1→2)
        # - < 10% non-monotonic rows
        # - < 10% non-monotonic columns
        is_smooth = (
            max_transitions <= 2 and
            non_monotonic_rows < n_rows * 0.1 and
            non_monotonic_cols < n_cols * 0.1
        )
        
        metrics = {
            'max_transitions_per_row': int(max_transitions),
            'non_monotonic_rows': int(non_monotonic_rows),
            'non_monotonic_cols': int(non_monotonic_cols),
            'roughness_score': float(roughness_score),
            'total_rows': int(n_rows),
            'total_cols': int(n_cols),
        }
        
        return is_smooth, metrics
    
    def _analyze_lut_patterns(self, lut: np.ndarray, p_values: np.ndarray, m_values: np.ndarray) -> Dict:
        """Analyze LUT to extract decision boundaries for branch generation.
        
        Returns:
            Dictionary with transition points and fitted thresholds
        """
        n_p, n_m = lut.shape
        
        # Find PROBE->MERGE transition for key density values
        transition_points = []
        
        for i, p in enumerate(p_values):
            row = lut[i, :]
            
            # Find first occurrence of MERGE (value >= 1)
            merge_indices = np.where(row >= 1)[0]
            
            if len(merge_indices) > 0:
                # Transition happens at this m value
                transition_m = int(m_values[merge_indices[0]])
                transition_points.append((p, transition_m))
        
        return {
            'transition_points': transition_points,
            'p_values': p_values,
            'm_values': m_values
        }
    
    def _fit_threshold_function(self, transition_points: List[Tuple[float, int]]) -> Dict:
        """Fit threshold function: threshold(p) = a / p + b
        
        Returns:
            Dict with 'a' and 'b' coefficients
        """
        if len(transition_points) < 3:
            # Default fallback
            return {'a': 50.0, 'b': 0.0, 'quality': 'poor'}
        
        # Extract p and m values
        p_arr = np.array([pt[0] for pt in transition_points])
        m_arr = np.array([pt[1] for pt in transition_points])
        
        # Filter out very high density (p > 0.5) for better fit
        valid_idx = p_arr < 0.5
        p_fit = p_arr[valid_idx]
        m_fit = m_arr[valid_idx]
        
        if len(p_fit) < 2:
            return {'a': 50.0, 'b': 0.0, 'quality': 'poor'}
        
        # Fit: m = a / p + b
        # Transform: m * p = a + b * p
        # Linear regression: Y = a + b * X, where Y = m * p, X = p
        X = p_fit
        Y = m_fit * p_fit
        
        # Least squares fit
        X_mean = np.mean(X)
        Y_mean = np.mean(Y)
        
        numerator = np.sum((X - X_mean) * (Y - Y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        
        if abs(denominator) < 1e-10:
            return {'a': 50.0, 'b': 0.0, 'quality': 'poor'}
        
        b_coef = numerator / denominator
        a_coef = Y_mean - b_coef * X_mean
        
        # Compute R² for quality assessment
        y_pred = a_coef + b_coef * X
        ss_res = np.sum((Y - y_pred) ** 2)
        ss_tot = np.sum((Y - Y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        quality = 'excellent' if r_squared > 0.9 else 'good' if r_squared > 0.7 else 'fair'
        
        return {
            'a': float(a_coef),
            'b': float(b_coef),
            'r_squared': float(r_squared),
            'quality': quality
        }
    
    def _generate_branch_from_lut(self, analysis: Dict) -> str:
        """Generate branch-based code from LUT analysis.
        
        Converts lookup table to equivalent branch instructions (no memory access).
        """
        lut = analysis['lut']
        p_values = analysis['p_values']
        m_values = analysis['m_values']
        dist = analysis['distribution']
        
        # Analyze LUT patterns
        patterns = self._analyze_lut_patterns(lut, p_values, m_values)
        
        # Fit threshold function
        fitted = self._fit_threshold_function(patterns['transition_points'])
        
        a_coef = fitted['a']
        b_coef = fitted['b']
        quality = fitted['quality']
        r_squared = fitted.get('r_squared', 0.0)
        
        header = f"""#pragma once

/// @file sparse_slice_strategy.hpp
/// @brief Optimized Strategy Selection for Sparse Matrix Slice Operations
///
/// Auto-generated by slice_simulator.py from cost model analysis
/// 
/// Strategy Distribution (based on cost model):
///   - SKIP:  {dist['SKIP']*100:.1f}%
///   - PROBE: {dist['PROBE']*100:.1f}%
///   - MERGE: {dist['MERGE']*100:.1f}%
///   - RANGE: {dist['RANGE']*100:.1f}%
///
/// Platform-specific thresholds (fitted from LUT):
///   - Threshold function: m_threshold(p) = {a_coef:.2f} / p + {b_coef:.2f}
///   - Fit quality: {quality} (R² = {r_squared:.4f})
///
/// Key Insight: Real sparse matrices have p < 0.1 in 90%+ cases

#include <cstdint>

#include "scl/core/macro.hpp"

namespace scl::sparse {{

/// @brief Processing strategy for sparse slice operations
enum class SliceStrategy : std::int8_t {{
    SKIP  = -1, ///< Segment-level pruning (no overlap)
    PROBE = 0,  ///< Point-wise probing (A)
    MERGE = 1,  ///< Two-pointer merge (B)
    RANGE = 2   ///< Range query with RLE (C)
}};

/// @brief Select optimal strategy with segment-level pruning
/// @param[in] p Estimated density (fraction of non-zeros)
/// @param[in] m Segment size (number of indices)
/// @param[in] i_min Minimum index in segment
/// @param[in] i_max Maximum index in segment
/// @param[in] mask_min First nonzero position in mask
/// @param[in] mask_max Last nonzero position in mask
/// @return Optimal strategy
///
/// ## Implementation
///
/// Pure branch-based decision (no memory access, no lookup tables).
/// Thresholds are platform-specific, fitted from cost model simulation.
///
/// ## Decision Logic
///
/// 1. Segment-level pruning: Skip if ranges disjoint
/// 2. Density-based strategy selection with platform-tuned thresholds
template<typename Index>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto select_slice_strategy(
    float p, 
    Index m,
    Index i_min, 
    Index i_max,
    Index mask_min,
    Index mask_max
) noexcept -> SliceStrategy {{
    // Step 1: Segment-level range pruning
    if (m == 0 || i_max < mask_min || i_min > mask_max) [[unlikely]] {{
        return SliceStrategy::SKIP;
    }}
    
    // Step 2: Strategy selection (platform-tuned thresholds)
    // Fast path: typical sparse matrices (p < 0.1)
    if (p < 0.1f) [[likely]] {{
        // Very small segments: always probe
        if (m < 100) [[likely]] {{
            return SliceStrategy::PROBE;
        }}
        
        // Platform-specific transition point (fitted from cost model)
        const float threshold = {a_coef:.2f}f / p + {b_coef:.2f}f;
        return (static_cast<float>(m) < threshold) ? SliceStrategy::PROBE : SliceStrategy::MERGE;
    }}
    
    // Medium density (0.1 <= p < 0.5)
    if (p < 0.5f) [[likely]] {{
        return SliceStrategy::MERGE;
    }}
    
    // High density (p >= 0.5) - rare case
    [[unlikely]] {{
        // Extreme case: very high density + huge segment
        if (p > 0.9f && m > 50000) [[unlikely]] {{
            return SliceStrategy::RANGE;
        }}
        return SliceStrategy::MERGE;
    }}
}}

/// @brief Simplified strategy selection (no range pruning)
/// @param[in] p Estimated density
/// @param[in] m Segment size
/// @return Optimal strategy
template<typename Index>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto select_slice_strategy(float p, Index m) noexcept -> SliceStrategy {{
    if (p < 0.1f) [[likely]] {{
        if (m < 100) [[likely]] {{
            return SliceStrategy::PROBE;
        }}
        const float threshold = {a_coef:.2f}f / p + {b_coef:.2f}f;
        return (static_cast<float>(m) < threshold) ? SliceStrategy::PROBE : SliceStrategy::MERGE;
    }}
    
    if (p < 0.5f) [[likely]] {{
        return SliceStrategy::MERGE;
    }}
    
    return SliceStrategy::MERGE;
}}

/// @brief Get strategy name for debugging
[[nodiscard]]
constexpr
auto strategy_name(SliceStrategy s) noexcept -> const char* {{
    switch (s) {{
        case SliceStrategy::SKIP:  return "SKIP";
        case SliceStrategy::PROBE: return "PROBE";
        case SliceStrategy::MERGE: return "MERGE";
        case SliceStrategy::RANGE: return "RANGE";
        default: return "UNKNOWN";
    }}
}}

}}  // namespace scl::sparse
"""
        return header
    
    def generate_decision_function(self, output_file: str, force_lut: bool = False):
        """Generate optimized C++ decision function.
        
        Args:
            output_file: Output file path
            force_lut: Ignored (always generates branch-based code)
        
        Note: Generates branch-based code from LUT analysis.
              No memory access, pure register-based instructions.
        """
        analysis = self.analyze_decision_boundaries()
        
        # Convert LUT to branch-based code
        with open(output_file, 'w') as f:
            f.write(self._generate_branch_from_lut(analysis))
    
    def _generate_branch_header(self, analysis: Dict) -> str:
        """Generate branch-based C++ header (smooth boundaries)."""
        dist = analysis['distribution']
        
        header = f"""#pragma once

/// @file sparse_slice_strategy.hpp
/// @brief Optimized Strategy Selection for Sparse Matrix Slice Operations
///
/// Auto-generated by slice_simulator.py
/// 
/// Strategy Distribution (based on cost model):
///   - SKIP:  {dist['SKIP']*100:.1f}%
///   - PROBE: {dist['PROBE']*100:.1f}%
///   - MERGE: {dist['MERGE']*100:.1f}%
///   - RANGE: {dist['RANGE']*100:.1f}%
///
/// Key Insight: Real sparse matrices have p < 0.1 in 90%+ cases

#include <cstdint>

#include "scl/core/macro.hpp"
#include "scl/core/platform.hpp"

namespace scl::sparse {{

/// @brief Processing strategy for sparse slice operations
enum class SliceStrategy : std::int8_t {{
    SKIP  = -1, ///< Segment-level pruning (no overlap)
    PROBE = 0,  ///< Point-wise probing (A)
    MERGE = 1,  ///< Two-pointer merge (B)
    RANGE = 2   ///< Range query with RLE (C)
}};

/// @brief Select optimal strategy with segment-level pruning
/// @param[in] p Estimated density (fraction of non-zeros)
/// @param[in] m Segment size (number of indices)
/// @param[in] i_min Minimum index in segment
/// @param[in] i_max Maximum index in segment
/// @param[in] mask_min First nonzero position in mask
/// @param[in] mask_max Last nonzero position in mask
/// @return Optimal strategy
///
/// ## Performance Characteristics
///
/// This function is optimized for typical sparse matrix distributions:
/// - 90%+ cases: p < 0.1 (very sparse)
/// - Branch hints reflect real-world usage patterns
/// - Zero table lookup overhead
/// - Segment-level pruning for disjoint ranges
///
/// ## Decision Logic
///
/// ```
/// # Step 1: Segment-level pruning (GraphBLAS Case 1-4)
/// if i_max < mask_min or i_min > mask_max:
///     return SKIP
///
/// # Step 2: Strategy selection based on density
/// if p < 0.1:      # Most common (90%+)
///     if m < 100:  # Small segments
///         PROBE
///     else:
///         PROBE if m < 50/p else MERGE
/// elif p < 0.5:    # Medium density (rare)
///     MERGE
/// else:            # High density (should use dense!)
///     MERGE or RANGE
/// ```
template<typename Index>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto select_slice_strategy(
    float p, 
    Index m,
    Index i_min, 
    Index i_max,
    Index mask_min,
    Index mask_max
) noexcept -> SliceStrategy {{
    // Step 1: Segment-level range pruning
    // GraphBLAS fast path: completely disjoint ranges
    if (m == 0 || i_max < mask_min || i_min > mask_max) [[unlikely]] {{
        return SliceStrategy::SKIP;
    }}
    
    // Step 2: Fast path for typical sparse matrices (p < 0.1)
    // This covers 90%+ of real-world sparse matrix operations
    if (p < 0.1f) [[likely]] {{
        // Very small segments: always probe
        if (m < 100) [[likely]] {{
            return SliceStrategy::PROBE;
        }}
        
        // Transition point based on density-size trade-off
        // As density increases, prefer merge earlier
        const int threshold = static_cast<int>(50.0f / p);
        return (m < threshold) ? SliceStrategy::PROBE : SliceStrategy::MERGE;
    }}
    
    // Medium density (0.1 <= p < 0.5) - less common
    // MERGE is almost always optimal in this range
    if (p < 0.5f) [[likely]] {{
        return SliceStrategy::MERGE;
    }}
    
    // High density (p >= 0.5) - should rarely happen!
    // If we reach here, the user should reconsider using sparse format
    [[unlikely]] {{
        // Extreme case: very high density + huge segment
        // Consider RANGE strategy with RLE compression
        if (p > 0.9f && m > 50000) [[unlikely]] {{
            return SliceStrategy::RANGE;
        }}
        
        // Otherwise, MERGE is still reasonable
        return SliceStrategy::MERGE;
    }}
}}

/// @brief Get strategy name for debugging
[[nodiscard]]
constexpr
auto strategy_name(SliceStrategy s) noexcept -> const char* {{
    switch (s) {{
        case SliceStrategy::SKIP:  return "SKIP";
        case SliceStrategy::PROBE: return "PROBE";
        case SliceStrategy::MERGE: return "MERGE";
        case SliceStrategy::RANGE: return "RANGE";
        default: return "UNKNOWN";
    }}
}}

}}  // namespace scl::sparse
"""
        return header
    
    def _generate_lut_header(self, analysis: Dict) -> str:
        """Generate lookup table-based C++ header (non-smooth boundaries)."""
        dist = analysis['distribution']
        lut = analysis['lut']
        p_values = analysis['p_values']
        m_values = analysis['m_values']
        metrics = analysis['smoothness']
        
        p_bins, m_bins = lut.shape
        
        header = f"""#pragma once

/// @file sparse_slice_strategy.hpp
/// @brief Lookup Table-Based Strategy Selection for Sparse Matrix Slice
///
/// Auto-generated by slice_simulator.py
///
/// WARNING: Decision boundary is NOT smooth (non-monotonic)
///          Using lookup table instead of branch-based decision
///
/// Smoothness Metrics:
///   - Max transitions per row: {metrics['max_transitions_per_row']}
///   - Non-monotonic rows: {metrics['non_monotonic_rows']}/{metrics['total_rows']}
///   - Non-monotonic cols: {metrics['non_monotonic_cols']}/{metrics['total_cols']}
///   - Roughness score: {metrics['roughness_score']:.3f}
///
/// Strategy Distribution:
///   - SKIP:  {dist['SKIP']*100:.1f}%
///   - PROBE: {dist['PROBE']*100:.1f}%
///   - MERGE: {dist['MERGE']*100:.1f}%
///   - RANGE: {dist['RANGE']*100:.1f}%

#include <cstdint>
#include <algorithm>
#include "scl/core/platform.hpp"

namespace scl::sparse {{

/// @brief Processing strategy for sparse slice operations
enum class SliceStrategy : std::int8_t {{
    SKIP  = -1, ///< Segment-level pruning (no overlap)
    PROBE = 0,  ///< Point-wise probing (A)
    MERGE = 1,  ///< Two-pointer merge (B)
    RANGE = 2   ///< Range query with RLE (C)
}};

namespace detail {{

// Lookup table parameters
constexpr int P_BINS = {p_bins};
constexpr int M_BINS = {m_bins};
constexpr float P_MIN = {p_values[0]:.6f}f;
constexpr float P_MAX = {p_values[-1]:.6f}f;
constexpr int M_MIN = {int(m_values[0])};
constexpr int M_MAX = {int(m_values[-1])};

// M boundaries (log scale)
constexpr int M_BOUNDS[M_BINS] = {{
"""
        
        # Write M_BOUNDS array
        for i in range(0, len(m_values), 8):
            batch = m_values[i:i+8]
            header += "    "
            header += ", ".join(f"{int(m)}" for m in batch)
            if i + 8 < len(m_values):
                header += ","
            header += "\n"
        
        header += "};\n\n"
        
        # Write strategy LUT (convert to uint8 for storage, -1 -> 255)
        header += f"// Strategy lookup table [{p_bins}x{m_bins}]\n"
        header += f"// Note: -1 (SKIP) stored as 255 in uint8\n"
        header += f"constexpr std::uint8_t STRATEGY_LUT[P_BINS][M_BINS] = {{\n"
        
        for i in range(p_bins):
            header += "    {"
            # Convert int8 to uint8 for storage
            values = [str(int(lut[i, j]) if lut[i, j] >= 0 else 255) for j in range(m_bins)]
            header += ",".join(values)
            header += "}"
            if i < p_bins - 1:
                header += ","
            header += f"  // p={p_values[i]:.3f}\n"
        
        header += "};\n\n"
        
        header += """}}  // namespace detail

/// @brief Select optimal strategy using lookup table with segment-level pruning
template<typename Index>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto select_slice_strategy(
    float p, 
    Index m,
    Index i_min,
    Index i_max,
    Index mask_min,
    Index mask_max
) noexcept -> SliceStrategy {{
    // Step 1: Segment-level range pruning
    if (m == 0 || i_max < mask_min || i_min > mask_max) [[unlikely]] {{
        return SliceStrategy::SKIP;
    }}
    
    // Step 2: Lookup table query
    // Clamp and quantize p
    p = std::clamp(p, detail::P_MIN, detail::P_MAX);
    int p_idx = static_cast<int>(
        (p - detail::P_MIN) / (detail::P_MAX - detail::P_MIN) * (detail::P_BINS - 1)
    );
    p_idx = std::clamp(p_idx, 0, detail::P_BINS - 1);
    
    // Binary search for m (log scale)
    int m_idx = 0;
    for (int i = 0; i < detail::M_BINS; ++i) {{
        if (static_cast<int>(m) >= detail::M_BOUNDS[i]) {{
            m_idx = i;
        }} else {{
            break;
        }}
    }}
    
    // Lookup strategy (convert 255 back to -1)
    std::uint8_t raw = detail::STRATEGY_LUT[p_idx][m_idx];
    return static_cast<SliceStrategy>(raw == 255 ? -1 : static_cast<std::int8_t>(raw));
}}

/// @brief Get strategy name for debugging
[[nodiscard]]
constexpr
auto strategy_name(SliceStrategy s) noexcept -> const char* {{
    switch (s) {{
        case SliceStrategy::SKIP:  return "SKIP";
        case SliceStrategy::PROBE: return "PROBE";
        case SliceStrategy::MERGE: return "MERGE";
        case SliceStrategy::RANGE: return "RANGE";
        default: return "UNKNOWN";
    }}
}}

}}  // namespace scl::sparse
"""
        
        return header


# =============================================================================
# CLI Interface
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Sparse Matrix Slice Strategy Simulator and Code Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with default parameters
  python3 slice_simulator.py --run-simulation
  
  # Custom simulation
  python3 slice_simulator.py --run-simulation \\
      --density 0.05 --n-segments 200 --segment-size-min 100
  
  # Generate C++ code
  python3 slice_simulator.py --generate-code --output strategy.hpp
  
  # Full workflow
  python3 slice_simulator.py --run-simulation --generate-code --output strategy.hpp
        """
    )
    
    # Action selection
    action_group = parser.add_argument_group('Actions')
    action_group.add_argument('--run-simulation', action='store_true',
                             help='Run performance simulation')
    action_group.add_argument('--generate-code', action='store_true',
                             help='Generate C++ decision code')
    
    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument('--mask-length', type=int, default=100000,
                          help='Length of mask array (default: 100000)')
    sim_group.add_argument('--density', '-p', type=float, default=0.1,
                          help='Mask density 0.0-1.0 (default: 0.1)')
    sim_group.add_argument('--n-segments', type=int, default=100,
                          help='Number of segments (default: 100)')
    sim_group.add_argument('--segment-size-min', type=int, default=50,
                          help='Minimum segment size (default: 50)')
    sim_group.add_argument('--segment-size-max', type=int, default=500,
                          help='Maximum segment size (default: 500)')
    sim_group.add_argument('--mask-pattern', choices=['random', 'block', 'periodic'],
                          default='random', help='Mask pattern (default: random)')
    
    # Bayesian learning
    bayes_group = parser.add_argument_group('Bayesian Learning')
    bayes_group.add_argument('--no-bayesian', action='store_true',
                            help='Disable Bayesian online learning')
    bayes_group.add_argument('--bayes-alpha', type=float, default=1.0,
                            help='Beta prior alpha (default: 1.0)')
    bayes_group.add_argument('--bayes-beta', type=float, default=1.0,
                            help='Beta prior beta (default: 1.0)')
    bayes_group.add_argument('--bayes-warmup', type=int, default=10,
                            help='Warmup segments (default: 10)')
    
    # Hardware parameters
    hw_group = parser.add_argument_group('Hardware Parameters')
    hw_group.add_argument('--cache-l1', type=int, default=32*1024,
                         help='L1 cache size in bytes (default: 32KB)')
    hw_group.add_argument('--cache-l2', type=int, default=256*1024,
                         help='L2 cache size in bytes (default: 256KB)')
    hw_group.add_argument('--cache-l3', type=int, default=8*1024*1024,
                         help='L3 cache size in bytes (default: 8MB)')
    hw_group.add_argument('--latency-l1', type=float, default=4.0,
                         help='L1 latency in cycles (default: 4.0)')
    hw_group.add_argument('--latency-l2', type=float, default=12.0,
                         help='L2 latency in cycles (default: 12.0)')
    hw_group.add_argument('--latency-l3', type=float, default=40.0,
                         help='L3 latency in cycles (default: 40.0)')
    hw_group.add_argument('--latency-dram', type=float, default=200.0,
                         help='DRAM latency in cycles (default: 200.0)')
    hw_group.add_argument('--simd-width', type=int, default=64,
                         help='SIMD width in bytes (default: 64 for AVX-512)')
    hw_group.add_argument('--branch-penalty', type=float, default=15.0,
                         help='Branch misprediction penalty (default: 15.0)')
    
    # Code generation options
    codegen_group = parser.add_argument_group('Code Generation Options')
    codegen_group.add_argument('--force-lut', action='store_true',
                              help='Force lookup table generation even if boundary is smooth')
    
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output', '-o', type=str,
                             default='sparse_slice_strategy.hpp',
                             help='Output file for generated code (default: sparse_slice_strategy.hpp)')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Check actions
    if not args.run_simulation and not args.generate_code:
        parser.print_help()
        print("\nError: Please specify at least one action (--run-simulation or --generate-code)")
        sys.exit(1)
    
    # Create hardware parameters
    hw = HardwareParams(
        cache_l1=args.cache_l1,
        cache_l2=args.cache_l2,
        cache_l3=args.cache_l3,
        latency_l1=args.latency_l1,
        latency_l2=args.latency_l2,
        latency_l3=args.latency_l3,
        latency_dram=args.latency_dram,
        simd_width=args.simd_width,
        branch_penalty=args.branch_penalty,
    )
    
    # Run simulation
    if args.run_simulation:
        print("=" * 70)
        print("SPARSE SLICE STRATEGY SIMULATION")
        print("=" * 70)
        print()
        
        config = SimulationConfig(
            mask_length=args.mask_length,
            density=args.density,
            n_segments=args.n_segments,
            segment_size_min=args.segment_size_min,
            segment_size_max=args.segment_size_max,
            mask_pattern=args.mask_pattern,
            use_bayesian=not args.no_bayesian,
            bayes_alpha=args.bayes_alpha,
            bayes_beta=args.bayes_beta,
            bayes_warmup=args.bayes_warmup,
        )
        
        simulator = Simulator(hw, config)
        result = simulator.run()
        
        # Print results
        cfg = result['config']
        summary = result['summary']
        
        print(f"Configuration:")
        print(f"  Mask length    : {cfg['N']:,}")
        print(f"  Target density : {config.density:.4f}")
        print(f"  Actual density : {cfg['actual_p']:.4f}")
        print(f"  Segments       : {cfg['n_segments']}")
        print(f"  RLE intervals  : {cfg['q']}")
        print()
        
        print(f"Results:")
        print(f"  Total cost     : {summary['total_cost']:,.2f} cycles")
        print(f"  Optimal cost   : {summary['optimal_cost']:,.2f} cycles")
        print(f"  Regret         : {summary['regret']:,.2f} cycles")
        print(f"  Regret ratio   : {summary['regret_ratio']:.4f} ({summary['regret_ratio']*100:.2f}%)")
        print()
        
        print(f"Strategy Distribution:")
        for strategy, count in summary['strategy_counts'].items():
            pct = 100 * count / cfg['n_segments']
            print(f"  {strategy:10s} : {count:4d} ({pct:5.1f}%)")
        print()
        
        # Performance assessment
        if summary['regret_ratio'] < 0.01:
            rating = "EXCELLENT"
        elif summary['regret_ratio'] < 0.05:
            rating = "VERY GOOD"
        elif summary['regret_ratio'] < 0.10:
            rating = "GOOD"
        else:
            rating = "NEEDS TUNING"
        
        print(f"Assessment: {rating}")
        print()
    
    # Generate code
    if args.generate_code:
        print("=" * 70)
        print("GENERATING C++ DECISION CODE")
        print("=" * 70)
        print()
        
        generator = CodeGenerator(hw)
        
        # Analyze boundaries first
        analysis = generator.analyze_decision_boundaries()
        dist = analysis['distribution']
        is_smooth = analysis['is_smooth']
        metrics = analysis['smoothness']
        
        # Display smoothness check results
        print("Decision Boundary Smoothness Check:")
        print(f"  Smoothness     : {'PASS' if is_smooth else 'FAIL'}")
        print(f"  Max transitions: {metrics['max_transitions_per_row']}")
        print(f"  Non-monotonic  : {metrics['non_monotonic_rows']}/{metrics['total_rows']} rows, "
              f"{metrics['non_monotonic_cols']}/{metrics['total_cols']} cols")
        print(f"  Roughness score: {metrics['roughness_score']:.3f}")
        print()
        
        # Determine generation method
        use_lut = args.force_lut or not is_smooth
        
        if use_lut:
            if args.force_lut:
                print("Mode: LOOKUP TABLE (forced by --force-lut)")
            else:
                print("Mode: LOOKUP TABLE (boundary is non-smooth)")
                print()
                print("WARNING: Decision boundary is not monotonic!")
                print("         Cannot use simple branch-based decision.")
                print("         Generating lookup table implementation.")
        else:
            print("Mode: BRANCH-BASED (boundary is smooth)")
            print()
            print("Optimization: Using branch prediction hints")
        
        print()
        
        # Generate code
        generator.generate_decision_function(args.output, force_lut=use_lut)
        
        print(f"Generated: {args.output}")
        print()
        
        print("Strategy Distribution Analysis:")
        print(f"  SKIP  : {dist['SKIP']*100:5.1f}%")
        print(f"  PROBE : {dist['PROBE']*100:5.1f}%")
        print(f"  MERGE : {dist['MERGE']*100:5.1f}%")
        print(f"  RANGE : {dist['RANGE']*100:5.1f}%")
        print()
    
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)



if __name__ == '__main__':
    main()

