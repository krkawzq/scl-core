"""
SCL Config - Strategy Configuration System

Provides property-based strategy configuration for sparse matrix operations.
Allows fine-grained control over computation behavior without modifying
function signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from enum import IntEnum, auto
import threading


# =============================================================================
# Strategy Enumerations
# =============================================================================

class MaterializeStrategy(IntEnum):
    """
    Strategy for materializing lazy operations.
    """
    IMMEDIATE = 0       # Materialize immediately when needed
    DEFERRED = 1        # Defer until explicitly requested
    STREAMING = 2       # Stream from source, never fully load


class ParallelStrategy(IntEnum):
    """
    Strategy for parallel execution.
    """
    AUTO = 0           # Automatic selection based on data size
    SEQUENTIAL = 1     # Force sequential execution
    PARALLEL = 2       # Force parallel execution


class MemoryStrategy(IntEnum):
    """
    Strategy for memory management.
    """
    COPY = 0           # Always copy data
    VIEW = 1           # Create views when possible
    ZERO_COPY = 2      # Zero-copy mode (for mapped files)


class NormType(IntEnum):
    """
    Normalization type.
    """
    L1 = 0             # L1 norm (sum of absolute values)
    L2 = 1             # L2 norm (Euclidean)
    MAX = 2            # Max norm (infinity norm)


class BackendType(IntEnum):
    """
    Backend type for sparse matrix storage.
    """
    IN_MEMORY = 0      # All data in RAM
    MAPPED = 1         # Memory-mapped file
    STREAMING = 2      # Streaming from source
    VIRTUAL = 3        # Virtual (composed from chunks)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class MaterializeConfig:
    """Configuration for materialization behavior."""
    strategy: MaterializeStrategy = MaterializeStrategy.DEFERRED
    chunk_size: int = 1024 * 1024  # 1MB chunks for streaming
    prefetch: bool = True
    max_memory_mb: int = 4096      # Max memory to use (for auto decisions)


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    strategy: ParallelStrategy = ParallelStrategy.AUTO
    num_threads: int = 0           # 0 = auto-detect
    min_elements_per_thread: int = 10000
    chunk_size: int = 64           # Rows per chunk


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    strategy: MemoryStrategy = MemoryStrategy.VIEW
    alignment: int = 64            # Memory alignment
    use_pool: bool = True          # Use memory pool
    max_pages: int = 64            # Max resident pages for mapped mode


@dataclass
class SliceConfig:
    """Configuration for slicing behavior."""
    lazy: bool = True              # Create lazy views
    copy_threshold: int = 1000     # Copy if selection smaller than this
    preserve_order: bool = True    # Preserve element order in slices


@dataclass
class ComputeConfig:
    """Configuration for compute operations."""
    norm_type: NormType = NormType.L2
    epsilon: float = 1e-10         # Small value for numerical stability
    ddof: int = 0                  # Degrees of freedom for variance
    count_zeros: bool = True       # Include zeros in statistics


# =============================================================================
# Global Configuration Manager
# =============================================================================

class SclConfig:
    """
    Global configuration manager for SCL.

    Provides thread-local configuration with context manager support.
    Configuration can be set globally or locally within a context.

    Example:
        # Global configuration
        scl.config.parallel.strategy = ParallelStrategy.SEQUENTIAL

        # Local configuration (context manager)
        with scl.config.local(parallel=ParallelConfig(num_threads=4)):
            # Use 4 threads here
            result = scl.ops.normalize(matrix)
        # Back to global config
    """

    def __init__(self):
        self._global_materialize = MaterializeConfig()
        self._global_parallel = ParallelConfig()
        self._global_memory = MemoryConfig()
        self._global_slice = SliceConfig()
        self._global_compute = ComputeConfig()

        # Thread-local storage for context overrides
        self._local = threading.local()

        # Callbacks for config changes
        self._callbacks: Dict[str, List[Callable]] = {
            "materialize": [],
            "parallel": [],
            "memory": [],
            "slice": [],
            "compute": [],
        }

    # -------------------------------------------------------------------------
    # Property Accessors (with thread-local override support)
    # -------------------------------------------------------------------------

    @property
    def materialize(self) -> MaterializeConfig:
        """Get materialize configuration."""
        if hasattr(self._local, "materialize") and self._local.materialize is not None:
            return self._local.materialize
        return self._global_materialize

    @materialize.setter
    def materialize(self, value: MaterializeConfig):
        """Set global materialize configuration."""
        self._global_materialize = value
        self._notify("materialize", value)

    @property
    def parallel(self) -> ParallelConfig:
        """Get parallel configuration."""
        if hasattr(self._local, "parallel") and self._local.parallel is not None:
            return self._local.parallel
        return self._global_parallel

    @parallel.setter
    def parallel(self, value: ParallelConfig):
        """Set global parallel configuration."""
        self._global_parallel = value
        self._notify("parallel", value)

    @property
    def memory(self) -> MemoryConfig:
        """Get memory configuration."""
        if hasattr(self._local, "memory") and self._local.memory is not None:
            return self._local.memory
        return self._global_memory

    @memory.setter
    def memory(self, value: MemoryConfig):
        """Set global memory configuration."""
        self._global_memory = value
        self._notify("memory", value)

    @property
    def slice(self) -> SliceConfig:
        """Get slice configuration."""
        if hasattr(self._local, "slice") and self._local.slice is not None:
            return self._local.slice
        return self._global_slice

    @slice.setter
    def slice(self, value: SliceConfig):
        """Set global slice configuration."""
        self._global_slice = value
        self._notify("slice", value)

    @property
    def compute(self) -> ComputeConfig:
        """Get compute configuration."""
        if hasattr(self._local, "compute") and self._local.compute is not None:
            return self._local.compute
        return self._global_compute

    @compute.setter
    def compute(self, value: ComputeConfig):
        """Set global compute configuration."""
        self._global_compute = value
        self._notify("compute", value)

    # -------------------------------------------------------------------------
    # Convenience Properties
    # -------------------------------------------------------------------------

    @property
    def lazy(self) -> bool:
        """Whether lazy evaluation is enabled."""
        return self.slice.lazy

    @lazy.setter
    def lazy(self, value: bool):
        """Set lazy evaluation."""
        self._global_slice.lazy = value

    @property
    def num_threads(self) -> int:
        """Number of threads for parallel execution."""
        return self.parallel.num_threads

    @num_threads.setter
    def num_threads(self, value: int):
        """Set number of threads."""
        self._global_parallel.num_threads = value

    @property
    def alignment(self) -> int:
        """Memory alignment."""
        return self.memory.alignment

    @alignment.setter
    def alignment(self, value: int):
        """Set memory alignment."""
        self._global_memory.alignment = value

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    def local(self, **kwargs) -> "_LocalConfigContext":
        """
        Create a local configuration context.

        Args:
            **kwargs: Configuration overrides (materialize, parallel, memory, slice, compute)

        Returns:
            Context manager
        """
        return _LocalConfigContext(self, **kwargs)

    def _set_local(self, **kwargs):
        """Set thread-local configuration."""
        for key, value in kwargs.items():
            if value is not None:
                setattr(self._local, key, value)

    def _clear_local(self, keys: List[str]):
        """Clear thread-local configuration."""
        for key in keys:
            if hasattr(self._local, key):
                setattr(self._local, key, None)

    # -------------------------------------------------------------------------
    # Callback Registration
    # -------------------------------------------------------------------------

    def on_change(self, config_name: str, callback: Callable):
        """
        Register callback for configuration changes.

        Args:
            config_name: Name of config ("materialize", "parallel", etc.)
            callback: Function to call when config changes
        """
        if config_name in self._callbacks:
            self._callbacks[config_name].append(callback)

    def _notify(self, config_name: str, value: Any):
        """Notify callbacks of configuration change."""
        for callback in self._callbacks.get(config_name, []):
            try:
                callback(value)
            except Exception:
                pass  # Ignore callback errors

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self):
        """Reset all configurations to defaults."""
        self._global_materialize = MaterializeConfig()
        self._global_parallel = ParallelConfig()
        self._global_memory = MemoryConfig()
        self._global_slice = SliceConfig()
        self._global_compute = ComputeConfig()

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "materialize": {
                "strategy": self.materialize.strategy.name,
                "chunk_size": self.materialize.chunk_size,
                "prefetch": self.materialize.prefetch,
                "max_memory_mb": self.materialize.max_memory_mb,
            },
            "parallel": {
                "strategy": self.parallel.strategy.name,
                "num_threads": self.parallel.num_threads,
                "min_elements_per_thread": self.parallel.min_elements_per_thread,
                "chunk_size": self.parallel.chunk_size,
            },
            "memory": {
                "strategy": self.memory.strategy.name,
                "alignment": self.memory.alignment,
                "use_pool": self.memory.use_pool,
                "max_pages": self.memory.max_pages,
            },
            "slice": {
                "lazy": self.slice.lazy,
                "copy_threshold": self.slice.copy_threshold,
                "preserve_order": self.slice.preserve_order,
            },
            "compute": {
                "norm_type": self.compute.norm_type.name,
                "epsilon": self.compute.epsilon,
                "ddof": self.compute.ddof,
                "count_zeros": self.compute.count_zeros,
            },
        }

    def __repr__(self) -> str:
        return f"SclConfig({self.to_dict()})"


class _LocalConfigContext:
    """Context manager for local configuration override."""

    def __init__(self, config: SclConfig, **kwargs):
        self._config = config
        self._kwargs = kwargs
        self._keys = list(kwargs.keys())

    def __enter__(self):
        self._config._set_local(**self._kwargs)
        return self._config

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._config._clear_local(self._keys)
        return False


# =============================================================================
# Global Instance
# =============================================================================

# Global configuration instance
config = SclConfig()


# =============================================================================
# Convenience Functions
# =============================================================================

def get_config() -> SclConfig:
    """Get the global configuration instance."""
    return config


def set_lazy(enabled: bool = True):
    """Enable or disable lazy evaluation globally."""
    config.lazy = enabled


def set_parallel(num_threads: int = 0, strategy: ParallelStrategy = ParallelStrategy.AUTO):
    """
    Configure parallel execution.

    Args:
        num_threads: Number of threads (0 = auto)
        strategy: Parallel strategy
    """
    config.parallel = ParallelConfig(
        strategy=strategy,
        num_threads=num_threads,
    )


def set_memory(max_pages: int = 64, alignment: int = 64):
    """
    Configure memory management.

    Args:
        max_pages: Maximum resident pages for mapped mode
        alignment: Memory alignment
    """
    config.memory = MemoryConfig(
        max_pages=max_pages,
        alignment=alignment,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Strategy enums
    "MaterializeStrategy",
    "ParallelStrategy",
    "MemoryStrategy",
    "NormType",
    "BackendType",
    # Config classes
    "MaterializeConfig",
    "ParallelConfig",
    "MemoryConfig",
    "SliceConfig",
    "ComputeConfig",
    # Main config class
    "SclConfig",
    # Global instance
    "config",
    # Convenience functions
    "get_config",
    "set_lazy",
    "set_parallel",
    "set_memory",
]
