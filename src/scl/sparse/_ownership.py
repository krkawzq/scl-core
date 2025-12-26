"""Ownership and Reference Management.

This module provides utilities for tracking data ownership and
maintaining reference chains to prevent use-after-free bugs.

Key Concepts:
    - Reference Chain: When matrix B is derived from A, B holds
      a reference to A to prevent A from being garbage collected.
    - Automatic Flattening: Nested reference chains are flattened
      to avoid deep hierarchies.
    - Weak vs Strong: Views use strong refs; borrowed data uses
      weak refs with validity checking.

Safety Model:
    1. OWNED data: No external dependencies, always safe
    2. BORROWED data: Caller responsible for source lifetime
    3. VIEW data: Automatic reference management
"""

from typing import Any, List, Optional, Set, Union
from weakref import ref, WeakSet
from dataclasses import dataclass, field

__all__ = [
    'RefChain',
    'OwnershipTracker',
    'ensure_alive',
]


# =============================================================================
# Reference Chain
# =============================================================================

@dataclass
class RefChain:
    """Maintains reference chain for view matrices.
    
    When a matrix is created as a view of another, the source
    must be kept alive. RefChain stores strong references to
    all ancestors in the view hierarchy.
    
    Attributes:
        _refs: List of strong references to ancestors.
        
    Design:
        - Strong refs prevent premature GC
        - Flattening avoids deep chains
        - Uses list to support unhashable types (scipy matrices)
        
    Example:
        >>> mat1 = SclCSR.from_dense(...)
        >>> mat2 = mat1[0:100, :]  # mat2._refs contains mat1
        >>> mat3 = mat2[0:50, :]   # mat3._refs contains mat1 (flattened)
        >>> del mat1, mat2          # mat3 still valid, holds refs
    """
    _refs: List[Any] = field(default_factory=list)
    
    def add(self, source: Any) -> None:
        """Add source to reference chain.
        
        If source has its own RefChain, flatten it by
        adding all its references too.
        
        Args:
            source: Source matrix to reference.
        """
        if source is None:
            return
        
        # Check if already in refs (by identity)
        for ref in self._refs:
            if ref is source:
                return
        
        # Add the direct source
        self._refs.append(source)
        
        # Flatten: add source's ancestors too
        if hasattr(source, '_ref_chain') and source._ref_chain:
            for ancestor in source._ref_chain._refs:
                # Check identity
                found = False
                for ref in self._refs:
                    if ref is ancestor:
                        found = True
                        break
                if not found:
                    self._refs.append(ancestor)
    
    def add_multiple(self, sources: List[Any]) -> None:
        """Add multiple sources to reference chain.
        
        Args:
            sources: List of source matrices.
        """
        for src in sources:
            self.add(src)
    
    def clear(self) -> None:
        """Clear all references (use with caution)."""
        self._refs.clear()
    
    @property
    def count(self) -> int:
        """Number of held references."""
        return len(self._refs)
    
    @property
    def is_empty(self) -> bool:
        """Check if chain is empty."""
        return len(self._refs) == 0
    
    def __repr__(self) -> str:
        return f"RefChain(count={self.count})"


# =============================================================================
# Ownership Tracker
# =============================================================================

class OwnershipTracker:
    """Tracks ownership and validity of borrowed data.
    
    For BORROWED ownership, tracks the source object and provides
    validity checking to detect dangling references.
    
    Attributes:
        _weak_ref: Weak reference to borrowed source.
        _strong_ref: Strong reference (for views).
        _is_owned: Whether we own the data.
        
    Warning:
        Accessing borrowed data after source is freed is undefined
        behavior. Use is_valid() to check before access.
        
    Example:
        >>> tracker = OwnershipTracker.borrowed(scipy_mat)
        >>> if tracker.is_valid:
        ...     # Safe to use
        >>> else:
        ...     raise RuntimeError("Source was freed!")
    """
    
    def __init__(
        self,
        source: Optional[Any] = None,
        owned: bool = True,
        use_weak_ref: bool = True
    ):
        """Initialize tracker.
        
        Args:
            source: Source object (for borrowed/view).
            owned: Whether we own the data.
            use_weak_ref: Use weak ref (borrowed) or strong ref (view).
        """
        self._is_owned = owned
        self._weak_ref: Optional[ref] = None
        self._strong_ref: Optional[Any] = None
        
        if source is not None and not owned:
            if use_weak_ref:
                try:
                    self._weak_ref = ref(source)
                except TypeError:
                    # Some objects can't be weakref'd
                    self._strong_ref = source
            else:
                self._strong_ref = source
    
    @classmethod
    def owned(cls) -> 'OwnershipTracker':
        """Create tracker for owned data."""
        return cls(source=None, owned=True)
    
    @classmethod
    def borrowed(cls, source: Any) -> 'OwnershipTracker':
        """Create tracker for borrowed data (weak ref)."""
        return cls(source=source, owned=False, use_weak_ref=True)
    
    @classmethod
    def view(cls, source: Any) -> 'OwnershipTracker':
        """Create tracker for view data (strong ref)."""
        return cls(source=source, owned=False, use_weak_ref=False)
    
    @property
    def is_owned(self) -> bool:
        """Check if data is owned."""
        return self._is_owned
    
    @property
    def is_borrowed(self) -> bool:
        """Check if data is borrowed."""
        return not self._is_owned and self._weak_ref is not None
    
    @property
    def is_view(self) -> bool:
        """Check if data is a view."""
        return not self._is_owned and self._strong_ref is not None
    
    @property
    def is_valid(self) -> bool:
        """Check if borrowed source is still alive.
        
        Returns:
            True if owned, view, or borrowed source still exists.
            False if borrowed source was garbage collected.
        """
        if self._is_owned:
            return True
        if self._strong_ref is not None:
            return True
        if self._weak_ref is not None:
            return self._weak_ref() is not None
        return True  # No source to track
    
    @property
    def source(self) -> Optional[Any]:
        """Get source object (if any).
        
        Returns:
            Source object or None.
            
        Raises:
            RuntimeError: If borrowed source was freed.
        """
        if self._is_owned:
            return None
        if self._strong_ref is not None:
            return self._strong_ref
        if self._weak_ref is not None:
            src = self._weak_ref()
            if src is None:
                raise RuntimeError(
                    "Borrowed source was garbage collected! "
                    "Ensure source outlives this matrix."
                )
            return src
        return None
    
    def ensure_valid(self) -> None:
        """Raise if borrowed source is invalid.
        
        Raises:
            RuntimeError: If source was garbage collected.
        """
        if not self.is_valid:
            raise RuntimeError(
                "Borrowed data source was garbage collected! "
                "This matrix is no longer valid."
            )
    
    def __repr__(self) -> str:
        if self._is_owned:
            return "OwnershipTracker(owned)"
        elif self._strong_ref is not None:
            return f"OwnershipTracker(view of {type(self._strong_ref).__name__})"
        elif self._weak_ref is not None:
            alive = "alive" if self.is_valid else "dead"
            return f"OwnershipTracker(borrowed, {alive})"
        return "OwnershipTracker(unknown)"


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_alive(obj: Any) -> None:
    """Ensure object's data source is still valid.
    
    Checks if object has ownership tracking and validates it.
    
    Args:
        obj: Object to check.
        
    Raises:
        RuntimeError: If data source was freed.
    """
    if hasattr(obj, '_ownership') and obj._ownership is not None:
        obj._ownership.ensure_valid()
    if hasattr(obj, '_ref_chain') and obj._ref_chain is not None:
        # All refs in chain should be alive
        for ref_obj in obj._ref_chain._refs:
            ensure_alive(ref_obj)

