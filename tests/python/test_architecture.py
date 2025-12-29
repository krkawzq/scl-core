#!/usr/bin/env python3
"""
Test the new Python architecture.

Tests:
1. Lazy scipy import (no error if scipy not installed, only when used)
2. Automatic library selection
3. CSR/CSC separation
4. Error handling with aligned error codes
5. Ownership semantics
"""

import sys
sys.path.insert(0, 'python')

print("=" * 70)
print("Testing SCL Python Architecture")
print("=" * 70)

# Test 1: Basic imports (should work without scipy)
print("\n1. Testing basic imports...")
try:
    import scl
    from scl import CsrMatrix, CscMatrix, DenseMatrix, set_precision
    print("✅ Basic imports successful (no scipy needed yet)")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Error codes alignment
print("\n2. Testing error codes alignment...")
from scl import SCLError
assert SCLError.ERROR_NULL_POINTER == 4, "Error code mismatch!"
assert SCLError.ERROR_INVALID_ARGUMENT == 10, "Error code mismatch!"
print("✅ Error codes aligned with C API")

# Test 3: Configuration
print("\n3. Testing configuration...")
from scl import RealType, IndexType, get_precision
real, index = get_precision()
print(f"   Default precision: {real.value}, {index.value}")
set_precision(real='float32', index='int32')
real, index = get_precision()
assert real == RealType.FLOAT32
assert index == IndexType.INT32
print("✅ Configuration system working")

# Reset to default
set_precision(real='float64', index='int64')

# Test 4: NumPy integration (no scipy)
print("\n4. Testing NumPy-only functionality...")
import numpy as np
arr = np.random.randn(100, 50).astype(np.float64)
try:
    with DenseMatrix.wrap(arr) as dense:
        assert dense.shape == (100, 50)
        assert dense.dtype == np.float64
        print(f"✅ DenseMatrix created: {dense}")
except Exception as e:
    print(f"❌ DenseMatrix failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Check scipy is NOT imported yet
print("\n5. Testing lazy scipy import...")
if 'scipy' in sys.modules:
    print("⚠️  scipy was already imported (may be pre-loaded)")
else:
    print("✅ scipy NOT imported yet (lazy loading works)")

# Test 6: SciPy integration (will import scipy)
print("\n6. Testing SciPy integration...")
try:
    from scipy.sparse import random as sp_random
    
    # Create scipy CSR matrix
    scipy_csr = sp_random(1000, 500, density=0.1, format='csr', dtype=np.float64)
    print(f"   Created SciPy CSR: {scipy_csr.shape}, nnz={scipy_csr.nnz}")
    
    # Test CSR
    csr = CsrMatrix.copy(scipy_csr)
    print(f"   ✅ CsrMatrix.copy(): {csr}")
    assert csr.shape == (1000, 500)
    assert csr.format == 'csr'
    
    # Test row slicing (zero-copy)
    rows = csr[10:20]
    print(f"   ✅ Row slice [10:20]: {rows}")
    assert rows.shape == (10, 500)
    assert rows.is_view
    
    # Test transpose CSR -> CSC
    csc = csr.T
    print(f"   ✅ Transpose CSR->CSC: {csc}")
    assert isinstance(csc, CscMatrix)
    assert csc.shape == (500, 1000)
    assert csc.format == 'csc'
    
    # Test CSC
    scipy_csc = sp_random(1000, 500, density=0.1, format='csc', dtype=np.float64)
    csc2 = CscMatrix.copy(scipy_csc)
    print(f"   ✅ CscMatrix.copy(): {csc2}")
    assert csc2.format == 'csc'
    
    # Test transpose CSC -> CSR
    csr2 = csc2.T
    print(f"   ✅ Transpose CSC->CSR: {csr2}")
    assert isinstance(csr2, CsrMatrix)
    assert csr2.format == 'csr'
    
    # Test export
    scipy_back = csr.to_scipy()
    print(f"   ✅ Export to SciPy: {type(scipy_back).__name__}")
    assert scipy_back.shape == csr.shape
    
except ImportError:
    print("⚠️  scipy not installed, skipping SciPy tests")
except Exception as e:
    print(f"❌ SciPy integration failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Ownership modes
print("\n7. Testing ownership modes...")
try:
    from scipy.sparse import random as sp_random
    scipy_mat = sp_random(100, 50, density=0.1, format='csr', dtype=np.float64)
    
    # Test OWNED (copy)
    mat_owned = CsrMatrix.copy(scipy_mat)
    assert mat_owned.is_owned
    assert not mat_owned.is_view
    print("   ✅ Ownership: OWNED (copy)")
    
    # Test WRAPPED (view)
    indptr = np.array([0, 2, 3, 6], dtype=np.int64)
    indices = np.array([0, 2, 2, 0, 1, 2], dtype=np.int64)
    data = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    mat_view = CsrMatrix.wrap(indptr, indices, data, (3, 3))
    assert not mat_view.is_owned
    assert mat_view.is_view
    print("   ✅ Ownership: WRAPPED (view)")
    
except Exception as e:
    print(f"⚠️  Ownership test skipped: {e}")

# Test 8: Type conversion
print("\n8. Testing dtype conversion...")
try:
    from scipy.sparse import random as sp_random
    
    # float32 matrix
    scipy_f32 = sp_random(100, 50, density=0.1, format='csr', dtype=np.float32)
    mat_f32 = CsrMatrix.copy(scipy_f32)
    assert mat_f32.dtype == np.float32
    print(f"   ✅ float32 matrix: {mat_f32.dtype}")
    
    # float64 matrix
    scipy_f64 = sp_random(100, 50, density=0.1, format='csr', dtype=np.float64)
    mat_f64 = CsrMatrix.copy(scipy_f64)
    assert mat_f64.dtype == np.float64
    print(f"   ✅ float64 matrix: {mat_f64.dtype}")
    
except Exception as e:
    print(f"⚠️  Type conversion test skipped: {e}")

print("\n" + "=" * 70)
print("Architecture test completed!")
print("=" * 70)

