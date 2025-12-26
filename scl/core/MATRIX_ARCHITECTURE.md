# SCL Matrix System Architecture

## 设计理念

基于**完全解耦**的三层架构：

```
Layer 0: Concepts (matrix.hpp)
  ↓ 定义接口契约
Layer 1: Base Implementations (sparse.hpp, dense.hpp)
  ↓ 纯数据类，无内存管理
Layer 2: Storage Variants (io/mmatrix.hpp, io/h5_tools.hpp)
  ↓ 具体存储策略
Algorithms (kernel/*)
  ↓ 只依赖Concepts
```

---

## 文件组织

```
scl/core/
├── matrix.hpp          # Concepts + Traits (接口定义)
├── sparse.hpp          # Base sparse types (CustomCSR, VirtualCSR, etc.)
└── dense.hpp           # Base dense types (DenseArray, DenseDeque)

scl/io/
├── mmatrix.hpp         # MountMatrix (mmap-based CSR)
└── h5_tools.hpp        # DequeCSR (HDF5-based CSR)
```

---

## Layer 0: Concepts (matrix.hpp)

### 核心Concepts

```cpp
// Dense matrices
concept DenseLike

// Sparse matrices (unified CSR/CSC)
concept SparseLike        // CSR or CSC
concept CSRLike           // CSR-specific
concept CSCLike           // CSC-specific

// Storage patterns
concept CustomSparseLike  // Contiguous storage
concept VirtualSparseLike // Indirection-based

// Refined concepts
concept CustomCSRLike     // CustomSparseLike + CSRLike
concept VirtualCSRLike    // VirtualSparseLike + CSRLike
```

### Unified Sparse Interface

**关键创新**: CSR和CSC共享`SparseLike` concept

```cpp
template <SparseLike M>
void algorithm(M& mat) {
    Index n = primary_size(mat);
    
    for (Index i = 0; i < n; ++i) {
        // 自动分派
        auto vals = primary_values(mat, i);  // CSR: row_values, CSC: col_values
        auto idxs = primary_indices(mat, i); // CSR: row_indices, CSC: col_indices
        // ...
    }
}
```

**辅助函数**:
- `primary_size(mat)`: rows for CSR, cols for CSC
- `primary_values(mat, i)`: 自动dispatch
- `primary_indices(mat, i)`: 自动dispatch
- `primary_length(mat, i)`: 自动dispatch

---

## Layer 1: Base Types (sparse.hpp, dense.hpp)

### 设计原则

1. **纯数据类** (POD-like)
   - 所有成员public
   - 无构造函数分配内存
   - 无析构函数释放内存

2. **零ABI复杂度**
   - 简单struct布局
   - 无虚函数
   - 跨语言友好

3. **生命周期外部化**
   - 用户管理内存
   - 指针借用语义
   - 无ownership转移

### sparse.hpp

#### CustomCSR / CustomCSC

```cpp
template <typename T>
struct CustomCSR {
    using ValueType = T;
    using Tag = TagCSR;
    
    // Pure data members (direct access)
    T* data;
    Index* indices;
    Index* indptr;
    const Index* row_lengths;  // Optional
    
    Index rows, cols, nnz;
    
    // Minimal interface
    Span<T> row_values(Index i) const;
    Span<Index> row_indices(Index i) const;
    Index row_length(Index i) const;
};
```

**特点**:
- ✅ 实现`CustomCSRLike`
- ✅ 连续存储，SIMD友好
- ✅ scipy兼容

#### VirtualCSR / VirtualCSC

```cpp
template <typename T>
struct VirtualCSR {
    using ValueType = T;
    using Tag = TagCSR;
    
    // Source + mapping (pure data)
    T* src_data;
    Index* src_indices;
    Index* src_indptr;
    const Index* row_map;      // Key: indirection array
    const Index* src_row_lengths;
    
    Index rows, cols, src_rows, nnz;
    
    // Interface with indirection
    Index row_length(Index i) const {
        Index phys = row_map[i];  // +1 indirection
        return src_indptr[phys+1] - src_indptr[phys];
    }
};
```

**特点**:
- ✅ 实现`VirtualCSRLike`
- ✅ 零拷贝切片
- ✅ +1% overhead

### dense.hpp

#### DenseArray

```cpp
template <typename T>
struct DenseArray {
    using ValueType = T;
    using Tag = TagDense;
    
    T* ptr;
    Index rows, cols;
    
    T& operator()(Index r, Index c) const;
};
```

#### DenseDeque

```cpp
template <typename T>
class DenseDeque {
    T* ptr;  // nullptr (placeholder)
    Index rows, cols;
    
    std::vector<std::deque<T>> _rows;  // Private storage
    
public:
    T& operator()(Index r, Index c) const;
    std::vector<T> materialize() const;
};
```

---

## Layer 2: Storage Implementations

### MountMatrix (mmatrix.hpp)

```cpp
template <typename T>
class MountMatrix {
    using ValueType = T;
    using Tag = TagCSR;
    
    // Mmap storage (private)
    MappedArray<T> _data;
    MappedArray<Index> _indices;
    MappedArray<Index> _indptr;
    
public:
    // Public accessors (concept interface)
    Index rows, cols, nnz;
    
    // CSRLike interface
    Span<T> row_values(Index i) const;
    Span<Index> row_indices(Index i) const;
    Index row_length(Index i) const;
    
    // Storage access
    const T* data() const { return _data.data(); }
    const Index* indices() const { return _indices.data(); }
    const Index* indptr() const { return _indptr.data(); }
};
```

**实现**: `CustomCSRLike` (通过data()等方法暴露指针)

### DequeCSR (h5_tools.hpp)

```cpp
template <typename T>
class DequeCSR {
    using ValueType = T;
    using Tag = TagCSR;
    
    std::vector<RowSegment<T>> _segments;  // Private
    
public:
    Index rows, cols, nnz;
    
    // CSRLike interface
    Span<const T> row_values(Index i) const;
    Span<const Index> row_indices(Index i) const;
    Index row_length(Index i) const;
    
    // Materialization
    OwnedCSR<T> materialize() const;
};
```

**实现**: `CSRLike` (但不是CustomCSRLike，因为非连续存储)

---

## Concept继承关系

```
                    SparseLike<M>
                    (CSR or CSC)
                         │
           ┌─────────────┴─────────────┐
           │                           │
      CSRLike<M>                  CSCLike<M>
      (CSR-specific)              (CSC-specific)
           │                           │
    ┌──────┴──────┐            ┌──────┴──────┐
    │             │            │             │
CustomCSRLike  VirtualCSRLike  CustomCSCLike  VirtualCSCLike
(contiguous)  (indirection)   (contiguous)   (indirection)
```

---

## 统一接口示例

### 示例1: SparseLike通用算法

```cpp
template <SparseLike M>
void normalize(M& mat) {
    Index n = primary_size(mat);
    
    for (Index i = 0; i < n; ++i) {
        auto vals = primary_values(mat, i);  // 自动dispatch CSR/CSC
        Index len = primary_length(mat, i);
        
        typename M::ValueType sum = 0;
        for (Index j = 0; j < len; ++j) {
            sum += vals[j];
        }
        
        if (sum > 0) {
            for (Index j = 0; j < len; ++j) {
                vals[j] /= sum;
            }
        }
    }
}

// 使用
CustomCSR<float> csr = ...;
normalize(csr);  // ✓

CustomCSC<float> csc = ...;
normalize(csc);  // ✓ 自动适配!
```

### 示例2: CSRLike专用算法

```cpp
template <CSRLike M>
void row_specific_algo(M& mat) {
    for (Index i = 0; i < mat.rows; ++i) {
        auto vals = mat.row_values(i);
        // CSR-specific logic
    }
}

// 使用
CustomCSR<float> csr = ...;
row_specific_algo(csr);  // ✓

CustomCSC<float> csc = ...;
row_specific_algo(csc);  // ✗ 编译错误（正确！）
```

### 示例3: CustomSparseLike SIMD优化

```cpp
template <CustomSparseLike M>
void simd_normalize(M& mat) {
    // 直接访问data指针，SIMD批处理
    simd::normalize_contiguous(mat.data, mat.nnz);
}

// 使用
CustomCSR<float> custom = ...;
simd_normalize(custom);  // ✓

VirtualCSR<float> virtual_csr = ...;
simd_normalize(virtual_csr);  // ✗ 编译错误（非连续存储）

DequeCSR<float> deque = ...;
simd_normalize(deque);  // ✗ 编译错误（非连续存储）
```

---

## 类型映射表

### Sparse Matrix Types

| Type | Tag | Storage | Custom? | Virtual? | Implements |
|------|-----|---------|---------|----------|-----------|
| CustomCSR | CSR | Heap contiguous | ✓ | ✗ | CustomCSRLike |
| CustomCSC | CSC | Heap contiguous | ✓ | ✗ | CustomCSCLike |
| VirtualCSR | CSR | View + mapping | ✗ | ✓ | VirtualCSRLike |
| VirtualCSC | CSC | View + mapping | ✗ | ✓ | VirtualCSCLike |
| MountMatrix | CSR | Mmap contiguous | ✓ | ✗ | CustomCSRLike |
| VirtualMountMatrix | CSR | Mmap + mapping | ✗ | ✓ | VirtualCSRLike |
| DequeCSR | CSR | Deque discontiguous | ✗ | ✗ | CSRLike only |
| OwnedCSR | CSR | Owned vectors | ✓ | ✗ | CustomCSRLike |

### Dense Matrix Types

| Type | Storage | Contiguous? | Implements |
|------|---------|-------------|-----------|
| DenseArray | Array | ✓ | DenseLike |
| DenseDeque | Deque | ✗ | DenseLike |

---

## 使用指南

### 算法设计

```cpp
// 1. 最通用：支持所有sparse类型
template <SparseLike M>
void generic_sparse(M& mat);

// 2. CSR专用
template <CSRLike M>
void csr_only(M& mat);

// 3. 需要连续存储（SIMD）
template <CustomCSRLike M>
void csr_simd(M& mat);

// 4. 支持CSR或CSC（手动dispatch）
template <SparseLike M>
void csr_or_csc(M& mat) {
    if constexpr (std::is_same_v<typename M::Tag, TagCSR>) {
        // CSR path
    } else {
        // CSC path
    }
}
```

### 类型选择

```cpp
// 场景1: 小数据，需要SIMD
CustomCSR<float> mat = allocate_csr(...);
kernel_simd(mat);  // ✓ CustomCSRLike

// 场景2: 大数据，零拷贝
MountMatrix<float> mat = mount_from_file(...);
kernel_generic(mat);  // ✓ CustomCSRLike

// 场景3: 切片操作
VirtualCSR<float> slice = make_virtual(mat, row_indices);
kernel_generic(slice);  // ✓ VirtualCSRLike

// 场景4: HDF5增量加载
DequeCSR<float> deque = h5::load_csr_masked(...);
kernel_generic(deque);  // ✓ CSRLike
```

---

## 核心优势

### 1. 完全解耦

**算子 ↔ 数据结构 ↔ 存储**

```
Algorithm:
  template <CSRLike M> void kernel(M& mat);
  ↓ 依赖
Concept:
  CSRLike
  ↓ 实现
Data Structure:
  CustomCSR, VirtualCSR
  ↓ 实例化
Storage:
  Heap, Mmap, Deque, ...
```

### 2. 统一Sparse接口

**单一算法支持CSR和CSC**:

```cpp
template <SparseLike M>
void algorithm(M& mat) {
    // 自动适配CSR/CSC
    for (Index i = 0; i < primary_size(mat); ++i) {
        auto vals = primary_values(mat, i);
        // ...
    }
}
```

### 3. 纯数据类设计

**避免ABI问题**:

```cpp
// ✓ Good: Pure data
struct CustomCSR {
    float* data;  // Direct access
    Index rows;   // Direct access
};

// ✗ Avoid: Hidden state
class BadCSR {
private:
    std::vector<float> _data;  // Managed memory
public:
    const float* data() const;  // Getter
    ~BadCSR();  // Destructor - ABI issues
};
```

**优势**:
- ✅ 跨语言FFI简单
- ✅ 无hidden state
- ✅ 用户完全控制
- ✅ 无生命周期陷阱

---

## 扩展新类型

### 示例: GPU CSR

```cpp
// 1. 在新文件 gpu/sparse_gpu.hpp 中定义
template <typename T>
struct CudaCSR {
    using ValueType = T;
    using Tag = TagCSR;
    
    // GPU指针（仍然是纯数据）
    T* data;  // device pointer
    Index* indices;
    Index* indptr;
    const Index* row_lengths;
    
    Index rows, cols, nnz;
    
    // CSRLike接口（返回device span）
    Span<T> row_values(Index i) const;
    Span<Index> row_indices(Index i) const;
    Index row_length(Index i) const;
};

// 2. 验证
static_assert(CSRLike<CudaCSR<float>>);  // ✓

// 3. 完成！所有算法自动支持
template <CSRLike M>
void kernel_generic(M& mat);

CudaCSR<float> gpu_mat = ...;
kernel_generic(gpu_mat);  // ✓ 自动支持GPU!
```

### 示例: 网络CSR (分布式)

```cpp
template <typename T>
struct NetworkCSR {
    using ValueType = T;
    using Tag = TagCSR;
    
    // 网络句柄（纯数据）
    int socket_fd;
    uint64_t remote_addr;
    
    Index rows, cols, nnz;
    
    // CSRLike接口（通过RPC获取）
    Span<T> row_values(Index i) const {
        // RPC call to fetch row
        auto buffer = rpc_fetch_row(socket_fd, i);
        return Span<T>(buffer.data(), buffer.size());
    }
};

static_assert(CSRLike<NetworkCSR<float>>);  // ✓
```

---

## 性能特性

### Concept开销

| Concept | 编译期开销 | 运行期开销 | 说明 |
|---------|----------|----------|------|
| SparseLike | 是 | 0% | 纯编译期检查 |
| CustomSparseLike | 是 | 0% | 额外约束 |
| VirtualSparseLike | 是 | ~1% | +1 indirection |

### 类型开销

| Type | 内存开销 | 访问开销 | Cache友好度 |
|------|---------|---------|-----------|
| CustomCSR | 1x | O(1) | ⭐⭐⭐ |
| VirtualCSR | 1x + mapping | O(1) + 1 | ⭐⭐ |
| MountMatrix | ~0 (mmap) | O(1) | ⭐⭐⭐ |
| DequeCSR | 1.2x | O(1) | ⭐ |

---

## 最佳实践

### Do ✅

1. **算法依赖Concept**
   ```cpp
   template <CSRLike M>
   void algorithm(M& mat);
   ```

2. **直接访问成员**
   ```cpp
   template <CustomCSRLike M>
   void simd_algo(M& mat) {
       simd::process(mat.data, mat.nnz);  // Direct access
   }
   ```

3. **使用辅助函数统一CSR/CSC**
   ```cpp
   template <SparseLike M>
   void unified(M& mat) {
       auto vals = primary_values(mat, i);  // Works for both
   }
   ```

### Don't ❌

1. **不要在基类中分配内存**
   ```cpp
   // Bad
   struct CustomCSR {
       std::vector<T> _data;  // Owns memory
   };
   ```

2. **不要隐藏成员**
   ```cpp
   // Bad
   class CustomCSR {
   private:
       T* data;
   public:
       const T* get_data() const;  // Getter overhead
   };
   ```

3. **不要使用虚函数（除非Layer 3）**
   ```cpp
   // Bad (unless type erasure needed)
   struct CustomCSR {
       virtual Span<T> row_values(Index i) const;
   };
   ```

---

## 编译期验证

所有类型在编译期自动验证：

```cpp
// In sparse.hpp
static_assert(CSRLike<CustomCSR<float>>);
static_assert(CustomCSRLike<CustomCSR<float>>);
static_assert(VirtualCSRLike<VirtualCSR<float>>);

// In mmatrix.hpp
static_assert(CSRLike<MountMatrix<float>>);
static_assert(CustomCSRLike<MountMatrix<float>>);

// In h5_tools.hpp
static_assert(CSRLike<DequeCSR<float>>);
```

**好处**: 接口变更立即在编译期发现

---

## 总结

### 架构优势

| 维度 | 传统设计 | SCL新架构 |
|------|---------|----------|
| 算子-存储耦合 | 强耦合 | 完全解耦 |
| CSR/CSC统一 | 分开实现 | 统一接口 |
| 内存管理 | 隐式 | 显式外部化 |
| ABI复杂度 | 高 | 零 |
| 扩展性 | 困难 | 极简 |
| 类型安全 | 运行期 | 编译期 |

### 关键创新

1. **Concept-Based Complete Decoupling**
   - 算法、数据、存储三层解耦
   - C++20 Concepts系统化应用

2. **Unified Sparse Interface**
   - CSR/CSC共享SparseLike
   - primary_*()辅助函数

3. **Pure Data Classes**
   - 零ABI复杂度
   - 生命周期外部化
   - FFI友好

4. **Multi-Storage Support**
   - Heap, Mmap, Deque统一接口
   - 易于添加新后端

---

**文档版本**: 4.0 (Architecture Refactored)  
**状态**: ✅ Complete  
**日期**: 2025-01

