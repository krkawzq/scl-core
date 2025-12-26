# SCL Sparse Module 架构文档

> 版本: 0.2.0  
> 最后更新: 2024

## 📋 概述

SCL Sparse Module 提供高性能稀疏矩阵数据结构，具有：

- **智能后端管理**: 自动在 Custom/Virtual/Mapped 后端之间切换
- **透明所有权跟踪**: 自动管理数据所有权和引用链
- **跨平台互操作**: 无缝支持 scipy、numpy、anndata
- **零拷贝视图**: 高效的切片和堆叠操作

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    SclCSR / SclCSC (Smart Matrix)               │
│  统一接口，自动后端管理，用户无需关心底层存储细节               │
├─────────────────────────────────────────────────────────────────┤
│                        Backend Types                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   CUSTOM     │  │   VIRTUAL    │  │   MAPPED     │          │
│  │  本地数组    │  │  零拷贝视图  │  │  内存映射    │          │
│  │  完全控制    │  │  vstack/切片 │  │  大数据支持  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                       Ownership Model                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    OWNED     │  │   BORROWED   │  │     VIEW     │          │
│  │  拥有数据    │  │  借用外部    │  │  派生视图    │          │
│  │  负责释放    │  │  不负责释放  │  │  维护引用链  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                       Storage Layer                             │
│  CustomStorage | VirtualStorage | MappedStorage                 │
│  引用链管理 (RefChain) + 所有权追踪 (OwnershipTracker)          │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 模块结构

```
scl/sparse/
├── __init__.py          # 公开 API 导出 (53个符号)
├── _array.py            # 轻量级数组容器 (无 numpy 依赖)
├── _dtypes.py           # 数据类型定义
├── _backend.py          # 后端类型和存储抽象
├── _ownership.py        # 所有权和引用管理
├── _csr.py              # 智能 CSR 实现
├── _csc.py              # 智能 CSC 实现
└── _ops.py              # 高级操作函数
```

## 🎯 核心概念

### 1. Backend (后端类型)

```python
from scl.sparse import Backend

class Backend(Enum):
    CUSTOM = 'custom'   # 本地数组，完全控制
    VIRTUAL = 'virtual' # 零拷贝视图，支持 vstack/切片
    MAPPED = 'mapped'   # 内存映射文件（大数据）
```

**特点:**
- `CUSTOM`: 数据存储在本地 Array 中，所有操作立即执行
- `VIRTUAL`: 逻辑视图，不复制数据，支持延迟计算
- `MAPPED`: 文件后端，数据保留在磁盘上，按需加载

### 2. Ownership (所有权模型)

```python
from scl.sparse import Ownership

class Ownership(Enum):
    OWNED = 'owned'       # 拥有数据，负责释放
    BORROWED = 'borrowed' # 借用外部数据
    VIEW = 'view'         # 派生视图
```

**内存安全:**
- `OWNED`: 数据由 SCL 管理，无外部依赖
- `BORROWED`: 来自 scipy 等外部库，调用者负责保持源数据存活
- `VIEW`: 自动维护引用链，防止上游被 GC 回收

### 3. Reference Chain (引用链)

当矩阵 B 是从矩阵 A 派生的视图时，B 会自动持有 A 的强引用：

```python
mat = SclCSR.from_dense([[1, 2], [3, 4]])
view = mat[0:1, :]  # view 持有 mat 的引用

del mat  # 安全！view 内部仍然引用 mat
print(view[0, 0])  # 正常工作
```

引用链会自动展平，避免深层嵌套。

## 🚀 快速开始

### 创建矩阵

```python
from scl.sparse import SclCSR, SclCSC, Backend, Ownership

# 从 dense 列表创建 (OWNED)
mat = SclCSR.from_dense([[1, 0, 2], [0, 3, 0]])
print(mat.backend)     # Backend.CUSTOM
print(mat.ownership)   # Ownership.OWNED

# 从 scipy 创建 (BORROWED - 零拷贝)
import scipy.sparse as sp
scipy_mat = sp.csr_matrix([[1, 2], [3, 4]])
borrowed = SclCSR.from_scipy(scipy_mat, copy=False)
print(borrowed.ownership)  # Ownership.BORROWED

# 从 scipy 创建 (OWNED - 复制)
owned = SclCSR.from_scipy(scipy_mat, copy=True)
print(owned.ownership)  # Ownership.OWNED
```

### 智能切片

```python
# 行切片 -> 自动切换到 Virtual 后端
view = mat.slice_rows([0, 10, 20], strategy='virtual')
print(view.backend)  # Backend.VIRTUAL

# 索引操作
val = mat[0, 0]       # 单元素
row = mat[1, :]       # 整行 (dense Array)
col = mat[:, 2]       # 整列 (dense Array)
sub = mat[0:2, :]     # 行范围

# 列切片 -> 惰性执行
col_view = mat.slice_cols([0, 2], lazy=True)
```

### 堆叠操作

```python
from scl.sparse import vstack_csr, hstack_csc

# vstack 创建 Virtual 后端 (零拷贝)
mat1 = SclCSR.from_dense([[1, 2], [3, 4]])
mat2 = SclCSR.from_dense([[5, 6]])
stacked = vstack_csr([mat1, mat2])
print(stacked.backend)  # Backend.VIRTUAL
print(stacked.shape)    # (3, 2)

# 物化为 Custom (复制数据)
owned_stacked = stacked.to_owned()
print(owned_stacked.backend)  # Backend.CUSTOM
```

### 格式转换

```python
# CSR <-> CSC
csc = mat.tocsc()
csr = csc.tocsr()

# scipy 互转
scipy_mat = mat.to_scipy()
scl_mat = SclCSR.from_scipy(scipy_mat)

# AnnData 互转
from scl.sparse import from_anndata, to_anndata
mat = from_anndata(adata)
adata = to_anndata(mat)
```

### 对齐操作

```python
from scl.sparse import align_rows, align_to_categories

# 按索引映射对齐
mapping = [2, 0, -1]  # 新矩阵: [row2, row0, empty_row]
aligned = align_rows(mat, mapping, new_rows=3)

# 按类别名称对齐
aligned = align_to_categories(
    mat,
    source_categories=adata.var_names.tolist(),
    target_categories=reference_genes,
    axis=1  # 列对齐
)
```

### 统计操作

```python
from scl.sparse import sum_rows, sum_cols, mean_rows, var_cols

row_sums = sum_rows(mat)   # Array (length = rows)
col_sums = sum_cols(csc)   # Array (length = cols)
row_means = mean_rows(mat)
col_vars = var_cols(csc)
```

## 📊 API 参考

### 核心类

| 类 | 描述 |
|---|---|
| `SclCSR` | 智能 CSR 稀疏矩阵 (行优先) |
| `SclCSC` | 智能 CSC 稀疏矩阵 (列优先) |
| `Array` | 轻量级连续数组 (无 numpy 依赖) |
| `Backend` | 后端类型枚举 |
| `Ownership` | 所有权模型枚举 |

### 工厂函数

| 函数 | 描述 |
|---|---|
| `from_scipy(mat, copy=False)` | 从 scipy 稀疏矩阵创建 |
| `from_anndata(adata, layer=None)` | 从 AnnData X 或 layer 创建 |
| `from_numpy(arr)` | 从 numpy 数组创建 |

### 堆叠函数

| 函数 | 描述 |
|---|---|
| `vstack_csr(matrices)` | 垂直堆叠 CSR 矩阵 (Virtual) |
| `hstack_csc(matrices)` | 水平堆叠 CSC 矩阵 (Virtual) |
| `vstack(matrices)` | 通用垂直堆叠 |
| `hstack(matrices)` | 通用水平堆叠 |

### 对齐函数

| 函数 | 描述 |
|---|---|
| `align_rows(mat, mapping, new_rows)` | 按映射对齐行 |
| `align_cols(mat, mapping, new_cols)` | 按映射对齐列 |
| `align_to_categories(mat, src, tgt, axis)` | 按类别名称对齐 |

### 统计函数

| 函数 | 描述 |
|---|---|
| `sum_rows(mat)` | 行求和 |
| `sum_cols(mat)` | 列求和 |
| `mean_rows(mat)` | 行均值 |
| `mean_cols(mat)` | 列均值 |
| `var_rows(mat, ddof=0)` | 行方差 |
| `var_cols(mat, ddof=0)` | 列方差 |

### 类型检查

| 函数 | 描述 |
|---|---|
| `is_sparse_like(obj)` | 检查是否为稀疏矩阵 |
| `is_csr_like(obj)` | 检查是否为 CSR 格式 |
| `is_csc_like(obj)` | 检查是否为 CSC 格式 |

## 🔧 高级用法

### 检查矩阵信息

```python
mat = SclCSR.from_dense([[1, 0], [0, 2]])

# 快速属性
print(mat.shape)      # (2, 2)
print(mat.nnz)        # 2
print(mat.dtype)      # 'float64'
print(mat.backend)    # Backend.CUSTOM
print(mat.ownership)  # Ownership.OWNED
print(mat.is_view)    # False

# 详细信息
print(mat.info())
# SclCSR Matrix:
#   shape: (2, 2)
#   nnz: 2
#   dtype: float64
#   backend: custom
#   ownership: owned
#   is_view: False
#   is_contiguous: True
#   memory: 0.07 KB
```

### 手动控制后端

```python
# 强制使用 Virtual 后端进行切片
view = mat.slice_rows([0, 1], strategy='virtual')

# 强制复制
copy = mat.slice_rows([0, 1], strategy='copy')

# 物化 Virtual 为 Custom
owned = view.to_owned()
```

### 获取底层指针 (C API 调用)

```python
# 获取 C 兼容指针
data_ptr, indices_ptr, indptr_ptr, lengths_ptr, rows, cols, nnz = mat.get_c_pointers()
```

## ⚠️ 注意事项

### 借用数据的生命周期

```python
# 危险: scipy 矩阵可能被 GC
def bad_example():
    scipy_mat = sp.csr_matrix([[1, 2]])
    return SclCSR.from_scipy(scipy_mat, copy=False)  # 借用

mat = bad_example()  # scipy_mat 已被 GC！
# mat 的数据现在是未定义的！

# 安全: 复制数据
def safe_example():
    scipy_mat = sp.csr_matrix([[1, 2]])
    return SclCSR.from_scipy(scipy_mat, copy=True)  # 复制
```

### Virtual 后端的限制

Virtual 后端不支持就地修改:

```python
stacked = vstack_csr([mat1, mat2])
# stacked.data[0] = 99  # 不支持！

# 先物化
owned = stacked.to_owned()
owned.data[0] = 99  # OK
```

## 📈 性能建议

1. **优先使用 Virtual**: vstack/hstack 操作使用 Virtual 是零拷贝的
2. **延迟物化**: 只在需要时调用 `to_owned()` 或访问 `.data`
3. **借用 scipy**: 对于临时操作，使用 `copy=False` 避免不必要的复制
4. **批量操作**: 使用 `align_to_categories` 而不是多次切片

## 🔄 从旧版本迁移

### 旧 API (已移除)

```python
# 旧代码 (不再工作)
from scl.sparse import VirtualCSR, VirtualCSC

virtual = VirtualCSR(scipy_mat)
virtual.vstack([mat2])
```

### 新 API

```python
# 新代码
from scl.sparse import SclCSR, vstack_csr, Backend

# 借用 scipy 矩阵
mat = SclCSR.from_scipy(scipy_mat)
print(mat.backend)  # Backend.CUSTOM

# 堆叠操作 (返回 Virtual 后端的 SclCSR)
stacked = vstack_csr([mat1, mat2])
print(stacked.backend)  # Backend.VIRTUAL
```

## 📝 更新日志

### v0.2.0 (当前)

- **新增**: 智能后端管理系统 (Custom/Virtual/Mapped)
- **新增**: 所有权跟踪 (Owned/Borrowed/View)
- **新增**: 引用链自动管理
- **新增**: `align_to_categories()` 函数
- **新增**: 统计函数 (`sum_rows`, `mean_cols`, `var_rows` 等)
- **移除**: `VirtualCSR`, `VirtualCSC` 类 (功能合并到 `SclCSR`, `SclCSC`)
- **改进**: 切片操作自动选择最优策略
- **改进**: 更完善的文档和类型提示

