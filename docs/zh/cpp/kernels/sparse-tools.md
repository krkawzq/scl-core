# 稀疏工具

稀疏矩阵基础设施工具，用于转换、验证和清理。

## 概览

稀疏工具提供：

- **格式转换** - 导出为 CSR/CSC/COO 数组
- **验证** - 检查结构完整性
- **清理** - 移除零值，修剪小值
- **内存信息** - 查询内存使用
- **布局转换** - 转换为连续布局，调整大小

## 格式转换

### to_contiguous_arrays

导出为连续 CSR/CSC 数组：

```cpp
#include "scl/kernel/sparse.hpp"

Sparse<Real, true> matrix = /* ... */;

// 导出为连续数组
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// arrays.data, arrays.indices, arrays.indptr 已注册
// 到 scl::Registry 以进行自动内存管理

// 使用数组...
external_library(arrays.data, arrays.indices, arrays.indptr,
                 arrays.nnz, arrays.primary_dim);

// 清理
auto& reg = scl::get_registry();
reg.unregister_ptr(arrays.data);
reg.unregister_ptr(arrays.indices);
reg.unregister_ptr(arrays.indptr);
```

**返回：** `ContiguousArraysT<T>`

```cpp
template <typename T>
struct ContiguousArraysT {
    T* data;             // 注册的值数组
    Index* indices;      // 注册的索引数组
    Index* indptr;       // 注册的偏移数组
    Index nnz;
    Index primary_dim;
};
```

**用例：**
- 与外部库集成（SciPy、cuSPARSE）
- Python 绑定（零拷贝传输）
- 性能关键的内核（连续格式更快）

### to_coo_arrays

导出为 COO（坐标）格式：

```cpp
auto coo = scl::kernel::sparse::to_coo_arrays(matrix);

// coo.row_indices, coo.col_indices, coo.values 已注册

// 使用 COO 格式...
external_coo_library(coo.row_indices, coo.col_indices, coo.values,
                     coo.nnz, coo.rows, coo.cols);

// 清理
auto& reg = scl::get_registry();
reg.unregister_ptr(coo.row_indices);
reg.unregister_ptr(coo.col_indices);
reg.unregister_ptr(coo.values);
```

**返回：** `COOArraysT<T>`

```cpp
template <typename T>
struct COOArraysT {
    Index* row_indices;  // 注册的行索引
    Index* col_indices;  // 注册的列索引
    T* values;           // 注册的值
    Index nnz;
    Index rows;
    Index cols;
};
```

**用例：**
- 格式转换
- 与基于 COO 的库接口
- 按不同顺序排序

### from_contiguous_arrays

从连续数组创建稀疏矩阵：

```cpp
// 现有 CSR 数组
Real* data = /* ... */;
Index* indices = /* ... */;
Index* indptr = /* ... */;

// 包装为 Sparse（零拷贝）
auto matrix = scl::kernel::sparse::from_contiguous_arrays<Real, true>(
    data, indices, indptr, rows, cols, nnz,
    false  // take_ownership = false（仅包装）
);

// 或取得所有权（注册到 Registry）
auto matrix = scl::kernel::sparse::from_contiguous_arrays<Real, true>(
    data, indices, indptr, rows, cols, nnz,
    true  // take_ownership = true（注册）
);
```

**参数：**
- `take_ownership`: 如果为 true，将数组注册到 Registry

**用例：**
- 从外部源加载
- Python 绑定（包装 NumPy 数组）
- 零拷贝集成

## 验证

### validate

检查结构完整性：

```cpp
auto result = scl::kernel::sparse::validate(matrix);

if (result.is_valid) {
    std::cout << "矩阵有效\n";
} else {
    std::cout << "矩阵无效：\n";
    for (const auto& error : result.errors) {
        std::cout << "  - " << error << "\n";
    }
}
```

**返回：** `ValidationResult`

```cpp
struct ValidationResult {
    bool is_valid;
    std::vector<std::string> errors;
};
```

**检查：**
- 索引边界（所有索引在 [0, secondary_dim) 内）
- 排序的索引（每行/列内）
- NNZ 一致性（长度总和 == nnz）
- 无重复（可选）

**用例：**
- 调试矩阵构建
- 验证外部数据
- 前置条件检查

## 内存信息

### memory_info

查询内存使用：

```cpp
auto info = scl::kernel::sparse::memory_info(matrix);

std::cout << "数据字节:    " << info.data_bytes << "\n";
std::cout << "索引字节: " << info.indices_bytes << "\n";
std::cout << "元数据字节:" << info.metadata_bytes << "\n";
std::cout << "总字节:   " << info.total_bytes << "\n";
std::cout << "块数:   " << info.block_count << "\n";
std::cout << "是否连续: " << info.is_contiguous << "\n";
```

**返回：** `MemoryInfo`

```cpp
struct MemoryInfo {
    Size data_bytes;       // 值的字节数
    Size indices_bytes;    // 索引的字节数
    Size metadata_bytes;   // 指针/长度的字节数
    Size total_bytes;      // 总内存使用
    Index block_count;     // 内存块数量
    bool is_contiguous;    // 如果为单块布局则为 true
};
```

**用例：**
- 内存分析
- 优化决策
- 调试内存使用

## 清理操作

### eliminate_zeros

移除零值元素：

```cpp
// 移除精确零值
scl::kernel::sparse::eliminate_zeros(matrix);

// 移除近零值（在容差内）
scl::kernel::sparse::eliminate_zeros(matrix, 1e-10);
```

**参数：**
- `tolerance`: 移除 |x| <= tolerance 的值

**效果：**
- 移除零/近零元素
- 更新 nnz 和长度
- 保留矩阵结构（行/列不变）

**性能：**
- 并行处理
- 两遍算法（先计数，后复制）
- 高效内存复用

**用例：**
- 算术运算后的清理
- 减少内存使用
- 提高性能（处理更少元素）

### prune

移除小值元素：

```cpp
// 移除 |value| < threshold 的元素
scl::kernel::sparse::prune(matrix, threshold);

// 或设置为零同时保留结构
scl::kernel::sparse::prune(matrix, threshold, true);
```

**参数：**
- `threshold`: 移除/置零 |x| < threshold 的元素
- `keep_structure`: 如果为 true，设置为零；如果为 false，移除

**效果：**
- 移除或置零小元素
- 如果移除则更新 nnz
- 如果 keep_structure=true 则保留结构

**用例：**
- 稀疏化
- 降噪
- 内存优化

## 布局转换

### make_contiguous

转换为连续布局：

```cpp
// 检查是否已连续
auto info = scl::kernel::sparse::memory_info(matrix);
if (!info.is_contiguous) {
    // 转换为连续
    auto contiguous = scl::kernel::sparse::make_contiguous(matrix);
    
    // 使用连续矩阵以获得更好性能
    process_fast(contiguous);
}
```

**效果：**
- 创建具有连续存储的新矩阵
- 所有行/列在单个内存块中
- 更好的缓存局部性

**用例：**
- 性能优化
- 为外部库准备
- 顺序访问模式

### resize_secondary

调整次要维度大小：

```cpp
// 调整列（对于 CSR）或行（对于 CSC）
scl::kernel::sparse::resize_secondary(matrix, new_secondary_dim);
```

**参数：**
- `new_secondary_dim`: 次要维度的新大小

**效果：**
- 更新次要维度元数据
- 调试模式：如果缩小则断言无越界索引
- 不修改数据（仅元数据操作）

**用例：**
- 维度调整
- 子集操作
- 矩阵切片

## 性能考虑

### 并行化

大多数操作都是并行的：

```cpp
// to_coo_arrays: 并行偏移计算和复制
// eliminate_zeros: 并行计数和复制
// validate: 并行检查
```

### 内存效率

- **两遍算法**：先计数，分配一次
- **尽可能原地操作**：eliminate_zeros、prune
- **Registry 管理**：自动清理

### 缓存优化

- **顺序访问**：针对缓存行优化
- **预取**：用于可预测模式
- **块处理**：更好的局部性

## 最佳实践

### 1. 验证外部数据

```cpp
// 始终验证来自外部源的数据
auto matrix = load_from_file(filename);

auto result = scl::kernel::sparse::validate(matrix);
if (!result.is_valid) {
    std::cerr << "无效矩阵：\n";
    for (const auto& error : result.errors) {
        std::cerr << "  " << error << "\n";
    }
    return;
}
```

### 2. 为性能转换

```cpp
// 为性能关键部分转换为连续
auto info = scl::kernel::sparse::memory_info(matrix);
if (!info.is_contiguous) {
    auto contiguous = scl::kernel::sparse::make_contiguous(matrix);
    // 使用连续矩阵以获得更好性能
    process_intensive(contiguous);
}
```

### 3. 操作后清理

```cpp
// 可能产生零的算术运算后
matrix = matrix + other_matrix;
scl::kernel::sparse::eliminate_zeros(matrix);

// 阈值处理后
apply_threshold(matrix, threshold);
scl::kernel::sparse::prune(matrix, threshold);
```

### 4. 分析内存使用

```cpp
void print_matrix_info(const auto& matrix) {
    auto info = scl::kernel::sparse::memory_info(matrix);
    
    std::cout << "矩阵: " << matrix.rows() << " x " << matrix.cols() 
              << ", nnz = " << matrix.nnz() << "\n";
    std::cout << "内存: " << info.total_bytes / 1024.0 / 1024.0 << " MB\n";
    std::cout << "块数: " << info.block_count 
              << (info.is_contiguous ? " (连续)" : " (非连续)") 
              << "\n";
}
```

## 示例

### Python 集成

```cpp
// C++ 端：导出为连续数组
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// Python 绑定：零拷贝传输
py::capsule data_deleter(arrays.data, [](void* ptr) {
    scl::get_registry().unregister_ptr(ptr);
});
py::capsule indices_deleter(arrays.indices, [](void* ptr) {
    scl::get_registry().unregister_ptr(ptr);
});
py::capsule indptr_deleter(arrays.indptr, [](void* ptr) {
    scl::get_registry().unregister_ptr(ptr);
});

return py::make_tuple(
    py::array_t<Real>({arrays.nnz}, {sizeof(Real)}, 
                      arrays.data, data_deleter),
    py::array_t<Index>({arrays.nnz}, {sizeof(Index)}, 
                       arrays.indices, indices_deleter),
    py::array_t<Index>({arrays.primary_dim + 1}, {sizeof(Index)}, 
                       arrays.indptr, indptr_deleter)
);
```

### 外部库集成

```cpp
// 转换为外部库期望的格式
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// 调用外部库（例如 cuSPARSE、MKL）
external_sparse_mv(arrays.data, arrays.indices, arrays.indptr,
                   arrays.primary_dim, matrix.cols(), arrays.nnz,
                   x, y);

// 清理
auto& reg = scl::get_registry();
reg.unregister_ptr(arrays.data);
reg.unregister_ptr(arrays.indices);
reg.unregister_ptr(arrays.indptr);
```

---

::: tip Registry 管理
所有导出的数组都注册到 `scl::Registry`。完成后记得取消注册以避免内存泄漏。
:::

