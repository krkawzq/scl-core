# H5 Tools 转换方法完整指南

## 类型层次结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Storage Types                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HDF5 File        DequeSparse           OwnedSparse          CustomSparse    │
│  (on-disk)       (deque storage)       (vector storage)     (view only)      │
│                                                                              │
│     │                  │                      │                   │          │
│     │ load_*           │ materialize()        │ view()            │          │
│     ▼                  ▼                      ▼                   │          │
│  ┌──────────────────────────────────────────────────────────────┐ │          │
│  │                    OwnedSparse                                │◄┘          │
│  │              (canonical owned type)                           │            │
│  └──────────────────────────────────────────────────────────────┘            │
│                           │                                                   │
│                           │ view()                                            │
│                           ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────┐            │
│  │                    CustomSparse                               │            │
│  │                 (algorithm interface)                         │            │
│  └──────────────────────────────────────────────────────────────┘            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 转换方法速查表

| 源类型 | 目标类型 | 方法 | 复杂度 |
|-------|---------|------|-------|
| H5 | OwnedSparse | `load_sparse_full()` | O(nnz) |
| H5 | OwnedSparse (rows) | `load_sparse_rows()` | O(selected_nnz) |
| H5 | DequeSparse (masked) | `load_sparse_masked()` | O(nnz) |
| H5 | OwnedSparse+View | `load_with_view()` | O(nnz) |
| H5 | .bin files | `export_h5_to_bin()` | O(nnz), streaming |
| DequeSparse | OwnedSparse | `materialize()` | O(nnz) |
| DequeSparse | OwnedSparse+View | `materialize_with_view()` | O(nnz) |
| DequeSparse | .bin files | `export_to_bin()` | O(nnz), streaming |
| OwnedSparse | CustomSparse | `view()` | O(1) |
| OwnedSparse | .bin files | `export_owned_to_bin()` | O(nnz) |
| OwnedSparse | H5 | `save_sparse()` | O(nnz) |
| CustomSparse | OwnedSparse | `scl::io::to_owned()` | O(nnz) |
| CustomSparse | .bin files | `export_custom_to_bin()` | O(nnz) |
| CustomSparse | H5 | `save_custom_sparse()` | O(nnz) |
| .bin files | H5 | `import_bin_to_h5()` | O(nnz) |

## 使用场景

### 场景 1: 加载 H5 数据并处理

```cpp
// 方式 A: 分步操作
auto owned = scl::io::h5::load_sparse_full<Real, true>("data.h5", "/X");
auto view = owned.view();
algorithm(view);

// 方式 B: 一步到位
auto [owned, view] = scl::io::h5::load_with_view<Real, true>("data.h5", "/X");
algorithm(view);  // owned 必须保持活跃!
```

### 场景 2: 筛选行+列后处理

```cpp
std::vector<Index> row_mask = {0, 5, 10, 100};
std::vector<Index> col_mask = {1, 2, 3, 50, 51};

// DequeSparse 避免 chunk 边界额外拷贝
auto deque = scl::io::h5::load_sparse_masked<Real, true>(
    "data.h5", "/X", row_mask, col_mask
);

// 转换为 CustomSparse 用于算法
auto [owned, view] = deque.materialize_with_view();
algorithm(view);
```

### 场景 3: H5 → .bin 转换 (大文件流式处理)

```cpp
// 直接流式导出，不需要全部加载到内存
scl::io::h5::export_h5_to_bin<Real, true>(
    "huge_data.h5",     // 100GB+ 文件
    "/X",
    "/output/sparse/",
    1024 * 1024         // 1M 元素的 buffer
);
```

### 场景 4: 分区导出

```cpp
std::vector<Index> partition1 = /* rows 0-9999 */;
std::vector<Index> partition2 = /* rows 10000-19999 */;

scl::io::h5::export_h5_to_bin_rows<Real, true>(
    "data.h5", "/X", "/partition1/", 
    Array<const Index>(partition1.data(), partition1.size())
);

scl::io::h5::export_h5_to_bin_rows<Real, true>(
    "data.h5", "/X", "/partition2/",
    Array<const Index>(partition2.data(), partition2.size())
);
```

### 场景 5: .bin → H5 导入

```cpp
// 从二进制文件创建 H5
scl::io::h5::import_bin_to_h5<Real, true>(
    "/data/sparse/",        // 包含 data.bin, indices.bin, indptr.bin
    "output.h5",
    "/X",
    rows, cols,
    {10000},                // chunk size
    6                       // compression level
);
```

### 场景 6: Python 数据处理

```cpp
// 从 Python 传入的 CustomSparse (非拥有)
CustomSparse<Real, true> py_sparse(py_data, py_indices, py_indptr, rows, cols);

// 深拷贝到 C++ 管理的内存
OwnedSparse<Real, true> owned = scl::io::to_owned(py_sparse);

// 保存到 H5
scl::io::h5::save_sparse<OwnedSparse<Real, true>, true>(
    "output.h5", "/X", owned
);

// 或者导出到 .bin
scl::io::h5::export_owned_to_bin(owned, "/output/");
```

### 场景 7: 内存映射 → 各种格式

```cpp
// 从 .bin 文件内存映射
auto mapped = scl::io::mount_standard_layout<Real, true>("/data/", rows, cols);

// 转换为 OwnedSparse (深拷贝)
auto owned = mapped.materialize();

// 或者直接获取视图 (零拷贝，mapped 必须保持活跃)
auto view = mapped.as_view();

// 保存到 H5
scl::io::h5::save_sparse<decltype(owned), true>("output.h5", "/X", owned);
```

## DequeSparse 设计说明

**为什么需要 DequeSparse?**

当从 H5 读取带 mask 的数据时:
- 数据分布在多个 chunk 中
- 每个 chunk 只取部分元素
- 如果直接用 vector，每次追加都可能触发重新分配

DequeSparse 使用 `std::deque` 存储:
- 追加操作 O(1)
- 无需预知最终大小
- 避免 chunk 边界处的额外内存拷贝

**转换到 CustomSparse:**
```cpp
DequeSparse<Real, true> deque = load_sparse_masked(...);

// 必须先 materialize 到连续存储
auto owned = deque.materialize();
auto view = owned.view();

// 或者一步到位
auto [owned, view] = deque.materialize_with_view();
```

## 性能建议

1. **大文件导出**: 使用 `export_h5_to_bin()` 流式处理，避免内存峰值
2. **频繁读取**: 转换为 .bin 后使用内存映射
3. **筛选查询**: `load_sparse_masked()` 利用 Zone Map 跳过不相关 chunk
4. **批量处理**: 使用 `load_with_view()` 避免两次创建对象

## 文件格式

### .bin 目录结构
```
/output/
├── data.bin      # T[] - 非零值
├── indices.bin   # Index[] - 列索引 (CSR) 或行索引 (CSC)
├── indptr.bin    # Index[] - 行/列指针
└── meta.txt      # 元数据
```

### meta.txt 格式
```
rows=10000
cols=5000
nnz=1000000
is_csr=true
dtype=float32
```

## 结论

- **SparseLike 统一接口**: 所有类型都可直接用于算法
- **零拷贝优先**: 使用 `view()` / `as_view()` 避免不必要拷贝
- **流式处理**: 大文件使用 `export_h5_to_bin()` 
- **生命周期安全**: View 的生命周期必须短于 Owner

这是一套完整的稀疏矩阵 I/O 和转换工具! 🚀
