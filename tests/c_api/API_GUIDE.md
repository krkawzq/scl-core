# SCL C API 测试开发指南

## 📖 文档目的

本文档为 SCL C API 的并行测试开发提供指导，重点说明：
1. **重要 API 的正确使用方法**（频繁使用的核心 API）
2. **容易出错的陷阱**（已在测试中发现的问题）
3. **测试命名和组织规范**（保证并行开发一致性）
4. **内存管理最佳实践**（避免泄漏和双重释放）

---

## 🎯 核心模块总览

| 模块 | 文件 | 函数数 | 测试数 | 状态 |
|------|------|--------|--------|------|
| **core.h** | test_core.cpp | 7 | 41 | ✅ 完成 |
| **dense.h** | test_dense.cpp | 13 | 39 | ✅ 完成 |
| **sparse.h** | test_sparse.cpp | 27 | 51 | ✅ 完成 |
| **unsafe.h** | test_unsafe.cpp | 10 | 27 | ✅ 完成 |
| **总计** | - | **57** | **158** | **100%** |

---

## 🔑 重要 API 使用指南

### 1. 错误处理 API（**必读**）

#### ⚠️ 关键要点
所有 C API 都使用 **thread-local** 存储错误状态：
```cpp
thread_local scl_error_t g_last_error_code;
thread_local std::array<char, 512> g_last_error_message;
```

#### 最佳实践

```cpp
// ✅ 正确：每个函数调用后立即检查
scl_error_t err = scl_sparse_create(...);
if (err != SCL_OK) {
    const char* msg = scl_get_last_error();
    // 处理错误
}

// ❌ 错误：依赖旧的错误状态
scl_sparse_create(...);  // 可能失败
// ... 做其他事情 ...
const char* msg = scl_get_last_error();  // 可能是过时的错误
```

#### 测试中的错误清理

```cpp
// 测试框架会在每个测试开始时自动调用 scl_clear_error()
// 这防止了测试间的状态污染

SCL_TEST_CASE(my_test) {
    // 这里已经清理了错误状态，可以安全测试
    scl_error_t err = scl_sparse_create(...);
    SCL_ASSERT_EQ(err, SCL_OK);
}
```

#### 已知陷阱

| 陷阱 | 原因 | 解决方案 |
|------|------|----------|
| 测试间状态污染 | thread_local 不会自动清理 | 测试框架在每个测试前调用 `scl_clear_error()` |
| 错误信息丢失 | 后续调用覆盖错误 | 立即检查并保存错误信息 |
| 多线程混淆 | thread_local 在每个线程独立 | 确保在同一线程检查错误 |

---

### 2. Sparse Matrix API

#### `scl_sparse_create` - 创建稀疏矩阵

```cpp
scl_error_t scl_sparse_create(
    scl_sparse_t* out,           // [out] 输出句柄
    scl_index_t rows,            // 行数
    scl_index_t cols,            // 列数
    scl_index_t nnz,             // 非零元素数
    const scl_index_t* indptr,   // 行/列指针数组
    const scl_index_t* indices,  // 列/行索引数组
    const scl_real_t* data,      // 数据数组
    scl_bool_t is_csr            // SCL_TRUE=CSR, SCL_FALSE=CSC
);
```

#### ⚠️ 关键要点

1. **数据所有权**：`create` 会**复制数据**，原始数组可以安全释放
2. **格式标志**：`is_csr` 决定 CSR/CSC 格式，影响所有后续操作
3. **索引约定**：`indptr` 大小为 `primary_dim + 1`（CSR: rows+1, CSC: cols+1）

#### 测试示例

```cpp
SCL_TEST_CASE(create_csr_basic) {
    std::vector<scl_index_t> indptr = {0, 2, 3, 6};  // 3+1 元素
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    Sparse mat = make_sparse_csr(3, 3, 6, 
        indptr.data(), indices.data(), data.data());
    
    // 数据已复制，原始数组可以销毁
    // mat 会在析构时自动清理
}
```

---

#### `scl_sparse_wrap` - 零拷贝包装

```cpp
scl_error_t scl_sparse_wrap(
    scl_sparse_t* out,
    scl_index_t rows, scl_index_t cols, scl_index_t nnz,
    scl_index_t* indptr,    // 非 const
    scl_index_t* indices,   // 非 const
    scl_real_t* data,       // 非 const
    scl_bool_t is_csr
);
```

#### ⚠️ 关键要点

1. **零拷贝**：不复制数据，直接使用提供的指针
2. **生命周期**：调用者**必须保证**指针在矩阵生命周期内有效
3. **不可变性**：虽然指针非 const，但**不应修改**（UB）

#### 已知陷阱

```cpp
// ❌ 错误：临时数组生命周期结束
Sparse bad_example() {
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0};
    Sparse mat = wrap_sparse_csr(1, 3, 3, ...data.data()...);
    return mat;  // 悬空指针！data 已销毁
}

// ✅ 正确：确保数据生命周期
class MatrixHolder {
    std::vector<scl_real_t> data_;
    Sparse mat_;
public:
    MatrixHolder() {
        data_ = {1.0, 2.0, 3.0};
        mat_ = wrap_sparse_csr(...data_.data()...);
    }
    // data_ 和 mat_ 一起销毁，安全
};
```

---

#### `scl_sparse_wrap_and_own` - 接管所有权

```cpp
scl_error_t scl_sparse_wrap_and_own(
    scl_sparse_t* out,
    scl_index_t rows, scl_index_t cols, scl_index_t nnz,
    scl_index_t* indptr,
    scl_index_t* indices,
    scl_real_t* data,
    scl_bool_t is_csr
);
```

#### ⚠️ 关键要点

1. **接管所有权**：矩阵销毁时会**调用 `free()`**
2. **内存来源**：指针**必须来自 `malloc`/`new`**，不能是栈内存或 `vector::data()`
3. **单一所有权**：一旦传递，调用者**不应再访问**这些指针

#### 🚨 常见错误（已在测试中发现）

```cpp
// ❌ 错误 1：使用 vector 管理的内存
std::vector<scl_real_t> data = {1.0, 2.0, 3.0};
scl_sparse_wrap_and_own(..., data.data(), ...);
// 💥 双重释放：1. vector 析构释放  2. sparse 析构调用 free()

// ❌ 错误 2：使用栈内存
scl_real_t data[] = {1.0, 2.0, 3.0};
scl_sparse_wrap_and_own(..., data, ...);
// 💥 free() 非堆内存

// ✅ 正确：使用堆内存
scl_real_t* data = (scl_real_t*)malloc(3 * sizeof(scl_real_t));
data[0] = 1.0; data[1] = 2.0; data[2] = 3.0;
scl_sparse_wrap_and_own(..., data, ...);
// sparse 销毁时会正确 free(data)
```

#### 测试最佳实践

```cpp
SCL_TEST_CASE(wrap_and_own_correct) {
    // 分配堆内存
    scl_index_t* indptr = (scl_index_t*)malloc(4 * sizeof(scl_index_t));
    scl_index_t* indices = (scl_index_t*)malloc(6 * sizeof(scl_index_t));
    scl_real_t* data = (scl_real_t*)malloc(6 * sizeof(scl_real_t));
    
    // 填充数据
    indptr[0] = 0; indptr[1] = 2; indptr[2] = 3; indptr[3] = 6;
    // ... 填充 indices 和 data ...
    
    Sparse mat;
    scl_error_t err = scl_sparse_wrap_and_own(
        mat.ptr(), 3, 3, 6,
        indptr, indices, data,
        SCL_TRUE
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // ✅ 不需要手动 free - mat 析构时会自动清理
}
```

---

### 3. Dense Matrix API

#### `scl_dense_wrap` - 创建视图

```cpp
scl_error_t scl_dense_wrap(
    scl_dense_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* data  // 行优先存储
);
```

#### ⚠️ 关键要点

1. **纯视图**：`DenseView` **永远不拥有数据**
2. **行优先**：`data[i * cols + j]` = 元素 `(i, j)`
3. **调用者责任**：必须保证 `data` 指针在视图生命周期内有效

#### 内存模型

```
DenseView 内存模型：
┌─────────────────────────────────────┐
│  DenseView (handle)                 │
│    - rows, cols, stride             │
│    - data pointer → [外部内存]      │
└─────────────────────────────────────┘
         |
         └──> [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  ← 调用者拥有
```

#### 测试示例

```cpp
SCL_TEST_CASE(dense_view_lifetime) {
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0};
    
    {
        Dense mat = wrap_dense(2, 2, data.data());
        
        // 视图有效期间可以访问
        scl_real_t val;
        scl_dense_get(mat, 0, 0, &val);
        SCL_ASSERT_NEAR(val, 1.0, 1e-10);
        
    }  // mat 销毁（只释放 handle，不释放 data）
    
    // data 仍然有效
    SCL_ASSERT_NEAR(data[0], 1.0, 1e-10);
}
```

---

### 4. Unsafe API（高级）

#### ⚠️ 警告：仅供专家使用

Unsafe API 提供零开销的直接内存访问，但：
1. **ABI 不稳定**：结构布局可能在任何版本变更
2. **无边界检查**（release 模式）：越界访问 = UB
3. **无所有权管理**：调用者负责指针生命周期
4. **需要深入理解**：内存模型和内部实现

#### `scl_sparse_unsafe_get_row` - 直接行访问

```cpp
scl_error_t scl_sparse_unsafe_get_row(
    scl_sparse_t matrix,  // 必须是 CSR
    scl_index_t row,
    scl_real_t** data,
    scl_index_t** indices,
    scl_index_t* length
);
```

#### 使用场景

```cpp
// ✅ 适用：热路径中的批量访问
void process_all_rows(scl_sparse_t csr_matrix, scl_index_t rows) {
    for (scl_index_t i = 0; i < rows; ++i) {
        scl_real_t* data;
        scl_index_t* indices;
        scl_index_t length;
        
        scl_sparse_unsafe_get_row(csr_matrix, i, &data, &indices, &length);
        
        // 零开销访问
        for (scl_index_t j = 0; j < length; ++j) {
            // 处理 data[j], indices[j]
        }
    }
}
```

#### 已知行为

1. **Debug 模式**：有边界检查，越界会抛出异常
2. **Release 模式**：无边界检查，越界 = UB
3. **指针有效性**：指针在矩阵销毁后失效
4. **CSR/CSC 要求**：`get_row` 仅适用于 CSR，`get_col` 仅适用于 CSC

---

## 🛠️ 测试工具库

### RAII 守卫 (`guard.hpp`)

```cpp
// Sparse 守卫
Sparse mat = make_sparse_csr(3, 3, 6, ...);  // 自动管理生命周期
// 自动调用 scl_sparse_destroy

// Dense 守卫
Dense mat = wrap_dense(2, 3, data.data());
// 自动调用 scl_dense_destroy
```

### 工厂函数

```cpp
// CSR 矩阵
Sparse csr = make_sparse_csr(rows, cols, nnz, indptr, indices, data);

// CSC 矩阵
Sparse csc = make_sparse_csc(rows, cols, nnz, indptr, indices, data);

// 零拷贝包装
Sparse view = wrap_sparse_csr(rows, cols, nnz, indptr, indices, data);

// 密集矩阵视图
Dense dense = wrap_dense(rows, cols, data);
```

### 测试数据生成 (`data.hpp`)

```cpp
// 随机稀疏矩阵
auto mat = random_sparse_csr(100, 50, 0.1);  // 100x50, 10% 密度

// 随机形状
auto mat = random_sparse_csr(
    {10, 100},    // 行数范围
    {10, 50},     // 列数范围
    0.05          // 密度
);

// 随机密集矩阵
auto data = random_dense(20, 30);
```

### Eigen 参考实现 (`oracle.hpp`)

```cpp
// 转换为 Eigen
EigenCSR eigen_mat = to_eigen_csr(mat);

// Eigen 操作
EigenCSR transposed = eigen_mat.transpose();

// 转回 SCL
Sparse result = from_eigen_csr(transposed);

// 比较矩阵
bool equal = matrices_equal(eigen_mat1, eigen_mat2, 1e-10);
```

### 精度比较 (`precision.hpp`)

```cpp
// 标准容差
SCL_ASSERT_NEAR(a, b, Tolerance::normal());  // 相对 1e-9

// 宽松容差
SCL_ASSERT_NEAR(a, b, Tolerance::loose());   // 相对 1e-6

// 统计容差
SCL_ASSERT_NEAR(a, b, Tolerance::statistical());  // 5*std
```

---

## 📐 测试命名和组织规范

### 文件命名

```
tests/c_api/src/test_<module>.cpp
```

| 模块类型 | 命名示例 |
|---------|---------|
| Core API | `test_core.cpp`, `test_sparse.cpp`, `test_dense.cpp` |
| Unsafe API | `test_unsafe.cpp` |
| Kernels | `test_algebra_spmv.cpp`, `test_comp_effect_size.cpp` |
| Tools | `test_guards.cpp`, `test_tools.cpp` |

### 测试套件命名

```cpp
// 按功能分组
SCL_TEST_SUITE(creation)      // 创建函数
SCL_TEST_SUITE(properties)    // 属性查询
SCL_TEST_SUITE(operations)    // 操作函数
SCL_TEST_SUITE(error_handling) // 错误处理
SCL_TEST_SUITE(edge_cases)    // 边界情况
```

### 测试用例命名

格式：`功能_场景`，使用小写+下划线

```cpp
// ✅ 好的命名
SCL_TEST_CASE(create_csr_basic)
SCL_TEST_CASE(create_null_output)
SCL_TEST_CASE(transpose_rectangular_matrix)
SCL_TEST_CASE(get_row_out_of_bounds)

// ❌ 不好的命名
SCL_TEST_CASE(test1)
SCL_TEST_CASE(CreateCSR)
SCL_TEST_CASE(sparse_matrix_creation_with_null_output_parameter)
```

### 测试组织结构

```cpp
SCL_TEST_BEGIN

// =============================================================================
// 1. Creation Functions
// =============================================================================

SCL_TEST_SUITE(creation)

SCL_TEST_CASE(create_csr_basic) { /* ... */ }
SCL_TEST_CASE(create_csc_basic) { /* ... */ }
SCL_TEST_CASE(create_null_output) { /* ... */ }
// ... 更多 creation 测试 ...

SCL_TEST_SUITE_END

// =============================================================================
// 2. Property Queries
// =============================================================================

SCL_TEST_SUITE(properties)

SCL_TEST_CASE(query_rows) { /* ... */ }
SCL_TEST_CASE(query_cols) { /* ... */ }
// ... 更多 properties 测试 ...

SCL_TEST_SUITE_END

// ... 更多套件 ...

SCL_TEST_END

SCL_TEST_MAIN()
```

---

## 🧪 测试要求清单

### 每个函数必须测试

- [ ] **正常路径**：有效输入的预期行为
- [ ] **NULL 指针**：所有指针参数的 NULL 检查
- [ ] **无效维度**：负数、零、超大值
- [ ] **边界值**：空矩阵、单元素、最大索引
- [ ] **错误码**：验证正确的错误返回值

### 算法函数额外要求

- [ ] **随机测试**：使用 `SCL_TEST_RETRY` 多次运行
- [ ] **参考实现**：与 Eigen/BLAS 结果比较
- [ ] **精度验证**：使用适当的容差
- [ ] **Monte Carlo**：统计算法需要多次试验
- [ ] **性能标注**：标记慢速测试 `[slow]`

### 示例测试模板

```cpp
SCL_TEST_CASE(function_normal) {
    // 1. 准备输入
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    // 2. 调用函数
    Sparse result;
    scl_error_t err = scl_sparse_transpose(mat, result.ptr());
    
    // 3. 验证结果
    SCL_ASSERT_EQ(err, SCL_OK);
    
    scl_index_t rows, cols;
    scl_sparse_rows(result, &rows);
    scl_sparse_cols(result, &cols);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 3);
    
    // 4. 可选：与参考实现比较
    EigenCSR eigen_result = to_eigen_csr(result);
    EigenCSR expected = to_eigen_csr(mat).transpose();
    SCL_ASSERT_TRUE(matrices_equal(eigen_result, expected));
}

SCL_TEST_CASE(function_null_handle) {
    Sparse result;
    scl_error_t err = scl_sparse_transpose(nullptr, result.ptr());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
    result.release();
}

SCL_TEST_CASE(function_null_output) {
    auto [indptr, indices, data] = tiny_3x3();
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    scl_error_t err = scl_sparse_transpose(mat, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}
```

---

## 🚀 并行开发流程

### 1. 领取任务

从 `TASKS.md` 选择未分配的模块：

```bash
cd tests/c_api
vim TASKS.md  # 标记负责人和状态
```

### 2. 创建测试文件

```bash
cd tests/c_api/src
cp test_template.cpp test_my_module.cpp  # 或参考现有测试
```

### 3. 编译和运行

```bash
cd tests/c_api

# 独立编译
make build-my_module

# 运行测试
make test-my_module

# 调试失败
./units/test_my_module --verbose
./units/test_my_module --filter "failing_test"
```

### 4. 迭代开发

```bash
# 编辑测试
vim src/test_my_module.cpp

# 重新测试
make test-my_module

# 如果发现源码 bug，修复后重新测试
```

### 5. 验收提交

```bash
# 确保所有测试通过
make test-my_module
make test-my_module VERBOSE=1

# 检查覆盖率（可选）
# ... 覆盖率工具 ...

# 提交
git add src/test_my_module.cpp
git commit -m "Add tests for <module>: X tests, 100% pass"
```

---

## 📚 参考文档

| 文档 | 用途 |
|------|------|
| `TEST_GUIDE.md` | 完整测试编写指南 |
| `TASKS.md` | 任务分配清单 |
| `API_GUIDE.md` | 本文档 - API 使用指南 |
| `README_FINAL.md` | 测试系统总体说明 |

---

## 💡 提示和技巧

### 内存调试

```bash
# 使用 Valgrind 检查内存泄漏
valgrind --leak-check=full ./units/test_my_module

# 使用 AddressSanitizer
g++ -fsanitize=address -g src/test_my_module.cpp ...
```

### 性能分析

```cpp
// 使用框架的基准测试功能
SCL_TEST_BENCHMARK(matrix_multiply, 1000) {
    // 重复执行 1000 次
    scl_algebra_spmv(...);
}
```

### 调试技巧

```cpp
// 打印中间值
SCL_TEST_CASE(debug_example) {
    Sparse mat = ...;
    
    // 导出数据检查
    scl_sparse_raw_t raw;
    scl_sparse_unsafe_get_raw(mat, &raw);
    
    std::cout << "Rows: " << raw.rows << std::endl;
    std::cout << "NNZ: " << raw.nnz << std::endl;
    
    // ... 继续测试 ...
}
```

---

## ✅ 核心模块测试总结

| 模块 | 函数数 | 测试数 | 覆盖率 | 状态 |
|------|--------|--------|--------|------|
| core.h | 7 | 41 | 100% | ✅ |
| dense.h | 13 | 39 | 100% | ✅ |
| sparse.h | 27 | 51 | 100% | ✅ |
| unsafe.h | 10 | 27 | 100% | ✅ |
| **总计** | **57** | **158** | **100%** | **✅** |

**下一步**：开始 Kernel 模块测试（Algebra, Statistics, Neighbors, etc.）

---

*最后更新：2025-12-30*
*作者：SCL Core Team*

