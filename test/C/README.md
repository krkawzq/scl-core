# SCL Core v0.5 - C API Tests

C API 测试套件，用于验证 `scl/api/core/` 中的 C 接口。

## 目录结构

```
test/C/
├── CMakeLists.txt      # 测试构建配置
├── README.md           # 本文件
├── include/
│   └── test.hpp        # 测试框架和辅助函数
├── src/
│   ├── test_error.cpp  # 错误处理测试
│   └── test_sparse.cpp # 稀疏矩阵测试
└── build/              # 构建目录（生成）
```

## 构建和运行测试

### 1. 构建 SCL 库

首先需要构建主库：

```bash
cd /path/to/scl-core
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
```

### 2. 构建测试

```bash
cd test/C
cmake -B build
cmake --build build
```

### 3. 运行测试

运行所有测试：

```bash
cd build
ctest
```

或使用自定义目标：

```bash
cmake --build build --target run_tests
```

运行单个测试：

```bash
cd build
./test_error
./test_sparse
```

详细输出：

```bash
ctest --output-on-failure --verbose
```

## 编写新测试

### 基本结构

```cpp
#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

SCL_TEST_SUITE(my_feature)

SCL_TEST_CASE(test_basic_functionality) {
    // 创建测试矩阵
    auto handle = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    
    // 断言
    SCL_ASSERT_NOT_NULL(handle);
    SCL_ASSERT_EQ(scl_sparse_nnz(handle), 0);
    
    // 清理
    scl_sparse_destroy(handle);
}

SCL_TEST_CASE(test_edge_cases) {
    // 测试边界情况
    auto handle = scl_sparse_zeros(0, 0, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_NULL(handle);
    SCL_ASSERT_TRUE(scl_has_error());
    scl_clear_error();
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
```

### 可用断言

- `SCL_ASSERT_TRUE(cond)` - 条件为真
- `SCL_ASSERT_FALSE(cond)` - 条件为假
- `SCL_ASSERT_EQ(a, b)` - 相等
- `SCL_ASSERT_NE(a, b)` - 不相等
- `SCL_ASSERT_LT(a, b)` - 小于
- `SCL_ASSERT_LE(a, b)` - 小于等于
- `SCL_ASSERT_GT(a, b)` - 大于
- `SCL_ASSERT_GE(a, b)` - 大于等于
- `SCL_ASSERT_NULL(ptr)` - 指针为空
- `SCL_ASSERT_NOT_NULL(ptr)` - 指针非空
- `SCL_ASSERT_NEAR(a, b, tol)` - 浮点数近似相等

### RAII 句柄包装

使用 `SparseHandle` 自动管理生命周期：

```cpp
SCL_TEST_CASE(test_with_raii) {
    SparseHandle mat(scl_sparse_identity(5, SCL_REAL64, SCL_INDEX64));
    
    SCL_ASSERT_NOT_NULL(mat.get());
    SCL_ASSERT_EQ(scl_sparse_nnz(mat.get()), 5);
    
    // 自动调用 scl_sparse_destroy
}
```

或使用辅助函数：

```cpp
SCL_TEST_CASE(test_with_helper) {
    auto mat = make_test_csr_3x3();  // 返回 SparseHandle
    
    // 使用 mat.get() 获取原始句柄
    SCL_ASSERT_EQ(scl_sparse_nnz(mat.get()), 6);
}
```

### 错误处理测试

```cpp
SCL_TEST_CASE(test_error_handling) {
    // 触发错误
    auto rows = scl_sparse_rows(nullptr);
    
    // 检查错误状态
    SCL_ASSERT_EQ(rows, 0);
    SCL_ASSERT_TRUE(scl_has_error());
    SCL_ASSERT_NE(scl_get_last_error(), scl_error_success());
    
    // 清理错误状态
    scl_clear_error();
}
```

## 添加新测试文件

1. 在 `src/` 中创建 `test_<feature>.cpp`
2. 在 `CMakeLists.txt` 中注册：

```cmake
add_scl_test(<feature>  src/test_<feature>.cpp)
```

3. 更新 `ALL_TEST_TARGETS` 列表

## 测试覆盖范围

### 已实现

- ✅ `test_error.cpp` - 错误处理 C-API
  - 版本信息查询
  - 错误代码查询和验证
  - 线程本地错误状态
  - 错误信息格式化
  - 错误代码范围查询

- ✅ `test_sparse.cpp` - 稀疏矩阵 C-API
  - 创建函数（zeros, identity, from_coo）
  - 属性查询（rows, cols, nnz, density, etc.）
  - 生命周期管理（clone, destroy）
  - 转换操作（transpose）
  - 原位操作（scale）
  - 元素访问（get, exists）
  - 错误处理

### 待实现

- ⏳ `test_type.cpp` - 类型系统测试
- ⏳ `test_operations.cpp` - 矩阵运算测试
- ⏳ `test_export.cpp` - 数据导出测试
- ⏳ `test_slice.cpp` - 切片操作测试（一旦实现）

## 故障排查

### 库未找到

```
SCL library not found!
```

**解决方案**：确保先构建了主库

```bash
cd ../..
cmake -B build
cmake --build build
```

### 运行时库未找到

```
error while loading shared libraries: libscl.so: cannot open shared object file
```

**解决方案**：设置 `LD_LIBRARY_PATH`

```bash
export LD_LIBRARY_PATH=/path/to/scl-core/build:$LD_LIBRARY_PATH
```

或使用 RPATH（CMake 已自动配置）

### 测试失败

1. 检查错误消息和堆栈跟踪
2. 使用 `--verbose` 运行以获取详细输出
3. 单独运行失败的测试以隔离问题
4. 检查是否有未清理的错误状态

## 与旧测试框架的区别

### 从 tests_v0.4 迁移

主要变化：

1. **API 变更**：
   - 旧：`scl_sparse_create(mat.ptr(), ...)`
   - 新：`auto handle = scl_sparse_from_coo(...)`

2. **句柄管理**：
   - 旧：使用 `Sparse` 包装类
   - 新：使用 `SparseHandle` RAII 或原始句柄 + 手动 `destroy`

3. **类型枚举**：
   - 旧：隐式或通过模板参数
   - 新：显式传递 `SCL_REAL64`, `SCL_INDEX64`, `SCL_LAYOUT_CSR`

4. **错误处理**：
   - 旧：返回 `scl_error_t`
   - 新：函数返回句柄/值，错误通过 `scl_get_last_error()` 查询

5. **简化**：
   - 移除了复杂的 Oracle/BLAS 验证（保留基本断言）
   - 专注于 C-API 接口验证而非数值正确性

## 开发流程

1. 修改 C-API 代码
2. 重新构建库：`cmake --build ../../build`
3. 重新构建测试：`cmake --build build`
4. 运行测试：`ctest` 或 `./test_<name>`
5. 修复失败的测试并重复

## 性能考虑

- 测试应快速执行（< 100ms 每个）
- 使用小矩阵（≤ 1000x1000）
- 避免密集计算（留给性能基准测试）
- 专注于正确性，而非性能

## 持续集成

测试可以集成到 CI/CD 流程中：

```bash
# CI 脚本示例
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd test/C
cmake -B build
cmake --build build
cd build && ctest --output-on-failure
```

## 许可证

与 SCL Core 主项目相同。

