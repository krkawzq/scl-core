# SCL C API 测试系统

现代化的 C++ 测试框架，专为 SCL C API 设计，支持并行开发和 CI/CD 集成。

## ✅ 当前状态

### Core 模块（100% 完成）

| 模块 | 测试数 | 状态 | 文件 |
|------|--------|------|------|
| core.h | 41 | ✅ | test_core.cpp |
| dense.h | 39 | ✅ | test_dense.cpp |
| sparse.h | 51 | ✅ | test_sparse.cpp |
| unsafe.h | 27 | ✅ | test_unsafe.cpp |
| **总计** | **158** | **✅ 100%** | **4 files** |

### 工具库

- ✅ **RAII 守卫** (guard.hpp) - 自动管理资源生命周期
- ✅ **测试框架** (core.hpp) - 3000+ 行完整框架
- ✅ **数据生成** (data.hpp) - 随机矩阵生成（支持随机形状）
- ✅ **Eigen 参考** (oracle.hpp) - 参考实现和结果验证
- ✅ **精度比较** (precision.hpp) - 多种容差模式
- ✅ **BLAS 参考** (blas.hpp) - BLAS 操作封装

---

## 🚀 快速开始

### 运行测试

```bash
cd tests/c_api

# 运行单个模块
make test-core
make test-sparse

# 运行所有 core 模块
make test-core test-dense test-sparse test-unsafe

# 运行所有测试
make test-all
```

### 编写新测试

```cpp
#include "test.hpp"

SCL_TEST_BEGIN

SCL_TEST_SUITE(my_feature)

SCL_TEST_CASE(basic_test) {
    // 准备测试数据
    auto data = get_tiny_3x3();
    
    // 执行操作
    Sparse mat = make_sparse_csr(...);
    
    // 验证结果
    SCL_ASSERT_EQ(mat.rows(), 3);
    SCL_ASSERT_NEAR(mat.data()[0], 1.0, 1e-10);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()
```

### 独立编译

```bash
# 编译单个测试
make build-my_test

# 运行测试
./units/test_my_test

# 调试模式
./units/test_my_test --verbose
./units/test_my_test --filter "failing_test"
```

---

## 📚 文档

| 文档 | 用途 | 读者 |
|------|------|------|
| **[API_GUIDE.md](API_GUIDE.md)** | API 使用指南、常见陷阱、最佳实践 | **开发前必读** |
| **[TEST_GUIDE.md](TEST_GUIDE.md)** | 测试编写完整指南、模板、示例 | 编写测试时参考 |
| **[TASKS.md](TASKS.md)** | 任务分配清单、进度追踪 | 领取任务 |

---

## 🎨 测试框架特性

### 输出格式

```bash
# pytest 风格彩色输出（默认）
./test --human

# CI/CD 集成
./test --tap              # Test Anything Protocol
./test --json report.json # JSON 报告
./test --junit out.xml    # JUnit XML
./test --github           # GitHub Actions 注解
```

### 测试控制

```bash
# 过滤测试
./test --filter "sparse"          # 名称匹配
./test --tag "slow"                # 标签过滤
./test --suite "creation"          # 套件过滤

# 执行控制
./test --fail-fast                 # 首次失败时停止
./test --retry 3                   # 失败重试
./test --shuffle --seed 42         # 随机顺序
./test --timeout 5.0               # 超时控制

# 调试
./test --verbose                   # 详细输出
./test --list                      # 列出所有测试
./test --dry-run                   # 不执行，仅列出
```

### 断言宏

```cpp
// 基础断言
SCL_ASSERT_TRUE(condition);
SCL_ASSERT_FALSE(condition);
SCL_ASSERT_EQ(a, b);
SCL_ASSERT_NE(a, b);

// 数值比较
SCL_ASSERT_LT(a, b);
SCL_ASSERT_LE(a, b);
SCL_ASSERT_GT(a, b);
SCL_ASSERT_GE(a, b);
SCL_ASSERT_NEAR(a, b, tolerance);

// 指针检查
SCL_ASSERT_NULL(ptr);
SCL_ASSERT_NOT_NULL(ptr);

// 字符串
SCL_ASSERT_STR_EQ(s1, s2);
SCL_ASSERT_STR_CONTAINS(s, substr);

// 异常
SCL_ASSERT_THROWS(expr, exception_type);
SCL_ASSERT_NO_THROW(expr);
```

### 高级功能

```cpp
// 测试标记
SCL_TEST_CASE_SKIP(name, "reason") { /* ... */ }
SCL_TEST_CASE_XFAIL(name, "known bug") { /* ... */ }
SCL_TEST_CASE_TAGS(name, "slow", "integration") { /* ... */ }

// 重试机制（用于随机算法）
SCL_TEST_RETRY(name, 5) { /* 失败时重试 5 次 */ }

// 基准测试
SCL_TEST_BENCHMARK(name, 1000) { /* 运行 1000 次 */ }

// Fixtures
SCL_TEST_FIXTURE(MyFixture) {
    void setup() { /* 每个测试前 */ }
    void teardown() { /* 每个测试后 */ }
};

SCL_TEST_F(MyFixture, test_name) { /* 使用 fixture */ }
```

---

## 🛠️ 构建系统

### Makefile（推荐）

独立构建系统，支持并行开发：

```makefile
# 编译指定测试
make build-sparse

# 运行指定测试
make test-sparse

# 同时编译和运行
make test-sparse

# 清理
make clean

# 列出所有测试
make list
```

### CMake（集成）

```bash
# 在项目根目录
mkdir build && cd build
cmake ..
make

# 运行所有测试
ctest
```

---

## 📦 目录结构

```
tests/c_api/
├── src/                    # 测试源文件
│   ├── test_core.cpp      # Core API 测试
│   ├── test_dense.cpp     # Dense 矩阵测试
│   ├── test_sparse.cpp    # Sparse 矩阵测试
│   ├── test_unsafe.cpp    # Unsafe API 测试
│   └── ...                # 其他测试
│
├── include/                # 测试工具库
│   ├── test.hpp           # 主入口
│   ├── core.hpp           # 测试框架（2986 行）
│   ├── guard.hpp          # RAII 守卫
│   ├── oracle.hpp         # Eigen 参考
│   ├── data.hpp           # 数据生成
│   ├── precision.hpp      # 精度比较
│   └── blas.hpp           # BLAS 参考
│
├── units/                  # 编译后的测试可执行文件
│
├── API_GUIDE.md           # 📚 API 使用指南
├── TEST_GUIDE.md          # 📚 测试编写指南
├── TASKS.md               # 📋 任务清单
├── Makefile               # 独立构建系统
└── CMakeLists.txt         # CMake 配置
```

---

## 🎯 开发工作流

### 1. 领取任务

```bash
vim TASKS.md  # 选择未分配的模块，标记负责人
```

### 2. 编写测试

```bash
cd src/
cp test_core.cpp test_my_module.cpp  # 从示例开始
# 编辑文件...
```

### 3. 迭代开发

```bash
# 编译
make build-my_module

# 运行
make test-my_module

# 调试失败
./units/test_my_module --verbose
./units/test_my_module --filter "failing_test"
```

### 4. 验收提交

```bash
# 确保所有测试通过
make test-my_module
make test-my_module VERBOSE=1

# 提交
git add src/test_my_module.cpp
git commit -m "Add tests for <module>: X tests, 100% pass"
```

---

## 🐛 常见问题

### 测试编译失败

```bash
# 检查是否链接了正确的库
make build-my_test VERBOSE=1

# 检查头文件路径
ls include/
```

### 测试运行失败

```bash
# 详细输出
./units/test_my_test --verbose

# 单独运行失败的测试
./units/test_my_test --filter "failing_test"

# 检查错误信息
./units/test_my_test 2>&1 | less
```

### 内存问题

```bash
# 使用 Valgrind
valgrind --leak-check=full ./units/test_my_test

# 使用 AddressSanitizer
make build-my_test CXXFLAGS="-fsanitize=address -g"
```

---

## 📊 测试要求

### 每个函数必须测试

- ✅ **正常路径**：有效输入的预期行为
- ✅ **NULL 指针**：所有指针参数的 NULL 检查
- ✅ **无效维度**：负数、零、超大值
- ✅ **边界值**：空矩阵、单元素、最大索引
- ✅ **错误码**：验证正确的错误返回值

### 算法测试额外要求

- ✅ **随机测试**：使用 `SCL_TEST_RETRY` 多次运行
- ✅ **参考实现**：与 Eigen/BLAS 结果比较
- ✅ **精度验证**：使用适当的容差
- ✅ **Monte Carlo**：统计算法需要多次试验

---

## 🏆 项目统计

- **测试框架**：2986 行（core.hpp）
- **工具库**：1800+ 行（6 个头文件）
- **测试代码**：5000+ 行
- **文档**：3 个主要文档（API、TEST、TASKS）
- **Core 模块覆盖**：57 个函数，158 个测试，100% 通过

---

## 📞 获取帮助

- **API 使用问题**：查看 [API_GUIDE.md](API_GUIDE.md)
- **测试编写问题**：查看 [TEST_GUIDE.md](TEST_GUIDE.md)
- **任务分配**：查看 [TASKS.md](TASKS.md)
- **示例代码**：查看 `src/test_core.cpp`, `src/test_sparse.cpp`

---

**下一步**：开始 Kernel 模块测试（60+ 模块，1000+ 函数）

参考：
- `src/test_algebra_spmv.cpp` - Algebra 模块示例
- `src/test_comp_effect_size.cpp` - Statistics 模块示例

---

*最后更新：2025-12-30*  
*SCL Core Team*
