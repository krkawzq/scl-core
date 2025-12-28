# 快速开始

欢迎使用 SCL-Core 开发！本指南将帮助您设置开发环境并开始贡献。

## 前置要求

### 必需

- **C++17 编译器**:
  - GCC >= 9.0
  - Clang >= 10.0
  - MSVC >= 19.14 (Visual Studio 2017 15.7)

- **CMake** >= 3.15

- **Git**

### 可选

- **Python** >= 3.8 (用于 Python 绑定)
- **Google Test** (用于测试)
- **Doxygen** (用于文档)

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
```

### 2. 构建

```bash
mkdir build
cd build
cmake ..
cmake --build . -j$(nproc)
```

### 3. 运行测试

```bash
ctest --output-on-failure
```

## 开发环境

### Linux

```bash
# 安装依赖 (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential cmake git

# 克隆并构建
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### macOS

```bash
# 安装依赖
brew install cmake

# 克隆并构建
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Windows

```powershell
# 安装 Visual Studio 2019 或更高版本（带 C++ 支持）
# 从 https://cmake.org/download/ 安装 CMake

# 克隆并构建
git clone https://github.com/krkawzq/scl-core.git
cd scl-core
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## 构建配置

### 精度选择

```bash
# Float32 (默认)
cmake -DSCL_USE_FLOAT32=ON ..

# Float64
cmake -DSCL_USE_FLOAT64=ON ..

# Float16 (需要 GCC >= 12 或 Clang >= 15)
cmake -DSCL_USE_FLOAT16=ON ..
```

### 索引类型选择

```bash
# Int32 (默认)
cmake -DSCL_USE_INT32=ON ..

# Int64
cmake -DSCL_USE_INT64=ON ..

# Int16 (用于小矩阵)
cmake -DSCL_USE_INT16=ON ..
```

### SIMD 配置

```bash
# 自动检测 (默认)
cmake ..

# 强制 AVX2
cmake -DSCL_SIMD_TARGET=AVX2 ..

# 强制 AVX-512
cmake -DSCL_SIMD_TARGET=AVX512 ..

# 禁用 SIMD
cmake -DSCL_ENABLE_SIMD=OFF ..
```

### 构建类型

```bash
# Debug (默认)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release (优化)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Release with debug info
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

## 项目结构

```
scl-core/
├── scl/                  # 源代码
│   ├── core/             # 核心类型和工具
│   ├── threading/        # 并行处理
│   ├── kernel/           # 计算算子
│   ├── math/             # 统计函数
│   ├── mmap/             # 内存映射数组
│   └── io/               # I/O 工具
├── tests/                # 单元测试
├── benchmarks/           # 性能基准测试
├── docs/                 # 文档
├── examples/             # 示例代码
├── CMakeLists.txt        # 构建配置
└── README.md             # 项目概览
```

## 下一步

- [从源码构建](/zh/cpp/getting-started/building) - 详细构建说明
- [贡献指南](/zh/cpp/getting-started/contributing) - 代码标准和流程
- [测试指南](/zh/cpp/getting-started/testing) - 编写和运行测试
- [架构概览](/zh/cpp/architecture/) - 理解设计

## 获取帮助

- **问题**: 在 [GitHub Issues](https://github.com/krkawzq/scl-core/issues) 上报告错误
- **讨论**: 在 [GitHub Discussions](https://github.com/krkawzq/scl-core/discussions) 中提问
- **贡献**: 查看 [贡献指南](/zh/cpp/getting-started/contributing)

---

::: tip 第一次使用？
从 [贡献指南](/zh/cpp/getting-started/contributing) 开始，了解我们的代码标准和流程。
:::

