#!/bin/bash
# ============================================================
# 模板生成库
# ============================================================

[[ -z "$NC" ]] && source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TEMPLATE_DIR="$(dirname "${BASH_SOURCE[0]}")/../templates"

# 生成基础提示词头部
generate_header() {
    local task_id=$1
    local total_tasks=$2
    local task_type=$3
    local target_files=$4
    local difficulty=${5:-"normal"}
    
    local difficulty_desc
    case "$difficulty" in
        easy)
            difficulty_desc="简单 - 直接按照模式修复"
            ;;
        hard)
            difficulty_desc="困难 - 需要深入分析和仔细推理"
            ;;
        *)
            difficulty_desc="普通 - 按照标准流程修复"
            ;;
    esac
    
    cat << EOF
你是编译错误修复专家。你是第 ${task_id}/${total_tasks} 个并行修复agent。

## 任务信息
- 任务编号: ${task_id}
- 任务类型: ${task_type}
- 目标文件: ${target_files}
- 难度级别: ${difficulty_desc}

## 重要约束
- 只修改指定的文件
- 不要引入新的依赖
- 保持代码风格一致
- 如果不确定，宁可不修改

EOF
}

# 生成完成协议
generate_completion_protocol() {
    local state_dir=$1
    local task_id=$2
    
    cat << EOF
## 完成协议

修复完成后，必须执行以下命令更新状态：

成功时:
\`\`\`bash
echo "${task_id}:completed:\$(date -Is):修复完成" >> "${state_dir}/status"
echo "任务 ${task_id} 完成"
\`\`\`

失败时:
\`\`\`bash
echo "${task_id}:failed:\$(date -Is):失败原因描述" >> "${state_dir}/status"
echo "任务 ${task_id} 失败"
\`\`\`

现在开始执行修复任务！
EOF
}

# 稀疏矩阵API修复模板
template_sparse_api() {
    cat << 'EOF'
## 问题描述
该文件使用了旧的稀疏矩阵访问API，需要迁移到新API。

### 错误模式（需要替换）
```cpp
// 错误模式1: 直接下标访问indptr
const Index start = matrix.row_indices_unsafe()[i];
const Index end = matrix.row_indices_unsafe()[i + 1];

// 错误模式2: 直接下标访问indices和values
for (Index j = start; j < end; ++j) {
    Index col = matrix.col_indices_unsafe()[j];
    Real val = matrix.values()[j];
}
```

### 正确的新API
```cpp
// 新API: 使用行访问函数
auto row_vals = matrix.row_values_unsafe(i);
auto row_idxs = matrix.row_indices_unsafe(i);
Index row_len = matrix.row_length_unsafe(i);

for (Index j = 0; j < row_len; ++j) {
    Index col = row_idxs.ptr[j];
    Real val = row_vals.ptr[j];
}
```

## 修复规则
- CSR矩阵使用 row_* 方法
- CSC矩阵使用 col_* 方法
- 通用代码使用 primary_* 方法
- 模板代码中使用 if constexpr (IsCSR) 区分

## 搜索命令
```bash
grep -n "row_indices_unsafe()\[" TARGET_FILE
grep -n "col_indices_unsafe()\[" TARGET_FILE
grep -n "\.values()\[" TARGET_FILE
```

EOF
}

# 模板参数修复模板
template_template_param() {
    cat << 'EOF'
## 问题描述
模板参数数量或类型不匹配。

## 诊断步骤
1. 首先查看模板类的定义文件，确认正确的模板参数
2. 查看错误位置的代码
3. 修正模板参数

## 常见问题
- DualWorkspacePool<Real, Index> 可能应该是 DualWorkspacePool<Real>
- 先查看定义文件确认正确参数数量

## 修复命令
```bash
# 查看模板定义
grep -n "template.*class DualWorkspacePool" scl/threading/workspace.hpp
```

EOF
}

# 类型转换修复模板
template_type_cast() {
    cat << 'EOF'
## 问题描述
不安全的类型转换，特别是原子指针转换。

## 错误代码
```cpp
int64_t* raw_ptr = allocate<int64_t>(size);
auto* atomic_ptr = static_cast<std::atomic<int64_t>*>(raw_ptr);  // 错误！
```

## 修复方案
方案A - 直接分配原子类型:
```cpp
std::atomic<int64_t>* atomic_ptr = allocate<std::atomic<int64_t>>(size);
```

方案B - 使用reinterpret_cast（确保对齐安全）:
```cpp
static_assert(alignof(std::atomic<int64_t>) <= alignof(int64_t), 
              "Atomic alignment must not exceed base type alignment");
auto* atomic_ptr = reinterpret_cast<std::atomic<int64_t>*>(raw_ptr);
```

EOF
}

# 语法错误修复模板
template_syntax() {
    cat << 'EOF'
## 问题描述
语法错误，可能是括号不匹配或 if constexpr 使用问题。

## 诊断步骤
1. 查看错误行附近的代码（前后50行）
2. 检查括号匹配
3. 特别注意 if constexpr 语句
4. 查找未闭合的 { } ( ) < >

## 常见原因
- if constexpr 语句缺少括号
- 宏展开导致的括号不匹配
- 复制粘贴时遗漏了闭合括号

## 辅助命令
```bash
# 统计括号数量
grep -o "{" TARGET_FILE | wc -l
grep -o "}" TARGET_FILE | wc -l
```

EOF
}

# 缺少头文件模板
template_missing_header() {
    cat << 'EOF'
## 问题描述
缺少必要的头文件包含。

## 常见缺失
- `#include <cmath>` - 用于 std::sqrt, std::abs 等
- `#include <algorithm>` - 用于 std::sort, std::min 等
- `#include <atomic>` - 用于 std::atomic

## 修复步骤
1. 查看错误信息确定缺失的符号
2. 在文件顶部添加相应的头文件

EOF
}

# C API绑定修复模板
template_api_binding() {
    cat << 'EOF'
## 问题描述
C API绑定函数签名与内核函数不匹配。

## 诊断步骤
1. 查看函数的头文件定义
2. 确认期望的参数类型
3. 在调用处添加必要的类型转换或调整参数

## 常见问题
- CSR vs CSC 矩阵格式不匹配
- 参数顺序变化
- 新增或删除了参数

## 搜索命令
```bash
# 查看函数定义
grep -rn "FUNCTION_NAME" scl/kernel/
```

EOF
}

# 主生成函数
generate_prompt() {
    local task_id=$1
    local total_tasks=$2
    local task_type=$3
    local target_files=$4
    local state_dir=$5
    local extra_context=${6:-""}
    local difficulty=${7:-"normal"}
    
    # 头部（包含难度信息）
    generate_header "$task_id" "$total_tasks" "$task_type" "$target_files" "$difficulty"
    
    # 类型特定内容
    case "$task_type" in
        sparse_api)       template_sparse_api ;;
        template_param)   template_template_param ;;
        type_cast)        template_type_cast ;;
        syntax)           template_syntax ;;
        missing_header)   template_missing_header ;;
        api_binding)      template_api_binding ;;
        *)
            echo "## 通用修复任务"
            echo ""
            echo "请根据编译错误信息进行修复。"
            echo ""
            ;;
    esac
    
    # 额外上下文
    if [[ -n "$extra_context" ]]; then
        echo "## 额外信息"
        echo "$extra_context"
        echo ""
    fi
    
    # 根据难度添加额外指导
    if [[ "$difficulty" == "hard" ]]; then
        cat << 'EOF'
## 复杂任务指南

这是一个复杂任务，请：
1. 仔细阅读相关代码，理解整体架构
2. 分析错误的根本原因，而不仅仅是表面症状
3. 考虑修复可能带来的副作用
4. 如果涉及多个相互依赖的修改，规划好修改顺序
5. 在修改前先理解代码的意图

EOF
    fi
    
    # 完成协议
    generate_completion_protocol "$state_dir" "$task_id"
}

