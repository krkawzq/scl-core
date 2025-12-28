# 快速参考卡片

## 常用命令

```bash
# ============================================================
# 基础工作流
# ============================================================

# 1. 完整流程（自动化）
./scripts/parallel_fix/quick_fix.sh

# 2. 手动步骤
make build 2>&1 | tee build.log
./scripts/parallel_fix/analyze.sh build.log -o tasks.json
./scripts/parallel_fix/run.sh tasks.json

# ============================================================
# 交互模式 🆕
# ============================================================

# 交互式修复第一个任务
./scripts/parallel_fix/run.sh -i tasks.json

# 交互 + hard模式
./scripts/parallel_fix/run.sh -i --difficulty hard tasks.json

# 一键交互式修复
./scripts/parallel_fix/quick_fix.sh --interactive

# ============================================================
# 难度控制
# ============================================================

# 查看当前模型配置
./scripts/parallel_fix/run.sh --show-config

# 全局难度覆盖
./scripts/parallel_fix/run.sh --difficulty hard tasks.json

# 预览将使用什么模型
./scripts/parallel_fix/run.sh --dry-run tasks.json

# ============================================================
# 监控和调试
# ============================================================

# 实时监控
./scripts/parallel_fix/run.sh --monitor tasks.json

# 单独启动监控
./scripts/parallel_fix/monitor.sh .parallel_fix_state

# 详细输出
./scripts/parallel_fix/run.sh --verbose tasks.json

# 查看任务状态
cat .parallel_fix_state/status

# 查看任务日志
tail -f .parallel_fix_state/logs/task_1.log
```

## 配置速查

```bash
# 配置文件位置
.parallel_fix.conf

# 关键配置项
MAX_PARALLEL=6              # 最大并行任务数

EASY_MODEL="claude-haiku-4-5-20251001"
EASY_THINK=false

NORMAL_MODEL="claude-sonnet-4-5-20250929"
NORMAL_THINK=false

HARD_MODEL="claude-opus-4-5-20251101"
HARD_THINK=true             # 启用深度思考
```

## 任务文件格式

### JSON 格式

```json
{
  "tasks": [
    {
      "id": "1",
      "type": "sparse_api",
      "files": ["src/kernel.hpp"],
      "description": "修复稀疏矩阵API",
      "difficulty": "normal",
      "context": "可选的额外上下文"
    }
  ]
}
```

### 文本格式

```
# 格式: ID|类型|文件|描述|难度
1|sparse_api|src/kernel.hpp|修复稀疏矩阵API|normal
2|syntax|src/parser.hpp|修复语法错误|hard
3|missing_header|src/util.hpp|添加头文件|easy
```

## 任务类型

| 类型 | 默认难度 | 说明 |
|------|---------|------|
| `missing_header` | easy | 缺少头文件 |
| `sparse_api` | normal | 稀疏矩阵API迁移 |
| `template_param` | normal | 模板参数错误 |
| `type_cast` | normal | 类型转换问题 |
| `api_binding` | normal | C API绑定 |
| `syntax` | hard | 严重语法错误 |
| `complex_refactor` | hard | 复杂重构 |
| `generic` | normal | 通用错误 |

## 选项速查

| 选项 | 短选项 | 说明 |
|------|--------|------|
| `--interactive` | `-i` | 交互模式 |
| `--difficulty LVL` | `-D LVL` | 设置难度 (easy/normal/hard) |
| `--dry-run` | `-d` | 只预览不执行 |
| `--monitor` | `-m` | 启用实时监控 |
| `--verbose` | `-v` | 详细输出 |
| `--state-dir DIR` | `-s DIR` | 指定状态目录 |
| `--show-config` | | 显示模型配置 |
| `--help` | `-h` | 显示帮助 |

## 场景选择

```
┌─────────────────────────────────────────────────────┐
│ 需要深入讨论和多轮调整？                                │
│   → ./run.sh -i tasks.json                         │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 多个简单任务需要快速批量处理？                          │
│   → ./run.sh tasks.json                            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 不确定任务复杂度，想先预览？                            │
│   → ./run.sh --dry-run tasks.json                  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 任务很复杂，需要最强模型？                              │
│   → ./run.sh -D hard tasks.json                    │
│   或 ./run.sh -i -D hard tasks.json (交互+最强)      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 想要一键完成所有步骤？                                  │
│   → ./quick_fix.sh                                 │
│   或 ./quick_fix.sh --interactive (交互式)          │
└─────────────────────────────────────────────────────┘
```

## 故障排查

```bash
# 查看最近错误
tail -100 .parallel_fix_state/logs/task_*.log

# 检查任务状态
cat .parallel_fix_state/status | grep failed

# 查看会话信息
cat .parallel_fix_state/session

# 清理状态重新开始
rm -rf .parallel_fix_state

# 验证配置
./scripts/parallel_fix/run.sh --show-config

# 测试交互模式
./scripts/parallel_fix/test_interactive.sh
```

## 进阶技巧

### 1. 分阶段处理

```bash
# 先交互处理最复杂的
./run.sh -i --difficulty hard complex_tasks.json

# 然后并行处理简单的
./run.sh simple_tasks.json
```

### 2. 自定义难度配置

编辑 `.parallel_fix.conf`：
```bash
# 如果你的任务都很简单，全部用haiku
DEFAULT_DIFFICULTY="easy"

# 或者normal任务也用思考模式
NORMAL_THINK=true
```

### 3. 手动任务管理

```bash
# 查看任务文件格式
head .parallel_fix_state/tasks

# 手动添加任务
echo "99|custom|my_file.hpp|自定义任务|hard" >> .parallel_fix_state/tasks
```

### 4. 批量重试失败任务

```bash
# 提取失败的任务
grep "failed" .parallel_fix_state/status | cut -d: -f1 > failed_ids.txt

# 创建重试任务文件
# (需要手动从原任务文件中筛选)
```

