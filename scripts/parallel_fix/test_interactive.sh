#!/bin/bash
# ============================================================
# 交互模式测试脚本
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "测试交互式模式..."
echo ""

# 创建一个简单的测试任务
cat > /tmp/test_task.json << 'EOF'
{
  "tasks": [
    {
      "id": "test1",
      "type": "missing_header",
      "files": ["test.hpp"],
      "description": "测试任务 - 添加头文件",
      "difficulty": "easy",
      "context": "这是一个测试任务，用于验证交互模式。"
    }
  ]
}
EOF

echo "已创建测试任务文件: /tmp/test_task.json"
echo ""
echo "运行命令:"
echo "  $SCRIPT_DIR/run.sh --interactive /tmp/test_task.json"
echo ""
echo "提示："
echo "- 系统会自动发送初始prompt"
echo "- 你可以继续输入来与Claude对话"
echo "- 输入 Ctrl+D 退出"
echo ""
read -p "按Enter继续..." -r

# 执行交互模式
"$SCRIPT_DIR/run.sh" --interactive /tmp/test_task.json

echo ""
echo "测试完成！"

