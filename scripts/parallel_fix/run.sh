#!/bin/bash
# ============================================================
# 并行编译修复 - 主控制器 v2.0
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载库
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/state.sh"
source "$SCRIPT_DIR/lib/templates.sh"

# 默认配置
TASKS_FILE=""
STATE_DIR=".parallel_fix_state"
MONITOR_MODE=false
DRY_RUN=false
VERBOSE=false
INTERACTIVE_MODE=false  # 交互式模式
GLOBAL_DIFFICULTY=""  # 全局难度覆盖

# 帮助信息
show_help() {
    cat << EOF
并行编译修复系统 v2.0

用法:
    $0 [选项] <tasks.json|tasks_dir>

选项:
    -m, --monitor         启用实时监控模式
    -d, --dry-run         只生成prompt，不执行
    -v, --verbose         详细输出
    -i, --interactive     交互式模式 (自动发送prompt后继续对话)
    -s, --state-dir DIR   状态目录 (默认: .parallel_fix_state)
    -D, --difficulty LVL  设置全局难度: easy, normal, hard
    --show-config         显示模型配置并退出
    -h, --help            显示帮助

难度级别:
    easy   - 简单任务 (缺少头文件、简单语法)
             使用快速模型，无深度思考
    normal - 常规任务 (API迁移、模板参数)
             使用平衡模型
    hard   - 复杂任务 (逻辑重构、跨文件依赖)
             使用最强模型，启用深度思考

示例:
    # 从任务文件运行
    $0 tasks.json

    # 交互式模式 - 自动发送第一个任务的prompt后继续对话
    $0 --interactive tasks.json

    # 强制所有任务使用hard模式
    $0 --difficulty hard tasks.json

    # 干运行模式（预览prompt和模型配置）
    $0 --dry-run tasks.json

    # 查看当前模型配置
    $0 --show-config

任务文件格式 (JSON):
    {
      "tasks": [
        {
          "id": "1",
          "type": "sparse_api",
          "files": ["scl/kernel/alignment.hpp"],
          "description": "修复稀疏矩阵API",
          "difficulty": "normal",
          "context": "可选的额外上下文"
        }
      ]
    }

或者使用简单文本格式 (每行一个任务):
    1|sparse_api|scl/kernel/alignment.hpp|修复稀疏矩阵API|normal
    2|template_param|scl/kernel/centrality.hpp|修复模板参数|hard
EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--monitor)
                MONITOR_MODE=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -i|--interactive)
                INTERACTIVE_MODE=true
                shift
                ;;
            -s|--state-dir)
                STATE_DIR="$2"
                shift 2
                ;;
            -D|--difficulty)
                GLOBAL_DIFFICULTY="$2"
                if [[ ! "$GLOBAL_DIFFICULTY" =~ ^(easy|normal|hard)$ ]]; then
                    log_error "无效的难度级别: $GLOBAL_DIFFICULTY (可选: easy, normal, hard)"
                    exit 1
                fi
                shift 2
                ;;
            --show-config)
                load_config
                print_difficulty_config
                exit 0
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
            *)
                TASKS_FILE="$1"
                shift
                ;;
        esac
    done
    
    if [[ -z "$TASKS_FILE" ]]; then
        log_error "请指定任务文件"
        show_help
        exit 1
    fi
}

# 解析JSON任务文件
parse_json_tasks() {
    local file=$1
    local state_dir=$2
    
    if ! command -v jq &> /dev/null; then
        log_error "需要安装 jq 来解析JSON文件"
        exit 1
    fi
    
    local count
    count=$(jq '.tasks | length' "$file")
    
    for ((i=0; i<count; i++)); do
        local task
        task=$(jq -r ".tasks[$i]" "$file")
        
        local id type files desc context difficulty
        id=$(echo "$task" | jq -r '.id')
        type=$(echo "$task" | jq -r '.type')
        files=$(echo "$task" | jq -r '.files | join(", ")')
        desc=$(echo "$task" | jq -r '.description // ""')
        context=$(echo "$task" | jq -r '.context // ""')
        difficulty=$(echo "$task" | jq -r '.difficulty // ""')
        
        # 全局难度覆盖
        if [[ -n "$GLOBAL_DIFFICULTY" ]]; then
            difficulty="$GLOBAL_DIFFICULTY"
        fi
        
        add_task "$state_dir" "$id" "$type" "$files" "$desc" "$difficulty"
        
        # 存储额外上下文
        if [[ -n "$context" ]]; then
            echo "$context" > "$state_dir/context_$id.txt"
        fi
    done
}

# 解析简单文本任务文件
# 格式: id|type|files|desc|difficulty (difficulty可选)
parse_text_tasks() {
    local file=$1
    local state_dir=$2
    
    while IFS='|' read -r id type files desc difficulty; do
        [[ -z "$id" || "$id" == "#"* ]] && continue
        
        # 全局难度覆盖
        if [[ -n "$GLOBAL_DIFFICULTY" ]]; then
            difficulty="$GLOBAL_DIFFICULTY"
        fi
        
        add_task "$state_dir" "$id" "$type" "$files" "$desc" "$difficulty"
    done < "$file"
}

# 加载任务
load_tasks() {
    local file=$1
    local state_dir=$2
    
    if [[ "$file" == *.json ]]; then
        parse_json_tasks "$file" "$state_dir"
    else
        parse_text_tasks "$file" "$state_dir"
    fi
}

# 运行单个任务
run_task() {
    local state_dir=$1
    local task_id=$2
    local total_tasks=$3
    
    local task_info
    task_info=$(get_task_info "$state_dir" "$task_id")
    
    IFS='|' read -r _ task_type target_files description difficulty <<< "$task_info"
    
    # 如果没有难度信息，使用默认值
    if [[ -z "$difficulty" ]]; then
        difficulty="$DEFAULT_DIFFICULTY"
    fi
    
    # 读取额外上下文
    local context=""
    if [[ -f "$state_dir/context_$task_id.txt" ]]; then
        context=$(cat "$state_dir/context_$task_id.txt")
    fi
    
    # 生成prompt（包含难度信息）
    local prompt
    prompt=$(generate_prompt "$task_id" "$total_tasks" "$task_type" "$target_files" "$state_dir" "$context" "$difficulty")
    
    # 构建Claude命令
    local claude_cmd
    claude_cmd=$(build_claude_cmd "$difficulty")
    
    if $DRY_RUN; then
        echo "========== Task $task_id =========="
        echo "难度: $difficulty"
        echo "命令: $claude_cmd"
        echo "模型: $(get_model_for_difficulty "$difficulty")"
        echo "思考: $(get_think_for_difficulty "$difficulty")"
        echo "--- Prompt ---"
        echo "$prompt"
        echo "=========================================="
        return 0
    fi
    
    # 更新状态为运行中
    local model_info
    model_info="$(get_model_for_difficulty "$difficulty")"
    update_status "$state_dir" "$task_id" "running" "使用 $model_info ($difficulty)"
    
    # 执行Claude
    local log_file="$state_dir/logs/task_$task_id.log"
    
    if $VERBOSE; then
        log_info "Task $task_id [$difficulty]: $description"
        log_info "  Model: $model_info, Think: $(get_think_for_difficulty "$difficulty")"
    fi
    
    # 记录命令信息到日志
    {
        echo "# Task: $task_id"
        echo "# Difficulty: $difficulty"
        echo "# Model: $model_info"
        echo "# Think: $(get_think_for_difficulty "$difficulty")"
        echo "# Command: $claude_cmd"
        echo "# Started: $(date -Is)"
        echo "---"
    } > "$log_file"
    
    # 交互式模式处理
    if $INTERACTIVE_MODE; then
        log_info "交互模式：自动发送初始prompt后进入对话..."
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "初始Prompt (将自动发送):"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "$prompt" | head -50
        if [[ $(echo "$prompt" | wc -l) -gt 50 ]]; then
            echo "... (共 $(echo "$prompt" | wc -l) 行)"
        fi
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        # 保存到临时文件
        local temp_file=$(mktemp)
        echo "$prompt" > "$temp_file"
        
        log_info "启动交互会话 (使用 $model_info)..."
        echo ""
        
        # 方法：先自动发送prompt，然后保持stdin开放让用户继续对话
        # 使用cat连接：先读取临时文件（自动发送），然后读取stdin（用户输入）
        if [[ -t 0 ]]; then
            # 有终端，可以直接交互
            (cat "$temp_file"; cat) | $claude_cmd 2>&1 | tee -a "$log_file"
        else
            # 没有终端，尝试连接到/dev/tty
            if [[ -e /dev/tty ]]; then
                (cat "$temp_file"; cat < /dev/tty) | $claude_cmd 2>&1 | tee -a "$log_file"
            else
                log_error "交互模式需要终端环境"
                echo "$prompt" | $claude_cmd 2>&1 | tee -a "$log_file"
            fi
        fi
        
        local exit_code=$?
        rm -f "$temp_file"
        
        # 更新状态
        if [[ $exit_code -eq 0 ]]; then
            update_status "$state_dir" "$task_id" "completed" "交互会话完成"
        else
            update_status "$state_dir" "$task_id" "failed" "交互会话异常退出"
        fi
        
        return $exit_code
    fi
    
    # 非交互模式：正常执行
    if echo "$prompt" | $claude_cmd 2>&1 | tee -a "$log_file"; then
        # 检查状态文件是否已被agent更新
        local current_status
        current_status=$(get_status "$state_dir" "$task_id")
        if [[ "$current_status" != "completed" && "$current_status" != "failed" ]]; then
            # Agent没有更新状态，标记为完成
            update_status "$state_dir" "$task_id" "completed" "执行完成"
        fi
    else
        local current_status
        current_status=$(get_status "$state_dir" "$task_id")
        if [[ "$current_status" != "failed" ]]; then
            update_status "$state_dir" "$task_id" "failed" "Claude执行失败"
        fi
    fi
}

# 监控进度
monitor_progress() {
    local state_dir=$1
    local interval=${2:-2}
    
    while true; do
        clear
        print_summary "$state_dir"
        
        # 检查是否全部完成
        eval "$(get_stats "$state_dir")"
        if [[ $pending -eq 0 && $running -eq 0 ]]; then
            break
        fi
        
        sleep "$interval"
    done
}

# 主函数
main() {
    # 先加载配置
    load_config
    
    parse_args "$@"
    
    log_step "初始化并行修复系统"
    
    # 显示难度配置
    if $VERBOSE || $DRY_RUN; then
        print_difficulty_config
    fi
    
    # 显示全局难度覆盖
    if [[ -n "$GLOBAL_DIFFICULTY" ]]; then
        log_warn "全局难度覆盖: $GLOBAL_DIFFICULTY"
    fi
    
    # 初始化状态
    local session_id
    session_id=$(generate_id)
    STATE_DIR=$(init_state "$session_id")
    log_success "状态目录: $STATE_DIR"
    
    # 加载任务
    log_step "加载任务文件: $TASKS_FILE"
    load_tasks "$TASKS_FILE" "$STATE_DIR"
    
    # 读取任务总数
    source "$STATE_DIR/session"
    log_success "已加载 $TOTAL_TASKS 个任务"
    
    if $DRY_RUN; then
        log_warn "干运行模式 - 只生成prompt"
        echo ""
    fi
    
    # 启动监控（如果启用）
    if $MONITOR_MODE && ! $DRY_RUN; then
        monitor_progress "$STATE_DIR" &
        MONITOR_PID=$!
        trap "kill $MONITOR_PID 2>/dev/null" EXIT
    fi
    
    # 交互模式处理
    if $INTERACTIVE_MODE; then
        log_warn "交互模式：只执行第一个任务"
        
        # 获取第一个任务
        local first_task
        first_task=$(head -1 "$STATE_DIR/tasks")
        local task_id
        task_id=$(echo "$first_task" | cut -d'|' -f1)
        
        log_step "执行任务 $task_id (交互模式)..."
        
        # 直接执行，不使用后台
        run_task "$STATE_DIR" "$task_id" "$TOTAL_TASKS"
        
        # 打印最终状态
        print_summary "$STATE_DIR"
        
        exit 0
    fi
    
    # 并行启动所有任务
    log_step "启动并行任务..."
    
    local -a pids=()
    while IFS='|' read -r task_id _ _ _; do
        run_task "$STATE_DIR" "$task_id" "$TOTAL_TASKS" &
        pids+=($!)
        log_info "启动任务 $task_id (PID: ${pids[-1]})"
    done < "$STATE_DIR/tasks"
    
    if $DRY_RUN; then
        wait
        log_success "干运行完成"
        exit 0
    fi
    
    # 等待所有任务完成
    log_step "等待任务完成..."
    wait "${pids[@]}" 2>/dev/null || true
    
    # 停止监控
    if [[ -n "${MONITOR_PID:-}" ]]; then
        kill "$MONITOR_PID" 2>/dev/null || true
    fi
    
    # 打印最终摘要
    print_summary "$STATE_DIR"
    
    # 检查是否有失败
    eval "$(get_stats "$STATE_DIR")"
    if [[ $failed -gt 0 ]]; then
        log_warn "有 $failed 个任务失败，请查看日志:"
        for log in "$STATE_DIR/logs"/*.log; do
            if grep -q "failed" "$log" 2>/dev/null; then
                echo "  - $log"
            fi
        done
        exit 1
    fi
    
    log_success "所有任务完成！"
    echo ""
    echo "下一步: 重新编译验证修复结果"
    echo "  make 2>&1 | tee build_after_fix.log"
}

main "$@"

