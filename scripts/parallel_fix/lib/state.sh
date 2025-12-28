#!/bin/bash
# ============================================================
# 状态管理库
# 使用简单的行式文件代替复杂的JSON
# ============================================================

# 依赖 common.sh
[[ -z "$NC" ]] && source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# 状态常量
STATUS_PENDING="pending"
STATUS_RUNNING="running"
STATUS_COMPLETED="completed"
STATUS_FAILED="failed"
STATUS_SKIPPED="skipped"

# 初始化状态目录
init_state() {
    local session_id=${1:-$(generate_id)}
    local state_dir="${STATE_DIR:-.parallel_fix_state}"
    
    ensure_dir "$state_dir"
    ensure_dir "$state_dir/logs"
    
    # 创建会话信息
    cat > "$state_dir/session" << EOF
SESSION_ID=$session_id
STARTED_AT=$(date -Is)
TOTAL_TASKS=0
EOF
    
    # 清空状态文件
    : > "$state_dir/status"
    
    echo "$state_dir"
}

# 添加任务
# 格式: task_id|task_type|target_files|description|difficulty
add_task() {
    local state_dir=$1
    local task_id=$2
    local task_type=$3
    local target_files=$4
    local description=$5
    local difficulty=${6:-""}  # 可选，为空时自动检测
    
    # 如果未指定难度，自动检测
    if [[ -z "$difficulty" ]]; then
        difficulty=$(auto_detect_difficulty "$task_type")
    fi
    
    # 追加到任务列表（包含难度）
    echo "${task_id}|${task_type}|${target_files}|${description}|${difficulty}" >> "$state_dir/tasks"
    
    # 设置初始状态
    update_status "$state_dir" "$task_id" "$STATUS_PENDING" ""
    
    # 更新总任务数
    local total
    total=$(wc -l < "$state_dir/tasks" | tr -d ' ')
    sed -i "s/^TOTAL_TASKS=.*/TOTAL_TASKS=$total/" "$state_dir/session"
}

# 获取任务难度
get_task_difficulty() {
    local state_dir=$1
    local task_id=$2
    
    local task_info
    task_info=$(get_task_info "$state_dir" "$task_id")
    echo "$task_info" | cut -d'|' -f5
}

# 更新任务状态
update_status() {
    local state_dir=$1
    local task_id=$2
    local status=$3
    local message=${4:-""}
    local timestamp
    timestamp=$(date -Is)
    
    # 使用文件锁保证原子性
    (
        flock -x 200
        echo "${task_id}:${status}:${timestamp}:${message}" >> "$state_dir/status"
    ) 200>"$state_dir/.lock"
}

# 获取任务当前状态
get_status() {
    local state_dir=$1
    local task_id=$2
    
    # 获取该任务的最后一条状态记录
    grep "^${task_id}:" "$state_dir/status" | tail -1 | cut -d: -f2
}

# 获取任务信息
get_task_info() {
    local state_dir=$1
    local task_id=$2
    
    grep "^${task_id}|" "$state_dir/tasks" | head -1
}

# 获取统计信息
get_stats() {
    local state_dir=$1
    
    # 获取每个任务的最新状态
    local -A latest_status
    while IFS=: read -r tid status _ _; do
        latest_status[$tid]=$status
    done < "$state_dir/status"
    
    local pending=0 running=0 completed=0 failed=0
    for status in "${latest_status[@]}"; do
        case "$status" in
            pending)   ((pending++)) ;;
            running)   ((running++)) ;;
            completed) ((completed++)) ;;
            failed)    ((failed++)) ;;
        esac
    done
    
    echo "pending=$pending running=$running completed=$completed failed=$failed"
}

# 打印状态摘要
print_summary() {
    local state_dir=$1
    
    # 读取会话信息
    source "$state_dir/session"
    
    # 获取统计
    eval "$(get_stats "$state_dir")"
    
    local total=$((pending + running + completed + failed))
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  会话: $SESSION_ID"
    echo "  开始: $STARTED_AT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  总计: %d  " "$total"
    printf "${GREEN}✓ %d${NC}  " "$completed"
    printf "${RED}✗ %d${NC}  " "$failed"
    printf "${YELLOW}⋯ %d${NC}  " "$running"
    printf "○ %d\n" "$pending"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 详细列表
    while IFS='|' read -r tid ttype tfiles tdesc tdifficulty; do
        local status
        status=$(get_status "$state_dir" "$tid")
        local icon diff_icon
        case "$status" in
            completed) icon="${GREEN}✓${NC}" ;;
            failed)    icon="${RED}✗${NC}" ;;
            running)   icon="${YELLOW}⋯${NC}" ;;
            *)         icon="○" ;;
        esac
        case "$tdifficulty" in
            easy)   diff_icon="${GREEN}E${NC}" ;;
            hard)   diff_icon="${RED}H${NC}" ;;
            *)      diff_icon="${YELLOW}N${NC}" ;;
        esac
        printf "  %b [%s|%b] %s: %s\n" "$icon" "$tid" "$diff_icon" "$ttype" "$tfiles"
    done < "$state_dir/tasks"
    echo ""
    echo "  难度: ${GREEN}E${NC}=Easy ${YELLOW}N${NC}=Normal ${RED}H${NC}=Hard"
    echo ""
}

# 生成完成协议命令（简化版）
generate_completion_cmd() {
    local state_dir=$1
    local task_id=$2
    local status=$3  # completed 或 failed
    local message=$4
    
    cat << EOF
echo "${task_id}:${status}:\$(date -Is):${message}" >> "${state_dir}/status"
EOF
}

