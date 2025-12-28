#!/bin/bash
# ============================================================
# 实时监控工具
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
source "$SCRIPT_DIR/lib/state.sh"

STATE_DIR="${1:-.parallel_fix_state}"
INTERVAL="${2:-2}"

if [[ ! -d "$STATE_DIR" ]]; then
    log_error "状态目录不存在: $STATE_DIR"
    echo "用法: $0 [state_dir] [interval_seconds]"
    exit 1
fi

log_info "监控状态目录: $STATE_DIR (刷新间隔: ${INTERVAL}s)"
echo "按 Ctrl+C 退出"
echo ""

while true; do
    clear
    
    echo "┌────────────────────────────────────────────────────────────┐"
    echo "│              并行编译修复 - 实时监控                        │"
    echo "└────────────────────────────────────────────────────────────┘"
    
    print_summary "$STATE_DIR"
    
    # 显示最近日志
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  最近状态更新:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    tail -5 "$STATE_DIR/status" 2>/dev/null | while IFS=: read -r tid status ts msg; do
        local icon
        case "$status" in
            completed) icon="${GREEN}✓${NC}" ;;
            failed)    icon="${RED}✗${NC}" ;;
            running)   icon="${YELLOW}⋯${NC}" ;;
            *)         icon="○" ;;
        esac
        printf "  %b Task %s: %s (%s)\n" "$icon" "$tid" "$msg" "$ts"
    done
    
    echo ""
    echo "刷新时间: $(date '+%H:%M:%S') | 按 Ctrl+C 退出"
    
    # 检查是否全部完成
    eval "$(get_stats "$STATE_DIR")"
    if [[ $pending -eq 0 && $running -eq 0 ]]; then
        echo ""
        if [[ $failed -eq 0 ]]; then
            log_success "所有任务已完成！"
        else
            log_warn "任务完成，但有 $failed 个失败"
        fi
        break
    fi
    
    sleep "$INTERVAL"
done

