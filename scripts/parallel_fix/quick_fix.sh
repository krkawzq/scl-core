#!/bin/bash
# ============================================================
# 快速修复脚本 - 一键运行完整流程
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# 切换到项目根目录 (scripts/parallel_fix -> 项目根)
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

log_info "工作目录: $PROJECT_ROOT"

show_help() {
    cat << EOF
快速编译修复 - 一键运行完整流程

用法:
    $0 [选项]

选项:
    --skip-build    跳过初始编译（使用现有日志）
    --log FILE      使用指定的编译日志
    --dry-run       只分析不执行
    --interactive   交互模式 - 只处理第一个任务并保持对话
    -h, --help      显示帮助

流程:
    1. 编译项目并捕获错误
    2. 分析错误并生成任务
    3. 并行执行修复（或交互式处理第一个）
    4. 重新编译验证
EOF
}

SKIP_BUILD=false
LOG_FILE=""
DRY_RUN=false
INTERACTIVE_MODE=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --log)
            LOG_FILE="$2"
            SKIP_BUILD=true
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --interactive)
            INTERACTIVE_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主流程
main() {
    local start_time
    start_time=$(date +%s)
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║           快速编译修复 - 自动化工作流                        ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Step 1: 编译
    if ! $SKIP_BUILD; then
        log_step "Step 1/4: 编译项目..."
        LOG_FILE="build_errors_$(date +%Y%m%d_%H%M%S).log"
        
        if make build 2>&1 | tee "$LOG_FILE"; then
            log_success "编译成功，无需修复"
            exit 0
        fi
        
        log_info "编译失败，日志保存到: $LOG_FILE"
    else
        log_step "Step 1/4: 跳过编译，使用现有日志"
        
        if [[ -z "$LOG_FILE" ]]; then
            # 查找最近的日志文件
            LOG_FILE=$(ls -t build*.log 2>/dev/null | head -1 || echo "")
            if [[ -z "$LOG_FILE" ]]; then
                log_error "找不到编译日志文件"
                exit 1
            fi
        fi
        
        log_info "使用日志文件: $LOG_FILE"
    fi
    
    # Step 2: 分析
    log_step "Step 2/4: 分析编译错误..."
    local tasks_file="tasks_$(date +%Y%m%d_%H%M%S).json"
    
    "$SCRIPT_DIR/analyze.sh" "$LOG_FILE" -o "$tasks_file"
    
    if [[ ! -s "$tasks_file" ]]; then
        log_warn "未生成任何任务"
        exit 0
    fi
    
    log_info "任务文件: $tasks_file"
    
    # 显示任务预览
    echo ""
    echo "生成的任务:"
    jq -r '.tasks[] | "  [\(.id)] \(.type): \(.files | join(", ")) - \(.description)"' "$tasks_file"
    echo ""
    
    if $DRY_RUN; then
        log_warn "干运行模式，停止"
        exit 0
    fi
    
    # 确认执行
    if $INTERACTIVE_MODE; then
        log_info "交互模式：将只处理第一个任务"
        read -p "是否开始交互式修复? [Y/n] " -n 1 -r
    else
        read -p "是否开始并行修复? [Y/n] " -n 1 -r
    fi
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_info "已取消"
        exit 0
    fi
    
    # Step 3: 执行修复
    if $INTERACTIVE_MODE; then
        log_step "Step 3/4: 执行交互式修复..."
        "$SCRIPT_DIR/run.sh" --interactive "$tasks_file"
    else
        log_step "Step 3/4: 执行并行修复..."
        "$SCRIPT_DIR/run.sh" "$tasks_file"
    fi
    
    # Step 4: 验证
    log_step "Step 4/4: 重新编译验证..."
    local verify_log="build_verify_$(date +%Y%m%d_%H%M%S).log"
    
    if make build 2>&1 | tee "$verify_log"; then
        local end_time
        end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        echo ""
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║                      ✓ 修复成功！                           ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
        log_success "编译通过"
        log_info "总耗时: $(format_duration $duration)"
    else
        log_error "验证失败，仍有编译错误"
        log_info "请查看日志: $verify_log"
        
        # 分析剩余错误
        echo ""
        log_info "剩余错误分析:"
        "$SCRIPT_DIR/analyze.sh" "$verify_log" -f text
        
        exit 1
    fi
}

main

