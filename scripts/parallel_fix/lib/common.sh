#!/bin/bash
# ============================================================
# 通用函数库
# ============================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 日志函数
log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_step()    { echo -e "${CYAN}[STEP]${NC} ${BOLD}$*${NC}"; }

# 进度条
show_progress() {
    local current=$1
    local total=$2
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "\r["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%% (%d/%d)" "$percent" "$current" "$total"
}

# 时间格式化
format_duration() {
    local seconds=$1
    if ((seconds < 60)); then
        echo "${seconds}s"
    elif ((seconds < 3600)); then
        echo "$((seconds / 60))m $((seconds % 60))s"
    else
        echo "$((seconds / 3600))h $((seconds % 3600 / 60))m"
    fi
}

# 获取脚本目录
get_script_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

# 获取项目根目录
get_project_root() {
    local script_dir
    script_dir="$(get_script_dir)"
    cd "$script_dir/../../.." && pwd
}

# 确保目录存在
ensure_dir() {
    local dir=$1
    [[ -d "$dir" ]] || mkdir -p "$dir"
}

# 检查命令是否存在
require_cmd() {
    local cmd=$1
    if ! command -v "$cmd" &> /dev/null; then
        log_error "Required command not found: $cmd"
        return 1
    fi
}

# 生成唯一ID
generate_id() {
    date +%Y%m%d_%H%M%S_$$
}

# 读取配置文件
load_config() {
    local config_file="${1:-.parallel_fix.conf}"
    
    # 默认配置
    MAX_PARALLEL=${MAX_PARALLEL:-6}
    VERIFY_AFTER_TASK=${VERIFY_AFTER_TASK:-false}
    RETRY_COUNT=${RETRY_COUNT:-1}
    STATE_DIR=${STATE_DIR:-".parallel_fix_state"}
    DEFAULT_DIFFICULTY=${DEFAULT_DIFFICULTY:-"normal"}
    
    # 默认难度模型配置
    EASY_MODEL=${EASY_MODEL:-"claude-haiku-4-5-20251001"}
    EASY_THINK=${EASY_THINK:-false}
    
    NORMAL_MODEL=${NORMAL_MODEL:-"claude-sonnet-4-5-20250929"}
    NORMAL_THINK=${NORMAL_THINK:-false}
    
    HARD_MODEL=${HARD_MODEL:-"claude-opus-4-5-20251101"}
    HARD_THINK=${HARD_THINK:-true}
    
    # 从文件加载（如果存在）
    if [[ -f "$config_file" ]]; then
        # shellcheck source=/dev/null
        source "$config_file"
        log_info "Loaded config from $config_file"
    fi
}

# 根据任务类型自动检测难度
auto_detect_difficulty() {
    local task_type=$1
    
    case "$task_type" in
        missing_header|simple_syntax)
            echo "easy"
            ;;
        sparse_api|template_param|type_cast|api_binding|generic)
            echo "normal"
            ;;
        complex_refactor|cross_file_dependency|syntax)
            echo "hard"
            ;;
        *)
            echo "$DEFAULT_DIFFICULTY"
            ;;
    esac
}

# 获取难度对应的模型配置
get_model_for_difficulty() {
    local difficulty=$1
    
    case "$difficulty" in
        easy)
            echo "$EASY_MODEL"
            ;;
        hard)
            echo "$HARD_MODEL"
            ;;
        *)  # normal 或其他
            echo "$NORMAL_MODEL"
            ;;
    esac
}

# 获取难度对应的思考模式
get_think_for_difficulty() {
    local difficulty=$1
    
    case "$difficulty" in
        easy)
            echo "$EASY_THINK"
            ;;
        hard)
            echo "$HARD_THINK"
            ;;
        *)
            echo "$NORMAL_THINK"
            ;;
    esac
}

# 构建Claude命令
build_claude_cmd() {
    local difficulty=$1
    local model think
    
    model=$(get_model_for_difficulty "$difficulty")
    think=$(get_think_for_difficulty "$difficulty")
    
    local cmd="claude"
    
    # 添加模型参数
    cmd+=" --model $model"
    
    # 添加思考模式参数
    if [[ "$think" == "true" ]]; then
        cmd+=" --thinking"
    fi
    
    # 添加prompt参数
    cmd+=" -p"
    
    echo "$cmd"
}

# 打印难度配置
print_difficulty_config() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  难度配置:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    printf "  ${GREEN}Easy${NC}:   %-35s think=%s\n" "$EASY_MODEL" "$EASY_THINK"
    printf "  ${YELLOW}Normal${NC}: %-35s think=%s\n" "$NORMAL_MODEL" "$NORMAL_THINK"
    printf "  ${RED}Hard${NC}:   %-35s think=%s\n" "$HARD_MODEL" "$HARD_THINK"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# 清理函数
cleanup() {
    local pids=("$@")
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
        fi
    done
}

# 等待所有进程完成
wait_all() {
    local -a pids=("$@")
    local -a failed=()
    
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed+=("$pid")
        fi
    done
    
    echo "${failed[*]}"
}

