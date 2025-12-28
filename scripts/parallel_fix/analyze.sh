#!/bin/bash
# ============================================================
# 编译错误分析器
# 从编译日志自动生成修复任务
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

# 加载配置以获取难度检测函数
load_config

# 默认输出
OUTPUT_FORMAT="json"
OUTPUT_FILE=""

show_help() {
    cat << EOF
编译错误分析器

用法:
    $0 [选项] <compile.log>

选项:
    -o, --output FILE   输出文件 (默认: stdout)
    -f, --format FMT    输出格式: json, text (默认: json)
    -h, --help          显示帮助

示例:
    # 分析并生成JSON任务文件
    make build 2>&1 | tee build.log
    $0 build.log -o tasks.json

    # 分析并输出简单文本格式
    $0 -f text build.log > tasks.txt
EOF
}

# 解析命令行参数
parse_args() {
    local log_file=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -o|--output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
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
                log_file="$1"
                shift
                ;;
        esac
    done
    
    if [[ -z "$log_file" ]]; then
        log_error "请指定编译日志文件"
        show_help
        exit 1
    fi
    
    echo "$log_file"
}

# 提取错误文件列表
extract_error_files() {
    local log_file=$1
    
    # 提取带有 error: 的文件
    grep -oE '[a-zA-Z0-9_/.-]+\.(hpp|cpp|h|cc):[0-9]+:[0-9]+: error:' "$log_file" 2>/dev/null | \
        cut -d: -f1 | sort -u
}

# 统计每个文件的错误数
count_errors_per_file() {
    local log_file=$1
    local file=$2
    
    grep -c "^${file}:" "$log_file" 2>/dev/null || echo 0
}

# 检测错误类型
detect_error_type() {
    local log_file=$1
    local file=$2
    
    # 检查各种错误模式
    if grep -q "${file}.*row_indices_unsafe()\[" "$log_file" 2>/dev/null || \
       grep -q "${file}.*col_indices_unsafe()\[" "$log_file" 2>/dev/null || \
       grep -q "${file}.*\.values()\[" "$log_file" 2>/dev/null; then
        echo "sparse_api"
        return
    fi
    
    if grep -q "${file}.*static_cast<std::atomic" "$log_file" 2>/dev/null; then
        echo "type_cast"
        return
    fi
    
    if grep -q "${file}.*DualWorkspacePool<.*,.*>" "$log_file" 2>/dev/null || \
       grep -q "${file}.*template argument" "$log_file" 2>/dev/null; then
        echo "template_param"
        return
    fi
    
    if grep -q "${file}.*expected ')'" "$log_file" 2>/dev/null || \
       grep -q "${file}.*expected '}'" "$log_file" 2>/dev/null || \
       grep -q "${file}.*function-definition is not allowed" "$log_file" 2>/dev/null; then
        echo "syntax"
        return
    fi
    
    if grep -q "${file}.*is not a member of 'std'" "$log_file" 2>/dev/null; then
        echo "missing_header"
        return
    fi
    
    if [[ "$file" == *"binding/c_api"* ]]; then
        echo "api_binding"
        return
    fi
    
    echo "generic"
}

# 获取错误行号
get_error_lines() {
    local log_file=$1
    local file=$2
    
    grep -oE "^${file}:[0-9]+" "$log_file" 2>/dev/null | \
        cut -d: -f2 | sort -nu | head -10 | tr '\n' ',' | sed 's/,$//'
}

# 生成任务描述
generate_description() {
    local error_type=$1
    local file=$2
    local error_count=$3
    
    case "$error_type" in
        sparse_api)
            echo "修复稀疏矩阵API ($error_count 个错误)"
            ;;
        template_param)
            echo "修复模板参数 ($error_count 个错误)"
            ;;
        type_cast)
            echo "修复类型转换 ($error_count 个错误)"
            ;;
        syntax)
            echo "修复语法错误 ($error_count 个错误)"
            ;;
        missing_header)
            echo "添加缺少的头文件"
            ;;
        api_binding)
            echo "修复C API绑定 ($error_count 个错误)"
            ;;
        *)
            echo "修复编译错误 ($error_count 个错误)"
            ;;
    esac
}

# 合并相同类型的任务
merge_tasks() {
    local -n tasks_ref=$1
    local -n merged_ref=$2
    
    declare -A type_files
    declare -A type_counts
    declare -A type_lines
    
    for task in "${tasks_ref[@]}"; do
        IFS='|' read -r _ type file desc lines count <<< "$task"
        
        # 对于某些类型，合并到一个任务
        if [[ "$type" == "template_param" || "$type" == "type_cast" || "$type" == "missing_header" ]]; then
            if [[ -z "${type_files[$type]:-}" ]]; then
                type_files[$type]="$file"
                type_counts[$type]=$count
                type_lines[$type]="$lines"
            else
                type_files[$type]="${type_files[$type]}, $file"
                type_counts[$type]=$((type_counts[$type] + count))
            fi
        else
            merged_ref+=("$task")
        fi
    done
    
    # 添加合并后的任务
    local merge_id=100
    for type in "${!type_files[@]}"; do
        local desc
        desc=$(generate_description "$type" "" "${type_counts[$type]}")
        merged_ref+=("$merge_id|$type|${type_files[$type]}|$desc||${type_counts[$type]}")
        ((merge_id++))
    done
}

# 输出JSON格式
output_json() {
    local -n tasks_ref=$1
    
    echo "{"
    echo '  "generated_at": "'$(date -Is)'",'
    echo '  "tasks": ['
    
    local first=true
    local id=1
    for task in "${tasks_ref[@]}"; do
        IFS='|' read -r _ type files desc lines count <<< "$task"
        
        # 自动检测难度
        local difficulty
        difficulty=$(auto_detect_difficulty "$type")
        
        if $first; then
            first=false
        else
            echo ","
        fi
        
        # 将文件列表转换为JSON数组
        local files_json="["
        local first_file=true
        IFS=', ' read -ra file_arr <<< "$files"
        for f in "${file_arr[@]}"; do
            if $first_file; then
                first_file=false
            else
                files_json+=","
            fi
            files_json+="\"$f\""
        done
        files_json+="]"
        
        cat << EOF
    {
      "id": "$id",
      "type": "$type",
      "files": $files_json,
      "description": "$desc",
      "difficulty": "$difficulty",
      "error_lines": "$lines",
      "error_count": $count
    }
EOF
        ((id++))
    done
    
    echo ""
    echo "  ]"
    echo "}"
}

# 输出文本格式
output_text() {
    local -n tasks_ref=$1
    
    echo "# 自动生成的修复任务"
    echo "# 格式: ID|类型|文件|描述|难度"
    echo ""
    
    local id=1
    for task in "${tasks_ref[@]}"; do
        IFS='|' read -r _ type files desc _ _ <<< "$task"
        
        # 自动检测难度
        local difficulty
        difficulty=$(auto_detect_difficulty "$type")
        
        echo "$id|$type|$files|$desc|$difficulty"
        ((id++))
    done
}

# 主函数
main() {
    local log_file
    log_file=$(parse_args "$@")
    
    if [[ ! -f "$log_file" ]]; then
        log_error "文件不存在: $log_file"
        exit 1
    fi
    
    log_info "分析编译日志: $log_file" >&2
    
    # 提取错误文件
    local -a error_files
    mapfile -t error_files < <(extract_error_files "$log_file")
    
    if [[ ${#error_files[@]} -eq 0 ]]; then
        log_warn "未找到编译错误" >&2
        exit 0
    fi
    
    log_info "发现 ${#error_files[@]} 个包含错误的文件" >&2
    
    # 分析每个文件
    local -a tasks=()
    local id=1
    
    for file in "${error_files[@]}"; do
        local error_type error_count error_lines description
        
        error_type=$(detect_error_type "$log_file" "$file")
        error_count=$(count_errors_per_file "$log_file" "$file")
        error_lines=$(get_error_lines "$log_file" "$file")
        description=$(generate_description "$error_type" "$file" "$error_count")
        
        tasks+=("$id|$error_type|$file|$description|$error_lines|$error_count")
        
        log_info "  [$error_type] $file: $error_count 个错误" >&2
        ((id++))
    done
    
    # 合并相似任务
    local -a merged_tasks=()
    merge_tasks tasks merged_tasks
    
    log_success "生成 ${#merged_tasks[@]} 个修复任务" >&2
    
    # 输出结果
    local output
    case "$OUTPUT_FORMAT" in
        json)
            output=$(output_json merged_tasks)
            ;;
        text)
            output=$(output_text merged_tasks)
            ;;
        *)
            log_error "未知格式: $OUTPUT_FORMAT"
            exit 1
            ;;
    esac
    
    if [[ -n "$OUTPUT_FILE" ]]; then
        echo "$output" > "$OUTPUT_FILE"
        log_success "任务已保存到: $OUTPUT_FILE" >&2
    else
        echo "$output"
    fi
}

main "$@"

