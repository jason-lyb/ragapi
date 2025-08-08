#!/bin/bash

# RAG API 로그 모니터링 스크립트
# logs/ 폴더의 로그 파일들을 실시간으로 모니터링합니다.

# 파일 경로 설정
LOGS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
OUT_LOG="out.log"
TODAY=$(date +"%Y-%m-%d")
TODAY_LOG="rag_api_${TODAY}.log"

# 색상 출력을 위한 함수
print_info() {
    echo -e "\033[32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[31m[ERROR]\033[0m $1"
}

print_success() {
    echo -e "\033[32m[SUCCESS]\033[0m $1"
}

print_header() {
    echo -e "\033[36m=== $1 ===\033[0m"
}

# 사용법 출력
show_usage() {
    echo "사용법: $0 [옵션] [날짜]"
    echo ""
    echo "옵션:"
    echo "  -h, --help     도움말 표시"
    echo "  -l, --list     사용 가능한 로그 파일 목록 표시"
    echo "  -o, --out      Hypercorn 출력 로그 (out.log) 모니터링"
    echo "  -f, --follow   실시간 로그 모니터링 (기본값)"
    echo "  -n, --lines N  마지막 N줄만 표시 (기본값: 50)"
    echo "  -g, --grep     특정 패턴 검색"
    echo ""
    echo "날짜 형식: YYYY-MM-DD (예: 2025-08-08)"
    echo "날짜를 지정하지 않으면 오늘 로그를 표시합니다."
    echo ""
    echo "예시:"
    echo "  $0                    # 오늘 로그 실시간 모니터링"
    echo "  $0 2025-08-07         # 특정 날짜 로그 실시간 모니터링"
    echo "  $0 -o                 # Hypercorn 출력 로그 모니터링"
    echo "  $0 -l                 # 사용 가능한 로그 파일 목록"
    echo "  $0 -n 100             # 마지막 100줄만 표시"
    echo "  $0 -g \"ERROR\"         # ERROR 패턴만 검색"
}

# 로그 파일 목록 표시
show_log_list() {
    print_header "사용 가능한 로그 파일"
    
    if [ ! -d "$LOGS_DIR" ]; then
        print_error "logs 디렉토리를 찾을 수 없습니다: $LOGS_DIR"
        return 1
    fi
    
    print_info "logs 디렉토리: $LOGS_DIR"
    echo ""
    
    # 애플리케이션 로그 파일들
    print_info "📋 애플리케이션 로그 파일들:"
    ls -la "$LOGS_DIR"/rag_api_*.log 2>/dev/null | while read -r line; do
        echo "  $line"
    done
    
    if [ ! -f "$LOGS_DIR"/rag_api_*.log ]; then
        print_warning "  애플리케이션 로그 파일이 없습니다."
    fi
    
    echo ""
    
    # Hypercorn 출력 로그
    print_info "🖥️  Hypercorn 출력 로그:"
    if [ -f "$OUT_LOG" ]; then
        ls -la "$OUT_LOG"
    else
        print_warning "  out.log 파일이 없습니다."
    fi
    
    echo ""
    print_info "💡 팁:"
    echo "  - 오늘 로그: $0"
    echo "  - 특정 날짜: $0 YYYY-MM-DD"
    echo "  - Hypercorn 로그: $0 -o"
}

# 로그 파일 모니터링
monitor_log() {
    local log_file="$1"
    local follow="$2"
    local lines="$3"
    local grep_pattern="$4"
    
    if [ ! -f "$log_file" ]; then
        print_error "로그 파일을 찾을 수 없습니다: $log_file"
        print_info "사용 가능한 로그 파일을 확인하세요: $0 -l"
        return 1
    fi
    
    print_header "로그 모니터링: $(basename "$log_file")"
    print_info "파일: $log_file"
    print_info "크기: $(du -h "$log_file" | cut -f1)"
    print_info "수정일: $(stat -c %y "$log_file" | cut -d. -f1)"
    
    if [ -n "$grep_pattern" ]; then
        print_info "필터: $grep_pattern"
    fi
    
    echo ""
    print_info "로그 모니터링을 시작합니다... (Ctrl+C로 종료)"
    echo ""
    
    # 실시간 모니터링 또는 마지막 N줄 표시
    if [ "$follow" = true ]; then
        if [ -n "$grep_pattern" ]; then
            tail -n "$lines" -f "$log_file" | grep --color=always "$grep_pattern"
        else
            tail -n "$lines" -f "$log_file"
        fi
    else
        if [ -n "$grep_pattern" ]; then
            tail -n "$lines" "$log_file" | grep --color=always "$grep_pattern"
        else
            tail -n "$lines" "$log_file"
        fi
    fi
}

# 기본값 설정
FOLLOW=true
LINES=50
GREP_PATTERN=""
TARGET_DATE=""
USE_OUT_LOG=false

# 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -l|--list)
            show_log_list
            exit 0
            ;;
        -o|--out)
            USE_OUT_LOG=true
            shift
            ;;
        -f|--follow)
            FOLLOW=true
            shift
            ;;
        --no-follow)
            FOLLOW=false
            shift
            ;;
        -n|--lines)
            LINES="$2"
            shift 2
            ;;
        -g|--grep)
            GREP_PATTERN="$2"
            shift 2
            ;;
        -*)
            print_error "알 수 없는 옵션: $1"
            show_usage
            exit 1
            ;;
        *)
            # 날짜 형식 검증 (YYYY-MM-DD)
            if [[ $1 =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
                TARGET_DATE="$1"
            else
                print_error "잘못된 날짜 형식: $1 (올바른 형식: YYYY-MM-DD)"
                exit 1
            fi
            shift
            ;;
    esac
done

# 메인 로직
if [ "$USE_OUT_LOG" = true ]; then
    # Hypercorn 출력 로그 모니터링
    monitor_log "$OUT_LOG" "$FOLLOW" "$LINES" "$GREP_PATTERN"
else
    # 애플리케이션 로그 모니터링
    if [ -z "$TARGET_DATE" ]; then
        TARGET_DATE="$TODAY"
    fi
    
    LOG_FILE="$LOGS_DIR/rag_api_${TARGET_DATE}.log"
    monitor_log "$LOG_FILE" "$FOLLOW" "$LINES" "$GREP_PATTERN"
fi
