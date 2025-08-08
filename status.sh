#!/bin/bash

# RAG API 서버 상태 확인 스크립트
# 서버의 실행 상태, 프로세스 정보, 포트 상태 등을 확인합니다.

# 파일 경로 설정
PIDFILE="/tmp/hypercorn.pid"
LOGFILE="out.log"
LOGS_DIR="logs"
HOST="0.0.0.0"
PORT="8002"
TODAY=$(date +"%Y-%m-%d")
TODAY_LOG="$LOGS_DIR/rag_api_${TODAY}.log"

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
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  -h, --help     도움말 표시"
    echo "  -v, --verbose  상세 정보 표시"
    echo "  -q, --quiet    간단한 상태만 표시"
    echo "  -p, --port     포트 상태만 확인"
    echo "  -l, --log      로그 정보만 표시"
    echo ""
    echo "예시:"
    echo "  $0             # 기본 상태 확인"
    echo "  $0 -v          # 상세 정보 표시"
    echo "  $0 -q          # 간단한 상태만"
    echo "  $0 -p          # 포트 상태만"
    echo "  $0 -l          # 로그 정보만"
}

# 프로세스 상태 확인
check_process_status() {
    local verbose="$1"
    
    print_header "프로세스 상태"
    
    if [ ! -f "$PIDFILE" ]; then
        print_warning "PID 파일이 존재하지 않습니다: $PIDFILE"
        echo "  🔴 서버 상태: NOT RUNNING"
        return 1
    fi
    
    PID=$(cat "$PIDFILE" 2>/dev/null)
    
    if [ -z "$PID" ]; then
        print_error "PID 파일이 비어있습니다."
        echo "  🔴 서버 상태: NOT RUNNING"
        return 1
    fi
    
    if ps -p "$PID" > /dev/null 2>&1; then
        print_success "서버가 실행 중입니다!"
        echo "  🟢 서버 상태: RUNNING"
        echo "  📋 PID: $PID"
        
        if [ "$verbose" = true ]; then
            # 상세 프로세스 정보
            echo ""
            print_info "상세 프로세스 정보:"
            ps -p "$PID" -o pid,ppid,cmd,etime,pcpu,pmem --no-headers | while read -r line; do
                echo "    $line"
            done
            
            # 프로세스 시작 시간
            START_TIME=$(ps -p "$PID" -o lstart --no-headers 2>/dev/null)
            if [ -n "$START_TIME" ]; then
                echo "  ⏰ 시작 시간: $START_TIME"
            fi
            
            # 메모리 사용량
            MEM_INFO=$(ps -p "$PID" -o pid,vsz,rss --no-headers 2>/dev/null)
            if [ -n "$MEM_INFO" ]; then
                echo "  💾 메모리 정보: $MEM_INFO"
            fi
        fi
        
        return 0
    else
        print_warning "PID $PID 프로세스가 존재하지 않습니다."
        echo "  🔴 서버 상태: NOT RUNNING"
        return 1
    fi
}

# 포트 상태 확인
check_port_status() {
    local verbose="$1"
    
    print_header "포트 상태"
    
    # 포트 리스닝 확인 (ss 명령어 우선, netstat 백업)
    PORT_LISTENING=false
    if command -v ss >/dev/null 2>&1; then
        if ss -tlnp 2>/dev/null | grep ":$PORT " > /dev/null; then
            PORT_LISTENING=true
            PORT_CMD="ss"
        fi
    elif command -v netstat >/dev/null 2>&1; then
        if netstat -tlnp 2>/dev/null | grep ":$PORT " > /dev/null; then
            PORT_LISTENING=true
            PORT_CMD="netstat"
        fi
    fi
    
    if [ "$PORT_LISTENING" = true ]; then
        print_success "포트 $PORT이 활성화되어 있습니다!"
        echo "  🟢 포트 상태: LISTENING"
        
        if [ "$verbose" = true ]; then
            echo ""
            print_info "포트 상세 정보:"
            if [ "$PORT_CMD" = "ss" ]; then
                ss -tlnp 2>/dev/null | grep ":$PORT " | while read -r line; do
                    echo "    $line"
                done
            else
                netstat -tlnp 2>/dev/null | grep ":$PORT " | while read -r line; do
                    echo "    $line"
                done
            fi
        fi
        
        # 연결 테스트
        if command -v curl >/dev/null 2>&1; then
            echo ""
            print_info "연결 테스트 중..."
            if curl -s --connect-timeout 3 "http://$HOST:$PORT" >/dev/null 2>&1; then
                echo "  ✅ HTTP 연결: 성공"
            else
                echo "  ❌ HTTP 연결: 실패"
            fi
        fi
        
        return 0
    else
        print_warning "포트 $PORT이 리스닝 상태가 아닙니다."
        echo "  🔴 포트 상태: NOT LISTENING"
        
        if [ "$verbose" = true ]; then
            echo ""
            print_info "디버깅 정보:"
            if command -v ss >/dev/null 2>&1; then
                echo "    ss 명령어로 모든 리스닝 포트 확인:"
                ss -tln | grep LISTEN | head -5 | sed 's/^/      /'
            elif command -v netstat >/dev/null 2>&1; then
                echo "    netstat 명령어로 모든 리스닝 포트 확인:"
                netstat -tln | grep LISTEN | head -5 | sed 's/^/      /'
            else
                echo "    포트 확인 도구(ss 또는 netstat)가 설치되지 않았습니다."
            fi
        fi
        
        return 1
    fi
}

# 로그 상태 확인
check_log_status() {
    local verbose="$1"
    
    print_header "로그 상태"
    
    # Hypercorn 출력 로그
    if [ -f "$LOGFILE" ]; then
        LOG_SIZE=$(du -h "$LOGFILE" | cut -f1)
        LOG_LINES=$(wc -l < "$LOGFILE" 2>/dev/null || echo "0")
        LOG_MODIFIED=$(stat -c %y "$LOGFILE" 2>/dev/null | cut -d. -f1)
        
        print_info "Hypercorn 출력 로그 ($LOGFILE):"
        echo "  📁 크기: $LOG_SIZE"
        echo "  📄 라인 수: $LOG_LINES"
        echo "  🕒 수정일: $LOG_MODIFIED"
        
        if [ "$verbose" = true ]; then
            echo ""
            print_info "최근 로그 (마지막 5줄):"
            tail -n 5 "$LOGFILE" 2>/dev/null | sed 's/^/    /'
        fi
    else
        print_warning "Hypercorn 출력 로그를 찾을 수 없습니다: $LOGFILE"
    fi
    
    echo ""
    
    # 애플리케이션 로그
    if [ -f "$TODAY_LOG" ]; then
        LOG_SIZE=$(du -h "$TODAY_LOG" | cut -f1)
        LOG_LINES=$(wc -l < "$TODAY_LOG" 2>/dev/null || echo "0")
        LOG_MODIFIED=$(stat -c %y "$TODAY_LOG" 2>/dev/null | cut -d. -f1)
        
        print_info "애플리케이션 로그 ($TODAY_LOG):"
        echo "  📁 크기: $LOG_SIZE"
        echo "  📄 라인 수: $LOG_LINES"
        echo "  🕒 수정일: $LOG_MODIFIED"
        
        if [ "$verbose" = true ]; then
            echo ""
            print_info "최근 애플리케이션 로그 (마지막 5줄):"
            tail -n 5 "$TODAY_LOG" 2>/dev/null | sed 's/^/    /'
        fi
    else
        print_warning "오늘의 애플리케이션 로그를 찾을 수 없습니다: $TODAY_LOG"
    fi
}

# 전체 상태 확인
check_full_status() {
    local verbose="$1"
    
    print_header "RAG API 서버 상태 점검"
    echo "  📅 점검 시간: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 프로세스 상태 확인
    PROCESS_RUNNING=false
    if check_process_status "$verbose"; then
        PROCESS_RUNNING=true
    fi
    
    echo ""
    
    # 포트 상태 확인
    PORT_LISTENING=false
    if check_port_status "$verbose"; then
        PORT_LISTENING=true
    fi
    
    echo ""
    
    # 로그 상태 확인
    check_log_status "$verbose"
    
    echo ""
    
    # 전체 상태 요약
    print_header "상태 요약"
    
    if [ "$PROCESS_RUNNING" = true ] && [ "$PORT_LISTENING" = true ]; then
        print_success "✅ RAG API 서버가 정상적으로 실행 중입니다!"
        echo "  🌐 접속 URL: http://$HOST:$PORT"
        echo "  📚 API 문서: http://$HOST:$PORT/docs"
        echo ""
        echo "  💡 유용한 명령어:"
        echo "    - 로그 모니터링: ./tlog.sh"
        echo "    - 서버 중지: ./stop.sh"
        echo "    - 서버 재시작: ./restart.sh"
    elif [ "$PROCESS_RUNNING" = true ]; then
        print_warning "⚠️  프로세스는 실행 중이지만 포트가 활성화되지 않았습니다."
        echo "  💡 서버가 시작 중이거나 문제가 있을 수 있습니다."
        echo "  📋 로그 확인: ./tlog.sh -o"
    else
        print_error "❌ RAG API 서버가 실행되지 않고 있습니다."
        echo "  💡 서버 시작: ./run.sh"
    fi
}

# 간단한 상태만 출력
check_simple_status() {
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE" 2>/dev/null)
        if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
            echo "🟢 RUNNING (PID: $PID)"
            return 0
        fi
    fi
    echo "🔴 NOT RUNNING"
    return 1
}

# 기본값 설정
VERBOSE=false
QUIET=false
PORT_ONLY=false
LOG_ONLY=false

# 인수 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        -p|--port)
            PORT_ONLY=true
            shift
            ;;
        -l|--log)
            LOG_ONLY=true
            shift
            ;;
        -*)
            print_error "알 수 없는 옵션: $1"
            show_usage
            exit 1
            ;;
        *)
            print_error "알 수 없는 인수: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 메인 로직
if [ "$QUIET" = true ]; then
    # 간단한 상태만 출력
    check_simple_status
elif [ "$PORT_ONLY" = true ]; then
    # 포트 상태만 확인
    check_port_status "$VERBOSE"
elif [ "$LOG_ONLY" = true ]; then
    # 로그 정보만 표시
    check_log_status "$VERBOSE"
else
    # 전체 상태 확인
    check_full_status "$VERBOSE"
fi
