#!/bin/bash

# RAG API 서버 재시작 스크립트
# stop.sh로 서버를 중지한 후 run.sh로 다시 시작합니다.

# 파일 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STOP_SCRIPT="$SCRIPT_DIR/stop.sh"
RUN_SCRIPT="$SCRIPT_DIR/run.sh"
PIDFILE="/tmp/hypercorn.pid"

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

# 스크립트 존재 확인
if [ ! -f "$STOP_SCRIPT" ]; then
    print_error "stop.sh 스크립트를 찾을 수 없습니다: $STOP_SCRIPT"
    exit 1
fi

if [ ! -f "$RUN_SCRIPT" ]; then
    print_error "run.sh 스크립트를 찾을 수 없습니다: $RUN_SCRIPT"
    exit 1
fi

# 스크립트 실행 권한 확인
if [ ! -x "$STOP_SCRIPT" ]; then
    print_error "stop.sh에 실행 권한이 없습니다. chmod +x stop.sh를 실행하세요."
    exit 1
fi

if [ ! -x "$RUN_SCRIPT" ]; then
    print_error "run.sh에 실행 권한이 없습니다. chmod +x run.sh를 실행하세요."
    exit 1
fi

print_info "=== RAG API 서버 재시작 ==="

# 현재 서버 상태 확인
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
        print_info "현재 실행 중인 서버를 발견했습니다 (PID: $PID)"
        SERVER_RUNNING=true
    else
        print_info "PID 파일이 있지만 서버가 실행 중이지 않습니다"
        SERVER_RUNNING=false
    fi
else
    print_info "서버가 실행 중이지 않습니다"
    SERVER_RUNNING=false
fi

# 1단계: 서버 중지
if [ "$SERVER_RUNNING" = true ]; then
    print_info "1단계: 서버 중지 중..."
    if "$STOP_SCRIPT"; then
        print_success "✓ 서버가 성공적으로 중지되었습니다"
    else
        print_error "서버 중지에 실패했습니다"
        exit 1
    fi
else
    print_info "1단계: 서버가 이미 중지되어 있습니다"
fi

# 잠시 대기 (완전한 정리를 위해)
print_info "시스템 정리를 위해 2초 대기..."
sleep 2

# 2단계: 서버 시작
print_info "2단계: 서버 시작 중..."
if "$RUN_SCRIPT"; then
    print_success "✓ 서버 재시작이 완료되었습니다!"
    
    # 새 PID 확인 및 출력
    if [ -f "$PIDFILE" ]; then
        NEW_PID=$(cat "$PIDFILE" 2>/dev/null)
        if [ -n "$NEW_PID" ]; then
            print_info "새 서버 PID: $NEW_PID"
        fi
    fi
else
    print_error "서버 시작에 실패했습니다"
    print_error "run.sh 로그를 확인하세요: cat out.log"
    exit 1
fi

print_success "🎉 RAG API 서버 재시작이 성공적으로 완료되었습니다!"
