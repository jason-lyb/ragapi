#!/bin/bash

# RAG API 서버 중지 스크립트
# run.sh로 시작된 Hypercorn 서버를 안전하게 중지합니다.

# 파일 경로 설정
PIDFILE="/tmp/hypercorn.pid"
LOGFILE="out.log"

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

# PID 파일 확인
if [ ! -f "$PIDFILE" ]; then
    print_warning "PID 파일이 존재하지 않습니다: $PIDFILE"
    print_info "서버가 실행 중이지 않거나 이미 중지되었습니다."
    exit 0
fi

# PID 읽기
PID=$(cat "$PIDFILE")

if [ -z "$PID" ]; then
    print_error "PID 파일이 비어있습니다."
    rm -f "$PIDFILE"
    exit 1
fi

print_info "=== RAG API 서버 중지 ==="
print_info "PID: $PID"

# 프로세스 존재 확인
if ! ps -p "$PID" > /dev/null 2>&1; then
    print_warning "PID $PID 프로세스가 이미 존재하지 않습니다."
    print_info "PID 파일을 정리합니다."
    rm -f "$PIDFILE"
    exit 0
fi

# 프로세스 정보 표시
PROCESS_INFO=$(ps -p "$PID" -o pid,ppid,cmd --no-headers 2>/dev/null)
if [ -n "$PROCESS_INFO" ]; then
    print_info "실행 중인 프로세스:"
    print_info "  $PROCESS_INFO"
fi

# SIGTERM으로 정상 종료 시도
print_info "SIGTERM 신호를 보내어 정상 종료를 시도합니다..."
kill -TERM "$PID"

# 정상 종료 대기 (최대 15초)
WAIT_TIME=0
MAX_WAIT=15

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        print_success "✓ 서버가 정상적으로 종료되었습니다."
        rm -f "$PIDFILE"
        exit 0
    fi
    
    sleep 1
    WAIT_TIME=$((WAIT_TIME + 1))
    
    if [ $((WAIT_TIME % 3)) -eq 0 ]; then
        print_info "종료 대기 중... (${WAIT_TIME}/${MAX_WAIT}초)"
    fi
done

# 정상 종료 실패 시 강제 종료
print_warning "정상 종료에 실패했습니다. 강제 종료를 시도합니다..."
kill -KILL "$PID" 2>/dev/null

# 강제 종료 확인 (최대 5초)
WAIT_TIME=0
MAX_WAIT=5

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        print_success "✓ 서버가 강제로 종료되었습니다."
        rm -f "$PIDFILE"
        exit 0
    fi
    
    sleep 1
    WAIT_TIME=$((WAIT_TIME + 1))
done

# 모든 시도 실패
print_error "서버 종료에 실패했습니다."
print_error "수동으로 프로세스를 확인하세요: ps -ef | grep hypercorn"
print_error "수동 종료: kill -9 $PID"
exit 1
