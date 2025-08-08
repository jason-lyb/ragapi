#!/bin/bash

# RAG API 서버 시작 스크립트
# main.py 설정에 기반하여 Hypercorn 서버를 실행합니다.

# 가상환경 활성화
source venv/bin/activate

# 환경변수 설정 (main.py의 기본값 사용)
export WORKERS=${WORKERS:-1}
export KEEPALIVE=${KEEPALIVE:-65}
export MAX_REQUESTS=${MAX_REQUESTS:-1000}
export MAX_REQUESTS_JITTER=${MAX_REQUESTS_JITTER:-50}
export TEMP_DIR=${TEMP_DIR:-"temp"}

# 파일 경로 설정
LOGFILE="out.log"
PIDFILE="/tmp/hypercorn.pid"
CERTFILE="../certs/node1.pem"
KEYFILE="../certs/node1-key.pem"
HOST="0.0.0.0"
PORT="8002"

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

# 이미 실행 중인지 확인 (PID 파일 존재 여부)
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        print_warning "이미 실행 중입니다. PID: $PID"
        print_info "서버를 재시작하려면 먼저 './stop.sh'를 실행하세요."
        exit 1
    else
        print_warning "PID 파일이 있으나, 프로세스가 존재하지 않습니다. PID 파일을 제거합니다."
        rm -f "$PIDFILE"
    fi
fi

# 임시 디렉토리 생성
if [ ! -d "$TEMP_DIR" ]; then
    print_info "임시 디렉토리 생성: $TEMP_DIR"
    mkdir -p "$TEMP_DIR"
fi

# SSL 인증서 파일 확인
if [ -f "$CERTFILE" ] && [ -f "$KEYFILE" ]; then
    print_info "SSL 인증서 파일을 찾았습니다. HTTPS로 시작합니다."
    SSL_OPTIONS="--certfile=$CERTFILE --keyfile=$KEYFILE"
else
    print_warning "SSL 인증서 파일을 찾을 수 없습니다. HTTP로 시작합니다."
    SSL_OPTIONS=""
fi

# 서버 설정 정보 출력
print_info "=== RAG API 서버 시작 ==="
print_info "애플리케이션: main:app"
print_info "바인드 주소: ${HOST}:${PORT}"
print_info "작업자 수: $WORKERS"
print_info "Keep-Alive 타임아웃: ${KEEPALIVE}초"
print_info "최대 요청 수: $MAX_REQUESTS"
print_info "로그 파일: $LOGFILE"
print_info "PID 파일: $PIDFILE"

# Hypercorn 서버 시작
print_info "Hypercorn 서버를 백그라운드로 시작합니다..."

if [ -n "$SSL_OPTIONS" ]; then
    # HTTPS로 시작
    nohup hypercorn main:app \
        --workers $WORKERS \
        --bind ${HOST}:${PORT} \
        --keep-alive $KEEPALIVE \
        --max-requests $MAX_REQUESTS \
        --max-requests-jitter $MAX_REQUESTS_JITTER \
        --worker-class uvloop \
        $SSL_OPTIONS \
        > "$LOGFILE" 2>&1 &
else
    # HTTP로 시작
    nohup hypercorn main:app \
        --workers $WORKERS \
        --bind ${HOST}:${PORT} \
        --keep-alive $KEEPALIVE \
        --max-requests $MAX_REQUESTS \
        --max-requests-jitter $MAX_REQUESTS_JITTER \
        --worker-class uvloop \
        > "$LOGFILE" 2>&1 &
fi

# PID 저장
SERVER_PID=$!
echo $SERVER_PID > "$PIDFILE"

# 서버 시작 확인
sleep 2
if ps -p $SERVER_PID > /dev/null 2>&1; then
    print_info "✓ Hypercorn 서버가 성공적으로 시작되었습니다!"
    print_info "  - PID: $SERVER_PID"
    if [ -n "$SSL_OPTIONS" ]; then
        print_info "  - URL: https://${HOST}:${PORT}"
        print_info "  - API 문서: https://${HOST}:${PORT}/docs"
    else
        print_info "  - URL: http://${HOST}:${PORT}"
        print_info "  - API 문서: http://${HOST}:${PORT}/docs"
    fi
    print_info "  - 로그 확인: tail -f $LOGFILE"
    print_info "  - 서버 중지: ./stop.sh"
else
    print_error "서버 시작에 실패했습니다."
    print_error "로그를 확인하세요: cat $LOGFILE"
    exit 1
fi
