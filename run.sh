#!/bin/bash
source venv/bin/activate

LOGFILE="out.log"
PIDFILE="/tmp/hypercorn.pid"

# 이미 실행 중인지 확인 (PID 파일 존재 여부)
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "이미 실행 중입니다. PID: $PID"
        exit 1
    else
        echo "PID 파일이 있으나, 프로세스가 존재하지 않습니다. PID 파일을 제거합니다."
        rm -f "$PIDFILE"
    fi
fi

echo "Hypercorn 프로세스를 시작합니다..."
#nohup authbind --deep hypercorn main:app --workers 4 --bind 0.0.0.0:443 --certfile=../certs/node1.pem --keyfile=../certs/node1-key.pem > "$LOGFILE" 2>&1 &
nohup hypercorn main:app --workers 1 --bind 0.0.0.0:443 --certfile=../certs/node1.pem --keyfile=../certs/node1-key.pem > "$LOGFILE" 2>&1 &
echo $! > "$PIDFILE"
echo "Hypercorn 프로세스가 PID $(cat "$PIDFILE")로 시작되었습니다."
