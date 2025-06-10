#!/bin/bash

PID_FILE="/tmp/hypercorn.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Stopping hypercorn process with PID $PID..."
    kill "$PID"
    # 프로세스가 완전히 종료될 때까지 잠시 기다린 후 PID 파일 삭제
    sleep 2
    if ps -p "$PID" > /dev/null; then
        echo "Process did not terminate, forcing kill..."
        kill -9 "$PID"
    fi
    rm -f "$PID_FILE"
    echo "Process stopped."
else
    echo "PID file not found. Process may not be running."
fi
