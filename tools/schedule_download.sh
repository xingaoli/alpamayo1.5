#!/bin/bash

# 定时任务管理脚本
# 功能: 每天18:00启动下载脚本,次日08:30停止

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_data_retry.sh"
PID_FILE="$SCRIPT_DIR/.download_script.pid"
LOG_FILE="$SCRIPT_DIR/output.log"

# 启动下载任务
start_download() {
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "[$(date)] 下载任务已在运行中 (PID: $(cat $PID_FILE))" | tee -a "$LOG_FILE"
        return
    fi

    echo "[$(date)] 启动下载任务..." | tee -a "$LOG_FILE"
    # 加载用户环境变量（包括 HF_ENDPOINT 等）
    source ~/.bashrc 2>/dev/null || true
    # 激活虚拟环境后运行
    source "$SCRIPT_DIR/../.venv/bin/activate"
    bash "$DOWNLOAD_SCRIPT" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "[$(date)] 下载任务已启动 (PID: $!)" | tee -a "$LOG_FILE"
}

# 停止下载任务
stop_download() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[$(date)] 停止下载任务 (PID: $PID)..." | tee -a "$LOG_FILE"
            
            # 先尝试优雅终止
            kill -TERM "$PID" 2>/dev/null
            sleep 5
            
            # 检查是否已停止
            if kill -0 "$PID" 2>/dev/null; then
                echo "[$(date)] 强制终止下载任务..." | tee -a "$LOG_FILE"
                kill -KILL "$PID" 2>/dev/null
            fi
            
            echo "[$(date)] 下载任务已停止" | tee -a "$LOG_FILE"
        else
            echo "[$(date)] 下载任务未在运行 (PID: $PID 已不存在)" | tee -a "$LOG_FILE"
        fi
        rm -f "$PID_FILE"
    else
        echo "[$(date)] 下载任务未在运行" | tee -a "$LOG_FILE"
    fi
}

# 清理所有相关的Python下载进程
stop_all_download() {
    echo "[$(date)] 清理所有下载进程..." | tee -a "$LOG_FILE"
    pkill -f "python tools/download_data.py" 2>/dev/null
    rm -f "$PID_FILE"
}

# 主循环
echo "[$(date)] 定时任务管理器已启动" | tee -a "$LOG_FILE"
echo "[$(date)] 下载任务将在每天 18:00 启动,08:30 停止" | tee -a "$LOG_FILE"

# 立即启动一次下载任务
echo "[$(date)] 立即启动首次下载任务..." | tee -a "$LOG_FILE"
start_download

while true; do
    CURRENT_TIME=$(date +%H%M)
    CURRENT_HOUR=$(date +%H)
    CURRENT_MIN=$(date +%M)
    
    # 18:00 启动
    if [ "$CURRENT_TIME" = "1800" ]; then
        start_download
        sleep 60  # 等待1分钟避免重复触发
    fi
    
    # 08:30 停止
    if [ "$CURRENT_TIME" = "0830" ]; then
        stop_download
        sleep 60  # 等待1分钟避免重复触发
    fi
    
    # 每分钟检查一次
    sleep 10
done
