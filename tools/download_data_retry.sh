#!/bin/bash

# 设置最大重试次数（默认 9999 次）
MAX_RETRIES=9999
RETRY_COUNT=0

# 下载命令
DOWNLOAD_CMD="python tools/download_data.py --chunk_id 0 --local_dir data/PhysicalAI-Autonomous-Vehicles-mini"

# 循环下载，直到成功或达到最大重试次数
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "尝试下载 (第 $((RETRY_COUNT + 1)) 次)..."
    $DOWNLOAD_CMD
    
    # 检查是否成功（$? 是上一条命令的退出状态码，0 表示成功）
    if [ $? -eq 0 ]; then
        echo "✅ 下载成功！"
        exit 0
    else
        echo "❌ 下载失败，5 秒后重试..."
        sleep 60
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

echo "❌ 已达到最大重试次数 ($MAX_RETRIES)，下载失败！"
exit 1
