#!/bin/bash

# 设置最大重试次数（默认 9999 次）
MAX_RETRIES=9999
RETRY_COUNT=0

# 激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."

# 加载用户环境变量（包括 HF_ENDPOINT 等）
source ~/.bashrc 2>/dev/null || true

# 显式设置 Hugging Face 镜像站
export HF_ENDPOINT=https://hf-mirror.com
# 可选：禁用 XET 网关，如果镜像站不支持 XET
export HF_HUB_DISABLE_XET=1
# 禁用 HF Transfer（因为未安装 hf_transfer 包）
export HF_HUB_ENABLE_HF_TRANSFER=0

# 调试：打印环境变量
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"

# 激活虚拟环境
source "$PROJECT_DIR/.venv/bin/activate"

# 切换到项目根目录
cd "$PROJECT_DIR"

# 下载命令
DOWNLOAD_CMD="python tools/download_data.py --chunk_id -1 --local_dir /mnt/hdd_data/public_data/PhysicalAI-Autonomous-Vehicles-only-4-cam"

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
