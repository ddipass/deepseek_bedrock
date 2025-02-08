#!/bin/bash
# deploy_deepseek.sh

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 帮助函数
function usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --use-detected          Use detected parameters from param_detector.py"
    echo "  --setup-only            Only setup environment without deploying model"
    echo "  -h, --help             Show this help message"
}

# S3设置和模型下载函数
setup_s3_and_model() {
    log_info "Setting up model storage..."

    # 获取S3 bucket名称
    INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")" http://169.254.169.254/latest/meta-data/instance-id)
    AWS_S3_BUCKET="deepseek-model-${INSTANCE_ID}"

    # 创建S3 bucket (如果不存在)
    aws s3 mb "s3://${AWS_S3_BUCKET}" || true

    # 安装s3fs
    if ! command -v s3fs &> /dev/null; then
        log_info "Installing s3fs-fuse..."
        sudo apt-get update && sudo apt-get install -y s3fs
    fi

    # 创建挂载点
    S3_MOUNT_PATH="${MODEL_DIR}/s3_models"
    mkdir -p "$S3_MOUNT_PATH"

    # 挂载S3并验证
    log_info "Mounting S3 bucket..."
    s3fs "${AWS_S3_BUCKET}" "$S3_MOUNT_PATH" -o allow_other
    if ! mountpoint -q "$S3_MOUNT_PATH"; then
        log_error "Failed to mount S3 bucket"
        exit 1
    fi

    # 直接下载到挂载的S3路径
    log_info "Downloading model from $MODEL_REPO directly to S3..."
    TEMP_DIR=$(mktemp -d)
    huggingface-cli download "$MODEL_REPO" --local-dir "$TEMP_DIR" && \
    mv "$TEMP_DIR"/* "$S3_MOUNT_PATH/$MODEL_NAME/" && \
    rm -rf "$TEMP_DIR"

    # 设置模型路径为S3挂载路径
    export MODEL_PATH="$S3_MOUNT_PATH/$MODEL_NAME"
}

# 安装VLLM函数
install_vllm() {
    log_info "Installing vLLM..."
    if [ ! -d "vllm" ]; then
        git clone https://github.com/vllm-project/vllm.git
    fi
    cd vllm
    pip install -U -r requirements-neuron.txt
    pip install .
}

# 设置监控函数
setup_monitoring() {
    log_info "Setting up monitoring..."

    # 确保prometheus和grafana配置目录存在
    mkdir -p examples/online_serving/prometheus_grafana
    cp docker-compose.yaml examples/online_serving/prometheus_grafana/
    cp prometheus.yaml examples/online_serving/prometheus_grafana/
    cp grafana.json examples/online_serving/prometheus_grafana/

    # 启动Prometheus和Grafana
    cd examples/online_serving/prometheus_grafana
    log_info "Starting Prometheus and Grafana..."
    docker-compose down
    docker-compose up -d

    # 等待服务启动
    sleep 5
}

# 启动服务函数
start_vllm_service() {
    log_info "Starting vLLM server..."
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$(basename "$MODEL_REPO")" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --block-size "$BLOCK_SIZE" \
        --use-v2-block-manager \
        --device neuron \
        --port "$MODEL_PORT"
}

# 清理函数
cleanup() {
    log_info "Shutting down..."
    kill $MONITOR_PID
    docker-compose -f examples/online_serving/prometheus_grafana/docker-compose.yaml down
    fusermount -u "$S3_MOUNT_PATH"
}

# 主函数
main() {
    # 初始化变量
    USE_DETECTED=false
    SETUP_ONLY=false

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --use-detected)
                USE_DETECTED=true
                shift
                ;;
            --setup-only)
                SETUP_ONLY=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # 导入配置
    eval "$(python3 -c 'from config import Config; print(Config.to_shell_exports())')"

    # 检查工作目录
    WORK_DIR="$(pwd)"
    if [ ! -d "$WORK_DIR/scripts" ]; then
        log_error "Please run this script from the project root directory"
        exit 1
    fi

    # 设置环境
    log_info "Setting up environment..."
    source scripts/setup_env.sh || {
        log_error "Environment setup failed"
        exit 1
    }

    if [ "$SETUP_ONLY" = true ]; then
        log_info "Environment setup completed. Exiting as requested."
        exit 0
    fi

    # 检测优化参数
    if [ "$USE_DETECTED" = true ]; then
        log_info "Detecting optimal parameters..."
        python3 scripts/param_detector.py || {
            log_warn "Parameter detection failed, using default parameters from config"
        }
    fi

    # 设置S3和下载模型
    setup_s3_and_model

    # 安装VLLM
    install_vllm

    # 设置监控
    setup_monitoring

    # 启动补充监控
    cd "$WORK_DIR"
    log_info "Starting supplementary monitor..."
    python3 scripts/monitor.py &
    MONITOR_PID=$!

    # 检查Neuron设备
    log_info "Checking Neuron devices..."
    neuron-ls

    # 启动VLLM服务
    cd "$WORK_DIR/vllm"
    start_vllm_service

    # 设置清理trap
    trap cleanup EXIT

    # 显示监控信息
    log_info "Monitoring available at:"
    log_info "  - Prometheus: http://localhost:${PROMETHEUS_PORT}"
    log_info "  - Grafana: http://localhost:${GRAFANA_PORT}"
    log_info "  - Terminal Monitor: Running in background (PID: $MONITOR_PID)"
}

# 运行主函数
main "$@"