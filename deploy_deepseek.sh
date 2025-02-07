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
    echo "  -t, --token TOKEN        Hugging Face token (required)"
    echo "  --use-detected          Use detected parameters from param_detector.py"
    echo "  --setup-only            Only setup environment without deploying model"
    echo "  -h, --help             Show this help message"
}

# 初始化变量
HF_TOKEN=""
USE_DETECTED=false
SETUP_ONLY=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--token)
            HF_TOKEN="$2"
            shift 2
            ;;
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

# 验证必需参数
if [ -z "$HF_TOKEN" ]; then
    log_error "Hugging Face token is required"
    usage
    exit 1
fi

# 检查工作目录
WORK_DIR="$(pwd)"
if [ ! -d "$WORK_DIR/scripts" ]; then
    log_error "Please run this script from the project root directory"
    exit 1
fi

# 设置环境
log_info "Setting up environment..."
source scripts/setup_env.sh
if [ $? -ne 0 ]; then
    log_error "Environment setup failed"
    exit 1
fi

if [ "$SETUP_ONLY" = true ]; then
    log_info "Environment setup completed. Exiting as requested."
    exit 0
fi

# 检测优化参数
if [ "$USE_DETECTED" = true ]; then
    log_info "Detecting optimal parameters..."
    python3 scripts/param_detector.py
    if [ $? -ne 0 ]; then
        log_warn "Parameter detection failed, using default parameters from config"
    fi
fi

# 创建模型目录
mkdir -p "$MODEL_DIR"

# 配置并下载模型
log_info "Configuring Hugging Face..."
export HUGGING_FACE_TOKEN="$HF_TOKEN"
huggingface-cli login "$HF_TOKEN"

log_info "Downloading model from $MODEL_REPO..."
huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR/$MODEL_NAME"

# 设置模型路径
export MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

# 安装 vLLM
log_info "Installing vLLM..."
if [ ! -d "vllm" ]; then
    git clone https://github.com/vllm-project/vllm.git
fi
cd vllm
pip install -U -r requirements-neuron.txt
pip install .

# 设置监控
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

# 启动补充监控（后台）
cd "$WORK_DIR"
log_info "Starting supplementary monitor..."
python3 scripts/monitor.py &
MONITOR_PID=$!

# 检查 Neuron 设备
log_info "Checking Neuron devices..."
neuron-ls

# 启动 vLLM 服务
cd "$WORK_DIR/vllm"
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

# 清理监控进程
trap 'log_info "Shutting down..."; kill $MONITOR_PID; docker-compose -f examples/online_serving/prometheus_grafana/docker-compose.yaml down' EXIT

log_info "Monitoring available at:"
log_info "  - Prometheus: http://localhost:${PROMETHEUS_PORT}"
log_info "  - Grafana: http://localhost:${GRAFANA_PORT}"
log_info "  - Terminal Monitor: Running in background (PID: $MONITOR_PID)"