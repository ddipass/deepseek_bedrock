#!/bin/bash

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

# 检查是否是Inferentia2实例
check_inferentia() {
    log_info "Checking for Inferentia2 device..."
    
    # 检查 neuron-ls 是否存在并可执行
    if [ ! -x "/opt/aws/neuron/bin/neuron-ls" ]; then
        log_error "neuron-ls not found or not executable"
        exit 1
    fi
    
    # 运行 neuron-ls 并检查输出
    NEURON_OUTPUT=$(/opt/aws/neuron/bin/neuron-ls 2>&1)
    if [ $? -eq 0 ] && echo "$NEURON_OUTPUT" | grep -q "inf2"; then
        log_info "Found Inferentia2 devices:"
        echo "$NEURON_OUTPUT"
        return 0
    else
        log_error "No Inferentia2 devices found"
        exit 1
    fi
}
# 检查并激活Neuron环境
setup_neuron_env() {
    log_info "Setting up Neuron environment..."
    
    # 检查Neuron环境是否存在
    NEURON_ENV="/opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate"
    if [ ! -f "$NEURON_ENV" ]; then
        log_error "Neuron environment not found at $NEURON_ENV"
        exit 1
    fi
    
    # 激活环境
    source "$NEURON_ENV"
    
    # 验证环境
    if ! python3 -c "import torch_neuronx" 2>/dev/null; then
        log_error "Failed to import torch_neuronx. Environment may not be properly set up."
        exit 1
    fi
    
    # 创建环境变量说明文件
    cat << EOF > ./neuron_env_setup.txt
# Add these lines to your ~/.bashrc to set up the environment permanently:
export PATH="/opt/aws/neuron/bin:\$PATH"
export AWS_S3_BUCKET="${AWS_S3_BUCKET}"
source ${NEURON_ENV}

# You can do this by running:
# cat ./neuron_env_setup.txt >> ~/.bashrc
# source ~/.bashrc
EOF
    
    log_info "Neuron environment activated for current session"
    log_info "Please check neuron_env_setup.txt for permanent environment setup instructions"
}

# 安装系统依赖
install_system_deps() {
    log_info "Installing system dependencies..."
    
    # 更新包列表
    log_info "Updating package list..."
    sudo apt-get update
    
    # 安装必要的系统包
    PACKAGES=(
        "python3-pip"
        "python3-venv"
        "git"
        "git-lfs"
        "jq"
        "build-essential"
        "python3-dev"
    )
    
    for pkg in "${PACKAGES[@]}"; do
        if ! dpkg -l | grep -q "^ii  $pkg "; then
            log_info "Installing $pkg..."
            sudo apt-get install -y "$pkg"
        else
            log_info "$pkg already installed"
        fi
    done
    
    # 配置git-lfs
    log_info "Configuring git-lfs..."
    git lfs install
}

# 检查Docker安装
check_docker() {
    log_info "Checking Docker installation..."
    if ! docker --version > /dev/null 2>&1; then
        log_error "Docker is not installed or not running properly"
        exit 1
    else
        log_info "Found $(docker --version)"
    fi

    if ! groups | grep -q docker; then
        log_warn "Current user is not in docker group. Some docker commands may require sudo."
    fi
}

# 安装Python依赖
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # 首先安装/降级特定版本的基础包
    log_info "Installing specific versions of core packages..."
    pip install "numpy<=1.25.2,>=1.24.3" --force-reinstall
    pip install "requests<2.32.0" --force-reinstall
    
    # 安装特定版本的其他包
    PYTHON_PACKAGES=(
        "datasets==2.14.0"
        "transformers==4.45.*"
        "prometheus_client"
        "pandas"
        "huggingface-hub<0.28.0"
    )
    
    for pkg in "${PYTHON_PACKAGES[@]}"; do
        log_info "Installing $pkg..."
        pip install -U "$pkg"
    done
    
    # 验证关键包版本
    log_info "Verifying package versions..."
    python3 -c "import numpy; import requests; import datasets; print(f'numpy: {numpy.__version__}\nrequests: {requests.__version__}\ndatasets: {datasets.__version__}')" || {
        log_error "Package version verification failed"
        exit 1
    }
}

# 验证安装
verify_installation() {
    log_info "Verifying installation..."
    
    # 检查Python包
    python3 -c "import torch_neuronx; import transformers; import datasets; import huggingface_hub" || {
        log_error "Failed to import required Python packages"
        exit 1
    }
    
    # 检查docker
    docker --version || {
        log_error "Docker installation failed"
        exit 1
    }
    
    # 检查git-lfs
    git lfs version || {
        log_error "git-lfs installation failed"
        exit 1
    }
    
    # 检查neuron-ls
    neuron-ls || {
        log_error "neuron-ls not available"
        exit 1
    }
}

# S3 用于模型存储
setup_s3_storage() {
    log_info "Setting up S3 storage..."
    
    # 使用 IMDSv2 获取实例 ID
    INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")" http://169.254.169.254/latest/meta-data/instance-id)
    
    AWS_S3_BUCKET="deepseek-model-${INSTANCE_ID}"
    
    # 创建bucket
    if ! aws s3 ls "s3://${AWS_S3_BUCKET}" &>/dev/null; then
        log_info "Creating S3 bucket: ${AWS_S3_BUCKET}"
        aws s3 mb "s3://${AWS_S3_BUCKET}"
    else
        log_info "S3 bucket ${AWS_S3_BUCKET} already exists"
    fi
    
    export AWS_S3_BUCKET
}

# 添加新的函数来设置 Hugging Face
setup_huggingface() {
    log_info "Setting up Hugging Face authentication..."
    
    # 检查是否已经登录
    if huggingface-cli whoami &>/dev/null; then
        log_info "Already logged in to Hugging Face as: $(huggingface-cli whoami)"
        # 询问是否要重新登录
        read -p "Would you like to login again? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return 0
        fi
    fi
    
    # 提示用户访问 HuggingFace token 页面
    log_info "Please visit https://huggingface.co/settings/tokens to create a new token if you haven't already"
    log_info "Make sure to give it at least 'read' permissions"
    
    # 等待用户准备好
    read -p "Press Enter when you're ready to enter your token..."
    
    # 尝试登录，最多重试 3 次
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Attempt $attempt of $max_attempts to log in to Hugging Face"
        
        if huggingface-cli login; then
            log_info "Successfully logged in to Hugging Face!"
            # 验证登录
            if huggingface-cli whoami &>/dev/null; then
                log_info "Verified login as: $(huggingface-cli whoami)"
                return 0
            fi
        fi
        
        attempt=$((attempt + 1))
        
        if [ $attempt -le $max_attempts ]; then
            log_warn "Login failed. Please try again."
            read -p "Press Enter to continue..."
        fi
    done
    
    log_error "Failed to log in to Hugging Face after $max_attempts attempts"
    return 1
}

# 创建工作目录
setup_workspace() {
    log_info "Setting up workspace..."
    
    # 创建必要的目录
    mkdir -p models
    mkdir -p logs
    mkdir -p cache
    
    # 设置权限
    chmod 755 models logs cache
}

# 主函数
main() {
    log_info "Starting environment setup..."
    
    # 检查是否以root运行
    if [ "$EUID" -eq 0 ]; then 
        log_error "Please do not run this script as root"
        exit 1
    fi
    
    # 执行设置步骤
    check_inferentia
    check_docker
    install_system_deps
    install_python_deps
    setup_s3_storage      # 移到 setup_neuron_env 前面
    setup_neuron_env      # 这样 setup_neuron_env 就能使用已设置的 AWS_S3_BUCKET
    verify_installation
    setup_workspace
    setup_huggingface || log_error "Hugging Face setup failed"
    
    log_info "Environment setup completed successfully!"
    log_info "To complete the setup, please run the following commands:"
    log_info "    cat ./neuron_env_setup.txt >> ~/.bashrc"
    log_info "    source ~/.bashrc"
    log_info "You can now proceed with model deployment"
}

# 运行主函数
main "$@"

