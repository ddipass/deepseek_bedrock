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
    if ! lspci | grep -q "Device 1d0f"; then
        log_error "This script requires an AWS Inferentia2 instance"
        exit 1
    fi
    log_info "Inferentia2 device found"
}

# 检查并激活Neuron环境
setup_neuron_env() {
    log_info "Setting up Neuron environment..."
    
    # 检查Neuron环境是否存在
    NEURON_ENV="/opt/aws_neuronx_venv_pytorch_2_5_nxd_inference/bin/activate"
    if [ ! -f "$NEURON_ENV" ]; then
        log_error "Neuron environment not found at $NEURON_ENV"
        exit 1
    }
    
    # 激活环境
    source "$NEURON_ENV"
    
    # 验证环境
    if ! python3 -c "import torch_neuronx" 2>/dev/null; then
        log_error "Failed to import torch_neuronx. Environment may not be properly set up."
        exit 1
    }
    
    log_info "Neuron environment activated successfully"
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
        "docker.io"
        "docker-compose"
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
    
    # 配置docker权限
    if ! groups | grep -q docker; then
        log_info "Adding user to docker group..."
        sudo usermod -aG docker $USER
        log_warn "Please log out and log back in for docker permissions to take effect"
    fi
}

# 安装Python依赖
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # 安装/升级pip
    python3 -m pip install --upgrade pip
    
    # 安装必要的Python包
    PYTHON_PACKAGES=(
        "requests"
        "datasets"
        "transformers"
        "prometheus_client"
        "numpy"
        "pandas"
        "huggingface_hub"
    )
    
    for pkg in "${PYTHON_PACKAGES[@]}"; do
        log_info "Installing $pkg..."
        pip install -U "$pkg"
    done
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
    }
    
    # 执行设置步骤
    check_inferentia
    setup_neuron_env
    install_system_deps
    install_python_deps
    verify_installation
    setup_workspace
    
    log_info "Environment setup completed successfully!"
    log_info "Please log out and log back in if docker group was added"
    log_info "You can now proceed with model deployment"
}

# 运行主函数
main "$@"