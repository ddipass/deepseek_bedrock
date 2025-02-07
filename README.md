```markdown
# DeepSeek Model Deployment

这个项目提供了在AWS Inferentia2实例上部署和管理DeepSeek模型的完整解决方案。包括环境设置、参数优化、模型部署、性能监控和测试功能。

## 项目结构

```
deepseek-deployment/
├── config.py               # 配置管理
├── scripts/
│   ├── setup_env.sh        # 环境初始化脚本
│   ├── param_detector.py   # 参数检测脚本
│   └── monitor.py          # 监控脚本
├── deploy_deepseek.sh      # 主部署脚本
├── test_model.py           # 模型测试脚本
└── README.md               # 项目文档
```

## 前置要求

- AWS Inferentia2实例 (如: inf2.48xlarge)
- Ubuntu 22.04 LTS
- Python 3.10+
- Docker
- Hugging Face账号和token

## 环境变量设置

1. 设置Hugging Face Token:
```bash
# 临时设置
export HUGGING_FACE_TOKEN="your_token_here"

# 或永久设置（添加到 ~/.bashrc）
echo 'export HUGGING_FACE_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

2. 获取Hugging Face Token:
- 访问 https://huggingface.co/settings/tokens
- 创建新token（需要有读取权限）
- 复制token并设置环境变量

## 快速开始

1. 克隆项目：
```bash
git clone <repository_url>
cd deepseek-deployment
```

2. 初始化环境：
```bash
./scripts/setup_env.sh
```

3. 检测推荐参数：
```bash
python3 scripts/param_detector.py
```

4. 部署模型：
```bash
./deploy_deepseek.sh -t $HUGGING_FACE_TOKEN --use-detected
```

## 配置参数

主要配置参数在 `config.py` 中定义：

```python
MODEL_REPO = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # 模型仓库
MODEL_DIR = "/home/ubuntu/models"                        # 模型存储目录
MODEL_NAME = "deepseek-8b"                               # 模型名称
MODEL_PORT = 8000                                        # 服务端口

# 模型参数（可由param_detector.py自动检测）
TENSOR_PARALLEL_SIZE = 2                                 # 张量并行度
MAX_MODEL_LEN = 2048                                     # 最大模型长度
MAX_NUM_SEQS = 4                                         # 最大并行序列数
BLOCK_SIZE = 8                                           # KV缓存块大小
```

## 监控

1. Prometheus + Grafana监控：
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

2. 终端监控：
```bash
python3 scripts/monitor.py
```

## 性能测试

运行完整测试套件：
```bash
python3 test_model.py --runs 3
```

测试结果保存在 `logs/test_results/` 目录下。

## 参数调优建议

1. 内存相关：
- 如果内存使用率>90%：降低 `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `BLOCK_SIZE`
- 如果内存使用率<50%：可以增加上述参数

2. 性能相关：
- 首字符延迟高：增加 `TENSOR_PARALLEL_SIZE`
- 吞吐量低：增加 `MAX_NUM_SEQS`, `BLOCK_SIZE`

## 目录说明

- `logs/`: 日志文件
- `models/`: 模型文件
- `configs/`: 配置文件
  - `recommended_params.json`: 推荐参数
  - `current_config.json`: 当前配置

## 常见问题

1. 内存不足：
```bash
# 降低参数
python3 scripts/param_detector.py --conservative
```

2. 性能问题：
```bash
# 查看监控
python3 scripts/monitor.py
```

3. Docker权限：
```bash
sudo usermod -aG docker $USER
# 需要重新登录
```

## 注意事项

1. 首次运行需要下载模型，确保有足够磁盘空间
2. 建议先使用参数检测功能获取推荐配置
3. 修改参数后需要重启服务生效
4. 监控数据会占用磁盘空间，定期清理


## 更新日志

- v1.0.0: 初始版本
```

这个README提供了：
1. 清晰的项目结构
2. 详细的安装和使用说明
3. 配置参数的说明
4. 常见问题的解决方案
5. 参数调优建议

您可以根据实际情况修改和补充内容。