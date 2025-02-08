import os
from dataclasses import dataclass
from typing import Dict, Any
import json
from pathlib import Path

@dataclass
class Config:
    """DeepSeek 部署配置"""
    
    # 模型配置
    # MODEL_REPO: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    MODEL_REPO: str = "deepseek-ai/DeepSeek-R1"
    MODEL_DIR: str = "/home/ubuntu/models"
    MODEL_NAME: str = "deepseek-685b"
    
    # API配置
    HF_TOKEN: str = os.getenv("HUGGING_FACE_TOKEN")
    MODEL_PORT: int = 8000
    
    # 模型参数
    TENSOR_PARALLEL_SIZE: int = 2
    MAX_MODEL_LEN: int = 2048
    MAX_NUM_SEQS: int = 4
    BLOCK_SIZE: int = 8
    TEMPERATURE: float = 0.6
    TOP_P: float = 0.95
    
    # 监控配置
    PROMETHEUS_PORT: int = 9090
    GRAFANA_PORT: int = 3000
    MONITOR_INTERVAL: int = 2  # 监控更新间隔（秒）
    
    # 路径配置
    LOG_DIR: str = "logs"
    CACHE_DIR: str = "cache"
    CONFIG_DIR: str = "configs"
    
    @classmethod
    def load_recommended_params(cls) -> Dict[str, Any]:
        """从推荐参数文件加载参数"""
        try:
            with open(f"{cls.CONFIG_DIR}/recommended_params.json", 'r') as f:
                params = json.load(f)
            return params
        except FileNotFoundError:
            return {
                'tensor_parallel_size': cls.TENSOR_PARALLEL_SIZE,
                'max_model_len': cls.MAX_MODEL_LEN,
                'max_num_seqs': cls.MAX_NUM_SEQS,
                'block_size': cls.BLOCK_SIZE
            }
    
    @classmethod
    def update_params(cls, new_params: Dict[str, Any]):
        """更新动态参数"""
        cls.TENSOR_PARALLEL_SIZE = new_params.get('tensor_parallel_size', cls.TENSOR_PARALLEL_SIZE)
        cls.MAX_MODEL_LEN = new_params.get('max_model_len', cls.MAX_MODEL_LEN)
        cls.MAX_NUM_SEQS = new_params.get('max_num_seqs', cls.MAX_NUM_SEQS)
        cls.BLOCK_SIZE = new_params.get('block_size', cls.BLOCK_SIZE)
    
    @classmethod
    def save_current_config(cls):
        """保存当前配置"""
        config = {
            'model_repo': cls.MODEL_REPO,
            'model_dir': cls.MODEL_DIR,
            'model_name': cls.MODEL_NAME,
            'tensor_parallel_size': cls.TENSOR_PARALLEL_SIZE,
            'max_model_len': cls.MAX_MODEL_LEN,
            'max_num_seqs': cls.MAX_NUM_SEQS,
            'block_size': cls.BLOCK_SIZE,
            'temperature': cls.TEMPERATURE,
            'top_p': cls.TOP_P,
            'model_port': cls.MODEL_PORT,
            'prometheus_port': cls.PROMETHEUS_PORT,
            'grafana_port': cls.GRAFANA_PORT
        }
        
        os.makedirs(cls.CONFIG_DIR, exist_ok=True)
        with open(f"{cls.CONFIG_DIR}/current_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        for directory in [cls.LOG_DIR, cls.CACHE_DIR, cls.CONFIG_DIR]:
            Path(directory).mkdir(exist_ok=True)
    
    @classmethod
    def to_shell_exports(cls) -> str:
        """生成shell环境变量导出命令"""
        exports = [
            f"export MODEL_REPO='{cls.MODEL_REPO}'",
            f"export MODEL_DIR='{cls.MODEL_DIR}'",
            f"export MODEL_NAME='{cls.MODEL_NAME}'",
            f"export MODEL_PORT={cls.MODEL_PORT}",
            f"export TENSOR_PARALLEL_SIZE={cls.TENSOR_PARALLEL_SIZE}",
            f"export MAX_MODEL_LEN={cls.MAX_MODEL_LEN}",
            f"export MAX_NUM_SEQS={cls.MAX_NUM_SEQS}",
            f"export BLOCK_SIZE={cls.BLOCK_SIZE}",
            f"export PROMETHEUS_PORT={cls.PROMETHEUS_PORT}",
            f"export GRAFANA_PORT={cls.GRAFANA_PORT}"
        ]
        return "\n".join(exports)

    @classmethod
    def get_vllm_args(cls) -> Dict[str, Any]:
        """获取vLLM参数"""
        return {
            'model': cls.MODEL_REPO,
            'tensor_parallel_size': cls.TENSOR_PARALLEL_SIZE,
            'max_model_len': cls.MAX_MODEL_LEN,
            'max_num_seqs': cls.MAX_NUM_SEQS,
            'block_size': cls.BLOCK_SIZE,
            'port': cls.MODEL_PORT
        }

# 在导入时创建必要的目录
Config.create_directories()