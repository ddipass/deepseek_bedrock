#!/usr/bin/env python3
# param_detector.py

import json
import subprocess
import psutil
import os, sys
import logging
from typing import Dict, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.LOG_DIR}/param_detector.log'),
        logging.StreamHandler()  # 添加这行来同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

class ResourceDetector:
    """资源检测类"""
    
    def __init__(self):
        self.system_resources = self._detect_system_resources()
    
    def _detect_system_resources(self) -> Dict[str, Any]:
        """检测系统资源"""
        try:
            # 获取CPU和内存信息
            cpu_count = psutil.cpu_count(logical=False)
            mem = psutil.virtual_memory()
            
            # 获取Neuron设备信息
            neuron_devices = []
            try:
                # 执行 neuron-ls 命令
                output = subprocess.check_output(['neuron-ls']).decode()
                
                # 解析输出前先打印看看完整输出，便于调试
                logger.info(f"Raw neuron-ls output:\n{output}")
                
                # 解析输出
                lines = output.split('\n')
                device_found = False
                for line in lines:
                    if '|' in line and any(c.isdigit() for c in line):  # 修改判断条件
                        parts = [p.strip() for p in line.split('|') if p.strip()]
                        if len(parts) >= 3:
                            try:
                                device_id = parts[0]
                                nc_count = int(parts[1])  # 核心数
                                # 处理形如 "32 GB" 的内存字符串
                                memory_str = parts[2].split()[0]  # 取第一个数字
                                memory = int(memory_str) * 1024 * 1024 * 1024  # GB转字节
                                
                                neuron_devices.append({
                                    'device_id': device_id,
                                    'memory_total': memory,
                                    'memory_available': memory,  # 假设全部可用
                                    'nc_count': nc_count
                                })
                                device_found = True
                            except (ValueError, IndexError) as e:
                                logger.error(f"Error parsing line '{line}': {e}")
                                continue
                
                if not device_found:
                    logger.warning("No Neuron devices found in the output")
                    
            except Exception as e:
                logger.error(f"Failed to get Neuron device info: {e}")
            
            return {
                'cpu_count': cpu_count,
                'memory_total': mem.total,
                'memory_available': mem.available,
                'neuron_devices': neuron_devices
            }
        except Exception as e:
            logger.error(f"Failed to detect system resources: {e}")
            sys.exit(1)

    def _calculate_tensor_parallel_size(self) -> int:
        """计算张量并行度"""
        device_count = len(self.system_resources['neuron_devices'])
        if device_count >= 8:  # inf2.48xlarge
            return 8  # 使用8个设备以获得最佳性能平衡
        elif device_count >= 4:
            return 4
        return 2

    def _calculate_block_size(self) -> int:
        """计算KV缓存块大小"""
        total_memory_gb = sum(d.get('memory_total', 0) for d in self.system_resources['neuron_devices']) / (1024**3)
        
        if total_memory_gb >= 384:  # inf2.48xlarge (12 * 32GB = 384GB)
            return 32
        elif total_memory_gb >= 256:
            return 24
        elif total_memory_gb >= 128:
            return 16
        return 8

    def _calculate_max_num_seqs(self) -> int:
        """计算最大并行序列数"""
        total_memory_gb = sum(d.get('memory_total', 0) for d in self.system_resources['neuron_devices']) / (1024**3)
        
        if total_memory_gb >= 384:  # inf2.48xlarge
            return 16  # 更保守的值以确保稳定性
        elif total_memory_gb >= 256:
            return 12
        elif total_memory_gb >= 128:
            return 8
        return 4

    def _calculate_max_model_len(self) -> int:
        """计算最大模型长度"""
        total_memory_gb = sum(d.get('memory_total', 0) for d in self.system_resources['neuron_devices']) / (1024**3)
        
        if total_memory_gb >= 384:  # inf2.48xlarge
            return 8192
        elif total_memory_gb >= 256:
            return 6144
        elif total_memory_gb >= 128:
            return 4096
        return 2048

    def get_recommended_params(self) -> Dict[str, Any]:
        """获取推荐参数"""
        params = {
            'tensor_parallel_size': self._calculate_tensor_parallel_size(),
            'block_size': self._calculate_block_size(),
            'max_num_seqs': self._calculate_max_num_seqs(),
            'max_model_len': self._calculate_max_model_len(),
            'temperature': Config.TEMPERATURE,
            'top_p': Config.TOP_P
        }
        return params

    def print_system_info(self):
        """打印系统信息"""
        logger.info("System Resources:")
        logger.info(f"CPU Cores: {self.system_resources['cpu_count']}")
        logger.info(f"Total Memory: {self.system_resources['memory_total'] / (1024**3):.2f} GB")
        logger.info(f"Available Memory: {self.system_resources['memory_available'] / (1024**3):.2f} GB")
        
        logger.info("\nNeuron Devices:")
        for device in self.system_resources['neuron_devices']:
            logger.info(f"Device ID: {device['device_id']}")
            logger.info(f"Total Memory: {device['memory_total'] / (1024**3):.2f} GB")
            logger.info(f"Available Memory: {device['memory_available'] / (1024**3):.2f} GB")
            logger.info(f"Neuron Cores: {device['nc_count']}")

def main():
    """主函数"""
    try:
        detector = ResourceDetector()
        
        # 打印系统信息
        detector.print_system_info()
        
        # 获取推荐参数
        params = detector.get_recommended_params()
        
        # 更新Config中的参数
        Config.update_params(params)
        
        # 保存推荐参数到文件
        with open(f'{Config.CONFIG_DIR}/recommended_params.json', 'w') as f:
            json.dump(params, f, indent=2)
            
        # 保存当前配置
        Config.save_current_config()
        
        logger.info("\nRecommended Parameters:")
        logger.info(json.dumps(params, indent=2))
        
        return params
        
    except Exception as e:
        logger.error(f"Error in parameter detection: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()