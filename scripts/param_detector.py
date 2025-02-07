#!/usr/bin/env python3
# param_detector.py

import json
import subprocess
import psutil
import sys
import logging
from typing import Dict, Any
from config import Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'{Config.LOG_DIR}/param_detector.log'
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
                neuron_info = json.loads(subprocess.check_output(['neuron-ls', '--json']).decode())
                for device in neuron_info.get('neuron_devices', []):
                    neuron_devices.append({
                        'device_id': device.get('device_id', ''),
                        'memory_total': device.get('memory', 0),
                        'memory_available': device.get('memory_available', 0),
                        'nc_count': device.get('nc_count', 0)
                    })
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
        total_cores = sum(device.get('nc_count', 0) for device in self.system_resources['neuron_devices'])
        if total_cores >= 8:
            return 8
        elif total_cores >= 4:
            return 4
        elif total_cores >= 2:
            return 2
        return 1

    def _calculate_block_size(self) -> int:
        """计算KV缓存块大小"""
        total_memory_gb = sum(d.get('memory_total', 0) for d in self.system_resources['neuron_devices']) / (1024**3)
        
        if total_memory_gb >= 32:
            return 32
        elif total_memory_gb >= 16:
            return 16
        elif total_memory_gb >= 8:
            return 8
        return 4

    def _calculate_max_num_seqs(self) -> int:
        """计算最大并行序列数"""
        total_memory_gb = sum(d.get('memory_total', 0) for d in self.system_resources['neuron_devices']) / (1024**3)
        
        if total_memory_gb >= 32:
            return 16
        elif total_memory_gb >= 16:
            return 8
        elif total_memory_gb >= 8:
            return 4
        return 2

    def _calculate_max_model_len(self) -> int:
        """计算最大模型长度"""
        total_memory_gb = sum(d.get('memory_total', 0) for d in self.system_resources['neuron_devices']) / (1024**3)
        
        if total_memory_gb >= 32:
            return 8192
        elif total_memory_gb >= 16:
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