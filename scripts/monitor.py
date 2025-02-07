#!/usr/bin/env python3
# monitor.py

import curses
import json
import time
import requests
import subprocess
from datetime import datetime
from typing import Dict, Any
import logging
from pathlib import Path
from config import Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'{Config.LOG_DIR}/monitor.log'
)
logger = logging.getLogger(__name__)

class DeepSeekMonitor:
    def __init__(self):
        self.last_token_count = 0
        self.last_request_count = 0
        self.last_check_time = time.time()

    def get_neuron_metrics(self) -> Dict[str, Any]:
        """获取Neuron设备指标"""
        try:
            result = subprocess.run(['neuron-ls', '--json'], 
                                 capture_output=True, text=True)
            data = json.loads(result.stdout)
            
            metrics = {
                'devices': [],
                'total_memory': 0,
                'used_memory': 0,
                'total_cores': 0,
                'active_cores': 0
            }
            
            for device in data.get('neuron_devices', []):
                device_metrics = {
                    'id': device.get('device_id'),
                    'memory_total': device.get('memory', 0) / (1024**3),  # GB
                    'memory_used': device.get('memory_used', 0) / (1024**3),  # GB
                    'cores': device.get('nc_count', 0),
                    'utilization': device.get('nc_utilization', 0)
                }
                metrics['devices'].append(device_metrics)
                metrics['total_memory'] += device_metrics['memory_total']
                metrics['used_memory'] += device_metrics['memory_used']
                metrics['total_cores'] += device_metrics['cores']
                metrics['active_cores'] += (device_metrics['cores'] * 
                                         device_metrics['utilization'] / 100)
            
            return metrics
        except Exception as e:
            logger.error(f"Error getting Neuron metrics: {e}")
            return {}

    def get_vllm_metrics(self) -> Dict[str, Any]:
        """获取vLLM性能指标"""
        try:
            response = requests.get(f'http://localhost:{Config.MODEL_PORT}/metrics')
            if response.status_code != 200:
                return {}
            
            metrics = {}
            current_time = time.time()
            time_diff = current_time - self.last_check_time
            
            for line in response.text.split('\n'):
                if line.startswith('#'):
                    continue
                    
                if 'time_to_first_token_seconds_sum' in line:
                    metrics['first_token_latency'] = float(line.split()[-1])
                elif 'time_per_output_token_seconds_sum' in line:
                    metrics['token_latency'] = float(line.split()[-1])
                elif 'generation_tokens_total' in line:
                    current_tokens = float(line.split()[-1])
                    metrics['token_throughput'] = (current_tokens - self.last_token_count) / time_diff
                    self.last_token_count = current_tokens
                elif 'request_success_total' in line:
                    current_requests = float(line.split()[-1])
                    metrics['requests_per_second'] = (current_requests - self.last_request_count) / time_diff
                    self.last_request_count = current_requests
                elif 'gpu_cache_usage_perc' in line:
                    metrics['cache_usage'] = float(line.split()[-1]) * 100
            
            self.last_check_time = current_time
            return metrics
        except Exception as e:
            logger.error(f"Error getting vLLM metrics: {e}")
            return {}

    def get_parameter_recommendations(self, neuron_metrics: Dict, vllm_metrics: Dict) -> Dict[str, Any]:
        """基于当前性能指标提供参数调整建议"""
        recommendations = {}
        
        # 内存使用率
        memory_usage = neuron_metrics.get('used_memory', 0) / neuron_metrics.get('total_memory', 1)
        
        # 根据内存使用情况调整参数
        if memory_usage > 0.9:
            recommendations['memory_high'] = [
                f"降低 max_model_len (当前: {Config.MAX_MODEL_LEN})",
                f"减少 max_num_seqs (当前: {Config.MAX_NUM_SEQS})",
                f"减小 block_size (当前: {Config.BLOCK_SIZE})"
            ]
        elif memory_usage < 0.5:
            recommendations['memory_low'] = [
                f"可以增加 max_model_len (当前: {Config.MAX_MODEL_LEN})",
                f"可以增加 max_num_seqs (当前: {Config.MAX_NUM_SEQS})",
                f"可以增大 block_size (当前: {Config.BLOCK_SIZE})"
            ]
        
        # 根据延迟情况调整参数
        if vllm_metrics.get('first_token_latency', 0) > 1.0:
            recommendations['latency_high'] = [
                f"增加 tensor_parallel_size (当前: {Config.TENSOR_PARALLEL_SIZE})",
                f"减少 max_num_seqs (当前: {Config.MAX_NUM_SEQS})"
            ]
        
        # 根据吞吐量调整参数
        if vllm_metrics.get('token_throughput', 0) < 10:
            recommendations['throughput_low'] = [
                f"增加 max_num_seqs (当前: {Config.MAX_NUM_SEQS})",
                f"增大 block_size (当前: {Config.BLOCK_SIZE})",
                f"检查 tensor_parallel_size (当前: {Config.TENSOR_PARALLEL_SIZE})"
            ]
        
        return recommendations

    def display(self, stdscr):
        """显示监控信息"""
        while True:
            try:
                stdscr.clear()
                
                # 获取指标
                neuron_metrics = self.get_neuron_metrics()
                vllm_metrics = self.get_vllm_metrics()
                recommendations = self.get_parameter_recommendations(neuron_metrics, vllm_metrics)
                
                # 显示标题
                stdscr.addstr(0, 0, "DeepSeek Model Monitor", curses.A_BOLD)
                stdscr.addstr(1, 0, "=" * 50)
                
                # 显示当前时间
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                stdscr.addstr(2, 0, f"Current Time: {current_time}")
                
                # 显示Neuron设备信息
                row = 4
                stdscr.addstr(row, 0, "Neuron Device Status:", curses.A_BOLD)
                row += 1
                for device in neuron_metrics.get('devices', []):
                    stdscr.addstr(row, 2, 
                        f"Device {device['id']}: "
                        f"{device['memory_used']:.1f}/{device['memory_total']:.1f}GB "
                        f"({device['utilization']:.1f}% util)")
                    row += 1
                
                # 显示vLLM指标
                row += 1
                stdscr.addstr(row, 0, "Performance Metrics:", curses.A_BOLD)
                row += 1
                stdscr.addstr(row, 2, 
                    f"First Token Latency: {vllm_metrics.get('first_token_latency', 0):.3f}s")
                row += 1
                stdscr.addstr(row, 2, 
                    f"Token Throughput: {vllm_metrics.get('token_throughput', 0):.1f} tokens/s")
                row += 1
                stdscr.addstr(row, 2, 
                    f"Requests/s: {vllm_metrics.get('requests_per_second', 0):.2f}")
                row += 1
                stdscr.addstr(row, 2, 
                    f"Cache Usage: {vllm_metrics.get('cache_usage', 0):.1f}%")
                
                # 显示当前参数
                row += 2
                stdscr.addstr(row, 0, "Current Parameters:", curses.A_BOLD)
                row += 1
                stdscr.addstr(row, 2, f"tensor_parallel_size: {Config.TENSOR_PARALLEL_SIZE}")
                row += 1
                stdscr.addstr(row, 2, f"max_model_len: {Config.MAX_MODEL_LEN}")
                row += 1
                stdscr.addstr(row, 2, f"max_num_seqs: {Config.MAX_NUM_SEQS}")
                row += 1
                stdscr.addstr(row, 2, f"block_size: {Config.BLOCK_SIZE}")
                
                # 显示参数调整建议
                if recommendations:
                    row += 2
                    stdscr.addstr(row, 0, "Parameter Recommendations:", curses.A_BOLD)
                    row += 1
                    for category, items in recommendations.items():
                        for item in items:
                            stdscr.addstr(row, 2, f"• {item}")
                            row += 1
                
                # 显示监控界面信息
                row += 2
                stdscr.addstr(row, 0, "Monitoring Dashboards:", curses.A_BOLD)
                row += 1
                stdscr.addstr(row, 2, f"Prometheus: http://localhost:{Config.PROMETHEUS_PORT}")
                row += 1
                stdscr.addstr(row, 2, f"Grafana: http://localhost:{Config.GRAFANA_PORT}")
                
                # 显示控制信息
                row += 2
                stdscr.addstr(row, 0, "Press 'q' to quit, 'r' to refresh")
                
                # 刷新屏幕
                stdscr.refresh()
                
                # 检查用户输入
                c = stdscr.getch()
                if c == ord('q'):
                    break
                
                time.sleep(Config.MONITOR_INTERVAL)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in monitor display: {e}")
                stdscr.addstr(0, 0, f"Error: {str(e)}")
                stdscr.refresh()
                time.sleep(5)

def main():
    monitor = DeepSeekMonitor()
    curses.wrapper(monitor.display)

if __name__ == "__main__":
    main()