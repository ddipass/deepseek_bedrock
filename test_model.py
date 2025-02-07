#!/usr/bin/env python3
# test_model.py

import asyncio
import json
import time
import logging
import argparse
import statistics
from typing import Dict, List, Any
import requests
from datetime import datetime
import pandas as pd
from pathlib import Path
from config import Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.LOG_DIR}/test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepSeekTester:
    def __init__(self, endpoint: str = None):
        self.endpoint = endpoint or f"http://localhost:{Config.MODEL_PORT}"
        self.results_dir = Path(f"{Config.LOG_DIR}/test_results")
        self.results_dir.mkdir(exist_ok=True)

    def get_test_cases(self) -> List[Dict[str, Any]]:
        """定义不同长度和类型的测试用例"""
        return [
            # 短文本测试 (< 100 tokens)
            {
                "category": "short",
                "name": "Basic Math",
                "prompt": "What is 2+2? Explain your reasoning.",
                "expected_length": "short"
            },
            {
                "category": "short",
                "name": "Simple Definition",
                "prompt": "Define what is a neural network.",
                "expected_length": "short"
            },
            
            # 中等长度测试 (100-500 tokens)
            {
                "category": "medium",
                "name": "Code Generation",
                "prompt": """Write a Python function that implements bubble sort. 
                Include comments explaining how it works and provide an example usage.""",
                "expected_length": "medium"
            },
            {
                "category": "medium",
                "name": "Complex Math",
                "prompt": """Solve this calculus problem:
                Find the derivative of f(x) = x^3 * sin(x).
                Show your step-by-step solution using the product rule.""",
                "expected_length": "medium"
            },
            
            # 长文本测试 (500+ tokens)
            {
                "category": "long",
                "name": "Essay Generation",
                "prompt": """Write a comprehensive essay about the impact of artificial 
                intelligence on society. Include sections about economic impact, 
                ethical considerations, and future predictions. Provide specific 
                examples and cite relevant research.""",
                "expected_length": "long"
            },
            {
                "category": "long",
                "name": "Technical Analysis",
                "prompt": """Provide a detailed technical analysis of how transformer 
                models work. Include explanations of self-attention, multi-head attention, 
                positional encoding, and the overall architecture. Use examples to 
                illustrate key concepts.""",
                "expected_length": "long"
            },
            
            # 特殊测试
            {
                "category": "special",
                "name": "Context Window Test",
                "prompt": "A" * 1000 + "\nSummarize this text.",  # 测试上下文窗口
                "expected_length": "medium"
            },
            {
                "category": "special",
                "name": "Multi-turn Dialogue",
                "prompt": """User: Hi, I'd like to learn about quantum computing.
                Assistant: I'll help you understand quantum computing. What specific aspects would you like to know about?
                User: Can you explain qubits?
                Assistant: I'll explain qubits in detail. Would you like me to start with the basics or go into more advanced concepts?
                User: Start with the basics please.""",
                "expected_length": "medium"
            }
        ]

    async def test_completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """测试单个完成请求"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.endpoint}/v1/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": Config.MODEL_NAME,
                    "prompt": prompt,
                    "temperature": kwargs.get('temperature', Config.TEMPERATURE),
                    "max_tokens": kwargs.get('max_tokens', Config.MAX_MODEL_LEN),
                    "top_p": kwargs.get('top_p', Config.TOP_P)
                },
                timeout=30
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "latency": elapsed,
                    "first_token_time": result.get("first_token_time", None),
                    "generation_time": result.get("generation_time", None),
                    "output_length": len(result["choices"][0]["text"].split()),
                    "output": result["choices"][0]["text"]
                }
            else:
                return {
                    "success": False,
                    "latency": elapsed,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "latency": time.time() - start_time,
                "error": str(e)
            }

    async def run_test_suite(self, num_runs: int = 3):
        """运行完整的测试套件"""
        test_cases = self.get_test_cases()
        all_results = []
        
        logger.info(f"Starting test suite with {len(test_cases)} test cases, {num_runs} runs each")
        logger.info(f"Current configuration: {Config.get_vllm_args()}")
        
        for test_case in test_cases:
            logger.info(f"\nTesting {test_case['name']} ({test_case['category']})...")
            
            case_results = []
            for run in range(num_runs):
                logger.info(f"Run {run + 1}/{num_runs}")
                
                result = await self.test_completion(
                    test_case["prompt"],
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_MODEL_LEN if test_case["expected_length"] != "long" else Config.MAX_MODEL_LEN * 2
                )
                
                result.update({
                    "test_name": test_case["name"],
                    "category": test_case["category"],
                    "expected_length": test_case["expected_length"],
                    "run": run + 1
                })
                
                case_results.append(result)
                
                if result["success"]:
                    logger.info(f"Latency: {result['latency']:.2f}s, "
                              f"Output length: {result['output_length']} words")
                else:
                    logger.error(f"Failed: {result['error']}")
                
                await asyncio.sleep(1)
            
            all_results.extend(case_results)
        
        return all_results

    def analyze_results(self, results: List[Dict[str, Any]]):
        """分析测试结果"""
        df = pd.DataFrame(results)
        
        # 按类别统计
        category_stats = df.groupby('category').agg({
            'latency': ['mean', 'std', 'min', 'max'],
            'success': 'mean',
            'output_length': ['mean', 'std']
        }).round(2)
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存结果
        results_path = self.results_dir / timestamp
        results_path.mkdir(exist_ok=True)
        
        # 保存原始结果
        df.to_csv(results_path / "raw_results.csv")
        
        # 保存统计结果
        category_stats.to_csv(results_path / "category_stats.csv")
        
        # 生成报告
        report = (
            f"Test Results Summary ({timestamp})\n"
            f"================================\n\n"
            f"Model Configuration:\n"
            f"{json.dumps(Config.get_vllm_args(), indent=2)}\n\n"
            f"Category Statistics:\n{category_stats.to_string()}\n\n"
            f"Success Rate: {df['success'].mean()*100:.1f}%\n"
            f"Average Latency: {df['latency'].mean():.2f}s\n"
            f"Average Output Length: {df['output_length'].mean():.1f} words\n"
        )
        
        # 保存报告
        with open(results_path / "test_report.txt", 'w') as f:
            f.write(report)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Test DeepSeek model deployment')
    parser.add_argument('--endpoint', 
                      help=f'Model endpoint URL (default: http://localhost:{Config.MODEL_PORT})')
    parser.add_argument('--runs', type=int, default=3,
                      help='Number of runs for each test case')
    args = parser.parse_args()
    
    tester = DeepSeekTester(args.endpoint)
    
    # 运行测试
    logger.info("Starting tests...")
    results = asyncio.run(tester.run_test_suite(args.runs))
    
    # 分析结果
    logger.info("\nAnalyzing results...")
    report = tester.analyze_results(results)
    
    # 打印报告
    print("\nTest Report:")
    print(report)

if __name__ == "__main__":
    main()
