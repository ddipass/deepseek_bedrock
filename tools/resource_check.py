#!/usr/bin/env python3
import boto3
import requests
import json
import os
import subprocess
from typing import Dict, List
from tabulate import tabulate
from datetime import datetime
from colorama import init, Fore, Style

init()  # 初始化colorama

class AWSEnvironmentChecker:
    def __init__(self):
        self.region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "aws_services": {},
            "inf2_info": {},
            "quotas": {},
            "huggingface": {},
            "system": {}
        }

    def print_header(self, message: str):
        """打印格式化的标题"""
        print(f"\n{Fore.BLUE}{'='*20} {message} {'='*20}{Style.RESET_ALL}")

    def print_result(self, message: str, status: bool, details: str = ""):
        """打印检查结果"""
        color = Fore.GREEN if status else Fore.RED
        symbol = "✓" if status else "✗"
        print(f"{color}{symbol}{Style.RESET_ALL} {message}{': ' + details if details else ''}")

    def check_aws_services(self):
        """检查AWS服务可用性"""
        self.print_header("AWS服务检查")
        services = {
            "ec2": "EC2服务",
            "bedrock": "Bedrock服务",
            "cloudwatch": "CloudWatch服务",
            "iam": "IAM服务",
            "sagemaker": "SageMaker服务",
            "service-quotas": "Service Quotas服务"
        }
        
        for service, display_name in services.items():
            try:
                client = boto3.client(service, region_name=self.region)
                client._client_config
                self.print_result(display_name, True, "可以访问")
                self.results["aws_services"][service] = True
            except Exception as e:
                self.print_result(display_name, False, str(e))
                self.results["aws_services"][service] = False

    def get_inf2_quotas(self):
        """获取Inf2相关的配额信息"""
        self.print_header("Inf2配额检查")
        
        try:
            quotas = boto3.client('service-quotas')
            ec2 = boto3.client('ec2')
            
            quota_data = []
            
            # 获取所有Inf2相关配额
            try:
                # 运行中的On-Demand Inf实例配额
                response = quotas.get_service_quota(
                    ServiceCode='ec2',
                    QuotaCode='L-1945791B'
                )
                quota_data.append([
                    "Running On-Demand Inf instances",
                    str(response['Quota']['Value'])
                ])

                # 获取vCPU配额
                vcpu_response = quotas.get_service_quota(
                    ServiceCode='ec2',
                    QuotaCode='L-34B43A08'  # vCPU limit for Inf instances
                )
                quota_data.append([
                    "Inf Instance vCPUs",
                    str(vcpu_response['Quota']['Value'])
                ])

                # 获取具体实例类型的可用数量
                instance_types = ['inf2.xlarge', 'inf2.8xlarge', 'inf2.24xlarge', 'inf2.48xlarge']
                for instance_type in instance_types:
                    response = ec2.describe_instance_types(
                        InstanceTypes=[instance_type]
                    )
                    if response['InstanceTypes']:
                        inst = response['InstanceTypes'][0]
                        # 检查可用区域中的容量
                        availability = ec2.describe_instance_type_offerings(
                            LocationType='region',
                            Filters=[{'Name': 'instance-type', 'Values': [instance_type]}]
                        )
                        is_available = len(availability['InstanceTypeOfferings']) > 0
                        status = "可用" if is_available else "不可用"
                        quota_data.append([
                            f"{instance_type}",
                            f"状态: {status}, vCPU: {inst['VCpuInfo']['DefaultVCpus']}"
                        ])

            except Exception as e:
                quota_data.append(["EC2 Quotas", f"错误: {str(e)}"])

            # 检查SageMaker资源限制
            try:
                sagemaker = boto3.client('sagemaker')
                for instance_type in ['ml.inf2.xlarge', 'ml.inf2.8xlarge', 'ml.inf2.24xlarge', 'ml.inf2.48xlarge']:
                    try:
                        response = sagemaker.list_training_jobs(
                            StatusEquals='InProgress',
                            MaxResults=1
                        )
                        quota_data.append([
                            f"SageMaker {instance_type}",
                            "可用于训练作业"
                        ])
                    except Exception as e:
                        quota_data.append([
                            f"SageMaker {instance_type}",
                            f"状态检查失败: {str(e)}"
                        ])
            except Exception as e:
                quota_data.append(["SageMaker Limits", f"错误: {str(e)}"])

            print(tabulate(quota_data, headers=["配额类型", "限制"], tablefmt="grid"))
            self.results["quotas"] = quota_data

        except Exception as e:
            print(f"{Fore.RED}获取配额信息失败: {str(e)}{Style.RESET_ALL}")

    def check_inf2_availability(self):
        """检查Inf2实例的可用性和详细信息"""
        self.print_header("Inf2实例详细信息")
        
        try:
            ec2 = boto3.client('ec2')
            pricing = boto3.client('pricing', region_name='us-east-1')

            instance_data = []
            
            # 获取所有Inf2实例类型
            response = ec2.describe_instance_types(
                Filters=[{'Name': 'instance-type', 'Values': ['inf2.*']}]
            )

            for instance in response['InstanceTypes']:
                instance_type = instance['InstanceType']
                
                # 检查可用性
                availability = ec2.describe_instance_type_offerings(
                    LocationType='region',
                    Filters=[{'Name': 'instance-type', 'Values': [instance_type]}]
                )
                is_available = len(availability['InstanceTypeOfferings']) > 0

                # 获取价格
                try:
                    price_response = pricing.get_products(
                        ServiceCode='AmazonEC2',
                        Filters=[
                            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                            {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': f'US East ({self.region})'},
                            {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'}
                        ]
                    )
                    if price_response['PriceList']:
                        price_details = json.loads(price_response['PriceList'][0])
                        on_demand = list(price_details['terms']['OnDemand'].values())[0]
                        price = list(on_demand['priceDimensions'].values())[0]['pricePerUnit']['USD']
                    else:
                        price = 'N/A'
                except Exception:
                    price = 'N/A'

                instance_data.append([
                    f"{Fore.GREEN}✓{Style.RESET_ALL}" if is_available else f"{Fore.RED}✗{Style.RESET_ALL}",
                    instance_type,
                    str(instance['VCpuInfo']['DefaultVCpus']),
                    f"{instance['MemoryInfo']['SizeInMiB']/1024:.1f} GiB",
                    instance.get('NetworkInfo', {}).get('NetworkPerformance', 'N/A'),
                    f"${price}/hour" if price != 'N/A' else 'N/A'
                ])

            print(tabulate(instance_data, 
                         headers=["状态", "实例类型", "vCPU", "内存", "网络性能", "按需价格"],
                         tablefmt="grid"))
            
            self.results["inf2_info"] = instance_data

        except Exception as e:
            print(f"{Fore.RED}获取Inf2实例信息失败: {str(e)}{Style.RESET_ALL}")

    def check_huggingface_env(self):
        """检查HuggingFace环境"""
        self.print_header("HuggingFace环境检查")
        
        # 检查HF_TOKEN
        hf_token = os.environ.get('HF_TOKEN')
        self.print_result(
            "HuggingFace Token配置",
            hf_token is not None,
            "未设置 (运行: export HF_TOKEN='your_token')" if not hf_token else "已设置"
        )
        
        # 检查必要的包
        packages = {
            "huggingface_hub": "huggingface_hub",
            "transformers": "transformers",
            "torch": "torch"
        }
        
        install_commands = []
        for package_name, import_name in packages.items():
            try:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
                self.print_result(f"{package_name}版本", True, version)
            except ImportError:
                self.print_result(f"{package_name}", False, "未安装")
                install_commands.append(f"pip install {package_name}")
        
        if install_commands:
            print(f"\n{Fore.YELLOW}安装缺失的包:{Style.RESET_ALL}")
            for cmd in install_commands:
                print(f"  {cmd}")

    def check_system_requirements(self):
        """检查系统要求"""
        self.print_header("系统环境检查")
        
        # 定义最低要求
        MIN_MEMORY = 16  # GB
        MIN_DISK_SPACE = 200  # GB
        
        # Python版本
        python_version = subprocess.check_output(["python3", "--version"]).decode().strip()
        self.print_result(
            "Python版本", 
            True, 
            f"{python_version} (推荐 3.8 或更高)"
        )
        
        # 系统内存
        with open('/proc/meminfo') as f:
            mem_total = int(f.readline().split()[1]) / 1024 / 1024  # 转换为GB
        self.print_result(
            "系统内存", 
            mem_total >= MIN_MEMORY,
            f"{mem_total:.1f}GB (最低要求: {MIN_MEMORY}GB)"
        )
        
        # 磁盘空间
        df = subprocess.check_output(['df', '-h', '/']).decode().split('\n')[1]
        available_space = float(df.split()[3].replace('G', ''))
        self.print_result(
            "可用磁盘空间",
            available_space >= MIN_DISK_SPACE,
            f"{available_space}GB (最低要求: {MIN_DISK_SPACE}GB)"
        )
        
        # 必要软件
        required_software = {
            'docker': '使用容器运行服务',
            'git': '下载代码和模型',
            'nvidia-smi': 'GPU监控（可选）'
        }
        
        for software, purpose in required_software.items():
            try:
                subprocess.check_output(['which', software])
                self.print_result(f"{software}", True, f"已安装 ({purpose})")
            except subprocess.CalledProcessError:
                self.print_result(
                    f"{software}", 
                    software == 'nvidia-smi',  # nvidia-smi是可选的
                    f"未安装 ({purpose})"
                )

    def save_results(self):
        """保存检查结果"""
        filename = 'aws_environment_check_results.json'
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n{Fore.GREEN}检查结果已保存到: {filename}{Style.RESET_ALL}")

    def run_all_checks(self):
        """运行所有检查"""
        print(f"{Fore.GREEN}开始AWS环境检查...{Style.RESET_ALL}")
        
        self.check_aws_services()
        self.check_inf2_availability()
        self.get_inf2_quotas()
        self.check_huggingface_env()
        self.check_system_requirements()
        
        self.save_results()
        print(f"\n{Fore.GREEN}检查完成!{Style.RESET_ALL}")

def main():
    checker = AWSEnvironmentChecker()
    checker.run_all_checks()

if __name__ == "__main__":
    main()