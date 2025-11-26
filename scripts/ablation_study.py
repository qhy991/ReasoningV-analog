#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验脚本 (Ablation Study)
逐步验证每种优化策略的独立效果
"""

import json
import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")


class AblationStudy:
    """消融实验类"""
    
    def __init__(self, model_path: str):
        """初始化"""
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        # 实验配置
        self.experiments = {
            "baseline": {
                "name": "基线配置",
                "prompt_template": "Question: {question}\n\nOptions:\n{options}\n\nAnswer:",
                "params": {
                    "max_new_tokens": 3,
                    "temperature": 0.1,
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 10
                }
            },
            "params_only": {
                "name": "仅参数优化",
                "prompt_template": "Question: {question}\n\nOptions:\n{options}\n\nAnswer:",
                "params": {
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "do_sample": False,
                    "top_p": 1.0,
                    "top_k": 1,
                    "use_cache": True
                }
            },
            "prompt_only": {
                "name": "仅提示词优化",
                "prompt_template": "Circuit Expert: {question}\n\nOptions:\n{options}\n\nAnswer:",
                "params": {
                    "max_new_tokens": 3,
                    "temperature": 0.1,
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 10
                }
            },
            "few_shot_1": {
                "name": "Few-shot 1示例",
                "use_few_shot": True,
                "num_examples": 1,
                "expert_instruction": "You are a circuit expert.",
                "params": {
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "do_sample": False,
                    "top_p": 1.0,
                    "top_k": 1,
                    "use_cache": True
                }
            },
            "few_shot_2": {
                "name": "Few-shot 2示例",
                "use_few_shot": True,
                "num_examples": 2,
                "expert_instruction": "You are a circuit expert.",
                "params": {
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "do_sample": False,
                    "top_p": 1.0,
                    "top_k": 1,
                    "use_cache": True
                }
            },
            "few_shot_3": {
                "name": "Few-shot 3示例",
                "use_few_shot": True,
                "num_examples": 3,
                "expert_instruction": "You are a circuit expert.",
                "params": {
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "do_sample": False,
                    "top_p": 1.0,
                    "top_k": 1,
                    "use_cache": True
                }
            },
            "full_optimization": {
                "name": "完整优化（参数+提示词+Few-shot）",
                "use_few_shot": True,
                "num_examples": 3,
                "expert_instruction": "You are an LDO circuit expert. Analyze LDO circuits by checking:\n1. Pass transistor (source fixed at VDD)\n2. Error amplifier (compares VREF with feedback)\n3. Stable bandgap reference\n4. Resistive divider feedback network\n",
                "params": {
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                    "do_sample": False,
                    "top_p": 1.0,
                    "top_k": 1,
                    "use_cache": True
                }
            }
        }
    
    def load_model(self):
        """加载模型"""
        print(f"📥 正在加载模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ 模型加载完成")
    
    def generate_answer(self, prompt: str, params: Dict) -> str:
        """生成答案"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **params
            )
        
        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        return answer[0] if answer else "A"
    
    def create_few_shot_prompt(self, question: str, options: str, num_examples: int, expert_instruction: str, examples: List[Dict]) -> str:
        """创建Few-shot提示词"""
        prompt = expert_instruction + "\n\n"
        prompt += "Examples:\n"
        
        for i, example in enumerate(examples[:num_examples], 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {example['question']}\n"
            prompt += f"Options:\n{example['options']}\n"
            prompt += f"Answer: {example['answer']}\n\n"
        
        prompt += "Now solve this:\n"
        prompt += f"Question: {question}\n"
        prompt += f"Options:\n{options}\n"
        prompt += "Answer:"
        
        return prompt
    
    def test_task(self, task_data: List[Dict], experiment_name: str, task_name: str = "LDO") -> Dict:
        """测试单个任务"""
        config = self.experiments[experiment_name]
        correct = 0
        total = len(task_data)
        times = []
        
        # 准备Few-shot示例（如果有）
        examples = []
        if config.get("use_few_shot", False) and task_name == "LDO":
            # LDO任务的Few-shot示例
            examples = [
                {
                    "question": "What determines the dropout voltage in an LDO?",
                    "options": "A. Input voltage level\nB. Pass transistor characteristics\nC. Load current\nD. Temperature",
                    "answer": "B"
                },
                {
                    "question": "How does the error amplifier work in an LDO?",
                    "options": "A. It compares input and output\nB. It compares reference and feedback\nC. It amplifies the load current\nD. It generates the reference voltage",
                    "answer": "B"
                },
                {
                    "question": "What is the role of the feedback network in an LDO?",
                    "options": "A. To sense the input voltage\nB. To divide the output voltage\nC. To control the pass transistor\nD. To generate the reference",
                    "answer": "B"
                }
            ]
        
        print(f"\n🧪 实验: {config['name']}")
        print(f"   测试题目数: {total}")
        
        for i, item in enumerate(task_data[:10]):  # 只测试前10题以节省时间
            question = item.get("question", "")
            options = item.get("options", "")
            groundtruth = item.get("ground_truth", item.get("groundtruth", "A"))
            
            # 构建提示词
            if config.get("use_few_shot", False):
                prompt = self.create_few_shot_prompt(
                    question, 
                    options, 
                    config.get("num_examples", 1),
                    config.get("expert_instruction", ""),
                    examples
                )
            else:
                prompt = config["prompt_template"].format(question=question, options=options)
            
            # 生成答案
            start_time = time.time()
            try:
                answer = self.generate_answer(prompt, config["params"])
                if answer.upper() == groundtruth.upper():
                    correct += 1
            except Exception as e:
                print(f"   错误 (题目 {i+1}): {e}")
                answer = "A"
            
            times.append(time.time() - start_time)
            
            if (i + 1) % 5 == 0:
                print(f"   进度: {i+1}/{min(10, total)}")
        
        accuracy = (correct / min(10, total)) * 100 if total > 0 else 0
        avg_time = sum(times) / len(times) if times else 0
        
        return {
            "experiment": experiment_name,
            "name": config["name"],
            "correct": correct,
            "total": min(10, total),
            "accuracy": accuracy,
            "avg_time": avg_time
        }
    
    def run_ablation_study(self, task_data: List[Dict], task_name: str = "LDO") -> Dict:
        """运行消融实验"""
        print("=" * 80)
        print(f"消融实验: {task_name} 任务")
        print("=" * 80)
        
        results = {}
        
        # 按顺序运行每个实验
        experiment_order = [
            "baseline",
            "params_only",
            "prompt_only",
            "few_shot_1",
            "few_shot_2",
            "few_shot_3",
            "full_optimization"
        ]
        
        for exp_name in experiment_order:
            result = self.test_task(task_data, exp_name, task_name)
            results[exp_name] = result
        
        return results
    
    def save_results(self, results: Dict, output_file: str):
        """保存结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 结果已保存到: {output_file}")


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python ablation_study.py <model_path> [task_data_file]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    task_data_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 创建消融实验对象
    study = AblationStudy(model_path)
    study.load_model()
    
    # 加载任务数据（示例）
    if task_data_file and os.path.exists(task_data_file):
        with open(task_data_file, 'r', encoding='utf-8') as f:
            task_data = json.load(f)
    else:
        # 使用示例数据
        task_data = [
            {
                "question": "What is the primary function of the pass transistor in an LDO regulator?",
                "options": "A. To provide voltage reference\nB. To control the output current\nC. To regulate the output voltage by adjusting its resistance\nD. To generate the feedback signal",
                "ground_truth": "C"
            }
        ] * 10
    
    # 运行消融实验
    results = study.run_ablation_study(task_data, "LDO")
    
    # 保存结果
    output_file = "results/ablation_study_results.json"
    os.makedirs("results", exist_ok=True)
    study.save_results(results, output_file)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("消融实验结果总结")
    print("=" * 80)
    for exp_name, result in results.items():
        print(f"{result['name']:30s} | 准确率: {result['accuracy']:6.2f}% | 平均时间: {result['avg_time']:.3f}s")


if __name__ == "__main__":
    main()
