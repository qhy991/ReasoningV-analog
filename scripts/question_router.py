#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题路由机制 (Question Router)
根据问题文本特征自动分类问题类型，并选择相应的优化策略
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum


class QuestionType(Enum):
    """问题类型枚举"""
    FACTUAL = "factual"  # 事实类：直接询问定义、概念
    REASONING = "reasoning"  # 推理类：需要逻辑推理
    CALCULATION = "calculation"  # 计算类：需要数值计算
    ANALYSIS = "analysis"  # 分析类：需要深入分析
    COMPARISON = "comparison"  # 比较类：需要比较多个选项


class QuestionRouter:
    """问题路由器 - 基于规则映射的方法"""
    
    def __init__(self):
        """初始化路由规则"""
        self.rules = self._build_routing_rules()
    
    def _build_routing_rules(self) -> Dict[QuestionType, Dict]:
        """构建路由规则"""
        return {
            QuestionType.FACTUAL: {
                "keywords": [
                    r"what is", r"what are", r"what does", r"what do",
                    r"define", r"definition", r"meaning", r"means",
                    r"which of the following", r"which one",
                    r"is defined as", r"refers to"
                ],
                "prompt_prefix": "Answer precisely:",
                "description": "事实类问题：直接询问定义、概念或事实"
            },
            QuestionType.REASONING: {
                "keywords": [
                    r"why", r"how does", r"how do", r"how is",
                    r"explain", r"reason", r"because", r"due to",
                    r"leads to", r"results in", r"causes",
                    r"relationship", r"relates to", r"depends on"
                ],
                "prompt_prefix": "Analyze carefully:",
                "description": "推理类问题：需要理解因果关系和逻辑推理"
            },
            QuestionType.CALCULATION: {
                "keywords": [
                    r"calculate", r"compute", r"determine", r"find",
                    r"what is the value", r"what is the result",
                    r"numerical", r"value of", r"equal to",
                    r"formula", r"equation", r"solve"
                ],
                "prompt_prefix": "Calculate precisely:",
                "description": "计算类问题：需要数值计算或公式应用"
            },
            QuestionType.ANALYSIS: {
                "keywords": [
                    r"analyze", r"analysis", r"examine", r"evaluate",
                    r"compare", r"contrast", r"difference", r"similarity",
                    r"advantage", r"disadvantage", r"benefit", r"drawback",
                    r"characteristic", r"feature", r"property"
                ],
                "prompt_prefix": "Analyze carefully:",
                "description": "分析类问题：需要深入分析和评估"
            },
            QuestionType.COMPARISON: {
                "keywords": [
                    r"better", r"best", r"worse", r"worst",
                    r"prefer", r"preferred", r"optimal", r"optimum",
                    r"superior", r"inferior", r"advantageous",
                    r"more efficient", r"less efficient"
                ],
                "prompt_prefix": "Compare and analyze:",
                "description": "比较类问题：需要比较多个选项或方案"
            }
        }
    
    def classify_question(self, question_text: str) -> Tuple[QuestionType, str]:
        """
        分类问题类型
        
        Args:
            question_text: 问题文本
            
        Returns:
            (问题类型, 提示词前缀)
        """
        question_lower = question_text.lower()
        
        # 计算每个类型的匹配分数
        scores = {}
        for qtype, rule in self.rules.items():
            score = 0
            for keyword in rule["keywords"]:
                matches = len(re.findall(keyword, question_lower))
                score += matches
            scores[qtype] = score
        
        # 选择得分最高的类型
        if max(scores.values()) > 0:
            best_type = max(scores, key=scores.get)
            prompt_prefix = self.rules[best_type]["prompt_prefix"]
            return best_type, prompt_prefix
        else:
            # 默认返回事实类
            return QuestionType.FACTUAL, "Answer precisely:"
    
    def get_strategy_for_question(
        self, 
        question_text: str, 
        task_name: str = "TQA"
    ) -> Dict:
        """
        为问题获取优化策略
        
        Args:
            question_text: 问题文本
            task_name: 任务名称
            
        Returns:
            策略配置字典
        """
        qtype, prompt_prefix = self.classify_question(question_text)
        
        # 基础参数配置
        base_params = {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "top_k": 1,
            "use_cache": True
        }
        
        # 根据任务类型和问题类型选择策略
        if task_name == "TQA":
            # TQA任务使用多策略
            strategy = {
                "prompt": f"{prompt_prefix} {{question}}\n\nOptions:\n{{options}}\n\nAnswer:",
                "params": base_params,
                "question_type": qtype.value,
                "description": self.rules[qtype]["description"]
            }
        else:
            # 其他任务使用任务特定策略
            strategy = {
                "prompt": f"{{question}}\n\nOptions:\n{{options}}\n\nAnswer:",
                "params": base_params,
                "question_type": qtype.value
            }
        
        return strategy
    
    def batch_classify(self, questions: List[str]) -> Dict[str, Dict]:
        """
        批量分类问题
        
        Args:
            questions: 问题文本列表
            
        Returns:
            分类结果字典 {index: {type, prompt_prefix, description}}
        """
        results = {}
        for idx, question in enumerate(questions):
            qtype, prompt_prefix = self.classify_question(question)
            results[str(idx)] = {
                "type": qtype.value,
                "prompt_prefix": prompt_prefix,
                "description": self.rules[qtype]["description"]
            }
        return results
    
    def get_statistics(self, questions: List[str]) -> Dict:
        """
        获取问题类型统计信息
        
        Args:
            questions: 问题文本列表
            
        Returns:
            统计信息字典
        """
        type_counts = {qtype.value: 0 for qtype in QuestionType}
        
        for question in questions:
            qtype, _ = self.classify_question(question)
            type_counts[qtype.value] += 1
        
        total = len(questions)
        statistics = {
            "total_questions": total,
            "type_distribution": {
                qtype: {
                    "count": count,
                    "percentage": round(count / total * 100, 2) if total > 0 else 0
                }
                for qtype, count in type_counts.items()
            }
        }
        
        return statistics


def test_router():
    """测试路由机制"""
    router = QuestionRouter()
    
    # 测试问题
    test_questions = [
        "What is the primary function of the pass transistor in an LDO regulator?",
        "Why does the error amplifier compare VREF with feedback?",
        "Calculate the output voltage when VIN = 5V and R1 = 10kΩ, R2 = 5kΩ.",
        "Analyze the advantages and disadvantages of different LDO topologies.",
        "Which LDO design is better for low-power applications?"
    ]
    
    print("=" * 80)
    print("问题路由机制测试")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        qtype, prompt_prefix = router.classify_question(question)
        strategy = router.get_strategy_for_question(question)
        
        print(f"\n问题 {i}: {question}")
        print(f"  类型: {qtype.value}")
        print(f"  提示词前缀: {prompt_prefix}")
        print(f"  描述: {strategy['description']}")
    
    # 统计信息
    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    stats = router.get_statistics(test_questions)
    print(f"总问题数: {stats['total_questions']}")
    for qtype, info in stats['type_distribution'].items():
        print(f"  {qtype}: {info['count']} ({info['percentage']}%)")


if __name__ == "__main__":
    test_router()
