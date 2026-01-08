import json
import yaml
from openai import OpenAI
from typing import Dict, List, Any
import os
from datetime import datetime


class MathModelingEvaluator:
    """数学建模精度评估器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化评估器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        llm_config = config['llm']
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url']
        )
        self.model = llm_config['model']
        
    def load_problems(self, json_path: str) -> List[Dict[str, Any]]:
        """
        加载问题数据
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            问题列表
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            problems = json.load(f)
        return problems
    
    def create_evaluation_prompt(self, problem: Dict[str, Any]) -> str:
        """
        创建评估提示词
        
        Args:
            problem: 单个问题数据
            
        Returns:
            评估提示词
        """
        prompt = f"""你是一位数学优化和运筹学专家，请评估以下线性规划问题的数学建模质量。

**问题编号**: {problem['problem_id']}

**原始问题描述**:
{problem['UserRequirement']}

**思考过程**:
{problem['Thinking']}

**数学建模（Formulation）**:
{json.dumps(problem['Formulation'], indent=2, ensure_ascii=False)}

请从以下5个维度对该数学建模进行评分（每项满分20分，总分100分）：

1. **变量定义准确性（20分）**：
   - 变量是否完整覆盖问题中的决策变量
   - 变量命名是否清晰易懂
   - 变量类型（连续/整数）是否正确标注

2. **目标函数正确性（20分）**：
   - 目标函数是否准确反映优化目标
   - 系数是否正确提取自问题描述
   - 优化方向（最大化/最小化）是否正确

3. **约束完整性（20分）**：
   - 是否包含所有必要的约束条件
   - 是否遗漏了隐含约束（如非负性、整数性）
   - 约束是否存在冗余或矛盾

4. **约束转化准确性（20分）**：
   - 文字描述到数学表达式的转化是否准确
   - 不等式方向是否正确
   - 复杂约束（如比例、倍数关系）是否正确处理

5. **表达规范性（20分）**：
   - JSON结构是否合理
   - 数学符号使用是否规范
   - 是否包含必要的描述/注释信息

**评估要求**：
1. 对每个维度给出具体分数和详细理由
2. 指出存在的错误或不足之处
3. 给出改进建议
4. 最后给出总分和综合评价

请按以下JSON格式输出评估结果：
```json
{{
  "problem_id": "{problem['problem_id']}",
  "scores": {{
    "variable_definition": {{"score": 分数, "reason": "详细理由"}},
    "objective_function": {{"score": 分数, "reason": "详细理由"}},
    "constraint_completeness": {{"score": 分数, "reason": "详细理由"}},
    "constraint_accuracy": {{"score": 分数, "reason": "详细理由"}},
    "expression_standardization": {{"score": 分数, "reason": "详细理由"}}
  }},
  "total_score": 总分,
  "issues": ["问题1", "问题2", ...],
  "suggestions": ["建议1", "建议2", ...],
  "overall_evaluation": "综合评价文字",
  "grade": "A/B/C/D/F"
}}
```

评分等级标准：
- A (90-100): 优秀，建模准确完整
- B (80-89): 良好，有minor issues
- C (70-79): 中等，有明显不足
- D (60-69): 较差，存在重要错误
- F (0-59): 不合格，建模严重错误

请开始评估。
"""
        return prompt
    
    def evaluate_single_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个问题
        
        Args:
            problem: 问题数据
            
        Returns:
            评估结果
        """
        prompt = self.create_evaluation_prompt(problem)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的数学优化专家，擅长评估线性规划建模质量。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 降低温度以获得更稳定的评估
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content
            
            # 提取JSON部分（先尝试查找```json块，如果没有则查找{}）
            if '```json' in result_text:
                json_start = result_text.find('```json') + 7
                json_end = result_text.find('```', json_start)
                if json_end != -1:
                    json_str = result_text[json_start:json_end].strip()
                else:
                    json_str = result_text[json_start:].strip()
            else:
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                else:
                    json_str = None
            
            if json_str:
                try:
                    evaluation = json.loads(json_str)
                    return evaluation
                except json.JSONDecodeError as je:
                    print(f"    JSON解析失败: {str(je)}")
                    print(f"    尝试解析的内容: {json_str[:200]}...")
                    return {
                        "problem_id": problem['problem_id'],
                        "error": f"JSON解析失败: {str(je)}",
                        "raw_response": result_text
                    }
            else:
                # 如果无法找到JSON，返回原始文本
                print(f" 未找到JSON内容")
                return {
                    "problem_id": problem['problem_id'],
                    "error": "未找到JSON内容",
                    "raw_response": result_text
                }
                
        except Exception as e:
            print(f"评估出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "problem_id": problem['problem_id'],
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def evaluate_all_problems(self, json_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        评估所有问题
        
        Args:
            json_path: 输入JSON文件路径
            output_path: 输出结果路径（可选）
            
        Returns:
            完整评估报告
        """
        print(f"正在加载问题数据: {json_path}")
        problems = self.load_problems(json_path)
        print(f"共加载 {len(problems)} 个问题\n")
        
        evaluations = []
        
        for i, problem in enumerate(problems, 1):
            print(f"正在评估 {problem['problem_id']} ({i}/{len(problems)})...")
            evaluation = self.evaluate_single_problem(problem)
            evaluations.append(evaluation)
            print(f"  总分: {evaluation.get('total_score', 'N/A')}, "
                  f"等级: {evaluation.get('grade', 'N/A')}\n")
        
        # 生成统计报告
        report = self.generate_report(evaluations)
        
        # 保存结果
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_report_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_problems": len(problems),
                "evaluations": evaluations,
                "statistics": report
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估完成！结果已保存至: {output_path}")
        
        return report
    
    def generate_report(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成统计报告
        
        Args:
            evaluations: 所有评估结果
            
        Returns:
            统计报告
        """
        valid_evals = [e for e in evaluations if 'total_score' in e]
        
        if not valid_evals:
            return {"error": "没有有效的评估结果"}
        
        total_scores = [e['total_score'] for e in valid_evals]
        grades = [e['grade'] for e in valid_evals]
        
        # 计算各维度平均分
        dimension_scores = {
            'variable_definition': [],
            'objective_function': [],
            'constraint_completeness': [],
            'constraint_accuracy': [],
            'expression_standardization': []
        }
        
        for eval_result in valid_evals:
            if 'scores' in eval_result:
                for dim, score_info in eval_result['scores'].items():
                    if dim in dimension_scores:
                        dimension_scores[dim].append(score_info['score'])
        
        report = {
            "overall": {
                "average_score": sum(total_scores) / len(total_scores),
                "max_score": max(total_scores),
                "min_score": min(total_scores),
                "median_score": sorted(total_scores)[len(total_scores) // 2]
            },
            "grade_distribution": {
                "A (90-100)": grades.count('A'),
                "B (80-89)": grades.count('B'),
                "C (70-79)": grades.count('C'),
                "D (60-69)": grades.count('D'),
                "F (0-59)": grades.count('F')
            },
            "dimension_averages": {
                dim: sum(scores) / len(scores) if scores else 0
                for dim, scores in dimension_scores.items()
            },
            "top_problems": sorted(
                valid_evals,
                key=lambda x: x['total_score'],
                reverse=True
            )[:5],
            "bottom_problems": sorted(
                valid_evals,
                key=lambda x: x['total_score']
            )[:5]
        }
        
        # 打印报告摘要
        print("\n" + "="*60)
        print("评估报告摘要")
        print("="*60)
        print(f"平均分: {report['overall']['average_score']:.2f}")
        print(f"最高分: {report['overall']['max_score']}")
        print(f"最低分: {report['overall']['min_score']}")
        print(f"\n等级分布:")
        for grade, count in report['grade_distribution'].items():
            print(f"  {grade}: {count} 个")
        print(f"\n各维度平均分:")
        for dim, score in report['dimension_averages'].items():
            print(f"  {dim}: {score:.2f}")
        print("="*60)
        
        return report


def main():
    """主函数"""
    # 配置文件路径
    config_path = "/home/wentian/HWtest/AutoOpt/config/config3.yaml"
    
    # 输入JSON文件路径
    json_path = "/home/wentian/HWtest/AutoOpt/Result/test_new_01.json"  # 修改为你的JSON文件路径
    
    # 输出结果路径（可选，默认自动生成）
    output_path = None
    
    # 创建评估器
    evaluator = MathModelingEvaluator(config_path)
    
    # 执行评估
    report = evaluator.evaluate_all_problems(json_path, output_path)


if __name__ == "__main__":
    main()