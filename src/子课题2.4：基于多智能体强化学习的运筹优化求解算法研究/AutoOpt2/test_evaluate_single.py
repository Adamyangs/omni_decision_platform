#!/usr/bin/env python3
"""
测试单个问题的评估功能
用于验证evaluate.py的修复是否有效
"""

import json
import yaml
from openai import OpenAI

def test_single_evaluation():
    """测试评估单个问题"""
    
    # 加载配置
    config_path = "/home/wentian/HWtest/AutoOpt/config/config3.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    llm_config = config['llm']
    
    # 初始化客户端
    client = OpenAI(
        api_key=llm_config['api_key'],
        base_url=llm_config['base_url']
    )
    
    # 加载一个测试问题
    json_path = "/home/wentian/HWtest/AutoOpt/Result/all_problems_extracted4.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    # 取第一个问题进行测试
    problem = problems[0]
    
    print("="*60)
    print(f"测试问题: {problem['problem_id']}")
    print("="*60)
    
    # 创建简化的测试提示
    prompt = f"""你是一位数学优化专家，请评估以下线性规划问题的数学建模质量。

**问题**: {problem['UserRequirement'][:200]}...

**建模**: {str(problem['Formulation'])[:200]}...

请用JSON格式给出评分：
```json
{{
  "problem_id": "{problem['problem_id']}",
  "total_score": 85,
  "grade": "B",
  "evaluation": "测试评估"
}}
```
"""
    
    print("\n📤 发送测试请求...")
    
    try:
        response = client.chat.completions.create(
            model=llm_config['model'],
            messages=[
                {"role": "system", "content": "你是一位专业的数学优化专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        print("✅ 请求成功！\n")
        
        # 正确的访问方式
        result_text = response.choices[0].message.content
        
        print("📝 LLM返回的完整响应:")
        print("-"*60)
        print(result_text)
        print("-"*60)
        
        # 尝试解析JSON
        print("\n🔍 尝试解析JSON...")
        
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
            print("✅ 找到JSON内容:")
            print(json_str)
            print()
            
            try:
                result = json.loads(json_str)
                print("✅ JSON解析成功！")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"❌ JSON解析失败: {e}")
        else:
            print("❌ 未找到JSON内容")
        
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_evaluation()

