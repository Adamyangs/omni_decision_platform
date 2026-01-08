import json

def renumber_problem_ids(input_file, output_file=None):
    """
    重新编号JSON文件中的problem_id，从problem_1开始依次递增
    
    Args:
        input_file: 输入的JSON文件路径
        output_file: 输出的JSON文件路径（如果为None，则覆盖原文件）
    """
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 重新编号
    for index, item in enumerate(data, start=1):
        item['problem_id'] = f"problem_{index}"
    
    # 确定输出文件路径
    if output_file is None:
        output_file = input_file
    
    # 写回JSON文件（保持格式美观）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"✓ 成功重新编号 {len(data)} 条数据")
    print(f"✓ 已保存到: {output_file}")
    print(f"✓ 编号范围: problem_1 到 problem_{len(data)}")

if __name__ == "__main__":
    # 输入文件路径
    input_file = "/home/wentian/HWtest/AutoOpt/Result/all_problems_extracted5.json"
    
    # 选项1: 覆盖原文件
    # renumber_problem_ids(input_file)
    
    # 选项2: 如果想保存到新文件，可以取消下面这行的注释
    renumber_problem_ids(input_file, "/home/wentian/HWtest/AutoOpt/Result/all_problems_extracted5_renumbered.json")

