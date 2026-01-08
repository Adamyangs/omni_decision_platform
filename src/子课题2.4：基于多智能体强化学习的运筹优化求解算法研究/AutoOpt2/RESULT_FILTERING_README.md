# AutoOpt2.py 结果过滤和保存功能说明

## 功能概述

修改后的 `AutoOpt2.py` 现在具有自动评估求解结果并选择性保存成功案例的功能。

## 主要新增功能

### 1. 结果提取 (`extract_result_from_team`)
- 从 MetaGPT team 的执行记忆中自动提取计算结果
- 查找 `Run_code` 或 `Write_review` 步骤的输出
- 使用正则表达式提取数值结果

### 2. 结果比较 (`is_similar_result`)
- 比较计算结果与数据集中的 `en_answer`
- **相对误差容忍度**: 默认 1% (可调整)
- **特殊值处理**:
  - "No Best Solution" → 任何正数结果都视为成功
  - "-99999" → 表示无解问题，任何正数都算好结果
  - "Unknown" → 任何正数都视为成功
- **零值处理**: 期望值为 0 时使用绝对误差

### 3. 结果管理 (`manage_result_folder`)
- 自动管理 `Result` 文件夹
- **最大保存数量**: 30 个（可配置）
- 超过限制时自动删除最旧的结果

### 4. 结果保存 (`save_successful_result`)
- 将成功的求解过程复制到 `Result` 文件夹
- 自动创建 `result_summary.json` 摘要文件，包含:
  - 问题编号
  - 计算结果
  - 期望结果
  - 匹配类型（similar/exact）

## 文件结构

```
/home/wentian/HWtest/AutoOpt/
├── AutoOpt2.py                    # 主脚本（已修改）
├── Dataset/
│   └── NL4OPT_with_optimal_solution.json  # 数据集
├── workspace/
│   └── storage/
│       └── team/
│           ├── problem_1/         # 所有问题都会保存在这里
│           ├── problem_2/
│           └── ...
└── Result/                        # 仅保存成功的结果（最多30个）
    ├── problem_1/
    │   ├── result_summary.json    # 结果摘要
    │   └── [其他序列化文件]
    ├── problem_3/
    └── ...
```

## 使用方法

运行脚本：
```bash
python AutoOpt2.py
```

脚本会自动：
1. 读取数据集中的每个问题
2. 执行建模和求解
3. 提取计算结果
4. 与 `en_answer` 比较
5. 如果结果相似，保存到 `Result` 文件夹

## 日志输出示例

```
================================================================================
Problem 1/246: [问题描述]
Optimal answer: 1160.0
================================================================================

[执行过程...]

================================================================================
Evaluating results for Problem 1...
Calculated result: 1160.0
✓ Result matches! Saving to Result folder...
✓ Successfully saved problem_1 to Result folder
  Calculated: 1160.0, Expected: 1160.0
================================================================================
```

## 配置参数

可以通过修改以下参数来调整行为：

### 在 `is_similar_result` 函数中：
- `tolerance=0.01`: 相对误差容忍度（1% = 0.01）

### 在 `manage_result_folder` 函数中：
- `max_count=30`: Result 文件夹最大保存数量

### 在 `main` 函数调用中：
```python
python AutoOpt2.py --investment=3.0 --n_round=3
```

## 结果摘要文件示例

`Result/problem_1/result_summary.json`:
```json
{
  "problem_number": 1,
  "calculated_result": 1160.0,
  "expected_result": "1160.0",
  "match": "similar"
}
```

## 注意事项

1. **Result 文件夹管理**: 当超过 30 个结果时，会自动删除最早保存的结果
2. **结果提取**: 如果无法从执行结果中提取数值，该问题不会被保存
3. **特殊值处理**: 对于 "No Best Solution" 等特殊答案，只要得到正数结果就视为成功
4. **容错性**: 即使结果提取失败，程序也会继续处理下一个问题

## 修改内容总结

1. **新增导入**: `re`, `shutil`
2. **新增函数**:
   - `extract_result_from_team()`: 提取结果
   - `is_similar_result()`: 比较结果
   - `manage_result_folder()`: 管理文件夹
   - `save_successful_result()`: 保存结果
3. **修改 `main` 函数**:
   - 创建 Result 文件夹
   - 在每个问题求解后添加评估逻辑
   - 选择性保存成功结果

## 故障排查

如果遇到问题：

1. **结果未被保存**: 检查日志中的 "Calculated result" 是否正确提取
2. **结果不匹配**: 可以调整 `tolerance` 参数增加容忍度
3. **文件夹已满**: Result 文件夹会自动清理，无需手动干预



