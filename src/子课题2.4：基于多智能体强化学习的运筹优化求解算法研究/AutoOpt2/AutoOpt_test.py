import os
# 注释掉代理设置 - 阿里云 DashScope API 使用直连即可
# os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
from pathlib import Path
current_file_path = Path(__file__).resolve()
parent_dir_path = current_file_path.parent.resolve()
os.environ["METAGPT_PROJECT_ROOT"] = str(parent_dir_path)
from metagpt.const import METAGPT_ROOT
print('root path: ',METAGPT_ROOT)

from metagpt.const import SERDESER_PATH
import fire
import json
import re
import shutil
from metagpt.actions import  UserRequirement
from metagpt.logs import logger
from metagpt.team import Team
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from actions import (Thinking,Formulation,Trans_2_latex,
                     Write_code, Run_code, Write_review,
                     Write_test
                     )


class Model_llm(Role):
    name: str = "Alice"
    profile: str = "model_llm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([Thinking,Formulation,Trans_2_latex,])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        # logger.info('todo: ',todo)

        msg = self.get_memories(k=1)[0]  # find the most k recent messages
        # logger.info(f'{msg.content=}')
        result = await todo.run(msg.content)
        # logger.info(f'{type(todo)} result: {result}')

        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg
    

class Code_llm(Role):
    name: str = "Alice"
    profile: str = "code_llm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([Trans_2_latex,])  # feel free to try this too

        self.set_actions([Write_code,Run_code,Write_review])
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        # By choosing the Action by order under the hood
        todo = self.rc.todo

        msg = self.get_memories(k=1)[0]  # find the most k recent messages
        if type(todo) is Write_review:
            run_result=self.get_memories(k=1)[0]
            code_text = self.get_memories(k=2)[0]
            result = await todo.run(code_text=code_text.content,result=run_result.content)
        else:
            result=await todo.run(msg.content)

        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg
    


class Test_llm(Role):
    name: str = "Bob"
    profile: str = "test_llm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([Write_test])
        self._watch([Write_code, Write_review])  # feel free to try this too

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo

        context = self.get_memories()  # use all memories as context

        code_text = await todo.run(context, k=5)  # specify arguments
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg


def extract_result_from_team(team):
    """从team的执行结果中提取计算得到的答案"""
    try:
        # 尝试从team的角色记忆中提取结果
        for role in team.roles:
            if role.profile == "code_llm":
                memories = role.rc.memory.get()
                # 查找Run_code或Write_review的结果
                for msg in reversed(memories):
                    if "Run_code" in str(type(msg.cause_by)) or "Write_review" in str(type(msg.cause_by)):
                        content = msg.content
                        # 尝试提取数值结果
                        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', content)
                        if numbers:
                            # 返回最后一个找到的数字（通常是最终结果）
                            return float(numbers[-1])
        return None
    except Exception as e:
        logger.error(f"Error extracting result: {e}")
        return None


def is_similar_result(calculated, expected, tolerance=0.01):
    """比较计算结果与期望结果是否相似
    
    Args:
        calculated: 计算得到的结果
        expected: 期望的结果（字符串）
        tolerance: 相对误差容忍度（默认1%）
    
    Returns:
        bool: 结果是否相似
    """
    try:
        # 处理特殊情况
        if calculated is None:
            return False
        
        # 处理expected中的特殊值
        if isinstance(expected, str):
            if expected.lower() in ['unknown', 'no best solution', '-99999.0', '-99999']:
                # 对于这些特殊值，如果有任何合理的正数结果就认为是好的
                return calculated > 0
            
            # 尝试转换为浮点数
            try:
                expected_num = float(expected)
            except ValueError:
                logger.warning(f"Cannot convert expected answer to float: {expected}")
                return False
        else:
            expected_num = float(expected)
        
        # 如果expected是负数且很大（如-99999），表示问题无解，此时任何正数都算好结果
        if expected_num < -9999:
            return calculated > 0
        
        # 计算相对误差
        if expected_num == 0:
            # 如果期望值是0，使用绝对误差
            return abs(calculated - expected_num) < 0.01
        else:
            # 使用相对误差
            relative_error = abs((calculated - expected_num) / expected_num)
            return relative_error <= tolerance
    
    except Exception as e:
        logger.error(f"Error comparing results: {e}")
        return False


def manage_result_folder(result_base_path, max_count=30):
    """管理Result文件夹，确保不超过最大数量
    
    Args:
        result_base_path: Result文件夹的路径
        max_count: 最大保存数量
    """
    result_base_path = Path(result_base_path)
    if not result_base_path.exists():
        result_base_path.mkdir(parents=True, exist_ok=True)
        return
    
    # 获取所有problem_*文件夹
    problem_folders = sorted([f for f in result_base_path.iterdir() if f.is_dir() and f.name.startswith("problem_")])
    
    # 如果超过最大数量，删除最旧的
    if len(problem_folders) >= max_count:
        # 删除最旧的文件夹
        oldest = problem_folders[0]
        logger.info(f"Result folder full, removing oldest: {oldest.name}")
        shutil.rmtree(oldest)


def save_successful_result(source_path, result_base_path, problem_num, calculated_result, expected_result):
    """保存成功的求解结果到Result文件夹
    
    Args:
        source_path: 源文件夹路径（problem_X）
        result_base_path: Result文件夹基础路径
        problem_num: 问题编号
        calculated_result: 计算结果
        expected_result: 期望结果
    """
    result_base_path = Path(result_base_path)
    manage_result_folder(result_base_path)
    
    # 目标路径
    dest_path = result_base_path / f"problem_{problem_num}"
    
    # 复制文件夹
    if Path(source_path).exists():
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(source_path, dest_path)
        
        # 创建一个结果摘要文件
        summary_file = dest_path / "result_summary.json"
        summary = {
            "problem_number": problem_num,
            "calculated_result": calculated_result,
            "expected_result": str(expected_result),
            "match": "similar" if is_similar_result(calculated_result, expected_result) else "exact"
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Successfully saved problem_{problem_num} to Result folder")
        logger.info(f"  Calculated: {calculated_result}, Expected: {expected_result}")
    else:
        logger.warning(f"Source path does not exist: {source_path}")


async def main(
    investment: float = 3.0,
    n_round: int = 3,
    add_human: bool = False,
    # file_path: str = './problem_descriptions.jsonl',
    file_path: str = './Dataset/NL4OPT_with_optimal_solution.json',
):

    # Read the JSONL file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # data_dict_list = [json.loads(line) for line in lines]
    # for i,data_d in enumerate(data_dict_list):
    #     problem=data_d['problem_description']
    #     logger.info(f'{problem=}')

    data_dict_list = [json.loads(line) for line in lines if line.strip()]  # 添加 if line.strip() 过滤空行
    
    # 设置Result文件夹路径
    result_folder = Path(parent_dir_path) / "Result"
    result_folder.mkdir(exist_ok=True)
    
    for i,data_d in enumerate(data_dict_list):
        problem = data_d['en_question']  # 改为 'en_question'
        optimal_answer = data_d.get('en_answer', 'Unknown')  # 可选：获取最优答案
        # logger.info(f'{problem=}')
        logger.info(f'\n{"="*80}')
        logger.info(f'Problem {i+1}/{len(data_dict_list)}: {problem}')
        logger.info(f'Optimal answer: {optimal_answer}')  # 可选：记录最优答案
        logger.info(f'{"="*80}\n')
    

        team = Team()
        team.hire(
            [
                Model_llm(),
                Code_llm(),
                Test_llm(),
            ]
        )

        team.invest(investment=investment)
        await team.run(n_round=n_round,idea=problem)
        team_path=SERDESER_PATH.joinpath(f"problem_{i+1}")
        team.serialize(stg_path=team_path)
        
        # ==================== 新增：结果比较和保存逻辑 ====================
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating results for Problem {i+1}...")
        
        # 提取计算结果
        calculated_result = extract_result_from_team(team)
        
        if calculated_result is not None:
            logger.info(f"Calculated result: {calculated_result}")
            
            # 比较结果
            if is_similar_result(calculated_result, optimal_answer):
                logger.info(f"✓ Result matches! Saving to Result folder...")
                save_successful_result(
                    source_path=team_path,
                    result_base_path=result_folder,
                    problem_num=i+1,
                    calculated_result=calculated_result,
                    expected_result=optimal_answer
                )
            else:
                logger.info(f"✗ Result does not match. Expected: {optimal_answer}, Got: {calculated_result}")
                logger.info(f"  Not saving to Result folder.")
        else:
            logger.info(f"✗ Could not extract result from team execution.")
            logger.info(f"  Not saving to Result folder.")
        
        logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    fire.Fire(main)