
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
import random
import fire
import json
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


async def main(
    investment: float = 3.0,
    n_round: int = 3,
    add_human: bool = False,
    # file_path: str = './problem_descriptions.jsonl',
    file_path: str = './Dataset/NL4OPT_with_optimal_solution.json',

    sample_size: int = 20,  # 新增参数：要测试的样本数量
    random_seed: int = 52,  # 新增参数：随机种子，用于复现结果
):

    # 设置随机种子（如果提供）
    if random_seed is not None:
        random.seed(random_seed)
        logger.info(f'Random seed set to: {random_seed}')

    # Read the JSONL file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # data_dict_list = [json.loads(line) for line in lines]
    # for i,data_d in enumerate(data_dict_list):
    #     problem=data_d['problem_description']
    #     logger.info(f'{problem=}')

    data_dict_list = [json.loads(line) for line in lines if line.strip()]  # 添加 if line.strip() 过滤空行

    total_problems = len(data_dict_list)
    logger.info(f'Total problems in dataset: {total_problems}')
    
    # 随机选择样本
    if sample_size > 0 and sample_size < total_problems:
        data_dict_list = random.sample(data_dict_list, sample_size)
        logger.info(f'Randomly selected {sample_size} problems for testing')
    else:
        logger.info(f'Using all {total_problems} problems')



    for i,data_d in enumerate(data_dict_list):
        problem = data_d['en_question']  # 改为 'en_question'
        optimal_answer = data_d.get('en_answer', 'Unknown')  # 可选：获取最优答案
        # logger.info(f'{problem=}')
        logger.info(f'Problem {i+1}/{len(data_dict_list)}: {problem}')
        logger.info(f'Optimal answer: {optimal_answer}')  # 可选：记录最优答案
    

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

if __name__ == "__main__":
    fire.Fire(main)
