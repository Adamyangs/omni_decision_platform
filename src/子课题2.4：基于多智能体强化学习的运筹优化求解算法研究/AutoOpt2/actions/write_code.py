

from metagpt.utils.common import parse_json_code_block, CodeParser, write_json_file
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger




class Write_code(Action):
    PROMPT_TEMPLATE: str = '''
    Imagine you are an expert in programming. Based on the correct mathematical model established above, please write a code to solve this math problem.
    Remember, the code you generate should exactly correspond to the mathematical model, especially the objective function and constraints.
    Ensure the code includes the necessary imports, defines the inputs, solves the problem, and prints the result."
    NOTE: You are ONLY allowed to using << Numpy, SciPy, OR-Tools >> libraries to solve the problem.
    Return ```python your_code_here ``` with NO other texts

    ## Mathematical Model
    {math_model}

    ## Code
    ```python 
    
'''
    name: str = "write_code"
    def __init__(self,**kwargs):
        # pass
        super(Write_code,self).__init__()
        self.config.prompt_schema='raw'

    async def run(self, model: str):
        prompt = self.PROMPT_TEMPLATE.format(math_model=str(model))

        raw_text = await self._aask(prompt)
        if '```python' in raw_text:
            code_text = CodeParser.parse_code('',raw_text,'python')
        else: code_text=raw_text.replace('```','')
        # if not code_text:
        #     code_text=raw_text.split('```')[0]
        return code_text