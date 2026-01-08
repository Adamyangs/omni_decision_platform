
from metagpt.utils.common import parse_json_code_block, CodeParser, write_json_file
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger



class Thinking(Action):
    prefix: str ='''
    You are a modeling expert specialized in the field of Operations Research and Optimization. Your expertise lies in Linear Programming (LP) models, 
    and you possess an in-depth understanding of various modeling techniques within the realm of operations research.

'''

    PROMPT_TEMPLATE: str = """
    At present, you are given an Operations Research problem, your role is to :
    1. Identify and extract relevant parameters from the problem description. Parameters refer to fixed values that define certain aspects of the problem but are not subject to change during the optimization process.
    2. Identify and extract the relevant variables from the problem statement. Variables represent the unknowns or decision variables in the optimization problem.
    3. Identify and extract the constraints from the problem description. Constraints represent the limitations or conditions that need to be satisfied in the optimization problem. Translate the constraints into mathematical expressions.
    4. Identify and extract the objective function from the problem statement. The objective function represents the goal of the optimization problem. Translate the objective funtion into mathematical expressions.

    The problem description is as following, please review carefully and organize in response format. 
    Note: ONLY give steps, do not output formulation, response in following json schema:
    {response_json_schema}

    Note: ONLY output the mathematical expression of the model, return ```json your_output_here ``` with NO other texts


    ## Given Problem
    {information}

    ## Thinking steps
    ```json
    """
    response_format:str='''{
    "title": "Steps",
    "type": "object",
    "properties": {
        "Step_1": {
            "description": "variables and sets mentioned in problem description",
            "type": "string"
        },
        "Step_2": {
            "description": "parameters in problem description",
            "type": "string"
        },
        "Step_3": {
            "description": "objective funtion consists of variables and numerical parameters",
            "type": "string"
        },
        "Step_4": {
            "description": "all the constraints consist of variables and numerical parameters",
            "type": "string"
        }
    },
    "required": ["Step_1", "Step_2", "Step_3", "Step_4"]
}
'''

    name: str = "thinking"

    def __init__(self,**kwargs):
        # pass
        super(Thinking,self).__init__()


    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(information=instruction,response_json_schema=self.response_format)

        thinking_steps = await self._aask(prompt)
        thinking_steps= Thinking.parse_json(thinking_steps)
        result=''
        if thinking_steps:
            if 'properties' in thinking_steps:
                for k,v in thinking_steps['properties'].items():
                    result+=f'{k}: {v["description"]}, '

            else:
                for k,v in thinking_steps.items():
                    if isinstance(v,str):
                        result+=v+', '
                    elif isinstance(v,dict):
                        result+=next(iter(v.values()))

        return result

    @staticmethod
    def parse_json(rsp):
        json_block=parse_json_code_block(rsp)
        # logger.info(f'{json_block=}')
        if isinstance(json_block,list) and json_block:
            rsp=json_block[0]
        elif json_block: rsp=json_block
        else: rsp=rsp
        # rsp=parse_json_code_block(rsp)[0]
        try: 
            result=eval(rsp)
        except Exception as e:
            result={}
            logger.debug(e)
        return result