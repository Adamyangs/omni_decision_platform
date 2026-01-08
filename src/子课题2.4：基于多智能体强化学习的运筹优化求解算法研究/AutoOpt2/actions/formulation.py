
from metagpt.utils.common import parse_json_code_block, CodeParser, write_json_file
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger


class Formulation(Action):
    prefix: str ='''
    You are a modeling expert specialized in the field of Operations Research and Optimization. Your expertise lies in Linear Programming (LP) models, 
    and you possess an in-depth understanding of various modeling techniques within the realm of operations research.

'''

    PROMPT_TEMPLATE: str = """
    Now, according to what you write on your scratchpads, summary the mathematical expression of the model directly and explicitly in response in JSON.
    Note: ONLY output the mathematical expression of the model, return ```json your_output_here ``` with NO other texts

    ## Scratch pad
    {scratch_pad}

    ## Model
    ```json
    """

    response_format:str='''{
    "title": "Formulation",
    "type": "object",
    "properties": {
        "objective_function": {
            "title": "objective_function",
            "type": "string",
            "description": "The objective function consists of parameters and variables from an operations research problem, using numerical representations for parameters and standard mathematical symbols for variables."
        },
        "constraints": {
            "title": "constraints",
            "type": "string",
            "description": "Constraints consist of numerical parameters and variables from an operations research problem, using numerical representations for parameters and standard mathematical symbols for variables."
        }
    },
    "required": ["objective_function", "constraints"]
}

'''

    name: str = "formulation"

    def __init__(self,**kwargs):
        # pass
        super(Formulation,self).__init__()



    async def run(self, thinking_steps: str):
        prompt = self.PROMPT_TEMPLATE.format(scratch_pad=str(thinking_steps))

        formulation = await self._aask(prompt)
        formulation= Formulation.parse_json(formulation)
        result=''
        # if formulation:
        #     for k,v in formulation['properties'].items():
        #         result+=v['description']+', '
        if formulation:
            if 'properties' in formulation:
                    for k,v in formulation['properties'].items():
                        result+=f'{k}: {v["description"]}, '
            else:
                for k,v in formulation.items():
                    if isinstance(v,str):
                        result+=v+', '
                    elif isinstance(v,dict):
                        result+=str(next(iter(v.values())))

        return str(formulation)
    
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