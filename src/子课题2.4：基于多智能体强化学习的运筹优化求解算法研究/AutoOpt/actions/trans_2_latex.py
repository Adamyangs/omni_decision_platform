

from metagpt.utils.common import parse_json_code_block, CodeParser, write_json_file
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger



class Trans_2_latex(Action):
    PROMPT_TEMPLATE: str = '''
    Now, according to what you write on your scratchpads, give the LATEX format of the model directly.
    NOTE: MUST USE LATEX MATH SYMBOLS
    ## Scratch pad
    {scratch_pad}

    ## Model
    ```latex
'''


    name: str = "TransToLATEX"

    def __init__(self,**kwargs):
        super(Trans_2_latex,self).__init__()

        self.config.prompt_schema='raw'
    
    async def run(self, formulation: str):
        prompt = self.PROMPT_TEMPLATE.format(scratch_pad=str(formulation))

        latex_formulation = await self._aask(prompt)
        # formulation= Thinking.parse_json(formulation)
        result=''

        return latex_formulation
    