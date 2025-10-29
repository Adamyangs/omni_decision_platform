

from metagpt.utils.common import parse_json_code_block, CodeParser, write_json_file
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger



class Write_review(Action):
    PROMPT_TEMPLATE: str = """
    ## Code
    {code_text}

    ## Execution result
    {result}

    Review the test code and result of execution provide one critical comments:
    """

    name: str = "SimpleWriteReview"
    def __init__(self,**kwargs):
        super(Write_review,self).__init__()


    async def run(self, code_text:str,result: str):
        prompt = self.PROMPT_TEMPLATE.format(code_text=code_text,result=result)

        rsp = await self._aask(prompt)

        return rsp