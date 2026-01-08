

from metagpt.utils.common import parse_json_code_block, CodeParser, write_json_file
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger




class Write_test(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}
    Wrap above code to a function for solving math problem and Write {k} unit tests using pytest for the solving function.
    Return ```python your_code_here ``` with NO other texts, 
    You must start with ```python.
    your code:
    """

    name: str = "SimpleWriteTest"
    def __init__(self,**kwargs):
        super(Write_test,self).__init__()

    async def run(self, context: str, k: int = 3):
        prompt = self.PROMPT_TEMPLATE.format(context=context, k=k)

        rsp = await self._aask(prompt)
        if '```python' in rsp:
            code_text = CodeParser.parse_code('',rsp,'python')
        else: code_text=rsp.replace('```','')

        return code_text
