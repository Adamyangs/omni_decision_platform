

from metagpt.utils.common import parse_json_code_block, CodeParser, write_json_file
from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
import subprocess 



class Run_code(Action):
    name: str = "run_code"
    def __init__(self,**kwargs):
        # pass
        super(Run_code,self).__init__()

    async def run(self, code_text: str):
        run_result=''
        try:
            result = subprocess.run(["python3", "-c", code_text],check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            run_result = result.stdout
        except subprocess.CalledProcessError as e:
            run_result = e.stderr
        logger.info(f"{run_result=}")
        return run_result
