# Introduction
AutoOpt is a framework for solving optimization problems based on large language models. The main idea is to decompose the problem-solving process into modeling and coding stages, which are implemented by the modeling module and the coding module, respectively.
The modeling module uses a large language model to receive the description of the optimization problem and constructs the corresponding mathematical model. This process requires the large language model to understand the data structure and problem type in the problem.
The coding module uses a large language model to write the solution code according to the mathematical model and tests the code. If the test passes, the code that passes the test is output; otherwise, the model is re-encoded using the test information. Both the modeling module and the coding module support multiple open-source and closed-source large language models to achieve hot-swapping capabilities.
Experimental results show that the AutoOpt framework exhibits high generalization in solving different types of optimization problems. Compared to using a large language model directly, the AutoOpt framework can improve the accuracy of problem-solving on various issues.
# Installation
1. Install dependencies
```shell
pip install -r requirements.txt
```
2. Install solvers
Currently, only OR-Tools and SciPy open-source solvers are used. If you want to use other solvers, you need to install them yourself and adjust the content in ** Write_code **.
* OR-Tools installation
[OR-Tools Installation Guide](https://developers.google.com/optimization/install?hl=zh-cn)
3. The intelligent agent framework of this project is based on MetaGPT, installation as follows:
```shell
pip install --upgrade metagpt
# or `pip install --upgrade git+https://github.com/geekan/MetaGPT.git`
# or `git clone https://github.com/geekan/MetaGPT && cd MetaGPT && pip install --upgrade -e .`
```
# Configuration
1. Configure the relevant parameters for closed-source models such as GPT-3.5, GPT4 in config/config2.yaml, refer to the configuration specification: https://github.com/geekan/MetaGPT/blob/main/config/config2.example.yaml
2. Open-source models such as Llama3, Mistral are used via ollama, and the open-source models supported by ollama are referred to: https://ollama.com/library
# Input and Output
1. The input is a problem_description.jsonl file, which can be replaced with the problem description as needed.
2. The output is in workspace/storage/, you can view the answers of each agent in each problem.

# Run
```bash
python AutoOpt.py --file_path /path/to/your/problem_descriptions.jsonl
```



