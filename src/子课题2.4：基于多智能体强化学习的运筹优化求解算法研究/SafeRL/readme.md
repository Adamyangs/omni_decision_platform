## Code repo of submisson paper
### Clarify
We use [Omnisafe](https://github.com/PKU-Alignment/omnisafe) as our benchmark and create our algorithm as a plugin in Omnisafe to train and test it.

### Quick Start
We have delete other irrelevant files for simplicity and make it a minimal reproduction.


OmniSafe requires Python 3.8+ and PyTorch 1.10+.

Install from source
```bash
# Create a conda environment
conda env create --file conda-recipe.yaml
conda activate omnisafe

# Install omnisafe
pip install -e .
```
Run the algorithm
```bash
# we have create a bash file for convience
cd example
bash run.sh
```
### License
OmniSafe is released under Apache License 2.0.