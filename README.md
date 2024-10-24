# AttSAGE-CFM
A Smart Contract Vulnerability Detection Method Based on Graph Neural Networks and Zero-Shot Learning

## Requirements
### Required Packages
* Python 3.6+
* Numpy 1.19+
* PyTorch 1.10+
* PyG 2+

Run the following script to install required packages.
```
pip install --upgrade pip
pip install numpy
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
```

## Data preprocessing
1. **Intermediate representation:** The smart contract source code is transformed into intermediate representation using [Slither](https://github.com/crytic/slither.git).
2. **Normalization:** We first use [solcx](https://pypi.org/project/py-solc-x/) to convert the smart contract source code into AST, then extract all function and variable names from the AST and standardize them into a unified format.
3. **Graph:** By normalizing the code and converting the intermediate representation into control flow graphs.

## Running Project
