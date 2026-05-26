# topsbi
This repository uses the simulation based inference techniques outlined by [J. Brehmer et al](https://arxiv.org/abs/1805.00020) by training a DNN binary classifier between two points in Wilson coefficient space under the Standard Model effective field theory hypothesis. This classifier can be converted to a likelihood ratio between these two points which can then be used for a binned or unbinned physics analysis. 
## Installation
This code uses PyToch to for neural network training. As outlined in the [PyTorch instalation instructions](https://pytorch.org/get-started/locally/), Conda is no longer supported. Using the current `environment.yml` to install via Conda or Mamba will install the last Conda-supported PyTorch version (2.5.1). To install current versions of PyTorch and Cuda, use the corresponding pip command for your GPU. These instructions will cover the system architecture of camlnd.crc.nd.edu. 

Start by cloning the repository. 
```sh
git clone https://github.com/emcgrady/topsbi.git
cd topsbi
```
Then instal and activate the repository Conda/Mamba environment.
```sh
mamba env create -f environment.yml
mamba activate topsbi
```
Instal Conda for GPU use (Optional)
Use the instructions found in the [PyTorch documentation](https://pytorch.org/get-started/locally/) for your OS and GPUs. For camlnd.crc.nd.edu, use the following command. 
```sh
pip3 install torch torchvision torchaudio
```
The validation framework uses [topcoffea](https://github.com/TopEFT/topcoffea) as a baseline to compare to when reweighting. In any directory, topcoffea can be installed by the following commands.
```sh
git clone https://github.com/TopEFT/topcoffea.git
cd topcoffea
pip install .
```
## Network Training
All of the training is done through `train.py` which takes a single argument, a path to a configuration yaml. Examples of these yaml files can be found in the examples directory. As the fileds in these configuration files will be updated regulary as various features are added, a serpate README can be found in this directory.
