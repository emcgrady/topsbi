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
## Network Training
All of the training is done through `train.py` which takes a single argument, a path to a configuration yaml. Examples of these yaml files can be found in the examples directory. As the fileds in these configuration files will be updated regulary as various features are added, a serpate README can be found in this directory. 

There are two cases for netwrok training, quadratic and other, which are determined by the configuration fields `signalTrainingTerm` and `signalTrainingPoint` respectively. To train a discriminator for a quadratic field, put `null` in the `signalTrainingPoint` field and put the name of the quadratic term (i.e. sm, ctq8, cdq1, etc.) in the `signalTrainingTerm` field.  Conversely, to train any other term, put `null` in the `signalTrainingTerm` field and a point in WC space (i.e. ctq8=2.5) in the `signalTrainingPoint` field. Note that for learning any cross-terms, the values for both BSM fields must be given. 

## Future Development Plans
- [ ] Add quantitative measure to LR ROC recreation
- [ ] Factor common fields for configuration yaml files
- [ ] Network combination
