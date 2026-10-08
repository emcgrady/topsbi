# topsbi
This repository uses the simulation based inference techniques outlined by [J. Brehmer et al](https://arxiv.org/abs/1805.00020) by training a DNN binary classifier between two points in Wilson coefficient space under the Standard Model effective field theory hypothesis. This classifier can be converted to a likelihood ratio between these two points which can then be used for a binned or unbinned physics analysis. 
## Installation
This code uses PyTorch for neural network training and [uv](https://docs.astral.sh/uv/) to manage the environment. These instructions cover the system architecture of camlnd.crc.nd.edu; the same steps work on a laptop for CPU-only development.

Start by cloning the repository. 
```sh
git clone https://github.com/emcgrady/topsbi.git
cd topsbi
```
Then install and activate the repository UV environment. `uv sync` installs `topsbi` itself in editable mode, so code changes take effect without reinstalling.
```sh
uv sync
source .venv/bin/activate
```
## Network Training
All of the training is done through `train.py` which takes a single argument, a path to a configuration yaml.
```sh
python -m topsbi.train examples/training/stitched.yml
```
Examples of these yaml files can be found in the examples directory. As the fields in these configuration files will be updated regularly as various features are added, a separate README can be found in this directory.
## Tests
A small synthetic-data smoke test runs on CPU in about 20 seconds:
```sh
uv run pytest
```
