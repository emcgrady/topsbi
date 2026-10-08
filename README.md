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
## Combining Processor Output
The ttbarEFT tensor processor writes many small `*.p` files per variation. Combine them into the single dataset that training reads:
```sh
python -m topsbi.combine <prefix>/SR_CHANNELS_2j3j/nominal/to_train --out <prefix>/SR_CHANNELS_2j3j/nominal/train.p
```
Files are read in sorted order, so the result is reproducible. Pass several directories to combine them (e.g. `to_train` and `validation` for all events). Events with m(ℓℓbb) > 2000 GeV, the reco-level stand-in for m(tt), are removed; change this with `--max-mllbb` (`inf` disables the cut).
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
## Code style
Code is formatted and linted with [ruff](https://docs.astral.sh/ruff/); CI checks both.
```sh
uv run ruff format .
uv run ruff check .
```
