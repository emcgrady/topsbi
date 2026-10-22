# topsbi
This repository uses the simulation based inference techniques outlined by [J. Brehmer et al](https://arxiv.org/abs/1805.00020) by training a DNN binary classifier between two points $\theta_0$ and $\theta_1$ in Wilson coefficient space under the Standard Model effective field theory hypothesis. Its output is converted to the learned ratio $\hat r(x;\theta_1)$ between these two points, which can then be used for a binned or unbinned physics analysis. Notation and the mapping of code names to symbols are in `AGENTS.md` (Notation).
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
python -m topsbi.train examples/training/eft.yml
```
Examples of these yaml files can be found in the examples directory, with their keys listed in `examples/training/README.md`. What a training writes, its metrics and plots are described in [`docs/training.md`](docs/training.md); [`docs/`](docs/README.md) covers every stage of the analysis.
## Tests
The synthetic-data tests run on CPU in about a minute:
```sh
uv run pytest
```
## Code style
Code is formatted and linted with [ruff](https://docs.astral.sh/ruff/); CI checks both.
```sh
uv run ruff format .
uv run ruff check .
```
