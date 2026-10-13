# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

Simulation-based inference for SMEFT (Brehmer et al., arXiv:1805.00020): a DNN binary classifier is trained to separate events weighted at two points in Wilson-coefficient (WC) space, `c0` and `c1`. The classifier output `s` is converted to a likelihood ratio `s / (1 - s)`, and an ensemble of such networks is combined to give the likelihood ratio at arbitrary WC points.

The methods are written up in https://www.overleaf.com/read/vbnhkwndrbpm#8e3a7a. The page renders client-side, so fetching it returns no content; ask the user for the `.tex` or PDF when you need the derivations.

## Commands

The environment is managed with uv (`uv sync` installs `topsbi` in editable mode).

```sh
uv run pytest                                   # synthetic-data smoke tests, CPU, ~20 s
uv run pytest tests/test_smoke.py::test_get_lower_tri_matches_processor_ordering   # single test
uv run ruff check .                             # lint
uv run ruff format .                            # format (CI runs --check)
python -m topsbi.train examples/training/stitched.yml   # train from a YAML config
```

CI (`.github/workflows/tests.yml`) runs `uv sync --locked`, `ruff check`, `ruff format --check`, and `pytest -q`. Ruff uses line length 120, preserves quote style, and excludes `tests/` and notebooks.

## Architecture

**Data format.** Training data are pickled `torch` `TensorDataset`s (`train.p`, `test.p` in the config's `data` dir) holding `(features [N, 22], quadratic fit coefficients [N, 153], ids [N, 3])`. Each event's weight at a WC point `c` (SM-inclusive, so `c[0] = 1`) is `coefs @ get_lower_tri(c)`. `get_lower_tri` in `tools/data.py` must keep the `(i, j<=i)` ordering used by `expand_array` in ttbarEFT's tensor processor; a test pins this. Feature column 21 is `year_int` (era index used to look up `lumi`).

**Training (`train.py`).** It is driven entirely by a YAML config (fields are documented in `examples/training/README.md`, and per-feature binning in `examples/features/README.md`). The flow is: load data (or with `bootstrap: true`, split `train.p` 75/25 via `sample_boostrap`) → `prepare_weights` computes `(features, w0, w1)` → `Model.loss` concatenates the features twice, labelling the `w0` copy 0 and the `w1` copy 1, and uses the weights as per-event BCE weights → train with early stopping and an LR scheduler (`plateau`/`cosine`/`none`). Outputs go under `config['name']`: `model.pt` (best checkpoint), `complete/` (plots, `performance.yml`, animations), and `incomplete/` (per-epoch scratch plots). Existing `complete/` and `incomplete/` directories are deleted at the start of a run.

**Weighting methods** (`method` in the config, dispatched in `prepare_weights` and `Model.loss`):
- `stitched`: `get_probabilities`, where each of p0/p1 is normalized by its mean.
- `weights_only`: `get_weights`, which gives stitched weights `L_year * coefs @ T(c) * 1000` computed in float64 and divided by the weight at `cg`. It requires `lumi`.
- `weight_shift` / `alice` / `kinematic_shift`: alternate loss branches in `Model`. `kinematic_shift` is trained by the separate `kinematic_training.py` (it uses two datasets, `data_0`/`data_1`).
- `parameterize_weights` exists for parameterized training, but `prepare_weights` does not currently call it.

**Model (`model/net.py`).** `Net` is built from the config's `network` list of layers (falling back to a 32-16-8-1 default). Input standardization lives *inside* the network as registered buffers (`feature_mean`, `feature_std`), which are set from training-set statistics and saved in `model.pt`. Callers should therefore pass raw, unnormalized features. Normalization must be per-feature constants, never computed from the batch being evaluated (see the tests in `test_normalization.py`).

**Inference (`tools/buildLikelihood.py`).**
- `likelihood` reloads one trained network from its training YAML (it reads `name/model.pt` and `name/complete/performance.yml`).
- `full_likelihood` takes a set of networks trained at different `c1` points and solves a least-squares fit (`lstsq`) of their ratios against `get_lower_tri(c1)`. This gives per-event quadratic coefficients `gammas`, so the LR at any WC point is `get_lower_tri(c)[mask] @ gammas`.
- `ensemble` averages `gammas` over several seeds.
- `validation.py` compares the parametric ensemble with a dedicated network. It currently fails to import (it uses the removed `expand_array`, issue #5), and the import test marks it as a strict xfail.

Config `data`/`name` paths in `examples/` point to shared storage on camlnd.crc.nd.edu (`/groups/klannon/...`). Tests build synthetic data in `tests/conftest.py` instead.
