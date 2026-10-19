# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

Simulation-based inference for SMEFT (Brehmer et al., arXiv:1805.00020): a DNN binary classifier is trained to separate events weighted at two points in Wilson-coefficient (WC) space, `c0` and `c1`. The classifier output `s` is converted to a likelihood ratio `s / (1 - s)`, and an ensemble of such networks is combined to give the likelihood ratio at arbitrary WC points.

The methods are written up in `/users/cmcgrad2/eft_nsbi_framework/main.tex` (repo: `git@github.com:emcgrady/eft_nsbi_framework.git`). Its sections cover stitching weighted SMEFT samples, training techniques, validation methods, and likelihood scans. Read it for the derivations behind the code.

## Commands

The environment is managed with uv (`uv sync` installs `topsbi` in editable mode).

```sh
uv run pytest                                   # synthetic-data tests, CPU, about a minute
uv run pytest tests/test_smoke.py::test_get_lower_tri_matches_processor_ordering   # single test
uv run ruff check .                             # lint
uv run ruff format .                            # format (CI runs --check)
python -m topsbi.train examples/training/stitched.yml   # train from a YAML config
```

CI (`.github/workflows/tests.yml`) runs `uv sync --locked`, `ruff check`, `ruff format --check`, and `pytest -q`. Ruff uses line length 120, preserves quote style, and excludes `tests/` and notebooks.

## Architecture

**Data format.** Training data are pickled `torch` `TensorDataset`s (`train.p`, `test.p` in the config's `data` dir) holding `(features [N, 22], quadratic fit coefficients [N, 153], ids [N, 3])`. Each event's weight at a WC point `c` (SM-inclusive, so `c[0] = 1`) is `coefs @ get_lower_tri(c)`. `get_lower_tri` in `tools/data.py` must keep the `(i, j<=i)` ordering used by `expand_array` in ttbarEFT's tensor processor; a test pins this. Feature column 21 is `year_int` (era index used to look up `lumi`).

**Training (`train.py`).** It is driven entirely by a YAML config (fields are documented in `examples/training/README.md`, and per-feature binning in `examples/features/README.md`). The flow is: load data (or with `bootstrap: true`, split `train.p` 75/25 via `sample_boostrap`) → `prepare_weights` computes `(features, w0, w1)` → all weights are divided by the mean training `w0` → `Model.loss` concatenates the features twice, labelling the `w0` copy 0 and the `w1` copy 1, and uses the weights as per-event BCE weights → train with early stopping and an LR scheduler (`plateau`/`cosine`/`none`), both driven by the test-split loss. After training, the best checkpoint is evaluated on `holdout` if the config sets it (a dataset never used in training, e.g. `validation.p`), otherwise on the test split. Outputs go under `config['name']`: `model.pt` (best checkpoint), `complete/` (plots, `performance.yml`, animations), and `incomplete/` (plots every 50 epochs and per-epoch animation frames). Existing `complete/` and `incomplete/` directories are deleted at the start of a run.

**Weighting methods** (`method` in the config, dispatched in `prepare_weights` and `Model.loss`):
- `weights_only` (recommended): `get_weights`, which gives the stitched weights `L_year * coefs @ T(c) * 1000` computed in float64. They keep the per-sample `xsec/nSumOfWeights` normalization stored in `coefs`, so samples are mixed by cross section times luminosity. It requires `lumi`. Do not divide by the weight at the generation point: that cancels the normalization and biases the learned ratio.
- `stitched`: `get_probabilities`, where each of p0/p1 is normalized by its mean.
- `weight_shift` / `alice` / `kinematic_shift`: alternate loss branches in `Model`. `kinematic_shift` is trained by the separate `kinematic_training.py` (it uses two datasets, `data_0`/`data_1`).
- `parameterize_weights` exists for parameterized training, but `prepare_weights` does not currently call it.

**Model (`model/net.py`).** `Net` is built from the config's `network` list of layers (falling back to a 32-16-8-1 default). Input standardization lives *inside* the network as registered buffers (`feature_mean`, `feature_std`), which are set from training-set statistics and saved in `model.pt`. Callers should therefore pass raw, unnormalized features. Normalization must be per-feature constants, never computed from the batch being evaluated (see the tests in `test_normalization.py`).

**Inference (`tools/buildLikelihood.py`).**
- `likelihood` reloads one trained network from its training YAML (it reads `name/model.pt` and `name/complete/performance.yml`).
- `full_likelihood` takes a set of networks trained at different `c1` points and solves a least-squares fit (`lstsq`) of their ratios against `get_lower_tri(c1)`. This gives per-event quadratic coefficients `gammas`, so the LR at any WC point is `get_lower_tri(c)[mask] @ gammas`.
- `ensemble` averages `gammas` over several seeds. Each seed's `gammas` cover only the events in its own `infFilter`, so they are placed at their event indices, and `ensemble.infFilter` keeps only events retained by every seed. All seeds must be trained at the same set of WC points.
- `ensemble(..., per_seed=...)` passes each seed's fit to a callback before it is discarded. `tools/ensembleCache.py` uses it to cache the ensemble under `{name}/ensemble/`: the mean and per-seed `gammas`, mask and filters, plus every network's per-event ratio and Θ (the expanded training points, one row per network, SM last) for each seed. `ensemble.yml` is written last and records the `model.pt` hashes, which `EnsembleCache.matches` uses to detect a stale cache.

**Validation (`validation.py`, `python -m topsbi.validation -c config.yml [--refit]`).** This is the last check before likelihood scans. It reads the ensemble cache (a missing or stale cache stops the run unless `--refit` is given) and requires `lumi` in the config. Its steps are:
- Bias test per γ component: each component k gets a binned bias Δ = Σ L w0 γ̂_k − Σ L w_k in equal-count quantile bins of every network input (bins merged to `n_eff >= 30`). Its χ² against the MC error E² plus the seed variance s² gives Z (flagging uses Z > 5) and a Gaussian-equivalent Z_gauss.
- Attribution: the flagged biases are split exactly into signed shares of the training networks via Θ⁺.
- Suspects: networks that carry a large share and close badly on their own are proposed for dropping or retraining. Each drop is tested by refitting the cache without that network and is accepted only if it helps and harms no other component.
- Point search: greedily proposes new training points that minimise tr(Σ_γ G), the propagated network error on the flagged components (or over the scan region with `G: scan`).

It writes `ensemble_diagnostics.yml`, `proposed_points.yml` and `plots/` under `config['name']`.

**Metrics (`tools/plots.py` `networkPlots`, `tools/metrics.py`).** All calibration metrics weight events by `w0`, so they use the same weighting as training. `lr_calibration_difference` gives `pull` and `ci_diff`, `lr_calibration_curve` gives `rms_bias` (noise-subtracted), and `group_closure` checks closure in cells of era x njets x r-hat bins. Each is 0 for a calibrated network, and `examples/training/README.md` documents them. Store metrics from coarse binning (about 50 r-hat bins): with very fine bins the pulls become non-Gaussian and insensitive to smooth biases. Closure groups must be built from network inputs, because latent labels (the generator-level mtt slice, lepton flavour from `ids`) do not close even for a perfect network. `kinematic_histogram` merges bins until each has `n_eff >= 30` and puts out-of-range events into the edge bins.

Config `data`/`name` paths in `examples/` point to shared storage on camlnd.crc.nd.edu (`/groups/klannon/...`). Tests build synthetic data in `tests/conftest.py` instead.
