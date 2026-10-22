# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

Simulation-based inference for SMEFT (Brehmer et al., arXiv:1805.00020): a DNN binary classifier is trained to separate events weighted at two points in Wilson-coefficient (WC) space, θ_0 and θ_1 (config keys `c0`, `c1`). The learned score ŝ(x; θ_1) is converted to the learned ratio r̂(x; θ_1) = ŝ / (1 − ŝ), and an ensemble of such networks is combined to give r̂(x; θ) at arbitrary WC points θ. Nuisance parameters ν are trained with the soft⁺ loss, whose linear output is f̂ = log r̂(x; θ, ν).

The methods are written up in `/users/cmcgrad2/eft_nsbi_framework/main.tex` (repo: `git@github.com:emcgrady/eft_nsbi_framework.git`) and in the dissertation `/users/cmcgrad2/dissertation` (chapter statistical_methods). They cover stitching weighted SMEFT samples, training techniques, validation methods, and likelihood scans. Read them for the derivations behind the code. Code, docstrings and documentation use their notation; the Notation section below maps code names to symbols.

## Commands

The environment is managed with uv (`uv sync` installs `topsbi` in editable mode).

```sh
uv run pytest                                   # synthetic-data tests, CPU, about a minute
uv run pytest tests/test_smoke.py::test_get_lower_tri_matches_processor_ordering   # single test
uv run ruff check .                             # lint
uv run ruff format .                            # format (CI runs --check)
python -m topsbi.train examples/training/eft.yml   # train from a YAML config
```

CI (`.github/workflows/tests.yml`) runs `uv sync --locked`, `ruff check`, `ruff format --check`, and `pytest -q`. Ruff uses line length 120, preserves quote style, and excludes `tests/` and notebooks.

## Architecture

**Data format** (`docs/data.md`). Training data are pickled `torch` `TensorDataset`s whose first two tensors are the features x [n_events, n_features] and the structure constants s_q(z) [n_events, n_q] (coefs), n_q = (n_wcs+1)(n_wcs+2)/2; nuisance trainings read the combined format (nominal and variation directories with `event_id`, `syst_ratio`, `w_ref`). Each event's weight at a WC point θ (SM-inclusive, so θ[0] = 1) is s_q(z) · T(θ) = `coefs @ get_lower_tri(theta)`. `get_lower_tri` in `tools/data.py` must keep the `(i, j<=i)` ordering used by `expand_array` in ttbarEFT's tensor processor; a test pins this. The era index column (`year_loc`, 21 in this analysis) looks up `lumi`.

**Training (`train.py`).** It is driven entirely by a YAML config (fields are listed in `examples/training/README.md` and the per-feature keys in `examples/features/README.md`; outputs, metrics and plots are described in `docs/training.md`). The flow is: load data (or with `bootstrap: true`, split `train.p` 75/25 via `sample_bootstrap`) → `prepare_weights` computes `(features, w0, w1)` = (x, w^s(θ_0), w^s(θ_1)) → all weights are divided by the mean training `w0` → `Model.loss` concatenates the features twice, labelling the `w0` copy 0 and the `w1` copy 1, and uses the weights as per-event BCE weights → train with early stopping and an LR scheduler (`plateau`/`cosine`/`none`), both driven by the test-split loss. After training, the best checkpoint is evaluated on `holdout` if the config sets it (a dataset never used in training, e.g. `validation.p`), otherwise on the test split. Outputs go under `config['name']`: `model.pt` (best checkpoint), `complete/` (plots, `performance.yml`, animations), and `incomplete/` (plots every 50 epochs and per-epoch animation frames). Existing `complete/` and `incomplete/` directories are deleted at the start of a run.

**Weighting methods** (`method` in the config, required; `EFT_METHODS` and `SOFT_PLUS_METHODS` in `model/net.py`, checked by `train.check_config` and `Model`):
- `eft`: `get_weights`, which gives the stitched weights w^s(θ) = L_era s_q(z) · T(θ) × 1000 computed in float64. They keep the per-sample `xsec/nSumOfWeights` normalization stored in `coefs`, so samples are mixed by cross section times luminosity. It requires `lumi`. Do not divide by the weight at the generation point: that cancels the normalization and biases the learned ratio.
- `weight_shift` / `kinematic_shift` (nuisance methods, soft⁺ loss, linear output f̂ = log r̂(x; θ, ν)): `train.load_training_data` → `data.load_nuisance_data`. `weight_shift` pairs every nominal event with w^s(θ, 0) and w^s(θ, ν) = w^s(θ, 0) r(x, z; θ, ν) (`syst_ratio`); `kinematic_shift` labels the nominal sample (x, w^s(θ, 0)) 0 and the shifted sample (x′, w^s(θ, ν)) 1, and carries each row's event index i (`matched_events`) for the two-sample metrics.

**Model (`model/net.py`).** `Net` is f_φ(x), built from the config's `network` list of layers by `create_model` (falling back to `DEFAULT_NETWORK`, 128-128-1). A Sigmoid output gives ŝ, a linear output f̂ = log r̂; `Net.ratio` and `Net.log_ratio` return r̂ and log r̂ for either. Input standardization lives *inside* the network as registered buffers (`feature_mean`, `feature_std`), which are set from training-set statistics and saved in `model.pt`. Callers should therefore pass raw, unnormalized features. Normalization must be per-feature constants, never computed from the batch being evaluated (see the tests in `test_normalization.py`).

**Inference (`tools/build_ratio.py`).**
- `NetworkRatio` reloads one trained network from its training YAML (it reads `name/model.pt` and `name/complete/performance.yml`) and returns r̂(x; θ_1).
- `RatioFit` takes a set of networks a trained at different points θ_1^(a) (config `c1`) and solves ΘΓ = R by least squares (`lstsq`), with Θ (`Theta`) one row T(θ_1^(a))[mask] per network (SM last) and R (`R`) one row r̂(x; θ_1^(a)) per network. This gives per-event quadratic coefficients Γ (`Gamma`), so r̂(x; θ) = T(θ)[mask] Γ at any WC point.
- `Ensemble` averages Γ over several seeds. Each seed's Γ covers only the events in its own `finite` mask, so they are placed at their event indices, and `Ensemble.finite` keeps only events retained by every seed. All seeds must be trained at the same set of WC points.
- `Ensemble(..., per_seed=...)` passes each seed's `RatioFit` to a callback before it is discarded. `tools/ensembleCache.py` uses it to cache the ensemble under `{name}/ensemble/`: the mean and per-seed Γ, mask and finite masks, plus R and Θ for each seed (file names in the Notation section). `ensemble.yml` is written last and records the `model.pt` hashes, which `EnsembleCache.matches` uses to detect a stale cache.

**Validation (`validation.py`, `python -m topsbi.validation -c config.yml [--refit]`).** This is the last check before likelihood scans. It reads the ensemble cache (a missing or stale cache stops the run unless `--refit` is given) and requires `lumi` in the config. Its steps are:
- Bias test per component γ_q (row q of Γ, q over the kept terms of T(θ)): the kinematic closure test term by term, Δ_q,b = ⟨ŵ_q,b − w_q,b⟩_e (`D` in code) with ŵ_q,b = Σ_i w^s_i(θ_0) γ̂_q,i 𝟙(x_i ∈ K_b) and w_q,b = Σ_i L s_q(x_i, z_i) × 1000 𝟙(x_i ∈ K_b), averaged over ensemble members e, in equal-count quantile bins K_b of every network input (bins merged to `n_eff >= 30`). Its χ² against the MC variance Var_MC plus the variance of the ensemble mean Var_seeds gives Z (flagging uses Z > 5) and a Gaussian-equivalent Z_gauss.
- Attribution: the flagged biases are split exactly into signed shares of the training networks via Θ⁺.
- Suspects: networks that carry a large share and close badly on their own are proposed for dropping or retraining. Each drop is tested by refitting the cache without that network and is accepted only if it helps and harms no other component.
- Point search: greedily proposes new training points that minimise tr(Cov(γ̂) G), the propagated network uncertainty on the flagged components, G = diag(g_q / Var(γ̂_q)) with objective weights g_q (or over the scan region with `G: scan`).

It writes `ensemble_diagnostics.yml`, `proposed_points.yml` and `plots/` under `config['name']`.

**Metrics (`tools/metrics.py` `closure_metrics`, `tools/plots.py` `network_plots`).** Every training type writes the same closure metrics, from `closure_metrics`: the reweighted reference yield ŵ_b = Σ r̂ w0 against the target yield w_b = Σ w1 per bin, each sample binned by its own features, with Cov(Δ_b, Δ_c) from `matched_closure` (an event in both samples counts once; paired data, eft and weight_shift, pass the same events as both samples). `closure_summary` turns a result into χ²/ndf, Z, the noise-subtracted rms bias (size) and the worst pull. Tests: rate (one bin), 50 equal-count bins R_b of f̂ = log r̂, group x R_b cells (config `closure_groups`, default era x njets; `train.closure_options` builds the columns, caps and labels from the config), and every feature in its merged binning K_b, each also with r̂ = 1 (`nominal_*`); plus `auc`. The R_b test also gives `r_pull` and `r_ci_diff` (summaries of its per-bin pulls, `closure_summary`) for every training type. Paired data add `r_spread.png` (`r_spread_plot`), the per-event spread of the true ratio around r̂. `docs/training.md` documents every key. χ²/ndf measures significance and grows with the sample; rms bias measures size; read them together. Closure groups must be built from network inputs, because latent labels (the generator-level mtt slice, lepton flavour from `ids`) do not close even for a perfect network. `merge_sparse_bins` (float64 sums, so bins do not depend on the CPU) merges bins until each has `n_eff >= 30`; out-of-range events go into the edge bins. `closure_plot` (`tools/plots.py`) draws one closure test: ŵ_b and w_b, and Δ_b/w_b in the matched band, with χ²/ndf, Δ_rms and Z, on equal-width bins labelled by `bin_tick_labels`. Training writes `complete/kinematics/<feature>.png` for every feature and `r_closure.png` from the final `closure_metrics` run, plus `animations/<feature>.gif` for `features_to_animate`.

## Notation

Symbols follow `main.tex` and the dissertation. Ratios are written with the numerator hypothesis as argument, r̂(x; θ) or r̂(x; θ, ν); the reference (θ_0 for EFT trainings, (θ, 0) for nuisances) is stated in the text. Means and weighted means are written ⟨·⟩ (`\left\langle … \right\rangle` in plot labels), never with a bar. σ is reserved for cross sections; uncertainties are written Var(·), Cov(·, ·), √Var(·). Bins: K_b (kinematic), R_b (quantiles of r̂), S_b (quantiles of ŝ). Indices the documents do not define (new): a for a training network (row of Θ), e for an ensemble member (seed), g_q for the point-search objective weights. Code names that differ from the symbol, including stored keys that are not renamed:

| Code | Symbol / meaning |
|---|---|
| config `c0`, `c1`, `wcs`, `lumi` | θ_0, θ_1, WC names, L per era |
| `coefs` | structure constants s_q(z) (× xsec/nSumOfWeights) |
| `get_lower_tri(theta)` | T(θ), the (i, j ≤ i) quadratic expansion of (1, θ) |
| `w0`, `w1` | w^s(θ_0), w^s(θ_1); for nuisances w^s(θ, 0), w^s(θ, ν) |
| `r`, `r_hat`, `f_hat`, `s`, `s_hat` | r(x, z; θ), r̂(x; θ), f̂_φ(x), s(x, z; θ), ŝ(x; θ) |
| `w_hat`, `w`, `w_hat_b`, `w_b` | per-event r̂ w^s(θ_0) and w^s(θ_1); bin totals ŵ_b, w_b |
| `r_bar_b`, `r_hat_bar_b`, `R_edges` | ⟨r⟩_b, ⟨r̂⟩_b, edges of R_b |
| `x`, `x_shift`, `i0`, `i1` | x, x′ (shifted reco features), event index i of nominal / shifted rows |
| `w_ref` (data column), `sample_weight` | weight at the reference point θ_ref (θ_SM or θ_g, from sample.yml), and L w_ref × 1000 |
| `syst_ratio` (data column) | r(x, z; θ, ν) = w^s(θ, ν) / w^s(θ, 0) per weight systematic and direction |
| `Theta`, `R`, `Gamma` | Θ, R, Γ of ΘΓ = R; cache files `network.npz` (Theta), `ratios.npy` (R), `gammas.npy` / `gamma_cache.npy` (Γ), `filter.npy` / `filter_cache.npy` (finite events), `mask_cache.npy` (kept columns of T) |
| `k` (component index in validation code and plots) | q, restricted to the kept terms of T(θ) |
| `D`, `mc_var`, `s2` (code and ensemble_diagnostics.yml) | Δ, Var_MC, Var_seeds |
| `seed` (config key, loop variable) | ensemble member e |
| `g` (objective_weights) | g_q |
| `sigma0`, `sigma1`, `Sigma`, keys `sigma_propagated`, `sigma_after_search` | √Var(γ̂_q) before / after the point search, Cov(γ̂) |
| `sm_yield`, `yield_ratio`, `misclosure` | λ_{K_b}(θ_SM), λ(θ_1) / λ(θ_SM), m_a |
| performance.yml `auc`, `rate_*`, `r_*` (incl. `r_pull`, `r_ci_diff`), `group_*`, `feature_*`, `nominal_*` | closure metrics (docs/training.md) |

Config `data`/`name` paths in `examples/` point to shared storage on camlnd.crc.nd.edu (`/groups/klannon/...`). Tests build synthetic data in `tests/conftest.py` instead.
