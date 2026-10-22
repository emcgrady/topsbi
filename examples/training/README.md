## Configuration file fields

What a training writes, its metrics and plots: [`docs/training.md`](../../docs/training.md).


- `animate_per_epoch`: how often (in epochs) a frame is added to the closure animations of `features_to_animate` (5 in the examples)
- `batchSize`: number of events to feed the network at once
- `bootstrap`: if `true`, split `train.p` 75/25 into training and test sets; if `false`, load `train.p` and `test.p` separately. Ignored for a combined sample (see `data`)
- `c0`, `c1`: `eft` only, the points $\theta_0$ (label 0) and $\theta_1$ (label 1): $n_{\rm wcs}+1$ values each, 1 (SM) followed by one value per WC in `wcs` order
- `closure_groups`: optional columns of the group closure test, in order, each `{loc: column, label: text, cap: value}` (`label` and `cap` optional; values above `cap` are merged into it and labelled "≥cap"). Group columns must be network inputs. Default: the era column (`year_loc`) and column 19 capped at 4 (njets in this analysis's features), `[{loc: <year_loc>}, {loc: 19, label: jets, cap: 4}]`; set it for other feature layouts
- `data`: directory of the training data ([`docs/data.md`](../../docs/data.md)): pickled `TensorDataset`s whose first tensors are the features and the structure constants $s_q(z)$. Training reads `train.p`, and also `test.p` when `bootstrap` is `false`. For `eft` on a combined sample, `data` is its `nominal/` directory (recognised by its `sample.yml`): `train.p` is split 75/25 by a hash of each event's key (`event_split`, the same split as the nuisance trainings with the same `seed`) and the holdout defaults to `validation.p`. For nuisance methods, `data` is the combined sample's root
- `device`: the device used for training: `cpu`, or `cuda` for a GPU (falls back to the CPU with a warning when none is visible). For more options, see the [PyTorch documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device)
- `epochs`: maximum number of epochs used to train
- `direction`: nuisance methods (required), `up` or `down` (case-insensitive)
- `eras`: optional era names for the plot labels, indexed like `lumi` (as many entries); without it, the eras are labelled by their index
- `features`: path to the feature configuration yaml ([`examples/features/README.md`](../features/README.md))
- `features_to_animate`: features from the feature configuration whose closure plot is animated over the epochs (`animations/<feature>.gif`)
- `holdout`: optional path to a dataset not used for training, early stopping or the learning-rate scheduler, e.g. the `validation.p` produced alongside `train.p` (the default for a combined sample). When set, the final plots and `performance.yml` in `complete/` are computed on it. Without it, they are computed on the early-stopping test split and are slightly optimistic
- `learningRate`: initial learning rate for network training
- `lumi`: integrated luminosity of each era ($n_{\rm eras}$ values), indexed by the era column (`year_loc`). Required by every method. Training uses only the ratios between eras; the expected events in the plots also use the absolute values ([`docs/data.md`](../../docs/data.md): `coefs` in pb with `lumi` in fb$^{-1}$)
- `method` (required):
    1. `eft`: EFT training with the stitched event weights $w^s(z;\theta) = L\,s_q(z)\cdot T(\theta) \times 1,000$, which keep the per-sample cross-section normalization, so the samples are mixed by cross section times luminosity
    1. `weight_shift`: nuisance training on the nominal sample, weighted by `w_ref` against `w_ref` times the systematic's `syst_ratio` (keys `systematic`, `direction`), with the soft$^+$ loss
    1. `kinematic_shift`: nuisance training of the nominal sample against the shifted sample of the variation directory (keys `systematic`, `direction`), with the soft$^+$ loss

    Both read a combined sample; see [`docs/training.md`](../../docs/training.md) and [`docs/data.md`](../../docs/data.md)
- `name`: directory for output of training
- `network`: optional list of layers, each `{type: Linear, out: int, activation: LeakyReLU | Sigmoid | None}` with optional `in` (checked against the previous layer's `out`) and `dropout`; `None` gives a linear output. The last activation sets the output: `Sigmoid` (probability) for `eft`, `None` (log ratio) for the nuisance methods. Without it, the network is $n_{\rm features}$ x 128 x 128 x 1 with LeakyReLU and a Sigmoid output, which suits `eft` only
- `patience`: early stopping, number of epochs without improvement of the test loss before training stops (default 10)
- `scheduler`: learning-rate scheduler, `plateau` (default), `cosine` or `none`
    - `factor`: `plateau` only, multiplicative learning-rate decay (default 0.5)
    - `lr_patience`: `plateau` only, epochs without improvement before the learning rate is reduced (default 5)
    - `warmup_epochs`: `cosine` only, length of the linear warmup (default 5)
- `seed`: random seed used for network initialization and the train/test split
- `systematic`: nuisance methods (required); `weight_shift`: a name in `nominal/systematics.yml`; `kinematic_shift`: the variation, read from the directories `<systematic>Up` and `<systematic>Down`
- `threads`: optional number of CPU threads: torch's intra-op threads and the BLAS / OpenMP pools of numpy (`threadpoolctl`). Without it, the libraries' defaults (usually every core). Batches are built in the training process, so a run with `threads: 1` uses one core
- `wcs`: names of the $n_{\rm wcs}$ Wilson coefficients, in the order of the `coefs` columns; `c0` and `c1` follow this order
- `year_loc`: optional column of the features holding the integer era index $0, \dots, n_{\rm eras}-1$; used to look up `lumi` and `eras` (default 21)
- `zero_terms`: optional path to a yml listing WC pairs whose quadratic term is structurally zero; their columns are dropped from the weights (format: [`examples/data/zero_terms.yml`](../data/zero_terms.yml))
