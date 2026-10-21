## Configuration file fields

What a training writes, its metrics and plots: [`docs/training.md`](../../docs/training.md).


- `animate_per_epoch`: how often (in epochs) a frame is added to the closure animations of `features_to_animate` (5 in the examples)
- `batchSize`: number of events to feed the network at once
- `bootstrap`: if `true`, split `train.p` 75/25 into training and test sets; if `false`, load `train.p` and `test.p` separately. Ignored for a combined sample (see `data`)
- `c0`: SM-inclusive WC point $\theta_0$ (first entry 1) of the label-0 events
- `c1`: SM-inclusive WC point $\theta_1$ of the label-1 events
- `closure_groups`: optional columns of the group closure test, in order, each `{loc: column, label: text, cap: value}` (`label` and `cap` optional; values above `cap` are merged into it and labelled "≥cap"). Group columns must be network inputs. Default: the era (`year_loc`) and njets, `[{loc: 21}, {loc: 19, label: jets, cap: 4}]`
- `data`: directory containing torch `TensorDataset`s with `(features [N, 22], structure constants` $s_q(z)$ `[N, 153], ids [N, 3])`. Training reads `train.p`, and also `test.p` when `bootstrap` is `false`. For `eft` on a combined sample, `data` is its `nominal/` directory (recognised by its `sample.yml`): `train.p` is split 75/25 by a hash of each event's key (`event_split`, the same split as the nuisance trainings with the same `seed`) and the holdout defaults to `validation.p`. For nuisance methods, `data` is the combined sample's root
- `device`: The device used for training. Use `cpu` for CPU training and `cuda` for GPU training. For more options, see the [PyTorch documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device).
- `epochs`: maximum number of epochs used to train
- `eras`: optional era names indexed like `lumi`, used for the era tick labels (default `[2016APV, 2016, 2017, 2018]`); must have as many entries as `lumi`
- `features`: path to the feature configuration yaml (see `examples/features/README.md`)
- `features_to_animate`: features from the feature configuration whose closure plot is animated over the epochs (`animations/<feature>.gif`)
- `holdout`: optional path to a dataset not used for training, early stopping or the learning-rate scheduler, e.g. the `validation.p` produced alongside `train.p` (the default for a combined sample). When set, the final plots and `performance.yml` in `complete/` are computed on it. Without it they are computed on the early-stopping test split and are slightly optimistic
- `learningRate`: initial learning rate for network training
- `lumi`: luminosity per era, indexed by the `year_int` feature (0: UL16APV, 1: UL16, 2: UL17, 3: UL18), e.g. `[19.52, 16.81, 41.48, 59.83]`. Required by every method; only relative values matter for training
- `method` (required):
    1. `eft`: EFT training with the stitched event weights $w^s(\theta) = L\,s_q(z)\cdot T(\theta) \times 1000$, which keep the per-sample cross-section normalization, so the samples are mixed by cross section times luminosity
    1. `weight_shift`, `kinematic_shift`: nuisance trainings with the soft$^+$ loss (keys `systematic`, `direction`; see AGENTS.md)
- `name`: directory for output of training
- `network`: optional list of layers, each `{type: Linear, out: int, activation: LeakyReLU | Sigmoid | None}` with optional `in` (checked against the previous layer's `out`) and `dropout`; `None` gives a linear output. Without it, the network is 128 x 128 x 1 with LeakyReLU and a Sigmoid output
- `patience`: early stopping, number of epochs without improvement of the test loss before training stops (default 10)
- `scheduler`: learning-rate scheduler, `plateau` (default), `cosine` or `none`
    - `factor`: `plateau` only, multiplicative learning-rate decay (default 0.5)
    - `lr_patience`: `plateau` only, epochs without improvement before the learning rate is reduced (default 5)
    - `warmup_epochs`: `cosine` only, length of the linear warmup (default 5)
- `seed`: random seed used for network initialization and the train/test split
- `threads`: optional number of CPU threads: torch's intra-op threads and the BLAS / OpenMP pools of numpy (`threadpoolctl`). Without it, the libraries' defaults (usually every core). Batches are built in the training process, so a run with `threads: 1` uses one core
- `wcs`: array of WC names for sample
- `year_loc`: optional column of the `year_int` feature, used to look up `lumi` and `eras` (default 21)
