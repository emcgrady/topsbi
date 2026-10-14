## Configuration file fields

- `animate_per_epoch`: how often (in epochs) a frame is added to the kinematic animations
- `batchSize`: number of events to feed the network at once
- `bootstrap`: if `true`, split `train.p` 75/25 into training and test sets; if `false`, load `train.p` and `test.p` separately
- `c0`: SM-inclusive set of WCs for label-0 events
- `c1`: SM-inclusive set of WCs for label-1 events
- `data`: directory containing torch `TensorDataset`s with `(features [N, 22], quadratic fit coefficients [N, 153], ids [N, 3])`. Training reads `train.p`, and also `test.p` when `bootstrap` is `false`
- `device`: The device used for training. Use `cpu` for CPU training and `cuda` for GPU training. For more options, see the [PyTorch documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch-device).
- `epochs`: maximum number of epochs used to train
- `features`: path to the feature configuration yaml (see `examples/features/README.md`)
- `features_to_animate`: features from the feature configuration that get per-epoch animations
- `holdout`: optional path to a dataset not used for training, early stopping or the learning-rate scheduler, e.g. the `validation.p` produced alongside `train.p`. When set, the final plots and `performance.yml` in `complete/` are computed on it. Without it they are computed on the early-stopping test split and are slightly optimistic
- `learningRate`: initial learning rate for network training
- `lumi`: luminosity per era, indexed by the `year_int` feature (0: UL16APV, 1: UL16, 2: UL17, 3: UL18), e.g. `[19.52, 16.81, 41.48, 59.83]`. Required by `weights_only`; only relative values matter for training
- `method`: 2 options
    1. `weights_only` (recommended): train with the stitched event weights `w^s(c) = L_y * coefs @ T(c)`, which keep the per-sample cross-section normalization, so the samples are mixed by cross section times luminosity
    1. `stitched`: weights `coefs @ T(c)` normalized by their mean, without luminosity
- `name`: directory for output of training
- `network`: optional argument used for variable network architecture
- `patience`: early stopping, number of epochs without improvement of the test loss before training stops (default 10)
- `scheduler`: learning-rate scheduler, `plateau` (default), `cosine` or `none`
    - `factor`: `plateau` only, multiplicative learning-rate decay (default 0.5)
    - `lr_patience`: `plateau` only, epochs without improvement before the learning rate is reduced (default 5)
    - `warmup_epochs`: `cosine` only, length of the linear warmup (default 5)
- `seed`: random seed used for network initialization and the train/test split
- `wcs`: array of WC names for sample
- `year_loc`: optional column of the `year_int` feature used to look up `lumi` (default 21)

The event weights of both classes, in every split, are divided by the mean training weight under `c0`. This keeps the weights O(1) for the optimizer without changing the learned ratio.

## Outputs

Everything is written under `name`:

- `model.pt`: the best network (lowest test loss), including the training-set feature normalization
- `complete/`: final plots and `performance.yml`, computed on `holdout` when it is set
- `incomplete/epoch_XXXX/`: the same plots every 50 epochs during training, on the test split

## Performance metrics (`performance.yml`)

The calibration metrics check that the learned ratio $\hat r(x)$ equals the average true ratio $\langle r(x,z)\rangle$ of the events it is applied to. They are weighted by the stitched weights under `c0`, so runs with different settings are directly comparable. Each is 0 for a perfectly calibrated network and positive when the calibration can be improved; the `_err` keys give the scatter a perfect network would show on a test set of the same size, and a value within about twice its `_err` is consistent with 0.

| key | measures | perfect network |
|---|---|---|
| `auc` | separation power of the classifier; blind to calibration | no reference, compare runs at the same `c0`, `c1` |
| `pull`, `pull_err` | mean of $\lvert\langle r\rangle - \langle\hat r\rangle\rvert / \sigma$ over 50 quantile bins of $\hat r$, minus $\sqrt{2/\pi}$ | 0 |
| `ci_diff`, `ci_diff_err` | expected minus observed fraction of those bins within 1, 2, 3 $\sigma$ | 0, 0, 0 |
| `rms_bias` | weighted RMS of the relative bias $(\langle\hat r\rangle - \langle r\rangle)/\langle r\rangle$ over the same bins, with the statistical scatter subtracted; the typical size of the miscalibration as a fraction | 0 |
| `group_chi2_excess`, `group_chi2_excess_err` | mean pull$^2$ minus 1 over cells of era x njets (>= 4 merged) x 5 $\hat r$ bins; catches miscalibration in regions of the inputs that cancels within $\hat r$ bins | 0 |
| `group_rms_bias` | the size of that regional bias, as a fraction | 0 |
| `group_worst_pull` | the largest single-cell pull, for diagnosis | about $\pm 3$ for ~60 cells |

`pull` and `ci_diff` measure significance, so a fixed bias becomes more significant on a larger test set; `rms_bias` and `group_rms_bias` measure size and do not grow with the test set. Group closure is only defined for groups built from network inputs: grouping by a latent label such as the generator-level $m_{t\bar t}$ slice does not close even for a perfect network.

## Plots in `complete/`

- `loss.png`, `lossLog.png`, `lr.png`: training and test loss, learning rate
- `roc.png`: weighted ROC curve
- `lr_excl.png`, `lr_incl.png` (and `_log`): calibration curve $\langle r\rangle$ vs $\langle\hat r\rangle$ in quantile bins of $\hat r$; `excl` drops the outer 1% of $\hat r$ on each side, `incl` keeps them
- `mean_diff_*.png`: $\langle r\rangle - \langle\hat r\rangle$ per bin with its error; `std_diff_*.png` also shows the per-event spread
- `group_closure.png`: pull of each era/njets/$\hat r$ cell; points of one group share a tick label (era above, jets below) and run from low to high $\hat r$
- `kinematics/`: learned and true yields $\sum w_0 \hat r$ and $\sum w_0 r$ per feature, with their ratio
- `animations/`: the kinematic plots of `features_to_animate` over the epochs
