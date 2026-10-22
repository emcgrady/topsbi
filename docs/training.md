# Training

What one training does and writes: the weight normalisation, the data splits, the training loop, the outputs, the metrics in `performance.yml`, the binning of the closure tests and the plots. The configuration keys are listed in [`examples/training/README.md`](../examples/training/README.md) and the per-feature keys in [`examples/features/README.md`](../examples/features/README.md). The derivations are in `main.tex` ([eft_nsbi_framework](https://github.com/emcgrady/eft_nsbi_framework)); notation follows it and the Notation section of `AGENTS.md`.

Run a training with

```sh
python -m topsbi.train <config.yml>
```

`device: cuda` trains on a GPU and falls back to the CPU with a warning when none is visible; `threads` sets the CPU threads (torch and numpy's BLAS). The data a training reads are described in [`data.md`](data.md).

## 1. Weight normalization

The event weights of both classes, in every split, are divided by the mean training $w^s(\theta_0)$ ($w^s(\theta, 0)$ for nuisance trainings). This keeps the weights $\mathcal{O}(1)$ for the optimizer without changing the learned ratio.

## 2. Data splits

- **Single-file eft data**: with `bootstrap: true`, `train.p` is split 75 / 25 into train and test by random rows (seeded by `seed`); otherwise `train.p` and `test.p` are used as given.
- **Combined data** (eft on `<root>/nominal`, every nuisance training): `train.p` is split 75 / 25 by a hash of each event's key and `seed` (`event_split`), so every training with a seed has the same test events, in nominal and in every variation.
- **Holdout**: `holdout`, by default `validation.p` for combined data. It is never used for training, early stopping or the learning-rate scheduler; the final metrics and plots are computed on it (on the test split when there is none).

## 3. Training loop

Each epoch shuffles the training rows with a generator seeded by `seed` (independent of the network's own random numbers) and steps through them in batches of `batchSize` (`epoch_batches`). The loss is the weighted binary cross entropy of the two samples, on a probability output for `eft` and on a log-ratio output (with `logits`, the $\text{Soft}^+$ form) for nuisance trainings. After every epoch the training and test losses are evaluated on the full splits, without `autograd`; the test loss drives the learning-rate scheduler (`scheduler`) and early stopping (`patience`), and the network with the lowest test loss is kept as `model.pt`.

Epoch numbering: `best_epoch` in `performance.yml` counts from 0 (the epoch whose updates gave the best network), while the loss plot and the animation frames count completed epochs, so the best network appears at `best_epoch + 1` there.

## 4. Outputs

Everything is written under the config's `name`:

- `model.pt`: the best network (lowest test loss), including the training-set feature normalization
- `complete/`: final plots, `performance.yml` and a copy of `model.pt`, computed on the holdout (`holdout`, by default `validation.p` for combined data), otherwise on the early-stopping test split
- `complete/finished`: an empty file, the last thing a training writes; a training without it did not finish (e.g. the job stopped while writing the final plots)
- `incomplete/epoch_XXXX/`: the summary plots and metrics every 50 epochs during training, on the test split (epoch 0, before any update, without `loss.png` and `learning_rate.png`)
- `incomplete/kinematics/<feature>/`: animation frames, drawn after training and deleted once `complete/animations/` is written

Existing `complete/` and `incomplete/` directories are deleted at the start of a run.

## 5. Performance metrics (`performance.yml`)

Every training type (`eft`, `weight_shift`, `kinematic_shift`) writes the same closure metrics, computed on the holdout (or the test split without one) by `closure_metrics` in `tools/metrics.py`. Each test compares, per bin, the reweighted reference yield $\hat w_b = \sum_i \hat r(x_i)\, w_{0,i}\, 1(x_i \in \text{bin})$ with the target yield $w_b = \sum_i w_{1,i}\, 1(x'_i \in \text{bin})$, each sample binned by its own features, with $\Delta_b = \hat w_b - w_b$ and its covariance $\mathrm{Cov}(\Delta_b, \Delta_c)$ summed over events, so an event in both samples counts once. For `eft` the reference and target are the same events weighted by $w^s(\theta_0)$ and $w^s(\theta_1)$; for `weight_shift` by $w^s(\theta, 0)$ and $w^s(\theta, \nu)$; for `kinematic_shift` they are the nominal and the shifted sample, which share most of their events.

Each `*_chi2_ndf` is $\Delta^T \mathrm{Cov}^+ \Delta / \mathrm{ndf}$, about 1 for a calibrated network, with `*_z` $= (\chi^2 - \mathrm{ndf})/\sqrt{2\,\mathrm{ndf}}$ about 0. They measure significance, so a fixed bias becomes more significant on a larger sample. Each `*_rms_bias` ($\Delta_{\rm rms}$ in the plots) is the yield-weighted RMS of $\Delta_b / w_b$ with the statistical scatter subtracted: the typical size of the misclosure as a fraction, which does not grow with the sample (it has a small positive floor from the subtraction in sparse bins). Read the two together: a large $\chi^2/\mathrm{ndf}$ with a tiny rms bias is a real but negligible residual.

| key | measures | calibrated network |
|---|---|---|
| `auc` | weighted AUC between the reference and target samples; separation power, blind to calibration | no reference value |
| `rate_closure`, `rate_pull` | $\sum \hat w / \sum w - 1$ over all events (signed normalisation misclosure) and its pull | 0, about $\pm 1$ |
| `r_chi2_ndf`, `r_ndf`, `r_z`, `r_rms_bias`, `r_worst_pull` | calibration in 50 equal-count bins $R_b$ of $\hat f = \log \hat r$ | 1, -, 0, 0, about $\pm 3$ |
| `group_chi2_ndf`, `group_ndf`, `group_z`, `group_rms_bias`, `group_worst_pull` | cells of the group columns (`closure_groups`) x 5 bins $R_b$ (cells with >= 200 reference events); catches miscalibration in regions of the inputs that cancels within the bins $R_b$ | 1, -, 0, 0, about $\pm 3$ |
| `feature_chi2_ndf`, `feature_rms_bias` | per feature, in its `features.yml` binning $K_b$ merged to $n_{\rm eff} \ge 30$ | 1, 0 per feature |
| `feature_worst`, `feature_worst_pull` | the feature with the largest $\chi^2/\mathrm{ndf}$ and its largest bin pull, for diagnosis | - |
| `r_pull`, `r_pull_err` | in the bins $R_b$: mean $\lvert\Delta_b\rvert / \sqrt{\mathrm{Cov}_{bb}}$ minus $\sqrt{2/\pi}$ (its value for standard normal pulls); robust to a single outlying bin | 0 within about $2\times$`r_pull_err` |
| `r_ci_diff`, `r_ci_diff_err` | in the bins $R_b$: expected minus observed fraction of bins within 1, 2, 3 $\sigma$; the shape of the pull distribution, including its tails | 0, 0, 0 within about $2\times$`r_ci_diff_err` |
| `nominal_*` | every test above with $\hat r = 1$: the size of the EFT effect or the systematic before any correction | - |
| `best_epoch`, `best_test_loss` | epoch and test loss of the saved network | - |

The `_err` keys give the scatter a calibrated network would show with as many bins.

Group closure is only defined for groups built from network inputs: grouping by a latent label, e.g. a generator-level $m_{t\bar t}$ slice, does not close even for a perfect network.

## 6. Binning of the closure tests

- **Features, bins $K_b$.** Each feature starts from `nbins` equal bins between `min` and `max` (`features.yml`); neighbouring bins are merged until each holds at least 30 effective events of the target sample, $n_{\rm eff} = (\sum w)^2 / \sum w^2$ (`merge_sparse_bins`, summed in float64 so the bins do not depend on the CPU). Sparse tails therefore end up in wider bins instead of noisy ones. Events outside `[min, max]` are put into the first and last bins, so the range should contain nearly all events (see the `min` / `max` keys).
- **Learned ratio, bins $R_b$.** 50 equal-count bins of $\hat f = \log \hat r$, with edges from the reference sample; target events outside them go into the edge bins.
- **Groups.** The group columns (`closure_groups`) x 5 bins $R_b$ inside each group, keeping cells with >= 200 reference events. The group columns must be network inputs. The default suits this analysis's layout: the era (`year_loc`) and njets (column 19) with values >= 4 merged; set `closure_groups` for other data.

The numbers of bins and the thresholds (50 bins $R_b$, 5 per group, 200 reference events per cell, $n_{\rm eff} \ge 30$) are defaults of `closure_metrics`, not configuration keys.

## 7. Plots in `complete/`

Every training type:

- `loss.png`, `loss.npz`: the training and test loss over epochs (from before training on), each relative to its value at the best network, $L - L_{\rm best}$ (the lowest test loss, saved as `model.pt`), which removes the different loss levels of the two samples, on a y axis linear up to the largest change after the first epoch and logarithmic beyond (so the first epoch's drop and the much smaller later changes are both visible); overfitting shows as the test curve rising after the best network while the training curve keeps falling. The `.npz` holds the full history (arrays `train`, `test`; entry 0 before training)
- `learning_rate.png`: learning rate per epoch
- `kinematics/<feature>.png`: closure of every feature in its bins $K_b$ (the `feature_*` keys). The bins are drawn with equal widths and labelled by their edges (integer features one label per value, the last bin "≥k"; eras by the names in `eras`).
  - Top: the target yield $w_b$ (black) and the reweighted reference yield $\hat w_b$ (blue), in expected events per bin at the luminosity of the eras in the sample (with `coefs` and `lumi` normalised as in [`data.md`](data.md)), each with its MC error $\sqrt{\sum_i w_i^2}$; log scale when the yields span more than a factor 50.
  - Bottom: $\Delta_b / w_b$ (filled) and the uncorrected residual with $\hat r = 1$ (hollow) inside the band $\pm\sqrt{\mathrm{Cov}_{bb}}/w_b$; points beyond the range are drawn as arrows at its edge, and bins with $w_b = 0$ are hatched.
  - Legend: $\chi^2/\mathrm{ndf}$, $\Delta_{\rm rms}$ and $Z$ of the test. Labels show $\theta$ for `eft` and $\theta,\nu$ for nuisance trainings.
- `r_closure.png`: the same layout in the 50 bins $R_b$ (the `r_*` keys), the bins labelled by their edges in $\hat r$
- `group_closure.png`: pull of each group/$\hat r$ cell; points of one group share a tick label (one line per group column, by default era above, jets below) and run from low to high $\hat r$
- `animations/<feature>.gif`: the closure plot of each feature in `features_to_animate` on the test split, every `animate_per_epoch` epochs and after the last epoch (frames labelled by completed epochs, 1 s each), ending on the restored best network ("Best: epoch N", held 3 s). The y ranges cover the best network and every frame after it, so changes after the best epoch are seen at their size; earlier frames beyond the range are drawn as arrows

Paired data (`eft`, `weight_shift`) only, from the per-event true ratio $r(x, z) = w_1 / w_0$:

- `roc.png`: weighted ROC curve
- `r_spread.png`: in the bins $R_b$ of `r_closure.png` (equal-width, labelled by their edges in $\hat r$), relative to $\langle r \rangle_b$: the mean difference $\langle r \rangle_b - \langle \hat r \rangle_b$ (black) with its uncertainty (blue) and the per-event spread $\sqrt{\mathrm{Var}(r - \hat r)}$ (light band), the part of the true ratio that depends on the latent $z$ and cannot be learned from $x$, which no closure plot shows. The calibration itself is in `r_closure.png`
