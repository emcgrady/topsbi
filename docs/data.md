# Data

What topsbi needs from the data a training reads. Sizes are not fixed: the number of features comes from the data
and the number of structure constants from the configured WCs. Settings that describe where things are in the data
(the era column, the closure groups) are configuration keys, listed in
[`examples/training/README.md`](../examples/training/README.md).

## 1. The minimum: eft training on a single file

The config's `data` directory holds `train.p` (and `test.p` unless `bootstrap: true`; a holdout can be given with
`holdout`). Each file is a pickled `torch.utils.data.TensorDataset` whose first two tensors are:

- **`features`**, [$n_{\rm events}$, $n_{\rm features}$] float: the network inputs, raw (the network standardises them with
  training-set statistics). Any number of features $n_{\rm features}$. One column, `year_loc`, holds an integer era index $0, \dots, n_{\rm eras} - 1$.
- **`coefs`**, [$n_{\rm events}$, $n_q$] float: per-event structure constants $s(z)$, such that the event weight at a point
  $\theta = (1, c_1, \dots, c_{n_{\rm wcs}})$ is $s(z) \cdot T(\theta)$, where $T(\theta)$ = `get_lower_tri(theta)` lists
  $\theta_i \theta_j$ for $j \le i$, so $n_q = (n_{\rm wcs}+1)(n_{\rm wcs}+2)/2$. They include the per-sample normalisation (cross section
  over sum of weights) but not the luminosity: topsbi multiplies by `lumi[era]` and by 1,000, so `coefs` in pb with
  `lumi` in $fb^{-1}$ gives expected events.

Further tensors are ignored. The config sets `wcs` (the $n_{\rm wcs}$ names, in the order of `coefs`), `c0` and `c1`
($n_{\rm wcs} + 1$ values each, the first 1), `lumi` ($n_{\rm eras}$ values) and `year_loc`, and its `features` yml has one entry (`loc` < $n_{\rm features}$) per
feature to bin and plot ([`examples/features/README.md`](../examples/features/README.md)).

## 2. The combined format: nuisance trainings (optional for eft)

Nuisance trainings compare two samples of the same events, so they need one directory per sample:

```
<root>/
  nominal/                train.p  validation.p  systematics.yml  sample.yml
  <systematic>Up/         train.p  validation.p  sample.yml
  <systematic>Down/       train.p  validation.p  sample.yml
```

A kinematic variation may cover only some eras; its nominal sample is then restricted to those eras. For eft on
combined data, `data` is `<root>/nominal`; for nuisance trainings it is `<root>`.

Tensor order in `train.p` and `validation.p`:

| sample | tensors |
|---|---|
| nominal | `features`, `coefs`, (any), `event_id`, `syst_ratio`, `w_ref` |
| variation | `features`, `w_ref`, (any), `event_id` |

- **`event_id`**, [$n_{\rm events}$, $n_{\rm id}$] integer, any $n_{\rm id}$: the identity of each event. It must be unique within a file and the same for
  the same event in nominal and in every variation: it drives the train / test split (`event_split`) and the
  matching of nominal and varied events in the kinematic_shift metrics. The event key also includes column 2 of the
  third tensor.
- **`syst_ratio`**, [$n_{\rm events}$, $n_{\rm systematics}$, 2] float: $w(\theta, \nu) / w(\theta, 0)$ of each of
  $n_{\rm systematics}$ weight systematics, in the order of
  `systematics.yml`; the last axis holds the directions. Only weight_shift reads it.
- **`w_ref`**, [$n_{\rm events}$, 1] float: the event weight at one fixed reference point, normalised as `coefs`; nominal and the
  variations use the same point.
- The third tensor is otherwise not read; any [$n_{\rm events}$, ...] tensor.

**Split.** `validation.p` (the holdout) must be disjoint from `train.p` and split by event identity, so every event
is on the same side in nominal and in every variation. topsbi splits `train.p` into train and test itself, with a
hash of the event key and the training seed.

**`nominal/systematics.yml`** (weight_shift; [example](../examples/data/nominal/systematics.yml)): `syst_ratio`, the systematic names in column order, and `directions`
(default `[Up, Down]`).

**`sample.yml`** ([nominal](../examples/data/nominal/sample.yml), [variation](../examples/data/JES_AbsoluteUp/sample.yml)): marks a directory as combined data (needed for eft on combined data). When nominal and a variation
both have one, their `reference` (the reference point of `w_ref`) and `salt_split` (the salt of the train /
validation split) must agree. Other keys are free.

## 3. Settings tied to the data layout

| config key | default | set it when |
|---|---|---|
| `year_loc` | 21 | the era index is in another column |
| `eras` | the era index | the eras should be labelled by name (names indexed like `lumi`) |
| `closure_groups` | era, and njets (column 19) with values >= 4 merged | other network inputs should define the closure groups |
| `zero_terms` | none | some WC pairs have structurally zero quadratic terms (a yml with `zero_terms`: a list of pairs; [example](../examples/data/zero_terms.yml)) |

## 4. What topsbi checks

Checked when loading: the number of tensors in each file, `reference` and `salt_split` between nominal and a
variation, the eras of a variation's `train.p` and `validation.p`, a non-empty test split, the era and closure-group
columns against the number of features, and (validation) `lumi` covering every era index. Not checked: that
`event_id` is unique and that `validation.p` was split by event identity.

## 5. One way to produce it

The reco_systematics analysis builds its data with ttbarEFT:
[`analysis/tensor_processor.py`](https://github.com/TopEFT/ttbarEFT/blob/main/analysis/tensor_processor.py) writes
per-chunk tensors from NanoAOD (features, structure constants, event identity, weight-systematic ratios), and
[`analysis/combine_tensors.py`](https://github.com/TopEFT/ttbarEFT/blob/main/analysis/combine_tensors.py) merges
them per variation, removes duplicate events, adds `w_ref` and makes the event-key train / validation split and
`sample.yml`. Its v3 production is `/groups/klannon/cmcgrad2/data/reco_systematics_v3/SR_CHANNELS_2j3j_gen`.
