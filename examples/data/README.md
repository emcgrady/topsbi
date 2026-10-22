## Data files

Examples of the yml files that accompany the training data, laid out as a combined data root (`<root>/nominal/` and one
directory per kinematic variation). The tensors themselves are described in [`docs/data.md`](../../docs/data.md).

## `nominal/systematics.yml`

Read by `weight_shift` trainings, next to the nominal `train.p` / `validation.p`.

- `syst_ratio`: names of the weight systematics, in the order of axis 1 of the `syst_ratio` tensor; a config's
  `systematic` must be one of them
- `directions`: order of the tensor's last axis (default `[Up, Down]`); a config's `direction` must be one of them

## `nominal/sample.yml`, `<variation>/sample.yml`

Next to `train.p` / `validation.p` in every directory of the combined data.

- presence: marks the directory as combined data (an `eft` training on `<root>/nominal` then splits train / test by
  event key and uses `validation.p` as the default holdout)
- `reference`: the point at which `w_ref` is the event weight; must be the same in nominal and every variation
- `salt_split`: the salt of the train / validation split; must be the same in nominal and every variation
- any other keys are free and not read (e.g. `variation`, `columns`)

## `zero_terms.yml`

Referenced by the training config's `zero_terms` key; it depends on the samples, not on a single training.

- `zero_terms`: list of WC pairs (names as in the config's `wcs`, any order within a pair) whose quadratic term is
  structurally zero; their columns of the `get_lower_tri` expansion are dropped
