## Configuration file fields

The top-level keys name the features; they are used for naming plots when saved in the validation and training scripts.

## Configuration file sub-fields

- `label`: label to use for axis labeling
- `loc`: index of the kinematic variable (column of the feature tensor)
- `min`: lower bound of binning
- `max`: upper bound of binning
- `nbins`: number of bins of the finest binning
- `unit`: optional unit of the variable, e.g. `GeV`; shown on the axes

## Binning

`kinematic_histogram` starts from `nbins` equal bins between `min` and `max`, then merges neighbouring bins until each holds at least 30 effective events, $n_{\rm eff} = (\sum w)^2 / \sum w^2$. Sparse tails therefore end up in wider bins instead of noisy ones, and the main panel shows yields per unit of the variable. Events outside `[min, max]` are put into the first and last bins, so `min` and `max` should contain nearly all events:

- Signed variables such as single-object $\eta$ need a symmetric range, e.g. `min: -2.4`, `max: 2.4`.
- Lower bounds should sit at the selection thresholds (e.g. lepton $p_T$ 40, b-jet $p_T$ 30, $m_{\ell\ell}$ 20). A range starting below a threshold leaves empty bins that are merged into an artificially wide first bin.
