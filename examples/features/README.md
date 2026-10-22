## Configuration file fields

The top-level keys name the features; they are used for naming plots when saved in the validation and training scripts.

## Configuration file sub-fields

- `label`: label to use for axis labeling
- `loc`: index of the kinematic variable (column of the feature tensor)
- `min`, `max`: range of the binning. Events outside it go into the first and last bins, so the range should contain nearly all events: a symmetric range for signed variables such as single-object $\eta$ (e.g. `min: -2.4`, `max: 2.4`), and lower bounds at the selection thresholds (e.g. lepton $p_T$ 40, b-jet $p_T$ 30, $m_{\ell\ell}$ 20); a range starting below a threshold leaves empty bins that are merged into an artificially wide first bin
- `nbins`: number of bins of the finest binning, before sparse bins are merged ([`docs/training.md`](../../docs/training.md), Binning)
- `unit`: optional unit of the variable, e.g. `GeV`; shown on the axes
