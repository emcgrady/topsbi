# Documentation

The analysis runs in four stages; each page describes one stage with the same sections: what it does and reads, how to run it, its outputs, its metrics (each stored key, what it measures and its value when everything is in order) and its plots. Configuration keys are listed next to the example configs in `examples/<stage>/README.md`. Derivations are in `main.tex` ([eft_nsbi_framework](https://github.com/emcgrady/eft_nsbi_framework)), and the notation and code-name mapping in `AGENTS.md` (Notation).

1. **Training** ([`training.md`](training.md)): one network per training point or nuisance direction (`python -m topsbi.train`), with its closure metrics and plots.
2. **Ensemble** (`ensemble.md`, not yet written): the fit of the per-event quadratic coefficients over the networks of each seed, averaged over seeds, and its cache under `{name}/ensemble/` that validation and the scanner read.
3. **Validation** (`validation.md`, not yet written): `python -m topsbi.validation`, the last check before likelihood scans: bias test per component, attribution, suspect networks and the point search.
4. **Likelihood scans** (`likelihood_scans.md`, not yet written): the scanner, once it is a script.
