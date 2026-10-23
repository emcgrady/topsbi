# Validation

The last check of an ensemble before it is used for likelihood scans: whether every coefficient of the learned ratio is unbiased, which training networks cause a bias, whether removing them helps, and which new training points would reduce the uncertainty that remains.

```sh
python -m topsbi.validation -c <config.yml> [--refit]
```

The configuration keys are listed in [`examples/validation/README.md`](../examples/validation/README.md). The ensemble it tests, its cache and `--refit` are described in [`ensemble.md`](ensemble.md); notation as there: $a$ a training network (row of $\Theta$), $e$ an ensemble member, $q$ a kept term of $T(\theta)$ (`k` in the code), $\langle \cdot \rangle_e$ the mean over members.

## 1. What it does and reads

1. **Ensemble.** Read the cache `{name}/ensemble/`, or fit it with `--refit`.
2. **Bias test** (section 3). Compare every component $\hat\gamma_q$ (row $q$ of $\hat\Gamma$) with its truth in bins of every network input, and flag the components with a significant bias.
3. **Attribution** (section 4). Split the bias of each flagged component exactly into the contributions of the training networks.
4. **Suspects and drop tests** (section 5). Networks that carry a large share of a bias and close badly on their own are proposed for dropping or retraining; each drop is tested by refitting without it.
5. **Point search** (section 6). Propose new training points that most reduce the uncertainty of the flagged components.

Besides the ensemble it reads `data` (the events the ensemble is evaluated on; the per-event structure constants $s_q(z)$ in it give the truth), `lumi` (luminosity per era, column `year_loc`), `c0` and `features` (the network inputs and their columns). Outputs go to `{name}/`: `ensemble_diagnostics.yml`, `proposed_points.yml` and `plots/`.

## 2. Running it

Without `--refit` the cache must exist and match the config ([`ensemble.md`](ensemble.md), section 3). The steps after the fit read it memory-mapped. `workers` (default 1) also runs the per-member steps in parallel, one process per member: the binned sums of every network (section 4), and the refits of the drop tests (section 5). Their memory per worker is of the order of one member's $R$ in float64; the needs of the fit itself are given in [`ensemble.md`](ensemble.md). The bias test, the attribution and the point search are fast compared with the fit.

## 3. Bias test

**Binning.** Each network input $f$ is divided into `bias.n_bins` equal-count bins (one bin per value for a discrete input), and neighbouring bins are merged until each holds `bias.min_neff` effective events, $n_{\rm eff} = (\sum w)^2 / \sum w^2$ with $w = w^s(\theta_0)$. These are the bins $K_b$, $b \in B_f$.

**Bias.** For component $q$ the true per-event contribution is $w_{q,i} = L\, s_q(z_i) \times 1000$ and the learned one $\hat w_{q,i} = w^s_i(\theta_0)\, \hat\gamma_{q,i}$. Per bin:

- $\Delta_{q,b} = \left\langle \sum_{i \in K_b} (\hat w_{q,i} - w_{q,i}) \right\rangle_e$, the bias of the ensemble mean (each member sums over the events it keeps);
- $\mathrm{Var}_{\rm MC} = \sum_{i \in K_b} (\langle \hat w_{q,i} \rangle_e - w_{q,i})^2$, its variance from the finite number of events;
- $\mathrm{Var}_{\rm seeds}$, the variance of the ensemble mean over members (the variance of the members' sums divided by their number).

**Z.** Per component and input, over the bins with events of the component, with $\rho_b = \mathrm{Var}_{\rm seeds} / \mathrm{Var}_{\rm MC}$:

$$\chi^2_{q,f} = \sum_{b \in B_f} \frac{\Delta_{q,b}^2}{\mathrm{Var}_{\rm MC}}, \qquad \mu = \sum_b (1 + \rho_b), \qquad V = 2 \sum_b (1 + \rho_b)^2, \qquad Z_{q,f} = \frac{\chi^2_{q,f} - \mu}{\sqrt{V}}.$$

Without bias each term is $(1 + \rho_b)$ times a $\chi^2_1$ variable, so $\mu$ and $V$ are the mean and variance of $\chi^2$ and $Z$ has mean 0 and variance 1. $Z$ is a screening statistic, not a Gaussian significance: the $\chi^2$ of a few tens of bins is right-skewed, so a large $Z$ overstates how unlikely the excess is. A component is flagged if its largest $Z$ over the inputs exceeds `bias.z_threshold`. The $Z$ of different inputs of one component are correlated, since all inputs bin the same events.

**rms bias.** The size of the bias, per component and input:

$$\text{rms bias}_{q,f} = \sqrt{\max\left(\left\langle \frac{\Delta_{q,b}^2 - \mathrm{Var}_{\rm MC} - \mathrm{Var}_{\rm seeds}}{\lambda_{K_b}(\theta_{\rm SM})^2} \right\rangle_{b \in B_f},\ 0\right)},$$

the noise-subtracted rms bias per bin as a fraction of the SM yield $\lambda_{K_b}(\theta_{\rm SM})$, per unit WC (or WC$^2$ for quadratic and cross terms). $Z$ measures significance and grows with the number of events; the rms bias measures size and does not. Read them together: a large $Z$ with a tiny rms bias is a real but negligible bias.

## 4. Attribution

Since $\hat\Gamma = \Theta^+ R$ for each member, the binned bias of a component is exactly a sum over the training networks,

$$\Delta_{q,b} = \sum_a (\Theta^+)_{q,a}\, \Delta_{a,b},$$

with $\Delta_{a,b} = \sum_{i \in K_b} w^s_i(\theta_0) \left(\hat r_a(x_i) - r_a(x_i, z_i)\right)$ the binned misclosure of network $a$. The share of network $a$ in the bias of component $q$ is the projection of its term on $\Delta_q$, weighted by $1 / \lambda_{K_b}(\theta_{\rm SM})^2$ and averaged over members; the shares of a component sum to 1 and can be negative (a network that cancels part of the bias). `attribution_identity` checks that the terms rebuild $\Delta$ (up to float32 rounding).

Each network's own closure, from the same sums, gives its misclosure $m_a$: the noise-subtracted rms of $\Delta_{a,b} / \lambda_{K_b}(\theta_{\rm SM})$ over bins, with $m_a^2$ averaged over members.

## 5. Suspects and drop tests

A network is a suspect if its share in some flagged component reaches `suspects.share_threshold`. It closes badly in a member if its own $Z$ (the test of section 3 on $\Delta_{a,b}$) exceeds `suspects.z_threshold` there. Bad in at least `suspects.seed_fraction` of the members: the training point itself is the problem, and it is a drop candidate. Bad in fewer: the bad copies are proposed for retraining (`retrain`) and the point stays.

Drop candidates are grouped (the points of one cross-term pair together, a single-WC point alone) and each group is tested by refitting every member without it. A drop is accepted if the largest $Z$ of some component it was blamed for falls by at least `suspects.min_gain`, and no component ends above `bias.z_threshold` with its largest $Z$ risen by more than `suspects.tolerance`. A group whose removal would leave $\Theta$ rank-deficient is not tested. Drops that pass alone may interact, so all accepted drops are then refitted together (`combined`).

The refits use the stored $R$, so no network is evaluated again.

## 6. Point search

With the misclosures as independent network errors, the propagated covariance of the coefficients for a design $\Theta$ is

$$\mathrm{Cov}(\hat\gamma) = H^+ \Theta^T \mathrm{diag}(m_a^2)\, \Theta\, H^+, \qquad H = \Theta^T \Theta,$$

and $\sqrt{\mathrm{Var}(\hat\gamma_q)}$ its diagonal (`sigma_propagated`). Components outside the range of $\Theta$ (a part of their direction in its null space) cannot be determined by the design; they are marked `constrained: false` and get no uncertainty.

The search adds candidate units one at a time, each time the one that most lowers the objective $\mathrm{tr}(\mathrm{Cov}(\hat\gamma)\, G)$, and stops after `point_search.n_units` units or when a step lowers it by less than the fraction `point_search.min_gain`.

- **Weights.** `G: flagged` (default): $G = \mathrm{diag}(g_q / \mathrm{Var}(\hat\gamma_q))$ with the variance of the current design, $g_q = 1$ for flagged components and `unflagged_weight` for the others, so the objective is the $g$-weighted sum of relative variance reductions. `G: scan`: $G = \langle T(\theta) T(\theta)^T \rangle$ over a scan region (`bounds` per WC, `mode` box or 1d, `shape` uniform or gaussian), the uncertainty of $\hat r$ averaged over the region.
- **Candidates.** For every cross term $\theta_i \theta_j$, units of the four points $(\pm\theta_i, \pm\theta_j)$ built from the existing single-WC training values, with the misclosure of the worse of the two source networks assumed for the new networks; and the dropped points, as candidates to retrain. Units whose points all exist, or with a point whose inclusive yield relative to the SM exceeds `max_yield_ratio` (default: the largest of the existing points), are left out.

The dropped networks leave the design before the search. `sigma_after_search` is the propagated uncertainty with the chosen units added. The search assumes the new networks close as well as existing ones; it cannot fix a bias that every network shares.

## 7. Outputs

### `ensemble_diagnostics.yml`

| Key | Measures | Healthy ensemble |
|---|---|---|
| `ensemble.path`, `created`, `seeds`, `rank` | the cache used; rank of each member's $\Theta$ | `rank` = number of kept terms |
| `ensemble.residual_outliers` | members whose median fit residual is more than 3 times the typical one | empty |
| `binning.<input>` | number of bins $K_b$ of each input after merging | |
| `flagged` | components with largest $Z$ above `bias.z_threshold`, largest first | empty |
| `attribution_identity` | max $\lvert$ rebuilt $-\ \Delta \rvert$ / max $\lvert \Delta \rvert$ (section 4) | float32 rounding, about $10^{-3}$ |
| `components.<q>.kind` | `sm`, `linear`, `quadratic` or `cross` | |
| `components.<q>.Z`, `max_Z`, `worst_feature` | $Z_{q,f}$ per input, its largest value and that input | $\lvert Z \rvert$ of a few at most, below the threshold |
| `components.<q>.rms_bias` | the largest rms bias over the inputs | small compared with the precision the analysis needs |
| `components.<q>.flagged` | in `flagged` | false |
| `components.<q>.constrained` | determined by the design (section 6) | true |
| `components.<q>.sigma_propagated`, `sigma_after_search` | $\sqrt{\mathrm{Var}(\hat\gamma_q)}$ for the current design and with the proposed points; null if unconstrained | below the rms bias the analysis can tolerate |
| `components.<q>.top_networks` | flagged components: the `n_top_networks` largest signed shares | |
| `components.<q>.refit.max_Z`, `refit.rms_bias` | after refitting without the accepted drops | |
| `suspects.drop` | networks dropped (accepted drop tests) | empty |
| `suspects.retrain` | {network: members whose copy closes badly} | empty |
| `suspects.drop_tests.<group>.members`, `accepted`, `gain`, `harmed` | the networks of a group; accepted; largest fall of $Z$ among the components it was blamed for; {component: [Z before, Z after]} for harmed components, or why the group could not be tested | |
| `suspects.combined` | the same for all accepted drops at once | |
| `network_misclosure.<network>` | $m_a$ of every training network (section 4) | similar for all networks of a kind |
| `point_search.G` | the weights used | |
| `point_search.candidate_units` | number of candidate units | |
| `point_search.objective` | objective before the search and after each added unit | |
| `point_search.proposed` | the chosen units: kind, point names, largest yield ratio of their points | |

The settings of a run (thresholds, `n_units`, ...) are those of its config.

### `proposed_points.yml`

`wcs` (the WC order) and `points`, each with `name`, `kind` (`cross` or `retrain`) and `theta` (SM-inclusive, first entry 1): the new training points, ready to be turned into training configs.

## 8. Plots (`plots/`)

| File | Shows | Look for |
|---|---|---|
| `fit_residuals.png`, `fit_residuals_log.png` | distribution of the per-event fit residuals: median over members, band of their spread, outlying members as separate lines | a member far from the others |
| `z_heatmap_full.png`, `z_heatmap_compact.png` | $Z$ of every component (columns, grouped by kind) in every input (rows); cells above the threshold outlined. Compact: the flagged components and the next 10 | rows (inputs) or columns (components) with many outlined cells |
| `closure/flagged/<q>.png`, `closure/passed/<q>.png` | one component in its worst input: learned and true sums per bin, and $\Delta$ relative to the SM yield with the noise band $\sqrt{\mathrm{Var}_{\rm MC} + \mathrm{Var}_{\rm seeds}}$ | where in the input the bias sits, and whether it is a shape or a normalisation |
| `attribution/<q>.png` | flagged components: the signed shares of the 10 largest networks and the sum of the others | one dominant network (a candidate to drop or retrain) or many small shares (a bias the networks share) |
| `drop_tests_compact.png`, `drop_tests_full.png` | change of the largest $Z$ of the affected components for every tested group and for the combined refit; blamed cells outlined, accepted groups in dark red | gains where the group was blamed, no new outlined cells elsewhere |
| `network_misclosure.png` | $m_a$ of every network against the inclusive yield of its training point, by kind, with the spread over members; suspects marked | networks far above the others of their kind |
| `point_search_objective.png` | the objective after each added unit, relative to the current design | how fast the gain falls off: a flat curve means more points will not help |
| `point_search_uncertainty_flagged.png`, `point_search_uncertainty_full.png` | $\sqrt{\mathrm{Var}(\hat\gamma_q)}$ before and after the search, over the measured rms bias | a bias well above the propagated uncertainty: more points will not remove it |

## 9. Reading the results

- **Nothing flagged.** The ensemble closes in every term within the noise of the validation sample; it can be used.
- **A flagged component with one dominant network.** Check that network's own closure (`network_misclosure`, its training plots); accept the drop if the test does, or retrain the copies listed in `retrain`. Rerun with `--refit` after retraining.
- **A flagged component with many small shares** and a bias well above `sigma_propagated`: every network carries the same bias, so neither dropping nor adding points removes it; the networks themselves need to learn that term better (capacity, training points closer to where the term matters, more training events), or the term may vanish by construction and belong in `zero_terms`.
- **A bias comparable to `sigma_propagated`**: more training points help; `proposed_points.yml` lists them.
- **Remaining flags after all of this** limit what the ensemble can be trusted for: test their effect on the scan (bias of the scan minimum on Asimov data) before using it.

The bias test can only resolve biases larger than the noise of the validation sample; an unflagged component is consistent with zero bias at that precision, not proven unbiased.
