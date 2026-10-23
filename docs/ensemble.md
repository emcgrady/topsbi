# Ensemble

The ensemble is built by the validation run, which also tests it ([`validation.md`](validation.md)):

```sh
python -m topsbi.validation -c <config.yml> --refit
```

The configuration keys are listed in [`examples/validation/README.md`](../examples/validation/README.md).

## 1. What the fit does

The ensemble consists of members $e$, each a complete set of networks trained with one seed. Every network $a$ of a member is trained at its own point $\theta_1^{(a)}$ against the same reference $\theta_0$ (config `c0`) and returns $\hat r(x; \theta_1^{(a)})$. Since the true ratio is quadratic in $\theta$, $r(x; \theta) = T(\theta)\, \Gamma(x)$ with $T(\theta)$ the expansion of $(1, \theta)$ into its constant, linear, quadratic and cross terms (`get_lower_tri`), the per-event coefficients follow from all networks of the member at once. For each member $e$ (`RatioFit`):

1. **Evaluate.** Every network, and the SM network ($\theta = 0$) last, is evaluated on the $N$ events of `data`, giving $R$ with row $a$ equal to $\hat r(x; \theta_1^{(a)})$ ([$n_a$, $N$]).
2. **Expand.** Row $a$ of $\Theta$ is $T(\theta_1^{(a)})$, restricted to the kept terms: a term is dropped if it is zero for every network, or listed in the networks' `zero_terms` (terms that vanish by construction). The kept terms are indexed by $q$ (`mask`, $K$ terms).
3. **Keep finite events.** An event is kept if every network's $\hat r$ is finite there (a probability network gives $\hat r = \infty$ where $\hat s = 1$).
4. **Solve.** $\Theta \hat\Gamma_e = R$ in least squares on the kept events (`torch.linalg.lstsq`, driver `gelsd`), giving $\hat\Gamma_e$ ([$K$, $N_e$]), the rank of $\Theta$ and, when $\Theta$ has full column rank and more rows than columns, the residual $\sum_a (\Theta \hat\Gamma_e - R)^2_{a,i}$ per event.

The members are then averaged (`Ensemble`) on the events every member keeps, $\langle \hat\Gamma \rangle_e$, and the learned ratio at any point is

$$\hat r(x; \theta) = T(\theta)[\text{mask}]\, \langle \hat\Gamma \rangle_e .$$

All members must be trained at the same set of points; a member with a different `mask` stops the fit. A member needs at least as many networks as kept terms, $n_a \ge K$, for $\Theta$ to have full column rank.

## 2. What it reads

From the validation config:

| Key | Used for |
|---|---|
| `seeds.<e>.networks`, `seeds.<e>.sm` | the training configs of member $e$'s networks and its SM network; each network's `c1` gives its row of $\Theta$, its `name` the directory with `model.pt` and `complete/performance.yml` |
| `data` | the events the networks are evaluated on; use events that no network was trained on |
| `c0`, `wcs` | the reference point and the WC order of $T(\theta)$ (every network must use the same `wcs` and `zero_terms`) |
| `device` | where the networks are evaluated (default `cpu`). The `device` in each training config is ignored, so networks trained on a GPU can be evaluated on a machine without one |
| `workers` | members fitted at once, one single-threaded process each (default 1) |

## 3. Running it

Without `--refit`, validation reads the cache in `{name}/ensemble/` and stops if it is missing or stale. A cache is stale if `data`, `c0` or `seeds` differ from the config, or if any network's `model.pt` changed since the fit (`ensemble.yml` keeps the sha256 of every `model.pt`). With `--refit` the ensemble is fitted again and the cache replaced.

The members are independent, so with `workers` > 1 they are fitted in parallel, one process each (spawned, single-threaded, results combined in member order so the result does not depend on `workers`).

What a run needs depends on the number of members $n_e$, networks per member $n_a$ and events $N$:

- **Disk.** The cache holds $R$ and $\hat\Gamma$ of every member in float32: about $n_e (n_a + K)\, N \times 4$ bytes, almost all of it $R$.
- **Memory.** A member's fit holds its $R$ ($n_a N \times 4$ bytes) and the working copies of the least-squares solve, about 3–5 times the size of $R$ in total. With `workers` > 1 that is per worker, so the peak is roughly `workers` times it, plus the main process (the data and the averages, a small multiple of $N K \times 8$ bytes). The later validation steps read the cache memory-mapped.
- **Time.** Dominated by evaluating every network on every event, proportional to $n_e n_a N$ and to the network size; divided by up to `workers` when members run in parallel.

Choose `workers` so that its memory fits the machine and the machine has that many free cores. Several workers on a shared interactive machine compete with other users; run them through a batch system with `workers` cores requested.

## 4. The cache `{name}/ensemble/`

Each member's files are written as soon as the member is fitted, so only one member's $R$ is in memory per process; the averages and `ensemble.yml` follow, `ensemble.yml` last, and the directory is moved into place, so an interrupted fit never leaves a cache that looks complete. Written by `SeedCollector` (per member) and `write_cache` in `tools/ensembleCache.py`.

| File / key | Content | Symbol | Shape |
|---|---|---|---|
| `gamma_cache.npy` | mean coefficients on the events every member keeps | $\langle \hat\Gamma \rangle_e$ | [$K$, $N_{\rm finite}$] float32 |
| `mask_cache.npy` | kept terms of $T(\theta)$ | range of $q$ | [$n_q$] bool |
| `filter_cache.npy` | events kept by every member | | [$N$] bool |
| `seed_<e>/gammas.npy` | member $e$'s coefficients on its own kept events | $\hat\Gamma_e$ | [$K$, $N_e$] float32 |
| `seed_<e>/filter.npy` | member $e$'s kept events | | [$N$] bool |
| `seed_<e>/residuals.npy` | least-squares residual per event (only with more networks than terms and full column rank) | | [$N_e$] float32 |
| `seed_<e>/ratios.npy` | every network's $\hat r$ for every event, row $a$, SM last; `inf` where a network diverged | $R$ | [$n_a$, $N$] float32 |
| `seed_<e>/network.npz`: `Theta`, `networks` | the expanded training points (row $a$, SM last) and the network names (relative to the member's results directory) | $\Theta$, $a$ | [$n_a$, $K$], [$n_a$] |
| `ensemble.yml`: `data`, `c0`, `wcs`, `lumi`, `seeds` | the inputs the cache was fitted from | $\theta_0$, $L$ | |
| `ensemble.yml`: `model_hash` | sha256 of every network's `model.pt`, per member in `seeds` order, SM last | | |
| `ensemble.yml`: `rank` | rank of each member's $\Theta$ | $\mathrm{rank}(\Theta_e)$ | |
| `ensemble.yml`: `created` | when the cache was written | | |

$R$ and $\Theta$ are kept so that validation can attribute biases to single networks and refit without some of them, without evaluating any network again.

## 5. Fit health

| Quantity | Where | Healthy fit |
|---|---|---|
| `rank` | `ensemble.yml`, `ensemble_diagnostics.yml` `ensemble.rank` | $K$ for every member (full column rank). A smaller rank means some combination of terms is not determined by the training points; validation marks the affected components `constrained: false` |
| residuals | `seed_<e>/residuals.npy`, plot `fit_residuals.png` (validation) | similar distributions for all members; validation lists members whose median residual is more than 3 times the typical one (`ensemble_diagnostics.yml` `ensemble.residual_outliers`, normally empty) |
| kept events | `filter.npy`, `filter_cache.npy` | nearly all events; a member that drops many has a diverging network |

Whether the ensemble is unbiased is tested by validation, term by term ([`validation.md`](validation.md)).

## 6. Reading the cache

```python
from topsbi.tools.ensembleCache import EnsembleCache

cache = EnsembleCache('<name>/ensemble')  # memory-maps the large arrays (mmap=False loads them)
r_hat = cache([1.0, *theta])  # r_hat(x; theta) on the events of cache.finite, float64
cache.Gamma, cache.mask, cache.finite  # <Gamma>_e, kept terms, kept events
cache.seed_Gamma[e], cache.seed_finite[e]  # member e
cache.network(e)  # {'Theta': ..., 'networks': [...]}
cache.seed_R(e)  # R of member e, memory-mapped
cache.matches(config)  # False if the cache is stale for this config
```

A likelihood scan needs only `gamma_cache.npy`, `mask_cache.npy` and `filter_cache.npy`.

## 7. Nuisance parameters

The ensemble describes the dependence on the Wilson coefficients only. Nuisance networks (`weight_shift`, `kinematic_shift`, [`training.md`](training.md)) are trained at a nominal reference $(\theta, 0)$ and give $\hat f = \log \hat r(x; \theta, \nu)$ for one direction of one nuisance; they are not part of this fit. With the factorisation $r(x; \theta, \nu) = r(x; \theta)\, r(x; \nu)$ the two enter the learned ratio as a sum of logarithms,

$$\log \hat r(x; \theta, \nu) = \log\left[T(\theta)[\text{mask}]\, \langle \hat\Gamma \rangle_e\right] + \sum_n \log \hat r(x; \nu_n).$$

Combining them is not yet implemented.
