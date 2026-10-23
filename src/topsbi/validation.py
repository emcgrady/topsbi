from topsbi.tools.build_ratio import Ensemble, seed_map
from topsbi.tools.data import get_lower_tri
from topsbi.tools.ensembleCache import (
    EnsembleCache,
    SeedCollector,
    binned_sums,
    component_labels,
    is_cross_term,
    merge_sparse,
    quantile_binning,
    write_cache,
)
from topsbi.tools.plots import (
    network_tex,
    plot_attribution,
    plot_closure,
    plot_drop_tests,
    plot_network_misclosure,
    plot_point_search,
    plot_residuals,
    plot_search_uncertainty,
    plot_z_heatmap,
)

import mplhep as mh
import numpy as np

import argparse, os, sys, torch, types, yaml

mh.style.use("CMS")


def load_ensemble(config, features, refit=False):
    """
    The ensemble cache of config: read from {name}/ensemble when it matches config (data, c0, networks and their
    model.pt), otherwise fitted and written, but only when refit is set.

    Args:
        config: validation config (name, data, c0, wcs, seeds, features, lumi, optional device and workers, the seeds
            fitted at once)
        features: [N, n_features] raw network inputs of config['data']
        refit: fit the ensemble and replace the cache even if one exists
    Returns:
        EnsembleCache
    """
    path = f'{config["name"]}/ensemble'
    if not refit:
        if not os.path.exists(f'{path}/ensemble.yml'):
            sys.exit(f'[error]: no ensemble cache in {path}; rerun with --refit to fit one')
        cache = EnsembleCache(path)
        if not cache.matches(config):
            sys.exit(f'[error]: {path} was built from other data, c0, networks or model.pt files; rerun with --refit')
        return cache
    config.setdefault('device', 'cpu')
    collector = SeedCollector(config['seeds'], path)
    ensemble = Ensemble(config, features, per_seed=collector, workers=config.get('workers', 1))
    write_cache(path, config, ensemble)
    return EnsembleCache(path)


def component_sums(cache, binning, w0, weights):
    """
    Binned sums for the bias test of every component gamma_q (row q of Gamma; q runs over the kept terms of T(theta),
    k in code), in kinematic bins K_b.

    Args:
        cache: EnsembleCache
        binning: binning dict (quantile_binning, then merge_sparse) built on the events of the cache's data
        w0: [N] w^s(theta_0) per event
        weights: [N, K] true per-event contribution of each kept component, L s_q(z) x 1000 (coefs[:, cache.mask])
    Returns:
        dict with
            bias: [S, K, B] per ensemble member e, hat w_q,b - w_q,b with hat w_q,b = sum_i w^s_i(theta_0) gamma_hat_q,i
                over the events in K_b that member kept
            mc_var: [K, B] Var_MC = sum_i (w^s_i(theta_0) <gamma_hat_q,i>_e - L s_q,i x 1000)^2 over the events every
                member kept
            truth: [K, B] w_q,b = sum_i L s_q,i x 1000 over the events in K_b, the true yield of each component, same
                events as mc_var
            sm_yield: [B] SM yield lambda_K_b(theta_SM) per bin (the sm component of truth)
            truth_var: [K, B] MC-statistics variance of truth, sum_i (L s_q,i x 1000)^2
            learned_var: [K, B] MC-statistics variance of the learned sum, sum_i (w^s_i(theta_0) <gamma_hat_q,i>_e)^2
    """
    w0 = torch.as_tensor(w0, dtype=torch.float64)
    weights = torch.as_tensor(weights, dtype=torch.float64)
    index, n_bins = binning['index'], binning['n_bins']
    truth_all = binned_sums(weights.T, index, n_bins)

    def truth_on(keep):
        # events dropped by a seed are rare, so subtract them instead of re-binning every event
        drop = ~keep
        return truth_all - binned_sums(weights[drop].T, index[:, drop], n_bins) if drop.any() else truth_all

    bias = []
    for seed in cache.seeds:
        keep = torch.from_numpy(cache.seed_finite[seed])
        Gamma = torch.from_numpy(np.asarray(cache.seed_Gamma[seed])).double()
        bias.append(binned_sums(Gamma * w0[keep], index[:, keep], n_bins) - truth_on(keep))
    keep = torch.from_numpy(cache.finite)
    Gamma = torch.from_numpy(np.asarray(cache.Gamma)).double()
    learned = Gamma * w0[keep]
    mc_var = binned_sums((learned - weights[keep].T) ** 2, index[:, keep], n_bins)
    truth = truth_on(keep)
    return {
        'bias': np.stack(bias),
        'mc_var': mc_var,
        'truth': truth,
        'sm_yield': truth[0],
        'truth_var': binned_sums(weights[keep].T ** 2, index[:, keep], n_bins),
        'learned_var': binned_sums(learned**2, index[:, keep], n_bins),
    }


def component_bias(sums, binning):
    """
    Significance and size of the bias of every component gamma_q, per feature.

    For each component q and feature f, the bias Delta_q,b (D in code) = <hat w_q,b - w_q,b>_e per bin K_b, with noise
    Var_MC (MC statistics) and Var_seeds (variance of the ensemble mean). chi2 = sum Delta^2 / Var_MC has expectation
    sum (1 + Var_seeds / Var_MC) and variance 2 sum (1 + Var_seeds / Var_MC)^2 without bias, so Z ~ N(0, 1) for an
    unbiased component.

    Args:
        sums: output of component_sums
        binning: the binning the sums were made with
    Returns:
        dict with
            Z: [K, F] excess chi2 in standard deviations
            Z_gauss: [K, F] Gaussian-equivalent significance: the noise-only chi2 approximated as a * chi2_nu with
                matching mean and variance (a = var / (2 mean), nu = 2 mean^2 / var), its tail probability above the
                observed chi2 converted to standard-normal sigma
            rms_bias: [K, F] sqrt(mean((Delta^2 - Var_MC - Var_seeds) / lambda_K_b(theta_SM)^2)), noise-subtracted rms
                bias per bin as a fraction of the SM yield, per unit WC (or WC^2)
            D: [K, B] ensemble-mean bias Delta
            s2: [K, B] Var_seeds, the variance of the ensemble mean
    """
    bias, mc_var, sm_yield = sums['bias'], sums['mc_var'], sums['sm_yield']
    n_seeds, n_comp, _ = bias.shape
    D = bias.mean(0)
    s2 = bias.var(0, ddof=1) / n_seeds if n_seeds > 1 else np.zeros_like(D)
    n_features = len(binning['names'])
    Z, rms_bias = np.full((n_comp, n_features), np.nan), np.full((n_comp, n_features), np.nan)
    Z_gauss = np.full((n_comp, n_features), np.nan)
    for f in range(n_features):
        b = (binning['feature'] == f) & (sm_yield > 0)
        e2, d2, v = mc_var[:, b], D[:, b] ** 2, s2[:, b]
        # a component with no events in a bin has Var_MC = 0 there and does not count in that bin
        ok = e2 > 0
        safe = np.where(ok, e2, 1)
        ratio = np.where(ok, v / safe, 0)
        chi2 = np.where(ok, d2 / safe, 0).sum(1)
        expected = (ok * (1 + ratio)).sum(1)
        variance = 2 * (ok * (1 + ratio) ** 2).sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            Z[:, f] = (chi2 - expected) / np.sqrt(variance)
            # P(a chi2_nu > chi2) is the regularized upper incomplete gamma Q(nu / 2, chi2 / (2 a))
            scale, dof = variance / (2 * expected), 2 * expected**2 / variance
            p = torch.special.gammaincc(torch.as_tensor(dof / 2), torch.as_tensor(chi2 / scale / 2))
            Z_gauss[:, f] = (-torch.special.ndtri(p)).numpy()
        rms_bias[:, f] = np.sqrt(np.maximum(np.mean((d2 - e2 - v) / sm_yield[b] ** 2, axis=1), 0))
    return {'Z': Z, 'Z_gauss': Z_gauss, 'rms_bias': rms_bias, 'D': D, 's2': s2}


def _network_sums_seed(seed, path, binning, w0, weights, chunk):
    """network_sums for one seed (through seed_map: the cache is opened here, so workers only memory-map it)."""
    cache = EnsembleCache(path)
    index, n_bins = binning['index'], binning['n_bins']
    keep = torch.from_numpy(cache.seed_finite[seed])
    network = cache.network(seed)
    Theta = torch.from_numpy(network['Theta']).double()
    R = cache.seed_R(seed)
    w, w0k, idx = weights[keep], w0[keep], index[:, keep]
    bias, mc_var = [], []
    for start in range(0, len(Theta), chunk):
        w0r = torch.from_numpy(np.asarray(R[start : start + chunk])[:, keep.numpy()]).double() * w0k
        diff = w0r - (w @ Theta[start : start + chunk].T).T  # learned - true w0 * r per network
        bias.append(binned_sums(diff, idx, n_bins))
        mc_var.append(binned_sums(diff**2, idx, n_bins))
    return {
        'networks': list(network['networks']),
        'Theta': network['Theta'],
        'bias': np.concatenate(bias),
        'mc_var': np.concatenate(mc_var),
    }


def network_sums(cache, binning, w0, weights, chunk=32, workers=1):
    """
    Binned misclosure of every training network of every seed, in the binning of the bias test.

    Args:
        cache: EnsembleCache
        binning: binning dict used for component_sums
        w0: [N] w^s(theta_0) per event
        weights: [N, K] true per-event contribution of each kept component, L s_q(z) x 1000 (coefs[:, cache.mask])
        chunk: networks binned at a time (memory)
        workers: seeds processed at once, one process each (seed_map)
    Returns:
        {seed: dict with
            networks: network names (rows of Theta)
            Theta: [n_networks, K] expanded training points T(theta_1^(a)), one row per network a
            bias: [n_networks, B] sum_i w^s_i(theta_0) (r_hat_a - r_a) over the events the member kept, with r_hat_a the
                network's row of R and r_a = r(x, z; theta_1^(a))
            mc_var: [n_networks, B] sum_i (w^s_i(theta_0) (r_hat_a - r_a))^2, the MC-statistics variance of bias
        }
    """
    w0 = torch.as_tensor(w0, dtype=torch.float64)
    weights = torch.as_tensor(weights, dtype=torch.float64)
    return seed_map(_network_sums_seed, cache.seeds, (cache.path, binning, w0, weights, chunk), workers)


def attribute(networks, components, sm_yield):
    """
    Split the bias of the given gamma components into the contributions of the training networks.

    gamma = Theta^+ r_hat per event, so the binned bias of component q is exactly
    sum_a Theta^+_qa * bias_a. Each network's share is the projection of its contribution on the total,
    weighted by 1 / lambda_K_b(theta_SM)^2 (bias relative to the SM yield); the shares of a component sum to 1.

    Args:
        networks: output of network_sums
        components: indices of the components to attribute
        sm_yield: [B] SM yield per bin
    Returns:
        shares: {network name: [len(components)]} signed shares, averaged over ensemble members
        total: [len(components), B] seed-mean bias rebuilt from the networks (equals component_bias D)
    """
    occupied = sm_yield > 0
    weight = np.where(occupied, 1 / np.where(occupied, sm_yield, 1) ** 2, 0)
    contributions = {}
    for s in networks.values():
        pinv = np.linalg.pinv(s['Theta'].astype(np.float64))[components]
        for a, name in enumerate(s['networks']):
            contributions[name] = contributions.get(name, 0) + pinv[:, a, None] * s['bias'][a] / len(networks)
    total = sum(contributions.values())
    norm = (total**2 * weight).sum(-1)
    shares = {name: (c * total * weight).sum(-1) / norm for name, c in contributions.items()}
    return shares, total


def network_z(networks, sm_yield, binning):
    """
    Own-misclosure significance of every training network, per seed and feature.

    Args:
        networks: output of network_sums
        sm_yield: [B] SM yield per bin
        binning: the binning of network_sums
    Returns:
        {seed: [n_networks, F] Z of sum w0 * r_hat_a - sum w0 * r_a, from MC statistics alone}
    """
    return {
        seed: component_bias({'bias': s['bias'][None], 'mc_var': s['mc_var'], 'sm_yield': sm_yield}, binning)['Z']
        for seed, s in networks.items()
    }


def suspects(shares, z, networks, z_threshold=5.0, share_threshold=0.5, seed_fraction=0.5):
    """
    Training points whose networks drive a flagged bias, split into points to drop and copies to retrain.

    A network is a candidate if |share| >= share_threshold for any flagged component. It is bad in a seed if its
    max-over-features own Z exceeds z_threshold there. Bad in at least seed_fraction of the seeds: the point is a
    problem, drop it from every seed. Bad in fewer: retrain the bad copies, keep the point.

    Args:
        shares: first output of attribute
        z: output of network_z
        networks: output of network_sums, for the network names of each seed
    Returns:
        drop: [names], retrain: {name: [seeds]}
    """
    drop, retrain = [], {}
    for name, share in shares.items():
        if name == 'sm' or np.max(np.abs(share)) < share_threshold:
            continue
        bad = [seed for seed, s in networks.items() if np.nanmax(z[seed][s['networks'].index(name)]) > z_threshold]
        if len(bad) >= seed_fraction * len(networks):
            drop.append(name)
        elif bad:
            retrain[name] = bad
    return drop, retrain


def _refit_seed(seed, path, drop):
    """refit_without for one seed (through seed_map): Gamma, kept events and rank of Theta without the drops."""
    cache = EnsembleCache(path)
    network = cache.network(seed)
    rows = np.array([name not in drop for name in network['networks']])
    Theta = torch.from_numpy(network['Theta'][rows])
    R = torch.from_numpy(np.asarray(cache.seed_R(seed))[rows])
    keep = ~torch.isinf(R).any(0)
    Gamma, _, rank, _ = torch.linalg.lstsq(Theta, R[:, keep], driver='gelsd')
    # lstsq returns the solution in an [n_networks, N_finite] buffer; keep only Gamma's rows
    return Gamma[: Theta.shape[1]].clone().numpy(), keep.numpy(), int(rank)


def refit_without(cache, drop, workers=1):
    """
    Refit every seed's Gamma without the dropped training points: Theta Gamma = R in least squares on the stored R.

    Args:
        cache: EnsembleCache
        drop: names of the networks to remove from every seed (not the SM anchor)
        workers: seeds refitted at once, one process each (seed_map)
    Returns:
        an object with the EnsembleCache attributes component_sums reads (seeds, seed_Gamma, seed_finite, Gamma,
        finite), plus rank: {seed: rank of the reduced Theta}
    """
    fits = seed_map(_refit_seed, cache.seeds, (cache.path, drop), workers)
    seed_Gamma = {s: f[0] for s, f in fits.items()}
    seed_finite = {s: f[1] for s, f in fits.items()}
    rank = {s: f[2] for s, f in fits.items()}
    common = np.logical_and.reduce(list(seed_finite.values()))
    mean = sum(g[:, common[f]] for g, f in zip(seed_Gamma.values(), seed_finite.values())) / len(seed_Gamma)
    return types.SimpleNamespace(
        seeds=list(cache.seeds),
        seed_Gamma=seed_Gamma,
        seed_finite=seed_finite,
        Gamma=mean,
        finite=common,
        mask=cache.mask,
        rank={s: int(r) for s, r in rank.items()},
    )


def _drop_bias_seed(seed, path, groups, binning, w0, weights, chunk):
    """
    drop_bias_sums for one seed (through seed_map): {group: binned bias of the seed with the group removed, or None if
    removing it leaves this seed's Theta rank-deficient}.
    """
    cache = EnsembleCache(path)
    index, n_bins = binning['index'], binning['n_bins']
    network = cache.network(seed)
    names, Theta = network['networks'], network['Theta'].astype(np.float64)
    keep = cache.seed_finite[seed]
    R = cache.seed_R(seed)
    H = Theta.T @ Theta
    # groups whose removal leaves Theta full rank: (their rows of Theta, H - Theta_J^T Theta_J)
    reduced, out = {}, {}
    for g, members in groups.items():
        rows = [names.index(n) for n in members]
        Hg = H - Theta[rows].T @ Theta[rows]
        if np.linalg.matrix_rank(Hg) < len(Hg):
            out[g] = None
            continue
        reduced[g] = (rows, Hg)
    keep_t = torch.from_numpy(keep)
    truth = binned_sums(weights[keep_t].T, index[:, keep_t], n_bins)
    seed_bias = {g: -truth for g in reduced}
    # a step of events at a time: Gamma is per event and binned sums add over events
    for start in range(0, len(keep), chunk):
        kept = np.flatnonzero(keep[start : start + chunk])
        Rc = np.asarray(R[:, start : start + chunk])[:, kept].astype(np.float64)
        kept = torch.from_numpy(start + kept)
        b = Theta.T @ Rc
        for g, (rows, Hg) in reduced.items():
            Gamma = torch.from_numpy(np.linalg.solve(Hg, b - Theta[rows].T @ Rc[rows]))
            seed_bias[g] += binned_sums(Gamma * w0[kept], index[:, kept], n_bins)
    return out | seed_bias


def drop_bias_sums(cache, groups, binning, w0, weights, sums, chunk=50_000, workers=1):
    """
    Binned bias of the components gamma_q with each group of training points removed, every group from one pass over
    each seed's R. Dropping rows J of Theta gives Gamma_-J = (H - Theta_J^T Theta_J)^-1 (Theta^T R - Theta_J^T R_J),
    H = Theta^T Theta, so a group costs one K x K solve per seed instead of a new least-squares fit (float64, since the
    normal equations square the condition number). Each seed keeps its cached finite-event mask, and Var_MC is taken
    from the full ensemble (removing a few networks hardly changes it).

    Args:
        cache: EnsembleCache
        groups: {group name: [network names removed together]}
        binning, w0, weights: as for component_sums
        sums: component_sums of the full ensemble (for mc_var, truth, sm_yield)
        chunk: events per step (memory: one step holds R for chunk events in float64)
        workers: seeds processed at once, one process each (seed_map)
    Returns:
        {group name: sums dict for component_bias, or None if removing the group leaves Theta rank-deficient}
    """
    w0 = torch.as_tensor(w0, dtype=torch.float64)
    weights = torch.as_tensor(weights, dtype=torch.float64)
    per_seed = seed_map(_drop_bias_seed, cache.seeds, (cache.path, groups, binning, w0, weights, chunk), workers)
    # a group that leaves any seed's Theta rank-deficient has no complete average
    bias = {
        g: None if any(per_seed[s][g] is None for s in cache.seeds) else [per_seed[s][g] for s in cache.seeds]
        for g in groups
    }
    return {
        g: None
        if b is None
        else {'bias': np.stack(b), 'mc_var': sums['mc_var'], 'truth': sums['truth'], 'sm_yield': sums['sm_yield']}
        for g, b in bias.items()
    }


def test_drops(baseline, after, blamed, z_threshold=5.0, min_gain=2.0, tolerance=1.0):
    """
    Keep a drop only if it helps what it was blamed for without harming anything else.

    Args:
        baseline: component_bias of the full ensemble
        after: {group: component_bias without the group, or None if the group cannot be removed}
        blamed: {group: indices of the flagged components the group was blamed for}
        z_threshold: flagging threshold
        min_gain: smallest required drop in max Z over features, for at least one blamed component
        tolerance: largest allowed rise in max Z for any component that ends above z_threshold
    Returns:
        {group: dict(accepted, gain, harmed)} with harmed = {component index: (Z before, Z after)}
    """
    before = np.nanmax(baseline['Z'], axis=1)
    result = {}
    for g, b in after.items():
        if b is None:
            result[g] = {'accepted': False, 'gain': None, 'harmed': 'removing it leaves Theta rank-deficient'}
            continue
        z = np.nanmax(b['Z'], axis=1)
        gain = float(max(before[k] - z[k] for k in blamed[g]))
        worse = np.flatnonzero((z > z_threshold) & (z - before > tolerance))
        harmed = {int(k): (float(before[k]), float(z[k])) for k in worse}
        result[g] = {'accepted': gain >= min_gain and not harmed, 'gain': gain, 'harmed': harmed}
    return result


def network_misclosure(networks, sm_yield, per_seed=False):
    """
    Noise-subtracted rms misclosure of every training network, as a fraction of the SM yield per bin,
    m_a = sqrt(mean_b max(bias_a,b^2 - mc_var_a,b, 0) / lambda_K_b(theta_SM)^2), with m_a^2 averaged over ensemble
    members by name.

    Returns:
        {network name: m_a}, or with per_seed {network name: [m_a of each member that has it]}
    """
    occupied = sm_yield > 0
    m2 = {}
    for s in networks.values():
        excess = np.maximum(s['bias'][:, occupied] ** 2 - s['mc_var'][:, occupied], 0) / sm_yield[occupied] ** 2
        for name, value in zip(s['networks'], excess.mean(1)):
            m2.setdefault(name, []).append(value)
    if per_seed:
        return {name: np.sqrt(np.array(v)) for name, v in m2.items()}
    return {name: float(np.sqrt(np.mean(v))) for name, v in m2.items()}


def scan_moments(wcs, bounds, mask, mode='box', shape='uniform'):
    """
    Second moments G = E[T(theta) T(theta)^T] over the kept get_lower_tri columns, for theta in the scan region.
    Every entry of T is a monomial of degree <= 2, so G needs moments of each theta up to 4th order:
    uniform on [-a, a]: E[theta^n] = a^n / (n + 1); gaussian N(0, a^2): E[theta^n] = (n - 1)!! a^n; 0 for odd n.

    Args:
        wcs: WC names in get_lower_tri order
        bounds: {wc: a} scan half-range (uniform) or width (gaussian); WCs not listed are fixed at 0
        mask: kept get_lower_tri columns
        mode: 'box' (all listed WCs varied together, profiled or multi-WC scans) or '1d' (average over 1D scans,
            one WC varied at a time)
        shape: 'uniform' or 'gaussian'
    Returns:
        [K, K] G
    """
    n = len(wcs) + 1
    rows, cols = np.tril_indices(n)
    powers = np.zeros((len(rows), n), dtype=int)  # exponent of each theta in each column of T
    np.add.at(powers, (np.arange(len(rows)), rows), 1)
    np.add.at(powers, (np.arange(len(rows)), cols), 1)
    powers = powers[np.asarray(mask)][:, 1:]  # drop the constant SM entry (theta_0 = 1)
    e = powers[:, None, :] + powers[None, :, :]  # exponents of each product T_k T_l, at most 4
    factor = 1 / (e + 1) if shape == 'uniform' else np.array([1, 0, 1, 0, 3])[e]

    def moments(a):
        return np.where(e % 2 == 0, a**e * factor, 0.0).prod(-1)

    a = np.array([bounds.get(wc, 0.0) for wc in wcs], dtype=np.float64)
    if mode == 'box':
        return moments(a)
    return np.mean([moments(np.where(np.arange(len(wcs)) == i, a, 0.0)) for i in np.flatnonzero(a)], axis=0)


def candidate_units(wcs, Theta, names, misclosure, mask, yield_ratio, max_yield, extra=()):
    """
    Candidate training units for the point search. Each unit lists its points theta, their rows T(theta)[mask],
    and the misclosure assumed for their networks.

    - cross: for every kept interference column (i, j) of T(theta), the 4 points (+-theta_i, +-theta_j) for every pair
      of single-WC magnitude levels of the two WCs, a level pairing the same-rank positive and negative single-WC
      values (assumed m: the worse of the two source networks)
    - extra: points to retrain, e.g. the points suspects() marks to drop, as (name, theta) with assumed m = median m
    Units whose points all exist already, or with any point above max_yield (inclusive yield / SM yield), are dropped.

    Args:
        wcs: WC names in get_lower_tri order
        Theta: [n_networks, K] expanded training points of the current design
        names: network names, rows of Theta
        misclosure: output of network_misclosure
        mask: kept get_lower_tri columns
        yield_ratio: callable theta -> inclusive yield at theta / SM yield
        max_yield: largest yield ratio allowed for a new point
    Returns:
        [dict(kind, points, rows, misclosure, max_yield_ratio)]
    """
    n = len(wcs) + 1
    rows, cols = np.tril_indices(n)
    kept = [(int(i), int(j)) for i, j, keep in zip(rows, cols, mask) if keep]
    # the linear columns of Theta are the WC values of each training point
    thetas = np.zeros((len(Theta), n))
    thetas[:, 0] = 1
    for i in range(1, n):
        if (i, 0) in kept:
            thetas[:, i] = Theta[:, kept.index((i, 0))]
    existing = {tuple(np.round(t, 6)) for t in thetas}

    def make(kind, points, m, new_only=True):
        yields = [yield_ratio(p['theta']) for p in points]
        if max(yields) > max_yield:
            return None
        if new_only and all(tuple(np.round(p['theta'], 6)) in existing for p in points):
            return None
        return {
            'kind': kind,
            'points': points,
            'rows': np.array([get_lower_tri(p['theta'], dtype=torch.float64).numpy()[mask] for p in points]),
            'misclosure': np.array(m),
            'max_yield_ratio': float(max(yields)),
        }

    single = {}
    for t, name in zip(thetas, names):
        nonzero = np.flatnonzero(t[1:]) + 1
        if len(nonzero) == 1:
            single.setdefault(int(nonzero[0]), []).append((t[nonzero[0]], misclosure[name], name))
    for i, values in single.items():
        pos = sorted([v for v in values if v[0] > 0], key=lambda v: v[0])
        neg = sorted([v for v in values if v[0] < 0], key=lambda v: -v[0])
        single[i] = list(zip(pos, neg))
    units = []
    for i, j in kept:
        if j == 0 or i == j or i not in single or j not in single:
            continue
        for level_a in single[i]:
            for level_b in single[j]:
                points, m = [], []
                for va, ma, la in level_a:
                    for vb, mb, lb in level_b:
                        theta = [1.0] + [0.0] * len(wcs)
                        theta[i], theta[j] = float(va), float(vb)
                        label = f'{wcs[i - 1]}_{os.path.basename(la)}_{wcs[j - 1]}_{os.path.basename(lb)}'
                        points.append({'name': label, 'theta': theta})
                        m.append(max(ma, mb))
                units.append(make('cross', points, m))
    median = float(np.median(list(misclosure.values())))
    for name, theta in extra:
        # a dropped point is still in Theta, so it is offered even though it exists
        units.append(make('retrain', [{'name': name, 'theta': list(theta)}], [median], new_only=False))
    return [u for u in units if u is not None]


def objective_weights(cfg, wcs, mask, flagged, Theta, m):
    """
    The weight matrix G of the point-search objective tr(Cov(gamma_hat) G).

    - G: scan  -> scan-region second moments E[T(theta) T(theta)^T] (scan_moments with cfg bounds, mode, shape)
    - G: flagged (default) -> diag(g_q / Var(gamma_hat_q)), Var(gamma_hat_q) propagated for the current design
      (sigma0_2 in code), with objective weights g_q (new symbol, g in code) = 1 for flagged components and
      cfg['unflagged_weight'] (0.1) for the others, or 1 for all if none is flagged. Dividing by Var(gamma_hat_q)
      makes the objective the g-weighted sum of relative variance reductions, so components with large
      uncertainties do not dominate.

    Args:
        cfg: point_search section of the config
        wcs: WC names in get_lower_tri order
        mask: kept get_lower_tri columns
        flagged: indices of the flagged components
        Theta: [n_networks, K] expanded training points of the current design
        m: [n_networks] misclosure of each network
    Returns:
        [K, K] G
    """
    if cfg.get('G', 'flagged') == 'scan':
        return scan_moments(wcs, cfg['bounds'], mask, cfg.get('mode', 'box'), cfg.get('shape', 'uniform'))
    Hi = np.linalg.pinv(Theta.T @ Theta)
    sigma0_2 = np.diag(Hi @ (Theta.T @ (m[:, None] ** 2 * Theta)) @ Hi)
    g = np.full(Theta.shape[1], cfg.get('unflagged_weight', 0.1) if flagged else 1.0)
    g[flagged] = 1.0
    return np.diag(np.where(sigma0_2 > 0, g / np.where(sigma0_2 > 0, sigma0_2, 1), 0))


def unconstrained_components(Theta, rtol=1e-10):
    """
    Components the training design cannot determine: those whose column direction has a part in the null space of
    Theta. The pseudo-inverse gives them zero propagated error, which would read as perfectly known.

    Args:
        Theta: [n_networks, K] expanded training points
        rtol: singular values below rtol times the largest count as zero
    Returns:
        [K] bool, True for an unconstrained component
    """
    _, s, Vt = np.linalg.svd(Theta, full_matrices=True)
    s = np.r_[s, np.zeros(Vt.shape[0] - len(s))]
    null = Vt[s <= rtol * s.max()]
    return (null**2).sum(0) > rtol


def search_points(Theta, m, units, G, n_units=10, min_gain=0.01, chunk=256):
    """
    Greedily add the unit that most reduces tr(Cov(gamma_hat) G), Cov(gamma_hat) = H^+ S H^+ (Sigma in code) with
    H = Theta^T Theta and S = Theta^T M^2 Theta (independent network errors of size m); stop after n_units or when a
    unit gains less than min_gain (relative).

    Returns:
        chosen units, sqrt(Var(gamma_hat_q)) before and after (sigma0, sigma1 in code), objective after each step
    """
    H = Theta.T @ Theta
    S = Theta.T @ (m[:, None] ** 2 * Theta)

    def objective(H, S):
        Hi = np.linalg.pinv(H)
        Sigma = Hi @ S @ Hi
        return np.einsum('...kl,lk->...', Sigma, G), np.sqrt(np.maximum(np.einsum('...kk->...k', Sigma), 0))

    current, sigma0 = objective(H, S)
    history, chosen, remaining = [float(current)], [], list(units)
    while remaining and len(chosen) < n_units:
        scores = np.concatenate(
            [
                objective(
                    H + np.stack([u['rows'].T @ u['rows'] for u in batch]),
                    S + np.stack([u['rows'].T @ (u['misclosure'][:, None] ** 2 * u['rows']) for u in batch]),
                )[0]
                for batch in (remaining[i : i + chunk] for i in range(0, len(remaining), chunk))
            ]
        )
        best = int(np.argmin(scores))
        if (history[-1] - scores[best]) / history[-1] < min_gain:
            break
        unit = remaining.pop(best)
        H = H + unit['rows'].T @ unit['rows']
        S = S + unit['rows'].T @ (unit['misclosure'][:, None] ** 2 * unit['rows'])
        chosen.append(unit)
        history.append(float(scores[best]))
    return chosen, sigma0, objective(H, S)[1], history


def drop_test_entry(test, labels):
    """A test_drops result with component labels instead of indices, for the yml."""
    harmed = test['harmed']
    if isinstance(harmed, dict):
        harmed = {labels[k]: [before, after] for k, (before, after) in harmed.items()}
    return {'accepted': bool(test['accepted']), 'gain': test['gain'], 'harmed': harmed}


def write_results(
    name,
    *,
    config,
    cache,
    labels,
    kinds,
    binning,
    bias,
    flagged,
    shares,
    total,
    drop,
    retrain,
    refit_bias,
    misclosure,
    units,
    chosen,
    sigma0,
    sigma1,
    history,
    residual_outliers,
    groups,
    drop_tests,
    combined,
):
    """
    Write ensemble_diagnostics.yml (fit health, bias test, attribution, suspects, point search) and
    proposed_points.yml (the points of the units chosen by the search) to name.
    """
    n_top = config.get('n_top_networks', 5)
    features = binning['names']
    components = {}
    for k, label in enumerate(labels):
        entry = {
            'kind': kinds[k],
            'flagged': k in flagged,
            'max_Z': float(np.nanmax(bias['Z'][k])),
            'worst_feature': features[int(np.nanargmax(bias['Z'][k]))],
            'rms_bias': float(np.nanmax(bias['rms_bias'][k])),
            'Z': {f: float(z) for f, z in zip(features, bias['Z'][k])},
            'max_Z_gauss': float(np.nanmax(bias['Z_gauss'][k])),
            'Z_gauss': {f: float(z) for f, z in zip(features, bias['Z_gauss'][k])},
            'constrained': bool(np.isfinite(sigma0[k])),
            'sigma_propagated': None if np.isnan(sigma0[k]) else float(sigma0[k]),
            'sigma_after_search': None if np.isnan(sigma1[k]) else float(sigma1[k]),
        }
        if k in flagged:
            i = flagged.index(k)
            top = sorted(shares, key=lambda n: -abs(shares[n][i]))[:n_top]
            entry['top_networks'] = {n: float(shares[n][i]) for n in top}
        if refit_bias is not None:
            entry['refit'] = {
                'max_Z': float(np.nanmax(refit_bias['Z'][k])),
                'rms_bias': float(np.nanmax(refit_bias['rms_bias'][k])),
            }
        components[label] = entry

    # the networks rebuild the measured bias exactly up to float32 rounding (gamma = Theta^+ r_hat)
    identity = None
    if total is not None:
        D = bias['D'][flagged]
        identity = float(np.abs(total - D).max() / np.abs(D).max())

    diagnostics = {
        'ensemble': {
            'path': cache.path,
            'created': cache.manifest['created'],
            'seeds': list(cache.seeds),
            'rank': cache.manifest['rank'],
            'residual_outliers': list(residual_outliers),
        },
        'binning': {f: int((binning['feature'] == i).sum()) for i, f in enumerate(features)},
        'flagged': [labels[k] for k in flagged],
        'attribution_identity': identity,
        'components': components,
        'suspects': {
            'drop': list(drop),
            'retrain': {n: list(s) for n, s in retrain.items()},
            'drop_tests': {g: drop_test_entry(drop_tests[g], labels) | {'members': list(m)} for g, m in groups.items()},
            'combined': None if combined is None else drop_test_entry(combined, labels),
        },
        'network_misclosure': {n: float(v) for n, v in misclosure.items()},
        'point_search': {
            'G': config.get('point_search', {}).get('G', 'flagged'),
            'candidate_units': len(units),
            'objective': [float(h) for h in history],
            'proposed': [
                {'kind': u['kind'], 'max_yield_ratio': u['max_yield_ratio'], 'points': [p['name'] for p in u['points']]}
                for u in chosen
            ],
        },
    }
    with open(f'{name}/ensemble_diagnostics.yml', 'w') as f:
        yaml.safe_dump(diagnostics, f, sort_keys=False)
    with open(f'{name}/proposed_points.yml', 'w') as f:
        yaml.safe_dump(
            {
                'wcs': list(config['wcs']),
                'points': [
                    {'name': p['name'], 'kind': u['kind'], 'theta': list(map(float, p['theta']))}
                    for u in chosen
                    for p in u['points']
                ],
            },
            f,
            sort_keys=False,
        )


def main(config_path, refit=False):
    """
    Run the validation of an ensemble (AGENTS.md, validation): bias test of every component gamma_q, attribution,
    suspects and drop tests, point search; writes ensemble_diagnostics.yml, proposed_points.yml and plots/ under
    config['name'].

    Args:
        config_path: validation config yml
        refit: fit the ensemble and replace its cache
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    torch.set_num_threads(config.get('n_threads', 1))
    # seeds processed at once by the per-seed stages (fit, network_sums, drop tests), one process each
    workers = config.get('workers', 1)
    if isinstance(config['features'], str):
        with open(config['features']) as f:
            config['features'] = yaml.safe_load(f)
    name = config['name']
    os.makedirs(f'{name}/plots', mode=0o755, exist_ok=True)

    # features and coefs are the first two tensors of every data format (combined: features, coefs, ids, event_id,
    # syst_ratio, w_ref; older files: features, coefs, ids); the memory map reads only these two from disk
    tensors = torch.load(config['data'], weights_only=False, mmap=True).tensors
    features, coefs = tensors[0].clone(), tensors[1].clone()
    if 'lumi' not in config:
        sys.exit('[error]: config needs lumi, the luminosity of each era')
    eras = features[:, config.get('year_loc', 21)].long()
    if eras.max() >= len(config['lumi']):
        sys.exit(f'[error]: lumi has {len(config["lumi"])} eras but the data use era index {int(eras.max())}')
    lumi = torch.tensor(config['lumi'], dtype=torch.float64)[eras]
    all_weights = lumi[:, None] * coefs.to(torch.float64) * 1000
    w0 = all_weights @ get_lower_tri(config['c0'], dtype=torch.float64)

    # ensemble: cached, or fitted only with --refit
    cache = load_ensemble(config, features, refit)
    residual_outliers = plot_residuals(cache.residuals, f'{name}/plots/fit_residuals.png')
    wcs = cache.manifest['wcs']
    weights = all_weights[:, torch.from_numpy(cache.mask)]
    labels, kinds = component_labels(wcs, cache.mask)

    # bias test of every gamma component
    cfg = config.get('bias', {})
    binning = quantile_binning(features, config['features'], cfg.get('n_bins', 20), cfg.get('features'))
    binning = merge_sparse(binning, features, config['features'], w0, cfg.get('min_neff', 30))
    sums = component_sums(cache, binning, w0, weights)
    bias = component_bias(sums, binning)
    max_z = np.nanmax(bias['Z'], axis=1)
    z_flag = cfg.get('z_threshold', 5.0)
    flagged = [int(k) for k in np.argsort(-max_z) if max_z[k] > z_flag]
    plot_z_heatmap(
        bias['Z'],
        labels,
        kinds,
        binning['names'],
        f'{name}/plots/z_heatmap_full.png',
        z_flag,
        feature_config=config['features'],
        Z_gauss=bias['Z_gauss'],
    )
    compact = [int(k) for k in np.argsort(-max_z)[: len(flagged) + 10]]
    plot_z_heatmap(
        bias['Z'],
        labels,
        kinds,
        binning['names'],
        f'{name}/plots/z_heatmap_compact.png',
        z_flag,
        components=compact,
        feature_config=config['features'],
        Z_gauss=bias['Z_gauss'],
    )

    # closure of every component in its worst feature, flagged and passed components in separate directories
    for group in ('flagged', 'passed'):
        os.makedirs(f'{name}/plots/closure/{group}', mode=0o755, exist_ok=True)
    for k in range(len(labels)):
        if np.isnan(bias['Z'][k]).all():
            continue
        safe = labels[k].replace('^2', '_sq').replace('*', '_x_')
        group = 'flagged' if k in flagged else 'passed'
        plot_closure(
            k,
            int(np.nanargmax(bias['Z'][k])),
            sums,
            bias,
            binning,
            labels,
            f'{name}/plots/closure/{group}/{safe}.png',
            feature_config=config['features'],
        )

    # attribution of the flagged biases to the training networks
    networks = network_sums(cache, binning, w0, weights, workers=workers)
    shares, total = attribute(networks, flagged, sums['sm_yield']) if flagged else ({}, None)

    # suspect networks, and the bias test refitted without the points to drop
    cfg = config.get('suspects', {})
    drop, retrain = [], {}
    if flagged:
        z = network_z(networks, sums['sm_yield'], binning)
        drop, retrain = suspects(
            shares,
            z,
            networks,
            cfg.get('z_threshold', 5.0),
            cfg.get('share_threshold', 0.5),
            cfg.get('seed_fraction', 0.5),
        )
    # each suspect family is removed on its own and kept out only if that helps what it was blamed for without
    # harming other components: the points of a cross-term pair go together, scan points alone
    groups, drop_tests, combined, refit_bias = {}, {}, None, None
    for n in drop:
        groups.setdefault(os.path.dirname(n) if is_cross_term(n) else n, []).append(n)
    if groups:
        threshold = cfg.get('share_threshold', 0.5)
        blamed = {
            g: [k for i, k in enumerate(flagged) if any(abs(shares[n][i]) >= threshold for n in members)]
            for g, members in groups.items()
        }
        after = {
            g: None if s is None else component_bias(s, binning)
            for g, s in drop_bias_sums(cache, groups, binning, w0, weights, sums, workers=workers).items()
        }
        drop_tests = test_drops(bias, after, blamed, z_flag, cfg.get('min_gain', 2.0), cfg.get('tolerance', 1.0))
        drop = [n for g, members in groups.items() if drop_tests[g]['accepted'] for n in members]
        if drop:
            # drops that pass alone may interact: refit exactly without all of them and test the combination
            refit = refit_without(cache, drop, workers=workers)
            refit_bias = component_bias(component_sums(refit, binning, w0, weights), binning)
            accepted_blame = sorted({k for g in groups if drop_tests[g]['accepted'] for k in blamed[g]})
            combined = test_drops(
                bias,
                {'all': refit_bias},
                {'all': accepted_blame},
                z_flag,
                cfg.get('min_gain', 2.0),
                cfg.get('tolerance', 1.0),
            )['all']

    first = cache.network(cache.seeds[0])
    tex = network_tex(first['Theta'], first['networks'], wcs, cache.mask)
    # what each suspect drop did to every affected component, and the combined refit
    if groups:
        for version, everything in [('compact', False), ('full', True)]:
            plot_drop_tests(
                bias,
                after,
                refit_bias,
                groups,
                blamed,
                drop_tests,
                labels,
                tex,
                f'{name}/plots/drop_tests_{version}.png',
                show_all=everything,
            )

    # attribution of every flagged component, with the suspects' outcome noted next to the networks
    if flagged:
        notes = {n: 'dropped' for n in drop}
        notes.update({n: 'retrain' for n in retrain})
        os.makedirs(f'{name}/plots/attribution', mode=0o755, exist_ok=True)
        for k in flagged:
            safe = labels[k].replace('^2', '_sq').replace('*', '_x_')
            plot_attribution(k, shares, flagged, labels, bias, tex, f'{name}/plots/attribution/{safe}.png', notes=notes)

    # point search on the design of the first seed (all seeds share the training points); dropped suspects leave
    # the design and come back only as candidates to retrain
    cfg = config.get('point_search', {})
    misclosure = network_misclosure(networks, sums['sm_yield'])
    # size of each training point: inclusive yield at theta_1 relative to the SM yield (the sm column comes first)
    component_totals = weights.sum(0).numpy()
    ratio = dict(zip(first['networks'], first['Theta'].astype(np.float64) @ component_totals / component_totals[0]))
    notes = {n: 'dropped' for n in drop}
    notes.update({n: 'retrain' for n in retrain})
    plot_network_misclosure(
        misclosure,
        ratio,
        tex,
        f'{name}/plots/network_misclosure.png',
        notes=notes,
        per_seed=network_misclosure(networks, sums['sm_yield'], per_seed=True),
    )
    design = cache.network(cache.seeds[0])
    Theta, names = design['Theta'].astype(np.float64), list(design['networks'])
    paths = cache.manifest['seeds'][cache.seeds[0]]
    paths = paths['networks'] + [paths['sm']]
    extra = []
    for n in drop:
        with open(paths[names.index(n)]) as f:
            extra.append((n, yaml.safe_load(f)['c1']))
    keep = np.array([n not in drop for n in names])
    Theta, names = Theta[keep], [n for n, k in zip(names, keep) if k]
    m = np.array([misclosure[n] for n in names])
    totals = all_weights.sum(0)

    def yield_ratio(theta):
        return float(get_lower_tri(theta, dtype=torch.float64) @ totals / totals[0])

    max_yield = cfg.get('max_yield_ratio') or float((Theta @ weights.sum(0).numpy()).max() / totals[0])
    units = candidate_units(wcs, Theta, names, misclosure, cache.mask, yield_ratio, max_yield, extra)
    G = objective_weights(cfg, wcs, cache.mask, flagged, Theta, m)
    chosen, sigma0, sigma1, history = search_points(
        Theta, m, units, G, cfg.get('n_units', 10), cfg.get('min_gain', 0.01)
    )
    # components the design cannot determine have no propagated error (the pseudo-inverse would report 0)
    sigma0 = np.where(unconstrained_components(Theta), np.nan, sigma0)
    after = np.vstack([Theta] + [u['rows'] for u in chosen])
    sigma1 = np.where(unconstrained_components(after), np.nan, sigma1)

    # what the proposed units buy: objective per unit, and the expected errors before and after
    if chosen:
        plot_point_search(
            history, chosen, wcs, f'{name}/plots/point_search_objective.png', scan=cfg.get('G', 'flagged') == 'scan'
        )
    else:
        print('[info]: the point search added no units, no point-search plots')
    for version, shown in [('flagged', flagged or None), ('full', None)]:
        if version == 'flagged' and not flagged:
            continue
        plot_search_uncertainty(
            sigma0,
            sigma1,
            bias['rms_bias'],
            labels,
            f'{name}/plots/point_search_uncertainty_{version}.png',
            components=shown,
        )

    write_results(
        name,
        config=config,
        cache=cache,
        labels=labels,
        kinds=kinds,
        binning=binning,
        bias=bias,
        flagged=flagged,
        shares=shares,
        total=total,
        drop=drop,
        retrain=retrain,
        refit_bias=refit_bias,
        misclosure=misclosure,
        units=units,
        chosen=chosen,
        sigma0=sigma0,
        sigma1=sigma1,
        history=history,
        residual_outliers=residual_outliers,
        groups=groups,
        drop_tests=drop_tests,
        combined=combined,
    )
    print(
        f'[info]: {len(flagged)} of {len(labels)} components flagged, {len(drop)} suspects dropped, '
        f'{len(chosen)} units proposed; see {name}/ensemble_diagnostics.yml'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='validation configuration yml file')
    parser.add_argument('--refit', action='store_true', help='fit the ensemble and replace the cache')
    args = parser.parse_args()
    main(args.config, args.refit)
