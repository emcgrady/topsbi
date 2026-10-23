"""
Validation functions on a small synthetic ensemble with a known truth, written through the real cache code (no
trained networks): two WCs, three members of twelve training points plus SM, "networks" that return the true ratio
with 1% noise per event, optionally with a bias injected into one network.
"""

import types

import numpy as np
import pytest
import torch
import yaml

import topsbi.validation as val
from topsbi.tools.data import get_lower_tri
from topsbi.tools.ensembleCache import (
    EnsembleCache,
    SeedCollector,
    binned_sums,
    component_labels,
    merge_sparse,
    quantile_binning,
    write_cache,
)

WCS = ['cA', 'cB']
N_EVENTS = 3000
MEMBERS = [0, 1, 2]
FEATURES = {'x0': {'loc': 0}, 'x1': {'loc': 1}, 'x2': {'loc': 2}}
POINTS = {
    'cA/p_one': (1, 0),
    'cA/n_one': (-1, 0),
    'cA/p_two': (2, 0),
    'cA/n_two': (-2, 0),
    'cB/p_one': (0, 1),
    'cB/n_one': (0, -1),
    'cB/p_two': (0, 2),
    'cB/n_two': (0, -2),
    'cA_cB/p_one_p_one': (1, 1),
    'cA_cB/p_one_n_one': (1, -1),
    'cA_cB/n_one_p_one': (-1, 1),
    'cA_cB/n_one_n_one': (-1, -1),
}
CROSS = [name for name in POINTS if name.startswith('cA_cB/')]
BAD = 'cA/p_two'  # the network a bias is injected into
NOISE = 0.01  # relative noise of every network's ratio, per event


def expanded(theta):
    """T(theta) of an SM-inclusive point as a float64 numpy row."""
    return get_lower_tri(list(map(float, theta)), dtype=torch.float64).numpy()


def truth():
    """Features, per-event component weights (L s_q x 1000 in validation) and w^s(theta_0) of the toy sample."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (N_EVENTS, 3))
    x[:, 2] = rng.integers(0, 3, N_EVENTS)  # a discrete input
    # w(theta) = (1, theta) A (1, theta)^T per event, A positive definite, so every weight is positive
    v = np.stack([np.ones(N_EVENTS), 0.5 * x[:, 0] + 0.2, 0.4 * x[:, 1] - 0.3], axis=1)
    a = v[:, :, None] * v[:, None, :] + 0.05 * np.eye(3)
    rows, cols = np.tril_indices(3)
    s = a[:, rows, cols] * np.where(rows == cols, 1.0, 2.0)
    weights = s * rng.uniform(0.5, 1.5, N_EVENTS)[:, None]  # unequal event weights, as in MC
    w0 = weights @ expanded([1, 0, 0])
    return x, weights, w0


def fake_fit(Theta, R):
    """What RatioFit hands to the per-seed callback, for given Theta and R (all events finite)."""
    Theta, R = torch.as_tensor(Theta, dtype=torch.float32), torch.as_tensor(R, dtype=torch.float32)
    finite = ~torch.isinf(R).any(0)
    solution, residuals, rank, _ = torch.linalg.lstsq(Theta, R[:, finite], driver='gelsd')
    return types.SimpleNamespace(
        Theta=Theta, R=R, finite=finite, Gamma=solution[: Theta.shape[1]].clone(), residuals=residuals, rank=rank
    )


def build_toy(root, inject=None, members=MEMBERS):
    """
    Write the toy ensemble cache under root and return what the validation functions need.

    Args:
        root: directory for the fake training configs and the cache
        inject: optional (size, members): network BAD returns r (1 + size) for events with x0 > 0.5 in those members
        members: the ensemble members (seeds)
    """
    x, weights, w0 = truth()
    names = list(POINTS) + ['sm']
    Theta = np.stack([expanded((1, *POINTS[n])) for n in POINTS] + [expanded((1, 0, 0))])
    gamma = weights / w0[:, None]  # true per-event coefficients, so hat w = w0 gamma reproduces weights exactly
    R_true = Theta @ gamma.T
    seeds = {}
    for e in members:
        paths = []
        for name in names:
            directory = root / f'seed_{e}' / name
            directory.mkdir(parents=True)
            (directory / 'model.pt').write_bytes(f'{name} {e}'.encode())
            (directory / 'training.yml').write_text(yaml.safe_dump({'name': str(directory)}))
            paths.append(str(directory / 'training.yml'))
        seeds[e] = {'networks': paths[:-1], 'sm': paths[-1]}
    config = {'data': 'toy', 'c0': [1.0, 0.0, 0.0], 'wcs': WCS, 'lumi': [1.0], 'seeds': seeds}

    path = str(root / 'ensemble')
    collector = SeedCollector(seeds, path)
    Gamma, rank = 0, {}
    for e in members:
        rng = np.random.default_rng(100 + e)
        R = R_true * (1 + NOISE * rng.standard_normal(R_true.shape))
        if inject is not None and e in inject[1]:
            R[names.index(BAD)] *= 1 + inject[0] * (x[:, 0] > 0.5)
        fit = fake_fit(Theta, R)
        collector(e, fit)
        Gamma, rank[e] = Gamma + fit.Gamma / len(members), int(fit.rank)
    ensemble = types.SimpleNamespace(
        Gamma=Gamma, mask=torch.ones(Theta.shape[1], dtype=torch.bool), finite=torch.ones(N_EVENTS, dtype=torch.bool)
    )
    ensemble.rank = rank
    write_cache(path, config, ensemble)

    features = torch.from_numpy(x).float()
    w0, weights = torch.from_numpy(w0), torch.from_numpy(weights)
    binning = merge_sparse(quantile_binning(features, FEATURES, 10), features, FEATURES, w0, 30)
    cache = EnsembleCache(path)
    sums = val.component_sums(cache, binning, w0, weights)
    return types.SimpleNamespace(
        cache=cache,
        config=config,
        Theta=Theta,
        names=names,
        R_true=R_true,
        x=x,
        w0=w0,
        weights=weights,
        binning=binning,
        sums=sums,
        bias=val.component_bias(sums, binning),
    )


@pytest.fixture(scope='module')
def null(tmp_path_factory):
    return build_toy(tmp_path_factory.mktemp('null'))


@pytest.fixture(scope='module')
def biased(tmp_path_factory):
    """Bias of 5% in network BAD of every member."""
    return build_toy(tmp_path_factory.mktemp('biased'), inject=(0.05, MEMBERS))


@pytest.fixture(scope='module')
def bad_copies(tmp_path_factory):
    """
    Ten members, bias of 10% in network BAD of four of them. A bad copy in few members mostly raises Var_seeds rather
    than the ensemble-mean bias (with one bad copy among three members, Delta^2 = Var_seeds exactly, so it never
    flags); four of ten still flag, and are fewer than seed_fraction (0.5) of the members.
    """
    return build_toy(tmp_path_factory.mktemp('bad_copies'), inject=(0.1, [0, 1, 2, 3]), members=list(range(10)))


def flagged(toy, threshold=5.0):
    max_z = np.nanmax(toy.bias['Z'], axis=1)
    return [int(k) for k in np.argsort(-max_z) if max_z[k] > threshold]


def equal(a, b):
    """Bit-identical nested results (dicts, lists, arrays, tensors, scalars)."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(u, v) for u, v in zip(a, b))
    if isinstance(a, (np.ndarray, torch.Tensor)):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def test_cache_round_trip(null, tmp_path):
    cache = null.cache
    assert cache.seeds == MEMBERS
    assert cache.mask.all() and cache.finite.all()
    assert cache.manifest['rank'] == {e: 6 for e in MEMBERS}
    for e in MEMBERS:
        network = cache.network(e)
        assert network['networks'] == null.names
        assert np.allclose(network['Theta'], null.Theta)
        assert cache.seed_R(e).shape == (len(null.names), N_EVENTS)
    assert np.allclose(cache.Gamma, np.mean([cache.seed_Gamma[e] for e in MEMBERS], axis=0), rtol=1e-6)
    assert cache.matches(null.config)
    assert not cache.matches({**null.config, 'c0': [1.0, 0.5, 0.0]})
    # a retrained network makes the cache stale
    model = yaml.safe_load(open(null.config['seeds'][0]['sm']))['name'] + '/model.pt'
    original = open(model, 'rb').read()
    try:
        open(model, 'wb').write(b'retrained')
        assert not cache.matches(null.config)
    finally:
        open(model, 'wb').write(original)


def test_bias_test_without_bias(null):
    Z = null.bias['Z']
    assert np.isfinite(Z).sum() > 0.9 * Z.size
    assert np.nanmax(Z) < 5
    assert abs(np.nanmean(Z)) < 2
    assert flagged(null) == []


def test_bias_test_finds_injected_bias(biased):
    # Gamma is linear in R, so the injected bias propagates through Theta^+ into every component
    shift = biased.R_true[biased.names.index(BAD)] * 0.05 * (biased.x[:, 0] > 0.5)
    gamma_shift = np.linalg.pinv(biased.Theta)[:, biased.names.index(BAD), None] * shift
    expected = binned_sums(gamma_shift * biased.w0.numpy(), biased.binning['index'], biased.binning['n_bins'])
    q = int(np.argmax(np.abs(expected).max(1)))
    assert q in flagged(biased)
    D = biased.bias['D'][q]
    assert np.abs(D - expected[q]).max() < 0.2 * np.abs(expected[q]).max()
    assert np.nanmax(biased.bias['rms_bias'][q]) > 0


def test_attribution(biased):
    components = flagged(biased)
    networks = val.network_sums(biased.cache, biased.binning, biased.w0, biased.weights)
    shares, total = val.attribute(networks, components, biased.sums['sm_yield'])
    D = biased.bias['D'][components]
    assert np.abs(total - D).max() / np.abs(D).max() < 1e-3
    assert np.allclose(sum(shares.values()), 1)
    top = max(shares, key=lambda n: abs(shares[n][0]))
    assert top == BAD


def test_suspects_drop_a_bad_point(biased):
    networks = val.network_sums(biased.cache, biased.binning, biased.w0, biased.weights)
    shares, _ = val.attribute(networks, flagged(biased), biased.sums['sm_yield'])
    z = val.network_z(networks, biased.sums['sm_yield'], biased.binning)
    drop, retrain = val.suspects(shares, z, networks)
    assert drop == [BAD] and retrain == {}


def test_suspects_retrain_bad_copies(bad_copies):
    toy = bad_copies
    networks = val.network_sums(toy.cache, toy.binning, toy.w0, toy.weights)
    shares, _ = val.attribute(networks, flagged(toy), toy.sums['sm_yield'])
    z = val.network_z(networks, toy.sums['sm_yield'], toy.binning)
    drop, retrain = val.suspects(shares, z, networks)
    assert flagged(toy)
    assert drop == [] and retrain == {BAD: [0, 1, 2, 3]}


def test_drop_shortcut_equals_exact_refit(biased):
    groups = {'bad': [BAD], 'cross': CROSS[:2]}
    shortcut = val.drop_bias_sums(biased.cache, groups, biased.binning, biased.w0, biased.weights, biased.sums)
    scale = np.abs(biased.sums['truth']).max()
    for g, members in groups.items():
        refit = val.refit_without(biased.cache, members)
        exact = val.component_sums(refit, biased.binning, biased.w0, biased.weights)['bias']
        assert np.abs(shortcut[g]['bias'] - exact).max() < 1e-5 * scale


def test_drop_tests(biased):
    groups = {'bad': [BAD], 'harmless': ['cB/p_one'], 'every cross point': CROSS}
    sums = val.drop_bias_sums(biased.cache, groups, biased.binning, biased.w0, biased.weights, biased.sums)
    after = {g: None if s is None else val.component_bias(s, biased.binning) for g, s in sums.items()}
    blamed = {g: flagged(biased) for g in groups}
    result = val.test_drops(biased.bias, after, blamed)
    assert result['bad']['accepted'] and result['bad']['gain'] >= 2
    assert not result['harmless']['accepted'] and result['harmless']['gain'] < 2
    # without every cross point the interference term is undetermined: not testable
    assert sums['every cross point'] is None
    assert not result['every cross point']['accepted'] and isinstance(result['every cross point']['harmed'], str)


def test_workers_give_identical_results(biased):
    args = (biased.cache, biased.binning, biased.w0, biased.weights)
    assert equal(val.network_sums(*args, workers=1), val.network_sums(*args, workers=2))
    one, two = val.refit_without(biased.cache, [BAD], workers=1), val.refit_without(biased.cache, [BAD], workers=2)
    assert equal(vars(one), vars(two))
    groups = {'bad': [BAD]}
    assert equal(
        val.drop_bias_sums(*args[:1], groups, *args[1:], biased.sums, workers=1),
        val.drop_bias_sums(*args[:1], groups, *args[1:], biased.sums, workers=2),
    )


def test_design_uncertainty(null):
    labels, _ = component_labels(WCS, np.ones(6, dtype=bool))
    no_cross = np.array([n not in CROSS for n in null.names])
    unconstrained = val.unconstrained_components(null.Theta[no_cross])
    assert [labels[k] for k in np.flatnonzero(unconstrained)] == ['cB*cA']
    assert not val.unconstrained_components(null.Theta).any()
    # propagated covariance: H^-1 Theta^T diag(m^2) Theta H^-1 for a full-rank design
    m = np.linspace(0.01, 0.02, len(null.Theta))
    _, sigma0, _, _ = val.search_points(null.Theta, m, [], np.eye(6))
    H = np.linalg.inv(null.Theta.T @ null.Theta)
    cov = H @ null.Theta.T @ np.diag(m**2) @ null.Theta @ H
    assert np.allclose(sigma0, np.sqrt(np.diag(cov)))


def test_point_search(null):
    labels, _ = component_labels(WCS, np.ones(6, dtype=bool))
    misclosure = {n: 0.01 for n in null.names}
    m = np.full(len(null.names), 0.01)

    def yield_ratio(theta):
        return 1 + theta[1] ** 2 + theta[2] ** 2

    def units(max_yield):
        return val.candidate_units(WCS, null.Theta, null.names, misclosure, np.ones(6, bool), yield_ratio, max_yield)

    # the existing (+-1, +-1) cross points are not offered again; the cap removes points above it
    offered = units(9.0)
    assert offered and all(u['kind'] == 'cross' and len(u['points']) == 4 for u in offered)
    assert all(max(abs(p['theta'][1]), abs(p['theta'][2])) == 2 for u in offered for p in u['points'])
    assert all(u['max_yield_ratio'] <= 6 for u in units(6.5))

    # the search improves the interference term most with the largest lever arm
    q = labels.index('cB*cA')
    G = np.zeros((6, 6))
    G[q, q] = 1
    chosen, sigma0, sigma1, history = val.search_points(null.Theta, m, offered, G, n_units=1)
    assert len(chosen) == 1 and history[1] < history[0] and sigma1[q] < sigma0[q]
    assert all(abs(p['theta'][1]) == 2 and abs(p['theta'][2]) == 2 for p in chosen[0]['points'])


def test_scan_moments():
    a, b = 0.5, 2.0
    G = val.scan_moments(['cA'], {'cA': a}, np.ones(3, bool))  # columns 1, theta, theta^2
    assert np.allclose(G, [[1, 0, a**2 / 3], [0, a**2 / 3, 0], [a**2 / 3, 0, a**4 / 5]])
    G = val.scan_moments(['cA'], {'cA': a}, np.ones(3, bool), shape='gaussian')
    assert np.isclose(G[1, 1], a**2) and np.isclose(G[2, 2], 3 * a**4)
    # two WCs, columns 1, cA, cA^2, cB, cB*cA, cB^2: varied together (box) or one at a time (1d)
    box = val.scan_moments(WCS, {'cA': a, 'cB': b}, np.ones(6, bool))
    one_d = val.scan_moments(WCS, {'cA': a, 'cB': b}, np.ones(6, bool), mode='1d')
    assert np.isclose(box[2, 5], a**2 / 3 * b**2 / 3) and np.isclose(one_d[2, 5], 0)
    assert np.isclose(one_d[2, 2], a**4 / 5 / 2)


def test_binning():
    x, _, _ = truth()
    features = torch.from_numpy(x).float()
    binning = quantile_binning(features, FEATURES, 10)
    counts = np.bincount(binning['index'][0].numpy(), minlength=binning['n_bins'])[binning['feature'] == 0]
    assert len(counts) == 10 and counts.max() - counts.min() <= 1
    assert binning['discrete'] == [False, False, True] and (binning['feature'] == 2).sum() == 3

    # heavy-tailed weights: merged bins reach the effective number of events, every event stays in one bin
    weights = np.exp(2 * np.random.default_rng(1).standard_normal(N_EVENTS))
    merged = merge_sparse(binning, features, FEATURES, weights, 30)
    for f in range(3):
        index = merged['index'][f].numpy()
        assert (merged['feature'][index] == f).all()
        w = np.bincount(index, weights=weights, minlength=merged['n_bins'])
        w2 = np.bincount(index, weights=weights**2, minlength=merged['n_bins'])
        own = merged['feature'] == f
        assert (w[own] ** 2 / w2[own] >= 30).all()
