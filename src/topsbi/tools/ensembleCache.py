from topsbi.tools.data import get_lower_tri
from topsbi.tools.plots import merge_sparse_bins

import datetime, hashlib, os, shutil, torch, yaml
import numpy as np


def quantile_binning(features, feature_config, nbins=20, names=None):
    """
    Quantile bins of network inputs, placed end to end on one axis so one index_add fills every feature.

    Args:
        features: [N, n_features] raw network inputs
        feature_config: contents of the features yml, {name: {'loc': column, ...}}
        nbins: quantile bins per feature; a feature with at most nbins distinct values gets one bin per value
        names: features to bin, default every feature in feature_config
    Returns:
        dict with
            names: binned features, in order
            edges: bin edges per feature (for a discrete feature, its distinct values plus max + 1)
            discrete: per feature, whether it is binned by value
            feature: [n_bins] index into names of each bin
            index: [F, N] bin of each event in each feature, offset into [0, n_bins)
            n_bins: total number of bins
    """
    names = list(feature_config) if names is None else list(names)
    edges, discrete, index, feature = [], [], [], []
    offset = 0
    for f, name in enumerate(names):
        x = features[:, feature_config[name]['loc']].double().numpy()
        values = np.unique(x)
        if len(values) <= nbins:
            e = np.append(values, values[-1] + 1)
            discrete.append(True)
        else:
            e = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
            discrete.append(False)
        k = np.clip(np.searchsorted(e, x, side='right') - 1, 0, len(e) - 2)
        n = len(e) - 1
        edges.append(e)
        index.append(k + offset)
        feature += [f] * n
        offset += n
    return {
        'names': names,
        'edges': edges,
        'discrete': discrete,
        'feature': np.array(feature),
        'index': torch.from_numpy(np.stack(index)),
        'n_bins': offset,
    }


def merge_sparse(binning, features, feature_config, weights, min_neff=30):
    """
    Merge neighbouring bins of each feature until each holds at least min_neff effective events,
    n_eff = (sum w)^2 / sum w^2, with the rule of plots.merge_sparse_bins (accumulate from the low edge,
    fold a sparse remainder into the last bin).

    Args:
        binning: output of quantile_binning
        features: [N, n_features] raw network inputs used to build binning
        feature_config: contents of the features yml
        weights: [N] event weights the test sums (w0 for the bias test)
        min_neff: minimum effective number of events per bin
    Returns:
        a new binning dict with the same keys
    """
    weights = np.asarray(weights, dtype=np.float64)
    edges, index, feature = [], [], []
    offset = 0
    for f, name in enumerate(binning['names']):
        x = features[:, feature_config[name]['loc']].double().numpy()
        e = merge_sparse_bins(x, weights, binning['edges'][f], min_neff)
        k = np.clip(np.searchsorted(e, x, side='right') - 1, 0, len(e) - 2)
        edges.append(e)
        index.append(k + offset)
        feature += [f] * (len(e) - 1)
        offset += len(e) - 1
    return {
        'names': binning['names'],
        'edges': edges,
        'discrete': binning['discrete'],
        'feature': np.array(feature),
        'index': torch.from_numpy(np.stack(index)),
        'n_bins': offset,
    }


def binned_sums(values, index, n_bins):
    """
    Sum values [M, N] into every bin of every feature: [M, n_bins].

    Args:
        values: per-event values, one row per quantity
        index: [F, N] bin of each event in each feature (binning['index'], restricted to the same events as values)
        n_bins: total number of bins (binning['n_bins'])
    """
    values = torch.as_tensor(values, dtype=torch.float64)
    out = torch.zeros(values.shape[0], n_bins, dtype=torch.float64)
    for k in index:
        out.index_add_(1, k, values)
    return out.numpy()


def component_labels(wcs, mask):
    """
    Label and kind of every kept get_lower_tri column, in the order of the rows of gamma.

    Column (i, j), j <= i, of get_lower_tri([1, theta_1, ...]) multiplies theta_i * theta_j (theta_0 = 1):
    (0, 0) 'sm', (i, 0) the linear term 'wc_i', (i, i) the quadratic term 'wc_i^2', otherwise the interference
    term 'wc_i*wc_j'.

    Args:
        wcs: WC names in get_lower_tri order
        mask: kept get_lower_tri columns
    Returns:
        labels: [K] names of the components
        kinds: [K] 'sm', 'linear', 'quadratic' or 'cross'
    """
    names = ['sm'] + list(wcs)
    rows, cols = np.tril_indices(len(names))
    if len(rows) != len(mask):
        raise ValueError(f'{len(wcs)} wcs give {len(rows)} columns but the mask has {len(mask)}')
    labels, kinds = [], []
    for i, j, keep in zip(rows, cols, mask):
        if not keep:
            continue
        if i == 0:
            labels.append('sm'), kinds.append('sm')
        elif j == 0:
            labels.append(names[i]), kinds.append('linear')
        elif i == j:
            labels.append(f'{names[i]}^2'), kinds.append('quadratic')
        else:
            labels.append(f'{names[i]}*{names[j]}'), kinds.append('cross')
    return labels, kinds


class SeedCollector:
    def __init__(self, seeds):
        """
        Keep each seed's fit while ensemble discards it: per-event gammas, kept events and fit residuals (layer 1 of
        the ensemble cache), and Theta, the network names and every network's per-event ratio (layer 2).
        Pass an instance as ensemble(..., per_seed=collector).

        Args:
            seeds: config['seeds'], for the network names
        """
        self.seeds = seeds
        self.layer = {}

    def __call__(self, seed, fit):
        paths = self.seeds[seed]['networks'] + [self.seeds[seed]['sm']]
        base = os.path.commonpath([os.path.dirname(p) for p in paths])
        self.layer[seed] = {
            'networks': [os.path.relpath(os.path.dirname(p), base) for p in paths],
            'Theta': fit.trainingMatrix.numpy(),
            'rank': int(fit.rank),
            'filter': fit.infFilter.numpy(),
            'gammas': fit.gammas.numpy(),
            # every network's ratio for every event (inf where a network diverged; filter marks the usable events)
            'ratios': fit.ratios.numpy(),
            # empty unless Theta has more rows than columns and full column rank
            'residuals': fit.residuals.numpy() if fit.residuals.numel() else None,
        }


def model_hashes(seeds):
    """sha256 of the model.pt of every network (SM last), per seed: {seed: [hash, ...]} in config['seeds'] order."""
    hashes = {}
    for seed, networks in seeds.items():
        hashes[seed] = []
        for path in networks['networks'] + [networks['sm']]:
            with open(path) as f:
                name = yaml.safe_load(f)['name']
            with open(f'{name}/model.pt', 'rb') as f:
                hashes[seed].append(hashlib.sha256(f.read()).hexdigest())
    return hashes


def write_cache(path, config, dre, layer):
    """
    Write an ensemble cache to path. Layer 1 (event level): mean gammas, mask, events kept by every seed, and per seed
    the gammas, kept events and fit residuals. Layer 2 (network level): per seed Theta, the network names and every
    network's per-event ratio. The cache is built in path.tmp and moved into place, and ensemble.yml is written last,
    so an interrupted write never looks complete.

    Args:
        path: cache directory, replaced if it exists
        config: validation config (data, c0, wcs, seeds, lumi)
        dre: the fitted ensemble
        layer: SeedCollector.layer
    """
    tmp = f'{path}.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    np.save(f'{tmp}/gamma_cache.npy', dre.gammas.numpy())
    np.save(f'{tmp}/mask_cache.npy', dre.zeros_mask.numpy())
    np.save(f'{tmp}/filter_cache.npy', dre.infFilter.numpy())
    for seed, s in layer.items():
        d = f'{tmp}/seed_{seed}'
        os.makedirs(d)
        np.save(f'{d}/gammas.npy', s['gammas'])
        np.save(f'{d}/filter.npy', s['filter'])
        if s['residuals'] is not None:
            np.save(f'{d}/residuals.npy', s['residuals'])
        np.save(f'{d}/ratios.npy', s['ratios'])
        np.savez(f'{d}/network.npz', Theta=s['Theta'], networks=np.array(s['networks']))
    manifest = {
        'data': config['data'],
        'c0': config['c0'],
        'wcs': config['wcs'],
        'lumi': config['lumi'],
        'seeds': config['seeds'],
        'model_hash': model_hashes(config['seeds']),
        'rank': {seed: s['rank'] for seed, s in layer.items()},
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(f'{tmp}/ensemble.yml', 'w') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(tmp, path)


class EnsembleCache:
    def __init__(self, path, mmap=True):
        """
        Read an ensemble cache written by write_cache. Large per-event arrays are memory-mapped unless mmap is False;
        layer 2 is read on demand with network(seed) and seed_ratios(seed).
        """
        if not os.path.exists(f'{path}/ensemble.yml'):
            raise FileNotFoundError(f'{path} is not a complete ensemble cache')
        with open(f'{path}/ensemble.yml') as f:
            self.manifest = yaml.safe_load(f)
        mode = 'r' if mmap else None
        self.path = path
        self.mode = mode
        self.gammas = np.load(f'{path}/gamma_cache.npy', mmap_mode=mode)
        self.mask = np.load(f'{path}/mask_cache.npy')
        self.filter = np.load(f'{path}/filter_cache.npy')
        self.seeds = list(self.manifest['seeds'])
        self.seed_gammas = {s: np.load(f'{path}/seed_{s}/gammas.npy', mmap_mode=mode) for s in self.seeds}
        self.seed_filter = {s: np.load(f'{path}/seed_{s}/filter.npy') for s in self.seeds}
        self.residuals = {
            s: np.load(f'{path}/seed_{s}/residuals.npy') if os.path.exists(f'{path}/seed_{s}/residuals.npy') else None
            for s in self.seeds
        }

    def network(self, seed):
        """Theta and the network names of one seed (layer 2); row j of Theta belongs to network j."""
        with np.load(f'{self.path}/seed_{seed}/network.npz') as f:
            out = {k: f[k] for k in f.files}
        out['networks'] = [str(n) for n in out['networks']]
        return out

    def seed_ratios(self, seed):
        """Every network's per-event ratio for one seed (layer 2): [n_networks, N], rows ordered as in network(seed)."""
        return np.load(f'{self.path}/seed_{seed}/ratios.npy', mmap_mode=self.mode)

    def matches(self, config):
        """Whether the cache was built from the same data, c0, networks and network weights (model.pt) as config."""
        if any(self.manifest[k] != config[k] for k in ('data', 'c0', 'seeds')):
            return False
        return self.manifest['model_hash'] == model_hashes(config['seeds'])

    def __call__(self, theta):
        """Learned likelihood ratio r_hat(x; theta) for the events kept by every seed."""
        return get_lower_tri(theta, dtype=torch.float64).numpy()[self.mask] @ self.gammas
