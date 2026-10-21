import numpy as np
import os, torch, tqdm, yaml


def find_zero_terms(coefs: torch.tensor, wcs: list, lumi_per_event: torch.tensor, tol: float = 1e-6):
    """
    WC pairs whose structure constants s_q(z) contribute negligibly to the total yield.

    Uses sum_i L |s_q(z_i)| / sum_i L |s_SM(z_i)| so every event counts with its weight and
    positive/negative contributions cannot cancel.
    """
    names = ['sm'] + list(wcs)
    w = lumi_per_event.to(coefs.dtype)[:, None] * coefs.abs()
    rel = w.sum(0) / w[:, 0].sum()
    rows, cols = torch.tril_indices(len(names), len(names))
    return [[names[i], names[j]] for i, j, r in zip(rows.tolist(), cols.tolist(), rel.tolist()) if r < tol]


def load_zero_terms(config: dict):
    """WC pairs listed in the file at config['zero_terms']; none if the key is absent."""
    if 'zero_terms' not in config:
        return []
    with open(config['zero_terms']) as f:
        return yaml.safe_load(f)['zero_terms']


def zero_term_mask(wcs: list, zero_terms: list):
    """Mask over the get_lower_tri columns that is False for the listed (structurally zero) WC pairs."""
    names = ['sm'] + list(wcs)
    mask = torch.ones(len(names) * (len(names) + 1) // 2, dtype=torch.bool)
    for a, b in zero_terms:
        i, j = sorted((names.index(a), names.index(b)), reverse=True)
        mask[i * (i + 1) // 2 + j] = False
    return mask


def kept_columns(config: dict, n_cols: int):
    """Columns of the get_lower_tri expansion to keep: all of them unless config lists zero_terms."""
    zero_terms = load_zero_terms(config)
    if not zero_terms:
        return torch.ones(n_cols, dtype=torch.bool, device=config['device'])
    keep = zero_term_mask(config['wcs'], zero_terms)
    if keep.numel() != n_cols:
        raise ValueError(f'wcs give {keep.numel()} columns but the data have {n_cols}')
    return keep.to(config['device'])


def prepare_weights(
    data: torch.utils.data.TensorDataset,
    config: dict,
):
    """
    Features and the two weights of an EFT training.

    Args:
        data: TensorDataset (or a Subset of one) whose first two tensors are (features x, coefs); further tensors
            (ids, and in the combined format event_id, syst_ratio, w_ref) are ignored
        config: training config; 'c0' / 'c1' are theta_0 / theta_1, 'lumi' the per-era luminosities
    Returns:
        features, w0, w1: x, w^s(theta_0) and w^s(theta_1) (get_weights)
    """
    if config['method'] != 'eft':
        raise ValueError(f"prepare_weights serves the method 'eft', not {config['method']!r}")
    year_loc = config.get('year_loc', 21)
    features, coefs = data[:][:2]
    features = features.to(torch.float32)
    w0, w1 = get_weights(coefs, config, features[:, year_loc])
    return features, w0, w1


def get_lower_tri(
    theta: list,
    dtype: type = torch.float32,
    off_diag: float = 1.0,
):
    """
    Quadratic expansion T(theta) of an SM-inclusive WC point theta = (1, theta_1, ..., theta_n): the lower triangle,
    in (i, j <= i) order, of the outer product theta theta^T, so that w(z; theta) = s_q(z) . T(theta).

    Args:
        theta: SM-inclusive WC point (first entry 1)
        dtype: dtype of the result
        off_diag: factor to multiply off-diagonal elements by
    Returns:
        T(theta), length (n + 1)(n + 2) / 2
    """
    if type(theta) != torch.tensor:
        theta = torch.tensor(theta)
    if theta.dtype != dtype:
        theta = theta.to(dtype)
    outer = torch.outer(theta, theta)
    n = outer.shape[0]
    rows, cols = torch.tril_indices(n, n)
    off_rows, off_cols = torch.tril_indices(n, n, offset=-1)
    outer[off_rows, off_cols] *= off_diag
    return outer[rows, cols]


def lumi_per_event(years: torch.tensor, config: dict):
    """
    Integrated luminosity of each event's era, from config['lumi'] indexed by the year_int feature.

    Args:
        years: era index of each event (year_int feature)
        config: dictionary with 'lumi' (list of per-era luminosities)
    Returns:
        float64 tensor [N] of luminosities, on the device of years
    """
    if 'lumi' not in config:
        raise KeyError("'lumi' is required: a list of luminosities indexed by the year_int feature")
    return torch.tensor(config['lumi'], dtype=torch.float64, device=years.device)[years.long()]


# tensor order of combined nominal / variation samples for nuisance training; w_ref is each event's weight at the
# reference WC point chosen when the sample was combined (the same point for nominal and every variation)
NUISANCE_COLUMNS = {
    'nominal': ('features', 'coefs', 'ids', 'event_id', 'syst_ratio', 'w_ref'),
    'variation': ('features', 'w_ref', 'ids', 'event_id'),
}


def sample_weight(tensors: list, layout: str, config: dict):
    """
    Event weight of a combined nominal or variation sample: w^s(theta_ref) = L_era w_ref x 1000, where w_ref is the
    stored weight at the reference point theta_ref (theta_SM or theta_g, chosen when the sample was combined).

    Args:
        tensors: tensors of a combined train.p / validation.p (NUISANCE_COLUMNS order)
        layout: 'nominal' or 'variation'
        config: dictionary with 'lumi' and 'device' (optional 'year_loc', default 21)
    Returns:
        float32 tensor [N] of weights
    """
    columns = dict(zip(NUISANCE_COLUMNS[layout], tensors))
    years = columns['features'][:, config.get('year_loc', 21)]
    w_ref = columns['w_ref'][:, 0].to(torch.float64)
    return (lumi_per_event(years, config) * w_ref * 1000).to(torch.float32)


# offset that keeps the train/test salt distinct from the salt (0) used for train/validation when combining
SPLIT_SALT_OFFSET = 1 << 32


def splitmix64(x: np.ndarray):
    """
    splitmix64 finalizer on a uint64 array: every input bit affects every output bit; arithmetic wraps modulo 2**64.

    The step adds the golden-ratio increment and applies the mix64 finalizer of SplitMix (G. L. Steele Jr., D. Lea and
    C. H. Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014, doi:10.1145/2660193.2660195), whose
    shifts and multipliers are D. Stafford's Mix13 ("Better Bit Mixing - Improving on MurmurHash3's 64-bit
    Finalizer", 2011), as in S. Vigna's reference splitmix64.c (https://prng.di.unimi.it/splitmix64.c). The same
    function makes the train/validation split in ttbarEFT analysis/combine_tensors.py (split_fraction), so
    event_split keys the train/test split the same way with a different salt.
    """
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def event_keys(tensors: list, layout: str, year_loc: int = 21):
    """
    Identity of each event: (era, mtt slice, every event_id column).

    The same event has the same key in nominal and in every variation. The era and mtt slice are included because
    (run, lumi block, event) can repeat between the private samples.

    Args:
        tensors: tensors of a combined train.p / validation.p (NUISANCE_COLUMNS order)
        layout: 'nominal' or 'variation'
        year_loc: feature column of the era index
    Returns:
        int64 numpy array [N, 2 + number of event_id columns]
    """
    columns = dict(zip(NUISANCE_COLUMNS[layout], tensors))
    return np.column_stack(
        [
            columns['features'][:, year_loc].numpy().astype(np.int64),
            columns['ids'][:, 2].numpy().astype(np.int64),
            columns['event_id'].numpy().astype(np.int64),
        ]
    )


def matched_events(keys0: np.ndarray, keys1: np.ndarray):
    """
    Event index i of each nominal and shifted row, over the set of distinct events.

    An event in both samples gets the same index in both, so two-sample metrics can treat each MC event as one
    independent unit. Rows with the same key in one sample share an index.

    Args:
        keys0: event_keys of the nominal rows
        keys1: event_keys of the shifted rows
    Returns:
        (i0, i1): int64 tensors [N0], [N1] with values in [0, number of distinct events)
    """
    if keys0.shape[1] != keys1.shape[1]:
        raise ValueError(f'event keys have {keys0.shape[1]} and {keys1.shape[1]} columns')
    _, events = np.unique(np.concatenate([keys0, keys1]), axis=0, return_inverse=True)
    events = torch.from_numpy(events.reshape(-1))
    return events[: len(keys0)], events[len(keys0) :]


def event_split(tensors: list, layout: str, seed: int, test_fraction: float = 0.25, year_loc: int = 21):
    """
    Test-split mask (early stopping, LR scheduler) from a hash of each event's identity.

    The key is the event_keys row (era, mtt slice, event_id), so an event is on the same side in nominal and in every
    variation, and in every nuisance training with the same seed. The seed changes the partition, as in
    sample_bootstrap; the salt is offset so it never reproduces the train/validation split made when combining.

    Args:
        tensors: tensors of a combined train.p (NUISANCE_COLUMNS order)
        layout: 'nominal' or 'variation'
        seed: training seed
        test_fraction: fraction of events in the test split
        year_loc: feature column of the era index
    Returns:
        boolean tensor [N], True for test events
    """
    key = np.full(len(tensors[0]), SPLIT_SALT_OFFSET + seed, dtype=np.uint64)
    for field in event_keys(tensors, layout, year_loc).T:
        key = splitmix64(key ^ field.astype(np.uint64))
    u = (key >> np.uint64(11)).astype(np.float64) / 2.0**53
    return torch.from_numpy(u < test_fraction)


def syst_ratio_column(root: str, systematic: str, direction: str):
    """
    Index of a weight systematic and direction in the nominal syst_ratio tensor.

    Args:
        root: combined data directory (holds nominal/systematics.yml)
        systematic: name as in systematics.yml, e.g. 'renorm'
        direction: 'up' or 'down' (case-insensitive)
    Returns:
        (k, d): syst_ratio[:, k, d] is r(x, z; theta, nu) = w^s(theta, nu) / w^s(theta, 0) of that systematic
    """
    with open(f'{root}/nominal/systematics.yml') as f:
        spec = yaml.safe_load(f)
    names = spec['syst_ratio']
    directions = [d.lower() for d in spec.get('directions', ['Up', 'Down'])]
    if systematic not in names:
        raise ValueError(f'unknown weight systematic {systematic!r}; available: {names}')
    if direction.lower() not in directions:
        raise ValueError(f'unknown direction {direction!r}; available: {directions}')
    return names.index(systematic), directions.index(direction.lower())


def load_combined(path: str, layout: str, names: tuple):
    """
    Selected columns of a combined train.p / validation.p, read through a memory map.

    Columns that are not requested (e.g. the 153 float64 coefs) are never read from disk.

    Args:
        path: file to load
        layout: 'nominal' or 'variation' (NUISANCE_COLUMNS)
        names: columns to load
    Returns:
        list in NUISANCE_COLUMNS order, None for columns not loaded (accepted by sample_weight and event_split)
    """
    tensors = torch.load(path, weights_only=False, mmap=True).tensors
    if len(tensors) != len(NUISANCE_COLUMNS[layout]):
        raise ValueError(f'{path}: {len(tensors)} tensors, expected {NUISANCE_COLUMNS[layout]}')
    return [t.clone() if name in names else None for name, t in zip(NUISANCE_COLUMNS[layout], tensors)]


def weight_shift_sample(tensors: list, config: dict, k: int, d: int):
    """
    Paired sample of a weight systematic: the nominal events with w^s(theta, 0) and w^s(theta, nu).

    Args:
        tensors: nominal tensors (load_combined)
        config: dictionary with 'lumi' and 'device'
        k: syst_ratio column of the systematic (syst_ratio_column)
        d: direction index, 0 up / 1 down
    Returns:
        features x, w0 = w^s(theta, 0) (sample_weight), w1 = w^s(theta, nu) = w0 r(x, z; theta, nu)
    """
    columns = dict(zip(NUISANCE_COLUMNS['nominal'], tensors))
    w0 = sample_weight(tensors, 'nominal', config)
    return columns['features'], w0, w0 * columns['syst_ratio'][:, k, d]


def kinematic_shift_sample(nominal: list, shifted: list, config: dict):
    """
    Two-sample data of a kinematic systematic.

    Args:
        nominal: nominal tensors (load_combined)
        shifted: variation tensors (load_combined)
        config: dictionary with 'lumi' and 'device'
    Returns:
        {'rows': (features, weights, labels) for Model.labelled_loss, label 0 nominal / 1 shifted,
         'nominal': (x, w0, i0), 'shifted': (x_shift, w1, i1) for plots and metrics, with w0 = w^s(theta, 0),
         w1 = w^s(theta, nu) and i0 / i1 the event index i (matched_events), the same for an event in both samples}
    """
    x = dict(zip(NUISANCE_COLUMNS['nominal'], nominal))['features']
    x_shift = dict(zip(NUISANCE_COLUMNS['variation'], shifted))['features']
    w0, w1 = sample_weight(nominal, 'nominal', config), sample_weight(shifted, 'variation', config)
    i0, i1 = matched_events(event_keys(nominal, 'nominal'), event_keys(shifted, 'variation'))
    labels = torch.cat([torch.zeros_like(w0), torch.ones_like(w1)])
    return {
        'rows': (torch.cat([x, x_shift]), torch.cat([w0, w1]), labels),
        'nominal': (x, w0, i0),
        'shifted': (x_shift, w1, i1),
    }


def _select(tensors: list, mask: torch.tensor):
    """rows of every loaded column (None columns stay None)"""
    return [None if t is None else t[mask] for t in tensors]


def _variation_eras(shifted: list, year_loc: int):
    """era indices (year_int feature) present in a variation sample"""
    features = dict(zip(NUISANCE_COLUMNS['variation'], shifted))['features']
    return torch.unique(features[:, year_loc].to(torch.int64))


def _keep_eras(nominal: list, eras: torch.tensor, year_loc: int):
    """
    Nominal rows of the given eras. A kinematic variation produced only in some eras (JER per year) leaves the other
    eras unchanged, r(x, z; theta, nu) = 1 there, so they are not part of its training.
    """
    features = dict(zip(NUISANCE_COLUMNS['nominal'], nominal))['features']
    return _select(nominal, torch.isin(features[:, year_loc].to(torch.int64), eras))


def _check_same_sample_setup(root: str, variation: str):
    """
    Nominal and the variation must share the reference point and the train/validation split salt.

    Only checked when both directories have a sample.yml; samples made without one are accepted.
    """
    specs = []
    for name in ('nominal', variation):
        path = f'{root}/{name}/sample.yml'
        if not os.path.exists(path):
            return
        with open(path) as f:
            specs.append(yaml.safe_load(f))
    for key in ('reference', 'salt_split'):
        if specs[0].get(key) != specs[1].get(key):
            raise ValueError(f'{key} differs: nominal {specs[0].get(key)!r}, {variation} {specs[1].get(key)!r}')


def load_nuisance_data(config: dict):
    """
    Train / test / holdout data of a nuisance training (weight_shift or kinematic_shift).

    train.p is split into train and test (early stopping) by event_split with the training seed, so an event is on
    the same side in nominal and the variation; validation.p is the holdout. The variation of kinematic_shift is the
    directory <systematic><Up|Down>; its nominal sample is restricted to the eras present in the variation (a
    variation produced only for some eras, e.g. JER per year, leaves the others unchanged).

    Args:
        config: dictionary with 'method', 'data', 'systematic', 'direction', 'seed', 'lumi', 'device'
    Returns:
        {'train', 'test', 'holdout'}: (features, w0, w1) for weight_shift, kinematic_shift_sample dicts otherwise
    """
    root, method, seed = config['data'], config['method'], config['seed']
    direction = config['direction'].lower()
    variation = None
    if method == 'weight_shift':
        k, d = syst_ratio_column(root, config['systematic'], direction)
        need = ('features', 'ids', 'event_id', 'syst_ratio', 'w_ref')
    elif method == 'kinematic_shift':
        variation = f"{config['systematic']}{direction.capitalize()}"
        if not os.path.isdir(f'{root}/{variation}'):
            raise FileNotFoundError(f'no variation {variation!r} in {root}')
        _check_same_sample_setup(root, variation)
        need = ('features', 'ids', 'event_id', 'w_ref')
    else:
        raise ValueError(f'{method!r} is not a nuisance method')

    def build(nominal, shifted):
        if method == 'weight_shift':
            return tuple(t.to(config['device']) for t in weight_shift_sample(nominal, config, k, d))
        sample = kinematic_shift_sample(nominal, shifted, config)
        return {key: tuple(t.to(config['device']) for t in value) for key, value in sample.items()}

    year_loc = config.get('year_loc', 21)
    eras = None
    out = {}
    for split_name, file in (('train', 'train.p'), ('holdout', 'validation.p')):
        nominal = load_combined(f'{root}/nominal/{file}', 'nominal', need)
        shifted = None
        if variation is not None:
            shifted = load_combined(f'{root}/{variation}/{file}', 'variation', need)
            split_eras = _variation_eras(shifted, year_loc)
            if eras is None:
                eras = split_eras
                print(f'[INFO] {variation}: eras {eras.tolist()}; nominal rows of other eras are left out')
            elif not torch.equal(split_eras, eras):
                raise ValueError(f'{variation}: eras {split_eras.tolist()} in {file} but {eras.tolist()} in train.p')
            nominal = _keep_eras(nominal, eras, year_loc)
        if split_name == 'holdout':
            out['holdout'] = build(nominal, shifted)
            continue
        # split on the CPU (event_split hashes numpy arrays), then move to the device
        test_nominal = event_split(nominal, 'nominal', seed)
        test_shifted = None if shifted is None else event_split(shifted, 'variation', seed)
        if not test_nominal.any() or (test_shifted is not None and not test_shifted.any()):
            raise ValueError(f'empty test split in {root}: too few events for early stopping')
        if shifted is None:
            out['train'] = build(_select(nominal, ~test_nominal), None)
            out['test'] = build(_select(nominal, test_nominal), None)
        else:
            out['train'] = build(_select(nominal, ~test_nominal), _select(shifted, ~test_shifted))
            out['test'] = build(_select(nominal, test_nominal), _select(shifted, test_shifted))
    return out


def is_combined(data: str):
    """True for a directory of the combined format (the combiner writes its sample.yml next to train.p)"""
    return os.path.exists(f'{data}/sample.yml')


def load_eft_combined(config: dict):
    """
    Train / test / holdout of an EFT training on the nominal directory of a combined sample (config['data']).

    train.p is split by event_split, so a seed has the same test events as the nuisance trainings with that seed and
    the split does not depend on the row order; the holdout is config['holdout'], by default <data>/validation.p.
    Only the features, coefs and event-key columns are read (load_combined).
    Args:
        config: EFT training config ('data', 'seed', 'c0', 'c1', 'lumi', 'device', optional 'holdout', 'year_loc')
    Returns:
        {'train', 'test', 'holdout'}: (x, w^s(theta_0), w^s(theta_1)) each (prepare_weights)
    """

    def weights(tensors):
        return prepare_weights([None if t is None else t.to(config['device']) for t in tensors], config)

    train = load_combined(f"{config['data']}/train.p", 'nominal', ('features', 'coefs', 'ids', 'event_id'))
    test = event_split(train, 'nominal', config['seed'], year_loc=config.get('year_loc', 21))
    if not test.any():
        raise ValueError(f"empty test split in {config['data']}: too few events for early stopping")
    holdout = load_combined(config.get('holdout') or f"{config['data']}/validation.p", 'nominal', ('features', 'coefs'))
    return {'train': weights(_select(train, ~test)), 'test': weights(_select(train, test)), 'holdout': weights(holdout)}


def get_weights(coefs: torch.tensor, config: dict, years: torch.tensor):
    """
    Stitched event weights w^s(theta_0) and w^s(theta_1).

    The stored structure constants s_q(z) (coefs) already carry xsec/nSumOfWeights and the detector scale factors,
    but not the luminosity, so w^s(theta) = L_era s_q(z) . T(theta) x 1000 (pb to fb), computed in float64.

    Args:
        coefs: structure constants s_q(z) per event (rows) and quadratic term (columns)
        config: dictionary with 'c0' (theta_0), 'c1' (theta_1) and the per-era luminosities 'lumi'
        years: era index (year_int feature) of each event, used to look up the luminosity
    Returns:
        w0: w^s(theta_0) per event
        w1: w^s(theta_1) per event
    """
    keep = kept_columns(config, coefs.shape[1])
    coefs = coefs[:, keep].to(torch.float64)
    lumi = lumi_per_event(years, config)
    w0 = lumi * (coefs @ get_lower_tri(config['c0'], dtype=torch.float64).to(config['device'])[keep]) * 1000
    w1 = lumi * (coefs @ get_lower_tri(config['c1'], dtype=torch.float64).to(config['device'])[keep]) * 1000
    return (w0).to(torch.float32), (w1).to(torch.float32)


def get_feature_normalization(features: torch.tensor):
    """
    Get mean and standard deviation to normalize a set of features.

    A feature that is constant in the sample (e.g. the era when only one era is present) gets std 1 instead of 0,
    so standardising it gives 0 rather than 0/0.
    Args:
        features: torch tensor whose rows are each event and whose columns are each features
    Returns:
        feature mean
        feature standard deviation
    """
    std = features.std(0)
    return features.mean(0), torch.where(std > 0, std, torch.ones_like(std))


def sample_bootstrap(sample: torch.utils.data.TensorDataset, seed: int):
    """
    Random 75/25 split of a dataset into training and test sets, reproducible with seed.
    """
    train, test = torch.utils.data.random_split(sample, [0.75, 0.25], generator=torch.Generator().manual_seed(seed))
    return train, test


def toy_builder(
    p_t: torch.tensor,
    p_c: torch.tensor,
    n_t: int,
    M: float = None,
    n_workers: int = 32,
) -> torch.tensor:
    """
    Rejection-sampled toy of the target theta_t from candidate events at theta_c (methods write-up, rejection
    sampled toys): event i is kept when U < alpha_i = p_t,i / (M p_c,i), until N_t ~ Poisson(n_t) events are kept.
    Can be used for Asimov-like data if enough toys are generated.

    Args:
        p_t: target weight per event, e.g. w^s(theta_t) / sum w^s(theta_t)
        p_c: candidate weight per event, e.g. w^s(theta_c) / sum w^s(theta_c)
        n_t: expected number of toy events (mean of the Poisson draw of N_t)
        M: bound with M > p_t / p_c for every event; the smallest valid value is used if None or too low
        n_workers: number of workers to use in rejection sample loop
    Returns:
        event_mask: indices of the candidate events in the toy
    """
    min_m = (p_t / p_c).max().item()
    print(f'Minimum M is {min_m:.2e}...')
    if M is None:
        M = min_m
        print(f'No M chosen. Setting M to 0.01% above its minimum ({M:.2e})')
    elif M <= min_m:
        M = min_m
        print(f'Choice of M is too low! Setting M to 0.01% above its minimum ({M:.2e})')
    else:
        print(f'Using M={M:.2f}')
    threshold = p_t / (M * p_c)
    indices = torch.tensor(range(0, p_c.shape[0]))
    batches = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(threshold, indices), batch_size=1_000, shuffle=True, num_workers=n_workers
    )
    event_mask = []
    print('Rejection sampler prepared')
    enough = False
    total = 0
    n_events = np.random.poisson(n_t)
    pbar = tqdm.tqdm(total=n_events)
    while not enough:
        last = total
        for t, index in batches:
            event_mask += [index[torch.where(torch.rand(len(t)) < t)[0]]]
        total = sum(len(m) for m in event_mask)
        pbar.update(total - last)
        if total > n_events:
            enough = True
    pbar.close()
    return torch.concatenate(event_mask)[:n_events]
