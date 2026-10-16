import numpy as np
import torch, tqdm, yaml


def lumi_norm(s, lumi=137.64, expected_sm=140983.03):
    return s * expected_sm / (lumi * 1000 * s[0].sum())


def find_zero_terms(coefs: torch.tensor, wcs: list, lumi_per_event: torch.tensor, tol: float = 1e-6):
    """
    WC pairs whose structure constants contribute negligibly to the total yield.

    Uses sum_events L |c_k| / sum_events L |c_sm| so every event counts with its weight and
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
    method = config['method']
    if method == 'weights_only':
        year_loc = config.get('year_loc', 21)
        features, coefficients, _ = data[:]
        features = features.to(torch.float32)
        w0, w1 = get_weights(coefficients, config, features[:, year_loc])
    elif method == 'stitched':
        features, coefficients, _ = data[:]
        w0, w1 = get_probabilities(coefficients, config)
    return features, w0, w1


def get_lower_tri(
    vec: list,
    dtype: type = torch.float32,
    off_diag: float = 1.0,
):
    """
    returns lower triangual matrix of outer product of vec
    Args:
        vec: vector to take the outer product of
        off_diag: factor to multiply off-diagonal elements by
    Returns:
        lower triangular matrix
    """
    if type(vec) != torch.tensor:
        vec = torch.tensor(vec)
    if vec.dtype != dtype:
        vec = vec.to(dtype)
    expanded_vec = torch.outer(vec, vec)
    expanded_dim = expanded_vec.shape[0]
    rows, cols = torch.tril_indices(expanded_dim, expanded_dim)
    off_rows, off_cols = torch.tril_indices(expanded_dim, expanded_dim, offset=-1)
    expanded_vec[off_rows, off_cols] *= off_diag
    return expanded_vec[rows, cols]


def parameterize_weights(coefs: torch.tensor, config: dict):
    """
    randomize WC values across ranges for c1, retrun probabilites of c1 and c0 as well as WC values used.

    Args:
        coefs: torch tensor whose rows represent each event and whose columns are the structure constants for the expanded quadratic
        config: dictionary containing lists of WC value ranges for c1 and data generation point
    Returns:
        p0:  event probabilites under c0
        p1:  event probabilits under randomized c1 from config ranges
        wcs: random WC values used to calculate p1
    """
    wcs = [torch.ones(coefs.shape[0])]
    # choose random WC values
    for wc in config['wcs']:
        wcs += [
            (config['ranges'][wc]['max'] - config['ranges'][wc]['min']) * torch.rand(coefs.shape[0])
            + config['ranges'][wc]['min']
        ]
    wcs = torch.vstack(wcs)
    expanded_wcs = []
    # expand the random WC values
    for i in range(wcs.shape[0]):
        for j in range(i + 1):
            expanded_wcs += [wcs[i] * wcs[j]]
    sig = (coefs * (torch.vstack(expanded_wcs).T)).sum(1)
    bkg = coefs @ get_lower_tri(config['c0'])

    return bkg / (bkg.mean()), sig / (sig.mean()), wcs.T


def get_weights(coefs: torch.tensor, config: dict, years: torch.tensor):
    """
    return stitched event weights w^s under the hypotheses in the passed config file.

    The stored structure constants already carry xsec/nSumOfWeights and the detector
    scale factors, but not the luminosity, so w^s(c) = L_y * coefs @ T(c) (Note: pb to fb conversion of lumi)

    Args:
        coefs: torch tensor whose rows represent each event and whose columns are the structure constants for the expanded quadratic
        config: dictionary containing lists of WC values at c0 and c1, and the per-era luminosities in lumi
        years: era index (year_int feature) of each event, used to look up the luminosity
    Returns:
        w0: stitched event weight under c0
        w1: stitched event weight under c1
    """
    if 'lumi' not in config:
        raise KeyError("weights_only requires 'lumi': a list of luminosities indexed by the year_int feature")
    keep = kept_columns(config, coefs.shape[1])
    coefs = coefs[:, keep].to(torch.float64)
    lumi = torch.tensor(config['lumi'], dtype=torch.float64, device=config['device'])[years.long()]
    w0 = lumi * (coefs @ get_lower_tri(config['c0'], dtype=torch.float64).to(config['device'])[keep]) * 1000
    w1 = lumi * (coefs @ get_lower_tri(config['c1'], dtype=torch.float64).to(config['device'])[keep]) * 1000
    return (w0).to(torch.float32), (w1).to(torch.float32)


def get_probabilities(coefs: torch.tensor, config: dict):
    """
    return probabilities based on hypotheses in pass config file.

    Args:
        coefs: torch tensor whose rows represent each event and whose columns are the structure constants for the expanded quadratic
        config: dictionary containing lists of WC values at c0 and c1
    Returns:
        p0: event probability ratio under c0 and normalized by the mean
        p1: event probability ratio under c1 and normalized by the mean
    """
    keep = kept_columns(config, coefs.shape[1])
    coefs = coefs[:, keep].to(torch.float64)
    p0 = (coefs @ get_lower_tri(config['c0'], dtype=torch.float64)[keep]).to(torch.float32)
    p1 = (coefs @ get_lower_tri(config['c1'], dtype=torch.float64)[keep]).to(torch.float32)
    p0 /= p0.mean()
    p1 /= p1.mean()
    return p0, p1


def get_feature_normalization(features: torch.tensor):
    """
    Get mean and standard deviation to normalize a set of features.
    Args:
        features: torch tensor whose rows are each event and whose columns are each features
    Returns:
        feature mean
        feature standard deviation
    """
    return features.mean(0), features.std(0)


def sample_boostrap(sample: torch.utils.data.TensorDataset, seed: int):
    """
    Splits and shuffles data into training and testing
    """
    train, test = torch.utils.data.random_split(sample, [0.75, 0.25], generator=torch.Generator().manual_seed(seed))
    return train, test


def toy_builder(
    tar_prob: torch.tensor,
    can_prob: torch.tensor,
    tar_events: int,
    M: float = None,
    n_workers: int = 32,
) -> torch.tensor:
    """
    Use rejection sampling to create toy(s) for a given target distribution from a cadidate distribution.
    Can be used for Azimov data generation if a sufficient number of toys is generated.

    Args:
        tar_prob: target toy probability distribution
        can_prob: candidate distribution to create toy with
        tar_events: target number of toy events to draw a poisson from
        M: constant, finite bound on the likelihood ratio between tarProb and canProb (note: M > tarProb/canProb)
        n_workers: number of workers to use in rejection sample loop
    Returns:
        event_mask: indices of events in canProb to create toys of tarProb
    """
    min_m = (tar_prob / can_prob).max().item()
    print(f'Minimum M is {min_m:.2e}...')
    if M is None:
        M = min_m
        print(f'No M chosen. Setting M to 0.01% above its minimum ({M:.2e})')
    elif M <= min_m:
        M = min_m
        print(f'Choice of M is too low! Setting M to 0.01% above its minimum ({M:.2e})')
    else:
        print(f'Using M={M:.2f}')
    threshold = tar_prob / (M * can_prob)
    indices = torch.tensor(range(0, can_prob.shape[0]))
    batches = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(threshold, indices), batch_size=1_000, shuffle=True, num_workers=n_workers
    )
    event_mask = []
    print('Rejection sampler prepared')
    enough = False
    total = 0
    n_events = np.random.poisson(tar_events)
    pbar = tqdm.tqdm(total=n_events)
    while not enough:
        last = total
        for t, index in batches:
            event_mask += [index[torch.where(torch.rand(len(t)) < t)[0]]]
        total = len(event_mask)
        pbar.update(total - last)
        if total > n_events:
            enough = True
    pbar.close()
    return torch.concatenate(event_mask)[:n_events]
