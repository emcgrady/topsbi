import numpy as np
import torch, tqdm


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
    coefs /= (coefs @ get_lower_tri(config['cg'])).mean()
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


def get_weights(coefs: torch.tensor, config: dict):
    """
    return probabilities based on hypotheses in pass config file.

    Args:
        coefs: torch tensor whose rows represent each event and whose columns are the structure constants for the expanded quadratic
        config: dictionary containing lists of WC values at c0 and c1
    Returns:
        w0: event weight ratio under c0
        w1: event weight ratio under c1
        wg: event weight ratio under cg
    """
    pg = coefs @ get_lower_tri(config['cg'])
    p0 = (coefs @ get_lower_tri(config['c0'])) / pg
    p1 = (coefs @ get_lower_tri(config['c1'])) / pg

    return p0, p1, pg


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
    pg = coefs @ get_lower_tri(config['cg'])
    p0 = coefs @ get_lower_tri(config['c0'])
    p1 = coefs @ get_lower_tri(config['c1'])
    pg /= pg.mean()
    p0 /= (p0.mean()) * pg
    p1 /= (p1.mean()) * pg
    return p0, p1, pg


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


def prepare_features(features: torch.tensor):
    """
    Normalize features for use in training.

    Args:
        features: torch tensor whose rows are each event and whose columns are each features
    Returns:
        features normalized with mean 0 and std 1
    """
    return (features - features.mean(0)) / features.std(0)


def sample_boostrap(sample, seed):
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
