import pytest
import torch
import yaml

N_FEATURES = 22
N_WCS = 16  # Wilson coefficients, SM term not included
WCS = [
    'cQd1', 'ctj1', 'cQj31', 'ctj8', 'ctd1', 'ctd8', 'ctGRe', 'ctGIm',
    'cQj11', 'cQj18', 'ctu8', 'cQd8', 'ctu1', 'cQu1', 'cQj38', 'cQu8',
]
CG = [1.0] + [1.5] * 6 + [-0.5] * 2 + [1.5] * 8


def make_dataset(n_events: int, seed: int = 0) -> torch.utils.data.TensorDataset:
    """
    Synthetic dataset in the format written by the ttbarEFT tensor processor + combine step:
    (features [N, 22], quadratic fit coefficients [N, 153], ids [N, 3]).

    Each event weight is w(c) = c^T A c with A positive semi-definite, so weights are positive
    at every WC point. Off-diagonal coefficients carry the factor 2, matching get_lower_tri(off_diag=1).
    """
    gen = torch.Generator().manual_seed(seed)
    n = N_WCS + 1
    features = torch.rand(n_events, N_FEATURES, generator=gen) * 200
    features[:, 21] = torch.randint(0, 4, (n_events,), generator=gen).float()  # year_int
    v = torch.ones(n_events, n, dtype=torch.float64)
    v[:, 1:] = 0.1 * torch.tanh((features[:, : n - 1].double() - 100) / 50)
    a = v[:, :, None] * v[:, None, :] + 1e-4 * torch.eye(n, dtype=torch.float64)
    rows, cols = torch.tril_indices(n, n)
    coefs = a[:, rows, cols] * torch.where(rows == cols, 1.0, 2.0).double()
    ids = torch.stack([
        torch.full((n_events,), 11.0),
        torch.full((n_events,), -13.0),
        torch.randint(0, 3, (n_events,), generator=gen).float(),
    ], dim=1)
    return torch.utils.data.TensorDataset(features, coefs, ids)


@pytest.fixture
def train_config(tmp_path):
    """Minimal weights_only training config pointing at a small synthetic dataset."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    torch.save(make_dataset(2000), data_dir / 'train.p')

    features = {
        f'feat{i}': {'label': f'feature {i}', 'loc': i, 'min': 0, 'max': 200, 'nbins': 10}
        for i in range(N_FEATURES)
    }
    features_path = tmp_path / 'features.yml'
    features_path.write_text(yaml.safe_dump(features))

    c1 = [1.0] + [0.0] * N_WCS
    c1[4] = 1.0
    return {
        'animate_per_epoch': 1,
        'batchSize': 500,
        'bootstrap': True,
        'c0': [1.0] + [0.0] * N_WCS,
        'c1': c1,
        'cg': CG,
        'data': str(data_dir),
        'device': 'cpu',
        'epochs': 2,
        'features': str(features_path),
        'features_to_animate': ['feat0', 'feat1'],
        'learningRate': 0.001,
        'lumi': [19.52, 16.81, 41.48, 59.83],
        'method': 'weights_only',
        'name': str(tmp_path / 'out'),
        'network': [
            {'activation': 'LeakyReLU', 'out': 16, 'type': 'Linear'},
            {'activation': 'Sigmoid', 'in': 16, 'out': 1, 'type': 'Linear'},
        ],
        'seed': 42,
        'wcs': WCS,
    }
