import pytest
import torch
import yaml

N_FEATURES = 22
N_WCS = 16  # Wilson coefficients, SM term not included
WCS = [
    'cQd1', 'ctj1', 'cQj31', 'ctj8', 'ctd1', 'ctd8', 'ctGRe', 'ctGIm',
    'cQj11', 'cQj18', 'ctu8', 'cQd8', 'ctu1', 'cQu1', 'cQj38', 'cQu8',
]


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
    """Minimal eft training config pointing at a small synthetic dataset."""
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
        'data': str(data_dir),
        'device': 'cpu',
        'epochs': 2,
        'features': str(features_path),
        'features_to_animate': ['feat0', 'feat1'],
        'learningRate': 0.001,
        'lumi': [19.52, 16.81, 41.48, 59.83],
        'method': 'eft',
        'name': str(tmp_path / 'out'),
        'network': [
            {'activation': 'LeakyReLU', 'out': 16, 'type': 'Linear'},
            {'activation': 'Sigmoid', 'in': 16, 'out': 1, 'type': 'Linear'},
        ],
        'seed': 42,
        'wcs': WCS,
    }


@pytest.fixture
def holdout_config(train_config, tmp_path):
    """train_config with a separate holdout dataset for the final evaluation."""
    holdout = tmp_path / 'data' / 'holdout.p'
    torch.save(make_dataset(1500, seed=1), holdout)
    return {**train_config, 'holdout': str(holdout)}


def make_combined_sample(root, n_events: int):
    """
    Small sample in the combined format read by load_nuisance_data (NUISANCE_COLUMNS):

    nominal/{train,validation}.p: (features, coefs, ids, event_id [N, 4] int64, syst_ratio [N, 2, 2], w_ref [N, 1])
    with nominal/systematics.yml naming the syst_ratio columns ISR and FSR, directions Up and Down;
    JES_XUp/, JES_XDown/: (features, w_ref, ids, event_id) of the same events with feature 0 scaled by 1.02 / 0.98,
    0.5% of the events dropped and 0.5% new ones added; JER_2017Up/: the era-2 events only, feature 0 scaled by 1.01.
    """
    root.mkdir(parents=True, exist_ok=True)
    for split, n, seed in (('train', n_events, 0), ('validation', n_events // 2, 1)):
        features, coefs, ids = make_dataset(n, seed=seed)[:]
        gen = torch.Generator().manual_seed(100 + seed)
        event_id = torch.stack([
            torch.ones(n, dtype=torch.int64),
            torch.full((n,), seed, dtype=torch.int64),
            torch.arange(n, dtype=torch.int64),
            torch.randint(-(2**62), 2**62, (n,), generator=gen),
        ], dim=1)
        isr = torch.exp(0.05 * torch.tanh((features[:, 1] - 100) / 50))
        fsr = torch.exp(0.03 * torch.tanh((features[:, 2] - 100) / 50))
        syst_ratio = torch.stack([torch.stack([isr, 1 / isr], 1), torch.stack([fsr, 1 / fsr], 1)], 1)
        w_ref = coefs[:, :1].clone()
        (root / 'nominal').mkdir(exist_ok=True)
        torch.save(
            torch.utils.data.TensorDataset(features, coefs, ids, event_id, syst_ratio.float(), w_ref),
            root / 'nominal' / f'{split}.p',
        )

        keep = torch.rand(n, generator=gen) >= 0.005
        extra_features, extra_coefs, extra_ids = make_dataset(n // 200 + 1, seed=50 + seed)[:]
        extra_id = event_id[: len(extra_features)].clone()
        extra_id[:, 2] += 10 * n
        for name, scale, rows in (
            ('JES_XUp', 1.02, keep),
            ('JES_XDown', 0.98, keep),
            ('JER_2017Up', 1.01, features[:, 21] == 2),
        ):
            shifted = features[rows].clone()
            shifted[:, 0] *= scale
            tensors = [shifted, w_ref[rows], ids[rows], event_id[rows]]
            if name.startswith('JES'):
                tensors = [
                    torch.cat([tensors[0], extra_features]),
                    torch.cat([tensors[1], extra_coefs[:, :1]]),
                    torch.cat([tensors[2], extra_ids]),
                    torch.cat([tensors[3], extra_id]),
                ]
            (root / name).mkdir(exist_ok=True)
            torch.save(torch.utils.data.TensorDataset(*tensors), root / name / f'{split}.p')
    # written by the combiner; marks the combined format for EFT training (is_combined)
    (root / 'nominal' / 'sample.yml').write_text(yaml.safe_dump({'variation': 'nominal', 'reference': 'sm'}))
    (root / 'nominal' / 'systematics.yml').write_text(
        yaml.safe_dump({'syst_ratio': ['ISR', 'FSR'], 'directions': ['Up', 'Down']})
    )
    return root


@pytest.fixture
def nuisance_config(train_config, tmp_path):
    """Minimal nuisance training config on a small combined sample; the test sets method and systematic."""
    root = make_combined_sample(tmp_path / 'combined', 2000)
    # three features keep the per-feature plots and tests short
    features = {f'feat{i}': {'label': f'feature {i}', 'loc': i, 'min': 0, 'max': 200, 'nbins': 10} for i in (0, 1, 2)}
    features_path = tmp_path / 'nuisance_features.yml'
    features_path.write_text(yaml.safe_dump(features))
    return {
        'data': str(root),
        'direction': 'up',
        'seed': 0,
        'lumi': train_config['lumi'],
        'device': 'cpu',
        'name': str(tmp_path / 'nuisance'),
        'features': str(features_path),
        'features_to_animate': ['feat0'],
        'animate_per_epoch': 1,
        'epochs': 2,
        'batchSize': 500,
        'learningRate': 0.001,
        'network': [
            {'activation': 'LeakyReLU', 'out': 16, 'type': 'Linear'},
            {'activation': 'None', 'in': 16, 'out': 1, 'type': 'Linear'},
        ],
    }
