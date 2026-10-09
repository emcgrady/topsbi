import importlib

import pytest
import torch
import yaml

from topsbi.tools.data import get_lower_tri

MODULES = [
    'topsbi.model.net',
    'topsbi.tools.buildLikelihood',
    'topsbi.tools.data',
    'topsbi.tools.metrics',
    'topsbi.tools.plots',
    'topsbi.train',
    pytest.param(
        'topsbi.validation',
        marks=pytest.mark.xfail(reason='imports removed expand_array, see emcgrady/topsbi#5', strict=True),
    ),
]


@pytest.mark.parametrize('module', MODULES)
def test_import(module):
    importlib.import_module(module)


def test_get_lower_tri_matches_processor_ordering():
    """get_lower_tri must use the same (i, j<=i) ordering as expand_array in ttbarEFT's tensor_processor."""
    wcs = [1.0, 2.0, 3.0, 5.0]
    expected = [wcs[i] * wcs[j] for i in range(len(wcs)) for j in range(i + 1)]
    assert get_lower_tri(wcs, dtype=torch.float64).tolist() == expected


def test_weights_only_training(train_config, tmp_path):
    from topsbi.train import main

    main(train_config)

    out = tmp_path / 'out'
    assert (out / 'model.pt').exists()
    performance = yaml.safe_load((out / 'complete' / 'performance.yml').read_text())
    assert 0.0 <= performance['auc'] <= 1.0
    assert (out / 'complete' / 'animations' / 'feat0_log.gif').exists()


def test_get_weights_stitched():
    """weights_only weights are L_y * S @ T(c): the per-sample normalization in S is kept, not divided out by cg."""
    from conftest import CG, N_WCS
    from topsbi.tools.data import get_weights

    gen = torch.Generator().manual_seed(1)
    coefs = torch.rand(50, (N_WCS + 1) * (N_WCS + 2) // 2, generator=gen, dtype=torch.float64) * 1e-8
    years = torch.randint(0, 4, (50,), generator=gen).float()
    c1 = [1.0] + [0.0] * N_WCS
    c1[4] = 0.1
    config = {'c0': [1.0] + [0.0] * N_WCS, 'c1': c1, 'cg': CG, 'lumi': [19.52, 16.81, 41.48, 59.83]}

    w0, w1, wg = get_weights(coefs, config, years)

    lumi = torch.tensor(config['lumi'], dtype=torch.float64)[years.long()]
    for w, c in [(w0, config['c0']), (w1, c1), (wg, CG)]:
        assert w.dtype == torch.float32
        torch.testing.assert_close(w.double(), lumi * (coefs @ get_lower_tri(c, dtype=torch.float64)), rtol=1e-6, atol=0)

    with pytest.raises(KeyError):
        get_weights(coefs, {k: v for k, v in config.items() if k != 'lumi'}, years)
