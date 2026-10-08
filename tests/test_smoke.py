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
