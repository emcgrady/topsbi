from pathlib import Path

import pytest
import yaml

from topsbi.tools.data import (
    _check_same_sample_setup,
    get_lower_tri,
    is_combined,
    load_zero_terms,
    syst_ratio_column,
    zero_term_mask,
)
from topsbi.train import check_config

EXAMPLES = Path(__file__).resolve().parent.parent / 'examples'
DATA = EXAMPLES / 'data'


def test_example_systematics():
    """examples/data/nominal/systematics.yml is read as the code reads a combined sample's"""
    assert syst_ratio_column(str(DATA), 'FSR', 'down') == (1, 1)
    assert syst_ratio_column(str(DATA), 'ISR', 'Up') == (0, 0)


def test_example_sample_files():
    """examples/data/*/sample.yml mark combined data and agree on reference and salt_split"""
    assert is_combined(str(DATA / 'nominal'))
    _check_same_sample_setup(str(DATA), 'JES_AbsoluteUp')  # raises if they disagree


def test_example_zero_terms():
    """examples/data/zero_terms.yml drops one get_lower_tri column per listed pair, for the WCs of the eft example"""
    wcs = yaml.safe_load((EXAMPLES / 'training' / 'eft.yml').read_text())['wcs']
    zero_terms = load_zero_terms({'zero_terms': str(DATA / 'zero_terms.yml')})
    mask = zero_term_mask(wcs, zero_terms)
    assert len(mask) == len(get_lower_tri([1.0] * (len(wcs) + 1)))
    assert int((~mask).sum()) == len(zero_terms) == 3


@pytest.mark.parametrize('name', ['eft', 'weight_shift', 'kinematic_shift'])
def test_example_training_configs(name):
    """the training examples pass check_config; their data and features exist where /groups is mounted"""
    config = yaml.safe_load((EXAMPLES / 'training' / f'{name}.yml').read_text())
    check_config(config)
    if not Path(config['data']).exists():
        pytest.skip('example data not mounted')
    assert Path(config['features']).exists()
    if 'zero_terms' in config:
        assert Path(config['zero_terms']).exists()
