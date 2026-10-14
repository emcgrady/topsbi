import pytest
import torch

import topsbi.tools.buildLikelihood as build
from topsbi.tools.data import get_lower_tri

N_EVENTS = 6
# two WCs: get_lower_tri([1, a, b]) has 6 terms, 3 of which are kept
ZEROS_MASK = torch.tensor([True, True, False, True, False, False])
SEEDS = {
    # seed: (events kept by full_likelihood, gammas for the kept events)
    's1': (torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool), torch.arange(18.0).reshape(3, 6)),
    's2': (torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool), 10 + torch.arange(15.0).reshape(3, 5)),
}


class FakeFullLikelihood:
    """Stands in for full_likelihood with known per-event coefficients."""

    zeros_masks = {}

    def __init__(self, config, features):
        self.infFilter, self.gammas = SEEDS[config['sm']]
        self.zerosMask = self.zeros_masks.get(config['sm'], ZEROS_MASK)
        self.residuals = torch.rand(int(self.infFilter.sum()))


@pytest.fixture
def ensemble_config(monkeypatch):
    monkeypatch.setattr(build, 'full_likelihood', FakeFullLikelihood)
    FakeFullLikelihood.zeros_masks = {}
    return {
        'device': 'cpu',
        'c0': [1.0, 0.0, 0.0],
        'features': {},
        'seeds': {seed: {'sm': seed, 'networks': []} for seed in SEEDS},
    }


def full_size(seed):
    keep, gammas = SEEDS[seed]
    out = torch.zeros(3, N_EVENTS)
    out[:, keep] = gammas
    return out


def test_ensemble_averages_aligned_events(ensemble_config):
    ens = build.ensemble(ensemble_config, torch.zeros(N_EVENTS, 2))

    kept_by_all = SEEDS['s1'][0] & SEEDS['s2'][0]
    expected = ((full_size('s1') + full_size('s2')) / 2)[:, kept_by_all]
    assert torch.equal(ens.infFilter, kept_by_all)
    assert torch.allclose(ens.gammas, expected)
    expected2 = ((full_size('s1') ** 2 + full_size('s2') ** 2) / 2)[:, kept_by_all]
    assert torch.allclose(ens.gammas2, expected2)

    wcs = [1.0, 2.0, 3.0]
    assert torch.allclose(ens(wcs), get_lower_tri(wcs)[ZEROS_MASK] @ expected)


def test_ensemble_residual_histograms(ensemble_config):
    ens = build.ensemble(ensemble_config, torch.zeros(N_EVENTS, 2), hist_residuals=True)
    assert [int(h[0].sum()) for h in ens.residuals] == [int(SEEDS[s][0].sum()) for s in SEEDS]


def test_ensemble_rejects_seeds_at_different_points(ensemble_config):
    other = ZEROS_MASK.clone()
    other[2] = True
    FakeFullLikelihood.zeros_masks = {'s2': other}
    with pytest.raises(ValueError):
        build.ensemble(ensemble_config, torch.zeros(N_EVENTS, 2))
