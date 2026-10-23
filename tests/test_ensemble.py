import pytest
import torch

import topsbi.tools.build_ratio as build
from topsbi.tools.data import get_lower_tri

N_EVENTS = 6
# two WCs: get_lower_tri([1, a, b]) has 6 terms, 3 of which are kept
MASK = torch.tensor([True, True, False, True, False, False])
SEEDS = {
    # seed: (events kept by RatioFit, Gamma for the kept events)
    's1': (torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.bool), torch.arange(18.0).reshape(3, 6)),
    's2': (torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool), 10 + torch.arange(15.0).reshape(3, 5)),
}


class FakeRatioFit:
    """Stands in for RatioFit with known per-event coefficients Gamma."""

    masks = {}

    def __init__(self, config, features):
        self.finite, self.Gamma = SEEDS[config['sm']]
        self.mask = self.masks.get(config['sm'], MASK)
        self.residuals = torch.rand(int(self.finite.sum()))
        self.rank = int(self.mask.sum())  # full column rank, as lstsq reports for a well-posed fit


@pytest.fixture
def ensemble_config(monkeypatch):
    monkeypatch.setattr(build, 'RatioFit', FakeRatioFit)
    FakeRatioFit.masks = {}
    return {
        'device': 'cpu',
        'c0': [1.0, 0.0, 0.0],
        'features': {},
        'seeds': {seed: {'sm': seed, 'networks': []} for seed in SEEDS},
    }


def full_size(seed):
    keep, Gamma = SEEDS[seed]
    out = torch.zeros(3, N_EVENTS)
    out[:, keep] = Gamma
    return out


def test_ensemble_averages_aligned_events(ensemble_config):
    ens = build.Ensemble(ensemble_config, torch.zeros(N_EVENTS, 2))

    kept_by_all = SEEDS['s1'][0] & SEEDS['s2'][0]
    expected = ((full_size('s1') + full_size('s2')) / 2)[:, kept_by_all]
    assert torch.equal(ens.finite, kept_by_all)
    assert torch.allclose(ens.Gamma, expected)
    expected2 = ((full_size('s1') ** 2 + full_size('s2') ** 2) / 2)[:, kept_by_all]
    assert torch.allclose(ens.Gamma2, expected2)

    wcs = [1.0, 2.0, 3.0]
    assert torch.allclose(ens(wcs), get_lower_tri(wcs)[MASK] @ expected)
    assert ens.rank == {seed: int(MASK.sum()) for seed in SEEDS}


def test_ensemble_passes_each_seed_fit_to_per_seed(ensemble_config):
    residuals = {}
    build.Ensemble(
        ensemble_config, torch.zeros(N_EVENTS, 2), per_seed=lambda seed, fit: residuals.update({seed: fit.residuals})
    )
    assert [len(residuals[s]) for s in SEEDS] == [int(SEEDS[s][0].sum()) for s in SEEDS]


def test_ensemble_rejects_seeds_at_different_points(ensemble_config):
    other = MASK.clone()
    other[2] = True
    FakeRatioFit.masks = {'s2': other}
    with pytest.raises(ValueError):
        build.Ensemble(ensemble_config, torch.zeros(N_EVENTS, 2))


def test_is_cross_term():
    from topsbi.tools.ensembleCache import is_cross_term

    assert is_cross_term('ctGRe_cQd1/p_one_p_point_five')
    assert not is_cross_term('ctGRe/p_one') and not is_cross_term('sm')
