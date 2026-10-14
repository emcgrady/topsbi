import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from topsbi.tools.metrics import group_closure
from topsbi.tools.plots import lr_calibration_curve, lr_calibration_difference, merge_sparse_bins

N_EVENTS = 200_000
BIAS = 0.05


@pytest.fixture
def synthetic():
    """
    A network output with known calibration.

    true_ratio plays the calibrated network output E[r | x]; the per-event truth r scatters around it
    with mean 1 (gamma), and the weights w0 are spread like the stitched weights.
    """
    rng = np.random.default_rng(0)
    true_ratio = rng.lognormal(-0.5, 0.4, N_EVENTS)
    lr = true_ratio * rng.gamma(4, 0.25, N_EVENTS)
    w0 = rng.lognormal(0, 0.4, N_EVENTS)
    groups = np.column_stack([rng.integers(0, 4, N_EVENTS), rng.integers(2, 5, N_EVENTS)])
    return true_ratio, lr, w0 / w0.mean(), groups


def calibration_difference(lrhat, lr, w0):
    fig, ax = plt.subplots()
    out = lr_calibration_difference(ax, lr, lrhat, w0, 50, threshold=0.0)
    plt.close(fig)
    return out


def calibration_curve(lrhat, lr, w0):
    fig, ax = plt.subplots()
    out = lr_calibration_curve(ax, lr, lrhat, w0, 50, threshold=0.0)
    plt.close(fig)
    return out


def test_pull_and_ci_diff_zero_when_calibrated(synthetic):
    true_ratio, lr, w0, _ = synthetic
    pull, pull_err, ci_diff, ci_diff_err = calibration_difference(true_ratio, lr, w0)
    assert abs(pull) < 3 * pull_err
    for value, err in zip(ci_diff, ci_diff_err):
        assert abs(value) < 3 * err + 1e-3


def test_pull_and_ci_diff_positive_when_biased(synthetic):
    true_ratio, lr, w0, _ = synthetic
    pull, pull_err, ci_diff, ci_diff_err = calibration_difference(true_ratio * (1 + BIAS), lr, w0)
    assert pull > 10 * pull_err
    assert ci_diff[0] > 3 * ci_diff_err[0]


def test_rms_bias_measures_bias_size(synthetic):
    true_ratio, lr, w0, _ = synthetic
    assert calibration_curve(true_ratio, lr, w0) < 0.01
    assert calibration_curve(true_ratio * (1 + BIAS), lr, w0) == pytest.approx(BIAS, abs=0.01)


def test_group_closure_calibrated(synthetic):
    true_ratio, lr, w0, groups = synthetic
    closure = group_closure(true_ratio, lr, w0, groups)
    assert closure['n_cells'] == 4 * 3 * 5
    assert abs(closure['chi2_excess']) < 4 * closure['chi2_excess_err']
    assert closure['rms_bias'] < 0.01


def test_group_closure_detects_regional_bias(synthetic):
    """A bias confined to one group cancels nowhere in lrhat bins but must show up in its group cells."""
    true_ratio, lr, w0, groups = synthetic
    lrhat = true_ratio * np.where(groups[:, 0] == 2, 1 + BIAS, 1.0)
    closure = group_closure(lrhat, lr, w0, groups)
    assert closure['chi2_excess'] > 10 * closure['chi2_excess_err']
    assert closure['worst_pull'] > 3


def test_group_closure_without_populated_cells_returns_nan(synthetic):
    true_ratio, lr, w0, groups = synthetic
    closure = group_closure(true_ratio[:100], lr[:100], w0[:100], groups[:100])
    assert closure['n_cells'] == 0
    assert np.isnan(closure['chi2_excess'])
    assert np.isnan(closure['rms_bias'])


def test_merge_sparse_bins():
    rng = np.random.default_rng(1)
    x = rng.exponential(100, 50_000)
    w = rng.lognormal(0, 1, 50_000)
    fine = np.linspace(0, 1000, 51)
    merged = merge_sparse_bins(x, w, fine, min_neff=30)

    assert merged[0] == fine[0] and merged[-1] == fine[-1]
    assert np.all(np.isin(merged, fine))
    assert np.all(np.diff(merged) > 0)
    sumw, _ = np.histogram(x, bins=merged, weights=w)
    sumw2, _ = np.histogram(x, bins=merged, weights=w**2)
    assert np.all(sumw**2 / sumw2 >= 30)
    # the populated core keeps the fine binning, the sparse tail is merged
    assert np.allclose(merged[:6], fine[:6])
    assert len(merged) < len(fine)
