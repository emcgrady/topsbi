import numpy as np
import pytest

from topsbi.tools.metrics import closure_metrics, matched_closure, merge_sparse_bins
from topsbi.tools.plots import bin_tick_labels

N_EVENTS = 200_000
BIAS = 0.05


@pytest.fixture
def synthetic():
    """
    A network output with known calibration.

    r_hat_ideal plays the calibrated network output E[r | x]; the per-event truth r scatters around it
    with mean 1 (gamma), and the weights w0 are spread like the stitched weights.
    """
    rng = np.random.default_rng(0)
    r_hat_ideal = rng.lognormal(-0.5, 0.4, N_EVENTS)
    r = r_hat_ideal * rng.gamma(4, 0.25, N_EVENTS)
    w0 = rng.lognormal(0, 0.4, N_EVENTS)
    return r_hat_ideal, r, w0 / w0.mean()


def paired_metrics(r_hat, r, w0):
    """closure_metrics on paired data: the same events weighted by w0 and by w0 r"""
    features = np.zeros((len(w0), 22))
    events = np.arange(len(w0))
    performance, _ = closure_metrics(np.log(r_hat), (features, w0, events), np.log(r_hat), (features, w0 * r, events), {})
    return performance


def calibration_difference(r_hat, r, w0):
    p = paired_metrics(r_hat, r, w0)
    return p['r_pull'], p['r_pull_err'], p['r_ci_diff'], p['r_ci_diff_err']


def r_rms_bias(r_hat, r, w0):
    return paired_metrics(r_hat, r, w0)['r_rms_bias']


def test_pull_and_ci_diff_zero_when_calibrated(synthetic):
    r_hat_ideal, r, w0 = synthetic
    pull, pull_err, ci_diff, ci_diff_err = calibration_difference(r_hat_ideal, r, w0)
    assert abs(pull) < 3 * pull_err
    for value, err in zip(ci_diff, ci_diff_err):
        assert abs(value) < 3 * err + 1e-3


def test_pull_and_ci_diff_positive_when_biased(synthetic):
    r_hat_ideal, r, w0 = synthetic
    pull, pull_err, ci_diff, ci_diff_err = calibration_difference(r_hat_ideal * (1 + BIAS), r, w0)
    assert pull > 10 * pull_err
    assert ci_diff[0] > 3 * ci_diff_err[0]


def test_rms_bias_measures_bias_size(synthetic):
    r_hat_ideal, r, w0 = synthetic
    assert r_rms_bias(r_hat_ideal, r, w0) < 0.01
    assert r_rms_bias(r_hat_ideal * (1 + BIAS), r, w0) == pytest.approx(BIAS, abs=0.01)


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


# njets probabilities before and after the shift (5% of events gain a jet, at most 5)
P0 = np.array([0.4, 0.3, 0.2, 0.1])
P1 = 0.95 * P0 + 0.05 * np.array([0, 0.4, 0.3, 0.3])
S1 = np.sqrt(1 + 0.05**2)
TWO_SAMPLE_FEATURES = {
    'x': {'label': 'x', 'loc': 0, 'min': -3, 'max': 3, 'nbins': 30},
    'njets': {'label': 'njets', 'loc': 19, 'min': 1.5, 'max': 5.5, 'nbins': 4},
    'year': {'label': 'year', 'loc': 21, 'min': -0.5, 'max': 3.5, 'nbins': 4},
}


def log_normal_pdf(x, mean, std):
    return -0.5 * ((x - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)


def exact_log_r(x, njets):
    """log of the exact ratio of the weighted densities of the two samples built by two_samples"""
    return 0.0002 + log_normal_pdf(x, 0.1, S1) - log_normal_pdf(x, 0, 1) + np.log(P1[njets - 2] / P0[njets - 2])


@pytest.fixture
def two_samples():
    """
    Reference and target samples sharing 99% of their events, with the exact log density ratio per row.

    x ~ N(0, 1), njets in 2..5, era in 0..3; the target moves x' = x + 0.1 + 0.05 e, gives 5% of events one more jet
    (at most 5) and reweights w1 = w0 exp(0.02 e'); 0.5% of events are only in one sample.
    """
    rng = np.random.default_rng(3)
    n = 60_000
    x = rng.normal(size=n)
    njets = rng.choice([2, 3, 4, 5], size=n, p=P0)
    era = rng.integers(0, 4, n)
    x_shift = x + 0.1 + 0.05 * rng.normal(size=n)
    njets_shift = np.where(rng.random(n) < 0.05, np.minimum(njets + 1, 5), njets)
    w0 = rng.exponential(1, n)
    w1 = w0 * np.exp(0.02 * rng.normal(size=n))
    kind = rng.random(n)
    in0, in1 = kind >= 0.005, kind < 0.995

    def features(values, jets):
        f = np.zeros((n, 22))
        f[:, 0], f[:, 19], f[:, 21] = values, jets, era
        return f

    events = np.arange(n)
    f0 = exact_log_r(x, njets)[in0]
    f1 = exact_log_r(x_shift, njets_shift)[in1]
    return f0, (features(x, njets)[in0], w0[in0], events[in0]), f1, (features(x_shift, njets_shift)[in1], w1[in1], events[in1])


def test_matched_closure_sample_variances():
    """w_hat_var / w_var are each sample's own sum of squared weights per bin; rows outside every bin are dropped"""
    w_hat = np.array([1.0, 2.0, 3.0, 4.0])
    bin0 = np.array([0, 1, 1, -1])
    w = np.array([0.5, 1.5, 2.5])
    bin1 = np.array([1, 1, 0])
    result = matched_closure(w_hat, bin0, np.arange(4), w, bin1, np.array([0, 1, 2]), 2)
    np.testing.assert_allclose(result['w_hat_var'], [1.0, 4.0 + 9.0])
    np.testing.assert_allclose(result['w_var'], [2.5**2, 0.5**2 + 1.5**2])
    np.testing.assert_allclose(result['var_independent'], result['w_hat_var'] + result['w_var'])


def test_closure_metrics_calibrated(two_samples):
    f0, nominal, f1, shifted = two_samples
    performance, _ = closure_metrics(f0, nominal, f1, shifted, TWO_SAMPLE_FEATURES)
    assert abs(performance['rate_pull']) < 4
    assert abs(performance['r_z']) < 4
    assert abs(performance['group_z']) < 4
    for name in TWO_SAMPLE_FEATURES:
        ndf = TWO_SAMPLE_FEATURES[name]['nbins']
        assert performance['feature_chi2_ndf'][name] < 1 + 4 * np.sqrt(2 / ndf)
    # without the correction the samples are clearly different, so the test can tell the two apart
    assert performance['nominal_r_chi2_ndf'] > 10
    assert performance['nominal_feature_chi2_ndf']['x'] > 10


def test_closure_metrics_detects_regional_bias(two_samples):
    """r_hat 5% too high in one era only must show up in that era's group cells"""
    f0, nominal, f1, shifted = two_samples
    bias = np.log(1.05)
    f0 = f0 + bias * (nominal[0][:, 21] == 2)
    f1 = f1 + bias * (shifted[0][:, 21] == 2)
    performance, details = closure_metrics(f0, nominal, f1, shifted, TWO_SAMPLE_FEATURES)
    assert performance['group_z'] > 10
    worst = details['group']['summary']['worst_bin']
    assert details['group']['cells'][worst][0][0] == 2
    assert details['group']['cols'] == (21, 19)  # the columns the cells were grouped by, for the plot labels


def test_closure_metrics_paired():
    """eft / weight_shift: the same events in both samples, w1 = w0 r with r scattering around r_hat"""
    rng = np.random.default_rng(4)
    n = 60_000
    x = np.zeros((n, 22))
    x[:, 0] = rng.normal(size=n)
    x[:, 19] = rng.integers(2, 6, n)
    x[:, 21] = rng.integers(0, 4, n)
    f_ideal = 0.2 * x[:, 0]
    w0 = rng.exponential(1, n)
    w1 = w0 * np.exp(f_ideal) * rng.gamma(4, 0.25, n)
    events = np.arange(n)
    features = {'x': TWO_SAMPLE_FEATURES['x']}
    calibrated, _ = closure_metrics(f_ideal, (x, w0, events), f_ideal, (x, w1, events), features)
    assert abs(calibrated['r_z']) < 4
    assert abs(calibrated['group_z']) < 4
    f_biased = f_ideal + np.log(1.05)
    biased, _ = closure_metrics(f_biased, (x, w0, events), f_biased, (x, w1, events), features)
    assert biased['rate_closure'] == pytest.approx(0.05, abs=0.01)
    assert biased['r_rms_bias'] == pytest.approx(0.05, abs=0.01)


def test_closure_metrics_constant_f_hat(two_samples):
    """a constant network output gives one bin R_b (and one per group) instead of failing"""
    _, nominal, _, shifted = two_samples
    performance, _ = closure_metrics(np.zeros(len(nominal[1])), nominal, np.zeros(len(shifted[1])), shifted, {})
    assert performance['r_ndf'] == 1
    assert 'feature_chi2_ndf' not in performance


def test_bin_tick_labels():
    njets = {'loc': 19, 'min': 1.5, 'max': 6.5, 'nbins': 5}
    positions, labels = bin_tick_labels([1.5, 2.5, 3.5, 4.5, 6.5], njets)
    assert labels == ['2', '3', '4', '≥5'] and positions == [0.5, 1.5, 2.5, 3.5]
    # merged integer bins
    assert bin_tick_labels([1.5, 3.5, 6.5], njets)[1] == ['2–3', '≥4']
    # named values (the eras) for whichever column has them
    era = {'loc': 5, 'min': -0.5, 'max': 3.5, 'nbins': 4}
    names = {5: {0: '2016APV', 1: '2016', 2: '2017', 3: '2018'}}
    assert bin_tick_labels([-0.5, 0.5, 1.5, 2.5, 3.5], era, names)[1] == ['2016APV', '2016', '2017', '2018']
    assert bin_tick_labels([-0.5, 0.5, 1.5, 2.5, 3.5], era)[1] == ['0', '1', '2', '≥3']
    # symmetric continuous feature: 7 labels with 0 at the zero crossing
    eta = {'loc': 3, 'min': -2.4, 'max': 2.4, 'nbins': 48}
    positions, labels = bin_tick_labels(np.linspace(-2.4, 2.4, 49), eta)
    assert len(labels) == 7 and labels[3] == '0' and positions[3] == 24
    # R_b edges close together around 1: labels stay distinct
    labels = bin_tick_labels(np.linspace(0.998, 1.003, 51))[1]
    assert len(labels) == len(set(labels)) == 6
