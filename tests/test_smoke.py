import importlib

import numpy as np
import pytest
import torch
import yaml

from topsbi.tools.data import get_lower_tri

MODULES = [
    'topsbi.model.net',
    'topsbi.tools.build_ratio',
    'topsbi.tools.data',
    'topsbi.tools.metrics',
    'topsbi.tools.plots',
    'topsbi.train',
    'topsbi.validation',
]


@pytest.mark.parametrize('module', MODULES)
def test_import(module):
    importlib.import_module(module)


def test_get_lower_tri_matches_processor_ordering():
    """get_lower_tri must use the same (i, j<=i) ordering as expand_array in ttbarEFT's tensor_processor."""
    wcs = [1.0, 2.0, 3.0, 5.0]
    expected = [wcs[i] * wcs[j] for i in range(len(wcs)) for j in range(i + 1)]
    assert get_lower_tri(wcs, dtype=torch.float64).tolist() == expected


def test_eft_training(train_config, tmp_path):
    from topsbi.train import main

    main(train_config)

    out = tmp_path / 'out'
    assert (out / 'model.pt').exists()
    performance = yaml.safe_load((out / 'complete' / 'performance.yml').read_text())
    assert 0.0 <= performance['auc'] <= 1.0
    for key in ['r_pull', 'r_pull_err', 'r_ci_diff', 'r_ci_diff_err', 'rate_closure', 'rate_pull', 'r_chi2_ndf', 'r_z',
                'r_rms_bias', 'group_chi2_ndf', 'group_z', 'group_rms_bias', 'group_worst_pull', 'feature_chi2_ndf',
                'feature_rms_bias', 'nominal_r_chi2_ndf', 'best_epoch']:
        assert key in performance
    assert len(performance['r_ci_diff']) == 3
    for plot in ['group_closure.png', 'r_closure.png', 'r_spread.png', 'kinematics/feat0.png']:
        assert (out / 'complete' / plot).exists()
    assert (out / 'complete' / 'animations' / 'feat0.gif').exists()


def test_holdout_training(holdout_config, tmp_path, capsys):
    """With holdout set, the final metrics are computed on it rather than on the early-stopping test split."""
    from topsbi.train import main

    main(holdout_config)

    assert '[INFO] final metrics on holdout' in capsys.readouterr().out
    performance = yaml.safe_load((tmp_path / 'out' / 'complete' / 'performance.yml').read_text())

    # recompute on the holdout directly; the metrics do not depend on the overall weight scale
    from topsbi.model.net import Net
    from topsbi.tools.data import prepare_weights
    from topsbi.tools.plots import network_plots

    from conftest import N_FEATURES

    net = Net(N_FEATURES, 'cpu', holdout_config['network'])
    net.load_state_dict(torch.load(tmp_path / 'out' / 'model.pt'))
    features, w0, w1 = prepare_weights(torch.load(holdout_config['holdout'], weights_only=False), holdout_config)
    network_plots(features, w0, w1, net, [1.0], [1.0], str(tmp_path / 'direct'))
    direct = yaml.safe_load((tmp_path / 'direct' / 'performance.yml').read_text())
    for key in ['auc', 'r_pull', 'r_rms_bias', 'r_chi2_ndf']:
        assert performance[key] == pytest.approx(direct[key], abs=1e-4)


def test_get_weights_stitched():
    """eft weights are L_y * S @ T(c): the per-sample normalization in S is kept, not divided out by cg."""
    from conftest import N_WCS
    from topsbi.tools.data import get_weights

    gen = torch.Generator().manual_seed(1)
    coefs = torch.rand(50, (N_WCS + 1) * (N_WCS + 2) // 2, generator=gen, dtype=torch.float64) * 1e-8
    years = torch.randint(0, 4, (50,), generator=gen).float()
    c1 = [1.0] + [0.0] * N_WCS
    c1[4] = 0.1
    config = {'c0': [1.0] + [0.0] * N_WCS, 'c1': c1, 'lumi': [19.52, 16.81, 41.48, 59.83], 'device': 'cpu'}

    w0, w1 = get_weights(coefs, config, years)

    lumi = torch.tensor(config['lumi'], dtype=torch.float64)[years.long()]
    for w, c in [(w0, config['c0']), (w1, c1)]:
        assert w.dtype == torch.float32
        torch.testing.assert_close(w.double(), lumi * (coefs @ get_lower_tri(c, dtype=torch.float64)) * 1000, rtol=1e-6, atol=0)

    with pytest.raises(KeyError):
        get_weights(coefs, {k: v for k, v in config.items() if k != 'lumi'}, years)


SHARED_KEYS = ['auc', 'rate_closure', 'rate_pull', 'r_chi2_ndf', 'r_z', 'r_rms_bias', 'r_pull', 'r_ci_diff',
               'nominal_r_pull', 'group_chi2_ndf', 'feature_chi2_ndf', 'feature_rms_bias', 'nominal_r_chi2_ndf',
               'best_epoch', 'best_test_loss']


def check_nuisance_outputs(out, paired):
    assert (out / 'model.pt').exists()
    performance = yaml.safe_load((out / 'complete' / 'performance.yml').read_text())
    for key in SHARED_KEYS:
        assert key in performance
    assert (out / 'complete' / 'r_spread.png').exists() == paired  # needs the per-event true ratio
    for plot in ['r_closure.png', 'group_closure.png', 'kinematics/feat0.png', 'animations/feat0.gif', 'loss.png']:
        assert (out / 'complete' / plot).exists()
    # the epoch-0 summary has no loss or learning-rate plot
    assert not any((out / 'incomplete' / 'epoch_0000' / plot).exists() for plot in ('loss.png', 'learning_rate.png'))
    assert (out / 'incomplete' / 'epoch_0000' / 'r_closure.png').exists()
    losses = np.load(out / 'complete' / 'loss.npz')
    assert len(losses['train']) == len(losses['test']) == 3  # before training and after each of the 2 epochs
    # frames after 0, 1 and 2 completed epochs, 1 s each, then the best network held 3 s (merged by Pillow into one)
    from PIL import Image

    gif = Image.open(out / 'complete' / 'animations' / 'feat0.gif')
    durations = []
    for i in range(gif.n_frames):
        gif.seek(i)
        durations.append(gif.info['duration'])
    assert gif.n_frames == 4 and durations[0] == 1000 and durations[-1] == 3000
    return performance


def test_weight_shift_training(nuisance_config, tmp_path):
    from topsbi.train import main

    main({**nuisance_config, 'method': 'weight_shift', 'systematic': 'ISR'})
    check_nuisance_outputs(tmp_path / 'nuisance', paired=True)


def test_kinematic_shift_training(nuisance_config, tmp_path):
    from topsbi.train import main

    main({**nuisance_config, 'method': 'kinematic_shift', 'systematic': 'JES_X'})
    performance = check_nuisance_outputs(tmp_path / 'nuisance', paired=False)
    assert len(performance['feature_rms_bias']) == len(performance['feature_chi2_ndf'])


def test_kinematic_shift_keeps_variation_eras(nuisance_config):
    """A variation produced for one era only is trained against the nominal events of that era."""
    from topsbi.tools.data import load_nuisance_data

    config = {**nuisance_config, 'method': 'kinematic_shift', 'systematic': 'JER_2017'}
    for split in load_nuisance_data(config).values():
        assert torch.all(split['nominal'][0][:, 21] == 2)
        assert torch.all(split['shifted'][0][:, 21] == 2)

    # validation.p with other eras than train.p is a broken sample
    path = f"{config['data']}/JER_2017Up/validation.p"
    features, w_ref, ids, event_id = torch.load(path, weights_only=False)[:]
    features = features.clone()
    features[:, 21] = 1
    torch.save(torch.utils.data.TensorDataset(features, w_ref, ids, event_id), path)
    with pytest.raises(ValueError, match='eras'):
        load_nuisance_data(config)


@pytest.mark.parametrize('method', ['eft', 'weight_shift', 'kinematic_shift'])
def test_closure_inputs(method, train_config, nuisance_config):
    """closure_inputs gives closure_metrics' sample arguments for both data kinds"""
    import numpy as np

    from topsbi.model.net import Model
    from topsbi.tools.metrics import closure_metrics
    from topsbi.tools.data import get_feature_normalization
    from topsbi.train import closure_inputs, load_training_data, nominal_features

    if method == 'eft':
        config = train_config
    else:
        config = {**nuisance_config, 'method': method, 'systematic': 'ISR' if method == 'weight_shift' else 'JES_X'}
    kind, splits = load_training_data(config)
    net = Model(22, config['method'], 'cpu', config.get('network'), config['seed']).net
    # as in main: without the input standardisation an untrained network saturates on raw features
    net.set_feature_normalization(*get_feature_normalization(nominal_features(kind, splits['train'])))
    f0, (x0, w0, i0), f1, (x1, w1, i1) = closure_inputs(net, kind, splits['test'])
    assert len(f0) == len(x0) == len(w0) == len(i0) and len(f1) == len(x1) == len(w1) == len(i1)
    assert all(isinstance(a, np.ndarray) for a in (f0, x0, w0, i0, f1, x1, w1, i1))
    if kind == 'paired':
        x, _, _ = splits['test']
        assert len(f0) == len(x)  # untrained networks give finite f_hat everywhere
        assert np.array_equal(f0, f1) and np.array_equal(x0, x1) and np.array_equal(i0, np.arange(len(x)))
    else:
        assert kind == 'labelled' and method == 'kinematic_shift'
        assert len(f1) == len(splits['test']['shifted'][0])  # every shifted row is kept
    closure_metrics(f0, (x0, w0, i0), f1, (x1, w1, i1), {})


def test_hypothesis_and_plot_luminosity():
    import numpy as np

    from topsbi.train import hypothesis, plot_luminosity

    assert hypothesis({'method': 'eft'}) == r'\theta'
    assert hypothesis({'method': 'weight_shift'}) == hypothesis({'method': 'kinematic_shift'}) == r'\theta,\nu'
    lumi = [19.52, 16.81, 41.48, 59.83]
    x = np.zeros((8, 22))
    x[:, 21] = [0, 1, 2, 3, 3, 2, 1, 0]
    assert plot_luminosity({'lumi': lumi}, x) == 138
    x[:, 21] = 2  # a variation produced for one era
    assert plot_luminosity({'lumi': lumi}, x) == 41
    # the era column follows year_loc
    x = np.zeros((4, 22))
    x[:, 5] = [0, 3, 3, 0]
    assert plot_luminosity({'lumi': lumi, 'year_loc': 5}, x) == 79


def test_eft_combined_training(train_config, nuisance_config, tmp_path):
    """EFT on the nominal directory of a combined sample: event_split train/test, validation.p as the holdout"""
    from topsbi.tools.data import event_split, load_combined
    from topsbi.train import load_training_data, main

    nominal = f"{nuisance_config['data']}/nominal"
    config = {**train_config, 'data': nominal, 'name': str(tmp_path / 'eft_combined')}
    del config['bootstrap']
    _, splits = load_training_data(config)
    rows = load_combined(f'{nominal}/train.p', 'nominal', ('features', 'ids', 'event_id'))
    test = event_split(rows, 'nominal', config['seed'])
    assert len(splits['test'][0]) == int(test.sum()) and len(splits['train'][0]) == int((~test).sum())
    assert len(splits['holdout'][0]) == len(load_combined(f'{nominal}/validation.p', 'nominal', ('features',))[0])
    main(config)
    out = tmp_path / 'eft_combined' / 'complete'
    for plot in ['r_closure.png', 'kinematics/feat0.png', 'animations/feat0.gif', 'performance.yml']:
        assert (out / plot).exists()


def test_closure_options():
    from topsbi.train import check_config, closure_options

    lumi = [19.52, 16.81, 41.48, 59.83]
    default = closure_options({'lumi': lumi})
    assert default['group_cols'] == (21, 19) and default['group_caps'] == {19: 4}
    assert default['group_labels'][19] == ('jets', {4: '≥4'})
    # without eras, the eras are labelled by their index
    assert default['value_names'][21] == {0: '0', 1: '1', 2: '2', 3: '3'} == default['group_labels'][21][1]
    named = closure_options({'lumi': lumi, 'eras': ['2016APV', '2016', '2017', '2018']})
    assert named['value_names'][21][0] == '2016APV' == named['group_labels'][21][1][0]
    custom = closure_options(
        {'lumi': [1, 2], 'year_loc': 5, 'eras': ['A', 'B'], 'closure_groups': [{'loc': 5}, {'loc': 7, 'cap': 2}]}
    )
    assert custom['group_cols'] == (5, 7) and custom['group_caps'] == {7: 2}
    assert custom['value_names'] == {5: {0: 'A', 1: 'B'}} and custom['group_labels'][7] == ('', {2: '≥2'})
    for bad in (
        {'eras': ['A'], 'lumi': [1, 2]},
        {'closure_groups': [{'loc': 5}, {'loc': 5}]},
        {'closure_groups': [{'column': 5}]},
        {'threads': 0},
    ):
        with pytest.raises(ValueError):
            check_config({'method': 'eft', **bad})


def test_moved_era_column_and_threads(train_config, tmp_path):
    """era index in column 5 instead of 21, custom era names and groups, one thread"""
    from threadpoolctl import threadpool_info

    from topsbi.train import main

    features, coefs, ids = torch.load(f"{train_config['data']}/train.p", weights_only=False)[:]
    features = features.clone()
    features[:, [5, 21]] = features[:, [21, 5]]
    data = tmp_path / 'moved'
    data.mkdir()
    torch.save(torch.utils.data.TensorDataset(features, coefs, ids), data / 'train.p')
    config = {
        **train_config,
        'data': str(data),
        'name': str(tmp_path / 'moved_out'),
        'year_loc': 5,
        'eras': ['A', 'B', 'C', 'D'],
        'closure_groups': [{'loc': 5}],
        'threads': 1,
    }
    main(config)
    assert (tmp_path / 'moved_out' / 'complete' / 'group_closure.png').exists()
    assert torch.get_num_threads() == 1
    assert all(pool['num_threads'] == 1 for pool in threadpool_info())


def test_epoch_batches():
    from topsbi.train import epoch_batches

    n, size = 1003, 100
    x, w = torch.arange(n).float()[:, None].repeat(1, 3), torch.arange(n).float()

    def order(seed):
        batches = list(epoch_batches((x, w), size, torch.Generator().manual_seed(seed)))
        assert [len(b[0]) for b in batches] == [size] * (n // size) + [n % size]
        assert all(torch.equal(b[0][:, 0], b[1]) for b in batches)  # rows of every tensor stay aligned
        return torch.cat([b[1] for b in batches])

    first = order(0)
    assert torch.equal(first.sort().values, w)  # every row exactly once
    assert torch.equal(order(0), first) and not torch.equal(order(1), first)
