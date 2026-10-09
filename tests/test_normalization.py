import torch

from topsbi.model.net import Net
from topsbi.tools.data import sample_boostrap

NETWORK = [
    {'activation': 'LeakyReLU', 'out': 8, 'type': 'Linear'},
    {'activation': 'Sigmoid', 'in': 8, 'out': 1, 'type': 'Linear'},
]


def test_normalization_saved_and_restored(tmp_path):
    torch.manual_seed(0)
    net = Net(3, 'cpu', NETWORK)
    net.set_feature_normalization(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.5, 4.0, 10.0]))
    torch.save(net.state_dict(), tmp_path / 'model.pt')

    loaded = Net(3, 'cpu', NETWORK)
    loaded.load_state_dict(torch.load(tmp_path / 'model.pt'))
    x = torch.randn(5, 3) * 10
    assert torch.equal(loaded.feature_mean, net.feature_mean)
    assert torch.equal(loaded.feature_std, net.feature_std)
    assert torch.equal(loaded(x), net(x))


def test_output_does_not_depend_on_other_events():
    """An event's prediction must not change with the sample it is evaluated in (emcgrady/topsbi#3)."""
    torch.manual_seed(0)
    net = Net(3, 'cpu', NETWORK)
    x = torch.rand(100, 3) * 100
    net.set_feature_normalization(x.mean(0), x.std(0))
    subset = x[x[:, 0] > 50]
    with torch.no_grad():
        assert torch.allclose(net(x)[x[:, 0] > 50], net(subset))


def test_training_stores_training_set_statistics(train_config, tmp_path):
    from topsbi.train import main

    main(train_config)

    state = torch.load(tmp_path / 'out' / 'model.pt')
    sample = torch.load(tmp_path / 'data' / 'train.p', weights_only=False)
    train, _ = sample_boostrap(sample, train_config['seed'])
    train_feats = train[:][0].to(torch.float32)
    assert torch.allclose(state['feature_mean'], train_feats.mean(0))
    assert torch.allclose(state['feature_std'], train_feats.std(0))
