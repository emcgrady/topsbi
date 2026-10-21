import torch

ACTIVATIONS = {'LeakyReLU': torch.nn.LeakyReLU, 'Sigmoid': torch.nn.Sigmoid, 'None': None}

# architecture of the current reco_regrouped results, used when the config gives no network
DEFAULT_NETWORK = [
    {'type': 'Linear', 'out': 128, 'activation': 'LeakyReLU'},
    {'type': 'Linear', 'in': 128, 'out': 128, 'activation': 'LeakyReLU'},
    {'type': 'Linear', 'in': 128, 'out': 1, 'activation': 'Sigmoid'},
]

# EFT methods, trained with binary cross entropy; their networks output the score s_hat and need a Sigmoid last layer
EFT_METHODS = ('eft',)
# methods trained with the soft+ loss; their networks output f_hat = log r_hat and need a linear last layer
SOFT_PLUS_METHODS = ('weight_shift', 'kinematic_shift')


def create_model(n_features, config):
    """
    Build a network from a list of layer dictionaries.

    Args:
        n_features: number of inputs for the network
        config: list of {'type': 'Linear', 'out': int, 'activation': name, optional 'in', optional 'dropout'}
    Returns:
        torch network
    """
    layers = []
    n_in = n_features
    for i, layer in enumerate(config):
        if layer['type'] != 'Linear':
            raise ValueError(f"layer {i}: unknown type {layer['type']!r}")
        if layer.get('in', n_in) != n_in:
            raise ValueError(f"layer {i}: 'in' is {layer['in']} but the previous layer gives {n_in}")
        activation = str(layer['activation'])  # YAML null and 'None' both mean a linear output
        if activation not in ACTIVATIONS:
            raise ValueError(f"layer {i}: unknown activation {activation!r}, expected one of {list(ACTIVATIONS)}")
        if 'dropout' in layer:
            layers.append(torch.nn.Dropout(layer['dropout']))
        layers.append(torch.nn.Linear(n_in, layer['out']))
        if ACTIVATIONS[activation] is not None:
            layers.append(ACTIVATIONS[activation]())
        n_in = layer['out']
    return torch.nn.Sequential(*layers)


class Net(torch.nn.Module):
    """
    The network f_phi(x), with the input standardisation inside it (registered buffers saved in model.pt), so callers
    pass raw features x. Its output is the learned score s_hat(x; theta) (Sigmoid output, binary cross entropy) or
    f_hat(x) = log r_hat(x; theta) (linear output, soft+ loss).
    """

    def __init__(self, n_features, device, config):
        """
        Build DNN.

        Without a config, the network is built from DEFAULT_NETWORK (n_features x 128 x 128 x 1).
        The last activation sets the output: 'Sigmoid' gives the score s_hat = r_hat / (1 + r_hat), 'None' (linear)
        gives f_hat = log r_hat of the soft+ loss.
        Args:
            n_features: number of inputs for network
            device: torch device used network
            config: list of layer dictionaries used to construct the network (see create_model)
        """
        super().__init__()
        layers = config or DEFAULT_NETWORK
        last = str(layers[-1]['activation'])
        if last not in ('Sigmoid', 'None'):
            raise ValueError(f"last activation must be 'Sigmoid' (probability) or 'None' (log ratio), got {last!r}")
        self.output = 'probability' if last == 'Sigmoid' else 'log_ratio'
        self.main_module = create_model(n_features, layers)
        self.device = device
        self.main_module.type(torch.float32)
        # training-set feature mean/std, set by set_feature_normalization and saved in the state dict
        self.register_buffer('feature_mean', torch.zeros(n_features))
        self.register_buffer('feature_std', torch.ones(n_features))
        self.to(device)

    def set_feature_normalization(self, mean, std):
        """
        Set the statistics used to standardise inputs in forward.

        Args:
            mean: per-feature mean of the training set
            std: per-feature standard deviation of the training set
        """
        self.feature_mean.copy_(mean).to(self.device)
        self.feature_std.copy_(std).to(self.device)

    def forward(self, x):
        """Raw network output for raw features x: s_hat(x; theta) or f_hat(x), shape [N, 1]."""
        return self.main_module((x - self.feature_mean) / self.feature_std)

    def log_ratio(self, x):
        """
        Learned log ratio log r_hat(x; theta) per event, shape [N]: f_hat for log-ratio networks, logit(s_hat) for
        probability networks.

        Args:
            x: raw (unnormalised) features
        """
        out = self(x).squeeze(-1)
        return out if self.output == 'log_ratio' else torch.logit(out)

    def ratio(self, x):
        """
        Learned ratio r_hat(x; theta) per event, shape [N]: s_hat / (1 - s_hat) for probability networks (inf where
        s_hat = 1), e^f_hat for log-ratio networks.

        Args:
            x: raw (unnormalised) features
        """
        out = self(x).squeeze(-1)
        return out / (1 - out) if self.output == 'probability' else torch.exp(out)


class Model:
    """A Net and the loss of its training method (binary cross entropy, or soft+ for the nuisance methods)."""

    def __init__(self, n_features, method, device, config, seed):
        """
        Args:
            n_features: number of network inputs
            method: training method (config['method']); SOFT_PLUS_METHODS need a linear output
            device: device used to train the neural network
            config: list of layer dictionaries (create_model), or None for DEFAULT_NETWORK
            seed: torch seed for the initial weights
        """
        if method not in EFT_METHODS + SOFT_PLUS_METHODS:
            raise ValueError(f'unknown method {method!r}; expected one of {EFT_METHODS + SOFT_PLUS_METHODS}')
        torch.manual_seed(seed)
        if not config and method in SOFT_PLUS_METHODS:
            # default architecture with a linear output for the soft+ loss
            config = [dict(layer) for layer in DEFAULT_NETWORK]
            config[-1]['activation'] = 'None'
        self.net = Net(n_features, device, config)
        expected = 'log_ratio' if method in SOFT_PLUS_METHODS else 'probability'
        if self.net.output != expected:
            last = 'None' if expected == 'log_ratio' else 'Sigmoid'
            raise ValueError(f"method {method!r} needs a {expected} network: set the last activation to {last!r}")
        self.device = device
        self.method = method
        if self.method in SOFT_PLUS_METHODS:
            # soft+ cross entropy: w0 log(1 + e^f_hat) + w1 log(1 + e^-f_hat), minimised at e^f_hat = w1 / w0
            self.cost = torch.nn.BCEWithLogitsLoss()
        else:
            self.cost = torch.nn.BCELoss()
        self.cost.to(self.device)

    def loss(self, features, w0, w1):
        """
        Weighted cross entropy of a batch for the paired methods (every method except kinematic_shift).

        Each event enters twice: with weight w0 and label 0, and with weight w1 and label 1.
        Args:
            features: inputs used to train the neural network
            w0: w^s(theta_0), or w^s(theta, 0) for weight_shift
            w1: w^s(theta_1), or w^s(theta, nu) for weight_shift
        Returns:
            weighted loss
        """
        if self.method == 'kinematic_shift':
            raise ValueError('kinematic_shift compares different events: use labelled_loss(features, weights, labels)')
        labels = torch.cat([torch.zeros_like(w0), torch.ones_like(w1)])
        return self.labelled_loss(torch.cat([features, features]), torch.cat([w0, w1]), labels)

    def labelled_loss(self, features, weights, labels):
        """
        Weighted cross entropy of labelled rows: one row per event, label 0 for w^s(theta, 0) rows (nominal),
        1 for w^s(theta, nu) rows (shifted).

        Used directly by kinematic_shift, where the nominal and shifted samples are different events.
        Args:
            features: inputs used to train the neural network
            weights: weight of every row
            labels: 0 or 1 for every row
        Returns:
            weighted loss
        """
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError('labels must be 0 (nominal) or 1 (shifted)')
        self.cost.weight = weights
        return self.cost(self.net(features).squeeze(-1), labels)
