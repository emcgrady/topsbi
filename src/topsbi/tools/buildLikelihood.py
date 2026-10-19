from topsbi.model.net import Net
from topsbi.tools.data import get_lower_tri, load_zero_terms, zero_term_mask

import torch, tqdm, yaml

torch.set_num_threads(1)


class likelihood:
    def __init__(self, config, nFeatures):
        """
        Load a trained network and prepare for conversion to the likelihood ratio.

        Args:
            config: path to yaml file used to configure the network training
            nFeatures: number of features used in training
        """
        with open(config) as f:
            self.config = yaml.safe_load(f)
        if 'network' in self.config.keys():
            network = self.config['network']
        else:
            network = None
        with open(f'{self.config["name"]}/complete/performance.yml') as f:
            self.performance = yaml.safe_load(f)
        self.model = Net(nFeatures, self.config['device'], network)
        self.model.load_state_dict(
            torch.load(f'{self.config["name"]}/model.pt', map_location=torch.device(self.config['device']))
        )
        self.model.eval()  # dropout off at inference

    def __call__(self, features: torch.tensor, network=None):
        """
        Convert the network output of a series of events to a likelihood ratio.

        Args:
            features: non-normalized feateures to be evaluated by the network
        Returns:
            lr: evaluated likelihood ratio
        """
        with torch.no_grad():
            s = self.model(features)
        lr = (s / (1 - s)).flatten()
        return lr


class full_likelihood:
    def __init__(
        self,
        config: dict,
        features: torch.tensor,
    ):
        """
        Prepare an ensemble of network to be used to find the likelihood ratio
        at an arbitrary point in WC space.

        Args:
            config: dictionary containing the networks and parameters for network ensemble
            features: non-normalized feateures to be evaluated by the networks
        """
        self.config = config
        self.trainingMatrix = []
        self.ratios = []
        if 'sm' in config.keys():
            sm_network = likelihood(self.config['sm'], len(self.config['features']))
            self.wcs = sm_network.config['wcs']
            sm_ratio = sm_network(features)

        zero_terms = load_zero_terms(sm_network.config)

        def same(terms):
            return {tuple(sorted(p)) for p in terms} == {tuple(sorted(p)) for p in zero_terms}

        for i, network_yaml in tqdm.tqdm(enumerate(self.config["networks"]), total=len(self.config['networks'])):
            network = likelihood(network_yaml, len(self.config["features"]))
            if network.config['wcs'] != self.wcs or not same(load_zero_terms(network.config)):
                raise ValueError(f'{network_yaml} uses different wcs or zero_terms than the SM network')
            self.trainingMatrix += [get_lower_tri(network.config['c1'])]
            self.ratios += [network(features)]
        self.trainingMatrix += [get_lower_tri([1] + [0] * len(self.wcs))]
        self.ratios += [sm_ratio]
        self.trainingMatrix = torch.vstack(self.trainingMatrix)
        self.zerosMask = ~(self.trainingMatrix == 0).all(dim=0) & zero_term_mask(self.wcs, zero_terms)
        self.trainingMatrix = self.trainingMatrix[:, self.zerosMask]
        self.ratios = torch.vstack(self.ratios)
        self.infFilter = ~torch.isinf(self.ratios).any(0)
        self.gammas, self.residuals, self.rank, self.singular_values = torch.linalg.lstsq(
            self.trainingMatrix, self.ratios[:, self.infFilter], driver='gelsd'
        )

    def __call__(self, coefs):
        """
        Evaluates the ensembled likelihood ratio for a given point in WC space.

        Args:
            coefs: SM-inclusive set of WCs to evalueate the ensemble at
        Returns:
            evaluated likelihood ratio
        """
        return get_lower_tri(coefs)[self.zerosMask] @ self.gammas


class ensemble:
    def __init__(
        self,
        config: dict,
        features: torch.tensor,
        per_seed=None,
    ):
        """
        Average the per-event morphing coefficients of several seeds.

        Args:
            config: dictionary with the shared settings and, per seed, the sm and dedicated network configs
            features: non-normalized features to be evaluated by the networks
            per_seed: optional callable per_seed(seed, fit) given each seed's full_likelihood before it is discarded
        """
        likelihood_config = {
            'device': config['device'],
            'c0': config['c0'],
            'features': config['features'],
        }
        n_events = int(features.shape[0])
        self.zeros_mask = None
        self.gammas = None
        self.gammas2 = None
        # events kept by every seed; an event dropped by any seed has no complete average
        self.infFilter = torch.ones(n_events, dtype=torch.bool)
        for seed in config['seeds'].keys():
            likelihood_config['sm'] = config['seeds'][seed]['sm']
            likelihood_config['networks'] = config['seeds'][seed]['networks']
            temp_likelihood = full_likelihood(likelihood_config, features)
            if per_seed is not None:
                per_seed(seed, temp_likelihood)
            if self.zeros_mask is None:
                self.zeros_mask = temp_likelihood.zerosMask
                self.gammas = torch.zeros((int(self.zeros_mask.sum()), n_events), dtype=temp_likelihood.gammas.dtype)
                self.gammas2 = torch.zeros((int(self.zeros_mask.sum()), n_events), dtype=temp_likelihood.gammas.dtype)
            elif not torch.equal(self.zeros_mask, temp_likelihood.zerosMask):
                raise ValueError(f'seed {seed} was trained at a different set of WC points than the other seeds')
            # place each seed's coefficients at their own event indices
            self.gammas[:, temp_likelihood.infFilter] += temp_likelihood.gammas
            self.gammas2[:, temp_likelihood.infFilter] += temp_likelihood.gammas**2
            self.infFilter &= temp_likelihood.infFilter
        self.gammas = self.gammas[:, self.infFilter] / len(config['seeds'])
        self.gammas2 = self.gammas2[:, self.infFilter] / len(config['seeds'])

    def __call__(self, coefs: list):
        return get_lower_tri(coefs)[self.zeros_mask] @ self.gammas


def get_np_parameterization(features, config, up_training, down_training):
    up_model = Net(features.shape[1], config['device'], config['network'])
    down_model = Net(features.shape[1], config['device'], config['network'])
    up_model.load_state_dict(torch.load(up_training, map_location=torch.device(config['device'])))
    down_model.load_state_dict(torch.load(down_training, map_location=torch.device(config['device'])))
    up_model.eval()
    down_model.eval()
    with torch.no_grad():
        up_out = up_model(features)
        down_out = down_model(features)
    return ((up_out + down_out) / 2).numpy().flatten()
