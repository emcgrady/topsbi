from topsbi.model.net import Net
from topsbi.tools.data import get_lower_tri, load_zero_terms, zero_term_mask

import concurrent.futures, multiprocessing, torch, tqdm, yaml

torch.set_num_threads(1)


class NetworkRatio:
    """One trained network as a function x -> r_hat(x; theta_1), theta_1 = its config['c1']."""

    def __init__(self, config, n_features, device='cpu'):
        """
        Load a trained network from its training config, on the device it is evaluated on (the training config's
        device is ignored).

        Args:
            config: path to yaml file used to configure the network training
            n_features: number of features used in training
            device: torch device to evaluate the network on
        """
        with open(config) as f:
            self.config = yaml.safe_load(f)
        if 'network' in self.config.keys():
            network = self.config['network']
        else:
            network = None
        with open(f'{self.config["name"]}/complete/performance.yml') as f:
            self.performance = yaml.safe_load(f)
        self.device = device
        self.model = Net(n_features, device, network)
        self.model.load_state_dict(torch.load(f'{self.config["name"]}/model.pt', map_location=torch.device(device)))
        self.model.eval()  # dropout off at inference

    def __call__(self, features: torch.tensor, network=None):
        """
        Learned ratio of a series of events.

        Args:
            features: non-normalized features x to be evaluated by the network
        Returns:
            r_hat: r_hat(x; theta_1) per event, on the CPU
        """
        with torch.no_grad():
            r_hat = self.model.ratio(features.to(self.device))
        return r_hat.cpu()


class RatioFit:
    """
    One seed's ensemble: the networks trained at points theta_1^(n) (and the SM network) give the matrix of ratios R,
    one row r_hat(x; theta_1^(n)) per network, and Theta, one row T(theta_1^(n))[mask] per network. The per-event
    quadratic coefficients Gamma solve Theta Gamma = R in least squares, so r_hat(x; theta) = T(theta)[mask] Gamma.
    """

    def __init__(
        self,
        config: dict,
        features: torch.tensor,
    ):
        """
        Fit Gamma for one seed.

        Args:
            config: dictionary with 'sm' (SM network config), 'networks' (network configs) and 'features'
            features: non-normalized features x to be evaluated by the networks
        Attributes:
            Theta: [n_networks, K] T(theta_1^(n)) of every network, SM last, kept columns only
            R: [n_networks, N] r_hat of every network for every event
            mask: kept columns of T(theta): not zero for every network and not a listed zero term
            finite: [N] events where every network's r_hat is finite
            Gamma: [K, N_finite] least-squares coefficients; residuals, rank, singular_values from lstsq
        """
        self.config = config
        self.Theta = []
        # filled row by row: no long-lived per-network tensors between the evaluation temporaries, so the heap does
        # not fragment (fitprobe, ensemble_validation diary)
        self.R = torch.empty((len(self.config['networks']) + 1, features.shape[0]))
        if 'sm' in config.keys():
            sm_network = NetworkRatio(self.config['sm'], len(self.config['features']), self.config['device'])
            self.wcs = sm_network.config['wcs']
            self.R[-1] = sm_network(features)

        zero_terms = load_zero_terms(sm_network.config)

        def same(terms):
            return {tuple(sorted(p)) for p in terms} == {tuple(sorted(p)) for p in zero_terms}

        for i, network_yaml in tqdm.tqdm(enumerate(self.config["networks"]), total=len(self.config['networks'])):
            network = NetworkRatio(network_yaml, len(self.config['features']), self.config['device'])
            if network.config['wcs'] != self.wcs or not same(load_zero_terms(network.config)):
                raise ValueError(f'{network_yaml} uses different wcs or zero_terms than the SM network')
            self.Theta += [get_lower_tri(network.config['c1'])]
            self.R[i] = network(features)
        self.Theta += [get_lower_tri([1] + [0] * len(self.wcs))]
        self.Theta = torch.vstack(self.Theta)
        self.mask = ~(self.Theta == 0).all(dim=0) & zero_term_mask(self.wcs, zero_terms)
        self.Theta = self.Theta[:, self.mask]
        self.finite = ~torch.isinf(self.R).any(0)
        Gamma, self.residuals, self.rank, self.singular_values = torch.linalg.lstsq(
            self.Theta, self.R[:, self.finite], driver='gelsd'
        )
        # lstsq returns the solution in an [n_networks, N_finite] buffer; keep only Gamma's rows
        self.Gamma = Gamma[: self.Theta.shape[1]].clone()

    def __call__(self, theta):
        """
        Learned ratio at a WC point: r_hat(x; theta) = T(theta)[mask] Gamma, for the finite events.

        Args:
            theta: SM-inclusive WC point to evaluate the ensemble at
        Returns:
            r_hat(x; theta) per finite event
        """
        return get_lower_tri(theta)[self.mask] @ self.Gamma


def seed_map(function, seeds, args, workers=1):
    """
    function(seed, *args) for every seed, in this process (workers 1) or in up to `workers` single-threaded spawn
    processes (fresh interpreters: fork is unsafe with torch / OpenMP threads; each imports this module, which sets
    one thread). Results come back in seed order, so whatever the caller sums adds in the same order either way.

    Returns:
        {seed: function(seed, *args)}
    """
    if workers <= 1:
        return {seed: function(seed, *args) for seed in seeds}
    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(min(workers, len(seeds)), mp_context=context) as pool:
        futures = {seed: pool.submit(function, seed, *args) for seed in seeds}
        return {seed: future.result() for seed, future in futures.items()}


def fit_seed(seed, fit_config, seeds, features, per_seed=None):
    """
    One seed's RatioFit for Ensemble (through seed_map): the per-seed callback runs here (SeedCollector writes the
    seed's files), and only what Ensemble averages is returned, so the fit and its R are freed with the call.

    Args:
        seed: the seed to fit
        fit_config: settings shared by every seed (device, c0, features)
        seeds: config['seeds'], the sm and dedicated network configs of every seed
    Returns:
        mask, finite, Gamma and rank of the seed's fit
    """
    fit = RatioFit({**fit_config, 'sm': seeds[seed]['sm'], 'networks': seeds[seed]['networks']}, features)
    if per_seed is not None:
        per_seed(seed, fit)
    return fit.mask, fit.finite, fit.Gamma, int(fit.rank)


class Ensemble:
    """Gamma averaged over seeds: one RatioFit per seed, mean Gamma on the events every seed keeps."""

    def __init__(
        self,
        config: dict,
        features: torch.tensor,
        per_seed=None,
        workers=1,
    ):
        """
        Average the per-event quadratic coefficients Gamma of several seeds.

        Args:
            config: dictionary with the shared settings and, per seed, the sm and dedicated network configs
            features: non-normalized features x to be evaluated by the networks
            per_seed: optional callable per_seed(seed, fit) given each seed's RatioFit before it is discarded (with
                workers > 1 it runs in the worker, so it must be picklable and keep no state it needs later)
            workers: seeds fitted at once, one single-threaded process each (1: one after another in this process)
        Attributes:
            mask: kept columns of T(theta), the same for every seed
            finite: [N] events kept by every seed
            Gamma, Gamma2: [K, N_finite] seed mean of Gamma and of Gamma^2
            rank: {seed: rank of the seed's Theta}
        """
        fit_config = {
            'device': config['device'],
            'c0': config['c0'],
            'features': config['features'],
        }
        n_events = int(features.shape[0])
        self.mask = None
        self.Gamma = None
        self.Gamma2 = None
        self.rank = {}
        # events kept by every seed; an event dropped by any seed has no complete average
        self.finite = torch.ones(n_events, dtype=torch.bool)
        seeds = list(config['seeds'])
        fits = seed_map(fit_seed, seeds, (fit_config, config['seeds'], features, per_seed), workers)
        for seed in seeds:
            mask, finite, Gamma, rank = fits.pop(seed)
            if self.mask is None:
                self.mask = mask
                self.Gamma = torch.zeros((int(self.mask.sum()), n_events), dtype=Gamma.dtype)
                self.Gamma2 = torch.zeros((int(self.mask.sum()), n_events), dtype=Gamma.dtype)
            elif not torch.equal(self.mask, mask):
                raise ValueError(f'seed {seed} was trained at a different set of WC points than the other seeds')
            # place each seed's coefficients at their own event indices
            self.Gamma[:, finite] += Gamma
            self.Gamma2[:, finite] += Gamma**2
            self.finite &= finite
            self.rank[seed] = rank
        self.Gamma = self.Gamma[:, self.finite] / len(config['seeds'])
        self.Gamma2 = self.Gamma2[:, self.finite] / len(config['seeds'])

    def __call__(self, theta: list):
        """r_hat(x; theta) = T(theta)[mask] Gamma for the events kept by every seed."""
        return get_lower_tri(theta)[self.mask] @ self.Gamma


def get_np_parameterization(features, config, up_training, down_training):
    """
    Mean of the up and down networks' outputs f_hat per event (placeholder; the nuisance parameterisation in the
    likelihood is still to be designed).
    """
    up_model = Net(features.shape[1], config['device'], config['network'])
    down_model = Net(features.shape[1], config['device'], config['network'])
    up_model.load_state_dict(torch.load(up_training, map_location=torch.device(config['device'])))
    down_model.load_state_dict(torch.load(down_training, map_location=torch.device(config['device'])))
    up_model.eval()
    down_model.eval()
    with torch.no_grad():
        f_hat_up = up_model(features)
        f_hat_down = down_model(features)
    return ((f_hat_up + f_hat_down) / 2).numpy().flatten()
