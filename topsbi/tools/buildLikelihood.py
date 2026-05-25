from topsbi.model.net import Net
from topsbi.tools.data import expand_array, prepare_features

import torch, tqdm, yaml

sm = [1, 0.,  0.,  0.,  0.,  0.,  0.,   0.,   0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
b
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
        self.model = Net(nFeatures, self.config['device'], network)
        self.model.load_state_dict(torch.load(f'{self.config["name"]}/complete/networkStateDict.p', 
                                              map_location=torch.device(self.config['device'])))   
    def __call__(self, 
                 features: torch.tensor, 
                 network=None):
        """
        Convert the network output of a series of events to a likelihood ratio. 
        
        Args:
            features: non-normalized feateures to be evaluated by the network 
        Returns:
            lr: evaluated likelihood ratio
        """
        with torch.no_grad():
            s   = self.model(prepare_features(features))
        lr  = (s/(1-s)).flatten()
        return lr

class fullLikelihood: 
    def __init__(
        self, 
        config: dict, 
        features: torch.tensor
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
        for i, yaml in tqdm.tqdm(enumerate(self.config['networks']), total=len(self.config['networks'])):
            network = likelihood(yaml, len(self.config['features']))
            if i==0:
                self.wcs = network.config['wcs']
            self.trainingMatrix += [expand_array(network.config['c1'])]
            if network.config['c1'] == network.config['c0']:
                self.ratios += [torch.ones(features.shape[0])]
            else:
                self.ratios += [network(features)]
        self.trainingMatrix = vstack(self.trainingMatrix)
        self.zerosMask = ~(self.trainingMatrix == 0).all(dim=0)
        self.trainingMatrix = self.trainingMatrix[:,self.zerosMask]
        self.ratios = torch.vstack(self.ratios)
        self.infFilter = ~torch.isinf(self.ratios).any(0)
        self.alphas, self.residuals, self.rank, self.singular_values = torch.linalg.lstsq(self.trainingMatrix, self.ratios[:,self.infFilter], rcond=-1)
    def __call__(self, coefs): 
        """
        Evaluates the ensembled likelihood ratio for a given point in WC space. 

        Args: 
            coefs: SM-inclusive set of WCs to evalueate the ensemble at
        Returns:
            evaluated likelihood ratio
        """
        return expand_array(coefs)[self.zerosMask]@self.alphas