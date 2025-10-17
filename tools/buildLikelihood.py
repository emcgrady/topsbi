from model.net import Net
from tools.data import expandArray
from torch import device, load, no_grad, tensor, vstack
from torch.linalg import lstsq
from yaml import safe_load

class likelihood:
    def __init__(self, config, nFeatures, network=None):
        with open(config) as f:
            self.config = safe_load(f)
        self.model = Net(nFeatures, self.config['device'], network)
        self.model.load_state_dict(load(f'{self.config["name"]}/complete/networkStateDict.p', 
                                                map_location=device(self.config['device'])))   
    def __call__(self, features, network=None):
        with no_grad():
            s   = self.model((features - features.mean(0))/features.std(0))
        lr  = (s/(1-s)).flatten()
        lr *= self.config['sig2bkg']
        return lr
            
class fullLikelihood: 
    def __init__(self, config, features):
        self.config = config
        self.trainingMatrix = []
        self.ratios = []
        for i, yaml in enumerate(self.config['networks']):
            network = likelihood(yaml, len(self.config['features']))
            if i==0:
                self.wcs = network.config['wcs']
            self.trainingMatrix += [expandArray(network.config['signalTrainingPoint'])]
            self.ratios += [network(features)]
        self.trainingMatrix = vstack(self.trainingMatrix)
        if 'toSkip' in self.config.keys():
            self.trainingMatrix[:, self.config['toSkip']] = 0
        self.zerosMask = ~(self.trainingMatrix ==0).all(dim=0)
        self.trainingMatrix = self.trainingMatrix[:,self.zerosMask]
        self.ratios = vstack(self.ratios)
        self.alphas, self.residuals, self.rank, self.singular_values = lstsq(self.trainingMatrix, self.ratios, rcond=-1, driver='gelsy')
    def __call__(self, coefs):
        return expandArray(coefs)[self.zerosMask]@self.alphas