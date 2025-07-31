from model.net import Net
from tools.data import expandArray
from torch import device, load, tensor, vstack
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
        s   = self.model((features - features.mean(0))/features.std(0))
        lr  = (s/(1-s)).flatten()
        lr *= self.config['sig2bkg']
        return lr
            
class fullLikelihood: 
    def __init__(self, config, features):
        self.config = config
        trainingMatrix = []
        ratios = []
        for i, network in enumerate(self.config['networks']):
            network = likelihood(network, len(self.config['features']))
            trainingMatrix += [expandArray(network.config['signalTrainingPoint'])]
            ratios += [network(features)]
        self.wcs = network.config['wcs']
        trainingMatrix = vstack(trainingMatrix)
        ratios = vstack(ratios)
        features = None
        self.alphas, _, _, _ = lstsq(trainingMatrix, ratios, rcond=-1)
    def __call__(self, coefs):
        return (expandArray(coefs)@self.alphas)