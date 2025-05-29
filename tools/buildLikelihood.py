from model.net import Net
from yaml import safe_load
from torch import device, load
from numpy import array, concatenate, float32
from numpy.linalg import lstsq
import torch

def expandArray(coefs):
    arrayOut = []
    for i in range(len(coefs)):
         for j in range(i+1):
             arrayOut += [coefs[i]*coefs[j]]
    return array(arrayOut).astype(float32)
    
class likelihood:
    def __init__(self, config, nFeatures, network=None):
        with open(config) as f:
            self.config = safe_load(f)
        self.model = Net(nFeatures, self.config['device'], network)
        self.model.load_state_dict(load(f'{self.config["name"]}/complete/networkStateDict.p', map_location=device(self.config['device'])))
    def __call__(self, data, coefs):
        features, fitCoefs, truth  = data[:] 
        weights = 0
        for i in range(len(coefs)):
            for j in range(i+1):
                weights += coefs[i]*coefs[j]*fitCoefs[:,i+j]
        s  = self.model(features.to(torch.float32))
        lr = (s/(1-s)).flatten()
        lr[truth == 0] *= weights[truth == 0].mean()
        lr[truth == 1] *= weights[truth == 1].mean()
        return lr.detach().numpy()
        
class fullLikelihood: 
    def __init__(self, config, data):
        self.config = config
        trainingMatrix = []
        ratios = []
        for i, network in enumerate(self.config['networks']):
            network = likelihood(f'{network}/training.yml', len(self.config['features']))
            self.wcs = network.config['wcs']
            trainingMatrix += [[expandArray(network.config['signalTrainingPoint'])]]
            ratios += [[network(data, network.config['signalTrainingPoint'])]]
        trainingMatrix = concatenate(trainingMatrix)
        ratios = concatenate(ratios).T
        self.fitCoefMean = data[:][1].detach().numpy().mean(0)
        data = None
        normalizations = (self.fitCoefMean@trainingMatrix.T)
        self.alphas, _, _, _ = lstsq(trainingMatrix,(ratios*normalizations).T, rcond=-1)
    def __call__(self, coefs):
        return (expandArray(coefs)@self.alphas)/(self.fitCoefMean@expandArray(coefs).T)      