from model.net import Net
from yaml import safe_load
from torch import device, float64, load
from numpy import array, concatenate
from numpy.linalg import lstsq

def expandArray(coefs):
    arrayOut = []
    for i in range(len(coefs)):
         for j in range(i+1):
             arrayOut += [coefs[i]*coefs[j]]
    return array(arrayOut)
    
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
        s  = self.model(features)
        lr = (s/(1-s)).flatten()
        lr[truth == 0] *= weights[truth == 0].mean()
        lr[truth == 1] *= weights[truth == 1].mean()
        return lr.detach().numpy()
        
class fullLikelihood: 
    def __init__(self, config, data):
        self.config = config
        self.trainingMatrix = []
        self.ratios = []
        for i, network in enumerate(self.config['networks']):
            network = likelihood(f'{network}/training.yml', len(self.config['features']))
            self.wcs = network.config['wcs']
            self.trainingMatrix += [[expandArray(network.config['signalTrainingPoint'])]]
            self.ratios += [[network(data, network.config['signalTrainingPoint'])]]
        self.trainingMatrix = concatenate(self.trainingMatrix)
        self.ratios = concatenate(self.ratios).T
        self.fitCoefs = data[:][1].detach().numpy()
        self.normalizations = self.fitCoefs@self.trainingMatrix.T
        self.alphas, self.residuals, _, _ = lstsq(self.trainingMatrix,(self.ratios*self.normalizations).T, rcond=-1)
    def __call__(self, coefs):
        return (expandArray(coefs)@self.alphas)/(self.fitCoefs@expandArray(coefs).T)      