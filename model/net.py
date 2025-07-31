from torch import cat, float64, nn, ones, zeros
from yaml import safe_load

cost =  nn.BCELoss(reduction='mean')

def createModel(nFeatures, config):
    layers = []
    for i, layer in enumerate(config):
        layerType = layer['type']
        if i == 0:
            if layerType == 'Linear':
                layers.append(nn.Linear(nFeatures, layer['out']))
        else:
            if layerType == 'Linear':
                layers.append(nn.Linear(layer['in'], layer['out']))
        if layer['activation'] == 'LeakyReLU':
            layers.append(nn.LeakyReLU())
        elif layer['activation'] == 'Sigmoid':
            layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)

class Net(nn.Module):
    def __init__(self, nFeatures, device, config):
        super().__init__()
        if config:
            self.main_module = createModel(nFeatures, config)
        else:
            self.main_module = nn.Sequential(
                nn.Linear(nFeatures, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 16),
                nn.LeakyReLU(),
                nn.Linear(16,8),
                nn.LeakyReLU(),
                nn.Linear(8,1),
                nn.Sigmoid(),
            )
        self.main_module.type(float64)
        self.main_module.to(device)
    def forward(self, x):
        return self.main_module(x)
    
class Model:
    def __init__(self, nFeatures, device, config):
        '''
        features: inputs used to train the neural network
        device: device used to train the neural network
        '''
        self.net  = Net(nFeatures, device, config)
        self.device = device
        cost.to(device)
        
    def loss(self, features, backgroundWeights, signalWeights):
        truth       = cat([zeros(backgroundWeights.shape[0], device=self.device), 
                           ones(signalWeights.shape[0], device=self.device)]).to(float64)
        features    = cat([features, features])
        cost.weight = cat([backgroundWeights, signalWeights])
        return cost(self.net(features).squeeze(), truth)