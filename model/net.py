from torch import float64, nn
from yaml import safe_load

cost =  nn.BCELoss(reduction='mean')

def createModel(nFeatures, config):
    with open(config, 'r') as f:
        config =safe_load(f)
    layers = []
    for i, layer in enumerate(config['model']['layers']):
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
        cost.to(device)
        
    def loss(self, features, weights, truth):
        cost.weight = weights
        return cost(self.net(features).squeeze(), truth)