import torch

cost =  torch.nn.BCELoss(reduction='mean')

def createModel(nFeatures, config):
    """
    Build a network based on a given dictionary. 
    
    Args:
        nFenFeatures: numner of inputs for network 
        config: dictionary used to construct the network
    Returns:
        torch network 
    """
    layers = []
    for i, layer in enumerate(config):
        layerType = layer['type']
        if 'dropout' in layer.keys():
            layers.append(torch.nn.Dropout(layer['dropout']))
        if i == 0:
            if layerType == 'Linear':
                layers.append(torch.nn.Linear(nFeatures, layer['out']))
        else:
            if layerType == 'Linear':
                layers.append(torch.nn.Linear(layer['in'], layer['out']))
        if layer['activation'] == 'LeakyReLU':
            layers.append(torch.nn.LeakyReLU())
        elif layer['activation'] == 'Sigmoid':
            layers.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*layers)

class Net(torch.nn.Module):
    def __init__(self, nFeatures, device, config):
        """
        Build DNN. 

        By default, network will build as nFeatures x 32 x 16 x 8 x 1. 
        Args:
            nFeatures: numner of inputs for network 
            device: torch device used network 
            config: dictionary used to construct the network
        """
        super().__init__()
        if config:
            self.main_module = createModel(nFeatures, config)
        else:
            self.main_module = torch.nn.Sequential(
                torch.nn.Linear(nFeatures, 32),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(32, 16),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(16,8),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(8,1),
                torch.nn.Sigmoid(),
            )
        self.main_module.type(torch.float32)
        self.main_module.to(device)
    def forward(self, x):
        return self.main_module(x)
    
class Model:
    def __init__(self, nFeatures, method, device, config, seed):
        """
        features: inputs used to train the neural network
        device: device used to train the neural network
        """
        torch.manual_seed(seed)
        self.net  = Net(nFeatures, device, config)
        self.device = device
        self.method = method
        cost.to(device)
        
    def loss(self, features, w0, w1):
        """
        Get the weighted binary cross-entropy loss for a set of events. 
        Args:
            features: inputs used to train the neural network
            w0: weight for events under theta0
            w1: weight for events under theta1
        Returns:
            weighted loss 
        """
        if self.method == 'alice': 
            truth       = w1/(w0 + w1)
            cost.weight = 2*torch.ones(truth.shape[0], device=self.device)
        else: 
            truth       = torch.cat([torch.zeros(w0.shape[0], device=self.device), 
                                     torch.ones(w1.shape[0], device=self.device)])
            features    = torch.cat([features, features])
            cost.weight = torch.cat([w0, w1])
        netOut = self.net(features).squeeze()
        return cost(netOut, truth)