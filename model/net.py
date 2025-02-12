from torch import float64
import torch.nn as nn

cost =  nn.BCELoss(reduction='mean')

class Net(nn.Module):
    def __init__(self, features, device):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self.main_module.type(float64)
        self.main_module.to(device)
    def forward(self, x):
        return self.main_module(x)
    
class Model:
    def __init__(self, features, device):
        '''
        features: inputs used to train the neural network
        device: device used to train the neural network
        '''
        self.net  = Net(features, device=device)
        cost.to(device)
        
    def loss(self, features, weights, targets):
        cost.weight = weights
        return cost(self.net(features).squeeze(), targets)