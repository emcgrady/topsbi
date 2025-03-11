from torch.utils import data
from os.path import isfile
from pickle import load

import torch
import yaml

class DataLoader(data.Dataset):
    def __init__(self, config, dataType='training', redoTensors=False):
        #pull in formations from the config file
        self.signalSample            = config['signalSample']
        self.backgroundSample        = config['backgroundSample']
        self.signalTrainingPoint     = config['signalTrainingPoint']
        self.backgroundTrainingPoint = config['backgroundTrainingPoint']
        self.device                  = config['device']
        self.name                    = config['name']
        if not config['redoTensors']:
            self.redoTensors = not isfile(f'{self.name}/data.pkl')
        else: 
            self.redoTensors = config['redoTensors']
        self.buildTensors()
        self.loadTensors()
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        return self.features[index], self.weights[index], self.targets[index], self.fitCoeffs[index]

    def buildTensors(self):
        if self.redoTensors:
            print('Building tensors...')
            if (self.signalSample == self.backgroundSample):
                self.backgroundStart, self.signalStart = torch.utils.data.random_split(torch.load(f"{self.signalSample}/fit_coefs.p"), 
                                                                                       [0.5, 0.5], 
                                                                                       generator=torch.Generator().manual_seed(42))
                self.backgroundFeatures, self.signalFeatures = torch.utils.data.random_split(torch.load(f"{self.signalSample}/features.p"), 
                                                                                             [0.5, 0.5],
                                                                                             generator=torch.Generator().manual_seed(42))
                self.backgroundStart    = self.backgroundStart[:].to(device=self.device)
                self.signalStart        = self.signalStart[:].to(device=self.device)
                self.backgroundFeatures = self.backgroundFeatures[:].to(device=self.device)
                self.signalFeatures     = self.signalFeatures[:].to(device=self.device)
            else: 
                self.backgroundStart = torch.load(f"{self.backgroundSample}/fit_coefs.p")
                self.signalStart     = torch.load(f"{self.signalSample}/fit_coefs.p")
                cap = self.backgroundStart.shape[0]
                if self.signalStart.shape[0] < cap:
                    cap = self.signalStart.shape[0]
                self.backgroundStart    = self.backgroundStart[:cap].to(device=self.device)
                self.signalStart        = self.signalStart[:cap].to(device=self.device)
                self.backgroundFeatures = torch.load(f"{self.backgroundSample}/features.p")[:cap].to(device=self.device)
                self.signalFeatures     = torch.load(f"{self.signalSample}/features.p")[:cap].to(device=self.device)

            index = 0
            self.signalWeight = torch.zeros(self.signalStart.shape[0])
            for i in range(len(self.signalTrainingPoint)):
                for j in range(i+1):
                    if index == 0:
                        self.signalWeight += self.signalStart[:,index]*self.signalTrainingPoint[i]
                    else:
                        self.signalWeight += self.signalStart[:,index]*self.signalTrainingPoint[i]*self.signalTrainingPoint[j]
                    index += 1
    
            index = 0
            self.backgroundWeight = torch.zeros(self.backgroundStart.shape[0])

            for i in range(len(self.backgroundTrainingPoint)):
                for j in range(i+1):
                    if index == 0:
                        self.backgroundWeight += self.backgroundStart[:,index]*self.backgroundTrainingPoint[i]
                    else:
                        self.backgroundWeight += self.backgroundStart[:,index]*self.backgroundTrainingPoint[i]*self.backgroundTrainingPoint[j]
                    index += 1
        else:
            print('Loading tensors...')
            with open(f'{self.name}/data.pkl', 'rb') as f:
                data = load(f)
            backgroundMask = data[:][2] == 0
            signalMask     = data[:][2] == 1
            self.backgroundFeatures = data[:][0][backgroundMask, :] * data.stdvs + data.means
            self.signalFeatures     = data[:][0][signalMask, :] * data.stdvs + data.means
            self.backgroundWeight   = data[:][1][backgroundMask]
            self.signalWeight       = data[:][1][signalMask]
            self.backgroundStart    = data[:][3][backgroundMask]
            self.signalStart        = data[:][3][signalMask]
 
    def loadTensors(self):
        self.features  = torch.cat([self.backgroundFeatures, self.signalFeatures])
        self.means     = self.features.mean(axis=0, keepdim=True)
        self.stdvs     = self.features.std(axis=0, keepdim=True)
        self.features -= torch.Tensor(self.means)
        self.features /= torch.Tensor(self.stdvs)
        self.weights   = torch.cat([self.backgroundWeight, self.signalWeight])
        self.targets   = torch.cat([torch.tensor([0], dtype=torch.float64, device=self.device).repeat(self.signalFeatures.size(0)), 
                                    torch.tensor([1], dtype=torch.float64, device=self.device).repeat(self.backgroundFeatures.size(0))])
        self.fitCoeffs = torch.cat([self.backgroundStart, self.signalStart])