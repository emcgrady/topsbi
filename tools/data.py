from os.path import isfile
from torch.utils import data
from torch import cat, float64, Generator, load, tensor, Tensor
from torch.utils.data import random_split
import pickle

class dataLoader(data.Dataset):
    def __init__(self, config, redoTensors=False):
        #pull in formations from the config file
        self.signalSample            = config['signalSample']
        self.backgroundSample        = config['backgroundSample']
        self.signalTrainingPoint     = config['signalTrainingPoint']
        self.backgroundTrainingPoint = config['backgroundTrainingPoint']
        self.device                  = config['device']
        self.name                    = config['name']
        self.data                    = config['data']
        if config['redoTensors'] or not isfile(self.data):
            self.buildTensors()
        else:
            self.loadTensors()
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        return self.features[index], self.fitCoeffs[index], self.targets[index]

    def buildTensors(self):
        print('Building tensors...')
        if (self.signalSample == self.backgroundSample):
            self.backgroundStart, self.signalStart       = random_split(load(f"{self.signalSample}/fit_coefs.p"), 
                                                                        [0.5, 0.5], 
                                                                        generator=Generator().manual_seed(42))
            self.backgroundFeatures, self.signalFeatures = random_split(load(f"{self.signalSample}/features.p"), 
                                                                        [0.5, 0.5],
                                                                        generator=Generator().manual_seed(42))
            self.backgroundCoefs    = self.backgroundStart[:].to(device=self.device)
            self.signalCoefs        = self.signalStart[:].to(device=self.device)
            self.backgroundFeatures = self.backgroundFeatures[:].to(device=self.device)
            self.signalFeatures     = self.signalFeatures[:].to(device=self.device)
        else: 
            self.backgroundCoefs = load(f"{self.backgroundSample}/fit_coefs.p")
            self.signalCoefs     = load(f"{self.signalSample}/fit_coefs.p")
            cap = self.backgroundCoefs.shape[0]
            if self.signalCoefs.shape[0] < cap:
                cap = self.signalCoefs.shape[0]
            self.backgroundCoefs    = self.backgroundCoefs[:cap].to(device=self.device)
            self.signalCoefs        = self.signalCoefs[:cap].to(device=self.device)
            self.backgroundFeatures = load(f"{self.backgroundSample}/features.p")[:cap].to(device=self.device)
            self.signalFeatures     = load(f"{self.signalSample}/features.p")[:cap].to(device=self.device)
        self.prepareTensors()
        with open(self.data, 'wb') as f:
             pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
        
    def loadTensors(self):
        print('Loading tensors...')
        with open(self.data, 'rb') as f:
            data = pickle.load(f)
        backgroundMask = data[:][2] == 0
        signalMask     = data[:][2] == 1
        self.backgroundFeatures = (data[:][0][backgroundMask, :] * data.stdvs + data.means).to(device=self.device)
        self.signalFeatures     = (data[:][0][signalMask, :] * data.stdvs + data.means).to(device=self.device)
        self.backgroundCoefs    = (data[:][1][backgroundMask]).to(device=self.device)
        self.signalCoefs        = (data[:][1][signalMask]).to(device=self.device)
        self.prepareTensors()

    def prepareTensors(self):
        self.features  = cat([self.backgroundFeatures, self.signalFeatures])
        self.means     = self.features.mean(axis=0, keepdim=True)
        self.stdvs     = self.features.std(axis=0, keepdim=True)
        self.features -= Tensor(self.means)
        self.features /= Tensor(self.stdvs)
        self.targets   = cat([tensor([0], dtype=float64, device=self.device).repeat(self.signalFeatures.size(0)), 
                                    tensor([1], dtype=float64, device=self.device).repeat(self.backgroundFeatures.size(0))])
        self.fitCoeffs = cat([self.backgroundCoefs, self.signalCoefs])