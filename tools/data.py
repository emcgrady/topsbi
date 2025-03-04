from torch.utils import data
from os.path import isfile

import torch

class DataLoader(data.Dataset):
    def __init__(self, config, dataType='training', redoTensors=False):
        #pull in formations from the config file
        self.signalSample = config['signalSample']
        self.backgroundSample = config['backgroundSample']
        self.signalTrainingPoint = config['signalTrainingPoint']
        self.backgroundTrainingPoint = config['backgroundTrainingPoint']
        self.signalTrainingTerm = config['signalTrainingTerm']
        self.device = config['device']       
        self.redoTensors = config['redoTensors']
        self.signalWCList = ['sm'] + config['signalStartingPoint'].replace(":", "_").replace("=", "_").split("_")[::2]
        self.backgroundWCList = ['sm'] + config['backgroundStartingPoint'].replace(":", "_").replace("=", "_").split("_")[::2]
        #build and load tensors
        self.buildMapping()
        if dataType == 'training':
            self.buildTensors()
            self.backgroundWeight = f"{self.backgroundSample}/{self.backgroundTrainingPoint.replace('=','_').replace(':','_')}.p"
            if self.signalTrainingTerm:
                self.signalWeight = f"{self.signalSample}/{self.signalTrainingTerm}.p"
            else: 
                self.signalWeight = f"{self.signalSample}/{self.signalTrainingPoint.replace('=','_').replace(':','_')}.p"
        elif dataType == 'HistEFT':
            self.backgroundWeight = f"{self.backgroundSample}/fit_coefs.p"
            self.signalWeight = f"{self.signalSample}/fit_coefs.p"
        self.loadTensors()
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        return self.features[index], self.weights[index], self.targets[index]
    
    def buildMapping(self):
        '''
        Builds a mapping between the WCs and the index of the coefficient in the tensor
        '''
        self.signalCoefMap = {}
        self.backgroundCoefMap = {}
        index = 0
        if self.signalSample == self.backgroundSample:
            for i in range(len(self.signalWCList)):
                for j in range(i+1):
                    self.signalCoefMap[(self.signalWCList[i],self.signalWCList[j])]=index
                    self.backgroundCoefMap[(self.signalWCList[i],self.signalWCList[j])]=index
                    index+=1
        else: 
            for i in range(len(self.signalWCList)):
                for j in range(i+1):
                    self.signalCoefMap[(self.signalWCList[i],self.signalWCList[j])]=index
                    index+=1
            index = 0
            for i in range(len(self.backgroundWCList)):
                for j in range(i+1):
                    self.backgroundCoefMap[(self.backgroundWCList[i],self.backgroundWCList[j])]=index
                    index+=1

    def buildTensors(self):
        if self.redoTensors:
            makeSignalFile = True
            makeBackgroundFile = True
        elif self.signalTrainingTerm: 
            makeSignalFile = not isfile(f"{self.signalSample}/{self.signalTrainingTerm}.p")
        else: 
            makeSignalFile = not isfile(f"{self.signalSample}/{self.signalTrainingPoint.replace('=','_').replace(':','_')}.p")
            makeBackgroundFile = not isfile(f"{self.backgroundSample}/{self.backgroundTrainingPoint.replace('=','_').replace(':','_')}.p")
        if not makeSignalFile and not makeBackgroundFile:
            return
        print("Building tensors...")

        if (self.signalSample == self.backgroundSample):
            backgroundStart, signalStart = torch.utils.data.random_split(torch.load(f"{self.signalSample}/fit_coefs.p"), 
                                                                         [0.5, 0.5], generator=torch.Generator().manual_seed(42))
            
            signalStart = signalStart[:].to(device=self.device); backgroundStart = backgroundStart[:].to(device=self.device)
        else: 
            signalStart     = torch.load(f"{self.signalSample}/fit_coefs.p")
            backgroundStart = torch.load(f"{self.backgroundSample}/fit_coefs.p")

        if makeSignalFile:
            if not self.signalTrainingTerm and not self.signalTrainingPoint:
                raise Exception('signalTrainingTerm nor signalTrainingPoint!\nPlease define one of these quantities!')
            elif self.signalTrainingTerm:
                signalWeight = signalStart[:,self.signalCoefMap[(self.signalTrainingTerm,self.signalTrainingTerm)]]
                torch.save(signalWeight, f"{self.signalSample}/{self.signalTrainingTerm}.p")
            elif self.signalTrainingPoint:
                signalWeight = signalStart[:,0]
                for i, rwgtPoint in enumerate(self.signalTrainingPoint.split(':')):
                    coef, value = rwgtPoint.split('=')
                    signalWeight += signalStart[:,self.signalCoefMap[(coef,'sm')]]*float(value)
                    signalWeight += signalStart[:,self.signalCoefMap[(coef,coef)]]*(float(value)**2)
                    torch.save(signalWeight, f"{self.signalSample}/{self.signalTrainingPoint.replace('=','_').replace(':','_')}.p")
        if makeBackgroundFile:
            backgroundWeight = backgroundStart[:,0]
            for i, rwgtPoint in enumerate(self.backgroundTrainingPoint.split(':')):
                coef, value = rwgtPoint.split('=')
                backgroundWeight += backgroundStart[:,self.backgroundCoefMap[(coef,'sm')]]*float(value)
                backgroundWeight += backgroundStart[:,self.backgroundCoefMap[(coef,coef)]]*(float(value)**2)
            torch.save(backgroundWeight, f"{self.backgroundSample}/{self.backgroundTrainingPoint.replace('=','_').replace(':','_')}.p")
 
    def loadTensors(self):
        backgroundWeight = torch.load(self.backgroundWeight).to(device=self.device)
        signalWeight     = torch.load(self.signalWeight).to(device=self.device)

        if self.signalSample == self.backgroundSample:
            backgroundFeatures, signalFeatures = torch.utils.data.random_split(torch.load(f"{self.signalSample}/features.p"), 
                                                                               [0.5, 0.5],  generator=torch.Generator().manual_seed(42))
            backgroundFeatures = backgroundFeatures[:].to(device=self.device)
            signalFeatures     = signalFeatures[:].to(device=self.device)
        else: 
            cap = backgroundWeight.size()[0]
            if signalWeight.size()[0] < 0:
                cap = signalWeight.size()[0]

            backgroundWeight = backgroundWeight[:cap]
            signalWeight     = signalWeight[:cap]
            
            backgroundFeatures = torch.load(f"{self.backgroundSample}/features.p").to(device=self.device)[:cap,:]
            signalFeatures     = torch.load(f"{self.signalSample}/features.p").to(device=self.device)[:cap,:]
        self.features  = torch.cat([signalFeatures, backgroundFeatures])
        self.means     = self.features.mean(axis=0, keepdim=True)
        self.stdvs     = self.features.std(axis=0, keepdim=True)
        self.features -= torch.Tensor(self.means)
        self.features /= torch.Tensor(self.stdvs)
        self.weights   = torch.cat([signalWeight, backgroundWeight])
        self.targets   = torch.cat([torch.tensor([1], dtype=torch.float64, device=self.device).repeat(signalFeatures.size(0)), 
                                    torch.tensor([0], dtype=torch.float64, device=self.device).repeat(backgroundFeatures.size(0))])

    def getFeatures(self):
        return self.features * self.stdvs + self.means