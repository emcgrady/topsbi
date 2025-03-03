from torch.utils import data
from os.path import isfile

import torch

class DataLoader(data.Dataset):
    def __init__(self, config, dataType='training'):
        self.config = config
        self.signalWCList = ['sm'] + self.config['signalStartingPoint'].replace(":", "_").replace("=", "_").split("_")[::2]
        self.backgroundWCList = ['sm'] + self.config['backgroundStartingPoint'].replace(":", "_").replace("=", "_").split("_")[::2]
        self.buildMapping()
        if dataType == 'training':
            self.buildTensors()
            self.backgroundWeight = f"{self.config['backgroundSample']}/{self.config['backgroundTrainingPoint'].replace('=','_').replace(':','_')}.p"
            if self.config['signalTrainingTerm']:
                self.signalWeight = f"{self.config['signalSample']}/{self.config['signalTrainingTerm']}.p"
            else: 
                self.signalWeight = f"{self.config['signalSample']}/{self.config['signalTrainingPoint'].replace('=','_').replace(':','_')}.p"
        elif dataType == 'HistEFT':
            self.backgroundWeight = f"{self.config['backgroundSample']}/fit_coefs.p"
            self.signalWeight = f"{self.config['signalSample']}/fit_coefs.p"
        self.loadTensors()
    
    def __len__( self ):
        return self.features.shape[0]
    
    def __getitem__( self, index ):
        return self.features[index], self.weights[index], self.targets[index]
    
    def buildMapping( self ):
        '''
        Builds a mapping between the WCs and the index of the coefficient in the tensor
        '''
        self.signalCoefMap = {}
        self.backgroundCoefMap = {}
        index = 0
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
        if self.config['signalTrainingTerm']: 
             makeSignalFile = not isfile(f"{self.config['signalSample']}/{self.config['signalTrainingTerm']}.p")
        else: 
            makeSignalFile = not isfile(f"{self.config['signalSample']}/{self.config['signalTrainingPoint'].replace('=','_').replace(':','_')}.p")
        makeBackgroundFile = not isfile(f"{self.config['backgroundSample']}/{self.config['backgroundTrainingPoint'].replace('=','_').replace(':','_')}.p")

        if not makeSignalFile and not makeBackgroundFile:
            return
        
        print("Building tensors...")
        if makeSignalFile: 
            signalStart = torch.load(f"{self.config['signalSample']}/fit_coefs.p")
            if self.config['signalTrainingTerm']:
                signalWeight = signalStart[:,self.signalCoefMap[(self.config['signalTrainingTerm'],self.config['signalTrainingTerm'])]]
                torch.save(signalWeight, f"{self.config['signalSample']}/{self.config['signalTrainingTerm']}.p")
            else: 
                signalStart = torch.load(f"{self.config['signalSample']}/fit_coefs.p")
                signalWeight = signalStart[:,0]
                for i, rwgtPoint in enumerate(self.config['signalTrainingPoint'].split(':')):
                    coef, value = rwgtPoint.split('=')
                    signalWeight += signalStart[:,self.signalCoefMap[(coef,'sm')]]*float(value)
                    signalWeight += signalStart[:,self.signalCoefMap[(coef,coef)]]*(float(value)**2)
                torch.save(signalWeight, f"{self.config['signalSample']}/{self.config['signalTrainingPoint'].replace('=','_').replace(':','_')}.p")

        if makeBackgroundFile: 
            signalStart = torch.load(f"{self.config['backgroundSample']}/fit_coefs.p")
            signalWeight = signalStart[:,0]
            for i, rwgtPoint in enumerate(self.config['backgroundTrainingPoint'].split(':')):
                coef, value = rwgtPoint.split('=')
                signalWeight += signalStart[:,self.backgroundCoefMap[(coef,'sm')]]*float(value)
                signalWeight += signalStart[:,self.backgroundCoefMap[(coef,coef)]]*(float(value)**2)
            torch.save(signalWeight, f"{self.config['backgroundSample']}/{self.config['backgroundTrainingPoint'].replace('=','_').replace(':','_')}.p")

    def loadTensors(self):
        backgroundWeight = torch.load(self.backgroundWeight).to(device=self.config['device'])
        signalWeight     = torch.load(self.signalWeight).to(device=self.config['device'])

        cap = backgroundWeight.size()[0]
        if signalWeight.size()[0] < 0:
            cap = signalWeight.size()[0]

        backgroundWeight = backgroundWeight[:cap]
        signalWeight     = signalWeight[:cap]
            
        backgroundFeatures = torch.load(f"{self.config['backgroundSample']}/features.p").to(device=self.config['device'])[:cap,:]
        signalFeatures     = torch.load(f"{self.config['signalSample']}/features.p").to(device=self.config['device'])[:cap,:]
        self.features      = torch.cat([signalFeatures, backgroundFeatures])
        self.features     -= torch.Tensor(self.config['featureMeans'])
        self.features     /= torch.Tensor(self.config['featureStdvs'])
        self.weights       = torch.cat([signalWeight, backgroundWeight])
        self.targets       = torch.cat([torch.tensor([1], dtype=torch.float64, device=self.config['device']).repeat(backgroundFeatures.size(0)), 
                                        torch.tensor([0], dtype=torch.float64, device=self.config['device']).repeat(signalFeatures.size(0))])

    def getFeatures(self):
        return self.features * torch.Tensor(self.config['featureStdvs']) + torch.Tensor(self.config['featureMeans'])