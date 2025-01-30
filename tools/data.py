from torch.utils import data
from os.path import isfile

import torch

class DataLoader(data.Dataset):
    def __init__(self, config):
        self.config = config
        self.buildMapping()
        self.buildTensors()
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
        for i in range(len(self.config['signalWCList'])):
            for j in range(i+1):
                self.signalCoefMap[(self.config['signalWCList'][i],self.config['signalWCList'][j])]=index
                index+=1
        index = 0
        for i in range(len(self.config['backgroundWCList'])):
            for j in range(i+1):
                self.backgroundCoefMap[(self.config['backgroundWCList'][i],self.config['backgroundWCList'][j])]=index
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
        backgroundWeight   = torch.load(f"{self.config['backgroundSample']}/{self.config['backgroundTrainingPoint'].replace('=','_').replace(':','_')}.p").to(device=self.config['device'])
        if self.config['signalTrainingTerm']:
            signalWeight = torch.load(f"{self.config['signalSample']}/{self.config['signalTrainingTerm']}.p")
        else: 
            signalWeight = torch.load(f"{self.config['signalSample']}/{self.config['signalTrainingPoint'].replace('=','_').replace(':','_')}.p").to(device=self.config['device'])
            
        backgroundFeatures = torch.load(f"{self.config['backgroundSample']}/features.p").to(device=self.config['device'])
        signalFeatures     = torch.load(f"{self.config['signalSample']}/features.p").to(device=self.config['device'])
        self.features      = torch.cat([signalFeatures, backgroundFeatures])
        self.weights       = torch.cat([signalWeight, backgroundWeight])
        self.targets       = torch.cat([torch.tensor([0], dtype=torch.float64, device=self.config['device']).repeat(backgroundFeatures.size(0)), 
                                        torch.tensor([1], dtype=torch.float64, device=self.config['device']).repeat(signalFeatures.size(0))])