from model.net import Net
import torch
import yaml 

class likelihood:
    def __init__(self, config):
        with open(config) as f:
            self.config = yaml.safe_load(f)
        self.model = Net(len(self.config['features'].split(',')), self.config['device'])
        self.model.load_state_dict(torch.load(f'{self.config["name"]}/last/networkStateDict.p',
                                              map_location=torch.device(self.config['device'])))
        self.norm = self.config['backgroundMean']/self.config['signalMean']

    def __call__(self, features):
        score = self.model(features.to(torch.float64))
        return self.norm*(score/(1-score))
    
class linearTerm:
    def __init__(self, sm, linear, quad):
        self.sm          = sm
        self.linear      = linear
        self.quad        = quad
        self.linearValue = linear.config['signalTrainingPoint'].split('=')[1]

    def __call__(self, features):
        return (self.linear(features) - self.sm(features) - self.quad(features)*self.linearValue**2)/self.linearValue

class interferenceTerm:
    def __init__(self, linear, quad, interference):
        self.sm = quad['sm']; self.linear = []; self.linearValue = []; self.quad = []
        terms   = interference.config['signalTrainingPoint'].split(':')
        for term in terms: 
            wc, value = term.split['=']
            self.linearValue += [value]
            self.quad        += [quad[wc]]
            self.linear      += [linear[wc]]
        self.interference = interference
    def __call__(self, features):
        return (self.interference(features)- self.linear[0](features)*self.linearValue[0] - self.quad[0](features)*self.linearValue[0]**2\
                - self.linear[1](features)*self.linearValue[1] - self.quad[1](features)*self.linearValue[1]**2\
                - self.sm(features))/(self.linearValue[0]*self.linearValue[1])
     
class fullLikelihood: 
    def __init__(self, config):
        with open(config) as f:
            self.config = yaml.safe_load(f.read())
        self.wcs = ['sm'] + self.config['wcs'].split(",")
        self.quad = {}; self.linear = {}; self.interference={}
        self.noLin = ['ctu1', 'cqd1', 'cqq13', 'cqu1', 'cqq11', 'ctd1', 'ctq1']
        for wc in self.wcs:
            self.quad[wc] = likelihood(self.config[f'{wc}_quad'])
        for i, wc0 in enumerate(self.wcs[:-1]):
            for wc1 in self.wcs[(i+1):]:
                if wc0 == 'sm' and wc1 not in self.noLin:
                    self.linear[wc1] = linearTerm(self.quad['sm'], likelihood(self.config[f'{wc1}_linear']), self.quad[wc1])
                else:
                    self.interference[(wc0,wc1)] = interferenceTerm(self.linear, self.quad, likelihood(self.config[f'{wc_0}_{wc_1}']))
    def __call__(self, features, wcValues):
        if set(self.wcs) != set(wcValues.keys()):
            raise RuntimeError(f'Coefficient mismatch!\ncoeffs passed: {wcValues}\nconfig coeffs: {self.wcs}')
            
        likelihoodRatio = 0
        for wc in self.wcs:
            likelihoodRatio += (self.quad[wc](features)*wcValues[wc]**2).flatten()
        for i, wc0 in enumerate(self.wcs[:-1]):
            for wc1 in self.wcs[(i+1):]:
                if wc0 == 'sm' and wc1 not in self.noLin:
                    likelihoodRatio += (self.linear[wc1](features)*wcValues[wc1]).flatten()
                else: 
                    likelihoodRatio += self.interference[(wc0,wc1)](features)*wcValues[wc0]*wcValues[wc1]
        return likelihoodRatio