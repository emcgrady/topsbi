from model.net import Net
from yaml import safe_load
from torch import load, device, float64

class likelihood:
    def __init__(self, config, nFeatures, network):
        with open(config) as f:
            self.config = safe_load(f)
        self.model = Net(nFeatures, self.config['device'])
        self.model.load_state_dict(load(f'{self.config["name"]}/{network}/networkStateDict.p',
                                        map_location=device(self.config['device'])))
    def __call__(self, features):
        score = self.model(features.to(float64))
        return score/(1-score)
class linearTerm:
    def __init__(self, sm, linear, linearValue, quad):
        self.sm          = sm
        self.linear      = linear
        self.quad        = quad
        self.linearValue = linearValue
    def __call__(self, features):
        return (self.linear(features) - self.sm(features) - self.quad(features)*self.linearValue**2)/self.linearValue
class fullLikelihood: 
    def __init__(self, config, network):
        with open(config) as f:
            self.config = safe_load(f.read())
        self.quad = {}; self.linear = {}; self.wcValues = {}; nFeatures = len(self.config['features'])
        self.quad['sm'] = likelihood(self.config['terms']['sm']['net'], nFeatures, network=network) 
        self.wcValues['sm'] = self.config['terms']['sm']['value']
        for term, params in self.config['terms'].items():
            self.wcValues[term] = params['value']
            if term != 'sm':
                self.quad[term] = likelihood(params['quad'], nFeatures, network)
                if 'linear' in params:
                    self.linear[term] = linearTerm(self.quad['sm'], 
                                                   likelihood(params['linear'], nFeatures, network), 
                                                   params['value'],
                                                   self.quad[term])
    def __call__(self, features, wcValues):
        likelihoodRatio = 0
        for wc, value in wcValues.items():
            likelihoodRatio += (self.quad[wc](features)*value**2).flatten()
            if wc in self.linear:
                likelihoodRatio += (self.linear[wc](features)*value).flatten()
        return likelihoodRatio