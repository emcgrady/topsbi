from torch.utils.data import TensorDataset
from torch import tensor, float64
from yaml import dump

def expandArray(coefs):
    arrayOut = []
    for i in range(len(coefs)):
         for j in range(i+1):
             arrayOut += [coefs[i]*coefs[j]]
    return tensor(arrayOut).to(float64)

def prepareWeights(train, test, config):
    if 'toSkip' in config.keys():
        train[:][1][:, config['toSkip']] = 0
        test[:][1][:,  config['toSkip']] = 0
    trainBW = train[:][1]@expandArray(config['backgroundTrainingPoint'])
    trainSW = train[:][1]@expandArray(config['signalTrainingPoint'])
    trainGW = train[:][1]@expandArray([1] + config['dataStartingPoint'])
    
    testBW = test[:][1]@expandArray(config['backgroundTrainingPoint'])
    testSW = test[:][1]@expandArray(config['signalTrainingPoint'])
    testGW = test[:][1]@expandArray([1] + config['dataStartingPoint'])

    nEvents = testSW.shape[0] + trainSW.shape[0]

    config['sig2bkg'] = ((trainSW.sum() + testSW.sum())/(trainBW.sum() + testBW.sum())).item()
    config['sig2gen'] = ((trainSW.sum() + testSW.sum())/(trainGW.sum() + testGW.sum())).item()
    config['bkg2gen'] = ((trainBW.sum() + testBW.sum())/(trainGW.sum() + testGW.sum())).item()

    with open(f'{config['name']}/training.yml', 'w') as f:
        dump(config, f)

    train = TensorDataset(train[:][0], trainBW/(trainGW*config['bkg2gen']), trainSW/(trainGW*config['sig2gen']))
    test  = TensorDataset(test[:][0],  testBW/(testGW*config['bkg2gen']),   testSW/(testGW*config['sig2gen']))
    return train, test