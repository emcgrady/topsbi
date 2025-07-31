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
    trainBW = train[:][1]@expandArray(config['backgroundTrainingPoint'])
    trainSW = train[:][1]@expandArray(config['signalTrainingPoint'])
    trainGW = train[:][1]@expandArray([1] + config['dataStartingPoint'])
    trainSM = train[:][1]@expandArray([1] + [0]*(len(config['signalTrainingPoint']) - 1))
    
    testBW = test[:][1]@expandArray(config['backgroundTrainingPoint'])
    testSW = test[:][1]@expandArray(config['signalTrainingPoint'])
    testGW = test[:][1]@expandArray([1] + config['dataStartingPoint'])
    testSM = test[:][1]@expandArray([1] + [0]*(len(config['signalTrainingPoint']) - 1))

    nEvents = testSW.shape[0] + trainSW.shape[0]
    
    config['sig2gen'] = (((trainSW/trainGW).sum() + (testSW/testGW).sum())/nEvents).item()
    config['bkg2gen'] = (((trainBW/trainGW).sum() + (testBW/testGW).sum())/nEvents).item()
    config['sig2bkg'] = (((trainSW/trainBW).sum() + (testSW/testBW).sum())/nEvents).item()
    config['bkg2sm']  = (((trainBW/trainSM).sum() + (testBW/testSM).sum())/nEvents).item()

    with open(f'{config['name']}/training.yml', 'w') as f:
        dump(config, f)

    train = TensorDataset(train[:][0], trainBW/(trainGW*config['bkg2gen']), trainSW/(trainGW*config['sig2gen']))
    test  = TensorDataset(test[:][0],  testBW/(testGW*config['bkg2gen']),   testSW/(testGW*config['sig2gen']))
    return train, test