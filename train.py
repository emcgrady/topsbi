from argparse import ArgumentParser
from model.net import Model
from torch import load, manual_seed, optim, float64, tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tools.buildLikelihood import expandArray
from tools.plots import networkPlots
from tools.data import prepareWeights
from yaml import safe_load, dump

def main(config):
    manual_seed(42)
    #Check for GPU availability and fall back on CPU if needed
    if config['device'] != 'cpu' and not cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        config['device'] = 'cpu'
    #load pre-split tensors
    train = load(f'{config["data"]}/train.p', weights_only=False)
    test  = load(f'{config["data"]}/test.p', weights_only=False)
    #Normalize features
    train[:][0][:] = (train[:][0] - train[:][0].mean(0))/train[:][0].std(0)
    test[:][0][:]  = (test[:][0] - test[:][0].mean(0))/test[:][0].std(0)

    train, test = prepareWeights(train, test, config)

    batches = DataLoader(train, batch_size=config['batchSize'], shuffle=True, num_workers=4)
    
    model     = Model(nFeatures=test[:][0].shape[1], device=config['device'], config=config['network'])
    optimizer = optim.Adam(model.net.parameters(), lr=config['learningRate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config['factor'], patience=config['patience'])
    trainLoss = [model.loss(train[:][0], train[:][1], train[:][2]).item()]
    testLoss  = [model.loss(test[:][0],  test[:][1],  test[:][2]).item()]
    
    for epoch in tqdm(range(config['epochs'])):
        if epoch%50==0: 
         networkPlots(model.net, test, testLoss, trainLoss, f'{config["name"]}/incomplete/epoch_{epoch:04d}')
        for features, backgroundWeights, signaWeights in batches:
            optimizer.zero_grad()
            loss = model.loss(features, backgroundWeights, signaWeights)
            loss.backward()
            optimizer.step()
        trainLoss.append(model.loss(train[:][0], train[:][1], train[:][2]).item())
        testLoss.append(model.loss(test[:][0], test[:][1], test[:][2]).item())
        scheduler.step(testLoss[epoch])
        
    networkPlots(model.net, test, testLoss, trainLoss, f'{config["name"]}/complete')

    return config
if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('config', help = 'configuration yml file used for training')
    
    #Load the configuration options and build the WC lists
    with open(parser.parse_args().config, 'r') as f:
        config = safe_load(f)
    config = main(config)
    