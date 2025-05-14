from argparse import ArgumentParser
from model.net import Model
from os import makedirs
from tools.buildLikelihood import expandArray
from tools.data import dataLoader
from tools.plots import networkPlots
from torch import cuda, float64, Generator, manual_seed, mean, optim, zeros
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm
from yaml import safe_load

def main(config):
    manual_seed(42)
    
    #Check for GPU availability and fall back on CPU if needed
    if config['device'] != 'cpu' and not cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        config['device'] = 'cpu'
    
    #Load the model, loss funtion, and data
    features, eftCoefs, truth = dataLoader(config)[:]
    weights = zeros(truth.shape, dtype=float64)
    weights[truth == 0] = eftCoefs[truth == 0]@expandArray(config['backgroundTrainingPoint'])
    weights[truth == 1] = eftCoefs[truth == 1]@expandArray(config['signalTrainingPoint'])
    model   = Model(nFeatures=features.shape[1],device=config['device'], config=config['network'])
    #Normalize weights with their respective means
    if config['normalization'] == 'weightMean':
        weights[truth == 1] /= mean(weights[truth == 1])
        weights[truth == 0] /= mean(weights[truth == 0])
    elif config['normalization'] == 'unity':
        weights[truth == 1] /= weights[truth == 1].max()
        weights[truth == 0] /= weights[truth == 0].max()
    
    train, test = random_split(TensorDataset(features, weights, truth), [0.7, 0.3], generator=Generator().manual_seed(42))
    dataloader  = DataLoader(train, batch_size=config['batchSize'], shuffle=True)
    
    optimizer = optim.Adam(model.net.parameters(), lr=config['learningRate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=config['factor'], patience=config['patience'])
    trainLoss = [model.loss(train[:][0], train[:][1], train[:][2]).item()]
    testLoss  = [model.loss(test[:][0],  test[:][1],  test[:][2]).item()]

    for epoch in tqdm(range(config['epochs'])):
        if epoch%50==0: 
         networkPlots(model.net, test, testLoss, trainLoss, f'{config["name"]}/incomplete/epoch_{epoch:04d}')
        for features, weights, truth in dataloader:
            optimizer.zero_grad()
            loss = model.loss(features, weights, truth)
            loss.backward()
            optimizer.step()
        trainLoss.append(model.loss(train[:][0], train[:][1], train[:][2]).item())
        testLoss.append(model.loss(test[:][0], test[:][1], test[:][2]).item())
        scheduler.step(testLoss[epoch])
        
    networkPlots(model.net, test, testLoss, trainLoss, f'{config["name"]}/complete')

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('config', help = 'configuration yml file used for training')
    
    #Load the configuration options and build the WC lists
    with open(parser.parse_args().config, 'r') as f:
        config = safe_load(f)
    main(config)
    