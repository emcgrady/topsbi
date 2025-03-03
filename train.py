from argparse import ArgumentParser
from tools.metrics import netEval
from tools.data import DataLoader
from model.net import Model
from torch import optim
from tqdm import tqdm
from os import makedirs

import matplotlib.pyplot as plt

import torch
import yaml

def savePlots(net, test, testLoss, trainLoss, label):
    try:
        makedirs(f'{label}')
    except:
        pass

    #save the network
    torch.save(net, f'{label}/network.p')
    torch.save(net.state_dict(), f'{label}/networkStateDict.p')

    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    
    #plot and save loss curves
    ax.plot( range(len(testLoss)), trainLoss, label="Training dataset")
    ax.plot( range(len(testLoss)), testLoss , label="Testing dataset")
    ax.set_title(label.split('/')[-1], fontsize=14)
    ax.legend()
    fig.savefig(f'{label}/loss.png')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lossLog.png')
    plt.clf()
    plt.close()

    backgroundMask = test[:][2] == 0
    signalMask     = test[:][2] == 1

    #plot the network output
    fig, ax = plt.subplots(1, 1, figsize=[12,7])
    bins = torch.linspace(0,1,200)
    ax.hist(net(test[:][0][backgroundMask]).ravel().detach().cpu().numpy(),
            weights=test[:][1][backgroundMask].detach().cpu().numpy(),
            bins=bins, alpha=0.5, label='Background', density=True)
    ax.hist(net(test[:][0][signalMask]).ravel().detach().cpu().numpy(),
            weights=test[:][1][signalMask].detach().cpu().numpy(),
            bins=bins, alpha=0.5, label='Signal', density=True)
    ax.set_xlabel('Network Output', fontsize=12)
    ax.set_title(label.split('/')[-1], fontsize=14)
    ax.legend()
    fig.savefig(f'{label}/netOut.png')
    ax.set_yscale('log')
    fig.savefig(f'{label}/netOutLog.png')
    plt.clf()
    plt.close()

    #get network performance metrics
    fpr, tpr, auc, a = netEval(net(test[:][0][backgroundMask]), net(test[:][0][signalMask]),
                         test[:][1][backgroundMask], test[:][1][signalMask])

    #make ROC curves
    fig, ax = plt.subplots(1, 1, figsize=[8,8])
    ax.plot(fpr, tpr, label='Network Performance')
    ax.plot([0,1],[0,1], ':', label='Baseline')
    ax.legend()
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(label.split('/')[-1], fontsize=14)
    fig.savefig(f'{label}/roc.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/rocLog.png')
    plt.clf()
    plt.close()
    
    #save performance metrics
    f = open(f'{label}/performance.txt','w+')
    f.write(f'Area under ROC: {auc}\nAccuracy:       {a}\n')
    f.close()

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help = 'configuration yml file used for training')
    
    #Load the configuration options and build the WC lists
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    

    torch.manual_seed(42)

    #Check for GPU availability and fall back on CPU if needed
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        config['device'] = 'cpu'

    #Load the model, loss funtion, and data
    model = Model(features=len(config['features'].split(",")),device=config['device'])
    data = DataLoader(config)
    #Normalize weights with their respective means
    signalMean = torch.mean(data[:][1][data[:][2] == 1]); backgroundMean = torch.mean(data[:][1][data[:][2] == 0])
    data[:][1][data[:][2] == 1] /= signalMean; data[:][1][data[:][2] == 0] /= backgroundMean
    
    train, test = torch.utils.data.random_split(data, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
    dataloader  = torch.utils.data.DataLoader(train, batch_size=config['batchSize'], shuffle=True)
    
    optimizer = optim.Adam(model.net.parameters(), lr=config['learningRate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=config['factor'], patience=config['patience'])
    trainLoss = [model.loss(train[:][0], train[:][1], train[:][2]).item()]
    testLoss  = [model.loss(test[:][0],  test[:][1],  test[:][2]).item()]

    for epoch in tqdm(range(config['epochs'])):
        if epoch%50==0: 
         savePlots(model.net, test, testLoss, trainLoss, f'{config["name"]}/epoch_{epoch:04d}')
        for features, weights, targets in dataloader:
            optimizer.zero_grad()
            loss = model.loss(features, weights, targets)
            loss.backward()
            optimizer.step()
        trainLoss.append(model.loss(train[:][0], train[:][1], train[:][2]).item())
        testLoss.append(model.loss(test[:][0], test[:][1], test[:][2]).item())
        scheduler.step(testLoss[epoch])
        
    savePlots(model.net, test, testLoss, trainLoss, f'{config["name"]}/last')

if __name__=="__main__":
    main()
    