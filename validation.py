from tools.buildLikelihood import fullLikelihood
from tools.data import DataLoader
from tools.metrics import netEval
from model.net import Net
from argparse import ArgumentParser
from os import makedirs

import matplotlib.pyplot as plt
import numpy as np

import torch
import yaml

def main():
    parser = ArgumentParser()
    parser.add_argument('--parametric', '-p' , help = 'configuration yml file used for parametric likelihood')
    parser.add_argument('--dedicated', '-d' , help = 'configuration yml file used for dedicated likelihood')
    parser.add_argument('--output', '-o' , help = 'location to save output plots')

    args = parser.parse_args()
    parametric = fullLikelihood(args.parametric)
    dedicated  = fullLikelihood(args.dedicated)
    with open(dedicated.config['sm_quad']) as f:
        config = yaml.safe_load(f)

    data = DataLoader(config)
    train,test  = torch.utils.data.random_split(data, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
    
    backgroundPLR     = parametric(test[:][0][test[:][2] == 0], parametric.config['wcValues'])
    backgroundDLR     = dedicated(test[:][0][test[:][2] == 0], dedicated.config['wcValues'])
    backgroundWeights = test[:][1][test[:][2] == 0]
    signalPLR         = parametric(test[:][0][test[:][2] == 1], parametric.config['wcValues'])
    signalDLR         = dedicated(test[:][0][test[:][2] == 1], dedicated.config['wcValues'])
    signalWeights     = test[:][1][test[:][2] == 1]

    dEval = netEval(backgroundDLR, signalDLR, backgroundWeights, signalWeights)
    pEval = netEval(backgroundPLR, signalPLR, backgroundWeights, signalWeights)

    backgroundPLR     = backgroundPLR.detach().numpy()
    backgroundDLR     = backgroundDLR.detach().numpy()
    backgroundWeights = backgroundWeights.detach().numpy()
    signalPLR         = signalPLR.detach().numpy()
    signalDLR         = signalDLR.detach().numpy()
    signalWeights     = signalWeights.detach().numpy()

    try:
        makedirs(args.output)
    except:
        pass

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    backgroundResiduals = (backgroundDLR - backgroundPLR)
    signalResiduals     = (signalDLR - signalPLR)
    bins = np.linspace(np.min([backgroundResiduals.min(), signalResiduals.min()]),
                       np.max([backgroundResiduals.max(), signalResiduals.max()]),
                       200
                      )
    ax.hist(backgroundResiduals, bins=bins, label='Background', alpha=0.6)
    ax.hist(signalResiduals, bins=bins, label='Signal', alpha=0.6)
    ax.set_xlabel('Residuals', fontsize=12)
    ax.legend()
    ax.set_yscale('log')
    fig.savefig(f'{args.output}/residuals.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    bins = np.linspace(
        np.min([backgroundDLR.min(), signalDLR.min()]),
        np.max([backgroundDLR.max(), signalDLR.max()]),
        200
    )
    ax.hist(backgroundDLR, weights=backgroundWeights, bins=bins, label='Background', alpha=0.6)
    ax.hist(signalDLR, weights=signalWeights, bins=bins, label='Signal', alpha=0.6)
    ax.set_title('Dedicated Likelihood Ratio', fontsize=14)
    ax.set_xlabel('Likelihood Ratio', fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'{args.output}/dedicatedLR.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    bins = np.linspace(
        np.min([backgroundPLR.min(), signalPLR.min()]),
        np.max([backgroundPLR.max(), signalPLR.max()]),
        200
    )
    ax.hist(backgroundPLR, weights=backgroundWeights,  bins=bins, label='Background', alpha=0.6)
    ax.hist(signalPLR, weights=signalWeights, bins=bins, label='Signal', alpha=0.6)
    ax.set_title('Parametric Likelihood Ratio', fontsize=14)
    ax.set_xlabel('Likelihood Ratio', fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'{args.output}/parametricLR.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(dEval[0], dEval[1], label='Dedicated')
    ax.plot(pEval[0], pEval[1], label='Parametric')
    ax.plot([0,1], [0,1], ':')
    ax.legend()
    fig.savefig(f'{args.output}/roc.png')
    plt.clf()
    plt.close()

if __name__=="__main__":
    main()