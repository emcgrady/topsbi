from tools.buildLikelihood import fullLikelihood
from tools.data import DataLoader
from tools.metrics import netEval
from model.net import Net
from matplotlib.gridspec import GridSpec
from argparse import ArgumentParser
from os import makedirs
from re import split

from topcoffea.modules.histEFT import HistEFT
import topcoffea.modules.utils as utils
import matplotlib.pyplot as plt
import numpy as np

import torch
import yaml
import hist


def ratioPlot(x, dedicatedLR, parametricLR, eftCoeffs, bins, wcs, outname, 
              plotLog=False, ratioLog=False, xlabel=None, showNoWeights=False, density=True):
    ax  = []
    fig = plt.figure(figsize=(12,9))
    gs  = GridSpec(6,6, figure=fig)

    histEFT = HistEFT(hist.axis.StrCategory(['histEFT'], name='category'),
                      hist.axis.Regular(
                          start=min(bins),
                          stop=max(bins),
                          bins=len(bins) - 1,
                          name="kin",
                          label='HistEFT'
                      ),
                      wc_names=wcs[::2]
                     )

    ax.append(fig.add_subplot(gs[0:5,0:5]))
    ax.append(fig.add_subplot(gs[5,0:5]))
    plt.subplots_adjust(hspace=0.2)

    histEFT.fill(kin=x, eft_coeff=eftCoeffs, category='histEFT')

    histEFTEval = histEFT.as_hist([float(j) for j in wcs[1::2]])
    histEFTEval.plot1d(ax=ax[0], density=density, yerr=False)
    nDedicated,_,_  = ax[0].hist(x, bins=bins, weights=dedicatedLR, label='Dedicated', histtype='step', density=density)
    nParametric,_,_ = ax[0].hist(x, bins=bins, weights=parametricLR, label='Parametric', histtype='step', density=density)
    if showNoWeights: 
        ax[0].hist(x, bins=bins, label='No Weights', histtype='step', color='k', linestyle='dashed', density=density)
    ax[0].legend()
    ax[0].set_xlabel('')
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('')
    if plotLog:
        ax[0].set_yscale('log')
    ax[0].autoscale() 

    if density:
        nHistEft = (histEFTEval.values().flatten()/(sum(histEFTEval.values().flatten())*np.diff(bins)))
        dedicatedRatio = np.ones(nHistEft.shape)
        dedicatedRatio[nHistEft != 0] = nHistEft[nHistEft != 0]/nDedicated[nHistEft != 0]
        paramtricRatio = np.ones(nHistEft.shape)
        paramtricRatio[nHistEft != 0] = nHistEft[nHistEft != 0]/nParametric[nHistEft != 0]
    else: 
        nHistEft = histEFTEval.values().flatten()
        dedicatedRatio = np.ones(nHistEft.shape)
        dedicatedRatio[nHistEft != 0] = nDedicated[nHistEft != 0]/nHistEft[nHistEft != 0]
        paramtricRatio = np.ones(nHistEft.shape)
        paramtricRatio[nHistEft != 0] = nParametric[nHistEft != 0]/nHistEft[nHistEft != 0]
        
    ax[1].hlines(1,x.min(), x.max(), color='k', linestyle='dashed')
    ax[1].plot((bins[1:] + bins[:-1])/2, dedicatedRatio, '^', label='Dedicated', color='orange')
    ax[1].plot((bins[1:] + bins[:-1])/2, paramtricRatio, 'v', label='Parametric', color='green')
    ax[1].legend()
    if ratioLog:
        ax[1].set_yscale('log')
    if xlabel:
        ax[1].set_xlabel(xlabel, fontsize=12)
    ax[1].set_xlim(ax[0].get_xlim())
    fig.savefig(f'{outname}.png')
    plt.clf()
    plt.close()

def main():
    parser = ArgumentParser()
    parser.add_argument('--parametric', '-p' , help = 'configuration yml file used for parametric likelihood')
    parser.add_argument('--dedicated', '-d' , help = 'configuration yml file used for dedicated likelihood')
    parser.add_argument('--output', '-o' , help = 'location to save output plots')
    parser.add_argument('--validation_set', '-v', default='all', help = 'which dataset to use for validation. Can choose from all, test, or train')

    args = parser.parse_args()
    parametric = fullLikelihood(args.parametric)
    dedicated  = fullLikelihood(args.dedicated)

    data = DataLoader(dedicated.config)

    if args.validation_set == 'test':
        _,data  = torch.utils.data.random_split(data, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
    elif args.validation_set == 'train':
        data,_  = torch.utils.data.random_split(data, [0.7, 0.3], generator=torch.Generator().manual_seed(42))

    plr = parametric(data[:][0], parametric.config['wcValues'])
    dlr = dedicated(data[:][0], dedicated.config['wcValues'])

    backgroundMask = data[:][2] == 0
    signalMask     = data[:][2] == 1
    
    dEval = netEval(dlr[backgroundMask], dlr[signalMask], data[:][1][backgroundMask], data[:][1][signalMask])
    pEval = netEval(plr[backgroundMask], plr[signalMask], data[:][1][backgroundMask], data[:][1][signalMask])

    try:
        makedirs(args.output)
    except:
        pass

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    backgroundResiduals = (dlr[backgroundMask] - plr[backgroundMask]).detach().numpy()
    signalResiduals     = (dlr[signalMask] - plr[signalMask]).detach().numpy()

    metrics = {
        'background': {
            'residualMean': float(backgroundResiduals.mean()),
            'residualMin':  float(backgroundResiduals.min()),
            'residualMax':  float(backgroundResiduals.max()),
            'residualMedian': float(np.median(backgroundResiduals)),
            'residualStdv': float(backgroundResiduals.std())
        },
        'signal': {
            'residualMean': float(signalResiduals.mean()),
            'residualMin':  float(signalResiduals.min()),
            'residualMax':  float(signalResiduals.max()),
            'residualMedian': float(np.median(signalResiduals)),
            'residualStdv': float(signalResiduals.std())
        }
    }
    
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
    bins = np.linspace(np.min([np.log10(backgroundResiduals.min()), np.log10(signalResiduals.min())]),
                       np.max([np.log10(backgroundResiduals.max()), np.log10(signalResiduals.max())]),
                       200
                      )
    ax.hist(np.log10(backgroundResiduals), bins=bins, label='Background', alpha=0.6)
    ax.hist(np.log10(signalResiduals), bins=bins, label='Signal', alpha=0.6)
    ax.set_xlabel('Log Residuals', fontsize=12)
    ax.legend()
    ax.set_yscale('log')
    fig.savefig(f'{args.output}/logResiduals.png')
    plt.clf()
    plt.close()

    dlr = dlr.detach().numpy()
    plr = plr.detach().numpy()

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    bins = np.linspace(
        np.min([dlr[backgroundMask].min(), dlr[signalMask].min()]),
        np.max([dlr[backgroundMask].max(), dlr[signalMask].max()]),
        200
    )
    ax.hist(dlr[backgroundMask], weights=data[:][1][backgroundMask], bins=bins, label='Background', alpha=0.6)
    ax.hist(dlr[signalMask], weights=data[:][1][signalMask], bins=bins, label='Signal', alpha=0.6)
    ax.set_title('Dedicated Likelihood Ratio', fontsize=14)
    ax.set_xlabel('Likelihood Ratio', fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'{args.output}/dedicatedLR.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    bins = np.logspace(
        np.log10(np.min([dlr[backgroundMask].min(), dlr[signalMask].min()])),
        np.log10(np.max([dlr[backgroundMask].max(), dlr[signalMask].max()])),
        200
    )
    ax.hist(dlr[backgroundMask], weights=data[:][1][backgroundMask], bins=bins, label='Background', alpha=0.6)
    ax.hist(dlr[signalMask], weights=data[:][1][signalMask], bins=bins, label='Signal', alpha=0.6)
    ax.set_title('Dedicated Likelihood Ratio', fontsize=14)
    ax.set_xlabel('Likelihood Ratio', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'{args.output}/dedicatedLR_Log.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    bins = np.linspace(
        np.min([plr[backgroundMask].min(), plr[signalMask].min()]),
        np.max([plr[backgroundMask].max(), plr[signalMask].max()]),
        200
    )
    ax.hist(plr[backgroundMask], weights=data[:][1][backgroundMask],  bins=bins, label='Background', alpha=0.6)
    ax.hist(plr[signalMask], weights=data[:][1][signalMask], bins=bins, label='Signal', alpha=0.6)
    ax.set_title('Parametric Likelihood Ratio', fontsize=14)
    ax.set_xlabel('Likelihood Ratio', fontsize=12)
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'{args.output}/parametricLR.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(12,8))
    bins = np.logspace(
        np.log10(np.min([plr[backgroundMask].min(), plr[signalMask].min()])),
        np.log10(np.max([plr[backgroundMask].max(), plr[signalMask].max()])),
        200
    )
    ax.hist(plr[backgroundMask], weights=data[:][1][backgroundMask],  bins=bins, label='Background', alpha=0.6)
    ax.hist(plr[signalMask], weights=data[:][1][signalMask], bins=bins, label='Signal', alpha=0.6)
    ax.set_title('Parametric Likelihood Ratio', fontsize=14)
    ax.set_xlabel('Likelihood Ratio', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(f'{args.output}/parametricLR_Log.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(dEval[0], dEval[1], label='Dedicated')
    ax.plot(pEval[0], pEval[1], '--', label='Parametric')
    ax.plot([0,1], [0,1], ':')
    ax.legend()
    fig.savefig(f'{args.output}/roc.png')
    plt.clf()
    plt.close()

    validationData = DataLoader(dedicated.config, dataType='HistEFT')

    if args.validation_set == 'all':
        validation = validationData
    elif args.validation_set == 'test':
        _, validation = torch.utils.data.random_split(validationData, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
    elif args.validation_set == 'train':
        validation, _ = torch.utils.data.random_split(validationData, [0.7, 0.3], generator=torch.Generator().manual_seed(42))

    backgroundStartingPoint = torch.Tensor([float(i) for i in split( '=|:', dedicated.config['backgroundStartingPoint'])[1::2]])
    
    ### FIXME TO WORK FOR ARBITRARY NUMBER OF WCS
    dlr *= (validation[:][1][:,0]
            + validation[:][1][:,1]*backgroundStartingPoint
            + validation[:][1][:,2]*backgroundStartingPoint**2).detach().numpy()
    plr *= (validation[:][1][:,0]
            + validation[:][1][:,1]*backgroundStartingPoint
            + validation[:][1][:,2]*backgroundStartingPoint**2).detach().numpy()

    featureKey = dedicated.config['featuresKey'].split(',')
    wcs = split('=|:', dedicated.config['signalTrainingPoint'])

    for key, value in dedicated.config['features'].items():
        i = featureKey.index(key)
        x = data[:][0][:,i].detach().numpy()*dedicated.config['featureStdvs'][0][i] + dedicated.config['featureMeans'][0][i]
        bins = np.linspace(value['min'], value['max'], value['nbins']+1)
        ratioPlot(x, dlr, plr, validation[:][1].detach().numpy(), 
                  bins, wcs, f'{args.output}/{key}', xlabel=value['label'], showNoWeights=True)
        ratioPlot(x, dlr, plr, validation[:][1].detach().numpy(), 
                  bins, wcs, f'{args.output}/{key}_Log', plotLog=True, xlabel=value['label'], showNoWeights=True)

    with open(f'{args.output}/metrics.yml', 'w') as f:
        f.write(yaml.dump(metrics))

if __name__=="__main__":
    main()