from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import clf, close, figure, subplots, subplots_adjust
from os import makedirs
from tools.metrics import netEval
from torch import save
from yaml import dump
import numpy as np

def histPlot(background, signal, label, outname=None, backgroundWeights=None, signalWeights=False, ylog=False, xlog=False):
    fig, ax = subplots(figsize=[12,8])
    if xlog:
        bins = np.logspace(np.log10(np.min([background[background > 0] .min(), signal[signal > 0].min()])), 
                           np.log10(np.max([background.max(), signal.max()])),
                           200
                          )
    else:
        bins = np.linspace(np.min([background.min(), signal.min()]), 
                           np.max([background.max(), signal.max()]),
                           200
                          )
    if backgroundWeights:
        ax.hist(background, weights=backgroundWeights, bins=bins, label='background', histtype='step')
    else:
        ax.hist(background, bins=bins, label='background', histtype='step')
    if signalWeights:
        ax.hist(signal, weights=signalWeights, bins=bins, label='signal', histtype='step')
    else:
        ax.hist(signal, bins=bins, label='signal', histtype='step')
    ax.set_xlabel(label, fontsize=12)
    ax.legend()
    if ylog:
        ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')
    if outname:
        fig.savefig(f'{outname}')
        clf()
        close()
    else:
        fig.show()

def ratioPlot(x, dedicatedLR, parametricLR, eftCoeffs, bins, wcs, outname=None, 
              plotLog=False, ratioLog=False, xlabel=None, showNoWeights=False, density=False):
    from hist.axis import Regular, StrCategory
    from topcoffea.modules.histEFT import HistEFT
    ax  = []
    fig = figure(figsize=(12,9))
    gs  = GridSpec(6,6, figure=fig)
    
    [terms,values] = zip(*wcs.items())
    
    histEFT = HistEFT(StrCategory(['histEFT'], name='category'),
                      Regular(
                          start=min(bins),
                          stop=max(bins),
                          bins=len(bins) - 1,
                          name="kin",
                          label='HistEFT'
                      ),
                      wc_names=terms
                     )

    ax.append(fig.add_subplot(gs[0:5,0:5]))
    ax.append(fig.add_subplot(gs[5,0:5]))
    subplots_adjust(hspace=0.2)

    histEFT.fill(kin=x, eft_coeff=eftCoeffs, category='histEFT')

    histEFTEval = histEFT.as_hist(values)
    histEFTEval.plot1d(ax=ax[0], density=density, yerr=False)
    nDedicated,_,_  = ax[0].hist(x, bins=bins, weights=dedicatedLR, label='Dedicated', histtype='step', density=density)
    nParametric,_,_ = ax[0].hist(x, bins=bins, weights=parametricLR, label='Parametric', histtype='step', density=density, linestyle='dashdot')
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
        dedicatedRatio[nHistEft != 0] = nDedicated[nHistEft != 0]/nHistEft[nHistEft != 0]
        paramtricRatio = np.ones(nHistEft.shape)
        paramtricRatio[nHistEft != 0] = nParametric[nHistEft != 0]/nHistEft[nHistEft != 0]
    else: 
        nHistEft = histEFTEval.values().flatten()
        mask = (nHistEft != 0) & (nParametric > 0)
        dedicatedRatio = np.ones(nHistEft.shape)
        dedicatedRatio[mask] = nDedicated[mask]/nHistEft[mask]
        parametricRatio = np.ones(nHistEft.shape)
        parametricRatio[mask] = nParametric[mask]/nHistEft[mask]
        
    ax[1].hlines(1,ax[0].get_xlim()[0], ax[0].get_xlim()[1], color='k', linestyle='dashed')
    ax[1].plot((bins[1:] + bins[:-1])/2, dedicatedRatio, '^', label='Dedicated', color='orange')
    ax[1].plot((bins[1:] + bins[:-1])/2, parametricRatio, 'v', label='Parametric', color='green')
    ax[1].legend() 

    if ax[1].get_ylim()[0] > 0:
        order = max([np.log10(ax[1].get_ylim()[1]), abs(np.log10(ax[1].get_ylim()[0]))])
    else: 
        lEdge = np.min((dedicatedRatio[dedicatedRatio != 0].min(), parametricRatio[parametricRatio != 0].min()))
        uEdge = np.max((dedicatedRatio[dedicatedRatio != 0].max(), parametricRatio[parametricRatio != 0].max()))
        order = np.max((abs(np.log10(lEdge)), abs(np.log10(uEdge))))
    if ratioLog or (abs(order) > 1):
        ax[1].set_yscale('log')
        ax[1].set_ylim(10**(-order), 10**order)
    else:
        deviation = max([ax[1].get_ylim()[1] - 1, 1- ax[1].get_ylim()[0]])
        if deviation < 1:
            ax[1].set_ylim(1-deviation, 1+deviation)
        else:
            ax[1].set_ylim(0, 1+deviation)
    if xlabel:
        ax[1].set_xlabel(xlabel, fontsize=12)
    ax[1].set_xlim(ax[0].get_xlim())
    if outname:
        fig.savefig(f'{outname}')
        clf()
        close()
    else:
        fig.show()

def networkPlots(net, test, testLoss, trainLoss, label):
    

    makedirs(f'{label}', mode=0o755, exist_ok=True)

    #save the network
    save(net, f'{label}/network.p')
    save(net.state_dict(), f'{label}/networkStateDict.p')

    fig, ax = subplots(figsize=[8,8])
    
    #plot and save loss curves
    ax.plot( range(len(testLoss)), trainLoss, label="Training dataset")
    ax.plot( range(len(testLoss)), testLoss , label="Testing dataset")
    ax.set_title(label.split('/')[-1], fontsize=14)
    ax.legend()
    fig.savefig(f'{label}/loss.png')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lossLog.png')
    clf()
    close()

    backgroundMask = test[:][2] == 0
    signalMask     = test[:][2] == 1

    #plot the network output
    fig, ax = subplots(figsize=[12,7])
    bins = np.linspace(0,1,200)
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
    clf()
    close()

    #get network performance metrics
    fpr, tpr, auc, a = netEval(net(test[:][0][backgroundMask]), net(test[:][0][signalMask]),
                         test[:][1][backgroundMask], test[:][1][signalMask])

    #make ROC curves
    fig, ax = subplots(1, 1, figsize=[8,8])
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
    clf()
    close()
    
    #save performance metrics
    performance = {
        'Area under ROC': auc,
        'Accuracy': a
    }
    with open(f'{label}/performance.yml','w') as f:
        f.write(dump(performance))