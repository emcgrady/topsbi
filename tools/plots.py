from topcoffea.modules.histEFT import HistEFT
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt
import numpy as np

import hist

def histPlot(background, signal, label, out, name, backgroundWeights=None, signalWeights=False, ylog=False, xlog=False):
    fig, ax = plt.subplots(figsize=[12,8])
    print(np.min([background.min(), signal.min()]))
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
        ax.hist(background, weights=backgroundWeights, bins=bins, label='background', alpha=0.6)
    else:
        ax.hist(background, bins=bins, label='background', alpha=0.6)
    if signalWeights:
        ax.hist(signal, weights=signalWeights, bins=bins, label='signal', alpha=0.6)
    else:
        ax.hist(signal, bins=bins, label='signal', alpha=0.6)
    ax.set_xlabel(label, fontsize=12)
    ax.legend()
    if ylog:
        ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')
    fig.savefig(f'{out}/{name}.png')
    plt.clf()
    plt.close()

def ratioPlot(x, dedicatedLR, parametricLR, eftCoeffs, bins, wcs, outname, 
              plotLog=False, ratioLog=False, xlabel=None, showNoWeights=False, density=False):
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