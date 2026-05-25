from hist import Hist
from hist.axis import Regular, StrCategory
from matplotlib.axes import Axes
from matplotlib.ticker import StrMethodFormatter
from topsbi.tools.buildLikelihood import expandArray
from topsbi.tools.metrics import netEval
from topcoffea.modules.histEFT import HistEFT

import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh

import os torch, yaml

def kinematicRatioPlot(
    x: np.array, 
    dlr: np.array, 
    plr: np.array, 
    fitCoefs: list[float], 
    **params
):
    """
    Plots histogram and ratio for dedicated and parametric training.
    Ratios are calculated with respect to HistEFT.

    Args:
        x: Kinemtaic to be plotted
        dlr: dedicated likelihood ratio 
        plr: parametric likelihood ratio
        fitCoefs: EFTFitCoefficients used to calculate the event weights
        params: dictionary containing plotting information
    """
    #initialize the figure
    ax   = []
    fig  = plt.figure(figsize=(12,9))
    grid = fig.add_gridspec(2, 1, hspace=0.05, height_ratios=[5, 1])
    ax  += [fig.add_subplot(grid[0])]
    ax  += [fig.add_subplot(grid[1], sharex=ax[0])]
    
    plt.setp(ax[0].get_xticklabels(), visible=False)

    #initialize the histograms
    correct = HistEFT(StrCategory([], name   = 'process', growth = True),
                      Regular(name ='correct',
                              label=params['label'],
                              bins=params['nbins'] - 1,
                              start=params['min'],
                              stop=params['max']
                              ),
                      wc_names = params['wcs'],
                      label = 'Events'
                      )
    dedicated  = Hist(Regular(name='dedicated',
                              label=params['label'],
                              bins=params['nbins'] - 1,
                              start=params['min'],
                               stop=params['max']
                              )
                      )
    parametric = Hist(Regular(name='parametric',
                              label=params['label'],
                              bins=params['nbins'] - 1,
                              start=params['min'],
                              stop=params['max']
                              )
                      )
    
    #convert the likelihood ratio to weights
    bkg  = fitCoefs@expandArray(params['backgroundTrainingPoint']).detach().numpy()
    sig  = fitCoefs@expandArray(params['signalTrainingPoint']).detach().numpy()
    norm = sig.sum()/bkg.sum()*bkg/x.shape[0]
    dedi = dlr*norm
    para = plr*norm
    
    #fill the histograms
    correct.fill(correct=x, eft_coeff=fitCoefs, process='process')
    correct = correct.as_hist(params['signalTrainingPoint'][1:]).integrate('process')
    dedicated.fill(dedicated=x, weight=dedi)
    parametric.fill(parametric=x, weight=para)

    #calculate error
    cNum, bins = correct.to_numpy()
    dNum = dedicated.values()
    pNum = parametric.values()
    cErr = []
    dErr = []
    pErr = []
    
    for i in range(params['nbins'] - 1):
        cErr.append((sig[(x >= bins[i]) & (x < bins[i+1])]**2).sum())
        dErr.append((dedi[(x >= bins[i]) & (x < bins[i+1])]**2).sum())
        pErr.append((para[(x >= bins[i]) & (x < bins[i+1])]**2).sum())
    cErr = np.sqrt(np.hstack(cErr))
    dErr = np.sqrt(np.hstack(dErr))
    pErr = np.sqrt(np.hstack(pErr))
    
    #plot the histograms
    correct.plot1d(ax=ax[0], yerr=cErr, label='Correct')
    dedicated.plot1d(ax=ax[0],  yerr=dErr, label='Dedicated',  linestyle='dashdot', color='orange')
    parametric.plot1d(ax=ax[0], yerr=pErr, label='Parametric', linestyle='dashed',  color='green')
    
    #plot the ratio and ratio errors
    ax[1].hlines(1, bins[0], bins[-1], color='k', linestyle='dashed')
    rBins = np.diff(bins)/2+bins[:-1]
    dVals = np.divide(cNum, dNum, out=np.repeat(np.nan, cNum.shape), where=dNum!=0)
    pVals = np.divide(cNum, pNum, out=np.repeat(np.nan, cNum.shape), where=pNum!=0)
    
    cRatio = np.divide(cErr, cNum, out=np.repeat(np.nan, cErr.shape), where=cNum!=0)
    dRatio = np.divide(dErr, dNum, out=np.repeat(np.nan, dErr.shape), where=dNum!=0)
    pRatio = np.divide(pErr, pNum, out=np.repeat(np.nan, pErr.shape), where=pNum!=0)
    
    ax[1].bar(rBins, 2*np.sqrt((cRatio + dRatio) * dVals), width=np.diff(bins), 
              bottom = dVals - np.sqrt((cRatio + dRatio) * dVals), edgecolor='orange', lw=0,
              hatch='//',  hatch_linewidth=0.8, color='none', label='Dedicated Uncertainty')
    ax[1].bar(rBins, 2*np.sqrt((cRatio + pRatio) * pVals), width=np.diff(bins), 
              bottom = pVals - np.sqrt((cRatio + pRatio) * pVals), edgecolor='green', lw=0,
              hatch='\\\\', hatch_linewidth=0.8, color='none', label='Parametric Uncertainty')
    ax[1].plot(rBins, dVals, '^', label='Dedicated', color='orange')
    ax[1].plot(rBins, pVals, 'v', label='Parametric', color='green')
    ax[1].set_ylim([0,2])
    
    #clean up formatting
    if params['title']:
        ax[0].set_title(params['title'], fontsize=14)
    if params['plotLog']:
        ax[0].set_yscale('log')
    ax[0].set_xlabel('') 
    ax[0].set_ylabel('counts', fontsize=12)
    ax[1].set_xlabel(params['label'], fontsize=12) 
    ax[1].set_ylabel('ratio', fontsize=12)
    ax[0].set_xlim(params['min'], params['max'])
    ax[0].legend()
    if params['outname']:
        fig.savefig(f'{params["outname"]}')
        plt.clf()
        plt.close()
    else:
        fig.show()

def loss_curve(
    ax: Axes,
    train_loss: list[float],
    test_loss: list[float]
) -> None:
    """
    Plot training and test loss over epochs.

    Args:
        ax: The matplotlib Axes to plot on.
        train_loss: loss for training data
        test_loss: loss for testing data
    """
    ax.plot(train_loss, label="Training dataset", linewidth=3)
    ax.plot(test_loss , label="Testing dataset", linewidth=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    
def networkPlots(features, p0, p1, net, train_loss, test_loss, label):
    mh.style.use("CMS")
    os.makedirs(f'{label}', mode=0o755, exist_ok=True)
    torch.save(net, f'{label}/network.p')
    torch.save(net.state_dict(), f'{label}/networkStateDict.p')

    performance = {}
    
    #convert tensors to np arrays
    s      = net(features).ravel().detach().cpu().numpy()
    noOnes = s < 1
    s      = s[noOnes]
    p0     = p0.detach().cpu().numpy()[noOnes]
    p1     = p1.detach().cpu().numpy()[noOnes]
    predLr = s/(1-s)

    #loss curves
    fig, ax = plt.subplots(figsize=[16,16])
    loss_curve(ax, train_loss, test_loss)
    mh.cms.label("Preliminary", data=False, lumi=137.64, com=13, ax=ax)
    fig.savefig(f'{label}/loss.png')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lossLog.png')
    plt.clf()
    plt.close()

    #plot the network output
    fig, ax = plt.subplots(figsize=[12,7])
    bins = np.linspace(0,1,200)
    ax.hist(s, weights=p0, bins=bins, alpha=0.5, label='Background', density=True)
    ax.hist(s, weights=p1, bins=bins, alpha=0.5, label='Signal', density=True)
    ax.set_xlabel('Network Output')
    mh.cms.label("Preliminary", data=False, lumi=137.64, com=13, ax=ax)
    ax.legend()
    fig.savefig(f'{label}/netOut.png')
    ax.set_yscale('log')
    fig.savefig(f'{label}/netOutLog.png')
    plt.clf()
    plt.close()

    #get network performance metrics
    fpr, tpr, auc, a = netEval(s, p0, p1)
    
    performance['Area under ROC'] =  auc
    performance['Accuracy'] = a

    #make ROC curves
    fig, ax = plt.subplots(figsize=[16,16])
    ax.plot(fpr, tpr, label='Network Performance', linewidth=3)
    ax.plot([0,1],[0,1], ':', label='Baseline', linewidth=3)
    ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    mh.cms.label("Preliminary", data=False, lumi=137.64, com=13, ax=ax)
    fig.savefig(f'{label}/roc.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/rocLog.png')
    plt.clf()
    plt.close()

    #check quantile binned s
    fig, ax = plt.subplots(figsize=[16,16])
    performance['sChiExcl'] = sMeanPlot(ax, s, p0, p1, 1000, 0.01)
    fig.savefig(f'{label}/sExcl.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/sExclLog.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=[16,16])
    performance['sChiIncl'] = sMeanPlot(ax, s, p0, p1, 1000, 0)
    fig.savefig(f'{label}/sIncl.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/sInclLog.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=[16,16])
    sMeanPlot(ax, s, p0, p1, 50, 0.01)
    fig.savefig(f'{label}/sExcl_lobin.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/sExclLog_lobin.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=[16,16])
    sMeanPlot(ax, s, p0, p1, 50, 0)
    fig.savefig(f'{label}/sIncl_lobin.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/sInclLog_lobin.png')
    plt.clf()
    plt.close()
    
    #check quantile binned lr
    fig, ax = plt.subplots(figsize=[16,16])
    performance['lrChiExcl'] = lrMeanPlot(ax, predLr, p1/p0, p0, 1000, 0.01)
    fig.savefig(f'{label}/lrExcl.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lrExclLog.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=[16,16])
    performance['lrChiIncl'] = lrMeanPlot(ax, predLr, p1/p0, p0, 1000, 0)
    fig.savefig(f'{label}/lrIncl.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lrInclLog.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=[16,16])
    lrMeanPlot(ax, predLr, p1/p0, p0, 50, 0.01)
    fig.savefig(f'{label}/lrExcl_lobin.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lrExclLog_lobin.png')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(figsize=[16,16])
    lrMeanPlot(ax, predLr, p1/p0, p0, 50, 0)
    fig.savefig(f'{label}/lrIncl_lobin.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lrInclLog_lobin.png')
    plt.clf()
    plt.close()

    #2d hist
    if (predLr == 0).any(): 
        print('Predicted lr of 0 found!')
        print('Filtering predicted lr=0 events for 2d hists of llr...')
        zeroMask = predLr > 0
        predLr   = predLr[zeroMask]
        p0       = p0[zeroMask]
        p1       = p1[zeroMask]
        
    if (predLr < 0).any():
        print('negative predicted LR found!')
        print('skipping histogram plots...')
    else:
        try:
            fig, ax = plt.subplots(figsize=[16,18])
            h = hist2d(ax, np.log10(predLr), np.log10((p1/p0)), p0, **{})
            fig.colorbar(h[3], ax=ax)
            fig.savefig(f'{label}/hist2d.png')
            plt.clf()
            plt.close()
        
            fig, ax = plt.subplots(figsize=[16,18])
            h = hist2d(ax, np.log10(predLr), np.log10((p1/p0)), p0, logbins=True, **{})
            fig.colorbar(h[3], ax=ax)
            fig.savefig(f'{label}/hist2dLog.png')
            plt.clf()
            plt.close()
        except:
            print('no 2d hists produced!')
            print('the following inputs were tried...')
            print('predicted LR:')
            print(predLr)
            print('true LR:')
            print(p1/p0)

    with open(f'{label}/performance.yml','w') as f:
        f.write(yaml.dump(performance))
    

def lrMeanPlot(
    ax: Axes, 
    lrhat: np.array, 
    lr: np.array, 
    p0: np.array, 
    nbins: int, 
    threshold: float = 0.01
):
    """
    Plot the mean and std of the true likelihood ratio p(x,z;c1)/p(x,z;c0) in quantiles of the predicted likelihood ratio lrhat(x;c0,c1).

    If the predicted likelihood ratio is perfect, the mean should lie on the y=x line and the std should be zero.
    In any case, the mean of the true likelihood should be plt.close to the predicted likelihood ratio.

    Args:
        ax: The matplotlib Axes to plot on.
        lrhat: The predicted predicted likelihood ratio phat(x;c1)/phat(x;c0) for each event.
        lr: The true likelihood ratio p(x,z;c1)/p(x,z;c0) for each event. 
        p0: The true probability distribution under c0 p(x,z;c0) for each event.
        nbins: The number of quantile bins to use.
        threshold: The quantile threshold to exclude outliers in the decision function.
    """
    qbins = np.quantile(lrhat, np.linspace(threshold, 1 - threshold, nbins + 1))
    
    sumw, _     = np.histogram(lrhat, bins = qbins, weights = p0)
    sumwlr, _   = np.histogram(lrhat, bins = qbins, weights = p0 * lr)
    sumw2lr2, _ = np.histogram(lrhat, bins = qbins, weights = p0**2 * lr**2)
    sumw2lr, _  = np.histogram(lrhat, bins = qbins, weights = p0**2 * lr)
    sumw2, _    = np.histogram(lrhat, bins = qbins, weights = p0**2)
    sumxw, _    = np.histogram(lrhat, bins = qbins, weights = p0 * lrhat)
    
    mean     = sumwlr / sumw
    err_mean = np.sqrt(sumw2lr2 - 2 * sumw2lr * mean + sumw2 * mean**2) / sumw
    
    xcenter  = sumxw / sumw 
    xerr = abs(np.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))
    
    mbins = 0.5 * (qbins[1:] + qbins[:-1])
    handle = ax.errorbar(
        xcenter, mean, xerr=xerr, yerr=err_mean, fmt='.',  markersize=10, capsize=6, linewidth=3
    )

    chiSquare = (xcenter - mean)**2/(lr.max() - lr.min())**2
    chiSquare = chiSquare[~np.isnan(chiSquare)]
    chiSquare = (chiSquare.sum()).item()
    
    ax.plot([qbins[0], qbins[-1]], [qbins[0], qbins[-1]], color = 'grey', linestyle = '--', label = '_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=137.64, com=13, ax=ax)
    ax.set_xlabel(r'$\hat{\overline{r}}\,(x;c_0,c_1)\cdot p(x,z;c_0)$')
    ax.set_ylabel(r'$\overline{r}\,(x,z;c_0,c_1)\cdot p(x,z;c_0)$')

    return chiSquare

def sMeanPlot(
    ax: Axes, 
    shat: np.array, 
    p0: np.array, 
    p1: np.array, 
    nbins: int, 
    threshold: float=0.01
):
    """
    Plot the mean and std of the true decision function s(x,z;c0,c1) in quantiles of the predicted decision function shat(x;c0,c1).

    If the predicted decision function is perfect, the mean should lie on the y=x line and the std should be zero.
    In any case, the mean of the true decision function should be plt.close to the predicted decision function.

    Args:
        ax: The matplotlib Axes to plot on.
        shat: The predicted  decision function shat(x;c0,c1) for each event.
        p0: The true probability distribution under c0 p(x,z;c0) for each event.
        p1: The true probability distribution under c1 p(x,z;c1) for each event.
        nbins: The number of quantile bins to use.
        threshold: The quantile threshold to exclude outliers in the decision function.
    """
    qbins = np.quantile(shat, np.linspace(threshold, 1 - threshold, nbins + 1))

    s = p1/(p1 + p0)
    w = p1 + p0
    
    sumw, _     = np.histogram(shat, bins = qbins, weights = w)
    sumws, _    = np.histogram(shat, bins = qbins, weights = w * s)
    sumw2s2, _  = np.histogram(shat, bins = qbins, weights = w**2 * s**2)
    sumw2s, _   = np.histogram(shat, bins = qbins, weights = w**2 * s)
    sumw2, _    = np.histogram(shat, bins = qbins, weights = w**2)
    sumxw, _    = np.histogram(shat, bins = qbins, weights = w * shat)
    
    mean     = sumws / sumw
    err_mean = np.sqrt(sumw2s2 - 2 * sumw2s * mean + sumw2 * mean**2) / sumw
    
    xcenter  = sumxw / sumw 
    xerr = abs(np.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))
    
    mbins = 0.5 * (qbins[1:] + qbins[:-1])
    handle = ax.errorbar(
        xcenter, mean, xerr=xerr, yerr=err_mean, fmt='.', markersize=10, capsize=6, linewidth=3
    )
    
    chiSquare = (xcenter - mean)**2/(s.max() - s.min())**2
    chiSquare = chiSquare[~np.isnan(chiSquare)]
    chiSquare = (chiSquare.sum()).item()
    
    ax.plot([qbins[0], qbins[-1]], [qbins[0], qbins[-1]], color = 'grey', linestyle = '--', label = '_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=137.64, com=13, ax=ax)
    ax.set_xlabel(r'$\hat{\overline{s}}\,(x;c_0,c_1)\cdot (p(x,z;c_0) + p(x,z;c_1))$')
    ax.set_ylabel(r'$\overline{s}\,(x,z;c_0,c_1)\cdot (p(x,z;c_0) + p(x,z;c_1))$')

    return chiSquare

def compareDistributions(
    ax: Axes, 
    pred: np.array, 
    true: np.array, 
    **plotParams
):
    """
    Scatter plot of a prediction as a function of the truth.

    Args:
        ax: matplotlib axis to plot on
        pred: prediction of the model aiming to approximate the truth
        true: truth values the prediction is aiming to approximate
    """
    if 'fmt' in plotParams.keys():
        plotParams.pop('fmt')
    
    ax.scatter(pred, true, s=1, **plotParams)
    
    amin = min(pred.min().item(), true.min().item())
    amax = max(pred.max().item(), true.max().item())
    
    ax.set_xlim(amin, amax)
    ax.set_ylim(amin, amax)
    ax.plot([0, 1], [0, 1], color = 'grey', linestyle = "--", transform = ax.transAxes, label = '_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=137.64, com=13, ax=ax)
    ax.set_xlabel(r'Learned $\log(\hat{r})$')
    ax.set_ylabel(r'Calculated $\log(\hat{r})$')

def hist2d(
    ax: Axes, 
    pred: np.array, 
    true: np.array, 
    weights: np.array, 
    logbins: bool=False, 
    **plotParams
):
    """
    Weighted 2D histogram of a prediction as a function of the truth.

    Args:
        ax: matplotlib axis to plot on
        pred: prediction of the model aiming to approximate the truth
        true: truth values the prediction is aiming to approximate
        weights: weights used for plotting
        logbins: bool to use log scale bins
    Returns:
        h: ax.hist2d used for plotting 
    """
    if logbins:
        from matplotlib import colors
        h = ax.hist2d(pred, true, weights=weights, bins=100, norm = colors.LogNorm(), **plotParams)
    else:
        h = ax.hist2d(pred, true, weights=weights, bins=100, **plotParams)
        
    amin = min(pred.min().item(), true.min().item())
    amax = max(pred.max().item(), true.max().item())
    
    ax.set_xlim(amin, amax)
    ax.set_ylim(amin, amax)
    ax.plot([0, 1], [0, 1], color = 'grey', linestyle = "--", transform = ax.transAxes, label = '_nolegend_', linewidth=3)
    mh.cms.label("Preliminary", data=False, lumi=137.64, com=13, ax=ax)
    ax.set_xlabel(r'Learned $\log(\hat{r})$')
    ax.set_ylabel(r'Calculated $\log(\hat{r})$')
    return h