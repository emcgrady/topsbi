from matplotlib.axes import Axes
from matplotlib.ticker import StrMethodFormatter
from matplotlib.animation import FuncAnimation, PillowWriter
from topsbi.tools.buildLikelihood import expand_array
from topsbi.tools.metrics import netEval
from topcoffea.modules.histEFT import HistEFT

import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh

import os, torch, yaml, hist



def animate_plots(plots, outname, fps=5):
    """
    Create an animation from a list of plots.

    Args:
        plots: list of file names for the plots to be included in the animation
        outname: name for saving the animation
        fps: frames per second for the animation
    """
    plots = sorted(p for p in plots if p.endswith('.png'))
    if not plots:
        raise ValueError('animate_plots requires at least one PNG frame')
    fig, ax = plt.subplots()
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.axis('off')
    image = ax.imshow(plt.imread(plots[0]))

    def animate(i):
        image.set_data(plt.imread(plots[i]))
        return (image,)

    anim = FuncAnimation(fig, animate, frames=len(plots), interval=200, blit=True)
    anim.save(outname, writer=PillowWriter(fps=fps))
    plt.close(fig)


def kinematic_histogram(x, params, epoch, learned_lr, true_lr, outname, ylim=None, epoch_title=True, x2=None, log=True, scale_ratio=False):
    '''
    Histogram of learned and true likelihood ratio as a function of a kinematic variable.

    Args:
        x: Kinematic to be plotted
        params: dictionary containing plotting information
        epoch: epoch number for plot title
        learned_lr: learned likelihood ratio for each event
        true_lr: true likelihood ratio for each event
        outname: name for saving the plot
    '''
    learned = hist.Hist(
        hist.axis.Regular(name='learned', 
            label= params['label'],
            bins=params['nbins'],
            start=params['min'],
            stop=params['max']
        )
    )
    correct = hist.Hist(
        hist.axis.Regular(
            name='correct', 
            label= params['label'],
            bins=params['nbins'],
            start=params['min'],
            stop=params['max']
        )
    )
    
    learned.fill(x, weight=learned_lr)
    correct.fill(x, weight=true_lr)
    
    mh.style.use("CMS")
    ax   = []
    fig  = plt.figure()
    grid = fig.add_gridspec(2, 1, hspace=0.1, height_ratios=[5, 1])
    ax  += [fig.add_subplot(grid[0])]
    ax  += [fig.add_subplot(grid[1], sharex=ax[0])]
    if epoch_title:
        fig.suptitle(f'Epoch {epoch:04d}')
    else:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax[0])

    n_learned, bins = learned.to_numpy()
    n_correct, _    = correct.to_numpy()

    lErr = []
    cErr = []
    for i in range(params['nbins']):
        lErr.append((learned_lr[ (x > bins[i]) & (x < bins[i+1])]**2).sum()) 
        cErr.append((true_lr[(x > bins[i]) & (x < bins[i+1])]**2).sum())
    lErr = np.sqrt(lErr)
    cErr = np.sqrt(cErr)

    correct.plot1d(ax=ax[0], yerr=cErr, label=r'$\hat{r}\,(x;\theta_1,\theta_0)$')
    learned.plot1d(ax=ax[0], yerr=lErr, linestyle='dashdot', label=r'$r\,(x,z;\theta_1,\theta_0)$')

    ratio = np.divide(n_learned, n_correct, where=(n_correct > 0), out=np.zeros(n_learned.shape))
    rErr = ratio * np.sqrt(np.divide(lErr, n_learned, where=(n_learned > 0), out=np.zeros(lErr.shape))**2 + np.divide(cErr, n_correct, where=(n_correct > 0), out=np.zeros(n_correct.shape))**2)

    ax[1].plot((bins[:-1] + bins[1:])/2, ratio, '.k') 
    ax[1].errorbar((bins[:-1] + bins[1:])/2, ratio, yerr=rErr, fmt='none', color='k')
    ax[1].hlines(1, bins[0], bins[-1], color='k', linestyle='dashed', alpha=0.5)
    if scale_ratio:
        bin_mask = n_correct >= 1
        err_max = np.array([((ratio + rErr - 1)[bin_mask]).max(), ((1 - ratio + rErr)[bin_mask]).max()]).max() * 1.05
        ax[1].set_ylim([1 - err_max, 1 + err_max])
    else:
        ax[1].set_ylim(0,2)
    ax[0].set_xlabel('')
    ax[0].set_xlim(bins[0], bins[-1])
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[1].set_xlabel(params['label'])
    ax[1].set_ylabel(r'$\frac{\hat{r}(x;\theta_1,\theta_0)}{r(x,z;\theta_1,\theta_0)}$')
    ax[0].legend()
    if log:
        ax[0].set_yscale('log')
    if ylim is None:
        ylim = ax[0].get_ylim()
        fig.savefig(outname, bbox_inches='tight')
        plt.close(fig)
        return ylim
    else:
        ax[0].set_ylim(ylim)
        fig.savefig(outname, bbox_inches='tight')
        plt.close(fig)

def kinematic_ratio_plot(
    x: np.array, 
    dlr: np.array, 
    plr: np.array, 
    true_lr: np.array, 
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
    mh.style.use("CMS")
    ax   = []
    fig  = plt.figure()
    grid = fig.add_gridspec(2, 1, hspace=0.05, height_ratios=[5, 1])
    ax  += [fig.add_subplot(grid[0])]
    ax  += [fig.add_subplot(grid[1], sharex=ax[0])]
    
    plt.setp(ax[0].get_xticklabels(), visible=False)

    #initialize the histograms
    correct_hist = hist.Hist(
        hist.axis.Regular(
            name='correct',
            label=params['label'],
            bins=params['nbins'] - 1,
            start=params['min'],
            stop=params['max']
            )
            )
    dedicated_hist = hist.Hist(
        hist.axis.Regular(
            name='dedicated',
            label=params['label'],
            bins=params['nbins'] - 1,
            start=params['min'],
            stop=params['max']
            )
            )
    parametric_hist = hist.Hist(
        hist.axis.Regular(
            name='parametric',
            label=params['label'],
            bins=params['nbins'] - 1,
            start=params['min'],
            stop=params['max']
            )
            )

    correct_hist.fill(correct=x, weight=true_lr)
    dedicated_hist.fill(dedicated=x, weight=dlr)
    parametric_hist.fill(parametric=x, weight=plr)

    #calculate error
    cNum, bins = correct_hist.to_numpy()
    dNum = dedicated_hist.values()
    pNum = parametric_hist.values()
    cErr = []
    dErr = []
    pErr = []

    for i in range(params['nbins'] - 1):
        cErr.append((true_lr[(x >= bins[i]) & (x < bins[i+1])]**2).sum())
        dErr.append((dlr[(x >= bins[i]) & (x < bins[i+1])]**2).sum())
        pErr.append((plr[(x >= bins[i]) & (x < bins[i+1])]**2).sum())
    cErr = np.sqrt(np.hstack(cErr))
    dErr = np.sqrt(np.hstack(dErr))
    pErr = np.sqrt(np.hstack(pErr))

    #plot the histograms
    correct_hist.plot1d(ax=ax[0], yerr=cErr, label=f'Correct ({params["wc_point"]})')
    dedicated_hist.plot1d(ax=ax[0],  yerr=dErr, label='Dedicated',  linestyle='dashdot', color='orange')
    parametric_hist.plot1d(ax=ax[0], yerr=pErr, label='Parametric', linestyle='dashed',  color='green')

    #plot the ratio and ratio errors
    ax[1].hlines(1, bins[0], bins[-1], color='k', linestyle='dashed')
    rBins = np.diff(bins)/2+bins[:-1]
    dVals = np.divide(cNum, dNum, where=dNum!=0, out=np.zeros(cNum.shape))
    pVals = np.divide(cNum, pNum, where=pNum!=0, out=np.zeros(cNum.shape))
    
    cRatio = np.divide(cErr, cNum, where=cNum!=0, out=np.zeros(cNum.shape))
    dRatio = np.divide(dErr, dNum, where=dNum!=0, out=np.zeros(dNum.shape))
    pRatio = np.divide(pErr, pNum, where=pNum!=0, out=np.zeros(pNum.shape))
    
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
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax[0])
    if params['plotLog']:
        ax[0].set_yscale('log')
    ax[0].set_xlabel('') 
    ax[0].set_ylabel('counts')
    ax[1].set_xlabel(params['label']) 
    ax[1].set_ylabel('ratio')
    ax[0].set_xlim(params['min'], params['max'])
    ax[0].legend()
    if params['outname']:
        fig.savefig(f'{params["outname"]}', bbox_inches='tight')
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

def lr_curve(
    ax: Axes,
    lr_history: list[float],
) -> None:
    ax.plot(lr_history, linewidth=3, color='tab:orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')

def networkPlots(features, p0, p1, pg, net, train_loss, test_loss, label, method='stitched', lr_history=None):
    mh.style.use("CMS")
    os.makedirs(f'{label}', mode=0o755, exist_ok=True)
    torch.save(net.state_dict(), f'{label}/model.pt')

    performance = {}
    f  = net(features).cpu().detach().numpy().flatten()
    p0 = p0.detach().cpu().numpy().flatten() 
    p1 = p1.detach().cpu().numpy().flatten() 
    pg = pg.detach().cpu().numpy().flatten() 

    
    #convert tensors to np arrays
    if (method == 'stitched') or (method == 'alice') or (method == 'weights_only'):
        noOnes = f != 1
        f      = f[noOnes]
        p0     = p0[noOnes]
        p1     = p1[noOnes]
        pg     = pg[noOnes]
        lrhat = f/(1-f)
    elif method == 'weight_shift':
        f      = 2 * f - 1
        lrhat = np.exp(f)

    lr = p1/p0
    lr_norm_factor = np.ones(lr.shape)

    #loss curves
    fig, ax = plt.subplots()
    loss_curve(ax, train_loss, test_loss)
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    fig.savefig(f'{label}/loss.png', bbox_inches='tight')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lossLog.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    if lr_history is not None:
        fig, ax = plt.subplots()
        lr_curve(ax, lr_history)
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
        fig.savefig(f'{label}/lr.png', bbox_inches='tight')
        plt.clf()
        plt.close()

    #get network performance metrics
    fpr, tpr, auc = netEval(f, p0, p1)
    
    performance['auc'] =  auc

    #make ROC curves
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='Network Performance', linewidth=3)
    ax.plot([0,1],[0,1], ':', label='Baseline', linewidth=3)
    ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    fig.savefig(f'{label}/roc.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    # lr calibration difference 
    fig, ax = plt.subplots()
    lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 20)
    ax.legend()
    fig.savefig(f'{label}/mean_diff_excl.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 20, threshold=0.)
    ax.legend()
    ax.set_xscale('log')
    fig.savefig(f'{label}/mean_diff_incl.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 20, include_stdv=True)
    ax.legend()
    fig.savefig(f'{label}/std_diff_excl.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 20, threshold=0., include_stdv=True)
    ax.legend()
    ax.set_xscale('log')
    fig.savefig(f'{label}/std_diff_incl.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    delta_sig_excl, ci_ratio_excl = lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 1000)
    ax.legend()
    fig.savefig(f'{label}/hibin_mean_diff_excl.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    performance['delta_sig_excl'] = delta_sig_excl
    performance['ci_ratio_excl']  = ci_ratio_excl

    fig, ax = plt.subplots()
    delta_sig_incl, ci_ratio_incl = lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 1000, threshold=0.)
    ax.legend()
    ax.set_xscale('log')
    fig.savefig(f'{label}/hibin_mean_diff_incl.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    performance['delta_sig_incl'] = delta_sig_incl
    performance['ci_ratio_incl']  = ci_ratio_incl

    # lr calibration curves
    fig, ax = plt.subplots()
    performance['chi_excl'] = lr_calibration_curve(ax, lr, lrhat, lr_norm_factor, 1000)
    fig.savefig(f'{label}/lr_excl.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lrExclLog.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    performance['chi_incl'] = lr_calibration_curve(ax, lr, lrhat, lr_norm_factor, 1000, threshold=0.)
    fig.savefig(f'{label}/lrIncl.png', bbox_inches='tight')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lr_incl_log.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    lr_calibration_curve(ax, lr, lrhat, lr_norm_factor, 50)
    fig.savefig(f'{label}/lr_excl_lobin.png', bbox_inches='tight')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lr_excl_log_lobin.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    lr_calibration_curve(ax, lr, lrhat, lr_norm_factor, 50, threshold=0.)
    fig.savefig(f'{label}/lr_incl_lobin.png', bbox_inches='tight')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lr_incl_log_lobin.png', bbox_inches='tight')
    plt.clf()
    plt.close()

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

    # remove negative weights from early training
    err_mean[err_mean < 0] = 0
    
    xcenter  = sumxw / sumw 
    xerr = abs(np.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))
    
    mbins = 0.5 * (qbins[1:] + qbins[:-1])
    ax.errorbar(xcenter, mean, xerr=xerr, yerr=err_mean, fmt='.')

    chiSquare = (xcenter - mean)**2#/(lr.max() - lr.min())**2
    chiSquare = chiSquare[~np.isnan(chiSquare)]
    chiSquare = (chiSquare.sum()).item()
    
    ax.plot([qbins[0], qbins[-1]], [qbins[0], qbins[-1]], color = 'grey', linestyle = '--', label = '_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\hat{\overline{r}}\,(x;\theta_1,\theta_0)$')
    ax.set_ylabel(r'$\overline{r}\,(x,z;\theta_1,\theta_0)$')

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
    sumws, _    = np.histogram(shat, bins = qbins, weights = w*s)
    sumw2s2, _  = np.histogram(shat, bins = qbins, weights = w**2 * s**2)
    sumw2s, _   = np.histogram(shat, bins = qbins, weights = w**2 * s)
    sumw2, _    = np.histogram(shat, bins = qbins, weights = w**2)
    sumxw, _    = np.histogram(shat, bins = qbins, weights = w*shat)
    
    mean     = sumws / sumw
    err_mean = np.sqrt(sumw2s2 - 2 * sumw2s * mean + sumw2 * mean**2) / sumw

    # remove negative weights from early training
    err_mean[err_mean < 0] = 0
    
    xcenter  = sumxw / sumw 
    xerr = abs(np.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))
    
    mbins = 0.5 * (qbins[1:] + qbins[:-1])
    ax.errorbar(xcenter, mean, xerr=xerr, yerr=err_mean, fmt='.')
    
    chiSquare = (xcenter - mean)**2#/(s.max() - s.min())**2
    chiSquare = chiSquare[~np.isnan(chiSquare)]
    chiSquare = (chiSquare.sum()).item()
    
    ax.plot([qbins[0], qbins[-1]], [qbins[0], qbins[-1]], color = 'grey', linestyle = '--', label = '_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\hat{\overline{s}}\,(x;\theta_1,\theta_0)$')
    ax.set_ylabel(r'$\overline{s}\,(x,z;\theta_1,\theta_0)$')

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
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
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
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\log[\hat{r}(x;\theta_1,\theta_0)]$')
    ax.set_ylabel(r'$\log[r(x,z:\theta_1,\theta_0)]$')
    return h

def f_calibration_curve(
    ax: Axes, 
    f: np.array, 
    fhat: np.array,
    f_norm_factor: np.array,
    nbins: int, 
    threshold: float=0.01
 ):
    qbins = np.quantile(fhat, np.linspace(threshold, 1 - threshold, nbins + 1))

    sumw, _      = np.histogram(fhat, bins=qbins, weights=f_norm_factor)
    sumwf, _     = np.histogram(fhat, bins=qbins, weights=f_norm_factor * f)
    sumw2f2, _   = np.histogram(fhat, bins=qbins, weights=f_norm_factor**2 * f**2)
    sumw2f, _    = np.histogram(fhat, bins=qbins, weights=f_norm_factor**2 * f)
    sumw2, _     = np.histogram(fhat, bins=qbins, weights=f_norm_factor**2)
    sumwfhat, _  = np.histogram(fhat, bins=qbins, weights=f_norm_factor * fhat)

    mean = sumwf/sumw
    err_mean = np.sqrt(sumw2f2 - 2*sumw2f * mean + sumw2 * mean**2)/sumw

    xcenter = sumwfhat/sumw
    xerr = abs(np.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))

    mbins = 0.5 * (qbins[1:] + qbins[:-1])
    ax.errorbar(xcenter, mean, xerr=xerr, yerr=err_mean, fmt='.')

    chi_square = (xcenter - mean)**2/(f.max() - f.min())**2
    chi_square = chi_square[~np.isnan(chi_square)]
    chi_square = (chi_square.sum()).item()

    ax.plot([qbins[0], qbins[-1]], [qbins[0], qbins[-1]], color = 'grey', linestyle = '--', label = '_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\left\langle\hat{{f}}\,(x;\theta_1,\theta_0)\right\rangle$')
    ax.set_ylabel(r'$\left\langle{f}\,(x,z;\theta_1,\theta_0)\right\rangle$')

    return chi_square

def lr_calibration_curve(
    ax: Axes,
    lr: np.array,
    lrhat: np.array,
    lr_norm_factor: np.array,
    nbins: int,
    threshold: float=0.01
):
    qbins = np.quantile(lrhat, np.linspace(threshold, 1 - threshold, nbins + 1))

    sumw, _      = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor)
    sumwlr, _    = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * lr)
    sumw2lr2, _  = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2 * lr**2)
    sumw2lr, _   = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2 * lr)
    sumw2, _     = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2)
    sumwlrhat, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * lrhat)

    mean = sumwlr / sumw
    err_mean = np.sqrt(sumw2lr2 - 2*sumw2lr*mean + sumw2*mean**2) / sumw

    xcenter = sumwlrhat / sumw
    xerr = abs(np.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))

    mbins = 0.5 * (qbins[1:] + qbins[:-1])
    ax.errorbar(xcenter, mean, xerr=xerr, yerr=err_mean, fmt='.')

    chi_square = (xcenter - mean)**2/mean
    chi_square = chi_square[~np.isnan(chi_square)]
    chi_square = (chi_square.sum()).item()

    ax.plot([qbins[0], qbins[-1]], [qbins[0], qbins[-1]], color = 'grey', linestyle = '--', label = '_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\left\langle\hat{{r}}\,(x;\theta_1,\theta_0)\right\rangle$')
    ax.set_ylabel(r'$\left\langle{r}\,(x,z;\theta_1,\theta_0)\right\rangle$')

    return chi_square

def kinematic_shift_histogram(x0, x1, weights_0, weights_1, params, epoch, lr, outname, ylim=None, epoch_title=True, log=True, scale_ratio=False):
    '''
    Histogram of learned likelihood ratio agsinst a true distribution as a function of a kinematic variable.

    Args:
        x: Kinematic to be plotted
        params: dictionary containing plotting information
        epoch: epoch number for plot title
        learned_lr: learned likelihood ratio for each event
        true_lr: true likelihood ratio for each event
        outname: name for saving the plot
    '''
    if type(x0) == torch.Tensor:
        x0 = x0.cpu().detach().numpy()
    if type(x1) == torch.Tensor:
        x1 = x1.cpu().detach().numpy()
    if type(weights_0) == torch.Tensor:
        weights_0 = weights_0.cpu().detach().numpy()
    if type(weights_1) == torch.Tensor:
        weights_1 = weights_1.cpu().detach().numpy()
    learned = hist.Hist(
        hist.axis.Regular(
            name='learned', 
            label= params['label'],
            bins=params['nbins'],
            start=params['min'],
            stop=params['max']
        )
    )
    correct = hist.Hist(
        hist.axis.Regular(
            name='correct', 
            label= params['label'],
            bins=params['nbins'],
            start=params['min'],
            stop=params['max']
        )
    )

    nominal =  hist.Hist(
        hist.axis.Regular(
            name='nominal', 
            label= params['label'],
            bins=params['nbins'],
            start=params['min'],
            stop=params['max']
        )
    )
    
    weights_0 = weights_0.flatten()
    weights_1 = weights_1.flatten()
    lr = lr.flatten()

    nominal.fill(x0, weight=weights_0)
    learned.fill(x0, weight=lr*weights_0)
    correct.fill(x1, weight=weights_1)
    
    mh.style.use("CMS")
    ax   = []
    fig  = plt.figure()
    grid = fig.add_gridspec(2, 1, hspace=0.1, height_ratios=[5, 1])
    ax  += [fig.add_subplot(grid[0])]
    ax  += [fig.add_subplot(grid[1], sharex=ax[0])]
    if epoch_title:
        fig.suptitle(f'Epoch {epoch:04d}')
    else:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax[0])

    n_learned, bins = learned.to_numpy()
    n_correct, _    = correct.to_numpy()

    lErr = []
    cErr = []
    for i in range(params['nbins']):
        lErr.append((lr[ (x0 > bins[i]) & (x0 < bins[i+1])]**2).sum()) 
        cErr.append(np.array((x1 > bins[i]) & (x1 < bins[i+1])).sum())
    lErr = np.sqrt(lErr)
    cErr = np.sqrt(cErr)

    nominal.plot1d(ax=ax[0], linestyle='dashed', color='k', label=r'$r\,(x,z;\theta,0)$')
    correct.plot1d(ax=ax[0], yerr=cErr, color='C0', label=r'$r\,(x,z;\theta,\nu)$',)
    learned.plot1d(ax=ax[0], yerr=lErr, linestyle='dashdot', color='C1', label=r'$\hat{r}\,(x;\theta,\nu)$', )  

    ratio = np.divide(n_learned, n_correct, where=((n_correct > 0) & (n_learned > 0)), out=np.zeros(n_learned.shape))
    rErr = ratio * np.sqrt(np.divide(lErr, n_learned, where=((n_correct > 0) & (n_learned > 0)), out=np.zeros(lErr.shape))**2 + np.divide(cErr, n_correct, where=((n_correct > 0) & (n_learned > 0)), out=np.zeros(n_correct.shape))**2)

    ax[1].plot((bins[:-1] + bins[1:])/2, ratio, '.k') 
    ax[1].errorbar((bins[:-1] + bins[1:])/2, ratio, yerr=rErr, fmt='none', color='k')
    ax[1].hlines(1, bins[0], bins[-1], color='k', linestyle='dashed', alpha=0.5)

    if scale_ratio:
        bin_mask = n_correct >= 1
        err_max = np.array([((ratio + rErr - 1)[bin_mask]).max(), ((1 - ratio + rErr)[bin_mask]).max()]).max() * 1.05
        ax[1].set_ylim([1 - err_max, 1 + err_max])
    else:
        ax[1].set_ylim(0,2)
    ax[0].set_xlabel('')
    ax[0].set_xlim(bins[0], bins[-1])
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[1].set_xlabel(params['label'])
    ax[1].set_ylabel(r'$\frac{\hat{r}(x;\theta,\nu)}{r(x,z;\theta,\nu)}$')
    ax[0].set_ylabel('Number of Events')
    ax[0].legend()
    if log:
        ax[0].set_yscale('log')
    if ylim is None:
        ylim = ax[0].get_ylim()
        fig.savefig(outname, bbox_inches='tight')
        plt.close(fig)
        return ylim
    else:
        ax[0].set_ylim(ylim)
        fig.savefig(outname, bbox_inches='tight')
        plt.close(fig)

def lr_calibration_difference(
    ax: Axes,
    lr: np.array,
    lrhat: np.array,
    lr_norm_factor: np.array,
    nbins: int,
    threshold: float=0.01,
    include_stdv: bool=False,
):
    qbins = np.quantile(lrhat, np.linspace(threshold, 1 - threshold, nbins + 1))

    sumw, _         = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor)
    sumwlr, _       = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * lr)
    sumw2, _        = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2)
    sumwlrhat, _    = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * lrhat)
    sumwlrdiff, _   = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * (lr - lrhat))
    sumw2lrdiff, _  = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2 * (lr - lrhat))
    sumw2lrdiff2, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2 * (lr - lrhat)**2)

    meanlr     = sumwlr / sumw 
    meanlrhat =  sumwlrhat / sumw
    meanlrdiff = sumwlrdiff / sumw

    mean_err  = np.sqrt(sumw2lrdiff2 + sumw2*meanlrdiff**2 - 2*meanlrdiff*sumw2lrdiff) / sumw
    err_width = 2*mean_err
    err_down  = (meanlr - meanlrhat) - mean_err

    mean_stdv  = mean_err * np.sqrt(sumw2)
    stdv_width = 2*mean_stdv
    stdv_down  = (meanlr - meanlrhat) - mean_stdv

    differ = meanlr - meanlrhat


    mean_error_formatting = {
        'align': 'edge',
        'facecolor': "#4086B8", 
        'linewidth': 0,
        'label': r'$\sigma_{\langle r  -\hat{r}\rangle}$',
        'alpha': 0.6,
    }

    mean_stdv_formatting = {
        'align': 'edge', 
        'facecolor': "#E6EEE6", 
        'linewidth': 0, 
        'label': r'$\sigma_{ r  -\hat{r}}$', 
    }

    if include_stdv:
        ax.bar(qbins[:-1], stdv_width, bottom=stdv_down, width=(qbins[1:] - qbins[:-1]), **mean_stdv_formatting)
    ax.plot([qbins[0], qbins[-1]], [0,0], color = 'grey', linestyle = ':', label = '_nolegend_')
    ax.bar(qbins[:-1], err_width, bottom=err_down, width=(qbins[1:] - qbins[:-1]),  **mean_error_formatting)
    
    ax.hist(meanlrhat, weights=differ, bins=qbins, histtype='step', color='k', label=r'$r - \hat{r}$')

    sign = np.sign(differ)
    delta_sig = []
    ci_ratio = []
    for sigma in range(1,4):
        pos = (differ + sigma*mean_err) * sign
        neg = (differ - sigma*mean_err) * sign
        temp_sig = (np.vstack((pos, neg)) / mean_err).min(0)
        delta_sig += [temp_sig.mean().item()]
        ci_ratio += [(temp_sig < 0).mean().item()]

    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\left\langle\hat{r}\,(x;\theta_1,\theta_0)\right\rangle$')
    ax.set_ylabel(r'$\left\langle{r}\,(x,z;\theta_1,\theta_0)\right\rangle - \left\langle\hat{{r}}\,(x;\theta_1,\theta_0)\right\rangle$')
    y_max = abs(np.array(ax.get_ylim())).max()*1.2
    ax.set_ylim([-y_max, y_max])
    ax.set_xlim([qbins[0], qbins[-1]])

    return delta_sig, ci_ratio