from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from mpl_toolkits.axes_grid1 import make_axes_locatable
from topsbi.tools.metrics import netEval, group_closure

import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh

import logging, os, re, torch, yaml, hist


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

    anim = FuncAnimation(fig, animate, frames=len(plots), interval=1000, blit=True)
    anim.save(outname, writer=PillowWriter(fps=fps))
    plt.close(fig)


def merge_sparse_bins(x: np.array, weights: np.array, edges: np.array, min_neff: float = 30):
    '''
    Merge neighbouring bins until each holds at least min_neff effective events.

    Bins are accumulated from the left; a remainder at the right edge that is still below
    min_neff is folded into the last complete bin, so the tail ends up in one wide bin.

    Args:
        x: values of the binned variable
        weights: event weights used to count effective events, n_eff = (sum w)^2 / sum w^2
        edges: initial (fine) bin edges
        min_neff: minimum effective number of events per merged bin
    Returns:
        merged bin edges
    '''
    sumw, _ = np.histogram(x, bins=edges, weights=weights)
    sumw2, _ = np.histogram(x, bins=edges, weights=weights**2)
    merged = [edges[0]]
    acc, acc2 = 0.0, 0.0
    for i in range(len(sumw)):
        acc += sumw[i]
        acc2 += sumw2[i]
        if acc2 > 0 and acc**2 / acc2 >= min_neff:
            merged.append(edges[i + 1])
            acc, acc2 = 0.0, 0.0
    if merged[-1] != edges[-1]:
        if len(merged) > 1:
            merged[-1] = edges[-1]
        else:
            merged.append(edges[-1])
    return np.array(merged)


def kinematic_histogram(
    x: np.array,
    params: dict,
    epoch: int,
    learned_lr: np.array,
    true_lr: np.array,
    outname: str,
    ylim: list = None,
    epoch_title: bool = True,
    log: bool = True,
    scale_ratio: bool = False,
    min_neff: float = 30,
    include_overflow: bool = True,
):
    '''
    Histogram of learned and true likelihood ratio as a function of a kinematic variable.

    The regular binning from params is merged in sparse regions so that every bin has at least
    min_neff effective events, and the main panel shows yields per unit of the variable.

    Args:
        x: Kinematic to be plotted
        params: dictionary containing plotting information
        epoch: epoch number for plot title
        learned_lr: learned likelihood ratio for each event (times the theta_0 weight)
        true_lr: true likelihood ratio for each event (times the theta_0 weight)
        outname: name for saving the plot
        min_neff: minimum effective number of events per bin
        include_overflow: put events outside [min, max] into the first and last bins
    '''
    x = np.asarray(x, dtype=np.float64)
    learned_lr = np.asarray(learned_lr, dtype=np.float64)
    true_lr = np.asarray(true_lr, dtype=np.float64)
    fine = np.linspace(params['min'], params['max'], params['nbins'] + 1)
    if include_overflow:
        # keep the tails in the test: clip into the outermost bins
        x = np.clip(x, fine[0], np.nextafter(fine[-1], fine[0]))
    bins = merge_sparse_bins(x, true_lr, fine, min_neff)
    widths = np.diff(bins)

    n_learned, _ = np.histogram(x, bins=bins, weights=learned_lr)
    n_correct, _ = np.histogram(x, bins=bins, weights=true_lr)
    sum_ll, _ = np.histogram(x, bins=bins, weights=learned_lr**2)
    sum_cc, _ = np.histogram(x, bins=bins, weights=true_lr**2)
    sum_lc, _ = np.histogram(x, bins=bins, weights=learned_lr * true_lr)
    lErr = np.sqrt(sum_ll)
    cErr = np.sqrt(sum_cc)

    mh.style.use("CMS")
    ax = []
    fig = plt.figure()
    grid = fig.add_gridspec(2, 1, hspace=0.1, height_ratios=[5, 1])
    ax += [fig.add_subplot(grid[0])]
    ax += [fig.add_subplot(grid[1], sharex=ax[0])]
    if epoch_title:
        fig.suptitle(f'Epoch {epoch:04d}')
    else:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax[0])

    # bins have different widths, so show yields per unit of the variable
    mh.histplot(n_correct / widths, bins, yerr=cErr / widths, ax=ax[0], label=r'$r\,(x,z;\theta_1,\theta_0)$')
    mh.histplot(
        n_learned / widths,
        bins,
        yerr=lErr / widths,
        ax=ax[0],
        linestyle='dashdot',
        label=r'$\hat{r}\,(x;\theta_1,\theta_0)$',
    )

    ratio = np.divide(n_learned, n_correct, where=(n_correct > 0), out=np.zeros(n_learned.shape))
    rVar = np.maximum(sum_ll - 2 * ratio * sum_lc + ratio**2 * sum_cc, 0)
    rErr = np.divide(np.sqrt(rVar), n_correct, where=(n_correct > 0), out=np.zeros(n_learned.shape))

    # merging leaves sparse bins only when the whole variable has too few effective events
    n_eff = np.divide(n_correct**2, sum_cc, where=(sum_cc > 0), out=np.zeros(n_correct.shape))
    sparce = n_eff < min_neff
    ratio[sparce] = np.nan
    rErr[sparce] = np.nan

    centers = (bins[:-1] + bins[1:]) / 2
    ax[1].errorbar(centers, ratio, xerr=widths / 2, yerr=rErr, fmt='none', color='k')
    ax[1].hlines(1, bins[0], bins[-1], color='k', linestyle='dashed', alpha=0.5)
    if scale_ratio and (~sparce).any():
        bin_mask = ~sparce
        err_max = np.array([((ratio + rErr - 1)[bin_mask]).max(), ((1 - ratio + rErr)[bin_mask]).max()]).max() * 1.05
        ax[1].set_ylim([1 - err_max, 1 + err_max])
    else:
        ax[1].set_ylim(0, 2)
    ax[0].set_xlabel('')
    unit = params.get('unit')
    ax[0].set_ylabel(f'Weighted events / {unit}' if unit else 'Weighted events / bin width')
    ax[0].set_xlim(bins[0], bins[-1])
    plt.setp(ax[0].get_xticklabels(), visible=False)
    ax[1].set_xlabel(params['label'] + (f' [{unit}]' if unit else ''))
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


def kinematic_ratio_plot(x: np.array, dlr: np.array, plr: np.array, true_lr: np.array, **params):
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
    # initialize the figure
    mh.style.use("CMS")
    ax = []
    fig = plt.figure()
    grid = fig.add_gridspec(2, 1, hspace=0.05, height_ratios=[5, 1])
    ax += [fig.add_subplot(grid[0])]
    ax += [fig.add_subplot(grid[1], sharex=ax[0])]

    plt.setp(ax[0].get_xticklabels(), visible=False)

    # initialize the histograms
    correct_hist = hist.Hist(
        hist.axis.Regular(
            name='correct', label=params['label'], bins=params['nbins'] - 1, start=params['min'], stop=params['max']
        )
    )
    dedicated_hist = hist.Hist(
        hist.axis.Regular(
            name='dedicated', label=params['label'], bins=params['nbins'] - 1, start=params['min'], stop=params['max']
        )
    )
    parametric_hist = hist.Hist(
        hist.axis.Regular(
            name='parametric', label=params['label'], bins=params['nbins'] - 1, start=params['min'], stop=params['max']
        )
    )

    correct_hist.fill(correct=x, weight=true_lr)
    dedicated_hist.fill(dedicated=x, weight=dlr)
    parametric_hist.fill(parametric=x, weight=plr)

    # calculate error
    cNum, bins = correct_hist.to_numpy()
    dNum = dedicated_hist.values()
    pNum = parametric_hist.values()
    cErr = []
    dErr = []
    pErr = []

    for i in range(params['nbins'] - 1):
        cErr.append((true_lr[(x >= bins[i]) & (x < bins[i + 1])] ** 2).sum())
        dErr.append((dlr[(x >= bins[i]) & (x < bins[i + 1])] ** 2).sum())
        pErr.append((plr[(x >= bins[i]) & (x < bins[i + 1])] ** 2).sum())
    cErr = np.sqrt(np.hstack(cErr))
    dErr = np.sqrt(np.hstack(dErr))
    pErr = np.sqrt(np.hstack(pErr))

    # plot the histograms
    correct_hist.plot1d(ax=ax[0], yerr=cErr, label=f'Correct ({params["wc_point"]})')
    dedicated_hist.plot1d(ax=ax[0], yerr=dErr, label='Dedicated', linestyle='dashdot', color='orange')
    parametric_hist.plot1d(ax=ax[0], yerr=pErr, label='Parametric', linestyle='dashed', color='green')

    # plot the ratio and ratio errors
    ax[1].hlines(1, bins[0], bins[-1], color='k', linestyle='dashed')
    rBins = np.diff(bins) / 2 + bins[:-1]
    dVals = np.divide(cNum, dNum, where=dNum != 0, out=np.zeros(cNum.shape))
    pVals = np.divide(cNum, pNum, where=pNum != 0, out=np.zeros(cNum.shape))

    cRatio = np.divide(cErr, cNum, where=cNum != 0, out=np.zeros(cNum.shape))
    dRatio = np.divide(dErr, dNum, where=dNum != 0, out=np.zeros(dNum.shape))
    pRatio = np.divide(pErr, pNum, where=pNum != 0, out=np.zeros(pNum.shape))

    ax[1].bar(
        rBins,
        2 * np.sqrt((cRatio + dRatio) * dVals),
        width=np.diff(bins),
        bottom=dVals - np.sqrt((cRatio + dRatio) * dVals),
        edgecolor='orange',
        lw=0,
        hatch='//',
        hatch_linewidth=0.8,
        color='none',
        label='Dedicated Uncertainty',
    )
    ax[1].bar(
        rBins,
        2 * np.sqrt((cRatio + pRatio) * pVals),
        width=np.diff(bins),
        bottom=pVals - np.sqrt((cRatio + pRatio) * pVals),
        edgecolor='green',
        lw=0,
        hatch='\\\\',
        hatch_linewidth=0.8,
        color='none',
        label='Parametric Uncertainty',
    )
    ax[1].plot(rBins, dVals, '^', label='Dedicated', color='orange')
    ax[1].plot(rBins, pVals, 'v', label='Parametric', color='green')
    ax[1].set_ylim([0, 2])

    # clean up formatting
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


def loss_curve(ax: Axes, train_loss: list[float], test_loss: list[float]) -> None:
    """
    Plot training and test loss over epochs.

    Args:
        ax: The matplotlib Axes to plot on.
        train_loss: loss for training data
        test_loss: loss for testing data
    """
    ax.plot(train_loss, label="Training dataset", linewidth=3)
    ax.plot(test_loss, label="Testing dataset", linewidth=3)
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


def networkPlots(
    features,
    p0,
    p1,
    net,
    train_loss,
    test_loss,
    label,
    method='stitched',
    lr_history=None,
    group_cols=(21, 19),
    group_caps={19: 4},
    group_labels={21: ('', {0: '2016APV', 1: '2016', 2: '2017', 3: '2018'}), 19: ('jets', {4: '≥4'})},
):
    mh.style.use("CMS")
    os.makedirs(f'{label}', mode=0o755, exist_ok=True)
    torch.save(net.state_dict(), f'{label}/model.pt')

    performance = {}
    f = net(features).cpu().detach().numpy().flatten()
    p0 = p0.detach().cpu().numpy().flatten()
    p1 = p1.detach().cpu().numpy().flatten()

    # convert tensors to np arrays
    if (method == 'stitched') or (method == 'alice') or (method == 'weights_only'):
        noOnes = f != 1
        f = f[noOnes]
        p0 = p0[noOnes]
        p1 = p1[noOnes]
        lrhat = f / (1 - f)
    elif method == 'weight_shift':
        f = 2 * f - 1
        lrhat = np.exp(f)

    lr = p1 / p0
    lr_norm_factor = p0 / p0.mean()

    # closure groups must come from network inputs: era (21) and njets (19, >= 4 merged) by default
    feats_np = features.detach().cpu().numpy()
    groups = np.column_stack(
        [np.minimum(feats_np[:, c], group_caps[c]) if c in group_caps else feats_np[:, c] for c in group_cols]
    )
    if method != 'weight_shift':
        groups = groups[noOnes]

    # loss curves
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

    # get network performance metrics
    fpr, tpr, auc = netEval(f, p0, p1)

    performance['auc'] = round(auc, 5)

    # make ROC curves
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='Network Performance', linewidth=3)
    ax.plot([0, 1], [0, 1], ':', label='Baseline', linewidth=3)
    ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    fig.savefig(f'{label}/roc.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    # lr calibration difference
    fig, ax = plt.subplots()
    lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 20, include_stdv=True)
    ax.legend()
    fig.savefig(f'{label}/std_diff_excl.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 20, threshold=0.0, include_stdv=True)
    ax.legend()
    fig.savefig(f'{label}/std_diff_incl.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 20)
    ax.legend()
    fig.savefig(f'{label}/mean_diff_excl.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 20, threshold=0.0)
    ax.legend()
    fig.savefig(f'{label}/mean_diff_incl.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    # get metrics with higher bins
    fig, ax = plt.subplots()
    pull, pull_err, ci_diff, ci_diff_err = lr_calibration_difference(ax, lr, lrhat, lr_norm_factor, 50, threshold=0.0)
    plt.clf()
    plt.close()

    performance['pull'] = round(pull, 5)
    performance['pull_err'] = round(pull_err, 5)
    performance['ci_diff'] = [round(v, 5) for v in ci_diff]
    performance['ci_diff_err'] = [round(v, 5) for v in ci_diff_err]

    # lr calibration curves
    fig, ax = plt.subplots()
    lr_calibration_curve(ax, lr, lrhat, lr_norm_factor, 50)
    fig.savefig(f'{label}/lr_excl.png', bbox_inches='tight')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lr_excl_log.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots()
    performance['rms_bias'] = round(lr_calibration_curve(ax, lr, lrhat, lr_norm_factor, 50, threshold=0.0), 5)
    fig.savefig(f'{label}/lr_incl.png', bbox_inches='tight')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{label}/lr_incl_log.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    # group closure test
    closure = group_closure(lrhat, lr, lr_norm_factor, groups)
    performance['group_chi2_excess'] = round(closure['chi2_excess'], 5)
    performance['group_chi2_excess_err'] = round(closure['chi2_excess_err'], 5)
    performance['group_rms_bias'] = round(closure['rms_bias'], 5)
    performance['group_worst_pull'] = round(closure['worst_pull'], 3)

    fig, ax = plt.subplots()
    ax.axhspan(-2, 2, color='#E6EEE6', label=r'$\pm 2\sigma$')
    ax.axhspan(-1, 1, color='#4086B8', alpha=0.6, label=r'$\pm 1\sigma$')
    ax.plot(closure['pulls'], '.k')
    cells = closure['cells']
    starts = [i for i in range(len(cells)) if i == 0 or cells[i][0] != cells[i - 1][0]] + [len(cells)]
    for s in starts[1:-1]:
        ax.axvline(s - 0.5, color='grey', linewidth=0.5)

    def describe(group):
        parts = []
        for col, value in zip(group_cols, group):
            name, names = group_labels.get(col, (str(col), {}))
            text = names.get(int(value), str(int(value)))
            parts.append(f'{text} {name}'.strip())
        return '\n'.join(parts)

    ax.set_xticks(
        [(a + b - 1) / 2 for a, b in zip(starts[:-1], starts[1:])],
        [describe(cells[s][0]) for s in starts[:-1]],
        rotation=90,
        fontsize=14,
    )
    ax.set_xlabel(r'Groups quantile binned in $\hat{r}(x;\theta_1)$')
    ax.set_ylabel(
        r'$\left[\sum w(\theta_0)\,\hat{r}(x;\theta_1) - \sum w(\theta_0) \,r(x,z;\theta_1)\right]\,/\,\sigma$'
    )
    ax.legend()
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    fig.savefig(f'{label}/group_closure.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    with open(f'{label}/performance.yml', 'w') as f:
        f.write(yaml.dump(performance))


def lrMeanPlot(ax: Axes, lrhat: np.array, lr: np.array, p0: np.array, nbins: int, threshold: float = 0.01):
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

    sumw, _ = np.histogram(lrhat, bins=qbins, weights=p0)
    sumwlr, _ = np.histogram(lrhat, bins=qbins, weights=p0 * lr)
    sumw2lr2, _ = np.histogram(lrhat, bins=qbins, weights=p0**2 * lr**2)
    sumw2lr, _ = np.histogram(lrhat, bins=qbins, weights=p0**2 * lr)
    sumw2, _ = np.histogram(lrhat, bins=qbins, weights=p0**2)
    sumxw, _ = np.histogram(lrhat, bins=qbins, weights=p0 * lrhat)

    mean = sumwlr / sumw
    err_mean = np.sqrt(sumw2lr2 - 2 * sumw2lr * mean + sumw2 * mean**2) / sumw

    # remove negative weights from early training
    err_mean[err_mean < 0] = 0

    xcenter = sumxw / sumw
    xerr = abs(np.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))

    ax.errorbar(xcenter, mean, xerr=xerr, yerr=err_mean, fmt='.')

    chiSquare = (xcenter - mean) ** 2  # /(lr.max() - lr.min())**2
    chiSquare = chiSquare[~np.isnan(chiSquare)]
    chiSquare = (chiSquare.sum()).item()

    ax.plot([qbins[0], qbins[-1]], [qbins[0], qbins[-1]], color='grey', linestyle='--', label='_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\hat{\overline{r}}\,(x;\theta_1,\theta_0)$')
    ax.set_ylabel(r'$\overline{r}\,(x,z;\theta_1,\theta_0)$')

    return chiSquare


def sMeanPlot(ax: Axes, shat: np.array, p0: np.array, p1: np.array, nbins: int, threshold: float = 0.01):
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

    s = p1 / (p1 + p0)
    w = p1 + p0

    sumw, _ = np.histogram(shat, bins=qbins, weights=w)
    sumws, _ = np.histogram(shat, bins=qbins, weights=w * s)
    sumw2s2, _ = np.histogram(shat, bins=qbins, weights=w**2 * s**2)
    sumw2s, _ = np.histogram(shat, bins=qbins, weights=w**2 * s)
    sumw2, _ = np.histogram(shat, bins=qbins, weights=w**2)
    sumxw, _ = np.histogram(shat, bins=qbins, weights=w * shat)

    mean = sumws / sumw
    err_mean = np.sqrt(sumw2s2 - 2 * sumw2s * mean + sumw2 * mean**2) / sumw

    # remove negative weights from early training
    err_mean[err_mean < 0] = 0

    xcenter = sumxw / sumw
    xerr = abs(np.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))

    ax.errorbar(xcenter, mean, xerr=xerr, yerr=err_mean, fmt='.')

    chiSquare = (xcenter - mean) ** 2  # /(s.max() - s.min())**2
    chiSquare = chiSquare[~np.isnan(chiSquare)]
    chiSquare = (chiSquare.sum()).item()

    ax.plot([qbins[0], qbins[-1]], [qbins[0], qbins[-1]], color='grey', linestyle='--', label='_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\hat{\overline{s}}\,(x;\theta_1,\theta_0)$')
    ax.set_ylabel(r'$\overline{s}\,(x,z;\theta_1,\theta_0)$')

    return chiSquare


def compareDistributions(ax: Axes, pred: np.array, true: np.array, **plotParams):
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
    ax.plot([0, 1], [0, 1], color='grey', linestyle="--", transform=ax.transAxes, label='_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'Learned $\log(\hat{r})$')
    ax.set_ylabel(r'Calculated $\log(\hat{r})$')


def hist2d(ax: Axes, pred: np.array, true: np.array, weights: np.array, logbins: bool = False, **plotParams):
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

        h = ax.hist2d(pred, true, weights=weights, bins=100, norm=colors.LogNorm(), **plotParams)
    else:
        h = ax.hist2d(pred, true, weights=weights, bins=100, **plotParams)

    amin = min(pred.min().item(), true.min().item())
    amax = max(pred.max().item(), true.max().item())

    ax.set_xlim(amin, amax)
    ax.set_ylim(amin, amax)
    ax.plot([0, 1], [0, 1], color='grey', linestyle="--", transform=ax.transAxes, label='_nolegend_', linewidth=3)
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\log[\hat{r}(x;\theta_1,\theta_0)]$')
    ax.set_ylabel(r'$\log[r(x,z:\theta_1,\theta_0)]$')
    return h


def f_calibration_curve(
    ax: Axes, f: np.array, fhat: np.array, f_norm_factor: np.array, nbins: int, threshold: float = 0.01
):
    qbins = np.quantile(fhat, np.linspace(threshold, 1 - threshold, nbins + 1))

    sumw, _ = np.histogram(fhat, bins=qbins, weights=f_norm_factor)
    sumwf, _ = np.histogram(fhat, bins=qbins, weights=f_norm_factor * f)
    sumw2f2, _ = np.histogram(fhat, bins=qbins, weights=f_norm_factor**2 * f**2)
    sumw2f, _ = np.histogram(fhat, bins=qbins, weights=f_norm_factor**2 * f)
    sumw2, _ = np.histogram(fhat, bins=qbins, weights=f_norm_factor**2)
    sumwfhat, _ = np.histogram(fhat, bins=qbins, weights=f_norm_factor * fhat)

    mean = sumwf / sumw
    err_mean = np.sqrt(sumw2f2 - 2 * sumw2f * mean + sumw2 * mean**2) / sumw

    xcenter = sumwfhat / sumw
    xerr = abs(np.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))

    ax.errorbar(xcenter, mean, xerr=xerr, yerr=err_mean, fmt='.')

    chi_square = (xcenter - mean) ** 2 / (f.max() - f.min()) ** 2
    chi_square = chi_square[~np.isnan(chi_square)]
    chi_square = (chi_square.sum()).item()

    ax.plot([qbins[0], qbins[-1]], [qbins[0], qbins[-1]], color='grey', linestyle='--', label='_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\left\langle\hat{{f}}\,(x;\theta_1,\theta_0)\right\rangle$')
    ax.set_ylabel(r'$\left\langle{f}\,(x,z;\theta_1,\theta_0)\right\rangle$')

    return chi_square


def lr_calibration_curve(
    ax: Axes, lr: np.array, lrhat: np.array, lr_norm_factor: np.array, nbins: int, threshold: float = 0.01
):
    qbins = np.quantile(lrhat, np.linspace(threshold, 1 - threshold, nbins + 1))

    sumw, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor)
    sumwlr, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * lr)
    sumw2lr2, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2 * lr**2)
    sumw2lr, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2 * lr)
    sumw2, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2)
    sumwlrhat, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * lrhat)

    mean = sumwlr / sumw
    err_mean = np.sqrt(sumw2lr2 - 2 * sumw2lr * mean + sumw2 * mean**2) / sumw

    xcenter = sumwlrhat / sumw
    xerr = abs(np.stack([xcenter - qbins[:-1], qbins[1:] - xcenter], axis=0))

    ax.errorbar(xcenter, mean, xerr=xerr, yerr=err_mean, fmt='.')

    rel = (xcenter - mean) / mean
    rel_err = err_mean / mean
    ok = np.isfinite(rel) & np.isfinite(rel_err)
    wk = sumw[ok] / sumw[ok].sum()
    # remove the statistical scatter a perfect model would show
    rms_bias = np.sqrt(max((wk * rel[ok] ** 2).sum() - (wk * rel_err[ok] ** 2).sum(), 0)).item()

    ax.plot([qbins[0], qbins[-1]], [qbins[0], qbins[-1]], color='grey', linestyle='--', label='_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\left\langle\hat{{r}}\,(x;\theta_1,\theta_0)\right\rangle$')
    ax.set_ylabel(r'$\left\langle{r}\,(x,z;\theta_1,\theta_0)\right\rangle$')

    return rms_bias


def kinematic_shift_histogram(
    x0, x1, weights_0, weights_1, params, epoch, lr, outname, ylim=None, epoch_title=True, log=True, scale_ratio=False
):
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
            name='learned', label=params['label'], bins=params['nbins'], start=params['min'], stop=params['max']
        )
    )
    correct = hist.Hist(
        hist.axis.Regular(
            name='correct', label=params['label'], bins=params['nbins'], start=params['min'], stop=params['max']
        )
    )

    nominal = hist.Hist(
        hist.axis.Regular(
            name='nominal', label=params['label'], bins=params['nbins'], start=params['min'], stop=params['max']
        )
    )

    weights_0 = weights_0.flatten()
    weights_1 = weights_1.flatten()
    lr = lr.flatten()

    nominal.fill(x0, weight=weights_0)
    learned.fill(x0, weight=lr * weights_0)
    correct.fill(x1, weight=weights_1)

    mh.style.use("CMS")
    ax = []
    fig = plt.figure()
    grid = fig.add_gridspec(2, 1, hspace=0.1, height_ratios=[5, 1])
    ax += [fig.add_subplot(grid[0])]
    ax += [fig.add_subplot(grid[1], sharex=ax[0])]
    if epoch_title:
        fig.suptitle(f'Epoch {epoch:04d}')
    else:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax[0])

    n_learned, bins = learned.to_numpy()
    n_correct, _ = correct.to_numpy()

    lErr = []
    cErr = []
    for i in range(params['nbins']):
        lErr.append((lr[(x0 > bins[i]) & (x0 < bins[i + 1])] ** 2).sum())
        cErr.append(np.array((x1 > bins[i]) & (x1 < bins[i + 1])).sum())
    lErr = np.sqrt(lErr)
    cErr = np.sqrt(cErr)

    nominal.plot1d(ax=ax[0], linestyle='dashed', color='k', label=r'$r\,(x,z;\theta,0)$')
    correct.plot1d(
        ax=ax[0],
        yerr=cErr,
        color='C0',
        label=r'$r\,(x,z;\theta,\nu)$',
    )
    learned.plot1d(
        ax=ax[0],
        yerr=lErr,
        linestyle='dashdot',
        color='C1',
        label=r'$\hat{r}\,(x;\theta,\nu)$',
    )

    ratio = np.divide(n_learned, n_correct, where=((n_correct > 0) & (n_learned > 0)), out=np.zeros(n_learned.shape))
    rErr = ratio * np.sqrt(
        np.divide(lErr, n_learned, where=((n_correct > 0) & (n_learned > 0)), out=np.zeros(lErr.shape)) ** 2
        + np.divide(cErr, n_correct, where=((n_correct > 0) & (n_learned > 0)), out=np.zeros(n_correct.shape)) ** 2
    )

    ax[1].plot((bins[:-1] + bins[1:]) / 2, ratio, '.k')
    ax[1].errorbar((bins[:-1] + bins[1:]) / 2, ratio, yerr=rErr, fmt='none', color='k')
    ax[1].hlines(1, bins[0], bins[-1], color='k', linestyle='dashed', alpha=0.5)

    if scale_ratio:
        bin_mask = n_correct >= 1
        err_max = np.array([((ratio + rErr - 1)[bin_mask]).max(), ((1 - ratio + rErr)[bin_mask]).max()]).max() * 1.05
        ax[1].set_ylim([1 - err_max, 1 + err_max])
    else:
        ax[1].set_ylim(0, 2)
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
    threshold: float = 0.01,
    include_stdv: bool = False,
):
    qbins = np.quantile(lrhat, np.linspace(threshold, 1 - threshold, nbins + 1))

    sumw, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor)
    sumwlr, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * lr)
    sumw2, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2)
    sumwlrhat, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * lrhat)
    sumwlrdiff, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * (lr - lrhat))
    sumwlrdiff2, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor * (lr - lrhat) ** 2)
    sumw2lrdiff, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2 * (lr - lrhat))
    sumw2lrdiff2, _ = np.histogram(lrhat, bins=qbins, weights=lr_norm_factor**2 * (lr - lrhat) ** 2)

    meanlr = sumwlr / sumw
    meanlrhat = sumwlrhat / sumw
    meanlrdiff = sumwlrdiff / sumw

    mean_err = np.sqrt(sumw2lrdiff2 + sumw2 * meanlrdiff**2 - 2 * meanlrdiff * sumw2lrdiff) / sumw
    err_width = 2 * mean_err
    err_down = (meanlr - meanlrhat) - mean_err

    mean_stdv = np.sqrt(np.maximum(sumwlrdiff2 / sumw - meanlrdiff**2, 0))
    stdv_width = 2 * mean_stdv
    stdv_down = (meanlr - meanlrhat) - mean_stdv

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
        'facecolor': "#D5E3D5",
        'linewidth': 0,
        'label': r'$\sigma_{ r  -\hat{r}}$',
        'alpha': 0.6,
    }

    ax.hist(qbins[:-1], weights=differ, bins=qbins, histtype='step', color='k', label=r'$r - \hat{r}$')

    if include_stdv:
        ax.bar(qbins[:-1], stdv_width, bottom=stdv_down, width=(qbins[1:] - qbins[:-1]), **mean_stdv_formatting)
    ax.plot([qbins[0], qbins[-1]], [0, 0], color='grey', linestyle=':', label='_nolegend_')
    ax.bar(qbins[:-1], err_width, bottom=err_down, width=(qbins[1:] - qbins[:-1]), **mean_error_formatting)

    pulls = np.abs(differ) / mean_err
    pulls = pulls[np.isfinite(pulls)]
    n = len(pulls)
    # for a perfectly calibrated model the pulls are standard normal: E|Z| = sqrt(2/pi), Var|Z| = 1 - 2/pi
    pull = (pulls.mean() - np.sqrt(2 / np.pi)).item()
    pull_err = np.sqrt((1 - 2 / np.pi) / n).item()
    expected = [0.6827, 0.9545, 0.9973]
    ci_diff = [(e - (pulls < s).mean()).item() for s, e in zip(range(1, 4), expected)]
    # binomial scatter of the fraction of bins inside each interval
    ci_diff_err = [np.sqrt(e * (1 - e) / n).item() for e in expected]

    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\left\langle\hat{r}\,(x;\theta_1,\theta_0)\right\rangle$')
    ax.set_ylabel(
        r'$\left\langle{r}\,(x,z;\theta_1,\theta_0)\right\rangle - \left\langle\hat{{r}}\,(x;\theta_1,\theta_0)\right\rangle$'
    )
    y_max = abs(np.array(ax.get_ylim())).max() * 1.2
    ax.set_ylim([-y_max, y_max])
    ax.set_xlim([qbins[0], qbins[-1]])

    return pull, pull_err, ci_diff, ci_diff_err


# reference categorical palette (fixed order), slot 1 for the ensemble, the rest for outlier seeds
ENSEMBLE_COLOR = '#2a78d6'
MEDIAN_COLOR = '#184f95'  # darker step of the same blue ramp, for the median over seeds
SEARCH_COLOR = '#008300'  # palette green, for what the point search proposes
OUTLIER_COLORS = ['#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']


def plot_residuals(residuals, outname, bins=200, outlier_factor=3.0):
    '''
    Distribution of the per-event least-squares residuals ||Theta gamma_hat - r_hat||^2 across seeds: the median
    histogram over seeds, the band between the smallest and largest count per bin, and, as separate lines, seeds whose
    median residual differs from the typical seed's by more than outlier_factor.

    Args:
        residuals: {seed: [N_s] residuals}, None for a seed whose fit returned none (rank-deficient Theta)
        outname: file to save the plot to; a log-count version is saved as <outname>_log next to it
        bins: bin edges of the residuals, or a number of log-spaced bins spanning the residuals of all seeds
        outlier_factor: ratio of a seed's median residual to the median over seeds beyond which it is drawn alone
    Returns:
        seeds drawn as outliers
    '''
    available = {seed: np.asarray(r) for seed, r in residuals.items() if r is not None}
    missing = [seed for seed in residuals if seed not in available]
    if missing:
        print(f'[info]: no fit residuals for seeds {missing}')
    if not available:
        print(f'[info]: no fit residuals stored, {outname} not made')
        return []

    # x range and default bins from the residuals of all seeds; the smallest positive value, since the axis is log
    values = np.concatenate(list(available.values()))
    lo, hi = values[values > 0].min(), values.max()
    if np.isscalar(bins):
        bins = np.logspace(np.log10(lo), np.log10(hi), int(bins) + 1)

    seeds = list(available)
    counts = np.array([np.histogram(available[seed], bins=bins)[0] for seed in seeds])
    medians = np.array([np.median(available[seed]) for seed in seeds])
    typical = np.median(medians)
    outliers = [seed for seed, m in zip(seeds, medians) if abs(np.log(m / typical)) > np.log(outlier_factor)]
    if len(outliers) > len(OUTLIER_COLORS):
        print(f'[info]: {len(outliers)} outlier seeds, only the first {len(OUTLIER_COLORS)} are drawn')

    fig, ax = plt.subplots()
    ax.stairs(
        counts.max(0),
        bins,
        baseline=counts.min(0),
        fill=True,
        color=ENSEMBLE_COLOR,
        alpha=0.3,
        label=f'min-max over {len(seeds)} seeds',
    )
    ax.stairs(np.median(counts, axis=0), bins, color=MEDIAN_COLOR, linewidth=2, label='median over seeds')
    for seed, color in zip(outliers, OUTLIER_COLORS):
        ax.stairs(counts[seeds.index(seed)], bins, color=color, linewidth=2, label=f'seed {seed}')
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylabel('Number of Events')
    ax.set_xlabel('Fit Residuals')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    fig.savefig(outname, bbox_inches='tight')
    # the same plot with a logarithmic count axis, saved next to it as <outname>_log
    root, ext = os.path.splitext(outname)
    ax.set_yscale('log')
    fig.savefig(f'{root}_log{ext}', bbox_inches='tight')
    plt.close(fig)
    return outliers


# sequential ramp of the reference palette (blue, light to dark)
SEQUENTIAL = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
KIND_ORDER = ['sm', 'linear', 'quadratic', 'cross']
# label colours for networks the suspects step dropped (dark red) or wants retrained (orange)
NOTE_COLORS = {'dropped': '#a51d1d', 'retrain': '#d95926'}


def wc_tex(wc):
    '''TeX of a WC name, the convention of likelihood_scan_dev.ipynb: cQj31 -> c_{Qj}^{31}, ctGRe -> \Re[c_{tG}].'''
    m = re.fullmatch(r'c([A-Za-z]+?)(Re|Im)', wc)
    if m:
        return rf'\{m.group(2)}\left[c_{{{m.group(1)}}}\right]'
    sub, sup = re.fullmatch(r'c([A-Za-z]+)(\d*)', wc).groups()
    return f'c_{{{sub}}}' + ('' if not sup else f'^{sup}' if len(sup) == 1 else f'^{{{sup}}}')


def component_tex(label):
    '''TeX label of a gamma component from component_labels: sm, c, |c|^2 or c_a . c_b.'''
    if label == 'sm':
        return 'sm'
    if label.endswith('^2'):
        return rf'$\left|{wc_tex(label[:-2])}\right|^2$'
    # the space only ends the command name (\cdotc would be read otherwise); TeX does not print it
    return '$' + r'\cdot '.join(wc_tex(wc) for wc in label.split('*')) + '$'


def _no_tex_mapping(record):
    '''Logging filter for mathtext's harmless notice that sized delimiters (\\left| ... \\right|) have no Unicode glyph.'''
    return not record.getMessage().startswith('No TeX to Unicode mapping')


def plot_z_heatmap(
    Z, labels, kinds, features, outname, z_threshold=5.0, z_max=15.0, components=None, feature_config=None, Z_gauss=None
):
    '''
    Significance of the bias of gamma components in every network input. Columns are components, grouped by kind and
    sorted by their largest Z within each group; rows are features, sorted by how many components exceed z_threshold
    in them (counted over all components, so every version of the plot shares the row order); cells above z_threshold
    are outlined.

    Args:
        Z: [K, F] output of component_bias
        labels, kinds: [K] component names and kinds (component_labels)
        features: [F] feature names (binning['names'])
        outname: file to save the plot to
        z_threshold: flagging threshold
        z_max: Z at the top of the colour scale (larger values share the darkest colour)
        components: indices of the components to show, default all
        feature_config: contents of the features yml; if given, features are labelled with their TeX label (as on
            the x axis of the training's kinematic plots, without the unit) instead of by name
        Z_gauss: [K, F] Gaussian-equivalent significance; cells with Z_gauss > z_threshold (a genuine 5 sigma) get a
            second, inner outline
    '''
    kinds = np.array(kinds)
    components = np.arange(len(labels)) if components is None else np.asarray(components)
    max_z = np.nanmax(Z, axis=1)
    columns = [k for kind in KIND_ORDER for k in sorted(components[kinds[components] == kind], key=lambda k: -max_z[k])]
    rows = np.argsort(-(Z > z_threshold).sum(0), kind='stable')
    shown = Z[np.ix_(columns, rows)].T

    cmap = LinearSegmentedColormap.from_list('sequential', SEQUENTIAL)
    cmap.set_bad('#d9d9d9')  # no events of the component in that feature
    # square cells, 0.55 inch on a side, with room for the labels, the colour bar and the header
    fig, ax = plt.subplots(figsize=(0.55 * len(columns) + 4, 0.55 * len(rows) + 3))
    image = ax.imshow(np.clip(shown, 0, z_max), cmap=cmap, vmin=0, vmax=z_max, aspect='equal', interpolation='none')
    for row, col in zip(*np.nonzero(shown > z_threshold)):
        ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor='#e34948', linewidth=1.5))
    if Z_gauss is not None:
        # a second outline inside the first where the flag is also a genuine 5 sigma
        strong = Z_gauss[np.ix_(columns, rows)].T > z_threshold
        for row, col in zip(*np.nonzero(strong)):
            ax.add_patch(Rectangle((col - 0.3, row - 0.3), 0.6, 0.6, fill=False, edgecolor='#e34948', linewidth=1.5))
    # separators between the kinds
    present = [kind for kind in KIND_ORDER if (kinds[columns] == kind).any()]
    edges = np.cumsum([(kinds[columns] == kind).sum() for kind in present])
    for e in edges[:-1]:
        ax.axvline(e - 0.5, color='k', linewidth=1)
    ax.set_xticks(range(len(columns)), [component_tex(labels[k]) for k in columns], rotation=90, fontsize=16)
    if feature_config is None:
        names = [features[f] for f in rows]
    else:
        names = [feature_config[features[f]]['label'] for f in rows]
    ax.set_yticks(range(len(rows)), names, fontsize=16)
    # colour bar 0.6 inch wide, attached to the grid so it always has the grid's height
    bar = fig.colorbar(image, cax=make_axes_locatable(ax).append_axes('right', size=0.6, pad=0.15))
    # Z upright above the bar, level with the kind names over the grid
    bar.ax.set_title('Z', fontsize=16, pad=4)
    # values above z_max share the top colour, which the last tick says
    ticks = list(range(0, int(z_max), 2)) + [z_max]
    bar.set_ticks(ticks, labels=[f'{t:g}' for t in ticks[:-1]] + [rf'$\geq${z_max:g}'])
    bar.ax.axhline(z_threshold, color='#e34948', linewidth=2)
    cms = mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    # kind names just above the grid, centred on their group of columns
    gap = ScaledTranslation(0, 4 / 72, fig.dpi_scale_trans)
    where = blended_transform_factory(ax.transData, ax.transAxes) + gap
    names = [
        ax.text((first + stop - 1) / 2, 1, kind, transform=where, ha='center', va='bottom', fontsize=16)
        for kind, first, stop in zip(present, np.r_[0, edges[:-1]], edges)
    ]

    # the TeX labels are drawn when the figure is; silence only mathtext's notice about the sized delimiters
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        # the CMS header moves up above the kind names, whose height is known only once they are drawn
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        height = max(t.get_window_extent(renderer).height for t in names) / fig.dpi
        lift = ScaledTranslation(0, height + 8 / 72, fig.dpi_scale_trans)
        for text in cms:
            if text is not None:
                text.set_transform(text.get_transform() + lift)
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)


def _place_legend(ax, samples, note, fontsize=20):
    '''
    Put the legend, with a note line under it, where the two cover the fewest data samples. The corners and the top and
    bottom centres of the axes are tried in turn; the axis ranges are not changed. For a bottom placement the legend is
    lifted by the height of the note, so the note still sits under it.

    Args:
        ax: axes holding the labelled artists
        samples: [N, 2] points in data coordinates that the legend should avoid (lines and error bars, densely sampled)
        note: text of the line under the legend
        fontsize: font size of the legend entries and the note
    Returns:
        the legend and the note text
    '''
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points = ax.transData.transform(samples)
    to_axes = ax.transAxes.inverted()
    probe = ax.text(0, 0, note, transform=ax.transAxes, fontsize=fontsize)
    lift = (probe.get_window_extent(renderer).height + 4) / ax.get_window_extent(renderer).height
    probe.remove()

    def place(loc):
        anchor = (0, lift, 1, 1 - lift) if loc.startswith('lower') else (0, 0, 1, 1)
        legend = ax.legend(fontsize=fontsize, loc=loc, bbox_to_anchor=anchor, bbox_transform=ax.transAxes)
        box = legend.get_window_extent(renderer)
        (x0, y0), (x1, _) = to_axes.transform([(box.x0, box.y0), (box.x1, box.y1)])
        side = 'right' if 'right' in loc else 'left' if 'left' in loc else 'center'
        x = {'right': x1, 'left': x0, 'center': (x0 + x1) / 2}[side]
        text = ax.text(x, y0, note, transform=ax.transAxes, ha=side, va='top', fontsize=fontsize)
        return legend, text

    scores = {}
    for loc in ['upper right', 'upper left', 'upper center', 'lower right', 'lower left', 'lower center']:
        legend, text = place(loc)
        boxes = [legend.get_window_extent(renderer), text.get_window_extent(renderer)]
        scores[loc] = sum(
            int(
                (
                    (points[:, 0] >= b.x0) & (points[:, 0] <= b.x1) & (points[:, 1] >= b.y0) & (points[:, 1] <= b.y1)
                ).sum()
            )
            for b in boxes
        )
        legend.remove()
        text.remove()
    return place(min(scores, key=scores.get))


def plot_closure(k, f, sums, bias, binning, labels, outname, feature_config=None):
    '''
    Closure of one gamma component in one feature: the learned sum w0 * gamma_k against its truth per bin (top), and
    their difference relative to the SM yield with the noise band sqrt(E^2 + s^2) of the bias test (bottom). Error bars
    show the MC statistics of each sum (and the seed spread for the learned one), both at the bin centre. Bins are
    drawn with equal widths and labelled by their edges; bins without events of the component are hatched.

    Args:
        k: component index
        f: feature index into binning['names']
        sums: output of component_sums
        bias: output of component_bias
        binning: the binning of sums
        labels: [K] component names (component_labels)
        outname: file to save the plot to
        feature_config: contents of the features yml, for the feature's TeX label
    '''
    b = binning['feature'] == f
    truth, D = sums['truth'][k, b], bias['D'][k, b]
    noise = np.sqrt(sums['mc_var'][k, b] + bias['s2'][k, b])
    sm = sums['sm_yield'][b]
    empty = sums['mc_var'][k, b] <= 0
    edges = binning['edges'][f]
    x = np.arange(len(edges))
    centers = x[:-1] + 0.5

    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0}
    )
    top.stairs(truth, x, color='k', linewidth=2, label='truth')
    top.stairs(truth + D, x, color=ENSEMBLE_COLOR, linewidth=2, label=r'$\sum w_0\,\hat{\gamma}$')
    # MC statistics of each sum; the learned one also carries the seed spread of the ensemble mean
    truth_err = np.sqrt(sums['truth_var'][k, b])
    learned_err = np.sqrt(sums['learned_var'][k, b] + bias['s2'][k, b])
    top.errorbar(centers, truth, yerr=truth_err, fmt='none', color='k', linewidth=2)
    top.errorbar(
        centers,
        truth + D,
        yerr=learned_err,
        fmt='none',
        color=ENSEMBLE_COLOR,
        linewidth=2,
    )
    # the component's coefficient in the yield: events it adds per bin when its WC product equals 1
    if labels[k] == 'sm':
        top.set_ylabel('Events (SM)')
    else:
        top.set_ylabel(f'Events per unit {component_tex(labels[k])}')
    band = np.where(empty, np.nan, noise / sm)
    bottom.stairs(
        band, x, baseline=-band, fill=True, color='#d9d9d9', label=r'$\pm\sqrt{E^2 + s^2} / \lambda(\mathrm{sm})$'
    )
    bottom.plot(centers[~empty], (D / sm)[~empty], 'o', color='k', label=r'$\Delta / \lambda(\mathrm{sm})$')
    bottom.axhline(0, color='k', linestyle='--', linewidth=1)
    # bins without events of the component: hatched in both panels
    for i in np.flatnonzero(empty):
        for ax in (top, bottom):
            ax.axvspan(x[i], x[i + 1], facecolor='none', edgecolor='#9a9a9a', hatch='//', linewidth=0)
    bottom.set_ylabel(r'$\Delta / \lambda(\mathrm{sm})$')
    name = binning['names'][f]
    # as on the x axis of the training's kinematic plots: label [unit]
    if feature_config:
        unit = feature_config[name].get('unit')
        bottom.set_xlabel(feature_config[name]['label'] + (f' [{unit}]' if unit else ''))
    else:
        bottom.set_xlabel(name)
    # equal-width bins labelled with the feature values at 6 evenly spaced edges, the first and last included
    # (fewer if the feature has fewer edges); 3 significant digits, no exponent
    tick = [np.format_float_positional(e, precision=3, unique=False, fractional=False, trim='-') for e in edges]
    # a feature symmetric about 0 (e.g. an eta) gets 7 labels instead, the middle one at the point where it is 0
    symmetric = edges[0] < 0 < edges[-1] and abs(edges[0] + edges[-1]) <= 0.05 * (edges[-1] - edges[0])
    shown = np.unique(np.round(np.linspace(0, len(edges) - 1, min(7 if symmetric else 6, len(edges)))).astype(int))
    positions, names = list(x[shown].astype(float)), [tick[i] for i in shown]
    if symmetric and len(shown) == 7:
        positions[3], names[3] = float(np.interp(0, edges, x)), '0'
    bottom.set_xticks(positions, names)
    # minor ticks on every bin edge, so major and minor ticks both sit on edges (shared by the top panel)
    bottom.set_xticks(x, minor=True)
    bottom.set_xlim(x[0], x[-1])
    # the CMS label measures the text already on the figure, so the filter must be on before it is added
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=top)
        # legend and Z_max where they cover the least of the lines and error bars (the ranges stay as they are)
        sampled = [
            np.c_[np.linspace(x[i], x[i + 1], 20), np.full(20, v[i])] for v in (truth, truth + D) for i in range(len(v))
        ]
        sampled += [
            np.c_[np.full(20, x[i + 1]), np.linspace(v[i], v[i + 1], 20)]
            for v in (truth, truth + D)
            for i in range(len(v) - 1)
        ]
        sampled += [
            np.c_[np.full(20, c), np.linspace(v - e, v + e, 20)]
            for vals, errs in ((truth, truth_err), (truth + D, learned_err))
            for c, v, e in zip(centers, vals, errs)
        ]
        samples = np.concatenate(sampled)
        _place_legend(top, samples[np.isfinite(samples).all(1)], rf'$Z_{{max}}$ = {np.nanmax(bias["Z"][k]):.1f}')
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)


def network_tex(Theta, names, wcs, mask):
    '''
    TeX label of the training point of every network: its nonzero WC values, read from the linear columns of Theta,
    e.g. "c_Qj^31 = 1.70" or "c_Qj^31, Re[c_tG] = -1.28, 4.27"; the SM anchor is "sm".

    Args:
        Theta: [n_networks, K] expanded training points
        names: network names, rows of Theta
        wcs: WC names in get_lower_tri order
        mask: kept get_lower_tri columns
    Returns:
        {network name: label}
    '''
    rows, cols = np.tril_indices(len(wcs) + 1)
    kept = [(int(i), int(j)) for i, j, keep in zip(rows, cols, mask) if keep]
    linear = {i: kept.index((i, 0)) for i in range(1, len(wcs) + 1) if (i, 0) in kept}
    out = {}
    for name, row in zip(names, Theta):
        point = [(wc_tex(wcs[i - 1]), row[c]) for i, c in linear.items() if row[c] != 0]
        if not point:
            out[name] = 'sm'
        elif len(point) == 1:
            out[name] = rf'${point[0][0]} = {point[0][1]:.2f}$'
        else:
            # interference point: c_a, c_b = x, y
            wc_part = ', '.join(t for t, _ in point)
            value_part = ', '.join(f'{v:.2f}' for _, v in point)
            out[name] = rf'${wc_part} = {value_part}$'
    return out


def plot_attribution(k, shares, flagged, labels, bias, network_tex, outname, n_top=10, notes=None):
    '''
    Signed shares of the training networks in the bias of one gamma component (shares sum to 1): the n_top networks
    with the largest |share| and the sum of the others, as horizontal bars.

    Args:
        k: component index
        shares: first output of attribute, {network name: [len(flagged)]}
        flagged: component indices attribute was run for (gives the column of k)
        labels: [K] component names (component_labels)
        bias: output of component_bias (for Z_max and rms bias above the plot)
        network_tex: {network name: TeX label of its training point}
        outname: file to save the plot to
        n_top: networks shown individually
        notes: {network name: 'dropped' or 'retrain'}; their labels are coloured and a legend names the colours
    '''
    column = flagged.index(k)
    ranked = sorted(shares, key=lambda n: -abs(shares[n][column]))
    top, rest = ranked[:n_top], ranked[n_top:]
    values = [shares[n][column] for n in top] + [sum(shares[n][column] for n in rest)]
    notes = notes or {}
    names = [network_tex[n] for n in top]
    names.append(f'other ({len(rest)} networks)')
    y = np.arange(len(values))[::-1]  # largest at the top

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(values) + 2))
    ax.barh(y, values, height=0.8, color=ENSEMBLE_COLOR)
    ax.axvline(0, color='k', linewidth=1)
    ax.axvline(1, color='k', linewidth=1, linestyle=':')
    ax.set_yticks(y, names, fontsize=16)
    # networks the suspects step dropped or wants retrained: coloured labels, named in a legend
    for tick, n in zip(ax.get_yticklabels(), top):
        if n in notes:
            tick.set_color(NOTE_COLORS[notes[n]])
    present = [kind for kind in NOTE_COLORS if kind in {notes[n] for n in top if n in notes}]
    ax.set_ylim(-0.5, len(values) - 0.5)
    ax.set_xlabel(r'share of $\Delta$')
    max_z = np.nanmax(bias['Z'][k])
    size = np.nanmax(bias['rms_bias'][k])
    # component and its bias centred just above the plot; the CMS header is lifted above it below
    gap = ScaledTranslation(0, 4 / 72, fig.dpi_scale_trans)
    note = ax.text(
        0.5,
        1,
        f'{component_tex(labels[k])}: $Z_{{max}}$ = {max_z:.1f}, $\\Delta_{{rms}}$ = {size:.1e}',
        transform=ax.transAxes + gap,
        ha='center',
        va='bottom',
        fontsize=18,
    )
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        cms = mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
        # the header's lift is the label's height, known only once drawn
        fig.canvas.draw()
        lift = ScaledTranslation(
            0, note.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi + 8 / 72, fig.dpi_scale_trans
        )
        for text in cms:
            if text is not None:
                text.set_transform(text.get_transform() + lift)
        if present:
            # legend on the line of the Z_max label, centred over the column of network labels
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            boxes = [t.get_window_extent(renderer) for t in ax.get_yticklabels()]
            center_x = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
            line = note.get_window_extent(renderer)
            fig.legend(
                handles=[Patch(color=NOTE_COLORS[kind], label=kind) for kind in present],
                fontsize=16,
                loc='center',
                bbox_to_anchor=fig.transFigure.inverted().transform((center_x, (line.y0 + line.y1) / 2)),
                borderaxespad=0,
                borderpad=0,
                ncols=len(present),
            )
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)


# diverging pair of the reference palette: blue (Z_max down) - neutral grey - red (Z_max up)
DIVERGING = ['#184f95', '#2a78d6', '#f0efec', '#e34948', '#a51d1d']


def plot_drop_tests(
    baseline, after, combined, groups, blamed, tests, labels, network_tex, outname, min_change=1.0, show_all=False
):
    '''
    Change of Z_max (without the group - full ensemble) of every component a suspect group affects, for every group the
    drop test tried and for the combined refit without all accepted groups. Cells a group was blamed for are outlined;
    row labels are dark red for accepted groups, black for rejected ones and grey for groups whose removal would leave
    Theta rank-deficient (hatched, no refit).

    Args:
        baseline: component_bias of the full ensemble
        after: {group: component_bias without the group, or None if Theta would lose rank}
        combined: component_bias without all accepted groups (None if nothing was accepted)
        groups: {group: [network names]}
        blamed: {group: [component indices it was blamed for]}
        tests: output of test_drops (accepted, gain, harmed per group)
        labels: [K] component names
        network_tex: {network name: TeX label of its training point}
        outname: file to save the plot to
        min_change: a component is shown if some group changes its Z_max by more than this
        show_all: show every component instead
    '''
    before = np.nanmax(baseline['Z'], axis=1)
    names = list(groups)
    rows = [None if after[g] is None else np.nanmax(after[g]['Z'], axis=1) - before for g in names]
    accepted = [g for g in names if tests[g]['accepted']]
    if combined is not None:
        rows.append(np.nanmax(combined['Z'], axis=1) - before)
    shown = {k for r in rows if r is not None for k in np.flatnonzero(np.abs(r) > min_change)}
    shown |= {k for g in names for k in blamed[g]}
    if show_all:
        shown = set(range(len(labels)))
    kinds = np.array(component_kinds(labels))
    columns = [k for kind in KIND_ORDER for k in sorted(shown, key=lambda k: -before[k]) if kinds[k] == kind]
    delta = np.array([np.full(len(columns), np.nan) if r is None else r[columns] for r in rows])

    def row_label(g):
        members = groups[g]
        if len(members) == 1:
            return network_tex[members[0]]
        return network_tex[members[0]].split(' = ')[0] + f'$ ({len(members)} points)'

    row_names = [row_label(g) for g in names] + (['all accepted (combined)'] if combined is not None else [])
    row_colors = [NOTE_COLORS['dropped'] if g in accepted else '#9a9a9a' if after[g] is None else 'k' for g in names]
    row_colors += ['k'] if combined is not None else []
    outlined = [blamed[g] for g in names] + (
        [sorted({k for g in accepted for k in blamed[g]})] if combined is not None else []
    )

    # matplotlib's RdBu, reversed so that a drop in Z_max is blue and a rise is red
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('white')
    fig, ax = plt.subplots(figsize=(0.55 * len(columns) + 6, 0.55 * len(row_names) + 4))
    image = ax.imshow(np.clip(delta, -10, 10), cmap=cmap, vmin=-10, vmax=10, aspect='equal', interpolation='none')
    for r, c in zip(*np.nonzero(np.isfinite(delta))):
        v = delta[r, c]
        # + 0.0 turns a value that rounds to -0.0 into 0.0
        text = f'{np.round(v, 1) + 0.0:.1f}'
        ax.text(c, r, text, ha='center', va='center', fontsize=11, color='white' if abs(v) > 6 else 'k')
    for r, ks in enumerate(outlined):
        for k in ks:
            if k in columns:
                c = columns.index(k)
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor='k', linewidth=2))
    for r in np.flatnonzero(~np.isfinite(delta).any(1)):
        ax.add_patch(
            Rectangle((-0.5, r - 0.5), len(columns), 1, facecolor='none', edgecolor='#9a9a9a', hatch='//', linewidth=0)
        )
    if combined is not None:
        ax.axhline(len(names) - 0.5, color='k', linewidth=2)
    # separators between the kinds of components
    present = [kind for kind in KIND_ORDER if (kinds[columns] == kind).any()]
    for e in np.cumsum([(kinds[columns] == kind).sum() for kind in present])[:-1]:
        ax.axvline(e - 0.5, color='k', linewidth=1)
    ax.set_xticks(range(len(columns)), [component_tex(labels[k]) for k in columns], rotation=90, fontsize=16)
    ax.set_yticks(range(len(row_names)), row_names, fontsize=16)
    for tick, color in zip(ax.get_yticklabels(), row_colors):
        tick.set_color(color)
    # rows and columns are labelled, so no tick marks reaching into the cells
    ax.tick_params(which='both', length=0)
    bar = fig.colorbar(image, cax=make_axes_locatable(ax).append_axes('right', size=0.6, pad=0.15))
    bar.set_label(r'$\Delta Z_{max}$ (without - with)')
    bar.set_ticks([-10, -5, 0, 5, 10], labels=[r'$\leq$-10', '-5', '0', '5', r'$\geq$10'])
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)


def component_kinds(labels):
    '''Kind of every component label from component_labels: sm, linear, quadratic or cross.'''
    return [
        'sm' if name == 'sm' else 'quadratic' if name.endswith('^2') else 'cross' if '*' in name else 'linear'
        for name in labels
    ]


def unit_tex(unit, wcs):
    '''
    TeX label of a point-search unit: the WCs it varies with their magnitudes; a cross-term unit (the 4 sign
    combinations) reads "c_a, c_b = +-x, +-y", a single point "c_a, c_b = x, y".
    '''
    theta = np.asarray(unit['points'][0]['theta'])
    nonzero = np.flatnonzero(theta[1:]) + 1
    names = ', '.join(wc_tex(wcs[i - 1]) for i in nonzero)
    if len(unit['points']) > 1:
        values = ', '.join(rf'\pm{abs(theta[i]):.2f}' for i in nonzero)
    else:
        values = ', '.join(f'{theta[i]:.2f}' for i in nonzero)
    return rf'${names} = {values}$'


def plot_point_search(history, chosen, wcs, outname, scan=False):
    '''
    Objective of the point search after each added unit, relative to the current design (1 = no units added), with
    every step labelled by the unit it added.

    Args:
        history: objective after each step, history[0] for the current design (search_points)
        chosen: the units added, in order (search_points)
        wcs: WC names in get_lower_tri order
        outname: file to save the plot to
        scan: the search used the scan-region weights (G: scan) rather than the flagged-component weights
    '''
    relative = np.asarray(history) / history[0]
    x = np.arange(len(relative))
    fig, ax = plt.subplots(figsize=(0.8 * len(x) + 6, 8))
    # one bin per step (the current design, then each added point), drawn as a step outline
    ax.hist(
        x, bins=np.arange(len(relative) + 1) - 0.5, weights=relative, histtype='step', color=SEARCH_COLOR, linewidth=2
    )
    # labels slanted at 45 degrees, anchored at their right end under their point
    names = ['current design'] + [unit_tex(u, wcs) for u in chosen]
    ax.set_xticks(x, names, rotation=45, ha='right', rotation_mode='anchor', fontsize=16)
    ax.set_xlim(-0.5, len(x) - 0.5)
    # y range from just below the lowest bin to just above the current design, so the steps fill the plot
    span = relative.max() - relative.min()
    pad = 0.1 * span if span > 0 else 0.05
    ax.set_ylim(relative.min() - pad, relative.max() + pad)
    ax.tick_params(axis='x', which='minor', length=0)
    # with the flagged weights the relative objective is the weighted mean of each component's squared error
    # relative to the current design; with the scan-region weights it is a trace over the scan region
    if scan:
        ax.set_ylabel('scan-region expected error, relative to the current design')
    else:
        ax.set_ylabel(r'$\left\langle (\sigma_k / \sigma_k^{\mathrm{current}})^2 \right\rangle_w$')
    ax.set_xlabel('point added')
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)


def plot_search_sigma(sigma0, sigma1, rms_bias, labels, outname, components=None):
    '''
    Expected error of the components (independent network errors) with the current design (black) and with the points
    the point search proposes (green), joined by a thin line, over the measured rms bias (grey bars), log scale.
    Components the design cannot determine (NaN) are marked on the upper edge, components with rms bias 0 (noise
    exceeds the bias) on the lower edge.

    Args:
        sigma0, sigma1: [K] propagated errors before and after the search, NaN for unconstrained components
        rms_bias: [K, F] output of component_bias
        labels: [K] component names (component_labels)
        outname: file to save the plot to
        components: indices of the components to show, default all
    '''
    size = np.nanmax(rms_bias, axis=1)
    components = np.arange(len(labels)) if components is None else np.asarray(components)
    order = sorted(components, key=lambda k: -size[k])
    x = np.arange(len(order))
    s0, s1, b = sigma0[order], sigma1[order], size[order]
    shown = np.concatenate([b, s0, s1])
    shown = shown[np.isfinite(shown) & (shown > 0)]
    low, high = shown.min() / 2, shown.max() * 2

    fig, ax = plt.subplots(figsize=(0.55 * len(order) + 4, 7))
    ax.bar(x[b > 0], b[b > 0], width=1, fill=False, edgecolor='#9a9a9a', linewidth=2, label=r'measured $\Delta_{rms}$')
    zero = ~(b > 0)
    if zero.any():
        # noise exceeds the bias: no bar, a marker on the lower edge instead
        ax.scatter(
            x[zero],
            np.full(zero.sum(), low),
            marker='^',
            s=120,
            color='#9a9a9a',
            clip_on=False,
            zorder=5,
            label=r'$\Delta_{rms}$ = 0 (noise exceeds $\Delta$)',
        )
    both = np.isfinite(s0) & np.isfinite(s1)
    ax.vlines(x[both], s1[both], s0[both], colors='k', linewidth=1, zorder=3)
    ax.hlines(s0, x - 0.5, x + 0.5, colors='k', linewidth=2, zorder=4, label='current design')
    # dashed, so the current-design tick shows through where the two coincide
    ax.hlines(
        s1,
        x - 0.5,
        x + 0.5,
        colors=SEARCH_COLOR,
        linewidth=2,
        linestyles='--',
        zorder=4,
        label='with the proposed points',
    )
    for values, color in ((s0, 'k'), (s1, SEARCH_COLOR)):
        free = np.isnan(values)
        if free.any():
            ax.scatter(x[free], np.full(free.sum(), high), marker='v', s=120, color=color, clip_on=False, zorder=5)
    ax.set_yscale('log')
    ax.set_ylim(low, high)
    ax.set_ylabel(r'$\sigma_k / \lambda(\mathrm{sm})$')
    ax.legend(fontsize=16)
    ax.set_xticks(x, [component_tex(labels[k]) for k in order], rotation=90, fontsize=16)
    ax.set_xlim(-0.5, len(order) - 0.5)
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)


# markers of the kinds of training point: (marker, colour, label); palette slots 1 and 3, the SM anchor in black
POINT_KINDS = {
    'scan': ('o', '#2a78d6', 'single-WC point'),
    'cross': ('^', '#1baf7a', 'interference point'),
    'sm': ('*', 'k', 'SM anchor'),
}


def plot_network_misclosure(misclosure, yield_ratio, network_tex, outname, notes=None, n_labels=3, per_seed=None):
    '''
    Misclosure m_n of every training network against the size of its training point (inclusive yield relative to the
    SM, lambda(theta_1) / lambda_SM, linear) on a log scale, by kind of point; suspects ringed in their attribution colours and the
    n_labels largest m_n labelled with their training point in a column on the right, joined by leader lines. The
    kind of point is shown by marker shape in neutral grey; suspects are filled in their status colour. With per_seed, every seed's m_n is drawn as a small
    marker, joined by a thin line from the smallest to the largest, with the seed average as the large marker.

    Args:
        misclosure: {network name: m_n} averaged over seeds (network_misclosure)
        yield_ratio: {network name: lambda(theta_1) / lambda_SM}
        network_tex: {network name: TeX label of its training point}
        outname: file to save the plot to
        notes: {network name: 'dropped' or 'retrain'}
        n_labels: networks labelled, largest m_n first
        per_seed: {network name: [m_n of every seed]} (network_misclosure with per_seed=True), optional
    '''
    notes = notes or {}
    names = [n for n in misclosure if n in yield_ratio]
    kind = {n: 'sm' if n == 'sm' else 'cross' if n.startswith('cross_terms/') else 'scan' for n in names}
    m = np.array([misclosure[n] for n in names])
    x = np.array([yield_ratio[n] for n in names])
    shown = np.concatenate([m] + ([np.concatenate([per_seed[n] for n in names])] if per_seed else []))
    shown = shown[shown > 0]
    low, high = shown.min() / 2, shown.max() * 2

    fig, ax = plt.subplots(figsize=(12, 9))
    # scheme A: the kind of point by marker shape, all in neutral grey; suspects filled in their status colour
    # the SM anchor is blue-green (cool against the warm suspect colours) and, with the suspects, drawn in front of
    # everything else
    base, seed_grey, anchor = '#9a9a9a', '#c8c8c8', '#0f8b8d'
    color = {n: anchor if kind[n] == 'sm' else NOTE_COLORS.get(notes.get(n), base) for n in names}
    front = {n: kind[n] == 'sm' or n in notes for n in names}
    if per_seed:
        for n, xi in zip(names, x):
            seeds = np.asarray(per_seed[n])
            seeds = seeds[seeds > 0]
            if len(seeds):
                marker = POINT_KINDS[kind[n]][0]
                c = color[n] if front[n] else seed_grey
                z = 4 if front[n] else 1
                ax.vlines(xi, seeds.min(), seeds.max(), colors=c, linewidth=1, alpha=0.6, zorder=z)
                ax.scatter(np.full(len(seeds), xi), seeds, marker=marker, s=10, color=c, alpha=0.6, zorder=z)
    for k, (marker, _, label) in POINT_KINDS.items():
        of_kind = np.array([kind[n] == k for n in names])
        size = 200 if k == 'sm' else 60
        plain = of_kind & np.array([not front[n] for n in names])
        ax.scatter(
            x[plain & (m > 0)],
            m[plain & (m > 0)],
            marker=marker,
            s=size,
            color=base,
            label=None if k == 'sm' else label,
            zorder=3,
        )
        if k == 'sm':
            sm = of_kind & (m > 0)
            ax.scatter(
                x[sm], m[sm], marker=marker, s=size, color=anchor, edgecolors='k', linewidths=0.8, label=label, zorder=6
            )
        zero = of_kind & (m <= 0)
        if zero.any():
            ax.scatter(
                x[zero],
                np.full(zero.sum(), low),
                marker=marker,
                s=size,
                facecolors='none',
                edgecolors=[color[n] for n, z in zip(names, zero) if z],
                clip_on=False,
                zorder=3,
            )
        for status, c in NOTE_COLORS.items():
            sel = of_kind & np.array([notes.get(n) == status for n in names]) & (m > 0)
            if sel.any():
                ax.scatter(
                    x[sel], m[sel], marker=marker, s=size * 1.6, color=c, edgecolors='k', linewidths=0.8, zorder=6
                )
    # legend entries for the statuses (the kinds are named by their grey markers)
    for status, c in NOTE_COLORS.items():
        if status in notes.values():
            ax.scatter([], [], marker='s', s=80, color=c, label=status)
    # labels stacked in a column at the top right, largest m_n first, each joined to its marker by a leader line
    for row, i in enumerate(np.argsort(-m)[:n_labels]):
        ax.annotate(
            network_tex[names[i]],
            (x[i], m[i]),
            xytext=(0.97, 0.95 - 0.07 * row),
            textcoords='axes fraction',
            ha='right',
            va='center',
            fontsize=14,
            arrowprops={'arrowstyle': '-', 'color': '#9a9a9a', 'linewidth': 0.8},
        )
    # the yield ratios span only about 1-1.2 (training points sized by their yield change), so x stays linear
    ax.set_yscale('log')
    ax.set_ylim(low, high)
    ax.set_xlabel(r'$\lambda(\theta_1) / \lambda(\mathrm{sm})$')
    ax.set_ylabel(r'$m_n$ / $\lambda(\mathrm{sm})$')
    ax.legend(fontsize=16, loc='upper left')
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)
