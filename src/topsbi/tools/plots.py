from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
from mpl_toolkits.axes_grid1 import make_axes_locatable
from topsbi.tools.metrics import roc_curve, closure_metrics
from topsbi.tools.ensembleCache import is_cross_term

import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh

import logging, os, re, torch, yaml, hist


# tick labels of the closure groups (era, njets with >= 4 merged): {column: (name, {value: text})}
def animate_plots(plots, outname, fps=5, sort=True):
    """
    Create an animation from a list of plots.

    Args:
        plots: list of file names for the plots to be included in the animation
        outname: name for saving the animation
        fps: frames per second for the animation
        sort: order the frames by file name; False keeps the given order (and repeated frames)
    """
    plots = [p for p in plots if p.endswith('.png')]
    plots = sorted(plots) if sort else plots
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


def kinematic_ratio_plot(x: np.array, w_hat_dedicated: np.array, w_hat_parametric: np.array, w: np.array, **params):
    """
    Kinematic closure of a dedicated and a parametric training: their reweighted bin totals w_hat_b against the
    direct w_b, with the ratio w_b / w_hat_b.

    Args:
        x: Kinematic to be plotted
        w_hat_dedicated: r_hat(x_i; theta_1) w^s(x_i, z_i; theta_0) of the dedicated network, per event
        w_hat_parametric: r_hat(x_i; theta_1) w^s(x_i, z_i; theta_0) of the parametric network, per event
        w: w^s(x_i, z_i; theta_1) per event
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

    correct_hist.fill(correct=x, weight=w)
    dedicated_hist.fill(dedicated=x, weight=w_hat_dedicated)
    parametric_hist.fill(parametric=x, weight=w_hat_parametric)

    # calculate error
    cNum, bins = correct_hist.to_numpy()
    dNum = dedicated_hist.values()
    pNum = parametric_hist.values()
    cErr = []
    dErr = []
    pErr = []

    for i in range(params['nbins'] - 1):
        cErr.append((w[(x >= bins[i]) & (x < bins[i + 1])] ** 2).sum())
        dErr.append((w_hat_dedicated[(x >= bins[i]) & (x < bins[i + 1])] ** 2).sum())
        pErr.append((w_hat_parametric[(x >= bins[i]) & (x < bins[i + 1])] ** 2).sum())
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


def loss_plot(train_loss: list[float], test_loss: list[float], best_epoch: int = None, outname: str = 'loss.png'):
    """
    Training and test loss over epochs, each relative to its value at the best network (the lowest test loss, saved
    as model.pt), L - L_best, and the loss history as an .npz next to the plot (train, test; entry 0 before
    training), so the plot can be redrawn without retraining.

    The two samples have different loss levels (their events differ), which the subtraction removes. The y axis is
    linear up to the largest change after the first epoch and logarithmic beyond it, so the first epoch's drop and the
    much smaller later changes are both visible. The test curve is 0 at the best network and above it elsewhere; overfitting shows as the test curve rising after the best
    network while the training curve keeps falling.

    Args:
        train_loss: loss for training data, entry 0 before training and entry i after epoch i - 1 (at least 2 entries)
        test_loss: loss for testing data, as train_loss
        best_epoch: epoch of the best network; None for the intermediate summaries (the lowest test loss so far)
        outname: file to write; the history goes to the same name with .npz
    """
    train, test = np.asarray(train_loss, dtype=np.float64), np.asarray(test_loss, dtype=np.float64)
    np.savez(os.path.splitext(outname)[0] + '.npz', train=train, test=test)
    epochs = np.arange(len(train))
    # the network after best_epoch's updates is entry best_epoch + 1
    best = best_epoch + 1 if best_epoch is not None else int(np.argmin(test[1:])) + 1

    train, test = train - train[best], test - test[best]
    # symmetric log scale: linear up to the largest change after the first epoch, logarithmic beyond it, so the
    # first epoch's drop and the much smaller later changes are both visible
    later = np.abs(np.concatenate([train[1:], test[1:]]))
    linthresh = later.max() if later.size and later.max() > 0 else max(np.abs(train[0]), 1e-12)

    mh.style.use('CMS')
    fig, ax = plt.subplots()
    ax.plot(epochs, train, label='Train', linewidth=3)
    ax.plot(epochs, test, label='Test', linewidth=3)
    ax.set_yscale('symlog', linthresh=linthresh)
    ax.axhline(0, color='grey', linestyle=':')
    ax.axvline(best, color='grey', linestyle='--', label='Best epoch' + ('' if best_epoch is not None else ' so far'))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$L - L_{\mathrm{best}}$')
    ax.legend()
    mh.cms.label('Preliminary', data=False, lumi=None, com=None, ax=ax)
    fig.savefig(outname, bbox_inches='tight')
    plt.close(fig)


def lr_curve(
    ax: Axes,
    lr_history: list[float],
) -> None:
    """
    Plot the learning rate over epochs, on a log scale when it changes by more than a factor 10 (the plateau
    scheduler's steps of factor 0.1, a cosine decay to eta_min).

    Args:
        ax: The matplotlib Axes to plot on.
        lr_history: learning rate per epoch
    """
    lr = np.asarray(lr_history, dtype=np.float64)
    ax.plot(lr, linewidth=3, color='tab:orange')
    if lr.size and lr.min() > 0 and lr.max() / lr.min() > 10:
        ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')


def training_plots(net, train_loss, test_loss, label, lr_history=None, performance=None):
    """
    Loss and learning-rate curves and the network for trainings without a per-event true ratio r(x, z; theta, nu)
    (kinematic_shift).

    Args:
        net: trained Net, saved as <label>/model.pt
        train_loss: loss on the training split per epoch
        test_loss: loss on the test split per epoch
        label: output directory
        lr_history: learning rate per epoch
        performance: entries written to <label>/performance.yml (written only if given)
    """
    mh.style.use("CMS")
    os.makedirs(f'{label}', mode=0o755, exist_ok=True)
    torch.save(net.state_dict(), f'{label}/model.pt')

    # loss and learning rate only once an epoch has run (the epoch-0 summary has nothing to show)
    if len(train_loss) > 1:
        loss_plot(train_loss, test_loss, (performance or {}).get('best_epoch'), f'{label}/loss.png')

    if lr_history is not None and len(train_loss) > 1:
        fig, ax = plt.subplots()
        lr_curve(ax, lr_history)
        mh.cms.label("Preliminary", data=False, lumi=None, com=None, ax=ax)
        fig.savefig(f'{label}/learning_rate.png', bbox_inches='tight')
        plt.close(fig)

    if performance is not None:
        with open(f'{label}/performance.yml', 'w') as f:
            f.write(yaml.dump(performance))


def group_pull_plot(pulls, cell_groups, group_cols, group_labels, xlabel, ylabel, outname):
    """
    Pull of every closure cell, cells ordered by group, with +-1 and +-2 sigma bands and one tick label per group.

    Args:
        pulls: pull of each cell
        cell_groups: group (tuple of group-column values) of each cell, cells of one group adjacent
        group_cols: feature columns that define the groups
        group_labels: {column: (name, {value: text})} for the group tick labels
        xlabel, ylabel: axis labels
        outname: file to write
    """
    fig, ax = plt.subplots()
    ax.axhspan(-2, 2, color='#E6EEE6', label=r'$\pm 2\sigma$')
    ax.axhspan(-1, 1, color='#4086B8', alpha=0.6, label=r'$\pm 1\sigma$')
    ax.plot(pulls, '.k')
    starts = [i for i in range(len(cell_groups)) if i == 0 or cell_groups[i] != cell_groups[i - 1]] + [len(cell_groups)]
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
        [describe(cell_groups[s]) for s in starts[:-1]],
        rotation=90,
        fontsize=14,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    fig.savefig(outname, bbox_inches='tight')
    plt.clf()
    plt.close()


def network_plots(
    features,
    w0,
    w1,
    net,
    train_loss,
    test_loss,
    label,
    lr_history=None,
    group_cols=(21, 19),
    group_caps={19: 4},
    group_labels=None,
    performance=None,
    features_config=None,
    hypothesis=r'\theta',
    lumi=None,
):
    """
    Final plots and metrics of a training with a per-event true ratio r(x, z; theta_1) = w^s(theta_1) / w^s(theta_0)
    (EFT methods and weight_shift), evaluated on the given split.

    Writes <label>/model.pt, the loss and learning-rate curves, roc.png, r_spread.png (the per-event spread of r
    around r_hat(x; theta_1) in the bins R_b, r_spread_plot), r_closure.png, group_closure.png, and
    <label>/performance.yml with the given performance entries (e.g. best_epoch, best_test_loss) and the closure
    metrics shared by every training type (closure_metrics with every event in both samples: auc, rate_*, r_*
    including r_pull and r_ci_diff, group_*, feature_* and their nominal_* baselines).
    Events where r_hat is not finite (probability output exactly 1) are left out.

    Args:
        features: raw network inputs x
        w0: w^s(x, z; theta_0) per event
        w1: w^s(x, z; theta_1) per event
        net: trained Net
        train_loss: loss on the training split per epoch
        test_loss: loss on the test split per epoch
        label: output directory
        lr_history: learning rate per epoch
        group_cols: feature columns that define the closure groups (network inputs only)
        group_caps: {column: cap}, values above the cap are merged into it (e.g. njets >= 4)
        group_labels: {column: (name, {value: text})} for the group tick labels (None: the plain values)
        performance: entries written to performance.yml along with the metrics
        features_config: plotting configuration of every feature for the per-feature closure (None: not computed)
        hypothesis: hypothesis in the closure-plot labels (theta for eft, theta, nu for weight_shift)
        lumi: integrated luminosity for the CMS label of r_closure.png
    Returns:
        the closure_metrics details, for the per-feature closure plots (closure_plot)
    """
    mh.style.use("CMS")
    os.makedirs(f'{label}', mode=0o755, exist_ok=True)
    torch.save(net.state_dict(), f'{label}/model.pt')

    performance = dict(performance or {})
    # raw output f_hat (probability or log ratio) is the ROC score; r_hat comes from the network's own output type
    with torch.no_grad():
        f_hat = net(features).cpu().numpy().flatten()
        r_hat = net.ratio(features).cpu().numpy()
        log_r_hat = net.log_ratio(features).cpu().numpy()
    w0 = w0.detach().cpu().numpy().flatten()
    w1 = w1.detach().cpu().numpy().flatten()

    # a probability network gives an infinite ratio where its output is exactly 1
    finite = np.isfinite(r_hat)
    f_hat, r_hat, log_r_hat, w0, w1 = f_hat[finite], r_hat[finite], log_r_hat[finite], w0[finite], w1[finite]

    # r(x, z; theta_1) = w^s(theta_1) / w^s(theta_0) per event
    r = w1 / w0
    w0_norm = w0 / w0.mean()

    # closure metrics shared with kinematic_shift: every event is in both samples (i0 = i1); closure groups come from
    # network inputs, era (21) and njets (19, >= 4 merged) by default
    feats_np = features.detach().cpu().numpy()[finite]
    events = np.arange(len(w0))
    closure, details = closure_metrics(
        log_r_hat,
        (feats_np, w0, events),
        log_r_hat,
        (feats_np, w1, events),
        features_config or {},
        group_cols=group_cols,
        group_caps=group_caps,
    )

    # loss and learning rate only once an epoch has run (the epoch-0 summary has nothing to show)
    if len(train_loss) > 1:
        loss_plot(train_loss, test_loss, (performance or {}).get('best_epoch'), f'{label}/loss.png')

    if lr_history is not None and len(train_loss) > 1:
        fig, ax = plt.subplots()
        lr_curve(ax, lr_history)
        mh.cms.label("Preliminary", data=False, lumi=None, com=None, ax=ax)
        fig.savefig(f'{label}/learning_rate.png', bbox_inches='tight')
        plt.clf()
        plt.close()

    # ROC curve of the raw output (the auc in performance.yml comes from closure_metrics)
    fpr, tpr, _ = roc_curve(f_hat, w0, w1)

    # make ROC curves
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='Network Performance', linewidth=3)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Baseline', linewidth=2)
    ax.legend()
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    mh.cms.label("Preliminary", data=False, lumi=None, com=None, ax=ax)
    fig.savefig(f'{label}/roc.png', bbox_inches='tight')
    plt.clf()
    plt.close()

    # per-event spread of the true ratio around the learned one, in the bins R_b of the closure test
    r_spread_plot(r, log_r_hat, w0_norm, details['r']['edges'], hypothesis, f'{label}/r_spread.png')

    # closure tests shared with kinematic_shift: r_closure plot and group pulls
    performance.update(closure)
    r_closure_plot(details['r'], label, hypothesis, lumi=lumi)
    group_pull_plot(
        details['group']['summary']['pulls'],
        [cell[0] for cell in details['group']['cells']],
        details['group']['cols'],
        group_labels or {},
        rf'Groups quantile binned in $\hat{{r}}(x;{hypothesis})$',
        rf'$\left[\hat{{w}}_b({hypothesis}) - w_b({hypothesis})\right]\,/\,\sqrt{{\mathrm{{Var}}}}$',
        f'{label}/group_closure.png',
    )

    with open(f'{label}/performance.yml', 'w') as f:
        f.write(yaml.dump(performance))
    return details


def r_mean_plot(ax: Axes, r_hat: np.array, r: np.array, w0: np.array, nbins: int, threshold: float = 0.01):
    """
    Plot r_bar_b, the w^s(theta_0)-weighted mean of the true ratio r(x, z; theta_1), against r_hat_bar_b, the mean of
    the learned ratio r_hat(x; theta_1), in quantile bins R_b of r_hat, with the uncertainty sqrt(Var(r_bar_b)).

    If the learned ratio is perfect, the points lie on the y = x line.

    Args:
        ax: The matplotlib Axes to plot on.
        r_hat: learned ratio r_hat(x; theta_1) for each event.
        r: true ratio r(x, z; theta_1) = w^s(theta_1) / w^s(theta_0) for each event.
        w0: w^s(x, z; theta_0) for each event.
        nbins: The number of quantile bins to use.
        threshold: fraction of events with the lowest and highest r_hat(x; theta_1) left out of the bins.
    """
    R_edges = np.quantile(r_hat, np.linspace(threshold, 1 - threshold, nbins + 1))

    sumw, _ = np.histogram(r_hat, bins=R_edges, weights=w0)
    sumwr, _ = np.histogram(r_hat, bins=R_edges, weights=w0 * r)
    sumw2r2, _ = np.histogram(r_hat, bins=R_edges, weights=w0**2 * r**2)
    sumw2r, _ = np.histogram(r_hat, bins=R_edges, weights=w0**2 * r)
    sumw2, _ = np.histogram(r_hat, bins=R_edges, weights=w0**2)
    sumxw, _ = np.histogram(r_hat, bins=R_edges, weights=w0 * r_hat)

    r_bar_b = sumwr / sumw
    r_bar_b_err = np.sqrt(sumw2r2 - 2 * sumw2r * r_bar_b + sumw2 * r_bar_b**2) / sumw

    # remove negative weights from early training
    r_bar_b_err[r_bar_b_err < 0] = 0

    r_hat_bar_b = sumxw / sumw
    xerr = abs(np.stack([r_hat_bar_b - R_edges[:-1], R_edges[1:] - r_hat_bar_b], axis=0))

    ax.errorbar(r_hat_bar_b, r_bar_b, xerr=xerr, yerr=r_bar_b_err, fmt='.')

    chiSquare = (r_hat_bar_b - r_bar_b) ** 2  # /(r.max() - r.min())**2
    chiSquare = chiSquare[~np.isnan(chiSquare)]
    chiSquare = (chiSquare.sum()).item()

    ax.plot([R_edges[0], R_edges[-1]], [R_edges[0], R_edges[-1]], color='grey', linestyle='--', label='_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\left\langle \hat{r} \right\rangle_b(\theta_1)$')
    ax.set_ylabel(r'$\left\langle r \right\rangle_b(\theta_1)$')

    return chiSquare


def s_mean_plot(ax: Axes, s_hat: np.array, w0: np.array, w1: np.array, nbins: int, threshold: float = 0.01):
    """
    Plot s_bar_b, the mean of the true score s(x, z; theta_1) weighted by w^s(theta_0) + w^s(theta_1), against
    s_hat_bar_b, the mean of the learned score s_hat(x; theta_1), in quantile bins S_b of s_hat, with the uncertainty
    sqrt(Var(s_bar_b)).

    If the learned score is perfect, the points lie on the y = x line.

    Args:
        ax: The matplotlib Axes to plot on.
        s_hat: learned score s_hat(x; theta_1) for each event.
        w0: w^s(x, z; theta_0) for each event.
        w1: w^s(x, z; theta_1) for each event.
        nbins: The number of quantile bins to use.
        threshold: fraction of events with the lowest and highest s_hat(x; theta_1) left out of the bins.
    """
    S_edges = np.quantile(s_hat, np.linspace(threshold, 1 - threshold, nbins + 1))

    s = w1 / (w1 + w0)
    w = w1 + w0

    sumw, _ = np.histogram(s_hat, bins=S_edges, weights=w)
    sumws, _ = np.histogram(s_hat, bins=S_edges, weights=w * s)
    sumw2s2, _ = np.histogram(s_hat, bins=S_edges, weights=w**2 * s**2)
    sumw2s, _ = np.histogram(s_hat, bins=S_edges, weights=w**2 * s)
    sumw2, _ = np.histogram(s_hat, bins=S_edges, weights=w**2)
    sumxw, _ = np.histogram(s_hat, bins=S_edges, weights=w * s_hat)

    s_bar_b = sumws / sumw
    s_bar_b_err = np.sqrt(sumw2s2 - 2 * sumw2s * s_bar_b + sumw2 * s_bar_b**2) / sumw

    # remove negative weights from early training
    s_bar_b_err[s_bar_b_err < 0] = 0

    s_hat_bar_b = sumxw / sumw
    xerr = abs(np.stack([s_hat_bar_b - S_edges[:-1], S_edges[1:] - s_hat_bar_b], axis=0))

    ax.errorbar(s_hat_bar_b, s_bar_b, xerr=xerr, yerr=s_bar_b_err, fmt='.')

    chiSquare = (s_hat_bar_b - s_bar_b) ** 2  # /(s.max() - s.min())**2
    chiSquare = chiSquare[~np.isnan(chiSquare)]
    chiSquare = (chiSquare.sum()).item()

    ax.plot([S_edges[0], S_edges[-1]], [S_edges[0], S_edges[-1]], color='grey', linestyle='--', label='_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\left\langle \hat{s} \right\rangle_b(\theta_1)$')
    ax.set_ylabel(r'$\left\langle s \right\rangle_b(\theta_1)$')

    return chiSquare


def compareDistributions(ax: Axes, pred: np.array, true: np.array, **plotParams):
    """
    Scatter plot of log r_hat(x; theta_1) against log r(x, z; theta_1).

    Args:
        ax: matplotlib axis to plot on
        pred: log r_hat(x; theta_1) per event
        true: log r(x, z; theta_1) per event
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
    ax.set_xlabel(r'Learned $\log\hat{r}(x;\theta_1)$')
    ax.set_ylabel(r'Calculated $\log r(x,z;\theta_1)$')


def hist2d(ax: Axes, pred: np.array, true: np.array, weights: np.array, logbins: bool = False, **plotParams):
    """
    Weighted 2D histogram of log r_hat(x; theta_1) against log r(x, z; theta_1).

    Args:
        ax: matplotlib axis to plot on
        pred: log r_hat(x; theta_1) per event
        true: log r(x, z; theta_1) per event
        weights: weights used for plotting, e.g. w^s(theta_0)
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
    ax.set_xlabel(r'$\log\hat{r}(x;\theta_1)$')
    ax.set_ylabel(r'$\log r(x,z;\theta_1)$')
    return h


def f_calibration_curve(
    ax: Axes, log_r: np.array, f_hat: np.array, w0_norm: np.array, nbins: int, threshold: float = 0.01
):
    """
    Calibration of a log-ratio network: the w^s(theta_0)-weighted mean of the true log r(x, z; theta_1) against the
    mean of f_hat(x) = log r_hat(x; theta_1), in quantile bins of f_hat.

    Args:
        ax: The matplotlib Axes to plot on.
        log_r: log r(x, z; theta_1) for each event
        f_hat: network output f_hat(x) for each event
        w0_norm: w^s(theta_0) / mean w^s(theta_0) for each event
        nbins: number of quantile bins
        threshold: quantile threshold to exclude outliers
    Returns:
        sum of the squared bin differences, normalised by the range of log r
    """
    f_edges = np.quantile(f_hat, np.linspace(threshold, 1 - threshold, nbins + 1))

    sumw, _ = np.histogram(f_hat, bins=f_edges, weights=w0_norm)
    sumwf, _ = np.histogram(f_hat, bins=f_edges, weights=w0_norm * log_r)
    sumw2f2, _ = np.histogram(f_hat, bins=f_edges, weights=w0_norm**2 * log_r**2)
    sumw2f, _ = np.histogram(f_hat, bins=f_edges, weights=w0_norm**2 * log_r)
    sumw2, _ = np.histogram(f_hat, bins=f_edges, weights=w0_norm**2)
    sumwfhat, _ = np.histogram(f_hat, bins=f_edges, weights=w0_norm * f_hat)

    log_r_bar_b = sumwf / sumw
    log_r_bar_b_err = np.sqrt(sumw2f2 - 2 * sumw2f * log_r_bar_b + sumw2 * log_r_bar_b**2) / sumw

    f_hat_bar_b = sumwfhat / sumw
    xerr = abs(np.stack([f_hat_bar_b - f_edges[:-1], f_edges[1:] - f_hat_bar_b], axis=0))

    ax.errorbar(f_hat_bar_b, log_r_bar_b, xerr=xerr, yerr=log_r_bar_b_err, fmt='.')

    chi_square = (f_hat_bar_b - log_r_bar_b) ** 2 / (log_r.max() - log_r.min()) ** 2
    chi_square = chi_square[~np.isnan(chi_square)]
    chi_square = (chi_square.sum()).item()

    ax.plot([f_edges[0], f_edges[-1]], [f_edges[0], f_edges[-1]], color='grey', linestyle='--', label='_nolegend_')
    mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
    ax.set_xlabel(r'$\left\langle\hat{f}_\phi(x)\right\rangle$')
    ax.set_ylabel(r'$\left\langle\log r(x,z;\theta_1)\right\rangle$')

    return chi_square


def bin_tick_labels(edges, params=None, value_names=None):
    """
    Tick positions and labels for a closure plot drawn with equal-width bins (bin b spans [b, b + 1] on the x axis,
    whatever its width in the feature).

    Integer features (unit-width fine bins centred on integers in features.yml) get one label per bin: the value,
    'lo–hi' for a merged bin, and '≥lo' for the last bin, which also counts the values above the range; a feature
    with named values (value_names, e.g. the era column) gets one name per bin. Any other feature, or R_b in r_hat
    (params None), gets labels at 6 evenly spread edges, or 7 with a '0' at the zero crossing when the range is
    symmetric about 0. Labels use 3 significant digits, more while two shown labels would be the same (edges close
    together, e.g. r_hat near 1).

    Args:
        edges: bin edges in the feature (after merge_sparse_bins), or in r_hat
        params: the feature's features.yml entry ('loc', 'min', 'max', 'nbins'); None for edges that are not a
            feature (R_b)
        value_names: {column: {value: name}} of features with named values (train.closure_options: the eras)
    Returns:
        positions (in bin units) and labels
    """
    edges = np.asarray(edges, dtype=np.float64)
    x = np.arange(len(edges))
    integer = (
        params is not None
        and np.isclose((params['max'] - params['min']) / params['nbins'], 1)
        and np.allclose((edges - 0.5) % 1, 0)
    )
    if integer:
        positions, labels = [], []
        for b in range(len(edges) - 1):
            lo, hi = int(round(edges[b] + 0.5)), int(round(edges[b + 1] - 0.5))
            names = (value_names or {}).get(params['loc'])
            if names is not None:
                text = names.get(lo, str(lo))
            elif b == len(edges) - 2:
                text = f'≥{lo}'  # values above the range are counted in the last bin
            else:
                text = str(lo) if lo == hi else f'{lo}–{hi}'
            positions.append(b + 0.5)
            labels.append(text)
        return positions, labels
    symmetric = edges[0] < 0 < edges[-1] and abs(edges[0] + edges[-1]) <= 0.05 * (edges[-1] - edges[0])
    shown = np.unique(np.round(np.linspace(0, len(edges) - 1, min(7 if symmetric else 6, len(edges)))).astype(int))
    for precision in range(3, 8):
        tick = [
            np.format_float_positional(e, precision=precision, unique=False, fractional=False, trim='-') for e in edges
        ]
        if len({tick[i] for i in shown}) == len(shown):
            break
    positions, labels = list(x[shown].astype(float)), [tick[i] for i in shown]
    if symmetric and len(shown) == 7:
        positions[3], labels[3] = float(np.interp(0, edges, x)), '0'
    return positions, labels


def place_legend(ax, samples, rows, fontsize=17, beside=True, gap=10):
    """
    Legend with rows of text, each drawn as 'label = value' and aligned on its '=' sign, placed where legend and rows
    together cover the fewest data samples. With beside, the rows sit to the right of the legend, top-aligned with it
    (bottom-aligned for a lower placement), and the legend moves left by their width so the pair stays inside the
    axes; otherwise they are stacked under the legend on its side (right, left or centred), and a lower legend is
    lifted by their height. The upper corners, the top centre, the lower corners and the bottom centre are tried in
    turn; placements whose rows would leave the axes are skipped (all of them only if none fits); ties go to the
    earlier placement. The axis ranges are not changed.

    Args:
        ax: axes holding the labelled artists
        samples: [N, 2] points in data coordinates to avoid (lines and error bars, densely sampled); non-finite
            points are ignored
        rows: (label, value) pairs, e.g. [(r'$Z$', '3.1')]
        fontsize: font size of the legend entries and the rows
        beside: rows to the right of the legend (True) or under it (False)
        gap: space between the legend and the rows, in pixels
    Returns:
        the legend
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    samples = np.asarray(samples, dtype=np.float64)
    points = ax.transData.transform(samples[np.isfinite(samples).all(1)])
    axes_box = ax.get_window_extent(renderer)

    def size(text):
        probe = ax.text(0, 0, text, fontsize=fontsize)
        box = probe.get_window_extent(renderer)
        probe.remove()
        return box.width, box.height

    label_w = max(size(label)[0] for label, _ in rows)
    value_w = max(size(f' = {value}')[0] for _, value in rows)
    row_h = max(size(f'{label} = {value}')[1] for label, value in rows) * 1.15
    block_w, block_h = label_w + value_w, row_h * len(rows)

    def layout(loc):
        vertical, horizontal = loc.split()
        if beside:
            # the legend moves left by the rows' width wherever the pair would otherwise leave the axes
            shift = (block_w + gap) / axes_box.width
            anchor_x = {'left': 0, 'center': 0.5 - shift / 2, 'right': 1 - shift}[horizontal]
            anchor = (anchor_x, 1 if vertical == 'upper' else 0)
        else:
            # a lower legend is lifted so the rows fit under it
            lift = (block_h + gap) / axes_box.height
            anchor = (0, lift, 1, 1 - lift) if vertical == 'lower' else (0, 0, 1, 1)
        legend = ax.legend(fontsize=fontsize, loc=loc, bbox_to_anchor=anchor, bbox_transform=ax.transAxes)
        box = legend.get_window_extent(renderer)
        if beside:
            x0 = box.x1 + gap
            y1 = box.y1 if vertical == 'upper' else box.y0 + block_h
        else:
            x0 = {'left': box.x0, 'center': (box.x0 + box.x1 - block_w) / 2, 'right': box.x1 - block_w}[horizontal]
            y1 = box.y0 - gap / 2
        return legend, box, (x0, y1 - block_h, x0 + block_w, y1)

    def covered(x0, y0, x1, y1):
        return int(((points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)).sum())

    scores, fits = {}, {}
    for loc in ['upper right', 'upper left', 'upper center', 'lower right', 'lower left', 'lower center']:
        legend, box, block = layout(loc)
        legend.remove()
        fits[loc] = axes_box.x0 <= block[0] and block[2] <= axes_box.x1 and axes_box.y0 <= block[1]
        scores[loc] = covered(box.x0, box.y0, box.x1, box.y1) + covered(*block)
    candidates = [loc for loc in scores if fits[loc]] or list(scores)
    legend, _, (x0, y0, x1, y1) = layout(min(candidates, key=scores.get))
    to_axes = ax.transAxes.inverted()
    for i, (label, value) in enumerate(rows):
        x_eq, y = to_axes.transform((x0 + label_w, y1 - (i + 0.5) * row_h))
        ax.text(x_eq, y, label, transform=ax.transAxes, ha='right', va='center_baseline', fontsize=fontsize)
        ax.text(x_eq, y, f' = {value}', transform=ax.transAxes, ha='left', va='center_baseline', fontsize=fontsize)
    return legend


def closure_plot(closure, params, outname, hypothesis, lumi=None, epoch=None, ylims=None, value_names=None, title=None):
    """
    Closure of one binned test of closure_metrics, drawn on equal-width bins labelled by their edges (bin_tick_labels).

    Top: the target yield w_b (black) and the reweighted reference yield w_hat_b (blue) per bin, each with its own MC
    error sqrt(sum w_i^2); log scale when the yields span more than a factor 50. Bottom: the relative misclosure
    Delta_b / w_b (filled) and the uncorrected residual with r_hat = 1 (hollow) inside the band +-sqrt(Cov_bb) / w_b of
    the matched covariance (events in both samples counted once). The bottom range is set by the filled points;
    points beyond 0.9 of it are drawn as arrows at its edge. Bins with w_b = 0 are hatched. The legend carries
    chi^2/ndf, Delta_rms and Z of the test.

    With epoch (or title) the plot is an animation frame titled by the epoch (or the title) instead of the CMS label,
    and ylims fixes both y ranges, so the axes and the marker / arrow choice do not change between frames. No tick
    label sits on the boundary between the panels: major ticks in the lowest 4% of the top panel and the highest 4%
    of the bottom panel are dropped (in axis coordinates, so log panels are covered too).

    Args:
        closure: one closure_metrics details entry ('result', 'nominal_result', 'summary', 'edges'), e.g.
            details['features'][name]
        params: the feature's features.yml entry ('label', optional 'unit', and 'loc', 'min', 'max', 'nbins' for the
            tick labels); for R_b, a dictionary with 'label' only (r_closure_plot)
        outname: file to write
        hypothesis: TeX of the hypothesis in the labels, e.g. r'\\theta,\\nu' or r'\\theta'
        lumi: integrated luminosity in fb^-1 for the CMS label (final plots only)
        epoch: epoch of an animation frame, None for a final plot
        ylims: ((top_low, top_high), (-span, span)) from the first frame; None to choose them
        value_names: {column: {value: name}} of features with named values, for the tick labels (bin_tick_labels)
        title: title of an animation frame, replacing 'Epoch NNNN' (e.g. the best network)
    Returns:
        the y ranges used, for the next frames
    """
    result, baseline, summary = closure['result'], closure['nominal_result'], closure['summary']
    edges = closure['edges']
    x = np.arange(len(edges))
    centers = x[:-1] + 0.5
    w, w_hat = result['w'], result['w_hat']
    w_err, w_hat_err = np.sqrt(result['w_var']), np.sqrt(result['w_hat_var'])
    empty = w <= 0

    def relative(values):
        return np.divide(values, w, where=~empty, out=np.full(len(w), np.nan))

    rel, rel0 = relative(result['delta']), relative(baseline['delta'])
    band = relative(np.sqrt(np.diag(result['cov'])))

    mh.style.use('CMS')
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 9), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0}
    )
    top.stairs(w, x, color='k', linewidth=2, label=rf'$w_b({hypothesis})$')
    top.stairs(w_hat, x, color=ENSEMBLE_COLOR, linewidth=2, label=rf'$\hat{{w}}_b({hypothesis})$')
    top.errorbar(centers, w, yerr=w_err, fmt='none', color='k', linewidth=2)
    top.errorbar(centers, w_hat, yerr=w_hat_err, fmt='none', color=ENSEMBLE_COLOR, linewidth=2)
    positive = w[w > 0]
    if positive.size and positive.max() / positive.min() > 50:
        top.set_yscale('log')
    top.set_ylabel('Expected events / bin')

    if ylims is None:
        # range set by the corrected residuals (filled points); uncorrected points beyond it become arrows
        largest = np.nanmax(np.abs(rel), initial=0)
        span = 1.25 * (largest if largest > 0 else np.nanmax(band, initial=0) or 1e-3)
    else:
        span = ylims[1][1]
    bottom.stairs(band, x, baseline=-band, fill=True, color='#d9d9d9')
    for values, face in ((rel0, 'none'), (rel, 'k')):
        # arrows for points beyond the panel's fixed range (not the current frame's largest residual), so a point
        # drawn as a marker stays one in every animation frame
        inside = np.abs(values) <= 0.9 * span
        outside = ~inside & np.isfinite(values)
        bottom.plot(centers[inside], values[inside], 'o', mfc=face, color='k', markersize=7)
        for c, v in zip(centers[outside], values[outside]):
            bottom.plot(c, np.sign(v) * span * 0.94, '^' if v > 0 else 'v', mfc=face, color='k', markersize=9)
    bottom.axhline(0, color='k', linestyle='--', linewidth=1)
    for b in np.flatnonzero(empty):
        for ax in (top, bottom):
            ax.axvspan(x[b], x[b + 1], facecolor='none', edgecolor='#9a9a9a', hatch='//', linewidth=0)
    bottom.set_ylabel(r'$\Delta_b / w_b$')
    unit = params.get('unit')
    bottom.set_xlabel(params['label'] + (f' [{unit}]' if unit else ''))
    positions, labels = bin_tick_labels(edges, params if 'loc' in params else None, value_names)
    bottom.set_xticks(positions, labels)
    bottom.set_xticks(x, minor=True)
    bottom.set_xlim(x[0], x[-1])

    if ylims is None:
        low, high = top.get_ylim()
        if top.get_yscale() == 'log':
            # room above the distribution for the legend and the metrics
            high = high * (high / low) ** 0.05
        else:
            # linear: the range follows the data (stairs would otherwise pull it down to 0), with room on top
            ends = np.concatenate([w - w_err, w_hat - w_hat_err, w + w_err, w_hat + w_hat_err])
            data_low, data_high = np.nanmin(ends), np.nanmax(ends)
            pad = data_high - data_low
            low, high = data_low - 0.05 * pad, data_high + 0.3 * pad
        ylims = ((low, high), (-span, span))
    top.set_ylim(ylims[0])
    bottom.set_ylim(ylims[1])

    def clear_of_boundary(ax, t, low_edge):
        # position of tick t in axis coordinates; the panels meet at the top panel's low edge
        lo, hi = ax.get_ylim()
        scale = np.log10 if ax.get_yscale() == 'log' else float
        u = (scale(t) - scale(lo)) / (scale(hi) - scale(lo))
        return 0 <= u <= 1 and (u > 0.04 if low_edge else u < 0.96)

    for ax, low_edge in ((top, True), (bottom, False)):
        ax.set_yticks([t for t in ax.get_yticks() if clear_of_boundary(ax, t, low_edge)])

    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        if title is not None:
            top.set_title(title, fontsize=22)
        elif epoch is None:
            # the yields are expected events at the luminosity of the eras in the sample
            mh.cms.label('Preliminary', data=False, lumi=lumi, com=13, ax=top)
        else:
            top.set_title(f'Epoch {epoch:04d}', fontsize=22)
        # the legend block avoids the stairs (sampled along each bin) and the error bars
        sampled = [np.c_[np.linspace(x[b], x[b + 1], 20), np.full(20, v[b])] for v in (w, w_hat) for b in range(len(v))]
        sampled += [
            np.c_[np.full(20, c), np.linspace(v - e, v + e, 20)]
            for values, errors in ((w, w_err), (w_hat, w_hat_err))
            for c, v, e in zip(centers, values, errors)
        ]
        rows = [
            (r'$\chi^2/\mathrm{ndf}$', f'{summary["chi2"]:.1f} / {summary["ndf"]}'),
            (r'$\Delta_{\mathrm{rms}}$', f'{100 * summary["rms_bias"]:.2f}%'),
            (r'$Z$', f'{summary["z"]:.1f}'),
        ]
        place_legend(top, np.concatenate(sampled), rows)
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)
    return ylims


def r_closure_plot(calibration, label, hypothesis=r'\theta,\nu', lumi=None):
    """
    Closure in the quantile bins R_b of f_hat (details['r'] of closure_metrics), for every training type, in the
    layout of the per-feature closure plots (closure_plot): w_b and w_hat_b per bin R_b, and Delta_b / w_b inside the
    matched band, with chi^2/ndf, Delta_rms and Z of the test. The bins are labelled by their edges in
    r_hat = exp(f_hat), so the ticks are values of r_hat itself, not bin means. Writes <label>/r_closure.png.

    Args:
        calibration: details['r'] of closure_metrics ('result', 'nominal_result', 'summary', 'edges' in f_hat)
        label: output directory
        hypothesis: hypothesis in the labels (theta for eft, theta, nu for nuisances)
        lumi: integrated luminosity in fb^-1 for the CMS label
    """
    shown = dict(calibration, edges=np.exp(calibration['edges']))
    params = {'label': rf'$\hat{{r}}(x;{hypothesis})$'}
    closure_plot(shown, params, f'{label}/r_closure.png', hypothesis, lumi=lumi)


def r_spread_plot(r, f_hat, w0, edges, hypothesis, outname):
    """
    Spread of the per-event true ratio r(x, z) around the learned r_hat(x) = exp(f_hat) in the bins R_b of the
    closure test (details['r']['edges'] of closure_metrics, in f_hat), drawn on equal-width bins labelled by their
    edges in r_hat, as r_closure.png. Paired data only (r = w1 / w0 per event).

    Relative to <r>_b (means weighted by w0): the mean difference (<r>_b - <r_hat>_b) / <r>_b (black, the misclosure
    of r_closure.png with the opposite sign) with its uncertainty sqrt(Var(<r>_b - <r_hat>_b)) / <r>_b (blue), and the
    per-event spread sqrt(Var(r - r_hat))_b / <r>_b (light band): the part of the true ratio that depends on the
    latent z and cannot be learned from x, which no closure plot shows.
    Args:
        r: true ratio per event
        f_hat: learned log ratio per event
        w0: reference weight per event (any normalisation)
        edges: bin edges R_b in f_hat
        hypothesis: hypothesis in the labels (theta for eft, theta, nu for weight_shift)
        outname: file to write
    """
    n = len(edges) - 1
    b = np.clip(np.searchsorted(edges, f_hat, side='right') - 1, 0, n - 1)
    w0 = np.asarray(w0, dtype=np.float64)
    d = r - np.exp(f_hat)

    def sums(values):
        return np.bincount(b, weights=values, minlength=n)

    with np.errstate(invalid='ignore', divide='ignore'):
        sumw = sums(w0)
        r_b = sums(w0 * r) / sumw
        d_b = sums(w0 * d) / sumw
        d_err = np.sqrt(sums(w0**2 * (d - d_b[b]) ** 2)) / sumw
        spread = np.sqrt(np.maximum(sums(w0 * d**2) / sumw - d_b**2, 0))
        mean, err, width = d_b / r_b, d_err / r_b, spread / r_b

    mh.style.use('CMS')
    fig, ax = plt.subplots()
    x = np.arange(n + 1)
    ax.stairs(
        mean + width, x, baseline=mean - width, fill=True, color='#D5E3D5', label=r'$\sqrt{\mathrm{Var}(r-\hat{r})}$'
    )
    ax.stairs(
        mean + err,
        x,
        baseline=mean - err,
        fill=True,
        color='#4086B8',
        alpha=0.6,
        label=r'$\sqrt{\mathrm{Var}(\left\langle r \right\rangle_b-\left\langle \hat{r} \right\rangle_b)}$',
    )
    ax.stairs(mean, x, color='k', label=r'$\left\langle r \right\rangle_b - \left\langle \hat{r} \right\rangle_b$')
    ax.axhline(0, color='grey', linestyle=':')
    positions, labels = bin_tick_labels(np.exp(edges))
    ax.set_xticks(positions, labels)
    ax.set_xticks(x, minor=True)
    ax.set_xlim(x[0], x[-1])
    extent = np.nanmax(np.abs(np.concatenate([mean + width, mean - width])), initial=0)
    ax.set_ylim(-1.3 * extent or -1, 1.3 * extent or 1)
    ax.set_xlabel(rf'$\hat{{r}}(x;{hypothesis})$')
    ax.set_ylabel(r'$(r - \hat{r}) / \left\langle r \right\rangle_b$')
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        ax.legend(loc='upper left', fontsize=16)
        mh.cms.label('Preliminary', data=False, lumi=None, com=13, ax=ax)
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)


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
    r'''TeX of a WC name, the convention of likelihood_scan_dev.ipynb: cQj31 -> c_{Qj}^{31}, ctGRe -> \Re[c_{tG}].'''
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


def plot_closure(k, f, sums, bias, binning, labels, outname, feature_config=None):
    '''
    Closure of one gamma component q in one feature: the learned sum hat w_q,b = sum_i w^s_i(theta_0) gamma_hat_q,i
    against its truth w_q,b per bin K_b (top), and their difference Delta relative to the SM yield lambda_K_b(theta_SM)
    with the noise band sqrt(Var_MC + Var_seeds) of the bias test (bottom). Error bars show the MC statistics of each
    sum (and the seed spread for the learned one), both at the bin centre. Bins are drawn with equal widths and
    labelled by their edges; bins without events of the component are hatched.

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
    top.stairs(truth + D, x, color=ENSEMBLE_COLOR, linewidth=2, label=r'$\sum_i w^s_i(\theta_0)\,\hat{\gamma}_{q,i}$')
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
        band,
        x,
        baseline=-band,
        fill=True,
        color='#d9d9d9',
        label=r'$\pm\sqrt{\mathrm{Var}_{\mathrm{MC}} + \mathrm{Var}_{\mathrm{seeds}}} / \lambda_{K_b}(\theta_{\mathrm{SM}})$',
    )
    bottom.plot(
        centers[~empty], (D / sm)[~empty], 'o', color='k', label=r'$\Delta / \lambda_{K_b}(\theta_{\mathrm{SM}})$'
    )
    bottom.axhline(0, color='k', linestyle='--', linewidth=1)
    # bins without events of the component: hatched in both panels
    for i in np.flatnonzero(empty):
        for ax in (top, bottom):
            ax.axvspan(x[i], x[i + 1], facecolor='none', edgecolor='#9a9a9a', hatch='//', linewidth=0)
    bottom.set_ylabel(r'$\Delta / \lambda_{K_b}(\theta_{\mathrm{SM}})$')
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
        z_max = np.nanmax(bias['Z'][k])
        place_legend(top, np.concatenate(sampled), [(r'$Z_{max}$', f'{z_max:.1f}')], fontsize=20, beside=False)
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
    # with the flagged weights the relative objective is the g_q-weighted mean of each component's variance
    # relative to the current design; with the scan-region weights it is a trace over the scan region
    if scan:
        ax.set_ylabel('scan-region expected error, relative to the current design')
    else:
        ax.set_ylabel(
            r'$\left\langle \mathrm{Var}(\hat{\gamma}_q) / \mathrm{Var}_{\mathrm{current}}(\hat{\gamma}_q) \right\rangle_g$'
        )
    ax.set_xlabel('point added')
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)


def plot_search_uncertainty(sigma0, sigma1, rms_bias, labels, outname, components=None):
    '''
    Expected uncertainty sqrt(Var(gamma_hat_q)) of the components (independent network errors) with the current design
    (black) and with the points the point search proposes (green), joined by a thin line, over the measured rms bias
    (grey bars), log scale.
    Components the design cannot determine (NaN) are marked on the upper edge, components with rms bias 0 (noise
    exceeds the bias) on the lower edge.

    Args:
        sigma0, sigma1: [K] propagated sqrt(Var(gamma_hat_q)) before and after the search, NaN for unconstrained
            components
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
    ax.set_ylabel(r'$\sqrt{\mathrm{Var}(\hat{\gamma}_q)} / \lambda_{K_b}(\theta_{\mathrm{SM}})$')
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
    Misclosure m_a of every training network a (a fraction of lambda_K_b(theta_SM), network_misclosure) against the
    size of its training point (inclusive yield relative to the SM, lambda(theta_1) / lambda(theta_SM), linear) on a
    log scale, by kind of point; suspects ringed in their
    attribution colours and the n_labels largest m_a labelled with their training point in a column on the right,
    joined by leader lines. The kind of point is shown by marker shape in neutral grey; suspects are filled in their
    status colour. With per_seed, every seed's m_a is drawn as a small marker, joined by a thin line from the smallest
    to the largest, with the seed average as the large marker.

    Args:
        misclosure: {network name: m_a} averaged over seeds (network_misclosure)
        yield_ratio: {network name: lambda(theta_1) / lambda(theta_SM)}
        network_tex: {network name: TeX label of its training point}
        outname: file to save the plot to
        notes: {network name: 'dropped' or 'retrain'}
        n_labels: networks labelled, largest m_a first
        per_seed: {network name: [m_a of every seed]} (network_misclosure with per_seed=True), optional
    '''
    notes = notes or {}
    names = [n for n in misclosure if n in yield_ratio]
    kind = {n: 'sm' if n == 'sm' else 'cross' if is_cross_term(n) else 'scan' for n in names}
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
    # labels stacked in a column at the top right, largest m_a first, each joined to its marker by a leader line
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
    ax.set_xlabel(r'$\lambda(\theta_1) / \lambda(\theta_{\mathrm{SM}})$')
    ax.set_ylabel(r'$m_a$')
    ax.legend(fontsize=16, loc='upper left')
    mathtext_log = logging.getLogger('matplotlib.mathtext')
    mathtext_log.addFilter(_no_tex_mapping)
    try:
        mh.cms.label("Preliminary", data=False, lumi=None, com=13, ax=ax)
        fig.savefig(outname, bbox_inches='tight')
    finally:
        mathtext_log.removeFilter(_no_tex_mapping)
    plt.close(fig)
