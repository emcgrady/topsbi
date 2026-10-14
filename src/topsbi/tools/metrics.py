import numpy as np


def netEval(netOut, bWeights, sWeights, threshold=0.5, nPoints=200):
    bins = np.linspace(netOut.min(), netOut.max(), nPoints + 1)

    bTotal = bWeights.sum()
    sTotal = sWeights.sum()

    tpr = []
    fpr = []
    for i in range(len(bins)):
        tpr += [(sWeights[(netOut >= bins[-(i + 1)]).ravel()].sum() / sTotal).item()]
        fpr += [(bWeights[(netOut >= bins[-(i + 1)]).ravel()].sum() / bTotal).item()]

    auc = np.trapezoid(tpr, x=fpr).item()

    return fpr, tpr, auc


def group_closure(lrhat, lr, w0, groups, n_rbins=5, min_events=200):
    """
    Closure of the learned ratio in cells of (group x lrhat quantile bin).

    For a calibrated network sum(w0 * lrhat) = sum(w0 * lr) in any set of events selected
    by the network inputs, so groups must be built from input features. Groups built from
    latent labels (e.g. the generator-level mtt slice) do not close even for a perfect network.

    Args:
        lrhat: learned likelihood ratio per event
        lr: true per-event likelihood ratio
        w0: event weight under theta_0
        groups: (n_events, n_cols) array of input-feature values defining the groups
        n_rbins: number of lrhat quantile bins inside each group
        min_events: cells with fewer events are skipped
    Returns:
        dictionary of summary metrics (0 for a calibrated network) and per-cell results
    """
    a, b = w0 * lrhat, w0 * lr
    cells = []
    for g in np.unique(groups, axis=0):
        in_g = (groups == g).all(1)
        ag, bg, lg = a[in_g], b[in_g], lrhat[in_g]
        q = np.quantile(lg, np.linspace(0, 1, n_rbins + 1))
        k = np.clip(np.searchsorted(q, lg, side='right') - 1, 0, n_rbins - 1)
        for j in range(n_rbins):
            m = k == j
            if m.sum() < min_events:
                continue
            A, B = ag[m], bg[m]
            R = A.sum() / B.sum()
            # learned and true sums share events, so propagate per-event residuals
            err = np.sqrt(((A - R * B) ** 2).sum()) / B.sum()
            cells.append((tuple(g.tolist()), j, R - 1, err, B.sum()))
    if not cells:
        nan = float('nan')
        return {
            'chi2_excess': nan,
            'chi2_excess_err': nan,
            'rms_bias': nan,
            'worst_pull': nan,
            'n_cells': 0,
            'cells': [],
            'pulls': np.array([]),
        }
    bias = np.array([c[2] for c in cells])
    err = np.array([c[3] for c in cells])
    W = np.array([c[4] for c in cells])
    pulls = bias / err
    n = len(pulls)
    return {
        'chi2_excess': ((pulls**2).mean() - 1).item(),
        'chi2_excess_err': np.sqrt(2 / n).item(),
        'rms_bias': np.sqrt(max((W * bias**2).sum() - (W * err**2).sum(), 0) / W.sum()).item(),
        'worst_pull': pulls[np.argmax(np.abs(pulls))].item(),
        'n_cells': n,
        'cells': cells,
        'pulls': pulls,
    }
