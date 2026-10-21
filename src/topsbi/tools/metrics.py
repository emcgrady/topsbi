import numpy as np


def roc_curve(f_hat, w0, w1, threshold=0.5, n_points=200):
    """
    Weighted ROC curve of the network output and its area: events weighted by w^s(theta_0) are the negative class,
    events weighted by w^s(theta_1) the positive one.

    Args:
        f_hat: network output f_hat(x) per event (any monotonic score)
        w0: w^s(x, z; theta_0) per event
        w1: w^s(x, z; theta_1) per event
        threshold: unused
        n_points: number of score thresholds
    Returns:
        fpr, tpr: false and true positive rates at each threshold
        auc: area under the ROC curve
    """
    bins = np.linspace(f_hat.min(), f_hat.max(), n_points + 1)

    w0_total = w0.sum()
    w1_total = w1.sum()

    tpr = []
    fpr = []
    for i in range(len(bins)):
        tpr += [(w1[(f_hat >= bins[-(i + 1)]).ravel()].sum() / w1_total).item()]
        fpr += [(w0[(f_hat >= bins[-(i + 1)]).ravel()].sum() / w0_total).item()]

    auc = np.trapezoid(tpr, x=fpr).item()

    return fpr, tpr, auc


def matched_closure(w_hat, bin0, i0, w, bin1, i1, n_bins):
    """
    Kinematic closure of two samples that share events, with the covariance from matched events.

    Per bin K_b, the reweighted nominal yield
    w_hat_b = sum_i r_hat(x_i; theta, nu) w^s(x_i, z_i; theta, 0) 1(x_i in K_b)
    is compared with the shifted yield w_b = sum_i w^s(x'_i, z_i; theta, nu) 1(x'_i in K_b); each sample is binned by
    its own features. Event i, in either or both samples, is one independent unit with contribution
    dw_ib = r_hat w^s(theta, 0) 1(x_i in K_b) - w^s(theta, nu) 1(x'_i in K_b), so
    Cov(Delta_b, Delta_c) = sum_i dw_ib dw_ic with Delta_b = w_hat_b - w_b. An event that moves between bins gives
    the off-diagonal terms.

    Args:
        w_hat: r_hat(x_i; theta, nu) w^s(x_i, z_i; theta, 0) per nominal row (w^s alone for the unreweighted baseline)
        bin0: bin of each nominal row, -1 outside every bin
        i0: event index i of each nominal row (matched_events)
        w: w^s(x'_i, z_i; theta, nu) per shifted row
        bin1: bin of each shifted row, -1 outside every bin
        i1: event index i of each shifted row
        n_bins: number of bins
    Returns:
        dictionary of per-bin arrays: 'w_hat', 'w', 'delta' (w_hat - w), 'cov' [n_bins, n_bins], 'w_hat_var'
        (sum_i w_hat_i^2 1(x_i in K_b)) and 'w_var' (sum_i w_i^2 1(x'_i in K_b)), each sample's own MC variance
        (the top-panel errors), and 'var_independent' = w_hat_var + w_var (the variance if the two samples were
        independent, for comparison)
    """
    bins = np.concatenate([bin0, bin1]).astype(np.int64)
    if bins.min(initial=-1) < -1 or bins.max(initial=-1) >= n_bins:
        raise ValueError(f'bins must be in [-1, {n_bins})')
    signed = np.concatenate([np.asarray(w_hat, np.float64), -np.asarray(w, np.float64)])
    events = np.concatenate([i0, i1]).astype(np.int64)
    inside = bins >= 0
    bins, signed, events = bins[inside], signed[inside], events[inside]

    # dw_ib: one entry per (event, bin); rows of the same event in the same bin are summed first
    codes, entry = np.unique(events * n_bins + bins, return_inverse=True)
    dw = np.bincount(entry, weights=signed)
    event, b = codes // n_bins, codes % n_bins

    # entries are sorted by event, so the entries of one event are adjacent; pair them at every offset
    cov = np.bincount(b * n_bins + b, weights=dw**2, minlength=n_bins**2)
    for d in range(1, len(dw)):
        same = np.nonzero(event[d:] == event[:-d])[0]
        if len(same) == 0:
            break
        p, q = same, same + d
        cov += np.bincount(b[p] * n_bins + b[q], weights=dw[p] * dw[q], minlength=n_bins**2)
        cov += np.bincount(b[q] * n_bins + b[p], weights=dw[p] * dw[q], minlength=n_bins**2)
    cov = cov.reshape(n_bins, n_bins)

    in0, in1 = bin0 >= 0, bin1 >= 0
    w_hat_b = np.bincount(bin0[in0], weights=w_hat[in0], minlength=n_bins)
    w_b = np.bincount(bin1[in1], weights=w[in1], minlength=n_bins)
    w_hat_var = np.bincount(bin0[in0], weights=w_hat[in0] ** 2, minlength=n_bins)
    w_var = np.bincount(bin1[in1], weights=w[in1] ** 2, minlength=n_bins)
    return {
        'w_hat': w_hat_b,
        'w': w_b,
        'delta': w_hat_b - w_b,
        'cov': cov,
        'w_hat_var': w_hat_var,
        'w_var': w_var,
        'var_independent': w_hat_var + w_var,
    }


def closure_summary(result, rcond=1e-10):
    """
    Summary statistics of a matched_closure result: is Delta_b = w_hat_b - w_b consistent with zero given Cov?

    chi2 = Delta^T Cov^+ Delta, with Cov^+ the pseudo-inverse over the eigenvectors of Cov whose eigenvalue exceeds
    rcond times the largest (empty bins and exactly degenerate combinations carry no information and drop out), and
    ndf the number of eigenvectors kept. For a calibrated network chi2 / ndf is about 1, Z = (chi2 - ndf) / sqrt(2 ndf)
    about 0 and rms_bias about 0. rms_bias is the w_b-weighted rms of Delta_b / w_b with the statistical scatter
    Cov_bb / w_b^2 removed; it measures the size of the misclosure and does not grow with the sample, while chi2 and
    the pulls measure its significance. The pulls are also summarised for a calibrated network's standard normal
    pulls: pull = mean |pull| - sqrt(2 / pi) (robust to a single outlying bin) and ci_diff, the expected minus the
    observed fraction of bins within 1, 2, 3 sigma (the shape of the pull distribution, including its tails); the
    _err entries are the scatter a calibrated network would show with as many bins.

    Args:
        result: output of matched_closure ('w_hat', 'w', 'delta', 'cov')
        rcond: relative eigenvalue cut of the pseudo-inverse
    Returns:
        dictionary: 'chi2', 'ndf', 'chi2_ndf', 'z', 'rms_bias', 'worst_pull' (signed), 'worst_bin', 'pulls'
        (Delta_b / sqrt(Cov_bb) per bin, NaN where Cov_bb = 0), 'pull', 'pull_err', 'ci_diff' and 'ci_diff_err'
        (lists over 1, 2, 3 sigma)
    """
    delta, cov, w = result['delta'], result['cov'], result['w']
    nan = float('nan')

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    keep = eigenvalues > rcond * eigenvalues.max(initial=0)
    projections = eigenvectors[:, keep].T @ delta
    chi2 = float((projections**2 / eigenvalues[keep]).sum())
    ndf = int(keep.sum())

    var = np.diag(cov)
    pulls = np.full(len(delta), np.nan)
    has_var = var > 0
    pulls[has_var] = delta[has_var] / np.sqrt(var[has_var])

    ok = has_var & (w > 0)
    weight = w[ok] / w[ok].sum() if ok.any() else w[ok]
    rel, rel_var = delta[ok] / w[ok], var[ok] / w[ok] ** 2
    rms_bias = np.sqrt(max((weight * rel**2).sum() - (weight * rel_var).sum(), 0)).item() if ok.any() else nan

    worst = int(np.nanargmax(np.abs(pulls))) if has_var.any() else -1

    # summaries of |pull| against the standard normal: E|Z| = sqrt(2/pi), Var|Z| = 1 - 2/pi, P(|Z| < 1, 2, 3)
    size = np.abs(pulls[has_var])
    expected = (0.6827, 0.9545, 0.9973)
    if size.size:
        pull, pull_err = float(size.mean() - np.sqrt(2 / np.pi)), float(np.sqrt((1 - 2 / np.pi) / size.size))
        ci_diff = [float(e - (size < s).mean()) for s, e in zip((1, 2, 3), expected)]
        # binomial scatter of the fraction of bins inside each interval
        ci_diff_err = [float(np.sqrt(e * (1 - e) / size.size)) for e in expected]
    else:
        pull, pull_err, ci_diff, ci_diff_err = nan, nan, [nan] * 3, [nan] * 3
    return {
        'chi2': chi2,
        'ndf': ndf,
        'chi2_ndf': chi2 / ndf if ndf else nan,
        'z': float((chi2 - ndf) / np.sqrt(2 * ndf)) if ndf else nan,
        'rms_bias': rms_bias,
        'worst_pull': float(pulls[worst]) if worst >= 0 else nan,
        'worst_bin': worst,
        'pulls': pulls,
        'pull': pull,
        'pull_err': pull_err,
        'ci_diff': ci_diff,
        'ci_diff_err': ci_diff_err,
    }


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
    # float64 sums: with float32 weights the rounding depends on the CPU, and a running n_eff close to min_neff
    # would merge differently on different machines
    weights = np.asarray(weights, dtype=np.float64)
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


def closure_metrics(
    f_hat0,
    nominal,
    f_hat1,
    shifted,
    features_config,
    n_R_bins=50,
    n_group_bins=5,
    group_cols=(21, 19),
    group_caps={19: 4},
    min_events=200,
    min_neff=30,
):
    """
    Closure metrics of a learned ratio, the same for every training type.

    Two samples that may share events: the reference sample (x_i, w0 = w^s at the denominator hypothesis) and the
    target sample (x'_i, w1 = w^s at the numerator hypothesis). kinematic_shift: nominal and the shifted sample,
    (theta, 0) and (theta, nu), with x'_i != x_i for shifted events. Paired data (eft, weight_shift): the same events
    in both, x'_i = x_i and i0 = i1, with w1 = w^s(theta_1) or w^s(theta, nu).
    Every test compares, per bin, the reweighted reference yield w_hat_b = sum_i r_hat(x_i) w0_i 1(x_i in bin) with
    the target yield w_b = sum_i w1_i 1(x'_i in bin), each sample binned by its own features, with the matched-event
    covariance (matched_closure, closure_summary). Each test is also run with r_hat = 1 (keys 'nominal_*'): the size of
    the effect before any correction. 'auc' is the weighted AUC of f_hat between the two samples.
    (a) rate: one bin holding every event; estimates lambda(theta, nu) / lambda(theta, 0).
    (b) r_hat calibration: n_R_bins equal-count quantile bins R_b of f_hat from the nominal events.
    (c) group closure: cells of era x njets (group_caps merged) x n_group_bins bins R_b inside each group; cells with
        fewer than min_events nominal events are left out. Groups come from network inputs, so a calibrated network
        closes in every cell.
    (d) per-feature closure: each feature of features_config in its fine binning K_b, out-of-range values in the edge
        bins, merged to n_eff >= min_neff on the shifted sample (merge_sparse_bins, as the closure plots).

    Args:
        f_hat0: f_hat(x_i) = log r_hat(x_i) per reference row
        nominal: (x, w0, i0): features, weight and event index i (matched_events) of the reference rows
        f_hat1: f_hat(x'_i) per target row
        shifted: (x', w1, i1) of the target rows
        features_config: plotting configuration of every feature ('loc', 'min', 'max', 'nbins'); empty: no
            per-feature tests
    Returns:
        performance: flat dictionary for performance.yml (scalars rounded; per-feature chi2/ndf and rms_bias as
            sub-dictionaries)
        details: per-test matched_closure results, summaries and bin edges, for the plots
    """
    (x0, w0, i0), (x1, w1, i1) = nominal, shifted
    r_hat0 = np.exp(f_hat0)

    def closure(b0, b1, n_bins):
        reweighted = matched_closure(r_hat0 * w0, b0, i0, w1, b1, i1, n_bins)
        baseline = matched_closure(w0, b0, i0, w1, b1, i1, n_bins)
        return {
            'result': reweighted,
            'summary': closure_summary(reweighted),
            'nominal_result': baseline,
            'nominal_summary': closure_summary(baseline),
        }

    def to_bins(values, edges):
        return np.clip(np.searchsorted(edges, values, side='right') - 1, 0, len(edges) - 2)

    def quantile_edges(values, n_bins):
        """equal-count edges; a constant f_hat (all quantiles equal) gives one bin holding every row"""
        edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        return edges if len(edges) > 1 else np.repeat(edges, 2)

    # (a) rate
    rate = closure(np.zeros(len(f_hat0), np.int64), np.zeros(len(f_hat1), np.int64), 1)

    # (b) quantile bins R_b of f_hat, edges from the nominal events
    R_edges = quantile_edges(f_hat0, n_R_bins)
    calibration = closure(to_bins(f_hat0, R_edges), to_bins(f_hat1, R_edges), len(R_edges) - 1)
    calibration['edges'] = R_edges

    # (c) era x njets x R_b cells; each row is grouped by its own features (njets can change under the shift)
    def group_of(x):
        return np.column_stack([np.minimum(x[:, c], group_caps[c]) if c in group_caps else x[:, c] for c in group_cols])

    g0, g1 = group_of(x0), group_of(x1)
    cell0, cell1, cells = np.full(len(x0), -1), np.full(len(x1), -1), []
    for g in np.unique(g0, axis=0):
        in0, in1 = np.flatnonzero((g0 == g).all(1)), np.flatnonzero((g1 == g).all(1))
        q = quantile_edges(f_hat0[in0], n_group_bins)
        b0, b1 = to_bins(f_hat0[in0], q), to_bins(f_hat1[in1], q)
        for j in np.flatnonzero(np.bincount(b0, minlength=len(q) - 1) >= min_events):
            cell0[in0[b0 == j]], cell1[in1[b1 == j]] = len(cells), len(cells)
            cells.append((tuple(g.tolist()), int(j)))
    group = closure(cell0, cell1, len(cells))
    group['cells'] = cells
    group['cols'] = tuple(group_cols)

    # (d) per feature, fine binning merged to n_eff >= min_neff on the shifted sample
    per_feature = {}
    for name, p in features_config.items():
        fine = np.linspace(p['min'], p['max'], p['nbins'] + 1)
        v0 = np.clip(x0[:, p['loc']], fine[0], np.nextafter(fine[-1], fine[0]))
        v1 = np.clip(x1[:, p['loc']], fine[0], np.nextafter(fine[-1], fine[0]))
        edges = merge_sparse_bins(v1, w1, fine, min_neff)
        per_feature[name] = closure(to_bins(v0, edges), to_bins(v1, edges), len(edges) - 1)
        per_feature[name]['edges'] = edges

    # weighted AUC: every row of both samples, with zero weight in the other class; scores on the probability scale
    score = 0.5 * (1 + np.tanh(0.5 * np.concatenate([f_hat0, f_hat1])))
    zeros0, zeros1 = np.zeros(len(f_hat0)), np.zeros(len(f_hat1))
    _, _, auc = roc_curve(score, np.concatenate([w0, zeros1]), np.concatenate([zeros0, w1]))
    performance = {'auc': round(auc, 5)}
    for prefix, summary in (('', 'summary'), ('nominal_', 'nominal_summary')):
        rate_result = rate['result' if not prefix else 'nominal_result']
        performance[f'{prefix}rate_closure'] = round(float(rate_result['w_hat'][0] / rate_result['w'][0] - 1), 6)
        performance[f'{prefix}rate_pull'] = round(rate[summary]['worst_pull'], 3)
        for key, test in (('r', calibration), ('group', group)):
            s = test[summary]
            performance.update(
                {
                    f'{prefix}{key}_chi2_ndf': round(s['chi2_ndf'], 4),
                    f'{prefix}{key}_ndf': s['ndf'],
                    f'{prefix}{key}_z': round(s['z'], 3),
                    f'{prefix}{key}_rms_bias': round(s['rms_bias'], 5),
                    f'{prefix}{key}_worst_pull': round(s['worst_pull'], 3),
                }
            )
        s = calibration[summary]
        performance.update(
            {
                f'{prefix}r_pull': round(s['pull'], 5),
                f'{prefix}r_pull_err': round(s['pull_err'], 5),
                f'{prefix}r_ci_diff': [round(v, 5) for v in s['ci_diff']],
                f'{prefix}r_ci_diff_err': [round(v, 5) for v in s['ci_diff_err']],
            }
        )
        if not per_feature:
            continue
        chi2_ndf = {name: round(t[summary]['chi2_ndf'], 4) for name, t in per_feature.items()}
        worst = max(chi2_ndf, key=lambda name: chi2_ndf[name])
        performance[f'{prefix}feature_chi2_ndf'] = chi2_ndf
        performance[f'{prefix}feature_rms_bias'] = {
            name: round(t[summary]['rms_bias'], 5) for name, t in per_feature.items()
        }
        performance[f'{prefix}feature_worst'] = worst
        performance[f'{prefix}feature_worst_pull'] = round(per_feature[worst][summary]['worst_pull'], 3)
    details = {'rate': rate, 'r': calibration, 'group': group, 'features': per_feature}
    return performance, details
