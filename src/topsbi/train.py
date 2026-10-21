from topsbi.model.net import Model, EFT_METHODS, SOFT_PLUS_METHODS
from topsbi.tools.plots import (
    network_plots,
    training_plots,
    closure_plot,
    r_closure_plot,
    group_pull_plot,
    animate_plots,
)
from topsbi.tools.metrics import closure_metrics
from topsbi.tools.data import (
    prepare_weights,
    sample_bootstrap,
    get_feature_normalization,
    load_nuisance_data,
    is_combined,
    load_eft_combined,
)

import argparse, glob, os, tqdm, torch, yaml
import numpy as np
from threadpoolctl import threadpool_limits

# config keys every nuisance training needs
NUISANCE_KEYS = ('data', 'systematic', 'direction', 'lumi', 'seed')
# era names indexed like lumi (config 'eras')
DEFAULT_ERAS = ['2016APV', '2016', '2017', '2018']


def check_config(config):
    """
    Raise before loading anything if the method is missing or unknown, a nuisance training misses a key, or the
    optional 'eras', 'closure_groups' or 'threads' are malformed.
    """
    methods = EFT_METHODS + SOFT_PLUS_METHODS
    if config.get('method') not in methods:
        raise ValueError(f"config 'method' must be one of {methods}, got {config.get('method')!r}")
    if config['method'] in SOFT_PLUS_METHODS:
        missing = [key for key in NUISANCE_KEYS if key not in config]
        if missing:
            raise KeyError(f"method {config['method']!r} needs {missing}")
    if 'eras' in config and 'lumi' in config and len(config['eras']) != len(config['lumi']):
        raise ValueError(f"config 'eras' has {len(config['eras'])} names but 'lumi' {len(config['lumi'])} values")
    locs = []
    for group in config.get('closure_groups', []):
        if not isinstance(group, dict) or set(group) - {'loc', 'label', 'cap'} or not isinstance(group.get('loc'), int):
            raise ValueError(f"each 'closure_groups' entry is {{loc: int, label: str, cap: int}}, got {group!r}")
        locs.append(group['loc'])
    if len(set(locs)) != len(locs):
        raise ValueError(f"'closure_groups' repeats a column: {locs}")
    threads = config.get('threads')
    if threads is not None and (not isinstance(threads, int) or threads < 1):
        raise ValueError(f"config 'threads' must be a positive integer, got {threads!r}")


def closure_options(config):
    """
    Column layout of the closure tests and plots from the config: the era column (year_loc) and its names (eras), and
    the group columns (closure_groups) with their caps and tick labels.

    Default groups: the era, and njets (column 19) with values >= 4 merged.
    Args:
        config: training config (optional 'year_loc', default 21; 'eras', default DEFAULT_ERAS; 'closure_groups', a
            list of {loc, optional label, optional cap})
    Returns:
        dict: 'year_loc'; 'group_cols' (tuple) and 'group_caps' ({loc: cap}) for closure_metrics; 'group_labels'
        ({loc: (label, {value: text})}) for group_pull_plot; 'value_names' ({loc: {value: text}}, the era names) for
        the per-feature tick labels (closure_plot)
    """
    year_loc = config.get('year_loc', 21)
    value_names = {year_loc: dict(enumerate(config.get('eras', DEFAULT_ERAS)))}
    groups = config.get('closure_groups', [{'loc': year_loc}, {'loc': 19, 'label': 'jets', 'cap': 4}])
    group_caps, group_labels = {}, {}
    for group in groups:
        names = dict(value_names.get(group['loc'], {}))
        if 'cap' in group:
            group_caps[group['loc']] = group['cap']
            names[group['cap']] = f"≥{group['cap']}"
        group_labels[group['loc']] = (group.get('label', ''), names)
    return {
        'year_loc': year_loc,
        'group_cols': tuple(group['loc'] for group in groups),
        'group_caps': group_caps,
        'group_labels': group_labels,
        'value_names': value_names,
    }


def load_training_data(config):
    """
    Train / test / holdout data and its kind.

    EFT methods on a combined sample (nominal directory with sample.yml): load_eft_combined (event_split, holdout
    config['holdout'] or validation.p); 'bootstrap' is ignored. Other EFT data: train.p (bootstrap split or test.p)
    and the optional config['holdout'], through prepare_weights.
    Nuisance methods: load_nuisance_data (event_split, holdout = validation.p); 'bootstrap' and 'holdout' are ignored.
    Returns:
        kind: 'paired' ({split: (x, w0, w1)}) or 'labelled' ({split: kinematic_shift_sample dict})
        splits: {'train', 'test', 'holdout'}; holdout is None for EFT methods without config['holdout']
    """
    if config['method'] in SOFT_PLUS_METHODS:
        for key in ('bootstrap', 'holdout'):
            if config.get(key):
                print(
                    f"[WARNING] {key!r} is ignored for {config['method']}: "
                    "train/test come from event_split, the holdout is validation.p"
                )
        kind = 'labelled' if config['method'] == 'kinematic_shift' else 'paired'
        return kind, load_nuisance_data(config)

    if is_combined(config['data']):
        if 'bootstrap' in config:
            print("[WARNING] 'bootstrap' is ignored for a combined sample: train/test come from event_split")
        return 'paired', load_eft_combined(config)
    if config['bootstrap']:
        train, test = sample_bootstrap(
            torch.load(f'{config["data"]}/train.p', weights_only=False, map_location=config['device']), config['seed']
        )
    else:
        test = torch.load(f'{config["data"]}/test.p', weights_only=False, map_location=config['device'])
        train = torch.load(f'{config["data"]}/train.p', weights_only=False, map_location=config['device'])
    splits = {'train': prepare_weights(train, config), 'test': prepare_weights(test, config), 'holdout': None}
    if config.get('holdout'):
        holdout = torch.load(config['holdout'], weights_only=False, map_location=config['device'])
        splits['holdout'] = prepare_weights(holdout, config)
    return 'paired', splits


def normalize_weights(kind, splits):
    """
    Divide every weight by the mean of w0 over the training split: w^s(theta_0) for EFT methods, w^s(theta, 0) (the
    nominal rows) for nuisance methods.

    One constant for all splits keeps the yield difference between the two hypotheses in the learned ratio.
    """
    norm = splits['train'][1].mean() if kind == 'paired' else splits['train']['nominal'][1].mean()
    for name, split in splits.items():
        if split is None:
            continue
        if kind == 'paired':
            splits[name] = (split[0], split[1] / norm, split[2] / norm)
        else:
            splits[name] = {key: (value[0], value[1] / norm, *value[2:]) for key, value in split.items()}


def loss_tensors(kind, split):
    """tensors passed to the loss: (x, w0, w1) for paired data, labelled rows (x, w, y) otherwise"""
    return split['rows'] if kind == 'labelled' else split


def epoch_batches(tensors, batch_size, generator):
    """
    Shuffled batches of one epoch: a permutation of the rows drawn from generator, cut into consecutive blocks of
    batch_size (the last block shorter, as DataLoader without drop_last), each block indexing every tensor.
    Args:
        tensors: equal-length tensors of the training split (loss_tensors)
        batch_size: rows per batch (config 'batchSize')
        generator: CPU torch.Generator, seeded once per training, so the batch order depends only on the seed
    Yields:
        tuple of the tensors' rows in one batch
    """
    order = torch.randperm(len(tensors[0]), generator=generator).to(tensors[0].device)
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        yield tuple(t[rows] for t in tensors)


def nominal_features(kind, split):
    """features of the nominal events, used for the input standardisation"""
    return split['nominal'][0] if kind == 'labelled' else split[0]


def closure_inputs(net, kind, split):
    """
    Inputs of closure_metrics for one split: the learned log ratio f_hat and (features, weights, event index) of the
    reference and target samples, as numpy arrays.

    Paired data: the same events twice, weighted by w0 (reference) and w1 (target), with f_hat the same in both and
    event index = row. Labelled data (kinematic_shift): the nominal rows (x, w^s(theta, 0), i) and the shifted rows
    (x', w^s(theta, nu), i), each with f_hat evaluated on its own features. Events whose f_hat is not finite (a
    probability network with output exactly 0 or 1) are left out of the reference sample, and for paired data also
    of the target, since they are the same events.
    Args:
        net: Net (trained, or during training)
        kind: 'paired' or 'labelled'
        split: data of one split (load_training_data)
    Returns:
        f_hat0, (x0, w0, i0), f_hat1, (x1, w1, i1), so closure_metrics(*closure_inputs(...), features_config)
    """
    numpy = lambda *tensors: tuple(t.detach().cpu().numpy() for t in tensors)  # noqa: E731
    if kind == 'paired':
        x, w0, w1 = split
        with torch.no_grad():
            f_hat = net.log_ratio(x).cpu().numpy()
        finite = np.isfinite(f_hat)
        x, w0, w1 = (a[finite] for a in numpy(x, w0.flatten(), w1.flatten()))
        events = np.arange(len(w0))
        return f_hat[finite], (x, w0, events), f_hat[finite], (x, w1, events)
    (x0, w0, i0), (x1, w1, i1) = split['nominal'], split['shifted']
    with torch.no_grad():
        f_hat0, f_hat1 = net.log_ratio(x0).cpu().numpy(), net.log_ratio(x1).cpu().numpy()
    finite = np.isfinite(f_hat0)
    x0, w0, i0 = (a[finite] for a in numpy(x0, w0, i0))
    return f_hat0[finite], (x0, w0, i0), f_hat1, numpy(x1, w1, i1)


def hypothesis(config):
    """TeX of the hypothesis in the closure-plot labels: theta, nu for nuisance trainings, theta for eft"""
    return r'\theta,\nu' if config['method'] in SOFT_PLUS_METHODS else r'\theta'


def plot_luminosity(config, x):
    """
    Integrated luminosity of the eras present in a sample, in fb^-1 and rounded, for the CMS label of the closure
    plots: the plotted yields are expected events at this luminosity (all four eras: 138; a variation produced for one
    era, e.g. JER_2017: that era's).
    Args:
        config: training config ('lumi' per era index, optional 'year_loc', default 21)
        x: features of the sample (numpy, e.g. x0 of closure_inputs)
    Returns:
        integer luminosity
    """
    eras = np.unique(x[:, config.get('year_loc', 21)]).astype(np.int64)
    return round(sum(config['lumi'][era] for era in eras))


def closure_frame(net, kind, split, config, features_config):
    """
    One animation frame: the closure_metrics details of config['features_to_animate'] on the given split, kept in
    memory and drawn after training (closure_animations), so the y ranges can come from the end of training.
    Args:
        net: Net being trained
        kind: 'paired' or 'labelled'
        split: data of one split (load_training_data)
        config: training config ('features_to_animate', column layout: closure_options)
        features_config: plotting configuration of every feature
    Returns:
        {feature: closure_metrics details entry}; empty without features to animate
    """
    animate = {feature: features_config[feature] for feature in config['features_to_animate']}
    if not animate:
        return {}
    groups = closure_options(config)
    _, details = closure_metrics(
        *closure_inputs(net, kind, split), animate, group_cols=groups['group_cols'], group_caps=groups['group_caps']
    )
    return details['features']


def closure_animations(frames, best, config, features_config, path, out, fps=1, hold=3):
    """
    Draw the stored frames and write one GIF per animated feature, <out>/<feature>.gif.

    The y ranges of a feature cover the best network and every frame after it (the widest top-panel range and the
    largest bottom-panel span among them, each as closure_plot chooses it), so changes after the best epoch are seen at
    their size; earlier frames that leave the range are drawn as arrows. The GIF runs through the frames in epoch
    order, one per second, and ends on the best network, held for `hold` seconds. Frames are labelled by completed
    epochs.
    Args:
        frames: [(completed epochs, closure_frame result)] in training order
        best: (completed epochs, closure_frame result) of the restored best network
        config: training config ('features_to_animate', 'method', column layout: closure_options)
        features_config: plotting configuration of every feature
        path: directory of the frame PNGs (main deletes it afterwards)
        out: directory of the GIFs
        fps: frames per second of the GIF
        hold: seconds the best frame is shown at the end
    """
    best_epoch, best_frame = best
    options = dict(hypothesis=hypothesis(config), value_names=closure_options(config)['value_names'])
    for feature in config['features_to_animate']:
        params = features_config[feature]
        os.makedirs(f'{path}/{feature}', mode=0o755, exist_ok=True)
        # y ranges: each candidate drawn once with its own ranges, then the union
        candidates = [best_frame[feature]] + [frame[feature] for epoch, frame in frames if epoch >= best_epoch]
        ranges = [closure_plot(c, params, f'{path}/{feature}/range.png', **options) for c in candidates]
        top = (min(r[0][0] for r in ranges), max(r[0][1] for r in ranges))
        span = max(r[1][1] for r in ranges)
        ylims = (top, (-span, span))
        names = []
        for epoch, frame in frames:
            names.append(f'{path}/{feature}/{epoch:04d}.png')
            closure_plot(frame[feature], params, names[-1], epoch=epoch, ylims=ylims, **options)
        best_name = f'{path}/{feature}/best.png'
        title = f'Best: epoch {best_epoch:04d}'
        closure_plot(best_frame[feature], params, best_name, ylims=ylims, title=title, **options)
        animate_plots(names + [best_name] * round(hold * fps), f'{out}/{feature}.gif', fps=fps, sort=False)


def summary_plots(
    net, kind, split, config, train_loss, test_loss, label, lr_history, features_config, lumi=None, performance=None
):
    """
    Paired data: network_plots (loss, ROC, calibration plots, the closure metrics of closure_metrics and the
    paired-only pull and ci_diff, performance.yml with the given entries).
    Labelled data (kinematic_shift): the closure metrics of closure_metrics on the given split, r_closure.png
    (r_closure_plot) and group_closure.png (group_pull_plot), then training_plots (loss and LR curves, model.pt,
    performance.yml with the given entries and the metrics).
    Args:
        net: trained Net (or the Net being trained, for the intermediate summaries)
        kind: 'paired' or 'labelled'
        split: data of one split (load_training_data)
        config: training config (method, for the hypothesis in the labels; column layout: closure_options)
        train_loss: loss on the training split per epoch
        test_loss: loss on the test split per epoch
        label: output directory
        lr_history: learning rate per epoch
        features_config: plotting configuration of every feature for the per-feature closure
        lumi: integrated luminosity for the CMS labels (plot_luminosity)
        performance: entries written to performance.yml along with the metrics
    Returns:
        the closure_metrics details, for the per-feature closure plots (closure_plot)
    """
    groups = closure_options(config)
    if kind == 'paired':
        return network_plots(
            *split,
            net,
            train_loss,
            test_loss,
            label,
            lr_history=lr_history,
            performance=performance,
            features_config=features_config,
            hypothesis=hypothesis(config),
            lumi=lumi,
            group_cols=groups['group_cols'],
            group_caps=groups['group_caps'],
            group_labels=groups['group_labels'],
        )
    metrics, details = closure_metrics(
        *closure_inputs(net, kind, split),
        features_config,
        group_cols=groups['group_cols'],
        group_caps=groups['group_caps'],
    )
    os.makedirs(label, mode=0o755, exist_ok=True)
    h = hypothesis(config)
    r_closure_plot(details['r'], label, h, lumi=lumi)
    group_pull_plot(
        details['group']['summary']['pulls'],
        [cell[0] for cell in details['group']['cells']],
        details['group']['cols'],
        groups['group_labels'],
        rf'Groups quantile binned in $\hat{{r}}(x;{h})$',
        rf'$\left[\hat{{w}}_b({h}) - w_b({h})\right]\,/\,\sqrt{{\mathrm{{Var}}}}$',
        f'{label}/group_closure.png',
    )
    print(
        f"[INFO] closure: rate {metrics['rate_closure']:+.4f} (pull {metrics['rate_pull']:+.2f}), "
        f"r_chi2_ndf {metrics['r_chi2_ndf']}, group_chi2_ndf {metrics['group_chi2_ndf']}, "
        f"worst feature {metrics['feature_worst']} ({metrics['feature_chi2_ndf'][metrics['feature_worst']]})"
    )
    training_plots(
        net, train_loss, test_loss, label, lr_history=lr_history, performance={**(performance or {}), **metrics}
    )
    return details


# gradients only inside the optimizer loop; evaluating the full training set with autograd costs ~10 GB
@torch.no_grad()
def main(config):
    """
    Train one network from a config (examples/training/README.md): load the data and weights, train with early
    stopping on the test loss, keep the best checkpoint in <name>/model.pt, and write the final plots and
    performance.yml to <name>/complete, computed on the holdout when there is one.

    Args:
        config: training configuration dictionary
    Returns:
        config (with defaults filled in)
    """
    with open(config['features'], 'r') as f:
        features_config = yaml.safe_load(f)
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        config['device'] = 'cpu'
    torch.manual_seed(config['seed'])
    check_config(config)
    if 'threads' in config:
        # torch's intra-op threads and the BLAS / OpenMP pools numpy uses (closure_metrics), for the whole process
        torch.set_num_threads(config['threads'])
        threadpool_limits(config['threads'])

    kind, splits = load_training_data(config)
    normalize_weights(kind, splits)
    train, test, holdout = splits['train'], splits['test'], splits['holdout']
    train_tensors, test_tensors = loss_tensors(kind, train), loss_tensors(kind, test)
    train_x = nominal_features(kind, train)
    groups = closure_options(config)
    outside = [loc for loc in (groups['year_loc'], *groups['group_cols']) if not 0 <= loc < train_x.shape[1]]
    if outside:
        raise ValueError(f'year_loc / closure_groups columns {outside} are not among the {train_x.shape[1]} features')

    # batch order from its own generator: it depends only on the seed, not on the network's use of random numbers
    shuffle = torch.Generator().manual_seed(config['seed'])
    model = Model(
        n_features=train_x.shape[1],
        method=config['method'],
        device=config['device'],
        config=config['network'],
        seed=config['seed'],
    )
    # the network standardises its inputs with the training-set (nominal) statistics, which are saved in model.pt
    model.net.set_feature_normalization(*get_feature_normalization(train_x))
    # paired methods duplicate each event with weights w0 / w1; kinematic_shift has one labelled row per event
    loss_fn = model.labelled_loss if kind == 'labelled' else model.loss
    optimizer = torch.optim.Adam(model.net.parameters(), lr=config['learningRate'])
    # dropout stays off for every evaluation (losses, early stopping, plots); on only for the optimizer steps
    model.net.eval()

    scheduler_type = config.get('scheduler', 'plateau')
    if scheduler_type == 'plateau':
        # ReduceLROnPlateau: steps LR down when val BCE stops improving.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('lr_patience', 5),
        )
        print(
            f"[INFO] scheduler: ReduceLROnPlateau  factor={config.get('factor', 0.5)}  lr_patience={config.get('lr_patience', 5)}"
        )
    elif scheduler_type == 'cosine':
        # Linear warmup → CosineAnnealingLR: smooth, deterministic
        warmup = config.get('warmup_epochs', 5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, config['epochs'] - warmup), eta_min=1e-6
                ),
            ],
            milestones=[warmup],
        )
        print(f"[INFO] scheduler: cosine+warmup  warmup_epochs={warmup}  T_max={max(1, config['epochs'] - warmup)}")
    else:
        scheduler = None
        print("[INFO] scheduler: none")

    train_loss = [loss_fn(*train_tensors).item()]
    test_loss = [loss_fn(*test_tensors).item()]
    lr_history = [optimizer.param_groups[0]['lr']]

    if len(glob.glob(f'{config["name"]}/complete')) > 0:
        os.system(f'rm -rf {config["name"]}/complete')
    os.makedirs(f'{config["name"]}/complete/animations')
    os.makedirs(f'{config["name"]}/complete/kinematics')
    if len(glob.glob(f'{config["name"]}/incomplete')) > 0:
        os.system(f'rm -rf {config["name"]}/incomplete')
    frames_dir = f'{config["name"]}/incomplete/kinematics'
    frames = []  # (completed epochs, closure_frame) every animate_per_epoch epochs, drawn after training
    test_lumi = plot_luminosity(config, nominal_features(kind, test).cpu().numpy())

    # early stopping parameters
    patience = config.get('patience', 10)
    best_test_loss = float('inf')
    best_epoch = 0
    patience_count = 0
    best_state = None

    for epoch in tqdm.tqdm(range(config['epochs'])):
        if epoch % config['animate_per_epoch'] == 0:
            frames.append((epoch, closure_frame(model.net, kind, test, config, features_config)))

        lr_history.append(optimizer.param_groups[0]['lr'])
        if epoch % 50 == 0:
            summary_plots(
                model.net,
                kind,
                test,
                config,
                train_loss,
                test_loss,
                f'{config["name"]}/incomplete/epoch_{epoch:04d}',
                lr_history,
                features_config,
                lumi=test_lumi,
            )
        model.net.train()
        with torch.enable_grad():
            for batch in epoch_batches(train_tensors, config['batchSize'], shuffle):
                optimizer.zero_grad()
                loss = loss_fn(*batch)
                loss.backward()
                optimizer.step()
        model.net.eval()
        # both losses after this epoch's updates, so train_loss[i] and test_loss[i] are the same network state
        train_loss.append(loss_fn(*train_tensors).item())
        current_test_loss = loss_fn(*test_tensors).item()
        test_loss.append(current_test_loss)

        # ── early stopping ──
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.net.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(
                    f"[INFO] early stopping at epoch {epoch}, best epoch was {best_epoch} (test loss {best_test_loss:.4f})"
                )
                break

        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(current_test_loss)
            else:
                scheduler.step()
    print('Training complete!')
    # the network after the last epoch's updates
    frames.append((epoch + 1, closure_frame(model.net, kind, test, config, features_config)))

    if best_state is not None:
        model.net.load_state_dict(best_state)
        print(f"[INFO] restored best checkpoint from epoch {best_epoch}")

    print('Creating animations...')
    best = (best_epoch + 1, closure_frame(model.net, kind, test, config, features_config))
    closure_animations(frames, best, config, features_config, frames_dir, f'{config["name"]}/complete/animations')
    os.system(f'rm -rf {frames_dir}')

    print('Creating final plots and saving best network..')
    # keep the best model for validation
    torch.save(model.net.state_dict(), f'{config["name"]}/model.pt')

    # final metrics on data not used for training, early stopping or the scheduler
    if holdout is not None:
        evaluation = holdout
        source = (
            f'{config["data"]}/<nominal, variation>/validation.p'
            if config['method'] in SOFT_PLUS_METHODS
            else config.get('holdout') or f'{config["data"]}/validation.p'
        )
        print(f"[INFO] final metrics on holdout {source}")
    else:
        evaluation = test
        print("[WARNING] no holdout given: final metrics use the early-stopping test split and are optimistic")

    lumi = plot_luminosity(config, nominal_features(kind, evaluation).cpu().numpy())
    details = summary_plots(
        model.net,
        kind,
        evaluation,
        config,
        train_loss,
        test_loss,
        f'{config["name"]}/complete',
        lr_history,
        features_config,
        lumi=lumi,
        performance={'best_epoch': best_epoch, 'best_test_loss': round(best_test_loss, 6)},
    )
    # per-feature closure of the final network, from the same closure_metrics run as performance.yml
    for feature, params in features_config.items():
        closure_plot(
            details['features'][feature],
            params,
            f'{config["name"]}/complete/kinematics/{feature}.png',
            hypothesis(config),
            lumi=lumi,
            value_names=groups['value_names'],
        )
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration yml file used for training')

    # Load the configuration options and build the WC lists
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    config = main(config)
