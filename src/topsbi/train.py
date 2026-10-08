from topsbi.model.net import Model
from topsbi.tools.plots import networkPlots, kinematic_histogram, animate_plots
from topsbi.tools.data import (
    parameterize_weights,
    get_probabilities,
    get_weights,
    sample_boostrap,
    get_feature_normalization,
)

import argparse, glob, os, tqdm, torch, yaml
import numpy as np


def main(config):
    with open(config['features'], 'r') as f:
        features_config = yaml.safe_load(f)
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        config['device'] = 'cpu'
    torch.manual_seed(config['seed'])

    if 'method' not in config.keys():
        config['method'] = 'stitched'

    if config['bootstrap']:
        train, test = sample_boostrap(torch.load(f'{config["data"]}/train.p', weights_only=False), config['seed'])

    else:
        test_feats, test_coefs = torch.load(f'{config["data"]}/test.p', weights_only=False)[:]
        train_feats, train_coefs = torch.load(f'{config["data"]}/train.p', weights_only=False)[:]

    if config['method'] == 'parameterized':
        train_feats, train_coefs = train[:]
        test_feats, test_coefs = test[:]
        test_p0, test_p1, test_wcs = parameterize_weights(test_coefs, config)
        train_p0, train_p1, train_wcs = parameterize_weights(train_coefs, config)
        test_feats = torch.concatenate([test_feats, test_wcs], dim=1)
        train_feats = torch.concatenate([train_feats, train_wcs], dim=1)
    elif config['method'] == 'stitched':
        train_feats, train_coefs, _ = train[:]
        # train_feats, train_coefs = train[:]
        train = None
        _ = None
        train_p0, train_p1, train_pg = get_probabilities(train_coefs, config)
        train_coefs = None
        norm_mean, norm_stdv = get_feature_normalization(train_feats)
        train_feats = (train_feats - norm_mean) / norm_stdv
        test_feats, test_coefs, _ = test[:]
        # test_feats,  test_coefs  = test[:]
        test = None
        _ = None
        test_p0, test_p1, test_pg = get_probabilities(test_coefs, config)
        test_coefs = None
        norm_test = (test_feats - norm_mean) / norm_stdv
        tlr = (test_p1 / test_p0).detach().cpu().numpy().flatten()
    elif config['method'] == 'weights_only':
        train_feats, train_coefs, _ = train[:]
        train_coefs = train_coefs.to(torch.float32)
        train_feats = train_feats.to(torch.float32)
        train = None
        _ = None
        train_p0, train_p1, train_pg = get_weights(train_coefs, config)
        train_coefs = None
        norm_mean, norm_stdv = get_feature_normalization(train_feats)
        train_feats = (train_feats - norm_mean) / norm_stdv
        test_feats, test_coefs, _ = test[:]
        test_coefs = test_coefs.to(torch.float32)
        test_feats = test_feats.to(torch.float32)
        test = None
        _ = None
        test_p0, test_p1, test_pg = get_weights(test_coefs, config)
        test_coefs = None
        norm_test = (test_feats - norm_mean) / norm_stdv
        tlr = (test_p1 / test_p0).detach().cpu().numpy().flatten()
    elif config['method'] == 'alice':
        train_feats, train_coefs = train[:]
        test_feats, test_coefs = test[:]
        test_p0, test_p1 = get_probabilities(test_coefs, config)
        train_p0, train_p1 = get_probabilities(train_coefs, config)
    elif config['method'] == 'weight_shift':
        train_feats, train_p0, train_p1, train_pg, _ = train[:]
        test_feats, test_p0, test_p1, test_pg, _ = test[:]
        train_pg /= train_pg.mean()
        train_p0 /= (train_p0.mean()) * train_pg
        train_p1 /= (train_p1.mean()) * train_pg
        test_pg /= test_pg.mean()
        test_p0 /= (test_p0.mean()) * test_pg
        test_p1 /= (test_p1.mean()) * test_pg

    batches = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_feats, train_p0, train_p1),
        batch_size=config['batchSize'],
        shuffle=True,
        num_workers=1,
    )
    model = Model(
        nFeatures=train_feats.shape[1],
        method=config['method'],
        device=config['device'],
        config=config['network'],
        seed=config['seed'],
    )
    optimizer = torch.optim.Adam(model.net.parameters(), lr=config['learningRate'])

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

    trainLoss = [model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item()]
    testLoss = [model.loss(norm_test, test_p0, test_p1).item()]
    lrHistory = [optimizer.param_groups[0]['lr']]

    if len(glob.glob(f'{config["name"]}/complete')) > 0:
        os.system(f'rm -rf {config["name"]}/complete')
    os.makedirs(f'{config["name"]}/complete/animations')
    os.makedirs(f'{config["name"]}/complete/kinematics')
    if len(glob.glob(f'{config["name"]}/incomplete')) > 0:
        os.system(f'rm -rf {config["name"]}/incomplete')
    for feature in config['features_to_animate']:
        os.makedirs(f'{config["name"]}/incomplete/kinematics/{feature}')

    # early stopping parameters
    patience = config.get('patience', 10)
    best_test_loss = float('inf')
    best_epoch = 0
    patience_count = 0
    best_state = None

    for epoch in tqdm.tqdm(range(config['epochs'])):
        f = model.net(norm_test).cpu().detach().numpy().flatten()
        if config['method'] == 'weight_shift':
            lr = np.exp(2 * f - 1)
            noOnes = np.ones(tlr.shape, dtype=bool)
        else:
            noOnes = f != 1
            f = f[noOnes]
            lr = f / (1 - f)
        for feature in config['features_to_animate']:
            params = features_config[feature]
            if epoch == 0:
                features_config[feature]['ylim_log'] = kinematic_histogram(
                    test_feats[noOnes, params['loc']].cpu().numpy(),
                    params,
                    epoch,
                    lr,
                    tlr[noOnes],
                    f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}_log.png',
                )
                features_config[feature]['ylim_linear'] = kinematic_histogram(
                    test_feats[noOnes, params['loc']].cpu().numpy(),
                    params,
                    epoch,
                    lr,
                    tlr[noOnes],
                    f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}_linear.png',
                    log=False,
                )
            elif epoch % config['animate_per_epoch'] == 0:
                kinematic_histogram(
                    test_feats[noOnes, params['loc']].cpu().numpy(),
                    params,
                    epoch,
                    lr,
                    tlr[noOnes],
                    f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}_log.png',
                    ylim=params['ylim_log'],
                )
                kinematic_histogram(
                    test_feats[noOnes, params['loc']].cpu().numpy(),
                    params,
                    epoch,
                    lr,
                    tlr[noOnes],
                    f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}_linear.png',
                    ylim=params['ylim_linear'],
                    log=False,
                )

        trainLoss.append(model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item())
        lrHistory.append(optimizer.param_groups[0]['lr'])
        if epoch % 50 == 0:
            networkPlots(
                norm_test,
                test_p0,
                test_p1,
                test_pg,
                model.net,
                trainLoss,
                testLoss,
                f'{config["name"]}/incomplete/epoch_{epoch:04d}',
                method=config['method'],
                lr_history=lrHistory,
            )
        for train_feats, train_p0, train_p1 in batches:
            optimizer.zero_grad()
            loss = model.loss(train_feats, train_p0, train_p1)
            loss.backward()
            optimizer.step()
        current_test_loss = model.loss(norm_test, test_p0, test_p1).item()
        testLoss.append(current_test_loss)

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
    print('Creating animations...')
    f = model.net(norm_test).cpu().detach().numpy().flatten()
    if config['method'] == 'weight_shift':
        lr = np.exp(2 * f - 1)
        noOnes = np.ones(tlr.shape, dtype=bool)
    else:
        noOnes = f != 1
        f = f[noOnes]
        lr = f / (1 - f)
    for feature in config['features_to_animate']:
        params = features_config[feature]
        kinematic_histogram(
            test_feats[noOnes, params['loc']].cpu().numpy(),
            params,
            epoch,
            lr,
            tlr[noOnes],
            f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}_log.png',
            ylim=params['ylim_log'],
        )
        kinematic_histogram(
            test_feats[noOnes, params['loc']].cpu().numpy(),
            params,
            epoch,
            lr,
            tlr[noOnes],
            f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}_linear.png',
            ylim=params['ylim_linear'],
            log=False,
        )

        plots = sorted(glob.glob(f'{config["name"]}/incomplete/kinematics/{feature}/*_log.png'))
        animate_plots(plots, f'{config["name"]}/complete/animations/{feature}_log.gif')
        plots = sorted(glob.glob(f'{config["name"]}/incomplete/kinematics/{feature}/*_linear.png'))
        animate_plots(plots, f'{config["name"]}/complete/animations/{feature}_linear.gif')

    print('Animations created!')
    print('deleting plots used for animations...')
    os.system(f'rm -rf {config["name"]}/incomplete/kinematics')
    print('Plots deleted!')

    if best_state is not None:
        model.net.load_state_dict(best_state)
        print(f"[INFO] restored best checkpoint from epoch {best_epoch}")

    print('Creating final plots and saving best network..')
    # keep the best model for validation
    torch.save(model.net.state_dict(), f'{config["name"]}/model.pt')

    networkPlots(
        norm_test,
        test_p0,
        test_p1,
        test_pg,
        model.net,
        trainLoss,
        testLoss,
        f'{config["name"]}/complete',
        method=config['method'],
        lr_history=lrHistory,
    )
    f = model.net(norm_test).cpu().detach().numpy().flatten()

    if config['method'] == 'weight_shift':
        lr = np.exp(2 * f - 1)
        noOnes = np.ones(tlr.shape, dtype=bool)
    else:
        noOnes = f != 1
        f = f[noOnes]
        lr = f / (1 - f)

    for feature, params in features_config.items():
        kinematic_histogram(
            test_feats[noOnes, params['loc']].cpu().numpy(),
            params,
            epoch,
            lr,
            tlr[noOnes],
            f'{config["name"]}/complete/kinematics/{feature}_log.png',
            epoch_title=False,
            scale_ratio=True,
        )
        kinematic_histogram(
            test_feats[noOnes, params['loc']].cpu().numpy(),
            params,
            epoch,
            lr,
            tlr[noOnes],
            f'{config["name"]}/complete/kinematics/{feature}_linear.png',
            log=False,
            epoch_title=False,
            scale_ratio=True,
        )

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='configuration yml file used for training')

    # Load the configuration options and build the WC lists
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    config = main(config)
