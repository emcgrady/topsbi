from topsbi.model.net import Model
from topsbi.tools.plots import networkPlots, kinematic_histogram, animate_plots
from topsbi.tools.data import parameterize_weights, get_probabilities

import argparse, glob, os, tqdm, torch, yaml

def main(config):
    with open(f'{config["data"]}/features.yml', 'r') as f:
        features_config = yaml.safe_load(f)
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        config['device'] = 'cpu'
    torch.manual_seed(config['seed'])

    test_feats,   test_coefs  = torch.load(f'{config["data"]}/test.p', weights_only=False)[:]
    train_feats,  train_coefs = torch.load(f'{config["data"]}/train.p', weights_only=False)[:]
    
    if 'method' not in config.keys():
        config['method'] = 'stitched'
    
    if config['method'] == 'parameterized':
        test_p0,  test_p1,  test_wcs  = parameterize_weights(test_coefs, config)
        train_p0, train_p1, train_wcs = parameterize_weights(train_coefs, config)
        test_feats  = torch.concatenate([test_feats,  test_wcs],  dim=1)
        train_feats = torch.concatenate([train_feats, train_wcs], dim=1)
    elif config['method'] == 'stitched':
        test_p0,  test_p1  = get_probabilities(test_coefs, config)
        train_p0, train_p1 = get_probabilities(train_coefs, config)
    elif config['method'] == 'alice':
        test_p0,  test_p1  = get_probabilities(test_coefs, config)
        train_p0, train_p1 = get_probabilities(train_coefs, config)

    test_coefs  = None
    train_coefs = None

    train_means = train_feats.mean(0)
    train_stds  = train_feats.std(0)
    train_feats = (train_feats - train_means) / train_stds
    norm_test   = (test_feats - train_means) / train_stds

    batches   = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_feats, train_p0, train_p1), 
                                            batch_size=config['batchSize'], shuffle=True, num_workers=16)
    model     = Model(nFeatures=train_feats.shape[1], method=config['method'], 
                      device=config['device'], config=config['network'], seed=config['seed'])
    optimizer = torch.optim.Adam(model.net.parameters(), lr=config['learningRate'])

    scheduler_type = config.get('scheduler', 'plateau')
    if scheduler_type == 'plateau':
        # ReduceLROnPlateau: steps LR down when val BCE stops improving.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=config.get('factor', 0.5),
            patience=config.get('lr_patience', 5),
        )
        print(f"[INFO] scheduler: ReduceLROnPlateau  factor={config.get('factor', 0.5)}  lr_patience={config.get('lr_patience', 5)}")
    elif scheduler_type == 'cosine':
        # Linear warmup → CosineAnnealingLR: smooth, deterministic
        warmup = config.get('warmup_epochs', 5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config['epochs'] - warmup), eta_min=1e-6),
            ],
            milestones=[warmup],
        )
        print(f"[INFO] scheduler: cosine+warmup  warmup_epochs={warmup}  T_max={max(1, config['epochs'] - warmup)}")
    else:
        scheduler = None
        print("[INFO] scheduler: none")

    trainLoss = [model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item()]
    testLoss  = [model.loss(norm_test, test_p0, test_p1).item()]
    lrHistory = [optimizer.param_groups[0]['lr']]

    os.makedirs(f'{config["name"]}/complete/animations', exist_ok=True)
    os.makedirs(f'{config["name"]}/complete/kinematics', exist_ok=True)
    for feature in features_config.keys():
        os.makedirs(f'{config["name"]}/incomplete/kinematics/{feature}', exist_ok=True)

    # early stopping parameters
    patience      = config.get('patience', 10)
    best_test_loss = float('inf')
    best_epoch     = 0
    patience_count = 0
    best_state     = None

    for epoch in tqdm.tqdm(range(config['epochs'])):
        s  = model.net(norm_test).cpu().detach().numpy().flatten()
        noOnes = s != 1
        s = s[noOnes]
        lr = s / (1 - s)
        tlr = (test_p1/test_p0).detach().cpu().numpy().flatten()
        for feature, params in features_config.items():
            if epoch == 0:
                ylim = kinematic_histogram(test_feats[noOnes, params['loc']].cpu().numpy(), params, epoch, lr, tlr[noOnes], 
                                           f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}.png')
            else: 
                kinematic_histogram(test_feats[noOnes, params['loc']].cpu().numpy(), params, epoch, lr, tlr[noOnes], 
                                    f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}.png', ylim=ylim)
        trainLoss.append(model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item())
        lrHistory.append(optimizer.param_groups[0]['lr'])
        if epoch%50 == 0:
            networkPlots(norm_test, test_p0, test_p1, model.net, trainLoss, 
                         testLoss, f'{config["name"]}/incomplete/epoch_{epoch:04d}', lr_history=lrHistory)
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
            best_epoch     = epoch
            best_state     = {k: v.clone() for k, v in model.net.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"[INFO] early stopping at epoch {epoch}, best epoch was {best_epoch} (test loss {best_test_loss:.4f})")
                break

        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(current_test_loss)
            else:
                scheduler.step()

    # NOTE: plots below reflect weights at the epoch training stopped on,
    # which may differ from the best checkpoint saved to model.pt below.
    networkPlots(norm_test, test_p0, test_p1, model.net, trainLoss, testLoss, f'{config["name"]}/complete', lr_history=lrHistory)
    s  = model.net(norm_test).cpu().detach().numpy().flatten()
    noOnes = s != 1
    s = s[noOnes]
    lr = s / (1 - s)
    tlr = (test_p1/test_p0).detach().cpu().numpy().flatten()
    for feature, params in features_config.items():
        kinematic_histogram(test_feats[noOnes, params['loc']].cpu().numpy(), params, epoch, lr, tlr[noOnes], 
                            f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}.png', ylim=ylim)
        kinematic_histogram(test_feats[noOnes, params['loc']].cpu().numpy(), params, epoch, lr, tlr[noOnes], 
                            f'{config["name"]}/complete/kinematics/{feature}.png', ylim=ylim, epoch_title=False)
        plots = sorted(glob.glob(f'{config["name"]}/incomplete/kinematics/{feature}/*.png'))
        animate_plots(plots, f'{config["name"]}/complete/animations/{feature}.gif')

    if best_state is not None:
        model.net.load_state_dict(best_state)
        print(f"[INFO] restored best checkpoint from epoch {best_epoch}")

    # keep the best model for validation
    torch.save(model.net.state_dict(), f'{config["name"]}/model.pt')

    return config

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help = 'configuration yml file used for training')
    
    #Load the configuration options and build the WC lists
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    config = main(config)
