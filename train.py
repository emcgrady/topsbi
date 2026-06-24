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
    trainLoss = [model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item()]
    testLoss  = [model.loss(norm_test, test_p0, test_p1).item()]

    os.makedirs(f'{config["name"]}/complete/animations', exist_ok=True)
    for feature in features_config.keys():
        os.makedirs(f'{config["name"]}/incomplete/kinematics/{feature}', exist_ok=True)

    for epoch in tqdm.tqdm(range(config['epochs'])):
        s  = model.net(norm_test).cpu().detach().numpy().flatten()
        noOnes = s != 1
        s = s[noOnes]
        lr = s / (1 - s)
        tlr = (test_p1/test_p0).detach().cpu().numpy().flatten()
        for feature, params in features_config.items():
            kinematic_histogram(test_feats[noOnes, params['loc']].cpu().numpy(), params, epoch, lr, tlr[noOnes], 
                                f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}.png')
        trainLoss.append(model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item())
        if epoch%50 == 0:
            networkPlots(norm_test, test_p0, test_p1, model.net, trainLoss, 
                         testLoss, f'{config["name"]}/incomplete/epoch_{epoch:04d}')
        for train_feats, train_p0, train_p1 in batches:
            optimizer.zero_grad()
            loss = model.loss(train_feats, train_p0, train_p1)
            loss.backward()
            optimizer.step()
        testLoss.append(model.loss(norm_test, test_p0, test_p1).item())
    networkPlots(norm_test, test_p0, test_p1, model.net, trainLoss, testLoss, f'{config["name"]}/complete')
    s  = model.net(norm_test).cpu().detach().numpy().flatten()
    noOnes = s != 1
    s = s[noOnes]
    lr = s / (1 - s)
    tlr = (test_p1/test_p0).detach().cpu().numpy().flatten()
    for feature, params in features_config.items():
        kinematic_histogram(test_feats[noOnes, params['loc']].cpu().numpy(), params, epoch, lr, tlr[noOnes], 
                            f'{config["name"]}/incomplete/kinematics/{feature}/{epoch:04d}.png')
        plots = sorted(glob.glob(f'{config["name"]}/incomplete/kinematics/{feature}/*.png'))
        animate_plots(plots, f'{config["name"]}/complete/animations/{feature}.gif')
    return config
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help = 'configuration yml file used for training')
    
    #Load the configuration options and build the WC lists
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    config = main(config)