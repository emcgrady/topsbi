from topsbi.model.net import Model
from topsbi.tools.buildLikelihood import expandArray
from topsbi.tools.plots import networkPlots
from topsbi.tools.data import prepare_weights, prepare_features, get_probabilities

import argparse, tqdm, torch, yaml

def main(config):
    if config['device'] != 'cpu' and not torch.cuda.is_available():
        print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        config['device'] = 'cpu'
    torch.manual_seed(config['seed'])

    test_feats,   test_coefs  = torch.load(f'{config["data"]}/test.p', weights_only=False)[:]
    train_feats,  train_coefs = torch.load(f'{config["data"]}/train.p', weights_only=False)[:]
    
    if 'method' not in config.keys():
        config['method'] = 'stitched'
    
    if config['method'] == 'parameterized':
        test_p0,  test_p1,  test_wcs  = prepare_weights(test_coefs, config)
        train_p0, train_p1, train_wcs = prepare_weights(train_coefs, config)
        test_feats  = torch.concatenate([test_feats,  test_wcs],  dim=1)
        train_feats = torch.concatenate([train_feats, train_wcs], dim=1)
    elif config['method'] == 'stitched':
        test_p0,  test_p1  = get_probabilities(test_coefs, config)
        train_p0, train_p1 = get_probabilities(train_coefs, config)

    test_coefs  = None
    train_coefs = None

    test_feats  = prepare_features(test_feats)
    train_feats = prepare_features(train_feats)

    batches   = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_feats, train_p0, train_p1), 
                                            batch_size=config['batchSize'], shuffle=True, num_workers=16)
    model     = Model(nFeatures=test_feats.shape[1], device=config['device'], config=config['network'], seed=config['seed'])
    optimizer = torch.optim.Adam(model.net.parameters(), lr=config['learningRate'])
    trainLoss = [model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item()]
    testLoss  = [model.loss(test_feats, test_p0, test_p1).item()]

    for epoch in tqdm.tqdm(range(config['epochs'])):
        if epoch%50 == 0:
            networkPlots(test_feats, test_p0, test_p1, model.net, trainLoss, testLoss, f'{config["name"]}/incomplete/epoch_{epoch:04d}')
        for train_feats, train_p0, train_p1 in batches:
            optimizer.zero_grad()
            loss = model.loss(train_feats, train_p0, train_p1)
            loss.backward()
            optimizer.step()
        trainLoss.append(model.loss(batches.dataset[:][0], batches.dataset[:][1], batches.dataset[:][2]).item())
        testLoss.append(model.loss(test_feats, test_p0, test_p1).item())
    networkPlots(test_feats, test_p0, test_p1, model.net, trainLoss, testLoss, f'{config["name"]}/complete')

    return config
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help = 'configuration yml file used for training')
    
    #Load the configuration options and build the WC lists
    with open(parser.parse_args().config, 'r') as f:
        config = yaml.safe_load(f)
    config = main(config)
    