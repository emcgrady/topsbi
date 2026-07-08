from topsbi.tools.buildLikelihood import full_likelihood, likelihood
from topsbi.tools.data import expand_array, get_probabilities
from topsbi.tools.plots import hist2d, kinematic_ratio_plot, lrMeanPlot

import matplotlib.pyplot as plt
import numpy as np

import argparse, os, torch, yaml


torch.set_num_threads(4)

def main(parametric, dedicated, output):
    with open(parametric, 'r') as f:
        config = yaml.safe_load(f)
    with open(config['features'], 'r') as f:
        config['features'] = yaml.safe_load(f)

    features, coefficients = torch.load(config['data'], weights_only=False)[:]
    
    plr          = full_likelihood(config, features)
    features     = features[plr.infFilter]
    coefficients = coefficients[plr.infFilter]
    dlr          = likelihood(dedicated, features.shape[1])
    c1           = dlr.config['c1']
    c0           = dlr.config['c0']
    wcs          = dlr.config['wcs']
    p0, p1       = get_probabilities(coefficients, dlr.config)

    # create title of dedicated point in WC space
    title = ''
    counter = 0
    for i, value in enumerate(c1[1:]):
        if value != 0:
            title += f'{wcs[i]}={value:.2f} '
            counter += 1
            if counter%6 == 0:
                title += '\n'

    os.makedirs(f'{output}/kinematics/linear', mode=0o755, exist_ok=True)
    os.makedirs(f'{output}/kinematics/log', mode=0o755, exist_ok=True)

    for kinematic, params in config['features'].items():
        params['c1']     = c1
        params['c0'] = c0
        params['title']   = title
        params['wcs']     = wcs
        params['plotLog'] = False
        params['outname'] = f'{output}/kinematics/linear/{kinematic}.png'
        kinematic_ratio_plot(features[:, params['loc']], dlr, plr, coefficients, **params)
        params['plotLog'] = True
        params['outname'] = f'{output}/kinematics/log/{kinematic}.png'
        kinematic_ratio_plot(features[:, params['loc']], dlr, plr, coefficients, **params)

    performance = {}
    #check binned LR
    fig, ax = plt.subplots()
    lrMeanPlot(ax, dlr, p1/p0, p0, 50, 0.01)
    lrMeanPlot(ax, plr, p1/p0, p0, 50, 0.01)
    ax.legend(['Dedicated', 'Parametric'])
    fig.savefig(f'{output}/lrExcl_lobin.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{output}/lrExclLog_lobin.png')
    fig.clf()
    fig.close()

    fig, ax = plt.subplots()
    performance['dChiExcl'] = lrMeanPlot(ax, dlr, p1/p0, p0, 1000, 0.01)
    performance['pChiExcl'] = lrMeanPlot(ax, plr, p1/p0, p0, 1000, 0.01)
    ax.legend(['Dedicated', 'Parametric'])
    fig.savefig(f'{output}/lrExcl.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{output}/lrExclLog.png')
    fig.clf()
    fig.close()

    fig, ax = plt.subplots()
    lrMeanPlot(ax, dlr, p1/p0, p0, 50, 0)
    lrMeanPlot(ax, plr, p1/p0, p0, 50, 0)
    ax.legend(['Dedicated', 'Parametric'])
    fig.savefig(f'{output}/lrIncl_lobin.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{output}/lrInclLog_lobin.png')
    fig.clf()
    fig.close()

    fig, ax = plt.subplots()
    performance['dChiIncl'] = lrMeanPlot(ax, dlr, p1/p0, p0, 1000, 0)
    performance['pChiIncl'] = lrMeanPlot(ax, plr, p1/p0, p0, 1000, 0)
    ax.legend(['Dedicated', 'Parametric'])
    fig.savefig(f'{output}/lrIncl.png')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.savefig(f'{output}/lrInclLog.png')
    fig.clf()
    fig.close()
    
    with open(f'{output}/performace.yml', 'w') as f:
        f.write(yaml.dump(performance))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parametric', '-p' , help = 'configuration yml file used for parametric likelihood')
    parser.add_argument('--dedicated', '-d' , help = 'configuration yml file used for dedicated likelihood')
    parser.add_argument('--output', '-o' , help = 'location to save output plots')

    args = parser.parse_args()
    main(args.parametric, args.dedicated, args.output)