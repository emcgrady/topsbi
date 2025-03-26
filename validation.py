from tools.buildLikelihood import fullLikelihood
from tools.plots import histPlot, ratioPlot
from tools.metrics import netEval
from argparse import ArgumentParser
from pickle import load
from os import makedirs
from yaml import dump

import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument('--parametric', '-p' , help = 'configuration yml file used for parametric likelihood')
    parser.add_argument('--dedicated', '-d' , help = 'configuration yml file used for dedicated likelihood')
    parser.add_argument('--output', '-o' , help = 'location to save output plots')
    parser.add_argument('--validation_set', '-v', default='all', help = 'which dataset to use for validation. Can choose from all, test, or train')
    parser.add_argument('--network', '-n', default='complete', help = 'which epoch to use for validation')

    args = parser.parse_args()
    makedirs(args.output, mode=0o755, exist_ok=True)
    parametric = fullLikelihood(args.parametric, args.network)
    dedicated  = fullLikelihood(args.dedicated, args.network)
    with open(dedicated.config['data'], 'rb') as f:
        data = load(f)
    if args.validation_set == 'test':
        _,data  = torch.utils.data.random_split(data, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
    elif args.validation_set == 'train':
        data,_  = torch.utils.data.random_split(data, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
        
    plr = parametric(data[:][0], dedicated.wcValues)
    dlr = dedicated(data[:][0], dedicated.wcValues)
    
    backgroundMask = data[:][2] == 0
    signalMask     = data[:][2] == 1
    
    dEval = netEval(dlr[backgroundMask], dlr[signalMask], data[:][1][backgroundMask], data[:][1][signalMask])
    pEval = netEval(plr[backgroundMask], plr[signalMask], data[:][1][backgroundMask], data[:][1][signalMask])
    
    backgroundResiduals = (dlr[backgroundMask] - plr[backgroundMask]).detach().numpy()
    signalResiduals     = (dlr[signalMask] - plr[signalMask]).detach().numpy()
    
    plr = plr.detach().numpy()
    dlr = dlr.detach().numpy()
    
    backgroundMask = backgroundMask.detach().numpy()
    signalMask     = signalMask.detach().numpy()
    
    metrics = {
        'background': {
            'residualMean':   float(backgroundResiduals.mean()),
            'residualMin':    float(backgroundResiduals.min()),
            'residualMax':    float(backgroundResiduals.max()),
            'residualMedian': float(np.median(backgroundResiduals)),
            'residualStdv':   float(backgroundResiduals.std())
        },
        'signal': {
            'residualMean':   float(signalResiduals.mean()),
            'residualMin':    float(signalResiduals.min()),
            'residualMax':    float(signalResiduals.max()),
            'residualMedian': float(np.median(signalResiduals)),
            'residualStdv':   float(signalResiduals.std())
        }
    }
    makedirs(f'{args.output}/likelihood', mode=0o755, exist_ok=True)
    histPlot(plr[backgroundMask], plr[signalMask], 'Parametric Likelihood Ratio', f'{args.output}/likelihood', 'parametricLR', ylog=True)
    histPlot(np.abs(plr[backgroundMask]), np.abs(plr[signalMask]), 'Log Parametric Likelihood Ratio', 
             f'{args.output}/likelihood', 'parametricLR_Log', ylog=True, xlog=True)
    histPlot(dlr[backgroundMask], dlr[signalMask], 'Dedicated Likelihood Ratio', f'{args.output}/likelihood', 'dedicatedLR', ylog=True)
    histPlot(np.abs(dlr[backgroundMask]), np.abs(dlr[signalMask]), 'Log Dedicated Likelihood Ratio', 
             f'{args.output}/likelihood', 'dedicatedLR_Log', ylog=True, xlog=True)
    histPlot(backgroundResiduals, signalResiduals, 'Residuals', f'{args.output}/likelihood', 'residuals', ylog=True)
    histPlot(np.abs(backgroundResiduals), np.abs(signalResiduals), 'Log Residuals', 
             f'{args.output}/likelihood', 'residuals_Log', ylog=True, xlog=True)

    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.plot(dEval[0], dEval[1], label='Dedicated')
    ax.plot(pEval[0], pEval[1], '--', label='Parametric')
    ax.plot([0,1], [0,1], ':')
    ax.legend()
    fig.savefig(f'{args.output}/likelihood/roc.png')
    plt.clf()
    plt.close()

    with open(f'{args.output}/likelihood/metrics.yml', 'w') as f:
        f.write(dump(metrics))

    dlr *= data[:][3][:,0].detach().numpy()
    plr *= data[:][3][:,0].detach().numpy()
    for term, params in dedicated.config['terms'].items():
        if term != 'sm':
            dlr *= (data[:][3][:,1]*params['value'] + data[:][3][:,2]*params['value']**2).detach().numpy()
            plr *= (data[:][3][:,1]*params['value'] + data[:][3][:,2]*params['value']**2).detach().numpy()
    makedirs(f'{args.output}/kinematics/noNorm/', mode=0o755, exist_ok=True)
    makedirs(f'{args.output}/kinematics/density/', mode=0o755, exist_ok=True)
    for key, value in dedicated.config['features'].items():
        x = (data[:][0][:,value['loc']]*data.stdvs[0][value['loc']]+data.means[0][value['loc']]).detach().numpy()
        bins = np.linspace(value['min'], value['max'], value['nbins']+1)
        ratioPlot(x, dlr, plr, data[:][3].detach().numpy(), bins, 
                  list(dedicated.config['terms'].keys())[1:], 
                  f'{args.output}/kinematics/noNorm/{key}', xlabel=value['label'])
        ratioPlot(x, dlr, plr, data[:][3].detach().numpy(), bins, 
                  list(dedicated.config['terms'].keys())[1:], 
                  f'{args.output}/kinematics/noNorm/{key}_Log', plotLog=True, xlabel=value['label'])
        ratioPlot(x, dlr, plr, data[:][3].detach().numpy(), bins, 
                  list(dedicated.config['terms'].keys())[1:], 
                  f'{args.output}/kinematics/density/{key}', xlabel=value['label'], showNoWeights=True, density=True)
        ratioPlot(x, dlr, plr, data[:][3].detach().numpy(), bins, 
                  list(dedicated.config['terms'].keys())[1:], 
                  f'{args.output}/kinematics/density/{key}_Log', plotLog=True, xlabel=value['label'], showNoWeights=True, density=True)

if __name__=="__main__":
    main()