from argparse import ArgumentParser
from numpy import linspace, median
from os import makedirs
from pickle import load
from tools.buildLikelihood import fullLikelihood, likelihood
from tools.plots import histPlot, ratioPlot
from tools.metrics import netEval
from torch import mean
from yaml import dump, safe_load

def main(parametric, dedicated, output):
    makedirs(output, mode=0o755, exist_ok=True)
    with open(parametric, 'r') as f:
        config = safe_load(f)

    dlr  = likelihood(dedicated, len(config['features']))
    dWcs = dlr.config['signalTrainingPoint']
    wcDict = {}
    for i, wc in enumerate(dlr.config['wcs']):
        wcDict[wc] = dWcs[i+1]

    with open(dlr.config['data'], 'rb') as f:
        data = load(f)
    features, fitCoefs, truth  = data[:]
    features = (features*data.stdvs+data.means).detach().numpy()
    fitCoefs = fitCoefs.detach().numpy()
    truth    = truth.detach().numpy()

    plr = fullLikelihood(config, data)
    dlr = dlr(data, dWcs)
    plr = plr(dWcs)

    plr[plr < 0] = 0
    dlr[dlr < 0] = 0
    
    residuals = abs((dlr - plr)/dlr)
    
    metrics = {
        'all': {
            'residualMean':   float(residuals.mean()),
            'residualMin':    float(residuals.min()),
            'residualMax':    float(residuals.max()),
            'residualMedian': float(median(residuals)),
            'residualStdv':   float(residuals.std())
        },
        'background': {
            'residualMean':   float(residuals[truth == 0].mean()),
            'residualMin':    float(residuals[truth == 0].min()),
            'residualMax':    float(residuals[truth == 0].max()),
            'residualMedian': float(median(residuals[truth == 0])),
            'residualStdv':   float(residuals[truth == 0].std())
        },
        'signal': {
            'residualMean':   float(residuals[truth == 1].mean()),
            'residualMin':    float(residuals[truth == 1].min()),
            'residualMax':    float(residuals[truth == 1].max()),
            'residualMedian': float(median(residuals[truth == 1])),
            'residualStdv':   float(residuals[truth == 1].std())
        },
    }

    makedirs(f'{output}/likelihood', mode=0o755, exist_ok=True)
    
    histPlot(plr[truth == 0], plr[truth == 1], 'Parametric Likelihood Ratio',
             outname=f'{output}/likelihood/parametricLR', ylog=True)
    histPlot(dlr[truth == 0], dlr[truth == 1], 'Dedicated Likelihood Ratio',  
             outname=f'{output}/likelihood/dedicatedLR', ylog=True)
    histPlot(residuals[truth == 0], residuals[truth == 1], 'Residuals',
              outname=f'{output}/likelihood/residuals', ylog=True)
    histPlot(plr[truth == 0], plr[truth == 1], 'Parametric Likelihood Ratio',   
             outname=f'{output}/likelihood/Log_parametricLR', ylog=True, xlog=True)
    histPlot(dlr[truth == 0], dlr[truth == 1], 'Dedicated Likelihood Ratio', 
             outname=f'{output}/likelihood/Log_dedicatedLR', ylog=True, xlog=True)
    histPlot(residuals[truth == 0], residuals[truth == 1], 'Residuals',
              outname=f'{output}/likelihood/Log_residuals', ylog=True, xlog=True)

    makedirs(f'{output}/kinematics/noNorm',  mode=0o755, exist_ok=True)
    makedirs(f'{output}/kinematics/density', mode=0o755, exist_ok=True)
    
    for kinematic, params in config['features'].items():
        bins = linspace(params['min'], params['max'], params['nbins'])
        ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict, 
                  outname=f'{output}/kinematics/noNorm/{kinematic}.png', 
                  xlabel=params['label'])
        ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict, 
                  outname=f'{output}/kinematics/noNorm/Log_{kinematic}.png', 
                  xlabel=params['label'], plotLog=True)
        ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict, 
                  outname=f'{output}/kinematics/density/{kinematic}.png', 
                  xlabel=params['label'], showNoWeights=True, density=True)
        ratioPlot(features[:,params['loc']], dlr, plr, fitCoefs, bins, wcDict, 
                  outname=f'{output}/kinematics/density/Log_{kinematic}.png', 
                  xlabel=params['label'], showNoWeights=True, density=True, plotLog=True)

    with open(f'{output}/likelihood/metrics.yml', 'w') as f:
        f.write(dump(metrics))

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--parametric', '-p' , help = 'configuration yml file used for parametric likelihood')
    parser.add_argument('--dedicated', '-d' , help = 'configuration yml file used for dedicated likelihood')
    parser.add_argument('--output', '-o' , help = 'location to save output plots')

    args = parser.parse_args()
    main(args.parametric, args.dedicated, args.output, args.validation_set, args.network)