import torch

from numpy import linspace, trapz


def netEval(netOut, bWeights, sWeights, threshold=0.5, nPoints=200):
    bins = linspace(netOut.min(), netOut.max(), nPoints + 1)

    bTotal = bWeights.sum()
    sTotal = sWeights.sum()

    tpr = []; fpr = []
    for i in range(len(bins)):
        tpr += [(sWeights[(netOut >= bins[-(i+1)]).ravel()].sum()/sTotal).item()]
        fpr += [(bWeights[(netOut >= bins[-(i+1)]).ravel()].sum()/bTotal).item()]

    a = ((sWeights[netOut >= threshold].sum() + bWeights[netOut <= threshold].sum())/bTotal + sTotal).item()
    auc = trapz(tpr, x=fpr).item()

    return fpr, tpr, auc, a