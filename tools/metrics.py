import torch

def netEval(backgroundOutput, signalOutput, backgroundWeights, signalWeights, threshold=0.5, nPoints=200):
    bins = torch.linspace(0, 1, nPoints + 1)
    signalTotal =  signalWeights.sum()
    backgroundTotal = backgroundWeights.sum()
    tpr = []; fpr = []
    for i in range(len(bins)):
        tpr += [(signalWeights[(signalOutput >= bins[-(i+1)]).ravel()].sum()/signalTotal).item()]
        fpr += [(backgroundWeights[(backgroundOutput >= bins[-(i+1)]).ravel()].sum()/backgroundTotal).item()]

    total = signalTotal + backgroundTotal
    a = ((signalWeights[signalOutput.ravel() >= threshold].sum() + backgroundWeights[backgroundOutput.ravel() <= threshold].sum())/total).item()
    auc = torch.trapz(torch.Tensor(tpr), x=torch.Tensor(fpr)).item()

    return fpr, tpr, auc, a