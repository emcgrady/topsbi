import torch

def netEval(backgroundOutput, signalOutput, background, signal, threshold=0.5, nPoints=200):

    backgroundHist, signalHist = makeHist(backgroundOutput, signalOutput, background, signal, nPoints)
    roc, auc = makeROC(backgroundHist, signalHist, nPoints)
    a = ( signal[(signalOutput <= threshold).flatten()].sum() + background[(backgroundOutput >= threshold).flatten()].sum() )/(signal.sum() + background.sum())
    
    return roc, auc, a

def makeHist(backgroundOutput, signalOutput, background, signal, nPoints):

    backgroundIndices = torch.argsort(backgroundOutput, axis=0)
    signalIndices     = torch.argsort(signalOutput, axis=0)

    backgroundDisc = len(background)%nPoints
    signalDisc     = len(signal)%nPoints

    if backgroundDisc == 0:
        background = torch.sum(torch.tensor_split(background[backgroundOutput], nPoints), axis=1)
    else: 
        background = torch.cat((torch.sum(torch.stack(torch.tensor_split(background, nPoints)[:backgroundDisc]), axis=1),
                                torch.sum(torch.stack(torch.tensor_split(background, nPoints)[backgroundDisc:]), axis=1)), axis=0)    

    if signalDisc == 0:
        signal = torch.sum(torch.tensor_split(signal[signalOutput],   nPoints), axis=1)
    else:
        signal = torch.cat((torch.sum(torch.stack(torch.tensor_split(signal, nPoints)[:signalDisc]), axis=1),
                            torch.sum(torch.stack(torch.tensor_split(signal, nPoints)[signalDisc:]), axis=1)), axis=0)
    return signal, background

def makeROC(backgroundHist, signalHist, nPoints):

    roc = torch.cat((torch.cumsum(signalHist, dim=0).reshape(nPoints,1),
                     torch.cumsum(backgroundHist, dim=0).reshape(nPoints,1)), axis=1)
    roc = 1 - roc/roc[-1]
    auc = -torch.trapz(roc[:,1], x=roc[:,0])
    
    return roc, auc