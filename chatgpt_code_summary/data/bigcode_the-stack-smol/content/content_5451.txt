from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ..w_transform import HaarTransform, InvHaarTransform

def _zeros_like(obj):
    zeros = [np.zeros_like(lev, dtype=float) for lev in obj]
    return zeros


__all__ = ['_findmin', '_findmax', '_BinData', '_NewColorMap',
           '_NSigmaFilter']

delta = 0.0 # Small number

def _findmin(array):
    minn = delta
    for i in array:
        if np.min(i) < minn:
            minn = np.min(i)
    return minn

def _findmax(array):
    maxx = delta
    for i in array:
        if np.max(i) > maxx:
            maxx = np.max(i)
    return maxx

def _BinData(data, bins):
    hist, edges = np.histogram(a=range(bins), bins=bins, weights=data)
    center = (edges[:-1]+edges[1:])/2.0
    width = edges[1:]-edges[:-1]
    return hist, edges, center, width

def _NewColorMap():
    R=float(0+172+242)
    G=(41.+181.+104.)
    B=(242.+81.+59.)
    #colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)] #RGB
    #colors = [(0.172, 0.521, 0.729), (0.870, 0.325, 0.129)]
    colors = [(0.152, 0.552, 0.607),
              (0.666, 0.882, 0.035),
              (0.945, 0.337, 0.074)]
    nbins=2**15
    cmap_name='New'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
    return cm


def _NSigmaFilter(data, hypothesis, nsigma,
                  nsigma_min=None, nsigma_percent=None):

    WaveDec_data = HaarTransform(data)
    DataCoeffs = WaveDec_data[:-1]
    DataFirstTrend = WaveDec_data[-1]
    
    WaveDec_hypo = HaarTransform(hypothesis)
    HypoCoeffs = WaveDec_hypo[:-1]
    HypoFirstTrend = WaveDec_hypo[-1]

    Level = len(DataCoeffs)

    flatNsigma = []
    flatAbsNsigma = []
    flatDataCoeffs = []
    flatHypoCoeffs = []
    flatLoc = []

    count = 0
    for l in range(Level):
        J = 2**(Level-l-1)
        for j in range(J):
            flatNsigma.append(nsigma[l][j])
            flatAbsNsigma.append(abs(nsigma[l][j]))
            flatDataCoeffs.append(DataCoeffs[l][j])
            flatHypoCoeffs.append(HypoCoeffs[l][j])
            flatLoc.append([l, j])
            count += 1

    ixsort = np.argsort(flatAbsNsigma)[::-1]
    sortNsigma = [flatNsigma[ix] for ix in ixsort]
    sortDataCoeffs = [flatDataCoeffs[ix] for ix in ixsort]
    sortHypoCoeffs = [flatHypoCoeffs[ix] for ix in ixsort]
    sortLoc = [flatLoc[ix] for ix in ixsort]

    keepNsigma = []
    keepDeltaCoeff = []
    keepLoc = []
    if nsigma_min is not None:
        for i in range(len(sortNsigma)):
            if abs(sortNsigma[i]) > nsigma_min:
                keepNsigma.append(sortNsigma[i])
                keepDeltaCoeff.append(sortDataCoeffs[i]-sortHypoCoeffs[i])
                keepLoc.append(sortLoc[i])
                
    elif nsigma_percent is not None:
        net = len(sortNsigma)
        netkeep = int(np.ceil(net*nsigma_percent))
        keepNsigma = sortNsigma[:netkeep]
        keepDeltaCoeff = np.subtract(sortDataCoeffs[:netkeep],
                                     sortHypoCoeffs[:netkeep])
        keepLoc = sortLoc[:netkeep]

    else:
        keepNsigma = sortNsigma
        keepDeltaCoeff = np.subtract(sortDataCoeffs,
                                     sortHypoCoeffs)
        keepLoc = sortLoc

    keep = _zeros_like(WaveDec_data)
    for i in range(len(keepDeltaCoeff)):
        l = keepLoc[i][0]
        j = keepLoc[i][1]
        keep[l][j] = keepDeltaCoeff[i]
    keep[-1][0] = DataFirstTrend-HypoFirstTrend

    return keep



