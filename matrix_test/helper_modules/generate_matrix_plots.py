#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import dill
import argparse
import pdb

from pathtype import PathType
from scipy.optimize import minimize

wordsCorrect = None
trackSNR = None
trialN = None
snrTrack = None

def find_nearest_idx(array, value):
    '''
    Adapted from: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def logisticFunction(L, L_50, s_50):
    '''
    Calculate logistic function for SNRs L, 50% SRT point L_50, and slope
    s_50
    '''
    return 1./(1.+np.exp(4.*s_50*(L_50-L)))


def logisticFuncLiklihood(args):
    '''
    Calculate the log liklihood for given L_50 and s_50 parameters.
    This function is designed for use with the scipy minimize optimisation
    function to find the optimal L_50 and s_50 parameters.

    args: a tuple containing (L_50, s_50)
    self.wordsCorrect: an n dimensional binary array of shape (N, 5),
        containing the correctness of responses to each of the 5 words for N
        trials
    self.trackSNR: A sorted list of SNRs of shape N, for N trials
    '''
    L_50, s_50 = args
    ck = wordsCorrect[np.arange(trackSNR.shape[0])]
    p_lf = logisticFunction(trackSNR, L_50, s_50)
    # Reshape array for vectorized calculation of log liklihood
    p_lf = p_lf[:, np.newaxis].repeat(5, axis=1)
    # Calculate the liklihood
    res = (p_lf**ck)*(((1.-p_lf)**(1.-ck)))
    with np.errstate(divide='raise'):
        try:
            a = np.concatenate(res)
            a[a == 0] = np.finfo(float).eps
            out = -np.sum(np.log(a))
        except:
            set_trace()
    if out == 0.:
        pdb.set_trace()
    return out


def fitLogistic():
    '''
    '''
    global wordsCorrect
    global trackSNR
    wordsCorrect = wordsCorrect[:trialN].astype(float)
    trackSNR = snrTrack[:trialN]
    res = minimize(logisticFuncLiklihood, np.array([np.mean(trackSNR),1.0]))
    pdb.set_trace()
    percent_correct = (np.sum(wordsCorrect, axis=1)/wordsCorrect.shape[1])*100.
    sortedSNRind = np.argsort(-trackSNR)
    sortedSNR = trackSNR[sortedSNRind]
    sortedPC = percent_correct[sortedSNRind]
    x = np.linspace(np.min(sortedSNR)-5, np.max(sortedSNR)+3, 3000)
    snr_50, s_50 = res.x
    x_y = logisticFunction(x, snr_50, s_50)
    x_y *= 100.
    print(snr_50)
    print(s_50)

    plt.clf()
    axes = plt.gca()
    psycLine, = axes.plot(x, x_y)
    plt.title("Predicted psychometric function")
    plt.xlabel("SNR (dB)")
    plt.ylabel("% Correct")
    srtLine, = axes.plot([snr_50,snr_50], [-50,50.], 'r--')
    axes.plot([-50.,snr_50], [50.,50.], 'r--')
    plt.xlim(x.min(), x.max())
    plt.ylim(x_y.min(), x_y.max())
    plt.yticks(np.arange(5)*25.)
    x_vals = np.array(axes.get_xlim())
    s_50 *= 100.
    b = 50. - s_50 * snr_50
    y_vals = s_50 * x_vals + b
    slopeLine, = axes.plot(x_vals, y_vals, '--')
    ticks = (np.arange((x.max()-x.min())/2.5)*2.5)+(2.5 * round(float(x.min())/2.5))
    ticks[find_nearest_idx(ticks, snr_50)] = snr_50
    labels = ["{:.2f}".format(x) for x in ticks]
    plt.xticks(ticks, labels)
    plt.legend((psycLine, srtLine, slopeLine), ("Psychometric function", "SRT={:.2f}dB".format(snr_50), "Slope={:.2f}%/dB".format(s_50)))
    plt.show()


def plotSNR():
    '''
    '''
    plt.clf()
    plt.plot(snrTrack, 'bo-')
    plt.ylim([20.0, -20.0])
    plt.xticks(np.arange(30))
    plt.xlim([-1, trialN])
    plt.xlabel("Trial N")
    plt.ylabel("SNR (dB)")
    plt.title("Adaptive track")
    for i, txt in enumerate(snrTrack[:trialN]):
        plt.annotate("{0}/{1}".format(
            np.sum(wordsCorrect[i]).astype(int),
            wordsCorrect[i].size),
            (i, snrTrack[i]),
            xytext=(0, 13),
            va="center",
            ha="center",
            textcoords='offset points'
        )
    plt.show()



def main(pklFile):
    '''
    '''
    global snrTrack
    global trialN
    global wordsCorrect

    with open(pklFile, 'rb') as f:
        l = dill.load(f)
        snrTrack = l['snrTrack']
        trialN = l['trialN']
        wordsCorrect =  l['wordsCorrect']
        plotSNR()
        fitLogistic()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pklFile', type=PathType(exists=True),
                        help='')
    args = parser.parse_args()
    pklFile = args.pklFile
    main(pklFile)
