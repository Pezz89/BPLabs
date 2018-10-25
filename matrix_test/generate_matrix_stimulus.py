#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import errno
import re
import fnmatch
import pdb
import numpy as np
import csv
from natsort import natsorted
from collections import namedtuple
import pysndfile
from pysndfile import PySndfile
import matplotlib.pyplot as plt

import scipy.signal as sgnl

from lpc import lpc

from filesystem import globDir, organiseWavs, prepareOutDir


def rolling_window_lastaxis(a, window):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
    if window < 1:
       raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
       raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def synthesizeTrial(wavFileMatrix, indexes):
    '''
    Using the matrix of alternative words and the selected words for each
    column, generate samples from audio files
    Returns an array of samples generated by concatenating the selected audio
    files
    '''
    columnNames = ['a', 'b', 'c', 'd', 'e']
    indexes = np.pad(indexes, ((0, 1)), 'constant', constant_values=0)
    indexes = rolling_window_lastaxis(indexes, 2)
    offset = 10
    y = np.array([])
    filenames = []
    for name, ind in zip(columnNames, indexes):
        if name == 'e':
            offset = 1
        wavFilename, wavFilepath = wavFileMatrix[name][(ind[0]*offset)+ind[1]]
        wav = PySndfile(wavFilepath)
        fs = wav.samplerate()
        x = wav.read_frames()
        y = np.append(y, x)
        filenames.append(wavFilename)
    return (y, {'rate': fs, 'format': wav.major_format_str(), 'enc': wav.encoding_str()}, filenames)


def generateTrialInds(n=1):
    '''
    Generate array of shape (n, 5), with each column representing the columns
    of the matrix (a-e)
    Indexes are generated randomly without replacement, ensuring no duplicate
    identical samples are generated
    '''
    choice = np.random.choice(100000, n, replace=False)
    indexes = np.zeros((n, 5), dtype=int)
    for ind, c in enumerate(choice):
        indexes[ind] = [int(i) for i in str(c).zfill(5)]
    return indexes


def generateStimulus(MatrixDir, OutDir, Length, socketio=None):
    # Get matrix wav file paths
    wavFiles = globDir(MatrixDir, '*.wav')
    wavFileMatrix = organiseWavs(wavFiles)
    # Randomly generate word choices for each trial
    indexes = generateTrialInds(100000)
    with open(os.path.join(OutDir, 'stim_parts.csv'), 'w') as csvfile:
        partwriter = csv.writer(csvfile)
        # Synthesize audio for each trial using generated word choices
        l = 0
        n = 0
        files = []
        while l < Length:
            if socketio:
                percent = (l / Length)*100.
                socketio.emit('update-progress', {'data': '{}%'.format(percent)}, namespace='/main')
            #print("Generating Trial_{0:05d}".format(n))
            y, wavInfo, partnames = synthesizeTrial(wavFileMatrix, indexes[n, :])
            partwriter.writerow(partnames)
            fileName = os.path.join(OutDir, 'Trial_{0:05d}.wav'.format(n))
            pysndfile.sndio.write(fileName, y, **wavInfo)
            n += 1
            l += y.size / wavInfo['rate']
            files.append(fileName)
    return files


def generateSpeechShapedNoise(x, order=500, plot=False):
    '''
    Generate speech shaped noise from input signal x.
    Linear Predictive Coding is used to estimate and FIR filter of the order
    specified. This is then used to filter white noise.
    '''
    a, e, k = lpc(x, order=order)
    noise = np.random.randn(x.size)*np.sqrt(e)
    b = np.zeros(a.size)
    b[0] = 1
    y = sgnl.lfilter(b,a,noise)
    if plot:
        M=fs/10;
        f, Px_den = sgnl.welch(x,window='hamming', nperseg=M, nfft=M)
        f, Py_den = sgnl.welch(y,window='hamming', nperseg=M, nfft=M)
        plt.semilogy(f, Px_den)
        plt.semilogy(f, Py_den)
        #plt.ylim([0.5e-3, 1])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()

    return y


if __name__ == "__main__":
    from pathtype import PathType
    # Create commandline interface
    parser = argparse.ArgumentParser(description='Generate stimulus for '
                                     'training TRF decoder by concatenating '
                                     'matrix test materials')
    parser.add_argument('--MatrixDir', type=PathType(exists=True, type='dir'),
                        default='./speech_components',
                        help='Matrix test speech data location')
    parser.add_argument('--OutDir', type=PathType(exists=None, type='dir'),
                        default='./out_trials', help='Output directory')
    parser.add_argument('--Length', type=int, default=60,
                        help='Concatenated length of trials in seconds')
    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    generateSpeechShapedNoise(order=500)
    exit()
    # Generate output directory if it doesn't exist
    prepareOutDir(args['OutDir'])

    # Generate stimulus from arguments provided on command line
    generateStimulus(**args)
