#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import errno
import shutil
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

from pathops import dir_must_exist
try:
    from signalops import rolling_window_lastaxis, block_lfilter
except ImportError:
    from .signalops import rolling_window_lastaxis, block_lfilter

import scipy.signal as sgnl
from scipy.stats import pearsonr

from pyswarm import pso

try:
    from lpc import lpc
except ImportError:
    from .lpc import lpc

try:
    from filesystem import globDir, organiseWavs, prepareOutDir
except ImportError:
    from .filesystem import globDir, organiseWavs, prepareOutDir


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
    pdb.set_trace()
    choice = np.random.choice(100000, n, replace=False)
    indexes = np.zeros((n, 5), dtype=int)
    for ind, c in enumerate(choice):
        indexes[ind] = [int(i) for i in str(c).zfill(5)]
    return indexes


def gen2(MatrixDir, OutDir, indexes):
    wavFiles = globDir(MatrixDir, '*.wav')
    wavFileMatrix = organiseWavs(wavFiles)
    files = []
    for sentenceList in indexes:
        for ind in sentenceList:
            y, wavInfo, partnames = synthesizeTrial(wavFileMatrix, ind)

    files.append(fileName)
    pdb.set_trace()



def generateAudioStimulus(MatrixDir, OutDir, Length, indexes, socketio=None):
    # Get matrix wav file paths
    wavFiles = globDir(MatrixDir, '*.wav')
    wavFileMatrix = organiseWavs(wavFiles)
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


def processNoise(x, order=500, plot=False, fs=None):
    '''
    Generate speech shaped noise from input signal x.
    Linear Predictive Coding is used to estimate and FIR filter of the order
    specified. This is then used to filter white noise.
    '''
    print("Calculating filter coefficients")
    x_fit = x[:fs*60*2]
    # Calculate filter coefficients using 2 minutes of speech
    b, a, e, k = calcLPC(x_fit, order, fs)
    # Generate and filter noise using calculted filter coefficients
    y = generateNoise(b, a, x.size, e, fs)
    # Calculate AIC of fitted filter
    #AIC = calcAIC(x, y, fs)
    if plot:
        plotLPC(x, y, fs)

    return y

def calcLPC(x, order, fs):
    a, e, k = lpc(x, order=order)
    b = np.zeros(a.size)
    b[0] = 1
    return b, a, e, k

def generateNoise(b, a, size, e, fs):
    print("Filtering white noise")
    noise = np.random.randn(size)*np.sqrt(e)
    y = block_lfilter(b, a, noise)
    return y


def calcAIC(x, y, fs):
    M=fs/10;
    f, Px_den = sgnl.welch(x,window='hamming', nperseg=M, nfft=M)
    f, Py_den = sgnl.welch(y,window='hamming', nperseg=M, nfft=M)
    resid = Px_den - Py_den
    sse = sum(resid**2)
    AIC= 2*order - 2*np.log(sse)
    return y, AIC


def plotLPC(x, y, fs):
    print("Plotting spectrum of the first 2 minutes of signal...")
    M=fs/10;
    f, Px_den = sgnl.welch(x[:fs*60*2],window='hamming', nperseg=M, nfft=M)
    f, Py_den = sgnl.welch(y[:fs*60*2],window='hamming', nperseg=M, nfft=M)
    plt.semilogy(f, Px_den)
    plt.semilogy(f, Py_den)
    #plt.ylim([0.5e-3, 1])
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()


def __calcLPCChunksPSOWrapper(params, *args):
    x = args[0]
    fs = args[1]
    length = args[2]
    order = int(round(params[0]))
    return calcLPCChunks(x, fs, plot=False, socketio=None, order=order, length=length)

def calcLPCChunks(x, fs, plot=False, socketio=None, order=500, length=1):
    # Define array of lengths in minutes to test
    length *= (fs * 60)
    length = int(length)
    chunkCount = 5
    start = ((np.arange(chunkCount)/chunkCount)*x.size).astype(int)
    end = length + start

    res = np.zeros(chunkCount)
    for j in range(start.size):
        print("Chunk: {0}".format(j))
        s = start[j]
        e = end[j]

        x_chunk = x[s:e]
        print("Chunk size: {0}".format((e-s)/fs))
        print("Order: {0}".format(order))
        y, aic = calcLPC(x_chunk, order, fs)
        res[j] = aic
    return np.mean(res)


def find_nearest(array, value):
    '''
    taken from:
        https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    '''
    array = np.array([array])
    value = np.array([value])
    idx = (np.abs(array - value.T)).argmin(axis=1)
    return idx

def generateDecoderAudio(SentenceDir, NoiseDir, OutDir, n_splits=8):
    '''
    Generate stimulus for training decoder at set SNRs using previous
    synthesized matrix test audio and spech shaped noise.
    '''
    n_splits = float(n_splits)
    wavFiles = globDir(SentenceDir, '*.wav')
    data = []
    markers = []
    i = 0
    for path in wavFiles:
        audio, fs, enc, fmt = pysndfile.sndio.read(path, return_format=True)
        silenceLen = int(0.1*fs)
        silence = np.zeros(silenceLen)
        chunk = np.append(audio, silence)
        data.append(chunk)
        markers.append(i)
        i += chunk.size
    markers = np.array(markers)
    split_size = i / n_splits
    splits = np.arange(n_splits) * split_size

    idx = find_nearest(markers, splits)
    idx = np.append(idx, len(data))
    idx = rolling_window_lastaxis(idx, 2)
    splitData = []
    for start, end in idx:
        splitData.append(np.concatenate([data[x] for x in range(start, end)]))
    x, fs, enc, fmt = pysndfile.sndio.read(os.path.join(NoiseDir, 'SSN.wav'))

    noiseData = []
    for start, end in idx:
        noiseData.append(x[start:end])

    pdb.set_trace()
    snr = 20*np.log10(np.sqrt(np.mean(signal**2))/np.sqrt(np.mean(noise**2)))


def generateNoiseFromSentences(SentenceDir, OutDir, order=500, plot=False, socketio=None):
    '''
    Fit speech shaped noise to all wav files found in SentenceDir. Output
    speech shaped noise of a length equal to the combined length of all found
    audio in SentenceDir to OutDir.
    '''
    wavFiles = globDir(SentenceDir, '*.wav')
    data = []
    for path in wavFiles:
        audio, fs, enc, fmt = pysndfile.sndio.read(path, return_format=True)
        # Add add silence after each clip
        silenceLen = int(0.1*fs)
        silence = np.zeros(silenceLen)
        chunk = np.append(audio, silence)
        data.append(chunk)
    x = np.concatenate(data)

    y = processNoise(x, order=order, plot=plot, fs=fs)
    noiseFile = os.path.join(OutDir, 'SSN.wav')
    print("Writing file...")
    pysndfile.sndio.write(os.path.join(OutDir, 'SSN.wav'), y, rate=fs, format=fmt, enc=enc)
    return noiseFile


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

    # Check directory for storing generated noise exists
    #noiseDir = os.path.join(args['OutDir'], 'noise')
    #dir_must_exist(noiseDir)
    #decoderDir = os.path.join(args['OutDir'], 'decoder')
    #dir_must_exist(noiseDir)
    ##generateDecoderAudio(args['OutDir'], noiseDir, decoderDir)
    #pdb.set_trace()
    #if os.path.exists(args['OutDir']):
    #    shutil.rmtree(args['OutDir'])
    #    os.makedirs(args['OutDir'])
    ## Generate output directory if it doesn't exist
    #prepareOutDir(args['OutDir'])


    # Generate audio stimulus from arguments provided on command line

    # Randomly generate word choices for each trial
    # indexes = generateTrialInds(100000)

    x = np.repeat(np.arange(10), 5)
    x = x.reshape(10, 5)

    y = np.zeros((50, 10, 5), dtype=int)

    # 50 lists
    for i in range(50):
        x[:, 1] = np.roll(x[:, 1], 1)
        x[:, 2] = np.roll(x[:, 2], 2)
        x[:, 3] = np.roll(x[:, 3], 3)
        x[:, 4] = np.roll(x[:, 4], 4)
        y[i] = x.copy()
    #indexes =
    gen2(args['MatrixDir'], args['OutDir'], y)
    generateAudioStimulus(**args)

    generateNoiseFromSentences(args['OutDir'], noiseDir)
    #generateDecoderAudio(args['OutDir'], noiseDir, decoderDir)



    '''
    wavFiles = globDir(args["OutDir"], '*.wav')
    data = []
    for path in wavFiles:
        audio, fs, enc, fmt = pysndfile.sndio.read(path, return_format=True)
        data.append(audio)
    x = np.concatenate(data)

    args = [x, fs, 0.05]
    lb = [1]
    ub = [1000]
    xopt, fopt = pso(__calcLPCChunksPSOWrapper, lb, ub, args=args)
    pdb.set_trace()
    '''
