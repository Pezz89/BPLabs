#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../helper_modules/")

import csv
import argparse
import os
import shutil
import errno
import re
import fnmatch
import pudb
import numpy as np
from natsort import natsorted
from collections import namedtuple
import pysndfile
from pysndfile import sndio, PySndfile

from signalops import gen_trigger
from pathtype import PathType
from tokens_to_words import tokens_to_words, load_component_map

def prepareOutDir(folder):
    '''
    Check that the specified output directory exists and remove any
    pre-existing files and folders from it
    '''
    try:
        os.mkdir(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def globDir(directory, pattern):
    '''
    Return all files in a directory matching the unix glob pattern, ignoring
    case
    '''
    def absoluteFilePaths(directory):
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))
    speech_file_pattern = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    filepaths = []
    for item in absoluteFilePaths(directory):
        if bool(speech_file_pattern.match(os.path.basename(item))):
            filepaths.append(item)
    return natsorted(filepaths)

def concatenateStimuli(MatrixDir, OutDir, Length, n):
    # Get matrix wav file paths
    wavFiles = globDir(MatrixDir, '*.wav')

    stim_parts = os.path.join(MatrixDir, "stim_parts.csv")
    stim_words = os.path.join(MatrixDir, "stim_words.csv")
    stim_part_rows = []
    with open(stim_parts, 'r') as csvfile:
        stim_part_rows = [line for line in csv.reader(csvfile)]
    with open(stim_words, 'r') as csvfile:
        stim_word_rows = [line for line in csv.reader(csvfile)]

    wavFiles = natsorted(wavFiles)
    totalSize = 0
    y = []
    parts = []
    questions = []
    i = 0
    for wav in wavFiles:
        if i == n:
            break
        wavObj = PySndfile(wav)
        fs = wavObj.samplerate()
        size = wavObj.frames()
        totalSize += size
        totalSize += int(0.1*fs)
        if (totalSize/fs) > Length:
            # total size + 2 second silence at start
            y.append(np.zeros((totalSize+2*fs, 3)))
            parts.append([])
            questions.append([])
            i += 1
            totalSize = 0

    writePtr = 2*fs
    idx = np.arange(0, writePtr)
    chunk = np.zeros(idx.size)
    chunk = np.vstack([chunk, chunk, chunk]).T
    trigger = gen_trigger(idx, 2., 0.01, fs)
    chunk[:, 2] = trigger
    for i, _ in enumerate(y):
        y[i][0:writePtr, :] = chunk

    i = 0
    for wav, word, part in zip(wavFiles, stim_word_rows, stim_part_rows):
        if writePtr >= y[i].shape[0]:
            i += 1
            writePtr = fs*2
        if i == n:
            break
        x, fs, encStr, fmtStr = sndio.read(wav, return_format=True)
        threeMs = int(0.1*fs)
        silence = np.zeros(threeMs)
        chunk = np.append(x, silence)

        idx = np.arange(writePtr, writePtr+chunk.shape[0])
        chunk = np.vstack([chunk, chunk, np.zeros(chunk.shape[0])]).T
        trigger = gen_trigger(idx, 2., 0.01, fs)
        chunk[:, 2] = trigger

        y[i][writePtr:writePtr + chunk.shape[0], :] = chunk
        questions[i].append(word)
        parts[i].append(part)

        writePtr += chunk.shape[0]

    for ind, (data, q, p) in enumerate(zip(y, questions, parts)):
        pysndfile.sndio.write(os.path.join(OutDir, 'stim_{}.wav'.format(ind)), data, format=fmtStr, enc=encStr)
        with open('./out/stim/stim_words_{}.csv'.format(ind), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(q)
        with open('./out/stim/stim_parts_{}.csv'.format(ind), 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(p)

if __name__ == "__main__":
    # Create commandline interface
    parser = argparse.ArgumentParser(description='Generate stimulus for '
                                     'training TRF decoder by concatenating '
                                     'matrix test materials')
    parser.add_argument('--MatrixDir', type=PathType(exists=True, type='dir'),
                        default='./out/parts',
                        help='Matrix test speech data location')
    parser.add_argument('--OutDir', type=PathType(exists=None, type='dir'),
                        default='./out/stim', help='Output directory')
    parser.add_argument('--Length', type=int, default=900,
                        help='Length of each concatenated trial in seconds')
    parser.add_argument('-n', type=int, default=4,
                        help='Number of trials to generate')
    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}

    # Generate output directory if it doesn't exist
    prepareOutDir(args['OutDir'])

    # Generate stimulus from arguments provided on command line
    concatenateStimuli(**args)
