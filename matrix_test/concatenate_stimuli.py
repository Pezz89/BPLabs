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
from natsort import natsorted
from collections import namedtuple
import pysndfile
from pysndfile import sndio, PySndfile

from pathtype import PathType

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
    return filepaths

def concatenateStimuli(MatrixDir, OutDir, Length):
    # Get matrix wav file paths
    wavFiles = globDir(MatrixDir, '*.wav')
    wavFiles = natsorted(wavFiles)
    totalSize = 0
    for wav in wavFiles:
        wavObj = PySndfile(wav)
        fs = wavObj.samplerate()
        size = wavObj.frames()
        totalSize += size
        totalSize += int(0.1*fs)
        if (totalSize/fs) > Length:
            break
    y = np.zeros(totalSize)

    writePtr = 0
    for wav in wavFiles:
        if writePtr >= y.size:
            break
        x, fs, encStr, fmtStr = sndio.read(wav, return_format=True)
        threeMs = int(0.1*fs)
        silence = np.zeros(threeMs)
        chunk = np.append(x, silence)
        y[writePtr:writePtr + chunk.size] = chunk

        writePtr += chunk.size

    pysndfile.sndio.write(os.path.join(OutDir, 'decoder_stim.wav'), y, format=fmtStr, enc=encStr)

if __name__ == "__main__":
    # Create commandline interface
    parser = argparse.ArgumentParser(description='Generate stimulus for '
                                     'training TRF decoder by concatenating '
                                     'matrix test materials')
    parser.add_argument('--MatrixDir', type=PathType(exists=True, type='dir'),
                        default='./speech_components',
                        help='Matrix test speech data location')
    parser.add_argument('--OutDir', type=PathType(exists=None, type='dir'),
                        default='./out_concat', help='Output directory')
    parser.add_argument('--Length', type=int, default=60,
                        help='Concatenated length of trials in seconds')
    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}

    # Generate output directory if it doesn't exist
    prepareOutDir(args['OutDir'])

    # Generate stimulus from arguments provided on command line
    concatenateStimuli(**args)
