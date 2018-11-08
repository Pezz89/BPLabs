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
from pysndfile import PySndfile, sndio
import matplotlib.pyplot as plt

from pathops import dir_must_exist
try:
    from signalops import rolling_window_lastaxis, calc_rms
except ImportError:
    from .signalops import rolling_window_lastaxis, block_lfilter, calc_rms

import scipy.signal as sgnl
from scipy.stats import pearsonr

import json

try:
    from lpc import lpc
except ImportError:
    from .lpc import lpc

try:
    from filesystem import globDir, organiseWavs, prepareOutDir
except ImportError:
    from .filesystem import globDir, organiseWavs, prepareOutDir

def gen_indexes(list_dir, speech_dir):
    list_files = globDir(list_dir, '*.txt')
    component_map_file = globDir(speech_dir, '*.json')[0]
    json_data=open(component_map_file).read()
    component_map = json.loads(json_data)

    column_names = ['a', 'b', 'c', 'd', 'e']
    list_indexes = np.array([])
    sentence_lists = {}
    for list_file in list_files:
        with open(list_file, 'r') as lfile:
            list_inds = []
            for line in lfile:
                line = line.split()
                trial_inds = []
                for ind, word in enumerate(line):
                    column_key = column_names[ind]
                    column_words = component_map[column_key]
                    trial_ind = column_words.index(word.upper())
                    trial_inds.append(trial_ind)
                list_inds.append(trial_inds)
            head, tail = os.path.split(list_file)
            sl_key = os.path.splitext(tail)[0]
            sentence_lists[sl_key] = np.array(list_inds)
    return sentence_lists

def generate_audio_stimulus(MatrixDir, OutDir, indexes, socketio=None):
    # Get matrix wav file paths
    wavFiles = globDir(MatrixDir, '*.wav')
    wavFileMatrix = organiseWavs(wavFiles)

    wav_dir = os.path.join(args['OutDir'], "wav")
    dir_must_exist(wav_dir)
    sentence_dir = os.path.join(wav_dir, "sentence-lists")
    dir_must_exist(sentence_dir)
    # Synthesize audio for each trial using generated word choices
    sentence_lists = {}
    for key in indexes.keys():
        files = []
        list_dir = os.path.join(sentence_dir, key)
        dir_must_exist(list_dir)
        with open(os.path.join(list_dir, 'stim_parts.csv'), 'w') as csvfile:
            partwriter = csv.writer(csvfile)
            for sentence_ind, component_inds in enumerate(indexes[key]):
                if socketio:
                    percent = (l / Length)*100.
                    socketio.emit('update-progress', {'data': '{}%'.format(percent)}, namespace='/main')
                y, wavInfo, partnames = synthesize_trial(wavFileMatrix, component_inds)
                partwriter.writerow(partnames)
                file_name = os.path.join(list_dir, 'Trial_{0:05d}.wav'.format(sentence_ind+1))
                sndio.write(file_name, y, **wavInfo)
                files.append(file_name)

            sentence_lists[key] = np.array(files)
    return sentence_lists



def synthesize_trial(wavFileMatrix, indexes):
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
                        default='./stimulus', help='Output directory')
    parser.add_argument('--ListDir', type=PathType(exists=None, type='dir'),
                        default='./lists', help='Output directory')

    args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}


    # Generate a dictionary containing indexes for every sentence list in
    # ListDir
    indexes = gen_indexes(args['ListDir'], args['MatrixDir'])
    generate_audio_stimulus(args['MatrixDir'], args['OutDir'], indexes)
    #wavFiles = gen_audio_stim(args['MatrixDir'], args['OutDir'], indexes)

    #silences = detect_silences(rmsFiles, 44100)
    #b = calc_spectrum(wavFiles, silences)
    #y = gen_noise(args['OutDir'], b, 44100)
