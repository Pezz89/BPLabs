#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "../helper_modules")

import os
from signalops import rolling_window_lastaxis, calc_rms
from filesystem import globDir
from pathops import dir_must_exist
from pysndfile import PySndfile
import numpy as np
import pdb

def detect_silences(rmsFile, fs):
    env = np.load(rmsFile)
    silence = env < 0.001
    # Get segment start end indexes for all silences in envelope
    silentSegs = np.where(np.concatenate(([silence[0]],silence[:-1]!=silence[1:],[True])))[0].reshape(-1, 2)
    validSegs = np.diff(silentSegs) > 0.02*fs
    return silentSegs[np.repeat(validSegs, 2, axis=1)].reshape(-1, 2)


def main():
    wavs = globDir('./out/stim/', '*.wav')
    rmss = globDir('./out/stim/', 'stim_*_env.npy')
    outDir = "./out/stim/"
    for wav, rms in zip(wavs, rmss):
        print("Detecting silence in wav file: {}".format(wav))
        snd = PySndfile(wav, 'r')
        fs = int(snd.samplerate())
        silences = detect_silences(rms, fs)

        head, tail = os.path.split(wav)
        tail = os.path.splitext(tail)[0]
        tail = tail + "_silence.npy"
        silence_filepath = os.path.join(outDir, tail)
        np.save(silence_filepath, silences)

if __name__ == "__main__":
    main()
