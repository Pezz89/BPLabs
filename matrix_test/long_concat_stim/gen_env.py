#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "../helper_modules")

import os
from signalops import rolling_window_lastaxis, calc_rms
from filesystem import globDir
from pathops import dir_must_exist
from pysndfile import sndio
import numpy as np
import multiprocessing
from helper import multiprocess_map

def flat_for(a, f, *args):
        a = a.reshape(-1)
        for i, v in enumerate(a):
            print("Sample {0} of {1}".format(i, a.size))
            a[i] = f(v, *args)

def window_rms(a, window_size):
    print("Squaring...")
    a2 = a**2
    print("Convolving...")
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'same'))

def gen_rms(file, OutDir):
    head, tail = os.path.split(file)
    tail = os.path.splitext(tail)[0]
    tail = tail + "_env.npy"
    dir_must_exist(OutDir)
    rmsFilepath = os.path.join(OutDir, tail)
    print("Generating: "+rmsFilepath)
    y, fs, _ = sndio.read(file)

    y = y[:, 0]
    y_rms = window_rms(y, round(0.02*fs))
    np.save(rmsFilepath, y_rms)
    return rmsFilepath

def main():
    wavs = globDir('./out/stim/', '*.wav')
    args = [(x, "./out/stim/") for x in wavs]

    multiprocess_map(gen_rms, args)


if __name__ == "__main__":
    main()
