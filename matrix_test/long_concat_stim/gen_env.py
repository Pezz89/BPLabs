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

def gen_rms(file, OutDir):
    head, tail = os.path.split(file)
    tail = os.path.splitext(tail)[0]
    tail = tail + "_env.npy"
    dir_must_exist(OutDir)
    rmsFilepath = os.path.join(OutDir, tail)
    print("Generating: "+rmsFilepath)
    y, fs, _ = sndio.read(file)

    y_rms = calc_rms(y, round(0.02*fs))
    np.save(rmsFilepath, y_rms)
    return rmsFilepath

def main():
    wavs = globDir('./out/stim/', '*.wav')
    args = [(x, "./out/stim/") for x in wavs]

    multiprocess_map(gen_rms, args)


if __name__ == "__main__":
    main()
