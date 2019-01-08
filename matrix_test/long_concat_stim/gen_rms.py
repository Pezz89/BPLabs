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
from helper import slice_to_bool

def main():
    wavs = globDir('./out/stim/', '*.wav')
    silences = globDir('./out/stim/', 'stim_*_silence.npy')
    outDir = "./out/stim/"
    for wav, sil in zip(wavs, silences):
        snd = PySndfile(wav, 'r')
        fs = int(snd.samplerate())
        s = np.load(sil)
        sil_bool = slice_to_bool(s, snd.frames())

        rms = np.sqrt(np.mean(np.abs(snd.read_frames()[~sil_bool]**2)))

        head, tail = os.path.split(wav)
        tail = os.path.splitext(tail)[0]
        tail = tail + "_rms.npy"
        rms_filepath = os.path.join(outDir, tail)
        np.save(rms_filepath, rms)

if __name__ == "__main__":
    main()
