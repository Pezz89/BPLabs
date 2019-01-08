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

from matplotlib import pyplot as plt

def main():
    wavs = globDir('./out/stim/', '*.wav')
    envs = globDir('./out/stim/', 'stim_*_env.npy')
    silences = globDir('./out/stim/', 'stim_*_silence.npy')
    for wavfp, envfp, silfp in zip(wavs, envs, silences):
        snd = PySndfile(wavfp, 'r')
        fs = int(snd.samplerate())
        env = np.load(envfp)
        sil_slices = np.load(silfp)
        sil = np.zeros(env.size)
        for sil_slice in sil_slices:
            sil[sil_slice[0]:sil_slice[1]] = 1
        pdb.set_trace()

        plt.plot(snd.read_frames(fs*60))
        plt.plot(sil[:fs*60])
        plt.show()


if __name__ == "__main__":
    main()
