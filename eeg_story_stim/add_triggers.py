#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../matrix_test/helper_modules")

from filesystem import globDir
from pudb import set_trace
from pysndfile import sndio
from signalops import gen_trigger
import os
import numpy as np

def main():
    '''
    '''
    wavs = globDir("./stimulus", "*.wav")
    for wav in wavs:
        x, fs, enc, fmt = sndio.read(wav, return_format=True)
        y_r = np.insert(x, 0, np.zeros(fs))
        idx = np.arange(x.shape[0])
        y = np.vstack([x, x, np.zeros(x.shape[0])]).T
        trigger = gen_trigger(idx, 2., 0.01, fs)
        y[:, 2] = trigger
        wav_out = os.path.splitext(wav)[0] + "_trig.wav"
        sndio.write(wav_out, y, rate=fs, format=fmt, enc=enc)

if __name__ == "__main__":
    main()
