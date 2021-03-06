#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../matrix_test/helper_modules")
from scipy.signal import square
from pysndfile import sndio
import numpy as np
import pdb
import matplotlib.pyplot as plt
from signalops import gen_trigger

def gen_click(idx, freq, length, fs):
    duty = length*freq
    trigger = square(2*np.pi*(idx/fs)*freq, duty=duty)
    trigger[trigger < 0] = 0
    return trigger

def main():
    '''
    '''
    freq = 20.0
    fs = 44100
    period = fs/freq
    length = period * 3000.
    y = (np.arange(length) % period == 0).astype(float)
    y[np.where(y == 1.0)[0][1::2]] = -1.0
    y = np.concatenate([np.zeros(fs), y, np.zeros(fs)])

    idx = np.arange(y.size)
    trigger = gen_trigger(idx, 2., 0.01, fs)
    y = np.vstack((y, y, trigger)).T

    print("Number of clicks generated: {}".format(np.sum(np.abs(y[:, 0]) == 1.0)))
    sndio.write('./click_3000_20Hz.wav', y, rate = fs, format='wav', enc='pcm16')

if __name__ == "__main__":
    main()
