#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../matrix_test/helper_modules")

import numpy as np
from pathops import dir_must_exist
from filesystem import globDir
from pysndfile import sndio
import os
from signalops import block_process_wav

import matplotlib.pyplot as plt

def main():
    '''
    '''
    fs = 44100
    f = 1000.0
    n = np.arange(fs * 60 * 5)
    y = np.sin(2*np.pi*f*n/fs)
    coef = np.load('./out/calibration_coefficients/click_cal_coef.npy')
    y *= coef
    dir_must_exist('./out/calibrated_stim/')
    sndio.write("./out/calibrated_stim/1k_tone.wav", y, fs, format='wav', enc='pcm16')
    coef = np.load('./out/calibration_coefficients/da_cal_coef.npy')
    y, fs, enc = sndio.read('./out/stimulus/da_cal_stim.wav')
    sndio.write('./out/calibrated_stim/da_cal_stim.wav', y*coef, fs, format='wav', enc='pcm16')
    coef = np.load('./out/calibration_coefficients/mat_cal_coef.npy')
    y, fs, enc = sndio.read('./out/stimulus/mat_cal_stim.wav')
    sndio.write('./out/calibrated_stim/mat_cal_stim.wav', y*coef, fs, format='wav', enc='pcm16')
    coef = np.load('./out/calibration_coefficients/story_cal_coef.npy')
    y, fs, enc = sndio.read('./out/stimulus/story_cal_stim.wav')
    sndio.write('./out/calibrated_stim/story_cal_stim.wav', y*coef, fs, format='wav', enc='pcm16')


if __name__ == "__main__":
    main()
