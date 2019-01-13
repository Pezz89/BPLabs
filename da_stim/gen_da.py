#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../matrix_test/helper_modules")

from pysndfile import sndio
import numpy as np
import pdb
import matplotlib.pyplot as plt
from pathops import dir_must_exist
from signalops import gen_trigger

def gen_da_stim(n, outpath):
    da_file = './BioMAP_da-40ms.wav'
    da_stim, fs, enc, fmt = sndio.read(da_file, return_format=True)
    prestim_size = 0.0158
    full_stim_size = 0.09174311926605504
    da_size = 0.04
    prestim = np.zeros(int(fs*prestim_size))
    poststim = np.zeros(int(fs*((full_stim_size-prestim_size)-da_size)))
    y_part = np.concatenate([prestim, da_stim, poststim])
    y_part_inv = -y_part

    y_2part = np.concatenate([y_part, y_part_inv])
    y_r = np.tile(y_2part, n)
    y_r = np.insert(y_r, 0, np.zeros(fs))
    y_l = np.zeros(y_r.size)

    idx = np.arange(y_l.size)
    trigger = gen_trigger(idx, 2., 0.01, fs)

    y = np.vstack((y_l, y_r, trigger))
    sndio.write(outpath, y.T, rate = fs, format = fmt, enc=enc)
    return outpath



def main():
    '''
    '''
    dir_must_exist('./stimulus')
    gen_da_stim(1500, './stimulus/3000_da.wav')

if __name__ == "__main__":
    main()
