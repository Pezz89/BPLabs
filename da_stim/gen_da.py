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
import resampy

def gen_da_stim(n, outpath):
    da_file = './BioMAP_da-40ms.wav'
    da_stim, fs, enc, fmt = sndio.read(da_file, return_format=True)
    prestim_size = 0.0158
    # Repetition rate in Hz
    repetition_rate = 10.9
    full_stim_size = 1./repetition_rate
    da_size = da_stim.size / fs
    prestim = np.zeros(int(fs*prestim_size))
    poststim = np.zeros(int(fs*((full_stim_size-prestim_size)-da_size)))
    y_part = np.concatenate([prestim, da_stim, poststim])
    pdb.set_trace()
    y_part_inv = -y_part
    loc_part = np.zeros(y_part.size)
    loc_part[prestim.size+1]=1

    y_2part = np.concatenate([y_part, y_part_inv])
    loc = np.concatenate([loc_part, loc_part])
    y_r = np.tile(y_2part, n)
    loc = np.tile(loc, n)
    loc = np.insert(loc, 0, np.zeros(fs))
    loc = np.where(loc)[0]

    y_r = np.insert(y_r, 0, np.zeros(fs))
    y_r = resampy.resample(y_r, fs, 44100)
    rat = 44100/fs
    fs = 44100
    y_l = np.zeros(y_r.size)
    loc = loc * rat
    loc=loc.round().astype(int)
    np.save('./stimulus/3000_da_locs.npy', loc)

    idx = np.arange(y_l.size)
    trigger = gen_trigger(idx, 2., 0.01, fs)

    y = np.vstack((y_l, y_r, trigger)).T
    sndio.write(outpath, y, rate = 44100, format = fmt, enc=enc)
    return outpath



def main():
    '''
    '''
    dir_must_exist('./stimulus')
    gen_da_stim(1500, './stimulus/3000_da.wav')

if __name__ == "__main__":
    main()
