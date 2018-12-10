#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pysndfile import sndio
import numpy as np
import pdb
import matplotlib.pyplot as plt

def main():
    '''
    '''
    da_file = './BioMAP_da-40ms.wav'
    da_stim, fs, enc, fmt = sndio.read(da_file, return_format=True)
    prestim_size = 0.0158
    full_stim_size = 0.09174311926605504
    da_size = 0.04
    prestim = np.zeros(int(fs*prestim_size))
    poststim = np.zeros(int(fs*((full_stim_size-prestim_size)-da_size)))
    y_part = np.concatenate([prestim, da_stim, poststim])
    y = np.tile(y_part, 3000)
    sndio.write('./3000_da.wav', y, rate = fs, format = fmt, enc=enc)


if __name__ == "__main__":
    main()
