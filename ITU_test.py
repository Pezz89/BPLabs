#!/usr/bin/env python3

from ITU_P56 import asl_P56
import numpy as np

def main():
    fs = 44100
    x = np.sin(2*np.pi*440*(np.arange(fs)/fs))
    asl_msq, actfact, c0 = asl_P56(x, fs, 16)
    breakpoint()


if __name__ == '__main__':
    main()
