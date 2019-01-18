#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../matrix_test/helper_modules")
import pdb
import numpy as np
import pandas as pd
import csv

def main():
    cal_txt = "./cal_vals.txt"
    vals = []
    with open(cal_txt, 'r') as txt:
        for line in txt:
            vals.append(line.split(" "))
            vals[-1][1] = float(vals[-1][1].rstrip())
    vals = {k: v for k, v in vals}

    v = []
    for key in vals.keys():
        v.append(vals[key])
    v = np.array(v)
    v = v/(v.max()*2)
    v[np.isnan(v)] = 0.0
    out = {}
    for key, val in zip(vals.keys(), v):
        out_file = './out/calibration_coefficients/{}_cal_coef.npy'.format(key)
        np.save(out_file, val)




if __name__ == "__main__":
    main()
