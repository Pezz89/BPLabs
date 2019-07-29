#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dill
import numpy as np
import argparse
from pathtype import PathType
import sys
import os
from loggerops import create_logger

def main(args):
    file = args.data_file
    with open(file, 'rb') as pkl:
        a = dill.load(pkl)
    del a['participant']
    np.save(os.path.basename(file)+'-new.npy', a)

if __name__ == '__main__':
    #peak_pick_test()
    parser = argparse.ArgumentParser(
        description='Script for removing BPLabs sepcific objects from participant data'
    )
    parser.add_argument(
        dest='data_file', type=PathType(),
        help='Configuration file for processing BDF', metavar='CONFIGFILE'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='count',
        help='Specifies level of verbosity in output. For example: \'-vvvvv\' '
        'will output all information. \'-v\' will output minimal information. '
    )
    args = parser.parse_args()

    # Set verbosity of logger output based on argument
    if not args.verbose:
        args.verbose = 10
    else:
        levels = [50, 40, 30, 20, 10]
        if args.verbose > 5:
            args.verbose = 5
        args.verbose -= 1
        args.verbose = levels[args.verbose]

    # Define path to module for storing log files
    modpath = sys.argv[0]
    modpath = os.path.splitext(modpath)[0]+'.log'

    logger = create_logger(
        logger_streamlevel=args.verbose,
        log_filename=modpath,
        logger_filelevel=args.verbose
    )
    main(args)
