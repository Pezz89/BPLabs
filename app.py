import pathops
import os
from matrix_test.generate_matrix_stimulus import generateStimulus
import time

import config

'''
Module containing functions to be processed asynchronously
'''

def generate_matrix_stimulus(n_part, snr_len, snr_num, mat_dir, save_dir, socketio=None):
    #from celery.contrib import rdb
    #rdb.set_trace()
    pathops.dir_must_exist(save_dir)
    pathops.dir_must_exist(mat_dir)
    n_part = int(n_part)
    snr_len = float(snr_len)
    snr_num = int(snr_num)

    genLength = (snr_len * snr_num) * 60.
    sentDir = os.path.join(save_dir, "sentences")
    pathops.dir_must_exist(sentDir)

    filenames = generateStimulus(mat_dir, sentDir, genLength, socketio=socketio)
    return filenames

def generate_speech_shaped_noise(, order=500):
    '''
    '''
