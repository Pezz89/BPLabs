import pathops
import os
from matrix_test.long_concat_stim.gen_long_stim import generateAudioStimulus
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

    filenames = generateAudioStimulus(mat_dir, sentDir, genLength, socketio=socketio)
    return filenames

def generate_speech_shaped_noise(sentence_dir, save_dir, order=500):
    '''
    '''
    pathops.dir_must_exist(save_dir)
    pathops.dir_must_exist(mat_dir)
    order = int(order)
    generateSpeechShapedNoise(sentence_dir, save_dir, order=500, plot=True)
