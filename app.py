import pathops
import os
from matrix_test.generate_matrix_stimulus import generateStimulus
import time

import config

'''
Module containing functions to be processed asynchronously
'''

def generate_matrix_stimulus(obj_response, n_part, snr_len, snr_num, mat_dir, save_dir):
    #from celery.contrib import rdb
    #rdb.set_trace()
    n_part = int(n_part)

    for participant_n in range(n_part):
        obj_response.attr('#progress-bar', 'style','width={}'.format((participant_n+1/n_part)*100.))
        time.sleep(1)
    return None
    '''
    save_dir = os.path.join(save_dir)
    mat_dir = os.path.join(mat_dir)
    pathops.dir_must_exist(save_dir)
    pathops.dir_must_exist(mat_dir)
    n_part = int(n_part)
    snr_len = float(snr_len)
    snr_num = int(snr_num)

    for participant_n in range(n_part):
        obj_response.attr('#progress-bar', 'style','width={}'.format((participant_n+1/n_part)*100.))
        genLength = (snr_len * snr_num) * 60.
        partDir = os.path.join(save_dir, "Partcipant{0:02d}".format(participant_n))
        pathops.dir_must_exist(partDir)

        filenames = generateStimulus(mat_dir, genLength, partDir)
    return {'current': n_part, 'total': n_part, 'status': 'Task completed!',
            'result': 42}
    '''
