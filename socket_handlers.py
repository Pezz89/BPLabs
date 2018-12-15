import pandas as pd
import os
from flask import Flask, url_for, render_template, jsonify, request, make_response, g
from flask_socketio import emit
import pdb
import csv
import io
import re
import base64
import shutil
import pdb

import webview
import webbrowser
import app
import time
from threading import Thread, Event
import numpy as np
import random
from pysndfile import sndio
import sounddevice as sd
from scipy.optimize import minimize
from WavPlayer import play_wav_async

from app import generate_matrix_stimulus
from matrix_test.filesystem import globDir, organiseWavs, prepareOutDir
from matrix_test_thread import MatTestThread
from pathops import dir_must_exist
from participant import Participant
from matrix_test.signalops import play_wav

from config import server, socketio, participants

from da_test_thread import DaTestThread
from eeg_test_thread import EEGTestThread
from matrix_test_thread import MatTestThread
from test_base import run_test_thread

thread_types = {
    'da_test': DaTestThread,
    'eeg_test': EEGTestThread,
    'mat_test': MatTestThread,
}

'''
Generic socket handlers
'''
@socketio.on('open_save_file_dialog', namespace='/main')
def openSaveFileDialog():
    # Open a file dialog interface for selecting a directory
    filepath = webview.create_file_dialog(dialog_type=webview.SAVE_DIALOG,
                                          file_types=("Python Pickle (*.pkl)",),
                                          save_filename="test_session.pkl")
    if filepath and len(filepath) > 0:
        filepath = filepath[0]
        if isinstance(filepath, bytes):
            filepath = filepath.decode("utf-8")
        # Make sure file ends with pickle file extension
        head, tail = os.path.splitext(filepath)
        filepath = head + ".pkl"
        # Send message with selected directory to the GUI
        socketio.emit('save_file_dialog_resp', {'data': filepath}, namespace='/main', broadcast=True, include_self=True)


@socketio.on('open_load_file_dialog', namespace='/main')
def openLoadFileDialog():
    # Open a file dialog interface for selecting a directory
    filepath = webview.create_file_dialog(dialog_type=webview.OPEN_DIALOG, file_types=("Python Pickle (*.pkl)",))
    if filepath and len(filepath) > 0:
        filepath = filepath[0]
        if isinstance(filepath, bytes):
            filepath = filepath.decode("utf-8")
        if not os.path.isfile(filepath):
            socketio.emit('main-notification', {'data': "\'{}\' is not a valid file".format(directory)}, namespace='/main')
        # Send message with selected directory to the GUI
        socketio.emit('load_file_dialog_resp', {'data': filepath}, namespace='/main', broadcast=True, include_self=True)


@socketio.on('open_save_dialog', namespace='/main')
def openSaveDirDialog():
    # Open a file dialog interface for selecting a directory
    dirs = webview.create_file_dialog(webview.FOLDER_DIALOG)
    if dirs and len(dirs) > 0:
        directory = dirs[0]
        if isinstance(directory, bytes):
            directory = directory.decode("utf-8")
        if not os.path.isdir(directory):
            socketio.emit('main-notification', {'data': "\'{}\' is not a valid directory".format(directory)}, namespace='/main')
            return None
        # Send message with selected directory to the GUI
        socketio.emit('save-file-dialog-resp', {'data': directory}, namespace='/main')



'''
Participant socket handlers
'''
@socketio.on('delete_participant', namespace='/main')
def manage_participant_delete(participant_str):
    shutil.rmtree(participants[participant_str].participant_dir)
    del participants[participant_str]
    return render_template("manage_participants.html", part_keys=participants.keys())

@socketio.on('update_participant_info', namespace='/main')
def manage_participant_save(data):
    key = "participant_{}".format(data['number'])
    participants[key].set_info(data)
    participants[key].save("info")

@socketio.on('get_part_info', namespace='/main')
def get_participant_info(key):
    socketio.emit('part_info', participants[key]['info'], namespace='/main')

'''
EEG test socket handlers
'''

@socketio.on('start_test', namespace='/main')
def start_test(msg):
    test_name = msg.pop('test_name')
    part_key = msg.pop('part_key')
    thread_type = thread_types[test_name]
    socketio.emit('participant_start_{}'.format(test_name), namespace='/main')

    if part_key != "--":
        participant = participants[part_key]
    else:
        raise ValueError("Participant must be selected...")

    socketio.emit('participant_start', test_name, namespace='/main', broadcast=True)
    run_test_thread(test_name, thread_type, participant=participant, **msg)


@socketio.on('load_backup_test', namespace='/main')
def load_backup_test(msg):
    '''
    Relay test start message to participant view
    '''

    test_name = msg.pop('test_name')
    part_key = msg.pop('part_key')
    thread_type = thread_types[test_name]
    socketio.emit('participant_start_{}'.format(test_name), namespace='/main')
    if part_key != "--":
        participant = participants[part_key]
        folder = participant.data_paths[test_name]
        backupPath = os.path.join(folder, "{}_state.pkl".format(test_name))
    else:
        raise ValueError("Participant must be selected...")

    socketio.emit('participant_start', test_name, namespace='/main', broadcast=True)
    run_test_thread(test_name, thread_type, sessionFilepath=backupPath, participant=participant, **msg)


@socketio.on('load_mat_session', namespace='/main')
def start_saved_mat_test(msg):
    '''
    Relay test start message to participant view
    '''
    filepath = webview.create_file_dialog(dialog_type=webview.OPEN_DIALOG, file_types=("Python Pickle (*.pkl)",))
    if filepath and len(filepath) > 0:
        filepath = filepath[0]
        if isinstance(filepath, bytes):
            filepath = filepath.decode("utf-8")
    else:
        return None

    part_key = msg['part_key']
    if part_key != "--":
        participant = participants[part_key]
    else:
        participant = None
    socketio.emit('participant_start_mat', {'data': ''}, namespace='/main', broadcast=True)
    run_matrix_thread(sessionFilepath=filepath, participant=participant)


@socketio.on('open_mat_dialog', namespace='/main')
def openMatDialog():
    # Open a file dialog interface for selecting a directory
    dirs = webview.create_file_dialog(webview.FOLDER_DIALOG)
    if dirs and len(dirs) > 0:
        directory = dirs[0]
        if isinstance(directory, bytes):
            directory = directory.decode("utf-8")
        # TODO: Add filepath checking here...
        # Send message with selected directory to the GUI
        socketio.emit('mat-dialog-resp', {'data': directory}, namespace='/main')

'''
Matrix test stimulus generation socket handlers
'''
@socketio.on('run_mat_stim_gen', namespace='/main')
def generateStim(msg):
    '''
    When process buton is clicked in GUI, start an asynchronous thread to run
    process
    '''
    global thread
    n_part = int(msg['n_part'])
    snr_len = float(msg['snr_len'])
    snr_num = int(msg['snr_num'])
    mat_dir = msg['mat_dir']
    save_dir = msg['save_dir']
    thread = StimGenThread(n_part, snr_len, snr_num, mat_dir, save_dir, socketio=socketio)
    thread.start()


@socketio.on('check-mat-processing-status', namespace='/main')
def checkMatProcessingStatus():
    global thread
    if thread.is_alive():
        socketio.emit('mat-processing-status', {'data': True}, namespace='/main')
    else:
        socketio.emit('mat-processing-status', {'data': False}, namespace='/main')


'''
Calibration socket handlers
'''
WavPlayer = None
@socketio.on('play_calibrate', namespace='/main')
def playCalibrate():
    WavPlayer = play_wav_async('./matrix_test/stimulus/wav/noise/noise.wav', 'stop_calibrate')

@socketio.on('stop_calibrate', namespace='/main')
def stop_playback():
    WavPlayer.join()

