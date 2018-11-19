#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
sns.set(style="ticks")
import matplotlib.pyplot as plt
import pandas as pd
import os
from flask import Flask, url_for, render_template, jsonify, request, make_response, g
from flask_socketio import emit
import pdb
import csv
import io
import re
import base64
import dill

import sounddevice as sd
import webview
import webbrowser
import app
import time
from threading import Thread, Event
import numpy as np
import random
from pysndfile import sndio
from scipy.optimize import minimize

from app import generate_matrix_stimulus
from matrix_test.filesystem import globDir, organiseWavs, prepareOutDir
from matrix_test_thread import MatTestThread
from pathops import dir_must_exist
from participant import Participant

import config

server = config.server
socketio = config.socketio

participants = []


class StimGenThread(Thread):
    '''
    Thread object for asynchronous processing of data in Python without locking
    up the GUI
    '''
    def __init__(self, *args, **kwargs):
        super(StimGenThread, self).__init__()
        self.args = args
        self.kwargs = kwargs


    def process_stimulus(self):
        '''
        An example process
        '''
        filenames = generate_matrix_stimulus(*self.args, **self.kwargs)
        generateSpeechShapedNoise(args['OutDir'], noiseDir, order=20, plot=True)
        #socketio.emit('update-progress', {'data': '{}%'.format(percent)}, namespace='/main')


    def run(self):
        '''
        This function is called when the thread starts
        '''
        self.process_stimulus()
        socketio.emit('processing-complete', {'data': ''}, namespace='/main')


def run_matrix_thread(listN=None, sessionFilepath=None):
    global matThread
    if 'matThread' in globals():
        if matThread.isAlive() and isinstance(matThread, MatTestThread):
            matThread.join()
    matThread = MatTestThread(socketio=socketio, listN=listN, sessionFilepath=sessionFilepath)
    matThread.start()


@server.after_request
def add_header(response):
    # Disable caching? unsure why...
    response.headers['Cache-Control'] = 'no-store'
    return response


@server.route("/")
def landing():
    """
    Render index.html
    """
    return render_template("index.html")


@server.route("/choose/path")
def choose_path():
    """
    Invokes a folder selection dialog
    """
    dirs = webview.create_file_dialog(webview.FOLDER_DIALOG)
    if dirs and len(dirs) > 0:
        directory = dirs[0]
        if isinstance(directory, bytes):
            directory = directory.decode("utf-8")

        response = {"status": "ok", "directory": directory}
    else:
        response = {"status": "cancel"}

    return jsonify(response)


@server.route("/fullscreen")
def fullscreen():
    webview.toggle_fullscreen()
    return jsonify({})

@server.route('/participant/create')
def create_participant_page():
    # Find all pre-existing participants
    participant_locs = find_participants()
    part_num = gen_participant_num(participant_locs)
    return render_template("create_participant.html", num=part_num)

def find_participants(folder='./participant_data/'):
    '''
    Returns a tuple of (participant number, participant filepath) for every
    participant folder found in directory provided
    '''
    part_folder = [os.path.join(folder, o) for o in os.listdir(folder)
                        if os.path.isdir(os.path.join(folder,o))]
    part_nums = []
    for path in part_folder:
        bn = os.path.basename(path)
        m = re.search(r'\d+$', bn)
        num = int(m.group())
        part_nums.append(num)
    return list(zip(part_nums, part_folder))

def gen_participant_num(participant_locs, N = 100):
    # generate array of numbers that haven't been taken between 0-100
    # if list is empty increment until list isnt empty
    # Choose a number
    taken_nums = []
    for num, loc in participant_locs:
        taken_nums.append(num)
    n = 0
    num_found = False
    while not num_found:
        potential_nums = np.arange(N)+n+1
        valid_nums = np.setdiff1d(potential_nums, taken_nums)
        if valid_nums.size:
            num_found = True
        else:
            n += N
    return np.random.choice(valid_nums)

@server.route('/participant/create/submit', methods=["POST"])
def create_participant_submit():
    data = request.form
    participants.append(Participant(participant_dir="./participant_data/participant_{}".format(data['number']), **data))
    return render_template("manage_participants.html")

@server.route('/participant_home')
def participant_homepage():
    title = "Welcome"
    paragraph = [
        "Please wait while the experimenter sets up the test..."
    ]

    try:
        return render_template("participant_home.html", title = title, paragraph=paragraph)
    except Exception as e:
        return str(e)


@server.route('/home')
def homepage():
    title = "Welcome"
    paragraph = [
        "This is the clinician view. Use the dropdown menus to access controls "
        "and feedback for the various tests available.",
        "This web app was developed for the generation of stimulus and "
        "running of experiments for the PhD project \"Predicting speech "
        "in noise performance using evoked responses\"."
    ]

    try:
        return render_template("home.html", title = title, paragraph=paragraph)
    except Exception as e:
        return str(e)


@server.route('/matrix_test')
def matrix_test_setup():
    return render_template("matrix_test_setup.html")

@server.route('/matrix_test/run')
def run_matrix_test():
    return render_template("mat_test_run.html")

@server.route('/matrix_test/complete')
def mat_end():
    return render_template("mat_test_end.html")

@server.route('/matrix_test/clinician/control')
def clinician_control_mat():
    return render_template("mat_test_clinician_view.html")

@server.route('/matrix_test/clinician/complete')
def clinician_mat_end():
    return render_template("mat_test_clinician_end.html")

@server.route('/matrix_test/stimulus_generation')
def matDecStim():
    return render_template("matrix_decode_stim.html")
@socketio.on('start_mat_test', namespace='/main')
def start_mat_test(msg):
    '''
    Relay test start message to participant view
    '''
    socketio.emit('participant_start_mat', {'data': ''}, namespace='/main', broadcast=True)
    listN = int(msg['listN'])

    run_matrix_thread(listN=listN)


@socketio.on('load_mat_backup', namespace='/main')
def start_backup_mat_test():
    '''
    Relay test start message to participant view
    '''
    socketio.emit('participant_start_mat', {'data': ''}, namespace='/main', broadcast=True)

    run_matrix_thread(sessionFilepath="./mat_state.pik")


@socketio.on('load_mat_session', namespace='/main')
def start_saved_mat_test():
    '''
    Relay test start message to participant view
    '''
    filepath = webview.create_file_dialog(dialog_type=webview.OPEN_DIALOG, file_types=("Python Pickle (*.pik)",))
    if filepath and len(filepath) > 0:
        filepath = filepath[0]
        if isinstance(filepath, bytes):
            filepath = filepath.decode("utf-8")
    else:
        return None
    socketio.emit('participant_start_mat', {'data': ''}, namespace='/main', broadcast=True)
    run_matrix_thread(sessionFilepath=filepath)


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


@socketio.on('open_save_file_dialog', namespace='/main')
def openSaveFileDialog():
    # Open a file dialog interface for selecting a directory
    filepath = webview.create_file_dialog(dialog_type=webview.SAVE_DIALOG,
                                          file_types=("Python Pickle (*.pik)",),
                                          save_filename="test_session.pik")
    if filepath and len(filepath) > 0:
        filepath = filepath[0]
        if isinstance(filepath, bytes):
            filepath = filepath.decode("utf-8")
        # Make sure file ends with pickle file extension
        head, tail = os.path.splitext(filepath)
        filepath = head + ".pik"
        # Send message with selected directory to the GUI
        socketio.emit('save_file_dialog_resp', {'data': filepath}, namespace='/main', broadcast=True, include_self=True)


@socketio.on('open_load_file_dialog', namespace='/main')
def openLoadFileDialog():
    # Open a file dialog interface for selecting a directory
    filepath = webview.create_file_dialog(dialog_type=webview.OPEN_DIALOG, file_types=("Python Pickle (*.pik)",))
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


@server.route('/click_stim')
def clickStim():
    return render_template("click_stim.html")


@server.route('/da_stim')
def daStim():
    return render_template("da_stim.html")


def set_trace():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log = logging.getLogger('engineio')
    log.setLevel(logging.ERROR)
    pdb.set_trace()


def run_server():
    '''
    Start the Flask server
    '''
    # SocketIO objects are defined in config.py
    socketio.run(server, host="127.0.0.1", port=23948)

if __name__ == "__main__":
    run_server()

