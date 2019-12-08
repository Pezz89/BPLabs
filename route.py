#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from flask import Flask, url_for, render_template, jsonify, request, make_response, g
from flask_socketio import emit
import pdb
import csv
import io
import re
import base64
import shutil

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
from matrix_test.helper_modules.filesystem import globDir, organiseWavs, prepareOutDir
from matrix_test_thread import MatTestThread
from pathops import dir_must_exist
from gen_participants import Participant, find_participants, gen_participant_num

from config import server, socketio, participants


'''
General routing
'''
@server.route("/")
def landing():
    """
    Render index.html
    """
    return render_template("index.html")

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

'''
Participant routing
'''
@server.route('/participant/manage')
def manage_participant_page():
    # Find all pre-existing participants
    participants = find_participants()
    return render_template("manage_participants.html", part_keys=participants.keys())

@server.route('/participant/create')
def create_participant_page():
    # Find all pre-existing participants
    participants = find_participants()
    part_num = gen_participant_num(participants)
    return render_template("create_participant.html", num=part_num)

def set_trace():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log = logging.getLogger('engineio')
    log.setLevel(logging.ERROR)
    pdb.set_trace()
@server.route('/participant/create/submit', methods=["POST"])
def create_participant_submit():
    data = request.form
    key = "participant_{}".format(data['number'])
    participants[key] = Participant(participant_dir="./participant_data/{}".format(key), **data)
    participants[key].save("info")
    return render_template("manage_participants.html", part_keys = participants.keys())

'''
EEG routing
'''
@server.route('/eeg')
def eeg_setup():
    participants = find_participants()
    return render_template("eeg_setup.html", part_keys=participants.keys())

@server.route('/eeg/test/run')
def eeg_test_run():
    return render_template("eeg_test_run.html")

@server.route('/eeg/test/complete')
def eeg_test_end():
    return render_template("eeg_test_end.html")

@server.route('/eeg/test/clinician/control')
def eeg_test_clinician_control():
    return render_template("eeg_test_clinician_view.html")

@server.route('/eeg/test/clinician/complete')
def eeg_test_clinician_end():
    return render_template("eeg_test_clinician_end.html")

@server.route('/eeg/train/story/run')
def eeg_story_train_run():
    return render_template("eeg_story_train_run.html")

@server.route('/eeg/train/story/complete')
def eeg_story_train_end():
    return render_template("eeg_story_train_end.html")

@server.route('/eeg/train/story/clinician/control')
def eeg_story_train_clinician_control():
    return render_template("eeg_story_train_clinician_view.html")

@server.route('/eeg/train/story/clinician/complete')
def eeg_story_train_clinician_end():
    return render_template("eeg_story_train_clinician_end.html")

@server.route('/eeg/train/mat/run')
def eeg_mat_train_run():
    return render_template("eeg_mat_train_run.html")

@server.route('/eeg/train/mat/complete')
def eeg_mat_train_end():
    return render_template("eeg_mat_train_end.html")

@server.route('/eeg/train/mat/clinician/control')
def eeg_mat_train_clinician_control():
    return render_template("eeg_mat_train_clinician_view.html")

@server.route('/eeg/train/mat/clinician/complete')
def eeg_mat_train_clinician_end():
    return render_template("eeg_mat_train_clinician_end.html")

'''
Matrix behavioral test routing
'''
@server.route('/matrix_test')
def matrix_test_setup():
    participants = find_participants()
    return render_template("matrix_test_setup.html", part_keys=participants.keys())

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

'''
Click stimulus routing
'''
@server.route('/click/setup')
def click_setup():
    participants = find_participants()
    return render_template("click_test_setup.html", part_keys=participants.keys())

@server.route('/click/clinician/run')
def click_clinician_run():
    return render_template("click_test_clinician_view.html")

@server.route('/click/clinician/complete')
def click_clinician_complete():
    return render_template("click_test_clinician_end.html")

@server.route('/click/run')
def click_run():
    return render_template("click_test_run.html")

@server.route('/click/complete')
def click_complete():
    return render_template("click_test_end.html")


'''
/da/ stimulus routing
'''
@server.route('/da/setup')
def da_setup():
    participants = find_participants()
    return render_template("da_test_setup.html", part_keys=participants.keys())

@server.route('/da/clinician/run')
def da_clinician_run():
    return render_template("da_test_clinician_view.html")

@server.route('/da/clinician/complete')
def da_clinician_complete():
    return render_template("da_test_clinician_end.html")

@server.route('/da/run')
def da_run():
    return render_template("da_test_run.html")

@server.route('/da/complete')
def da_complete():
    return render_template("da_test_end.html")

'''
Calibration routing
'''
@server.route('/calibrate')
def calibrate():
    return render_template("calibrate.html")

'''
Basic audiology test routing
'''
@server.route('/pta_test')
def pta():
    return render_template("pta.html")

@server.route('/tympanometry')
def typms():
    return render_template("tympanometry.html")
