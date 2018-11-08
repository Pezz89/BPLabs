#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from flask import Flask, url_for, render_template, jsonify, request, make_response, g
from flask_socketio import emit
import pdb

import sounddevice as sd
import webview
import webbrowser
import app
import time
from threading import Thread, Event
import numpy as np
import random

from pysndfile import sndio
from app import generate_matrix_stimulus
from matrix_test.filesystem import globDir, organiseWavs, prepareOutDir

import config

server = config.server
socketio = config.socketio


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
def matrix_test():
    return render_template("matrix_test_setup.html")

@server.route('/matrix_test/run')
def run_matrix_test():
    return render_template("run_matrix_test.html")

@server.route('/matrix_test/stimulus_generation')
def matDecStim():
    return render_template("matrix_decode_stim.html")

# thread = Thread()
# thread_stop_event = Event()

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


class MatTestThread(Thread):
    '''
    Thread for running server side matrix test operations
    '''
    def __init__(self, noiseFilepath="./matrix_test/stimulus/wav/noise/noise.wav",
        listFolder="./matrix_test/stimulus/wav/sentence-lists/", socketio=None):
        super(MatTestThread, self).__init__()
        self.newResp = False
        self.foundSRT = False
        self.pageLoaded = False
        self.response = ['','','','','']
        self.currentSRT = 0.0
        self.socketio = socketio
        self.socketio.on_event('submit_mat_response', self.submitMatResponse, namespace='/main')
        self.socketio.on_event('mat_page_loaded', self.setPageLoaded, namespace='/main')

        self.loadedLists = []
        self.noise = None
        self.lists = []
        self.listsRMS = []
        self.fs = None

        self.currentList = None
        self.availableSentenceInds = []
        self.usedLists = []

        self.loadStimulus(listFolder)
        self.loadNoise(noiseFilepath)


    def waitForResponse(self):
        while not self.newResp:
            time.sleep(0.75)
        return


    def waitForPageLoad(self):
        while not self.pageLoaded:
            time.sleep(0.75)
        return


    def testLoop(self):
        '''
        An example process
        '''
        self.waitForPageLoad()
        while not self.foundSRT:
            self.playStimulus()
            self.waitForResponse()
        #socketio.emit('update-progress', {'data': '{}%'.format(percent)}, namespace='/main')


    def playStimulus(self):
        self.newResp = False
        socketio.emit("mat_stim_playing", namespace="/main")
        y = self.generateTrial(0.0)
        sd.play(y, self.fs, blocking=True)
        # Play audio
        socketio.emit("mat_stim_done", namespace="/main")

    def preloadStimulus(self, listDir):
        return


    def loadStimulus(self, listDir, n=20, demo=False):
        lists = next(os.walk(listDir))[1]
        lists.pop(lists.index("demo"))
        pop = [lists.index(x) for x in self.loadedLists]
        for i in sorted(pop, reverse=True):
            del lists[i]
        # Randomly select n lists
        inds = list(range(n))
        random.shuffle(inds)
        for ind in inds:
            listAudiofiles = globDir(os.path.join(listDir, lists[ind]), "*.wav")
            self.lists.append([])
            self.listsRMS.append([])
            for fp in listAudiofiles:
                x, self.fs, _ = sndio.read(fp)
                x_rms = np.sqrt(np.mean(x**2))
                self.lists[-1].append(x)
                self.listsRMS[-1].append(x_rms)
        self.currentListInd = random.randint(0, n)
        self.usedLists.append(self.currentList)
        self.availableSentenceInds = list(range(len(self.lists[self.currentListInd])))
        random.shuffle(self.availableSentenceInds)



    def loadNoise(self, noiseFilepath):
        x, _, _ = sndio.read(noiseFilepath)
        self.noise = x

    def setPageLoaded(self):
        self.pageLoaded = True

    def generateTrial(self, snr):
        # Load speech
        currentSentenceInd = self.availableSentenceInds.pop(0)
        x = self.lists[self.currentListInd][currentSentenceInd]
        x_rms = self.listsRMS[self.currentListInd][currentSentenceInd]
        # Load noise
        noiseLen = x.size + self.fs
        start = random.randint(0, self.noise.size-noiseLen)
        end = start + noiseLen
        x_noise = self.noise[start:end]
        set_trace()
        y = x_noise
        sigStart = round(self.fs/2.)
        y[sigStart:sigStart+x.size] = x
        # Mix speech and noise at set SNR
        return y


    def submitMatResponse(self, msg):
        '''
        '''
        self.response = msg['resp']
        self.newResp = True


    def run(self):
        '''
        This function is called when the thread starts
        '''
        self.testLoop()
        socketio.emit('processing-complete', {'data': ''}, namespace='/main')


@socketio.on('start_mat_test', namespace='/main')
def start_mat_test():
    '''
    Relay test start message to participant view
    '''
    socketio.emit('participant_start_mat', {'data': ''}, namespace='/main', broadcast=True)

    global matThread
    thread = MatTestThread(socketio=socketio)
    thread.start()


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



@socketio.on('open_save_dialog', namespace='/main')
def openSaveDialog():
    # Open a file dialog interface for selecting a directory
    dirs = webview.create_file_dialog(webview.FOLDER_DIALOG)
    if dirs and len(dirs) > 0:
        directory = dirs[0]
        if isinstance(directory, bytes):
            directory = directory.decode("utf-8")
    # TODO: Add filepath checking here...
    # Send message with selected directory to the GUI
    socketio.emit('save-dialog-resp', {'data': directory}, namespace='/main')


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
    socketio.init_app(server)
    socketio.run(server, host="127.0.0.1", port=23948)

if __name__ == "__main__":
    run_server()

