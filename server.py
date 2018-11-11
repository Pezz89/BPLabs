#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
sns.set(style="ticks")
import matplotlib.pyplot as plt
import os
from flask import Flask, url_for, render_template, jsonify, request, make_response, g
from flask_socketio import emit
import pdb
import csv
import io
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

@server.route('/matrix_test/control')
def control_matrix_test():
    return render_template("mat_test_clinician_view.html")

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
        self.socketio = socketio
        # Attach messages from gui to class methods
        self.socketio.on_event('submit_mat_response', self.submitMatResponse, namespace='/main')
        self.socketio.on_event('mat_page_loaded', self.setPageLoaded, namespace='/main')
        self.socketio.on_event('mat_save_state', self.saveState, namespace='/main')
        self.socketio.on_event('mat_load_state', self.loadState, namespace='/main')

        self.loadedLists = []
        self.lists = []
        self.listsRMS = []
        self.listsString = []
        self.noise = None
        self.noise_rms = None
        self.fs = None

        self.response = ['','','','','']
        self.snr = 0.0
        self.direction = 0
        # Record SNRs presented with each trial of the adaptive track
        self.snrTrack = np.empty(30)
        self.snrTrack[:] = np.nan
        # Count number of presented trials
        self.trialN = 0

        self.currentList = None
        self.availableSentenceInds = []
        self.usedLists = []

        # Preload audio at start of the test
        self.loadStimulus(listFolder)
        self.loadNoise(noiseFilepath)

        # Adaptive track parameters
        self.slope = 0.15
        self.i = 0

        # Plotting parameters
        self.img = io.BytesIO()



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
        Main loop for iteratively finding the SRT
        '''
        self.waitForPageLoad()
        while not self.foundSRT:
            self.plotSNR()
            self.playStimulus()
            self.waitForResponse()
            self.calcSNR()
        #socketio.emit('update-progress', {'data': '{}%'.format(percent)}, namespace='/main')

    def plotSNR(self):
        '''
        '''
        self.snrTrack[self.trialN] = self.snr
        plt.plot(self.snrTrack, 'bo-')
        plt.ylim([20.0, -20.0])
        dpi = 300
        plt.savefig(self.img, format='png', figsize=(800/dpi, 800/dpi), dpi=dpi)
        self.img.seek(0)
        plot_url = base64.b64encode(self.img.getvalue()).decode()
        plot_url = "data:image/png;base64,{}".format(plot_url)
        socketio.emit("mat_plot_ready", {'data': plot_url}, namespace="/main")
        self.trialN += 1

    def calcSNR(self):
        '''
        '''
        correct = np.array([x == y for x, y in zip(self.currentWords, self.response)])
        self.nCorrect = np.sum(correct)/correct.size
        prevSNR = self.snr
        self.snr -= (((1.5*1.41**-self.i)*(self.nCorrect - 0.5))/self.slope)
        currentDirection = np.sign(np.diff([prevSNR, self.snr]))
        if self.direction != currentDirection:
            if currentDirection == 0:
                pass
            else:
                if self.direction != 0:
                    self.i += 1
                self.direction = currentDirection

        if not len(self.lists):
            self.foundSRT = True
            return
        print(self.snr)


    def playStimulus(self, replay=False):
        self.newResp = False
        socketio.emit("mat_stim_playing", namespace="/main")
        if not replay:
            self.y = self.generateTrial(self.snr)
        # Play audio
        sd.play(self.y, self.fs, blocking=True)
        socketio.emit("mat_stim_done", namespace="/main")


    def loadStimulus(self, listDir, n=3, demo=False):
        # Get folder path of all lists in the list directory
        lists = next(os.walk(listDir))[1]
        lists.pop(lists.index("demo"))
        # Don't reload an lists that have already been loaded
        pop = [lists.index(x) for x in self.loadedLists]
        for i in sorted(pop, reverse=True):
            del lists[i]
        # Randomly select n lists
        inds = list(range(len(lists)))
        random.shuffle(inds)
        # Pick first n shuffled lists
        inds = inds[:n]
        for ind in inds:
            # Get filepaths to the audiofiles and word csv file for the current
            # list
            listAudiofiles = globDir(os.path.join(listDir, lists[ind]), "*.wav")
            listCSV = globDir(os.path.join(listDir, lists[ind]), "*.csv")
            with open(listCSV[0]) as csv_file:
                csv_reader = csv.reader(csv_file)
                # Allocate empty lists to store audio samples, RMS and words of
                # each list sentence
                self.lists.append([])
                self.listsRMS.append([])
                self.listsString.append([])
                # Get data for each sentence
                for fp, words in zip(listAudiofiles, csv_reader):
                    # Read in audio file and calculate it's RMS
                    x, self.fs, _ = sndio.read(fp)
                    x_rms = np.sqrt(np.mean(x**2))
                    self.lists[-1].append(x)
                    self.listsRMS[-1].append(x_rms)
                    self.listsString[-1].append(words)

        # Shuffle order of sentence presentation
        self.availableSentenceInds = list(range(len(self.lists[0])))
        random.shuffle(self.availableSentenceInds)


    def loadNoise(self, noiseFilepath):
        '''
        Read noise samples and calculate the RMS of the signal
        '''
        x, _, _ = sndio.read(noiseFilepath)
        self.noise = x
        self.noise_rms = np.sqrt(np.mean(self.noise**2))


    def setPageLoaded(self):
        self.pageLoaded = True


    def generateTrial(self, snr):
        # If all sentences in the current list have been presented...
        if not self.availableSentenceInds:
            # Set subsequent list as the current list
            del self.lists[0]
            self.availableSentenceInds = list(range(len(self.lists[0])))
            random.shuffle(self.availableSentenceInds)
        currentSentenceInd = self.availableSentenceInds.pop(0)
        # Convert desired SNR to dB FS
        snr_fs = 10**(snr/20)
        # Get speech data
        x = self.lists[0][currentSentenceInd]
        x_rms = self.listsRMS[0][currentSentenceInd]
        self.currentWords = self.listsString[0][currentSentenceInd]
        # Get noise data
        noiseLen = x.size + self.fs
        start = random.randint(0, self.noise.size-noiseLen)
        end = start + noiseLen
        x_noise = self.noise[start:end]
        # Calculate RMS of noise
        noise_rms = np.sqrt(np.mean(x_noise**2))
        # Scale noise to match the RMS of the speech
        x_noise = x_noise*(x_rms/noise_rms)
        y = x_noise
        # Set speech to start 500ms after the noise, scaled to the desired SNR
        sigStart = round(self.fs/2.)
        y[sigStart:sigStart+x.size] += x*snr_fs
        return y


    def submitMatResponse(self, msg):
        '''
        Get and store participant response for current trial
        '''
        self.response = [x.upper() for x in msg['resp']]
        self.newResp = True


    def saveState(self):
        toSave = ['listsRMS', 'y', 'currentList', 'foundSRT', 'slope',
                  'response', 'snr', 'snrTrack', 'direction', 'noise_rms', 'i',
                  'currentWords', 'usedLists', 'availableSentenceInds',
                  'newResp', 'trialN', 'listsString', 'noise', 'fs',
                  'nCorrect', 'loadedLists', 'lists']
        saveDict = {k:self.__dict__[k] for k in toSave}
        with open('mat_state.pik', 'wb') as f:
            dill.dump(saveDict, f)


    def loadState(self):
        with open('mat_state.pik', 'rb') as f:
            self.__dict__.update(dill.load(f))
        self.plotSNR()


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

