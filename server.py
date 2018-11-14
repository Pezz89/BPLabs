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
from pathops import dir_must_exist
from scipy.optimize import minimize

import config

server = config.server
socketio = config.socketio

def find_nearest_idx(array, value):
    '''
    Adapted from: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


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
    def __init__(self, listN=3, sessionFilepath=None, noiseFilepath="./matrix_test/stimulus/wav/noise/noise.wav",
        listFolder="./matrix_test/stimulus/wav/sentence-lists/", socketio=None):
        super(MatTestThread, self).__init__()
        self.newResp = False
        self.foundSRT = False
        self.pageLoaded = False
        self.socketio = socketio
        # Attach messages from gui to class methods
        self.socketio.on_event('submit_mat_response', self.submitMatResponse, namespace='/main')
        self.socketio.on_event('mat_page_loaded', self.setPageLoaded, namespace='/main')
        self.socketio.on_event('save_file_dialog_resp', self.manualSave, namespace='/main')
        self.socketio.on_event('load_file_dialog_resp', self.loadStateSocketHandle, namespace='/main')
        self.socketio.on_event('repeat_stimulus', self.playStimulusSocketHandle, namespace='/main')

        self.listN = listN
        self.loadedLists = []
        self.lists = []
        self.listsRMS = []
        self.listsString = []
        self.noise = None
        self.noise_rms = None
        self.fs = None

        self.response = ['','','','','']
        self.nCorrect = None
        self.snr = 0.0
        self.direction = 0
        # Record SNRs presented with each trial of the adaptive track
        self.snrTrack = np.empty(30)
        self.wordsCorrect = np.full((30, 5), False, dtype=bool)
        self.snrTrack[:] = np.nan
        self.snrTrack[0] = 0.0
        # Count number of presented trials
        self.trialN = 1

        self.currentList = None
        self.availableSentenceInds = []
        self.usedLists = []

        # Adaptive track parameters
        self.slope = 0.15
        self.i = 0

        # Plotting parameters
        self.img = io.BytesIO()

        # If loading session from file, load session variables from the file
        if sessionFilepath:
            self.loadState(sessionFilepath)
        else:
            # Preload audio at start of the test
            self.loadStimulus(listFolder, n=self.listN)
            self.loadNoise(noiseFilepath)



    def waitForResponse(self):
        while not self.newResp:
            socketio.sleep(0.75)
        return


    def waitForPageLoad(self):
        while not self.pageLoaded:
            socketio.sleep(0.2)
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
            self.saveState()
        self.plotSNR()
        self.fitLogistic()
        #socketio.emit('update-progress', {'data': '{}%'.format(percent)}, namespace='/main')

    @staticmethod
    def logisticFunction(L, L_50, s_50):
        '''
        Calculate logistic function for SNRs L, 50% SRT point L_50, and slope
        s_50
        '''
        return 1./(1.+np.exp(4.*s_50*(L_50-L)))

    def logisticFuncLiklihood(self, args):
        '''
        Calculate the log liklihood for given L_50 and s_50 parameters.
        This function is designed for use with the scipy minimize optimisation
        function to find the optimal L_50 and s_50 parameters.

        args: a tuple containing (L_50, s_50)
        self.wordsCorrect: an n dimensional binary array of shape (N, 5),
            containing the correctness of responses to each of the 5 words for N
            trials
        self.trackSNR: A sorted list of SNRs of shape N, for N trials
        '''
        L_50, s_50 = args
        ck = self.wordsCorrect[np.arange(self.trackSNR.shape[0])]
        p_lf = self.logisticFunction(self.trackSNR, L_50, s_50)
        # Reshape array for vectorized calculation of log liklihood
        p_lf = p_lf[:, np.newaxis].repeat(5, axis=1)
        # Calculate the liklihood
        res = (p_lf**ck)*(((1.-p_lf)**(1.-ck)))
        out = -np.sum(np.log(np.concatenate(res)))
        return out


    def fitLogistic(self):
        '''
        '''
        self.wordsCorrect = self.wordsCorrect[:self.trialN].astype(float)
        self.trackSNR = self.snrTrack[:self.trialN]
        res = minimize(self.logisticFuncLiklihood, np.array([-5.0,1.0]), method='L-BFGS-B')
        percent_correct = (np.sum(self.wordsCorrect, axis=1)/self.wordsCorrect.shape[1])*100.
        sortedSNRind = np.argsort(-self.trackSNR)
        sortedSNR = self.trackSNR[sortedSNRind]
        sortedPC = percent_correct[sortedSNRind]
        x = np.linspace(np.min(sortedSNR)-5, np.max(sortedSNR)+3, 3000)
        snr_50, s_50 = res.x
        x_y = self.logisticFunction(x, snr_50, s_50)
        # np.savez('./plot.npz', x, x_y*100., sortedSNR, sortedPC)

        # snrPC = pd.DataFrame(sortedPC, sortedSNR)
        # sns.kdeplot(sortedSNR, sortedPC, cmap="Blues", shade=True)
        # sns.relplot(data=snrPC)
        # sns.relplot(x, x_y, kind="line")

        plt.clf()
        #plt.plot(sortedSNR, sortedPC, "x")
        #sbnplot = sns.relplot(data=pd.DataFrame(x_y*100., x), kind="line")
        x_y *= 100.
        axes = plt.gca()
        psycLine, = axes.plot(x, x_y)
        plt.xlabel("SNR (dB)")
        plt.ylabel("% Correct")
        srtLine, = axes.plot([snr_50,snr_50], [-50,50.], 'r--')
        axes.plot([-50.,snr_50], [50.,50.], 'r--')
        plt.xlim(x.min(), x.max())
        plt.ylim(x_y.min(), x_y.max())
        plt.yticks(np.arange(5)*25.)
        x_vals = np.array(axes.get_xlim())
        s_50 *= 100.
        b = 50. - s_50 * snr_50
        y_vals = s_50 * x_vals + b
        slopeLine, = axes.plot(x_vals, y_vals, '--')
        ticks = (np.arange((x.max()-x.min())/2.5)*2.5)+(2.5 * round(float(x.min())/2.5))
        ticks[find_nearest_idx(ticks, snr_50)] = snr_50
        labels = ["{:.2f}".format(x) for x in ticks]
        plt.xticks(ticks, labels)
        plt.legend((psycLine, srtLine, slopeLine), ("Psychometric function", "SRT={:.2f}dB".format(snr_50), "Slope={:.2f}%/dB".format(s_50)))
        dpi = 300
        plt.savefig("./test_2.png", format='png', figsize=(800/dpi, 800/dpi), dpi=dpi)
        self.img.seek(0)
        plot_url = base64.b64encode(self.img.getvalue()).decode()
        plot_url = "data:image/png;base64,{}".format(plot_url)
        socketio.emit("mat_mle_plot_ready", {'data': plot_url}, namespace="/main")


    def plotSNR(self):
        '''
        '''
        plt.plot(self.snrTrack, 'bo-')
        plt.ylim([20.0, -20.0])
        plt.xticks(np.arange(30))
        plt.xlim([-1, self.trialN])
        plt.xlabel("Trial N")
        plt.ylabel("SNR (dB)")
        dpi = 300
        plt.savefig(self.img, format='png', figsize=(800/dpi, 800/dpi), dpi=dpi)
        self.img.seek(0)
        plot_url = base64.b64encode(self.img.getvalue()).decode()
        plot_url = "data:image/png;base64,{}".format(plot_url)
        socketio.emit("mat_plot_ready", {'data': plot_url}, namespace="/main")


    def calcSNR(self):
        '''
        '''
        correct = np.array([x == y for x, y in zip(self.currentWords, self.response)])
        print("Current words: {}".format(self.currentWords))
        print("Response: {}".format(self.response))
        print("Correct: {}".format(correct))
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

        # If all sentences in the current list have been presented...
        if not self.availableSentenceInds:
            # Set subsequent list as the current list
            del self.lists[0]
            del self.listsRMS[0]
            del self.listsString[0]
            if not len(self.lists):
                self.foundSRT = True
                self.wordsCorrect[self.trialN-1] = correct
                return None
            self.availableSentenceInds = list(range(len(self.lists[0])))
            random.shuffle(self.availableSentenceInds)
        self.snrTrack[self.trialN] = self.snr
        self.wordsCorrect[self.trialN-1] = correct
        self.trialN += 1

    def playStimulusSocketHandle(self):
        self.playStimulus(replay=True)

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


    def saveState(self, out="mat_state.pik"):
        toSave = ['listsRMS', 'y', 'currentList', 'foundSRT', 'slope', 'snr',
                  'snrTrack', 'direction', 'noise_rms', 'i', 'currentWords',
                  'usedLists', 'availableSentenceInds', 'trialN',
                  'listsString', 'noise', 'fs', 'nCorrect', 'loadedLists',
                  'lists', 'listN', 'wordsCorrect']
        saveDict = {k:self.__dict__[k] for k in toSave}
        with open(out, 'wb') as f:
            dill.dump(saveDict, f)


    def manualSave(self, msg):
        '''
        Get and store participant response for current trial
        '''
        filepath = msg['data']
        self.saveState(out=filepath)


    def loadStateSocketHandle(self, msg):
        filepath = msg['data']
        self.loadState(filepath)


    def loadState(self, filepath):
        with open(filepath, 'rb') as f:
            self.__dict__.update(dill.load(f))


    def run(self):
        '''
        This function is called when the thread starts
        '''
        self.testLoop()
        socketio.emit('processing-complete', {'data': ''}, namespace='/main')


@socketio.on('start_mat_test', namespace='/main')
def start_mat_test(msg):
    '''
    Relay test start message to participant view
    '''
    socketio.emit('participant_start_mat', {'data': ''}, namespace='/main', broadcast=True)
    listN = int(msg['listN'])

    global matThread
    matThread = MatTestThread(socketio=socketio, listN=listN)
    socketio.start_background_task(matThread.run)


@socketio.on('load_mat_backup', namespace='/main')
def start_backup_mat_test():
    '''
    Relay test start message to participant view
    '''
    socketio.emit('participant_start_mat', {'data': ''}, namespace='/main', broadcast=True)

    global matThread
    matThread = MatTestThread(sessionFilepath="./mat_state.pik", socketio=socketio)
    matThread.start()
    # socketio.start_background_task(matThread.run)


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
    global matThread
    matThread = MatTestThread(sessionFilepath=filepath, socketio=socketio)
    matThread.start()
    #task = socketio.start_background_task(matThread.run)
    #set_trace()


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

