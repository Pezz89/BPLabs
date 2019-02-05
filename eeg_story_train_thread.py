from threading import Thread, Event
import os
import numpy as np
from pysndfile import PySndfile, sndio
from random import randint, shuffle
from shutil import copyfile
from natsort import natsorted
import numpy as np
import pandas as pd
from shutil import copy2

from test_base import BaseThread

from matrix_test.helper_modules.signalops import play_wav
from matrix_test.helper_modules.filesystem import globDir
from scipy.special import logit
from config import socketio
import csv
import pdb
import dill

symb_dict = {
    True: 10003,
    False: 10007
}

def set_trace():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log = logging.getLogger('engineio')
    log.setLevel(logging.ERROR)
    pdb.set_trace()


class EEGStoryTrainThread(BaseThread):
    '''
    Thread for running server side matrix test operations
    '''
    def __init__(self, sessionFilepath=None,
                 stimFolder='./eeg_story_stim/', nTrials=2,
                 socketio=None, participant=None, srt_50=None, s_50=None):
        self.test_name = 'eeg_story_train'
        self.stimDir = stimFolder
        self.nTrials = nTrials
        self.trial_ind = 0

        self.selected_q = []
        self.question = []
        self.answers = [''] * 8
        self.wav_files = []
        self.q_files = []

        self._stopevent = Event()

        super(EEGStoryTrainThread, self).__init__(self.test_name,
                                           sessionFilepath=sessionFilepath,
                                           socketio=socketio,
                                           participant=participant)

        self.toSave = ['trial_ind', 'answers', 'question', 'selected_q', 'nTrials', 'wav_files', 'test_name']


        self.socketio.on_event('submit_response', self.submitTestResponse, namespace='/main')
        self.socketio.on_event('finalise_results', self.finaliseResults, namespace='/main')
        self.loadStimulus()

        self.dev_mode = False

    def setQuestion(self, q):
        self.socketio.emit('set_question', data=q[0], namespace='/main')

    def testLoop(self):
        '''
        Main loop for iteratively finding the SRT
        '''
        self.waitForPageLoad()
        self.fillTable()
        self.socketio.emit('test_ready', namespace='/main')
        # For each stimulus
        trials = list(zip(self.wav_files, self.question))[self.trial_ind:]
        for (wav, q) in trials:
            self.saveState(out=self.backupFilepath)
            self.displayInstructions()
            self.setQuestion(q)
            self.waitForPartReady()
            if self._stopevent.isSet() or self.finishTest:
                break
            # Play concatenated matrix sentences at set SNR
            self.playStimulus(wav)
            self.waitForResponse()
            if self._stopevent.isSet() or self.finishTest:
                break
            self.processResponse()
            self.trial_ind += 1
        self.saveState(out=self.backupFilepath)
        if not self._stopevent.isSet():
            self.unsetPageLoaded()
            self.socketio.emit('processing-complete', namespace='/main')
            self.waitForPageLoad()
            self.fillTable()
            self.waitForFinalise()

    def submitTestResponse(self, msg):
        '''
        Get and store participant response for current trial
        '''
        self.answer = msg
        self.newResp = True

    def processResponse(self):
        '''
        '''
        self.newResp = False
        self.answers[self.trial_ind] = self.answer
        self.socketio.emit('test_resp', {'trial_ind': self.trial_ind, "ans": self.answer}, namespace='/main')

    def fillTable(self):
        '''
        '''
        for ind, ans in enumerate(self.answers):
            self.socketio.emit('test_resp', {'trial_ind': ind, "ans": ans}, namespace='/main')

    def loadStimulus(self):
        '''
        '''
        self.wav_files = natsorted(globDir(self.stimDir, '*.wav'))
        q_files = natsorted(globDir(self.stimDir, '*.csv'))
        set_trace()
        for wav_file, q_file in zip(self.wav_files, q_files):
            q_lines = []
            with open(q_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    q_lines.append((int(line[0]), line[1:]))
            q_ind = randint(0, len(q_lines)-1)
            self.question.append(q_lines[q_ind][1])


    def displayInstructions(self):
        self.socketio.emit('display_instructions', namespace='/main')


    def playStimulus(self, wav_file, replay=False):
        self.newResp = False
        self.socketio.emit("stim_playing", namespace="/main")
        # if not replay:
        #     self.y = self.generateTrial(self.snr)
        # Play audio
        # sd.play(self.y, self.fs, blocking=True)
        if not self.dev_mode:
            self.play_wav(wav_file, 'finish_test')
        else:
            self.play_wav('./da_stim/DA_170.wav', 'finish_test')

        self.socketio.emit("stim_done", namespace="/main")


    def saveState(self, out="test_state.pkl"):
        saveDict = {k:self.__dict__[k] for k in self.toSave}
        with open(out, 'wb') as f:
            dill.dump(saveDict, f)
