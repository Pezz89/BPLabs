from threading import Thread, Event
import os
import numpy as np
from matrix_test.filesystem import globDir
from pysndfile import PySndfile, sndio
from random import randint, shuffle
from shutil import copyfile
from natsort import natsorted
import numpy as np
import pandas as pd
from shutil import copy2

from test_base import BaseThread

from matrix_test.signalops import play_wav
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
        self._stopevent = Event()

        super(EEGStoryTrainThread, self).__init__(self.test_name,
                                           sessionFilepath=sessionFilepath,
                                           socketio=socketio,
                                           participant=participant)

        self.toSave = ['trial_ind', 'nTrials', 'wav_file', 'test_name']


        self.socketio.on_event('finalise_results', self.finaliseResults, namespace='/main')
        self.loadStimulus()

        self.dev_mode = True


    def testLoop(self):
        '''
        Main loop for iteratively finding the SRT
        '''
        self.waitForPageLoad()
        self.loadResponse()
        self.socketio.emit('test_ready', namespace='/main')
        # For each stimulus
        trials = list(zip(self.wav_files, self.question))[self.trial_ind:]
        for (wav, q) in trials:
            self.displayInstructions()
            self.waitForPartReady()
            if self._stopevent.isSet() or self.finishTest:
                break
            # Play concatenated matrix sentences at set SNR
            self.playStimulus(wav)
            self.setMatrix(q)
        self.saveState(out=self.backupFilepath)
        if not self._stopevent.isSet():
            self.unsetPageLoaded()
            self.socketio.emit('processing-complete', namespace='/main')
            self.waitForPageLoad()
            self.fillTable()


    def fillTable(self):
        '''
        '''
        symb = [[symb_dict[x], symb_dict[y]] for x, y in self.answers if not np.isnan([x, y]).any()]
        self.socketio.emit('test_fill_table', {'data': symb}, namespace='/main')


    def loadStimulus():
        wav_files = natsorted(globDir(self.stimDir, '*.wav'))
        q_files = natsorted(globDir(self.stimDir, '*.csv'))
        set_trace()


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
            self.play_wav('./test.wav', 'finish_test')

        self.socketio.emit("stim_done", namespace="/main")


    def loadStimulus(self):
        '''
        '''
        #audio, fs, enc, fmt = sndio.read(wav, return_format=True)
        self.answers = np.empty(np.shape(self.question)[:2])
        self.answers[:] = np.nan


    def saveState(self, out="test_state.pkl"):
        saveDict = {k:self.__dict__[k] for k in self.toSave}
        with open(out, 'wb') as f:
            dill.dump(saveDict, f)
