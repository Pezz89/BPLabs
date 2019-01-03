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


class DaTestThread(BaseThread):
    '''
    Thread for running server side matrix test operations
    '''
    def __init__(self, sessionFilepath=None,
                 stimFolder='./da_stim/', nTrials=2,
                 socketio=None, participant=None, srt_50=None, s_50=None):
        self.wav_file = os.path.join(stimFolder, '3000_da.wav')

        self.test_name = 'da_test'
        self.nTrials = nTrials
        self.trial_ind = 0
        self._stopevent = Event()

        super(DaTestThread, self).__init__(self.test_name,
                                           sessionFilepath=sessionFilepath,
                                           socketio=socketio,
                                           participant=participant)

        self.toSave = ['trial_ind', 'nTrials', 'wav_file', 'test_name']

        self.socketio.on_event('finalise_results', self.finaliseResults, namespace='/main')

        self.dev_mode = False


    def testLoop(self):
        '''
        Main loop for iteratively finding the SRT
        '''
        self.waitForPageLoad()
        self.socketio.emit('test_ready', namespace='/main')
        for self.trial_ind in range(self.nTrials):
            self.displayInstructions()
            self.waitForPartReady()
            if self._stopevent.isSet() or self.finishTest:
                break
            # Play concatenated matrix sentences at set SNR
            self.playStimulus(self.wav_file)
            self.saveState(out=self.backupFilepath)
        if not self._stopevent.isSet():
            self.unsetPageLoaded()
            self.socketio.emit('processing-complete', namespace='/main')


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


    def saveState(self, out="test_state.pkl"):
        saveDict = {k:self.__dict__[k] for k in self.toSave}
        with open(out, 'wb') as f:
            dill.dump(saveDict, f)
