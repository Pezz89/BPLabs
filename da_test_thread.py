from threading import Thread, Event
import os
import numpy as np
from matrix_test.helper_modules.filesystem import globDir
from pysndfile import PySndfile, sndio
from random import randint, shuffle
from shutil import copyfile
from natsort import natsorted
import numpy as np
import pandas as pd
from shutil import copy2

from test_base import BaseThread

from matrix_test.helper_modules.signalops import play_wav, block_mix_wavs
from pathops import dir_must_exist, delete_if_exists
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
                 stimFolder='./da_stim/',
                 noiseFilepath="./da_stim/noise/wav/noise/noise_norm.wav",
                 noiseRMSFilepath="./da_stim/noise/rms/noise_rms.npy",
                 daPeakFilepath="./da_stim/stimulus/peak/10min_da_peak.npy",
                 red_coef="./calibration/out/reduction_coefficients/da_red_coef.npy",
                 cal_coef="./calibration/out/calibration_coefficients/da_cal_coef.npy",
                 nTrials=2, socketio=None, participant=None, srt_50=None,
                 s_50=None):

        self.reduction_coef = np.load(red_coef)*np.load(cal_coef)
        self.wav_file = os.path.join(stimFolder, '3000_da.wav')
        self.noise_path = noiseFilepath
        self.noise_rms = np.load(noiseRMSFilepath)
        self.stim_folder = stimFolder
        self.stim_paths = []

        self.test_name = 'da_test'
        self.nTrials = nTrials
        self.trial_ind = 0
        self._stopevent = Event()
        # (Completely clean stimulus and stimulus at +10dB added by default later)
        self.si = np.array([19.0, 50.0, 81.0])

        super(DaTestThread, self).__init__(self.test_name,
                                           sessionFilepath=sessionFilepath,
                                           socketio=socketio,
                                           participant=participant)

        self.toSave = ['stim_paths', 'trial_ind', 'nTrials', 'wav_file', 'test_name', 'si']

        self.socketio.on_event('finalise_results', self.finaliseResults, namespace='/main')

        self.dev_mode = False


    def testLoop(self):
        '''
        Main loop for iteratively finding the SRT
        '''
        self.waitForPageLoad()
        self.socketio.emit('test_ready', namespace='/main')
        set_trace()
        for wav in self.stim_paths[self.trial_ind:]:
            self.saveState(out=self.backupFilepath)
            self.displayInstructions()
            self.waitForPartReady()
            if self._stopevent.isSet() or self.finishTest:
                break
            # Play concatenated matrix sentences at set SNR
            self.playStimulusWav(wav)
            self.trial_ind += 1
        self.saveState(out=self.backupFilepath)
        if not self._stopevent.isSet():
            self.unsetPageLoaded()
            self.socketio.emit('processing-complete', namespace='/main')
            self.waitForPageLoad()
            self.waitForFinalise()


    def displayInstructions(self):
        self.socketio.emit('display_instructions', namespace='/main')


    def loadStimulus(self):
        '''
        '''
        self.participant.load('mat_test')
        try:
            srt_50=self.participant.data['mat_test']['srt_50']
            s_50=self.participant.data['mat_test']['s_50']
        except KeyError:
            raise KeyError("Behavioural matrix test results not available, make "
                           "sure the behavioural test has been run before "
                           "running this test.")

        #reduction_coef = float(np.load(os.path.join(self.listDir, 'reduction_coef.npy')))

        # Calculate SNRs based on behavioural measures
        s_50 *= 0.01
        shuffle(self.si)
        x = logit(self.si * 0.01)
        snrs = (x/(4*s_50))+srt_50
        snrs = np.append(snrs, np.inf)
        snrs = np.append(snrs, 10.0)
        self.si = np.append(self.si, np.inf)
        self.si = np.append(self.si, 10.0)

        self.snr_fs = 10**(-snrs/20)
        self.snr_fs[self.snr_fs == np.inf] = 0.
        if (self.snr_fs == -np.inf).any():
            raise ValueError("Noise infinitely louder than signal for an SNR (SNRs: {})".format(self.snr_fs))


        wavs = globDir(self.stim_folder, "3000_da.wav") * len(snrs)
        rms_files = globDir(self.stim_folder, "overall_da_rms.npy") * len(snrs)

        self.socketio.emit('test_stim_load', namespace='/main')
        # Add noise to audio files at set SNRs and write to participant
        # directory
        self.data_path = self.participant.data_paths[self.test_name]
        out_dir = os.path.join(self.data_path, "stimulus")
        delete_if_exists(out_dir)
        out_info = os.path.join(out_dir, "stim_info.csv")
        dir_must_exist(out_dir)

        with open(out_info, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['wav', 'snr_fs', 'rms', 'si', 'snr'])
            for wav, snr_fs, rms, si, snr in zip(wavs, self.snr_fs, rms_files, self.si, snrs):
                fp = os.path.splitext(os.path.basename(wav))[0]+"_{}.wav".format(snr)
                out_wavpath =  os.path.join(out_dir, fp)
                stim_rms = np.load(rms)
                match_ratio = stim_rms/self.noise_rms
                block_mix_wavs(wav, self.noise_path, out_wavpath,
                               1.*self.reduction_coef,
                               snr_fs*match_ratio*self.reduction_coef,
                               mute_left=True)
                self.stim_paths.extend([out_wavpath] * self.nTrials)
                writer.writerow([wav, snr_fs, rms, si, snr])
                # TODO: Output SI/snrs of each file to a CSV file
        #audio, fs, enc, fmt = sndio.read(wav, return_format=True)


    def saveState(self, out="test_state.pkl"):
        saveDict = {k:self.__dict__[k] for k in self.toSave}
        with open(out, 'wb') as f:
            dill.dump(saveDict, f)
