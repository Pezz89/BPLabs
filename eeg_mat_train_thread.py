from threading import Thread, Event
import os
import numpy as np
from matrix_test.helper_modules.filesystem import globDir
from pathops import dir_must_exist
from matrix_test.helper_modules.signalops import block_mix_wavs
from pysndfile import PySndfile, sndio
from random import randint, shuffle
from shutil import copyfile
from natsort import natsorted
import numpy as np
import pandas as pd
from shutil import copy2

from test_base import BaseThread, run_test_thread
from scipy.special import logit
from config import socketio
import csv
import pdb
import dill

symb_dict = {
    True: 10003,
    False: 10007
}

def roll_independant(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]

    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]

    result = A[rows, column_indices]
    return result

def set_trace():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log = logging.getLogger('engineio')
    log.setLevel(logging.ERROR)
    pdb.set_trace()


class EEGMatTrainThread(BaseThread):
    '''
    Thread for running server side matrix test operations
    '''
    def __init__(self, sessionFilepath=None,
                 stimFolder="./matrix_test/long_concat_stim/out/stim",
                 noiseFilepath="./matrix_test/behavioural_stim/stimulus/wav/noise/noise.wav",
                 noiseRMSFilepath="./matrix_test/behavioural_stim/stimulus/rms/noise_rms.npy",
                 red_coef="./matrix_test/short_concat_stim/out/reduction_coef.npy",
                 socketio=None, participant=None, srt_50=None, s_50=None):
        self.noise_path = noiseFilepath
        self.noise_rms = np.load(noiseRMSFilepath)
        self.stim_folder = stimFolder
        self.stim_paths = []

        self.reduction_coef = np.load(red_coef)


        self.wav_files = []
        self.marker_files = []
        self.question_files = []
        self.question = []
        self.response = []

        # Percent speech inteligibility (estimated using behavioural measure)
        # to present stimuli at
        self.si = np.array([20.0, 50.0, 90.0, 100.0])
        self.trial_ind = 0
        self._stopevent = Event()

        super(EEGMatTrainThread, self).__init__('eeg_mat_train',
                                            sessionFilepath=sessionFilepath,
                                            socketio=socketio,
                                            participant=participant)

        self.socketio.on_event('submit_eeg_response', self.submitTestResponse, namespace='/main')
        self.socketio.on_event('finalise_results', self.finaliseResults, namespace='/main')

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
        self.snr_fs = 10**(-snrs/20)
        self.snr_fs[self.snr_fs == np.inf] = 0.
        if (self.snr_fs == -np.inf).any():
            raise ValueError("Noise infinitely louder than signal for an SNR (SNRs: {})".format(self.snr_fs))


        wavs = globDir(self.stim_folder, "*.wav")
        questions = globDir(self.stim_folder, "stim_questions_*.csv")
        rms_files = globDir(self.stim_folder, "stim_*_rms.npy")

        self.socketio.emit('test_stim_load', namespace='/main')
        # Add noise to audio files at set SNRs and write to participant
        # directory
        self.data_path = self.participant.data_paths[self.test_name]
        out_dir = os.path.join(self.data_path, "stimulus")
        out_info = os.path.join(out_dir, "stim_info.csv")
        dir_must_exist(out_dir)

        with open(out_info, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['wav', 'snr_fs', 'rms', 'si', 'snr'])
            for wav, snr_fs, rms, si, snr in zip(wavs, self.snr_fs, rms_files, self.si, snrs):
                out_wavpath =  os.path.join(out_dir, os.path.basename(wav))
                stim_rms = np.load(rms)
                match_ratio = stim_rms/self.noise_rms
                block_mix_wavs(wav, self.noise_path, out_wavpath, 1.*self.reduction_coef, snr_fs*match_ratio*self.reduction_coef)
                self.stim_paths.append(out_wavpath)
                writer.writerow([wav, snr_fs, rms, si, snr])
                # TODO: Output SI/snrs of each file to a CSV file


        for q_file_path in questions:
            q = []
            with open(q_file_path, 'r') as q_file:
                q_reader = csv.reader(q_file)
                for line in q_reader:
                    q.append(line)
            self.question.append(q)
        self.answers = np.empty(np.shape(self.question)[:2])
        set_trace()


    def testLoop(self):
        '''
        Main loop for iteratively finding the SRT
        '''
        self.waitForPageLoad()
        self.loadResponse()
        self.socketio.emit(
            'test_ready',
            {'sentence_1': self.question[0][0][0], 'sentence_2':
             self.question[0][1][0]}, namespace='/main'
        )
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

    def displayInstructions(self):
        self.socketio.emit(
            'display_instructions',
            {'sentence_1': self.question[self.trial_ind][0][0], 'sentence_2':
            self.question[self.trial_ind][1][0]}, namespace='/main'
        )

    def fillTable(self):
        '''
        '''
        symb = [[symb_dict[x], symb_dict[y]] for x, y in self.answers if not np.isnan([x, y]).any()]
        self.socketio.emit('test_fill_table', {'data': symb}, namespace='/main')


    def setMatrix(self, questions):
        '''
        '''
        for self.q_ind, q in enumerate(questions):
            self.answer = q[1]
            question = q[0]
            self.socketio.emit('set_matrix', {'data': question}, namespace='/main')
            self.waitForResponse()
            if self._stopevent.isSet() or self.finishTest:
                return
            self.processResponse()
        self.trial_ind += 1
        self.saveState(out=self.backupFilepath)

    def processResponse(self):
        '''
        '''
        self.newResp = False
        self.answers[self.trial_ind, self.q_ind] = self.answer in self.response
        symb = symb_dict[self.answers[self.trial_ind, self.q_ind]]
        self.socketio.emit('test_resp', {'q_ind': self.q_ind, 'trial_ind': self.trial_ind, "ans": symb}, namespace='/main')

    def loadResponse(self):
        incomplete_responses = np.isnan(self.answers).any(axis=1)[:, np.newaxis].repeat(2, axis=1)
        self.answers[incomplete_responses] = np.nan
        self.fillTable()

    def finaliseResults(self):
        toSave = ['marker_files', 'clinPageLoaded', 'wav_files', 'participant',
                  'response', 'backupFilepath', 'noise_path', 'question_files',
                  'si', 'question', 'answers', 'trial_ind']
        saveDict = {k:self.__dict__[k] for k in toSave}
        self.participant['eeg_test'].update(saveDict)
        self.participant.save("eeg_test")

        backup_path = os.path.join(self.participant.data_paths['eeg_test'],
                        'finalised_backup.pkl')
        copy2(self.backupFilepath, backup_path)
        self.finalised = True


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

    def submitTestResponse(self, msg):
        '''
        Get and store participant response for current trial
        '''
        self.response = [x.upper() for x in msg['resp']]
        self.newResp = True


    def saveState(self, out="eeg_test_state.pkl"):
        toSave = ['marker_files', 'wav_files', 'participant', 'response',
                  'backupFilepath', 'noise_path', 'question_files', 'si',
                  'question', 'answers', 'trial_ind']
        saveDict = {k:self.__dict__[k] for k in toSave}
        with open(out, 'wb') as f:
            dill.dump(saveDict, f)
