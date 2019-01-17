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


class EEGTestThread(BaseThread):
    '''
    Thread for running server side matrix test operations
    '''
    def __init__(self, sessionFilepath=None,
                 listFolder="./matrix_test/short_concat_stim/out",
                 noiseFilepath="./matrix_test/behavioural_stim/stimulus/wav/noise/noise.wav",
                 red_coef="./calibration/out/reduction_coefficients/mat_red_coef.npy",
                 cal_coef="./calibration/out/calibration_coefficients/da_cal_coef.npy",
                 socketio=None, participant=None, srt_50=None, s_50=None):
        self.noise_path = noiseFilepath
        self.listDir = listFolder

        self.wav_files = []
        self.marker_files = []
        self.question_files = []
        self.question = []
        self.response = []

        self.reduction_coef = np.load(red_coef)*np.load(cal_coef)

        # Percent speech inteligibility (estimated using behavioural measure)
        # to present stimuli at
        self.si = np.array([20.0, 35.0, 50.0, 65.0, 80.0, 90.0])
        self.trial_ind = 0
        self._stopevent = Event()

        super(EEGTestThread, self).__init__('eeg_test',
                                            sessionFilepath=sessionFilepath,
                                            socketio=socketio,
                                            participant=participant)

        self.socketio.on_event('submit_eeg_response', self.submitTestResponse, namespace='/main')
        self.socketio.on_event('finalise_results', self.finaliseResults, namespace='/main')


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


    @staticmethod
    def logisticFunction(L, L_50, s_50):
        '''
        Calculate logistic function for SNRs L, 50% SRT point L_50, and slope
        s_50
        '''
        return 1./(1.+np.exp(4.*s_50*(L_50-L)))


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
        self.participant.load('mat_test')
        try:
            srt_50=self.participant.data['mat_test']['srt_50']
            s_50=self.participant.data['mat_test']['s_50']
        except KeyError:
            raise KeyError("Behavioural matrix test results not available, make "
                           "sure the behavioural test has been run before "
                           "running this test.")
        # Estimate speech intelligibility thresholds using predicted
        # psychometric function
        s_50 *= 0.01
        x = logit(self.si * 0.01)
        snrs = (x/(4*s_50))+srt_50
        snrs = np.append(snrs, np.inf)
        snr_map = pd.DataFrame({"speech_intel" : np.append(self.si, 0.0), "snr": snrs})
        save_dir = self.participant.data_paths['eeg_test/stimulus']
        snr_map_path = os.path.join(save_dir, "snr_map.csv")
        snr_map.to_csv(snr_map_path)
        snrs = np.repeat(snrs[np.newaxis], 4, axis=0)
        snrs = roll_independant(snrs, np.array([0,-1,-2,-3]))
        noise_file = PySndfile(self.noise_path, 'r')
        stim_dirs = [x for x in os.listdir(self.listDir) if os.path.isdir(os.path.join(self.listDir, x))]
        shuffle(stim_dirs)
        wav_files = []
        question = []
        marker_files = []
        self.socketio.emit('test_stim_load', namespace='/main')
        for ind, dir_name in enumerate(stim_dirs):
            stim_dir = os.path.join(self.listDir, dir_name)
            wav = globDir(stim_dir, "*.wav")[0]
            csv_files = natsorted(globDir(stim_dir, "*.csv"))
            marker_file = csv_files[0]
            question_files = csv_files[1:]
            rms_file = globDir(stim_dir, "*.npy")[0]
            speech_rms = float(np.load(rms_file))
            snr = snrs[:, ind]
            audio, fs, enc, fmt = sndio.read(wav, return_format=True)

            speech = audio[:, :2]
            triggers = audio[:, 2]
            wf = []
            for ind2, s in enumerate(snr):
                start = randint(0, noise_file.frames()-speech.shape[0])
                noise_file.seek(start)
                noise = noise_file.read_frames(speech.shape[0])
                noise_rms = np.sqrt(np.mean(noise**2))
                snr_fs = 10**(-s/20)
                if snr_fs == np.inf:
                    snr_fs = 0.
                elif snr_fs == -np.inf:
                    raise ValueError("Noise infinitely louder than signal at snr: {}".format(snr))
                noise = noise*(speech_rms/noise_rms)
                out_wav_path = os.path.join(save_dir, "Stim_{0}_{1}.wav".format(ind, ind2))
                out_meta_path = os.path.join(save_dir, "Stim_{0}_{1}.npy".format(ind, ind2))
                with np.errstate(divide='raise'):
                    try:
                        out_wav = (speech+(np.stack([noise, noise], axis=1)*snr_fs))*self.reduction_coef
                    except:
                        set_trace()
                out_wav = np.concatenate([out_wav, triggers[:, np.newaxis]], axis=1)
                sndio.write(out_wav_path, out_wav, fs, fmt, enc)
                np.save(out_meta_path, snr)
                wf.append(out_wav_path)
            wav_files.append(wf)
            out_marker_path = os.path.join(save_dir, "Marker_{0}.csv".format(ind))
            marker_files.append(out_marker_path)
            copyfile(marker_file, out_marker_path)
            for q_file in question_files:
                out_q_path = os.path.join(save_dir, "Questions_{0}_{1}.csv".format(ind, ind2))
                self.question_files.append(out_q_path)
                copyfile(q_file, out_q_path)

            for q_file_path in question_files:
                q = []
                with open(q_file_path, 'r') as q_file:
                    q_reader = csv.reader(q_file)
                    for line in q_reader:
                        q.append(line)
                question.append(q)

        self.wav_files = [item for sublist in wav_files for item in sublist]

        self.question.extend(question)

        for item in marker_files:
            self.marker_files.extend([item] * 4)

        c = list(zip(self.wav_files, self.marker_files, self.question))
        shuffle(c)
        self.wav_files, self.marker_files, self.question = zip(*c)

        self.answers = np.empty(np.shape(self.question)[:2])
        self.answers[:] = np.nan


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
