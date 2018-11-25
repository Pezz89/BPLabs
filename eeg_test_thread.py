from threading import Thread, Event
import os
import numpy as np
from matrix_test.filesystem import globDir
from pysndfile import PySndfile

from scipy.special import logit
from config import socketio
import pdb

def run_eeg_test_thread(sessionFilepath=None, participant=None):
    global eegTestThread
    if 'eegTestThread' in globals():
        if eegTestThread.isAlive() and isinstance(eegTestThread, EEGTestThread):
            eegTestThread.join()
    eegTestThread = EEGTestThread(socketio=socketio,
                              sessionFilepath=sessionFilepath,
                              participant=participant)
    eegTestThread.start()

def set_trace():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log = logging.getLogger('engineio')
    log.setLevel(logging.ERROR)
    pdb.set_trace()


class EEGTestThread(Thread):
    '''
    Thread for running server side matrix test operations
    '''
    def __init__(self, sessionFilepath=None,
                 listFolder="./matrix_test/short_concat_stim/",
                 noiseFilepath="./matrix_test/stimulus/wav/noise/noise.wav",
                 socketio=None, participant=None, srt_50=None, s_50=None):
        super(EEGTestThread, self).__init__()
        self.participant=participant
        self.socketio = socketio
        self.noise_path = noiseFilepath

        # Percent speech inteligibility (estimated using behavioural measure)
        # to present stimuli at
        self.si = np.array([20.0, 35.0, 50.0, 65.0, 80.0, 90.0, 100.0])

        self.pageLoaded = False
        self._stopevent = Event()
        # Attach messages from gui to class methods
        if self.participant:
            folder = self.participant.data_paths['eeg_test_data']
            self.backupFilepath=os.path.join(folder, 'eeg_test_state.pkl')
        else:
            self.backupFilepath='./eeg_test_state.pkl'

        # If loading session from file, load session variables from the file
        if sessionFilepath:
            self.loadState(sessionFilepath)
        elif self.participant:
            # Preload audio at start of the test
            self.participant.load('adaptive_matrix_data')
            srt_50 = self.participant.data['adaptive_matrix_data']['srt_50']
            s_50 = self.participant.data['adaptive_matrix_data']['s_50']
            self.loadStimulus(listFolder, srt_50, s_50)
        else:
            self.loadStimulus(listFolder, srt_50, s_50)


    def testLoop(self):
        '''
        Main loop for iteratively finding the SRT
        '''
        self.waitForPageLoad()
        # For each stimulus
        while not self.foundSRT and not self._stopevent.isSet():
            # Play concatenated matrix sentences at set SNR
            self.playStimulus()
            self.waitForResponse()
            if self._stopevent.isSet():
                return
            self.saveState(out=self.backupFilepath)
        self.saveState(out=self.backupFilepath)
        if not self._stopevent.isSet():
            self.unsetPageLoaded()
            self.socketio.emit('processing-complete', {'data': ''}, namespace='/main')
            self.waitForPageLoad()


    def finishTestEarly(self):
        '''
        '''


    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        Thread.join(self, timeout)


    def waitForResponse(self):
        while not self.newResp and not self._stopevent.isSet() and not self.foundSRT:
            self._stopevent.wait(0.2)
        return


    def waitForPageLoad(self):
        while not self.pageLoaded and not self._stopevent.isSet():
            self.socketio.emit("check-loaded", namespace='/main')
            self._stopevent.wait(0.5)
        return

    def waitForFinalise(self):
        while not self.finalised and not self._stopevent.isSet():
            self._stopevent.wait(0.2)
        return


    def finaliseResults(self):
        toSave = ['snrTrack', 'trialN', 'wordsCorrect', 'presentedWords', 'responses']
        saveDict = {k:self.__dict__[k] for k in toSave}
        if self.participant:
            self.participant['adaptive_matrix_data'].update(saveDict)
            self.participant.save("adaptive_matrix_data")
            backup_path = os.path.join(self.participant.data_paths['adaptive_matrix_data'],
                         'finalised_backup.pkl')
            copy2(self.backupFilepath, backup_path)
        else:
            copy2(self.backupFilepath, './finalised_backup.pkl')
            with open('./Matrix_test_results.pkl', 'wb') as f:
                dill.dump(saveDict, f)
        self.finalised = True



    @staticmethod
    def logisticFunction(L, L_50, s_50):
        '''
        Calculate logistic function for SNRs L, 50% SRT point L_50, and slope
        s_50
        '''
        return 1./(1.+np.exp(4.*s_50*(L_50-L)))


    def playStimulus(self, replay=False):
        self.newResp = False
        self.socketio.emit("mat_stim_playing", namespace="/main")
        if not replay:
            self.y = self.generateTrial(self.snr)
        # Play audio
        sd.play(self.y, self.fs, blocking=True)
        self.socketio.emit("mat_stim_done", namespace="/main")


    def loadStimulus(self, listDir, srt_50, s_50):
        '''
        '''
        # Estimate speech intelligibility thresholds using predicted
        # psychometric function
        s_50 *= 0.01
        x = logit(self.si * 0.01)
        snrs = (x/(4*s_50))+srt_50
        noise_file = PySnfile(self.noise_path, 'r')
        for ind, dir_name in enumerate(os.listdir(listDir)):
            stim_dir = os.path.join(listDir, dir_name)
            wav = globDir(stim_dir, "*.wav")[0]
            marker = globDir(stim_dir, "*.csv")[0]
            snr = snrs[ind]
            speech, fs, enc, fmt = sndio.read(wav, return_format=True)
            start = randint(0, noise.frames()-speech.size)
            noise_file.seek(start)
            noise = noise_file.read_frames(speech.size)

            set_trace()

        # Generate wav files for noise/stim mixtures based on psychometric
        # function


    def loadNoise(self, noiseFilepath):
        '''
        Read noise samples and calculate the RMS of the signal
        '''
        x, _, _ = sndio.read(noiseFilepath)
        self.noise = x
        self.noise_rms = np.sqrt(np.mean(self.noise**2))


    def unsetPageLoaded(self):
        self.pageLoaded = False
        self.partPageLoaded = False
        self.clinPageLoaded = False

    def setPageLoaded(self, msg):
        if msg['data'] == "clinician":
            self.clinPageLoaded = True
        else:
            self.partPageLoaded = True
        self.pageLoaded = self.clinPageLoaded and self.partPageLoaded


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


    def saveState(self, out="mat_state.pkl"):
        toSave = ['listsRMS', 'y', 'currentList', 'slope', 'snr', 'snrTrack',
                  'direction', 'noise_rms', 'i', 'currentWords', 'usedLists',
                  'availableSentenceInds', 'trialN', 'listsString', 'noise',
                  'fs', 'nCorrect', 'loadedLists', 'lists', 'listN',
                  'wordsCorrect', 'responses', 'presentedWords']
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
        return self.testLoop()
