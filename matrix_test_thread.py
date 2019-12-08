import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, Event
import io
import dill
import base64
import os
import random
from scipy.optimize import minimize
import csv
from shutil import copy2

from pysndfile import sndio, PySndfile
from matrix_test.helper_modules.filesystem import globDir
from test_base import BaseThread
import sounddevice as sd
import pdb

from config import socketio


def run_matrix_thread(listN=3, sessionFilepath=None, participant=None):
    global matThread
    if 'matThread' in globals():
        if matThread.isAlive() and isinstance(matThread, MatTestThread):
            matThread.join()
    matThread = MatTestThread(socketio=socketio, listN=listN,
                              sessionFilepath=sessionFilepath,
                              participant=participant)
    matThread.start()


def set_trace():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log = logging.getLogger('engineio')
    log.setLevel(logging.ERROR)
    pdb.set_trace()


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


class MatTestThread(BaseThread):
    '''
    Thread for running server side matrix test operations
    '''
    def __init__(self, listN=5, sessionFilepath=None,
                 noiseFilepath="./matrix_test/behavioural_stim/stimulus/wav/noise/noise_norm.wav",
                 noiseRMSFilepath="./matrix_test/behavioural_stim/stimulus/rms/noise_rms.npy",
                 listFolder="./matrix_test/behavioural_stim/stimulus/wav/sentence-lists/",
                 red_coef="./calibration/out/reduction_coefficients/mat_red_coef.npy",
                 cal_coef="./calibration/out/calibration_coefficients/mat_cal_coef.npy",
                 track_targets=[0.2, 0.5, 0.8],
                 socketio=None, participant=None):

        self.listDir = listFolder

        self.adaptiveTracks = [AdaptiveTrack(x, red_coef, cal_coef) for x in track_targets]
        self.trackOrder = []


        self.listN = int(listN)
        self.loadedLists = []
        self.lists = []
        self.listsRMS = []
        self.listsString = []
        self.presentedWords = []
        self.noise = None
        self.noise_rms = None
        self.fs = None
        self.nCorrect = None

        self.response = ['','','','','']
        self.responses = []
        self.currentWords = []
        self.srt_50 = None
        self.s_50 = None

        self.wordsCorrect = np.full((180, 5), False, dtype=bool)
        self.trialN = 0

        self.availableSentenceInds = []

        # Plotting parameters
        self.img = io.BytesIO()
        self.img.seek(0)
        self.img.truncate(0)

        super(MatTestThread, self).__init__('mat_test',
                                            sessionFilepath=sessionFilepath,
                                            socketio=socketio,
                                            participant=participant)

        self.toSave = ['trackOrder', 'presentedWords', 'responses', 'srt_50', 's_50',
                       'wordsCorrect', 'trialN', 'test_name', 'backupFilepath',
                       'currentWords', 'nCorrect', 'availableSentenceInds',
                       'lists', 'listsRMS', 'listsString']
        self.toFinalise = ['trackSNR', 'trackOrder', 'trialN', 'wordsCorrect',
                           'presentedWords', 'responses', 'srt_50', 's_50']

        # Attach messages from gui to class methods
        self.socketio.on_event('submit_response', self.submitMatResponse, namespace='/main')
        self.socketio.on_event('page_loaded', self.setPageLoaded, namespace='/main')
        self.socketio.on_event('repeat_stimulus', self.playStimulusSocketHandle, namespace='/main')
        self.socketio.on_event('finalise_results', self.finaliseResults, namespace='/main')

        self.loadNoise(noiseFilepath, noiseRMSFilepath)

        self.dev_mode = False


    def displayInstructions(self):
        self.socketio.emit('display_instructions', namespace='/main')


    def renderSNRPlot(self):
        dpi = 300

        maxTrialN = np.max([x.trialN for x in self.adaptiveTracks])
        plt.xlim([-1, maxTrialN+1])
        plt.savefig(self.img, format='png', figsize=(800/dpi, 800/dpi), dpi=dpi)
        self.img.seek(0)
        plot_url = base64.b64encode(self.img.getvalue()).decode()
        plot_url = "data:image/png;base64,{}".format(plot_url)
        self.socketio.emit("mat_plot_ready", {'data': plot_url}, namespace="/main")
        plt.clf()

    def testLoop(self):
        '''
        Main loop for iteratively finding the SRT
        '''
        self.waitForPageLoad()

        self.displayInstructions()
        self.waitForPartReady()
        while not self.finishTest and not self._stopevent.isSet() and len(self.availableSentenceInds) and len(self.trackOrder):
            # Plot SNR of current trial to the clinician screen
            for at in self.adaptiveTracks:
                at.plotSNR()
            self.renderSNRPlot()
            # Get the index of the sentence to be played for the current trial
            currentSentenceInd = self.availableSentenceInds.pop(0)
            # Get the index of the current adaptive track to use
            self.adTrInd = self.trackOrder.pop(0)
            # Generate trial audioself.wordsCorrect
            self.y = self.adaptiveTracks[self.adTrInd].generateTrial(
                self.lists[0][currentSentenceInd],
                self.listsRMS[0][currentSentenceInd]
            )
            # Define words presented in the current trial
            self.currentWords = self.listsString[0][currentSentenceInd]

            self.playStimulus(self.y, self.fs)
            self.waitForResponse()
            self.checkSentencesAvailable()
            if self.finishTest:
                break
            if self._stopevent.isSet():
                return
            self.adaptiveTracks[self.adTrInd].calcSNR(self.nCorrect)
            self.checkSentencesAvailable()
            self.saveState(out=self.backupFilepath)
            self.trialN += 1
            self.adaptiveTracks[self.adTrInd].incrementTrialN()
        self.saveState(out=self.backupFilepath)
        if not self._stopevent.isSet():
            self.unsetPageLoaded()
            self.socketio.emit('processing-complete', {'data': ''}, namespace='/main')
            self.waitForPageLoad()
            # Plot SNR of current trial to the clinician screen
            for at in self.adaptiveTracks:
                at.plotSNR()
            self.renderSNRPlot()
            self.fitLogistic()
            self.waitForFinalise()



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
        with np.errstate(divide='raise'):
            try:
                a = np.concatenate(res)
                a[a == 0] = np.finfo(float).eps
                out = -np.sum(np.log(a))
            except:
                set_trace()
        return out


    def fitLogistic(self):
        '''
        '''
        self.wordsCorrect = np.concatenate([x.getWordsCorrect() for x in self.adaptiveTracks])
        self.trackSNR = np.concatenate([x.getSNRTrack() for x in self.adaptiveTracks])
        res = minimize(self.logisticFuncLiklihood, np.array([np.mean(self.trackSNR),1.0]))
        percent_correct = (np.sum(self.wordsCorrect, axis=1)/self.wordsCorrect.shape[1])*100.
        sortedSNRind = np.argsort(-self.trackSNR)
        sortedSNR = self.trackSNR[sortedSNRind]
        sortedPC = percent_correct[sortedSNRind]
        x = np.linspace(np.min(sortedSNR)-5, np.max(sortedSNR)+3, 3000)
        srt_50, s_50 = res.x
        x_y = self.logisticFunction(x, srt_50, s_50)
        x_y *= 100.
        # np.savez('./plot.npz', x, x_y*100., sortedSNR, sortedPC)

        # snrPC = pd.DataFrame(sortedPC, sortedSNR)
        # sns.kdeplot(sortedSNR, sortedPC, cmap="Blues", shade=True)
        # sns.relplot(data=snrPC)
        # sns.relplot(x, x_y, kind="line")

        #plt.plot(sortedSNR, sortedPC, "x")
        #sbnplot = sns.relplot(data=pd.DataFrame(x_y*100., x), kind="line")
        plt.clf()
        axes = plt.gca()
        psycLine, = axes.plot(x, x_y)
        plt.title("Predicted psychometric function")
        plt.xlabel("SNR (dB)")
        plt.ylabel("% Correct")
        srtLine, = axes.plot([srt_50,srt_50], [-50,50.], 'r--')
        axes.plot([-50.,srt_50], [50.,50.], 'r--')
        plt.xlim(x.min(), x.max())
        plt.ylim(x_y.min(), x_y.max())
        plt.yticks(np.arange(5)*25.)
        x_vals = np.array(axes.get_xlim())
        s_50 *= 100.
        b = 50. - s_50 * srt_50
        y_vals = s_50 * x_vals + b
        slopeLine, = axes.plot(x_vals, y_vals, '--')
        ticks = (np.arange((x.max()-x.min())/2.5)*2.5)+(2.5 * round(float(x.min())/2.5))
        ticks[find_nearest_idx(ticks, srt_50)] = srt_50
        labels = ["{:.2f}".format(x) for x in ticks]
        plt.xticks(ticks, labels)
        plt.legend((psycLine, srtLine, slopeLine), ("Psychometric function", "SRT={:.2f}dB".format(srt_50), "Slope={:.2f}%/dB".format(s_50)))
        dpi = 300
        plt.savefig(self.img, format='png', figsize=(800/dpi, 800/dpi), dpi=dpi)
        self.img.seek(0)
        plot_url = base64.b64encode(self.img.getvalue()).decode()
        plot_url = "data:image/png;base64,{}".format(plot_url)
        self.srt_50, self.s_50 = srt_50, s_50
        self.socketio.emit("mat_mle_plot_ready", {'data': plot_url}, namespace="/main")



    def checkSentencesAvailable(self):
        # If all sentences in the current list have been presented...
        if not self.availableSentenceInds:
            # Set subsequent list as the current list
            del self.lists[0]
            del self.listsRMS[0]
            del self.listsString[0]
            if not len(self.lists):
                self.finishTest = True
                return None
            self.availableSentenceInds = list(range(len(self.lists[0])))
            random.shuffle(self.availableSentenceInds)


    def playStimulusSocketHandle(self):
        self.playStimulus(self.y, self.fs)

    def loadStimulus(self):
        # Get folder path of all lists in the list directory
        lists = next(os.walk(self.listDir))[1]
        lists.pop(lists.index("demo"))
        # Don't reload an lists that have already been loaded
        pop = [lists.index(x) for x in self.loadedLists]
        for i in sorted(pop, reverse=True):
            del lists[i]
        # Randomly select n lists
        inds = list(range(len(lists)))
        random.shuffle(inds)
        # Pick first n shuffled lists
        inds = inds[:self.listN]
        for ind in inds:
            # Get filepaths to the audiofiles and word csv file for the current
            # list
            listAudiofiles = globDir(os.path.join(self.listDir, lists[ind]), "*.wav")
            listCSV = globDir(os.path.join(self.listDir, lists[ind]), "*.csv")
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

        # Number of trials to split between adaptive tracks
        n = len(self.lists[0])*self.listN
        tn = len(self.adaptiveTracks)
        self.trackOrder = list(np.repeat(np.arange(tn), np.floor(n/tn)))
        random.shuffle(self.trackOrder)

        # Shuffle order of sentence presentation
        self.availableSentenceInds = list(range(len(self.lists[0])))
        random.shuffle(self.availableSentenceInds)


    def loadNoise(self, noiseFilepath, noiseRMSFilepath):
        '''
        Read noise samples and calculate the RMS of the signal
        '''
        noise = PySndfile(noiseFilepath, 'r')
        noise_rms = np.load(noiseRMSFilepath)
        for ind, _ in enumerate(self.adaptiveTracks):
            self.adaptiveTracks[ind].setNoise(noise, noise_rms)




    def submitMatResponse(self, msg):
        '''
        Get and store participant response for current trial
        '''
        self.response = [x.upper() for x in msg['resp']]
        self.responses.append(self.response)
        self.newResp = True
        correct = np.array([x == y for x, y in zip(self.currentWords, self.response)])
        self.nCorrect = np.sum(correct)/correct.size
        self.wordsCorrect[self.trialN] = correct
        self.adaptiveTracks[self.adTrInd].wordsCorrect[self.adaptiveTracks[self.adTrInd].trialN] = correct
        self.presentedWords.append(self.currentWords)


    def loadState(self, filepath):
        '''
        Restore thread state from a saved session filepath
        '''
        with open(filepath, 'rb') as f:
            state = dill.load(f)
            aTrack = state.pop('adaptiveTracks')
            for ind, aTrackDict in enumerate(aTrack):
                self.adaptiveTracks[ind].loadFromDict(aTrackDict)
            self.__dict__.update(state)

    def saveState(self, out=None):
        '''
        Save the state of the thread to a pickle file
        '''
        if not out:
            out = "{}_state.pkl".format(self.test_name)
        saveDict = {k:self.__dict__[k] for k in self.toSave}

        saveDict['adaptiveTracks'] = []
        for ind, _ in enumerate(self.adaptiveTracks):
            atDict = self.adaptiveTracks[ind].createSaveDict()
            saveDict['adaptiveTracks'].append(atDict)
        with open(out, 'wb') as f:
            dill.dump(saveDict, f)


    def finaliseResults(self):
        saveDict = {k:self.__dict__[k] for k in self.toFinalise}
        saveDict['adaptiveTracks'] = []
        for ind, _ in enumerate(self.adaptiveTracks):
            atDict = self.adaptiveTracks[ind].createSaveDict()
            saveDict['adaptiveTracks'].append(atDict)
        self.participant[self.test_name].update(saveDict)
        self.participant.save(self.test_name)
        backup_path = os.path.join(self.participant.data_paths[self.test_name],
                        'finalised_backup.pkl')
        copy2(self.backupFilepath, backup_path)
        self.finalised = True

class AdaptiveTrack():
    '''
    '''
    def __init__(self, target, red_coef, cal_coef):
        '''
        '''
        self.snr = 10.0
        self.direction = 0
        # Record SNRs presented with each trial of the adaptive track
        self.snrTrack = np.empty(180)
        self.snrTrack[:] = np.nan
        self.snrTrack[0] = 0.0
        # Count number of presented trials
        self.trialN = 0
        self.reduction_coef = np.load(red_coef)*np.load(cal_coef)
        self.wordsCorrect = np.full((180, 5), False, dtype=bool)

        # Adaptive track parameters
        self.slope = 0.15
        self.i = 0
        self.target = target

        self.fs = 44100

        # Plotting parameters
        self.img = io.BytesIO()
        self.img.seek(0)
        self.img.truncate(0)

    def setNoise(self, noise, noise_rms):
        self.noise = noise
        self.noise_rms = noise_rms

    def getWordsCorrect(self):
        return self.wordsCorrect[:self.trialN].astype(float)

    def getSNRTrack(self):
        return self.snrTrack[:self.trialN]

    def processResponse(resp):
        pass

    def incrementTrialN(self):
        self.trialN += 1


    def generateTrial(self, x, x_rms):
        # Convert desired SNR to dB FS
        snr_fs = 10**(self.snr/20.)
        # Get speech data
        # Get noise data
        noiseLen = x.size + self.fs*2.5
        start = random.randint(0, self.noise.frames()-noiseLen)
        end = start + noiseLen
        self.noise.seek(start)
        x_noise = self.noise.read_frames(end-start)
        # Scale noise to match the RMS of the speech
        x_noise *= x_rms/self.noise_rms
        y = x_noise
        # Set speech to start 500ms after the noise, scaled to the desired SNR
        sigStart = random.randint(self.fs, round(2*self.fs))
        y[sigStart:sigStart+x.size] += x*snr_fs
        y *= self.reduction_coef
        return y


    def calcSNR(self, nCorrect):
        '''
        '''
        prevSNR = self.snr
        self.snr -= (((1.5*1.41**-self.i)*(nCorrect - self.target))/self.slope)
        currentDirection = np.sign(np.diff([prevSNR, self.snr]))
        if self.direction != currentDirection:
            if currentDirection == 0:
                pass
            else:
                if self.direction != 0:
                    self.i += 1
                self.direction = currentDirection
        self.snrTrack[self.trialN] = self.snr


    def plotSNR(self):
        '''
        '''
        plt.plot(self.snrTrack, 'o-')
        plt.ylim([20.0, -30.0])
        plt.xticks(np.arange(180))
        plt.xlabel("Trial N")
        plt.ylabel("SNR (dB)")
        plt.title("Adaptive track")
        for i, txt in enumerate(self.snrTrack[:self.trialN]):
            plt.annotate("{0}/{1}".format(
                np.sum(self.wordsCorrect[i]).astype(int),
                self.wordsCorrect[i].size),
                (i, self.snrTrack[i]),
                xytext=(0, 13),
                va="center",
                ha="center",
                textcoords='offset points'
            )

    def createSaveDict(self):
        toSave = ['snr', 'direction', 'snrTrack', 'trialN', 'reduction_coef', 'slope',
         'i', 'fs', 'wordsCorrect']
        saveDict = {k:self.__dict__[k] for k in toSave}
        return saveDict


    def loadFromDict(self, stateDict):
        self.__dict__.update(stateDict)
