import pdb

import os
import sounddevice as sd
import dill

from shutil import copy2
from threading import Thread, Event
from config import socketio

from WavPlayer import WavPlayer

def run_test_thread(name, thread_type, sessionFilepath=None, participant=None, **kwargs):
    thread_name = '{}TestThread'.format(name)
    if thread_name in globals():
        thread = globals()[thread_name]
        if thread.isAlive() and isinstance(thread, thread_type):
            daTestThread.join()
    thread = thread_type(socketio=socketio, sessionFilepath=sessionFilepath,
                         participant=participant, **kwargs)
    thread.start()

def set_trace():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log = logging.getLogger('engineio')
    log.setLevel(logging.ERROR)
    pdb.set_trace()


class BaseThread(Thread):
    '''
    Thread for running server side matrix test operations
    '''
    def __init__(self, test_name, sessionFilepath=None,
                 socketio=None, participant=None, **kwargs):
        super(BaseThread, self).__init__()
        self.participant=participant
        self.socketio = socketio
        self.test_name = test_name

        self.pageLoaded = False
        self.clinPageLoaded = False
        self.partPageLoaded = False
        self.finishTest = False
        self.partReady = False
        self.newResp = False
        self.finalised = False

        # Define variables to be saved in state files. Should be implemented in
        # derived class
        self.toSave = None
        self.toFinalise = None

        self.wavThread = None

        # Attach handler methods to socketio messages
        self.socketio.on_event('page_loaded', self.setPageLoaded, namespace='/main')
        self.socketio.on_event('part_ready', self.setPartReady, namespace='/main')
        self.socketio.on_event('finalise_results', self.finaliseResults, namespace='/main')
        self.socketio.on_event('finish_test', self.finishTestEarly, namespace='/main')
        self.socketio.on_event('save_file_dialog_resp', self.manualSave, namespace='/main')
        self.socketio.on_event('load_file_dialog_resp', self.loadStateSocketHandle, namespace='/main')

        self._stopevent = Event()
        # Attach messages from gui to class methods
        folder = self.participant.data_paths[test_name]
        self.backupFilepath=os.path.join(folder, '{}_state.pkl'.format(test_name))

        # If loading session from file, load session variables from the file
        if sessionFilepath:
            self.loadState(sessionFilepath)
        else:
            # Preload audio at start of the test
            self.loadStimulus()

        self.dev_mode = False


    def play_wav(self, wav_file, stop_string='stop_audio'):
        self.wavThread = WavPlayer(wav_file, socketio=socketio, stop_string=stop_string)
        self.wavThread.run()

    def testLoop(self):
        '''
        Main loop
        '''
        raise NotImplemented("Test loop code should not be called from the base "
                             "class. This should be implemented in the derived "
                             "test class")

    def displayInstructions(self):
        '''
        Emit signal to display test instructions
        '''
        self.socketio.emit('display_instructions', data=self.test_name, namespace='/main')


    def finishTestEarly(self):
        '''
        Set variables to finish the test as soon as possible and exit the
        thread
        '''
        self.finishTest = True
        if self.wavThread:
            self.wavThread._stopevent.set()


    def join(self, timeout=None):
        '''
        Stop the thread.
        '''
        self._stopevent.set()
        Thread.join(self, timeout)


    def waitForResponse(self):
        '''
        Test waits for a response from the participant. To use his function
        correctly, self.newResp must be set to True via a socketio handler in
        order to continue the test.
        '''
        while not self.newResp and not self._stopevent.isSet() and not self.finishTest:
            self._stopevent.wait(0.2)
        return


    def waitForPageLoad(self):
        '''
        Wait for page to load and poll for a socketio message to be sent
        informing the thread of a successfully loaded page
        '''
        while not self.pageLoaded and not self._stopevent.isSet():
            self.socketio.emit("check-loaded", namespace='/main')
            self._stopevent.wait(0.5)
        self.pageLoaded = False


    def waitForPartReady(self):
        '''
        Test waits for the participant to finish reading instructions. To use
        this function correctly, self.partReady must be set to True via a socketio
        handler in order to continue the test.
        '''
        while not self.partReady and not self._stopevent.isSet() and not self.finishTest:
            self._stopevent.wait(0.5)
        self.partReady = False
        return

    def waitForFinalise(self):
        '''
        Wait for results to be finalised by socketio handler
        '''
        while not self.finalised and not self._stopevent.isSet() and not self.finishTest:
            self._stopevent.wait(0.2)
        self.socketio.emit("test_finished", namespace='/main')
        return


    def finaliseResults(self):
        saveDict = {k:self.__dict__[k] for k in self.toSave}
        self.participant[self.test_name].update(saveDict)
        self.participant.save(self.test_name)
        backup_path = os.path.join(self.participant.data_paths[self.test_name],
                        'finalised_backup.pkl')
        copy2(self.backupFilepath, backup_path)
        self.finalised = True


    def playStimulusWav(self, wav_file, replay=False):
        '''
        output audio stimulus from wav file
        '''
        self.newResp = False
        self.socketio.emit("{}_stim_playing".format(self.test_name), namespace="/main")
        if not self.dev_mode:
            self.play_wav(wav_file)
        else:
            self.play_wav('./da_stim/DA_170.wav')

        self.socketio.emit("{}_stim_done".format(self.test_name), namespace="/main")


    def playStimulus(self, y, fs):
        '''
        Output audio stimulus from numpy array
        '''
        self.newResp = False
        self.socketio.emit("stim_playing", namespace="/main")
        # Play audio
        if not self.dev_mode:
            sd.play(y, fs, blocking=True)
        else:
            self.play_wav('./da_stim/DA_170.wav', '')
        self.socketio.emit("stim_done", namespace="/main")


    def loadStimulus(self):
        '''
        Method for preloading stimulus before the start of the test. Should be
        implemented in child class.
        '''
        raise NotImplemented("loadStimulus code should not be called from the base "
                             "class. This should be implemented in the derived "
                             "test class")

    def unsetPageLoaded(self):
        '''
        For use in the main loop when a new page is loaded for
        participant/clinician
        '''
        self.pageLoaded = False
        self.partPageLoaded = False
        self.clinPageLoaded = False


    def setPartReady(self):
        '''
        Set variables indicating that the participant is ready to proceed with
        the test
        '''
        self.partReady = True


    def setPageLoaded(self, msg):
        '''
        Indicate that either the clinician or participant page has been loaded
        '''
        if msg['data'] == "clinician":
            self.clinPageLoaded = True
        else:
            self.partPageLoaded = True
        self.pageLoaded = self.clinPageLoaded and self.partPageLoaded


    def saveState(self, out=None):
        '''
        Save the state of the thread to a pickle file
        '''
        if not out:
            out = "{}_state.pkl".format(self.test_name)
        saveDict = {k:self.__dict__[k] for k in self.toSave}
        with open(out, 'wb') as f:
            dill.dump(saveDict, f)


    def manualSave(self, msg):
        '''
        Get and store participant response for current trial
        '''
        filepath = msg['data']
        self.saveState(out=filepath)


    def loadStateSocketHandle(self, msg):
        '''
        Catch messages indicating that the thread should be loaded from a
        previously generated pickle file
        '''
        filepath = msg['data']
        self.loadState(filepath)


    def loadState(self, filepath):
        '''
        Restore thread state from a saved session filepath
        '''
        with open(filepath, 'rb') as f:
            self.__dict__.update(dill.load(f))


    def run(self):
        '''
        This function is called when the thread starts
        '''
        return self.testLoop()
