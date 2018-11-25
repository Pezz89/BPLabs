
def run_eeg_train_thread(sessionFilepath=None, participant=None):
    global eegTrainThread
    if 'eegTrainThread' in globals():
        if eegTrainThread.isAlive() and isinstance(eegTrainThread, EEGTrainThread):
            eegTrainThread.join()
    eegTrainThread = EEGTrainThread(socketio=socketio, listN=listN,
                              sessionFilepath=sessionFilepath,
                              participant=participant)
    eegTrainThread.start()

