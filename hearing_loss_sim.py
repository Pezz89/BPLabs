from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

def apply_hearing_loss_sim(x, fs, channels=[0, 1]):
    b, a = signal.butter(4, 1170.0/(fs/2.), 'low')
    if len(x.shape) < 2:
        x = x[:, np.newaxis]
    for channel in channels:
        x[:, channel] = signal.filtfilt(b, a, x[:, channel])
    return x
    # w, h = signal.freqs(b, a)

