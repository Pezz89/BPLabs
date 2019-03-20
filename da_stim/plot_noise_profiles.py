# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pysndfile.sndio as sndio
import pandas as pd

def voss(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.

    nrows: number of values to generate
    rcols: number of random sources to add

    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values


da_x, da_fs, da_enc = sndio.read('./stimulus/wav/10min_da.wav')
sp_x, sp_fs, sp_enc = sndio.read('./noise_source/male_speech_resamp.wav')

# %load test.ipy
pink_n = voss(sp_x.size, 1)
da_rms = np.sqrt(np.mean(da_x**2))
sp_rms = np.sqrt(np.mean(sp_x**2))
pink_n_rms = np.sqrt(np.mean(pink_n**2))
da_x *= sp_rms / da_rms
pink_n *= sp_rms / pink_n_rms
f, Pxx_den = signal.welch(pink_n, sp_fs, nperseg=1024)
plt.semilogy(f, Pxx_den)
f, Pxx_den = signal.welch(sp_x, sp_fs, nperseg=1024)
plt.semilogy(f, Pxx_den)
f, Pxx_den = signal.welch(da_x[:, 1], da_fs, nperseg=1024)
plt.semilogy(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.xlim([0, 10000])
plt.legend(['Pink noise', 'Speech shaped noise', 'Da'])
plt.savefig('./test.png')
