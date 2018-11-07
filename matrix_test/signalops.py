import numpy as np
import scipy.signal as sgnl
import matplotlib.pyplot as plt


def rolling_window_lastaxis(a, window):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
    if window < 1:
       raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
       raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def block_lfilter(b, a, x, blocksize=8192):
    '''
    Filter 1D signal in blocks. For use with large signals
    '''
    new_state = np.zeros(a.size-1)
    out = np.zeros(x.size)
    i = 0
    while i < x.size:
        print("Filtering {0} to {1} of {2}".format(i, i+blocksize, out.size))
        if i+blocksize > x.size:
            y, new_state = sgnl.lfilter(b,a,x[i:-1], zi=new_state)
            out[i:-1]=y
        else:
            y, new_state = sgnl.lfilter(b,a,x[i:i+blocksize], zi=new_state)
            out[i:i+blocksize]=y
        i += blocksize
    return out



def calc_rms(y, window, plot=False):
    y_2 = y**2
    rms = np.zeros(y_2.size + round(window/2.))
    y_i = rolling_window_lastaxis(y_2, window)
    for ind, frame in enumerate(y_i):
        rms[ind+round(window/2.)] = np.sqrt(np.mean(frame))
    rms[np.isnan(rms)] = 0
    if plot:
        plt.plot(y)
        plt.plot(rms)
        plt.show()
    return rms
