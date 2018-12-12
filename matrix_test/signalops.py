import numpy as np
import scipy.signal as sgnl
import matplotlib.pyplot as plt
import queue
import sys
import threading

import sounddevice as sd
import soundfile as sf



def play_wav(wav_file, buffersize=20, blocksize=1024, socketio=None):
    q = queue.Queue(maxsize=buffersize)
    event = threading.Event()


    def callback(outdata, frames, time, status):
        assert frames == blocksize
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort
        assert not status
        try:
            data = q.get_nowait()
        except queue.Empty:
            print('Buffer is empty: increase buffersize?', file=sys.stderr)
            raise sd.CallbackAbort
        if len(data) < len(outdata):
            outdata[:len(data)] = data
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
            raise sd.CallbackStop
        else:
            outdata[:] = data

    with sf.SoundFile(wav_file) as f:
        for _ in range(buffersize):
            data = f.buffer_read(blocksize, dtype='float32')
            if not data:
                break
            q.put_nowait(data)  # Pre-fill queue

        stream = sd.RawOutputStream(
            samplerate=f.samplerate, blocksize=blocksize,
            channels=f.channels, dtype='float32',
            callback=callback, finished_callback=event.set)
        with stream:
            timeout = blocksize * buffersize / f.samplerate
            while data:
                data = f.buffer_read(blocksize, dtype='float32')
                q.put(data, timeout=timeout)
            event.wait()  # Wait until playback is finished
        return stream

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
