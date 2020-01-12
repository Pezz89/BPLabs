import numpy as np
import scipy.signal as sgnl
import matplotlib.pyplot as plt
import queue
import sys
import threading
from pysndfile import PySndfile, construct_format
from scipy.signal import square

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

def block_process_wav(wavpath, out_wavpath, func, block_size=4096, **args):
    '''
    Mix two wav files, applying gains to each
    '''
    wav = PySndfile(wavpath, 'r')

    out_wav = PySndfile(out_wavpath, 'w', construct_format('wav', 'pcm16'), wav.channels(), wav.samplerate())

    i = 0
    while i < wav.frames():
        if i+block_size > wav.frames():
            block_size = wav.frames()-i
        x = wav.read_frames(block_size)
        y = func(x, **args)
        out_wav.write_frames(y)
        i += block_size
    del out_wav

def window_rms(a, window_size):
    print("Squaring...")
    a2 = a**2
    print("Convolving...")
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'same'))

def block_mix_wavs(wavpath_a, wavpath_b, out_wavpath, a_gain=1., b_gain=1., block_size=4096, mute_left=False):
    '''
    Mix two wav files, applying gains to each
    '''
    wav_a = PySndfile(wavpath_a, 'r')
    wav_b = PySndfile(wavpath_b, 'r')

    out_wav = PySndfile(out_wavpath, 'w', construct_format('wav', 'pcm16'), wav_a.channels(), wav_a.samplerate())

    i = 0
    while i < wav_a.frames():
        if i+block_size > wav_a.frames():
            block_size = wav_a.frames()-i
        x1 = wav_a.read_frames(block_size)
        x2 = wav_b.read_frames(block_size)
        x1[:, :2] *= a_gain
        x2 *= b_gain
        if x1.shape[1] == 3:
            y = np.zeros(x1.shape)
            y[:, 0] = x1[:, 0] + x2
            y[:, 1] = x1[:, 1] + x2
            y[:, 2] = x1[:, 2]
            if mute_left:
                y[:, 0] = 0.0
        else:
            y = x1 + x2
        out_wav.write_frames(y)
        i += block_size


def gen_trigger(x, freq, length, fs):

    duty = length*freq
    trigger = square(2*np.pi*(x/fs)*freq, duty=duty)
    trigger[trigger < 0] = 0
    return trigger


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
    return rms[:-round(window/2.)]
