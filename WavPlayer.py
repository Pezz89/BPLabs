from threading import Thread, Event
import sounddevice as sd
import soundfile as sf
from config import socketio

import queue


def play_wav_async(wav_file, stop_string):
    wavThread = WavPlayer(wav_file, socketio=socketio, stop_string=stop_string)
    wavThread.start()
    return wavThread

def play_wav(wav_file, stop_string):
    wavThread = WavPlayer(wav_file, socketio=socketio, stop_string=stop_string)
    wavThread.run()
    return wavThread

class WavPlayer(Thread):
    '''
    Thread for running server side matrix test operations
    '''
    def __init__(self, wav_file, socketio=None, stop_string="stop"):
        super(WavPlayer, self).__init__()
        self.socketio = socketio
        self._stopevent = Event()
        self.socketio.on_event(stop_string, self.join, namespace='/main')
        self.wav_file = wav_file

    def join(self, timeout=None):
        """ Stop the thread. """
        self._stopevent.set()
        try:
            Thread.join(self, timeout)
        except RuntimeError:
            pass


    def play_wav_async(self, wav_file, buffersize=20, blocksize=1024, socketio=None):
        q = queue.Queue(maxsize=buffersize)

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
                callback=callback, finished_callback=self._stopevent.set)
            with stream:
                timeout = blocksize * buffersize / f.samplerate
                while data and not self._stopevent.isSet():
                    data = f.buffer_read(blocksize, dtype='float32')
                    q.put(data, timeout=timeout)
                self._stopevent.wait()  # Wait until playback is finished
            return stream

    def run(self):
        '''
        This function is called when the thread starts
        '''
        return self.play_wav_async(self.wav_file)
