from pysndfile import sndio
import numpy as np
import pdb
import matplotlib.pyplot as plt

def main():
    '''
    Generate train of equally spaced clicks
    '''
    fs = 44100
    n = np.arange(fs*10)
    trig_s = np.where(np.mod(n, fs/2.) == 0)
    click = np.ones(int(0.01*fs))
    y = np.zeros(n.size)
    for i in trig_s[0]:
        y[i:i+click.size] = click

    sndio.write("./trig_test.wav", y, fs, format='wav', enc='pcm16')

    pdb.set_trace()
if __name__ == "__main__":
    main()
