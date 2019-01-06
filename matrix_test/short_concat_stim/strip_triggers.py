import sys
sys.path.insert(0, "../helper_modules/")
from filesystem import globDir
from pysndfile import sndio
import pdb
import os

def main():
    wavs = globDir('./', 'stim.wav')
    for wav in wavs:
        x, fs, enc, fmt = sndio.read(wav, return_format=True)
        y = x[:, :2]
        head, tail = os.path.splitext(wav)
        out_filepath = "{0}_old{1}".format(head, tail)
        os.rename(wav, out_filepath)
        sndio.write(wav, y, rate=fs, format=fmt, enc=enc)

if __name__ == '__main__':
    main()
