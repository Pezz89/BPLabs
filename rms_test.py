from pysndfile import sndio
from snrops import rms_no_silences

def main():
    x, fs, enc = sndio.read('./matrix_test/behavioural_stim/stimulus/wav/sentence-lists/ukmatrix10.1/Trial_00001.wav')
    rms = rms_no_silences(x, fs, -30)
    breakpoint()


if __name__ == '__main__':
    main()
