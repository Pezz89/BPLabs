import sys
sys.path.insert(0, "../helper_modules/")
from filesystem import globDir
import pdb
import os

def main():
    wavs = globDir('./', 'stim.wav')
    for wav in wavs:
        out_filepath = "{0}_old{1}".format(head, tail)
        out_temppath = "{0}_temp{1}".format(head, tail)
        os.rename(wav, out_temppath)
        os.rename(out_filepath, wav)
        os.rename(out_temppath, out_filepath)

if __name__ == '__main__':
    main()
