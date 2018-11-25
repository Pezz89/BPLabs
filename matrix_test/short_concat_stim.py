#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from filesystem import globDir
import argparse
import pdb
from shutil import rmtree
from natsort import natsorted
from pysndfile import sndio, PySndfile, construct_format
from random import shuffle
from pathops import dir_must_exist, delete_if_exists
import numpy as np
import csv

def main():
    base_dir = "./stimulus/wav/sentence-lists/"
    out_dir = "./short_concat_stim/"
    folders = os.listdir(base_dir)
    folders = natsorted(folders)[1:15]
    folders = list(zip(folders[::2], folders[1::2]))

    for ind, (list_folder_1, list_folder_2) in enumerate(folders):
        out_folder_name = 'Stim_{}'.format(ind)
        out_folder = os.path.join(out_dir, out_folder_name)
        delete_if_exists(out_folder)
        dir_must_exist(out_folder)
        out_wav_path = os.path.join(out_folder, "stim.wav")
        out_csv_path = os.path.join(out_folder, "markers.csv")
        out_wav = PySndfile(out_wav_path, 'w', construct_format('wav', 'pcm16'), 1, 44100)
        list_1_wav = globDir(os.path.join(base_dir, list_folder_1), '*.wav')
        list_2_wav = globDir(os.path.join(base_dir, list_folder_2), '*.wav')
        merged_wavs = list_1_wav + list_2_wav
        shuffle(merged_wavs)
        with open(out_csv_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            counter = 0
            for wav in merged_wavs:
                csv_line = [counter]
                x, fs, enc = sndio.read(wav)
                out_wav.write_frames(x)
                counter += x.size
                csv_line.append(counter)
                csv_line.append("Speech")
                writer.writerow(csv_line)
                csv_line = [counter]
                silence = np.zeros(np.random.uniform(int(0.8*44100), int(1.2*44100), 1).astype(int))
                out_wav.write_frames(silence)
                counter += silence.size
                csv_line.append(counter)
                writer.writerow(csv_line)
                csv_line.append("Silence")

if __name__ == "__main__":
    main()
