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
        out_rms_path = os.path.join(out_folder, "rms.npy")
        out_wav = PySndfile(out_wav_path, 'w', construct_format('wav', 'pcm16'), 1, 44100)
        list_1_wav = globDir(os.path.join(base_dir, list_folder_1), '*.wav')
        list_2_wav = globDir(os.path.join(base_dir, list_folder_2), '*.wav')
        list_1_csv = globDir(os.path.join(base_dir, list_folder_1), '*.csv')
        list_2_csv = globDir(os.path.join(base_dir, list_folder_2), '*.csv')
        merged_wavs = list_1_wav + list_2_wav
        merged_csvs = list_1_csv + list_2_csv
        words = []
        for c in merged_csvs:
            with open(c, 'r') as csvfile:
                for line in csv.reader(csvfile):
                    words.append(line)
        c = list(zip(merged_wavs, words))
        shuffle(c)
        merged_wavs, words = zip(*c)
        sum_sqrd = 0.
        n = 0
        with open(out_csv_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            counter = 0
            for wav, txt in zip(merged_wavs, words):
                csv_line = [counter]
                silence = np.zeros(np.random.uniform(int(0.8*44100), int(1.2*44100), 1).astype(int))
                out_wav.write_frames(silence)
                counter += silence.size
                csv_line.append(counter)
                csv_line.append("#")
                writer.writerow(csv_line)
                csv_line = [counter]
                x, fs, enc = sndio.read(wav)
                sum_sqrd = np.sum(x**2)
                n += x.size
                out_wav.write_frames(x)
                counter += x.size
                csv_line.append(counter)
                csv_line.append(" ".join(txt))
                writer.writerow(csv_line)
            csv_line = [counter]
            silence = np.zeros(np.random.uniform(int(0.8*44100), int(1.2*44100), 1).astype(int))
            out_wav.write_frames(silence)
            counter += silence.size
            csv_line.append(counter)
            csv_line.append("#")
            writer.writerow(csv_line)
            rms = np.sqrt(sum_sqrd/n)
            np.save(out_rms_path, rms)

if __name__ == "__main__":
    main()
