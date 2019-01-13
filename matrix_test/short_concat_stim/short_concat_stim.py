#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../helper_modules/")

import os
from filesystem import globDir
import argparse
import pdb
from shutil import rmtree
from natsort import natsorted
from pysndfile import sndio, PySndfile, construct_format
from random import shuffle, randint, sample
from pathops import dir_must_exist, delete_if_exists
import numpy as np
import csv
from copy import copy
from contextlib import ExitStack
from scipy.signal import square

def calc_potential_max(stim_folder, noise_filepath, out_dir):
    max_wav_samp = 0
    max_wav_rms = 0
    wavs = globDir(stim_folder, '*.wav')
    for wav in wavs:
        x, fs, enc = sndio.read(wav)
        max_wav_samp = np.max([max_wav_samp, np.max(np.abs(x))])
        max_wav_rms = np.max([max_wav_rms, np.sqrt(np.mean(x**2))])
    x, fs, enc = sndio.read(noise_filepath)
    noise_rms = np.sqrt(np.mean(x**2))
    max_noise_samp = max(np.abs(x))

    snr = -15.0
    snr_fs = 10**(-snr/20)
    max_noise_samp *= max_wav_rms/noise_rms
    max_sampl = max_wav_samp+(max_noise_samp*snr_fs)
    reduction_coef = 1.0/max_sampl
    np.save(os.path.join(out_dir, "reduction_coef.npy"), reduction_coef)


def gen_trigger(idx, freq, length, fs):

    duty = length*freq
    trigger = square(2*np.pi*(idx/fs)*freq, duty=duty)
    trigger[trigger < 0] = 0
    return trigger

def main():
    stim_dir = "../behavioural_stim/stimulus"
    wav_dir = "../behavioural_stim/stimulus/wav"
    base_dir = "../behavioural_stim/stimulus/wav/sentence-lists/"
    noise_dir = "../behavioural_stim/stimulus/wav/noise/"
    out_dir = "./out"
    dir_must_exist(base_dir)
    dir_must_exist(out_dir)
    dir_must_exist(wav_dir)
    dir_must_exist(noise_dir)

    noise_filepath = "../behavioural_stim/stimulus/wav/noise/noise.wav"

    folders = os.listdir(base_dir)
    folders = natsorted(folders)[1:15]
    folders = list(zip(folders[::2], folders[1::2]))
    calc_potential_max(base_dir, noise_filepath, out_dir)
    n_questions = 4
    fs = 44100

    for ind, (list_folder_1, list_folder_2) in enumerate(folders):
        out_folder_name = 'Stim_{}'.format(ind)
        out_folder = os.path.join(out_dir, out_folder_name)
        delete_if_exists(out_folder)
        dir_must_exist(out_folder)
        out_wav_path = os.path.join(out_folder, "stim.wav")
        out_csv_path = os.path.join(out_folder, "markers.csv")
        out_rms_path = os.path.join(out_folder, "rms.npy")
        out_q_path = [os.path.join(out_folder, "questions_{}.csv".format(x)) for x in range(n_questions)]
        out_wav = PySndfile(out_wav_path, 'w', construct_format('wav', 'pcm16'), 3, 44100)
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
        with open(out_csv_path, 'w') as csvfile, ExitStack() as stack:
            # Open all question files
            qfiles = [
                stack.enter_context(open(qfile, 'w'))
                for qfile in out_q_path
            ]
            writer = csv.writer(csvfile)
            qwriters = [csv.writer(qfile) for qfile in qfiles]

            counter = 0
            stim_count = len(merged_wavs)
            stim_count_half = stim_count//2
            q_inds = np.array([
                sample(range(0, stim_count_half), n_questions),
                sample(range(stim_count_half, stim_count-1), n_questions)
            ]).T
            a = 0
            silence = np.zeros((88200, 3))
            idx = np.arange(0, silence.shape[0])
            trigger = gen_trigger(idx, 2., 0.01, fs)
            silence[:, 2] = trigger
            out_wav.write_frames(silence)
            for ind, (wav, txt) in enumerate(zip(merged_wavs, words)):
                csv_line = [counter]
                silence = np.zeros((int(np.random.uniform(int(0.3*44100), int(0.4*44100), 1)), 3))
                idx = np.arange(counter, counter+silence.shape[0])
                trigger = gen_trigger(idx, 2., 0.01, fs)
                silence[:, 2] = trigger
                out_wav.write_frames(silence)
                counter += silence.shape[0]
                csv_line.append(counter)
                csv_line.append("#")
                writer.writerow(csv_line)
                csv_line = [counter]
                x, fs, enc = sndio.read(wav)
                sum_sqrd += np.sum(x**2)
                n += x.size

                y = np.vstack([x, x, np.zeros(x.size)]).T
                idx = np.arange(counter, counter+y.shape[0])
                trigger = gen_trigger(idx, 2., 0.01, fs)
                y[:, 2] = trigger
                out_wav.write_frames(y)
                counter += y.shape[0]
                csv_line.append(counter)
                csv_line.append(" ".join(txt))
                writer.writerow(csv_line)
                if ind in q_inds:
                    writer_ind = int(np.where(ind == q_inds)[0])
                    blank_ind = randint(0, len(txt)-1)
                    q_list = copy(txt)
                    q_list[blank_ind] = '_'
                    qwriters[writer_ind].writerow([" ".join(q_list), txt[blank_ind]])
                    a += 1
            if a != 8:
                pdb.set_trace()

            csv_line = [counter]
            silence = np.zeros((int(np.random.uniform(int(0.3*44100), int(0.4*44100), 1)), 3))
            idx = np.arange(counter, counter+silence.shape[0])
            trigger = gen_trigger(idx, 2., 0.01, fs)
            silence[:, 2] = trigger
            out_wav.write_frames(silence)
            counter += silence.size
            csv_line.append(counter)
            csv_line.append("#")
            writer.writerow(csv_line)
            rms = np.sqrt(sum_sqrd/n)
            np.save(out_rms_path, rms)

            x, fs, enc = sndio.read(out_wav_path)

if __name__ == "__main__":
    main()
