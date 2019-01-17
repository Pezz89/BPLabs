#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../matrix_test/helper_modules")

import numpy as np
from pathops import dir_must_exist
from filesystem import globDir
from pysndfile import sndio
import os
from signalops import block_process_wav
from shutil import copyfile

def calc_potential_max(wavs, noise_filepath, out_dir, out_name):
    max_wav_samp = 0
    max_wav_rms = 0
    for wav in wavs:
        x, fs, enc = sndio.read(wav)
        max_wav_samp = np.max([max_wav_samp, np.max(np.abs(x))])
        max_wav_rms = np.max([max_wav_rms, np.sqrt(np.mean(x**2))])
    x, fs, enc = sndio.read(noise_filepath)
    noise_rms = np.sqrt(np.mean(x**2))
    max_noise_samp = max(np.abs(x))

    snr = -15.
    snr_fs = 10**(-snr/20)
    max_noise_samp *= max_wav_rms/noise_rms
    max_sampl = max_wav_samp+(max_noise_samp*snr_fs)
    reduction_coef = 1.0/max_sampl
    np.save(os.path.join(out_dir, "{}.npy".format(out_name)), reduction_coef)
    return reduction_coef

def main():
    '''
    '''
    da_files = ["../da_stim/stimulus/3000_da.wav"]
    story_dir = "../eeg_story_stim/stimulus"
    mat_dir = "../matrix_test/speech_components"
    noise_file = "../matrix_test/behavioural_stim/stimulus/wav/noise/noise_norm.wav"
    da_noise_file = "../da_stim/noise/wav/noise/noise_norm.wav"

    story_wavs = globDir(story_dir, '*.wav')
    mat_wavs = globDir(mat_dir, '*.wav')

    out_dir = "./out"
    out_red_dir = os.path.join(out_dir, 'reduction_coefficients')
    out_stim_dir = os.path.join(out_dir, 'stimulus')
    dir_must_exist(out_dir)
    dir_must_exist(out_red_dir)
    dir_must_exist(out_stim_dir)
    import pdb
    pdb.set_trace()
    story_coef = calc_potential_max(story_wavs, noise_file, out_red_dir, "story_red_coef")
    mat_coef = calc_potential_max(mat_wavs, noise_file, out_red_dir, "mat_red_coef")
    da_coef = calc_potential_max(da_files, da_noise_file, out_red_dir, "da_red_coef")

    mat_cal_stim = "../matrix_test/long_concat_stim/out/stim/stim_0.wav"
    da_cal_stim = "../da_stim/stimulus/wav/10min_da.wav"
    click_cal_stim = "../click_stim/click_3000_20Hz.wav"
    story_cal_stim = "../eeg_story_stim/stimulus/odin_1_1.wav"

    mat_out_stim = os.path.join(out_stim_dir, "mat_cal_stim.wav")
    click_out_stim = os.path.join(out_stim_dir, "click_cal_stim.wav")
    da_out_stim = os.path.join(out_stim_dir, "da_cal_stim.wav")
    story_out_stim = os.path.join(out_stim_dir, "story_cal_stim.wav")

    block_process_wav(mat_cal_stim, mat_out_stim, lambda x: x * mat_coef)
    block_process_wav(story_cal_stim, story_out_stim, lambda x: x * story_coef)
    block_process_wav(da_cal_stim, da_out_stim, lambda x: x * da_coef)
    copyfile(click_cal_stim, click_out_stim)


if __name__ == "__main__":
    main()
