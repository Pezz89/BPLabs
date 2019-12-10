#!/usr/bin/env python3
from pathops import dir_must_exist
import os
import dill
import numpy as np
import pdb
import json
from natsort import natsorted
import random
random.seed(42)
np.random.seed(42)
import itertools
import copy

from config import server, socketio, participants

def set_trace():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log = logging.getLogger('engineio')
    log.setLevel(logging.ERROR)
    pdb.set_trace()


def find_participants(folder='./participant_data/'):
    '''
    Returns a tuple of (participant number, participant filepath) for every
    participant folder found in directory provided
    '''
    part_folder = [os.path.join(folder, o) for o in os.listdir(folder)
                        if os.path.isdir(os.path.join(folder,o))]
    for path in part_folder:
        part_key = os.path.basename(path)
        participants[part_key] = Participant(participant_dir=path)
        participants[part_key].load('info')
    return participants

def gen_participant_num(participants, N = 100):
    # generate array of numbers that haven't been taken between 0-100
    # if list is empty increment until list isnt empty
    # Choose a number
    taken_nums = []
    for part_key in participants.keys():
        participant = participants[part_key]
        taken_nums.append(int(participant['info']['number']))
    inds = np.arange(N)+1
    taken_inds = np.in1d(inds, taken_nums)

    inds = inds[~taken_inds]
    return inds


class Participant:
    def __init__(self, participant_dir=None, number=None, age=None, gender=None, handedness=None, general_notes=None, parameters={}):
        '''
        '''
        dir_must_exist(participant_dir)
        self.participant_dir = participant_dir
        self.data_paths = {}
        self.generate_folder_hierachy()
        self.parameters = parameters

        self.data = {
            "info": {
                "number": number,
                "age": age,
                "gender": gender,
                "handedness": handedness,
                "general_notes": general_notes
            },

            "mat_test": {
                "notes": ''
            },

            "eeg_story_train": {
                "notes": ''
            },
            "eeg_mat_train": {
                "notes": ''
            },

            "eeg_test": {
                "notes": ''
            },

            "da_test": {
                "notes": ''
            },

            "click_test": {
                "notes": ''
            },

            "pta": {
                "notes": ''
            }
        }
        self.data['parameters'] = parameters

    def generate_folder_hierachy(self):
        '''
        '''
        sub_dirs = ["mat_test", "da_test", "pta", "click_test", "info",
                    "eeg_story_train", "eeg_mat_train", "eeg_test",
                    "eeg_test/stimulus", "parameters"]
        for dir_name in sub_dirs:
            dn = os.path.join(*dir_name.split('/'))
            path = os.path.join(self.participant_dir, dn)
            dir_must_exist(path)
            self.data_paths[dir_name] = path

    def __setitem__(self, key, item):
        self.data[key] = item

    def __getitem__(self, key):
        return self.data[key]

    def set_info(self, info):
        self.data['info'] = info

    def save(self, data_key):
        '''
        '''
        directory = self.data_paths[data_key]
        with open(os.path.join(directory, '{}.pkl'.format(data_key)), 'wb') as f:
            dill.dump(self.data[data_key], f)


    def load(self, data_key):
        '''
        '''
        folder = os.path.join(self.participant_dir, data_key)
        with open(os.path.join(folder, "{}.pkl".format(data_key)), 'rb') as f:
            self.data[data_key].update(dill.load(f))

def roll_independant(A, r):
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]

    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]

    result = A[rows, column_indices]
    return result


def main():
    '''
    '''
    print("***REMEMBER THIS SCRIPTS WILL NOT OVERWRITE ANY EXISTING PARTICIPANT DATA. PLEASE DELETE THIS MANUALLY IF NEEDED!***")
    participants = find_participants()
    with open('./test_params.json') as json_file:
        general_params = json.load(json_file)
    # Generate all permutations of tests for couterbalancing
    tests = general_params['tests']
    cb_tests = list(itertools.permutations(tests))
    tone_freqs = general_params['tone_freq']
    cb_tone_freqs = list(itertools.permutations(tone_freqs))

    # Make sure that the number of participants is a multiple of the number of
    # counterbalanced tests
    cb_lcm = np.lcm.reduce([len(cb_tests), len(cb_tone_freqs)])
    n_participants = cb_lcm * 6
    part_nums = gen_participant_num(participants, N=n_participants)
    n_decoder_repeats = general_params['decoder_test_SNR_repeats']
    # Get all decoder test stimuli
    listDir = "./matrix_test/short_concat_stim/out"
    stim_dirs = natsorted([x for x in os.listdir(listDir) if os.path.isdir(os.path.join(listDir, x))])


    for i in part_nums:
        participant_params = {}
        # Set the order of tests to be presented to the current participant,
        # using previous counterbalancing above
        participant_params['tests'] = list(cb_tests[(i-1) % len(cb_tests)])
        # Randomy shuffle order of stimuli to be presented in decoder testing
        sd_copy = copy.copy(stim_dirs)
        random.shuffle(sd_copy)
        participant_params['decoder_test_lists'] = sd_copy
        # Generate randomised stimulus/SNR combinations for each participant
        snrs = np.array(general_params['decoder_SNRs'], dtype=float)
        np.random.shuffle(snrs)
        snrs = np.repeat(snrs[np.newaxis], n_decoder_repeats, axis=0)
        snrs = roll_independant(snrs, np.array(np.arange(n_decoder_repeats)+1-n_decoder_repeats))
        participant_params['decoder_test_SNRs'] = snrs

        # Is the hearing loss simulator active for this participant?
        # Even numbers yes, odd numbers no
        hl_sim_active = (i-1 % 2) == 0
        participant_params['hl_sim_active'] = hl_sim_active
        # What order are the decoder stories presented?
        # What order are the behavioral test stimuli presented?
        # What order are the tone SNRs presented at?
        n_tone_repeats = general_params['tone_repeats']
        tone_snrs = np.array(general_params['tone_SNRs'], dtype=float)
        np.random.shuffle(tone_snrs)
        snrs = np.repeat(tone_snrs[np.newaxis], n_tone_repeats, axis=1)
        participant_params['tone_SNRs'] = snrs

        # What order are the tones presented at?
        # Set the order of tone frequencies to be presented to the current
        # participant, using previous counterbalancing above
        participant_params['tone_freqs'] = list(cb_tone_freqs[(i-1) % len(cb_tone_freqs)])



        key = "participant_{}".format(i)
        print(f"Generating: {key}")
        participants[key] = Participant(participant_dir="./participant_data/{}".format(key), number=i, parameters=participant_params)
        participants[key].save("info")
    print(f"Generated {part_nums.size} new participant databases")



if __name__ == '__main__':
    main()