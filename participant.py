from pathops import dir_must_exist
import os
import dill
import numpy as np
import pdb

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
        taken_nums.append(int(participant['info']['number'][0]))
    n = 0
    num_found = False
    while not num_found:
        potential_nums = np.arange(N)+n+1
        try:
            valid_nums = np.setdiff1d(potential_nums, taken_nums)
        except:
            set_trace()
        if valid_nums.size:
            num_found = True
        else:
            n += N
    return np.random.choice(valid_nums)


class Participant:
    def __init__(self, participant_dir=None, number=None, age=None, gender=None, handedness=None, general_notes=None):
        '''
        '''
        dir_must_exist(participant_dir)
        self.participant_dir = participant_dir
        self.data_paths = {}
        self.generate_folder_hierachy()

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

            "eeg_train": {
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

    def generate_folder_hierachy(self):
        '''
        '''
        sub_dirs = ["mat_test", "da_test", "pta", "click_test", "info",
                    "eeg_story_train", "eeg_mat_train", "eeg_test",
                    "eeg_test/stimulus", ]
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


